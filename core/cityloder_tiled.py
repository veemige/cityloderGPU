"""Process all recursive LAS/LAZ sources tile by tile, then export one model."""

from __future__ import annotations

import os
import json
import hashlib
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import laspy
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from cityloder_gpu import filter_classes, nearest_xy, CudaExecutionError, prefetch_chunks

from cityloder_cityJSONgen import main as cityJSONgen
from cityloder_compute_buildings import main as pc2offs
from cityloder_preprocess_footprint import footprint_preprocess
from cityloder_preprocess_streetgraph import preprocess_graph
from cityloder_processPC import generate_ground_polygon

TILE_SIZE = 1000.0

_COORD_RE = re.compile(r"(\d{6,7})[-_](\d{6,7})")
_BUILDING_DIR_RE = re.compile(r"^building\d+$")


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


@dataclass
class LasSource:
    path: Path
    cell: tuple[int, int]
    mins: np.ndarray
    maxs: np.ndarray
    point_count: int
    priority: int
    merged: bool = False

    @property
    def name(self) -> str:
        return f"{self.cell[0]}-{self.cell[1]}"


def _is_copc(path: Path) -> bool:
    return ".copc." in path.name.lower()


def _source_priority(path: Path) -> int:
    name = path.name.lower()
    if name.endswith(".las"):
        return 0
    if _is_copc(path):
        return 1
    return 2


def _nominal_cell(path: Path, mins: np.ndarray, tile_size: float) -> tuple[int, int]:
    match = _COORD_RE.search(path.stem)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (
        int(np.floor(mins[0] / tile_size) * tile_size),
        int(np.floor(mins[1] / tile_size) * tile_size),
    )


def _read_header(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with laspy.open(path) as reader:
        header = reader.header
        return (
            np.asarray(header.mins[:2], dtype=float),
            np.asarray(header.maxs[:2], dtype=float),
            int(header.point_count),
        )


def discover_sources(
    root: Path,
    tile_size: float = TILE_SIZE,
    include_ept: bool = True,
    include_merged: bool = True,
    dedupe: bool = False,
) -> tuple[dict[tuple[int, int], list[LasSource]], list[LasSource]]:
    """Index all recursive sources by their header extents.

    All formats and folders are included by default. Explicit deduplication
    selects one product per identical extent, preserving other tile coverage.
    """

    sources: list[LasSource] = []
    merged_sources: list[LasSource] = []
    seen_paths: set[Path] = set()
    skipped_ept = 0
    skipped_merged = 0
    skipped_empty = 0

    for dirpath, dirnames, filenames in os.walk(root):
        if not include_ept:
            kept = []
            for name in dirnames:
                if name.lower().startswith("ept"):
                    skipped_ept += 1
                    continue
                kept.append(name)
            dirnames[:] = kept

        for filename in filenames:
            suffix = Path(filename).suffix.lower()
            if suffix not in {".las", ".laz"}:
                continue
            path = Path(dirpath) / filename
            try:
                path = path.resolve()
            except OSError:
                pass
            if path in seen_paths:
                continue
            seen_paths.add(path)

            try:
                mins, maxs, point_count = _read_header(path)
            except Exception as exc:  # unreadable/corrupt file: report and skip
                print(f"  [tiles] skipping unreadable source {path}: {exc}")
                continue
            if point_count == 0:
                skipped_empty += 1
                continue

            extent_x = maxs[0] - mins[0]
            extent_y = maxs[1] - mins[1]
            merged = max(extent_x, extent_y) > 1.5 * tile_size
            if merged and not include_merged:
                skipped_merged += 1
                continue

            cell = _nominal_cell(path, mins, tile_size)
            sources.append(
                LasSource(
                    path=path,
                    cell=cell,
                    mins=mins,
                    maxs=maxs,
                    point_count=point_count,
                    priority=_source_priority(path),
                    merged=merged,
                )
            )

    # Index every source by its actual header extent, never by its filename.
    cell_sources: dict[tuple[int, int], list[LasSource]] = defaultdict(list)
    for source in sources:
        if source.merged:
            merged_sources.append(source)
        for cell in _cell_range(source.mins, source.maxs, tile_size):
            cell_sources[cell].append(source)
    if dedupe:
        # Only identical extents are alternative products; a shared tile alone
        # does not imply duplicate coverage.
        for cell, items in cell_sources.items():
            unique = {}
            for source in sorted(items, key=lambda item: (item.priority, -item.point_count, str(item.path))):
                key = (tuple(source.mins), tuple(source.maxs))
                unique.setdefault(key, source)
            cell_sources[cell] = list(unique.values())

    print(
        f"  [tiles] sources: {len(sources)} | native cells: {len(cell_sources)} | "
        f"merged kept: {len(merged_sources)} | skipped EPT folders: {skipped_ept}, "
        f"merged files: {skipped_merged}, empty files: {skipped_empty}"
    )
    return cell_sources, merged_sources


# ---------------------------------------------------------------------------
# Point loading
# ---------------------------------------------------------------------------


def _filter_by_bounds(
    record,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    class_building: int,
    class_ground: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(record.x, dtype=np.float64)
    y = np.asarray(record.y, dtype=np.float64)
    z = np.asarray(record.z, dtype=np.float64)
    classification = np.asarray(record.classification)

    gpu_result = filter_classes(x, y, z, classification, bounds_min, bounds_max,
                                class_building, class_ground)
    if gpu_result is not None:
        return gpu_result
    inside = (
        (x >= bounds_min[0])
        & (x <= bounds_max[0])
        & (y >= bounds_min[1])
        & (y <= bounds_max[1])
    )
    building = np.column_stack((x, y, z))[inside & (classification == class_building)]
    ground = np.column_stack((x, y, z))[inside & (classification == class_ground)]
    return building, ground


def load_points(
    path: Path,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    class_building: int,
    class_ground: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load only building and ground classes inside a 2D bounding box."""

    if _is_copc(path):
        with laspy.CopcReader.open(path) as reader:
            record = reader.query(
                laspy.copc.Bounds(mins=bounds_min, maxs=bounds_max)
            )
        return _filter_by_bounds(
            record, bounds_min, bounds_max, class_building, class_ground
        )

    buildings, grounds = [], []
    with laspy.open(path) as reader:
        for record in prefetch_chunks(reader):
            building, ground = _filter_by_bounds(
                record, bounds_min, bounds_max, class_building, class_ground
            )
            if building.size:
                buildings.append(building)
            if ground.size:
                grounds.append(ground)
    return (
        np.concatenate(buildings) if buildings else np.empty((0, 3)),
        np.concatenate(grounds) if grounds else np.empty((0, 3)),
    )


# ---------------------------------------------------------------------------
# Footprint helpers
# ---------------------------------------------------------------------------


def _finite_xy(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if xy.size == 0:
        return np.empty((0, 2), dtype=float)
    return xy[np.isfinite(xy).all(axis=1)]


def assign_buildings_to_tiles(
    building_xy: Sequence[np.ndarray], tile_size: float = TILE_SIZE
) -> tuple[dict[tuple[int, int], list[int]], dict[int, tuple[np.ndarray, np.ndarray]]]:
    per_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
    bboxes: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for index, xy in enumerate(building_xy):
        points = _finite_xy(xy)
        if points.shape[0] == 0:
            continue
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = 0.5 * (mins + maxs)
        cell = (
            int(np.floor(center[0] / tile_size) * tile_size),
            int(np.floor(center[1] / tile_size) * tile_size),
        )
        per_cell[cell].append(index)
        bboxes[index] = (mins, maxs)

    return dict(per_cell), bboxes


def _cell_range(
    bounds_min: np.ndarray, bounds_max: np.ndarray, tile_size: float
) -> Iterable[tuple[int, int]]:
    x0 = int(np.floor(bounds_min[0] / tile_size) * tile_size)
    y0 = int(np.floor(bounds_min[1] / tile_size) * tile_size)
    x1 = int(np.floor(bounds_max[0] / tile_size) * tile_size)
    y1 = int(np.floor(bounds_max[1] / tile_size) * tile_size)
    for cx in range(x0, x1 + 1, int(tile_size)):
        for cy in range(y0, y1 + 1, int(tile_size)):
            yield cx, cy


def _select_cells(
    cell_sources: dict[tuple[int, int], list[LasSource]],
    tile_filter: str = "",
    bbox: str = "",
    only_cells: set[tuple[int, int]] | None = None,
    tile_size: float = TILE_SIZE,
) -> list[tuple[int, int]]:
    selected = sorted(cell_sources)

    if tile_filter:
        wanted = {
            part.strip()
            for part in re.split(r"[,;\s]+", tile_filter)
            if part.strip()
        }
        selected = [cell for cell in selected if f"{cell[0]}-{cell[1]}" in wanted]

    if bbox:
        try:
            minx, miny, maxx, maxy = [float(v) for v in bbox.split(",")]
        except ValueError as exc:
            raise ValueError(
                "--bbox must be four comma-separated numbers: minx,miny,maxx,maxy"
            ) from exc
        selected = [
            cell
            for cell in selected
            if cell[0] + tile_size >= minx
            and cell[0] <= maxx
            and cell[1] + tile_size >= miny
            and cell[1] <= maxy
        ]

    if only_cells is not None:
        selected = [cell for cell in selected if cell in only_cells]

    return selected


# ---------------------------------------------------------------------------
# Ground sampling
# ---------------------------------------------------------------------------


def sample_ground(
    ground: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    step: float,
) -> np.ndarray:
    """Pick an actual ground point near every grid node inside the cell."""

    if ground.shape[0] == 0 or step <= 0:
        return np.empty((0, 3), dtype=float)

    xs = np.arange(bounds_min[0] + step * 0.5, bounds_max[0], step)
    ys = np.arange(bounds_min[1] + step * 0.5, bounds_max[1], step)
    if xs.size == 0 or ys.size == 0:
        return np.empty((0, 3), dtype=float)

    grid = np.column_stack(
        (
            np.repeat(xs, ys.size),
            np.tile(ys, xs.size),
        )
    )
    distance, index = nearest_xy(ground, grid, distance_upper_bound=step * 0.75)
    keep = np.isfinite(distance)
    if not np.any(keep):
        return np.empty((0, 3), dtype=float)

    samples = ground[index[keep]]
    # A grid node may pick the same point as its neighbour; keep one copy.
    return np.unique(samples, axis=0)


def local_hull(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=float)
    if points.shape[0] < 3:
        return points.copy()
    try:
        hull = ConvexHull(points[:, :2])
        return points[hull.vertices]
    except Exception:
        return points.copy()


# ---------------------------------------------------------------------------
# Tile working data
# ---------------------------------------------------------------------------


@dataclass
class TileCache:
    samples: np.ndarray
    hull: np.ndarray
    building_count: int = 0
    elapsed: float = 0.0


def _cache_path(cache_folder: Path, cell: tuple[int, int]) -> Path:
    return cache_folder / f"{cell[0]}-{cell[1]}.npz"


def _save_cache(cache_folder: Path, cell: tuple[int, int], cache: TileCache) -> None:
    cache_folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _cache_path(cache_folder, cell),
        samples=cache.samples,
        hull=cache.hull,
        building_count=np.asarray([cache.building_count], dtype=np.int64),
        elapsed=np.asarray([cache.elapsed], dtype=np.float64),
    )


def _load_cache(cache_folder: Path, cell: tuple[int, int]) -> TileCache | None:
    path = _cache_path(cache_folder, cell)
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            return TileCache(
                samples=np.asarray(data["samples"], dtype=float),
                hull=np.asarray(data["hull"], dtype=float),
                building_count=int(data["building_count"][0]),
                elapsed=float(data["elapsed"][0]),
            )
    except Exception:
        return None


def _clean_temp(temp_fold: Path) -> None:
    """Remove generated building folders and tile cache from a previous run."""

    if not temp_fold.is_dir():
        return
    for child in temp_fold.iterdir():
        if child.is_dir() and _BUILDING_DIR_RE.match(child.name):
            shutil.rmtree(child, ignore_errors=True)
    cache_folder = temp_fold / "tiles"
    if cache_folder.is_dir():
        shutil.rmtree(cache_folder, ignore_errors=True)
    ground = temp_fold / "ground_polygon.off"
    if ground.is_file():
        ground.unlink()


# ---------------------------------------------------------------------------
# Main tiled pipeline
# ---------------------------------------------------------------------------


def run_cityloder_tiled(
    input2D_path: str,
    graph_path: str,
    las_path: str,
    outname: str,
    class_building: int = 6,
    class_ground: int = 2,
    temp_fold: str = "temp",
    ground_polygon_name: str = "ground_polygon",
    graph_target_crs: str = "",
    dedupe: bool = False,
    include_ept: bool = True,
    include_merged: bool = True,
    tile_size: float = TILE_SIZE,
    overhang: float = 50.0,
    tile_filter: str = "",
    bbox: str = "",
    only_tiles_with_buildings: bool = False,
    ground_sample_step: float = 25.0,
    resume: bool = True,
) -> dict:
    """Run the CityLODer pipeline one 1 km tile at a time.

    Parameters follow :func:`cityloder_main.run_cityloder_pipeline`, with
    additional tile-level controls documented in ``cityloder_runner.py``.
    """

    if not np.isfinite(tile_size) or tile_size < 1 or tile_size != int(tile_size):
        raise ValueError("tile_size must be a positive integer in map units")
    start = time.perf_counter()
    root = Path(las_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Tiled mode expects a LAS/LAZ folder: {root}")

    temp_path = Path(temp_fold).expanduser().resolve()
    temp_path.mkdir(parents=True, exist_ok=True)
    cache_folder = temp_path / "tiles"
    if not resume:
        print("  [tiles] clearing previous temporary results (--no-resume)")
        _clean_temp(temp_path)
    cache_folder.mkdir(parents=True, exist_ok=True)

    print("Discovering LAS/LAZ sources ...")
    cell_sources, _ = discover_sources(
        root,
        tile_size=tile_size,
        include_ept=include_ept,
        include_merged=include_merged,
        dedupe=dedupe,
    )
    if not cell_sources:
        raise FileNotFoundError(f"No usable LAS/LAZ files found under: {root}")

    # Invalidate old results when sources, footprints or processing options change.
    source_paths = sorted({source.path for items in cell_sources.values() for source in items})
    footprint_files = sorted(Path(input2D_path).parent.glob(Path(input2D_path).stem + ".*"))
    signature = hashlib.sha256(json.dumps({
        "version": 3,
        "files": [(str(path), path.stat().st_size, path.stat().st_mtime_ns)
                  for path in source_paths + footprint_files],
        "options": [tile_size, overhang, class_building, class_ground, ground_sample_step,
                    dedupe, include_ept, include_merged, tile_filter, bbox, only_tiles_with_buildings],
    }, sort_keys=True).encode()).hexdigest()
    manifest = cache_folder / "signature.txt"
    if not manifest.is_file() or manifest.read_text() != signature:
        _clean_temp(temp_path)
        cache_folder.mkdir(parents=True, exist_ok=True)
        manifest.write_text(signature)

    print("Preprocessing footprints ...")
    preprocess_start = time.perf_counter()
    M, final_id, final_check_adj, M_ori = footprint_preprocess(input2D_path)
    preprocess_seconds = time.perf_counter() - preprocess_start
    print(
        f"  [tiles] footprints: {len(M_ori)} original, {len(M)} after pre-processing "
        f"({preprocess_seconds:.2f}s)"
    )

    buildings_by_cell, building_bboxes = assign_buildings_to_tiles(M, tile_size)
    if not buildings_by_cell:
        print("  [tiles] WARNING: no footprints were assigned to any tile")

    only_cells = set(buildings_by_cell) if only_tiles_with_buildings else None
    tile_cells = _select_cells(
        cell_sources,
        tile_filter=tile_filter,
        bbox=bbox,
        only_cells=only_cells,
        tile_size=tile_size,
    )
    if not tile_cells:
        raise ValueError(
            "No tiles selected. Check --tile-filter/--bbox or the LAS folder."
        )

    id_to_final: dict[int, int] = {}
    for final_index, ids in enumerate(final_id):
        for original_id in np.ravel(ids):
            id_to_final[int(original_id)] = final_index

    ground_by_index: dict[int, np.ndarray] = {}
    missing_building_cells = sorted(set(buildings_by_cell) - set(cell_sources))
    if missing_building_cells:
        missing_count = sum(len(buildings_by_cell[c]) for c in missing_building_cells)
        print(
            f"  [tiles] WARNING: {missing_count} buildings are in "
            f"{len(missing_building_cells)} cells without a LAS source"
        )

    all_samples: list[np.ndarray] = []
    hull_points: list[np.ndarray] = []
    total_buildings = 0
    processed_tiles = 0
    reused_tiles = 0
    failed_tiles: list[tuple[tuple[int, int], str]] = []
    tile_start = time.perf_counter()

    for tile_counter, cell in enumerate(tile_cells, start=1):
        tile_name = f"{cell[0]}-{cell[1]}"
        tile_begin = time.perf_counter()

        cached = _load_cache(cache_folder, cell) if resume else None
        if cached is not None:
            all_samples.append(cached.samples)
            hull_points.append(cached.hull)
            total_buildings += cached.building_count
            reused_tiles += 1
            processed_tiles += 1
            print(
                f"[{tile_counter}/{len(tile_cells)}] {tile_name}: cached "
                f"({cached.building_count} buildings)"
            )
            continue

        cell_min = np.asarray(cell, dtype=float)
        cell_max = cell_min + tile_size
        building_indices = buildings_by_cell.get(cell, [])

        # The own cell is loaded with a small buffer so the ~1 m overlap
        # between native products is not lost.  Buildings crossing a cell
        # boundary extend the load area into the neighbouring cells.
        load_min = cell_min - 2.0
        load_max = cell_max + 2.0
        for index in building_indices:
            mins, maxs = building_bboxes[index]
            load_min = np.minimum(load_min, mins - overhang)
            load_max = np.maximum(load_max, maxs + overhang)

        building_chunks: list[np.ndarray] = []
        ground_chunks: list[np.ndarray] = []
        loaded_paths: set[Path] = set()
        tile_failed = False

        for source_cell in _cell_range(load_min, load_max, tile_size):
            candidates = cell_sources.get(source_cell)
            if not candidates:
                continue
            for source in candidates:
                if source.path in loaded_paths:
                    continue
                loaded_paths.add(source.path)
                try:
                    building_points, ground_points = load_points(
                        source.path,
                        load_min,
                        load_max,
                        class_building,
                        class_ground,
                    )
                except CudaExecutionError:
                    raise
                except Exception as exc:
                    message = f"{source.path.name}: {exc}"
                    print(f"  [tiles] WARNING: could not read {message}")
                    failed_tiles.append((cell, message))
                    tile_failed = True
                    continue
                if building_points.size:
                    building_chunks.append(building_points)
                if ground_points.size:
                    ground_chunks.append(ground_points)

        if tile_failed:
            continue

        if building_chunks:
            xyz_building = np.concatenate(building_chunks, axis=0)
        else:
            xyz_building = np.empty((0, 3), dtype=float)
        if ground_chunks:
            xyz_ground = np.concatenate(ground_chunks, axis=0)
        else:
            xyz_ground = np.empty((0, 3), dtype=float)

        # Keep only the actual cell when sampling the terrain.  Neighbour
        # strips were loaded solely to complete buildings that cross a border.
        if xyz_ground.size:
            inside_cell = (
                (xyz_ground[:, 0] >= cell_min[0])
                & (xyz_ground[:, 0] < cell_max[0])
                & (xyz_ground[:, 1] >= cell_min[1])
                & (xyz_ground[:, 1] < cell_max[1])
            )
            ground_cell = xyz_ground[inside_cell]
        else:
            ground_cell = xyz_ground

        samples = sample_ground(ground_cell, cell_min, cell_max, ground_sample_step)
        tile_hull = local_hull(ground_cell)
        all_samples.append(samples)
        hull_points.append(tile_hull)

        tile_building_count = 0
        if building_indices:
            try:
                pc2offs(
                    str(temp_path),
                    xyz_building,
                    xyz_ground,
                    M,
                    final_id,
                    final_check_adj,
                    indices=building_indices,
                    ground_by_index=ground_by_index,
                    id_to_final=id_to_final,
                    progress_prefix=f"[{tile_name}] ",
                )
                tile_building_count = len(building_indices)
            except CudaExecutionError:
                raise
            except Exception as exc:
                message = str(exc)
                print(f"  [tiles] ERROR while processing {tile_name}: {message}")
                failed_tiles.append((cell, message))
                # Do not write a cache entry: the tile will be retried on the
                # next run (or can be inspected immediately).
                continue

        total_buildings += tile_building_count
        elapsed = time.perf_counter() - tile_begin
        _save_cache(
            cache_folder,
            cell,
            TileCache(
                samples=samples,
                hull=tile_hull,
                building_count=tile_building_count,
                elapsed=elapsed,
            ),
        )
        processed_tiles += 1
        print(
            f"[{tile_counter}/{len(tile_cells)}] {tile_name}: "
            f"{len(xyz_building):,} building pts, {len(xyz_ground):,} ground pts, "
            f"{tile_building_count} buildings, {len(samples)} samples, "
            f"{elapsed:.2f}s"
        )

    tile_seconds = time.perf_counter() - tile_start

    if not all_samples:
        raise ValueError("No ground points were loaded from the selected tiles.")

    samples = (
        np.vstack([item for item in all_samples if item.size])
        if any(item.size for item in all_samples)
        else np.empty((0, 3), dtype=float)
    )
    hulls = [item for item in hull_points if item.size]
    if not hulls:
        raise ValueError("Could not build a ground boundary: no ground points found.")
    hull = np.vstack(hulls)

    print("Generating global ground polygon ...")
    ground_start = time.perf_counter()
    ground_base = str(Path(ground_polygon_name))
    generate_ground_polygon(
        hull,
        samples if samples.size else hull,
        ground_base,
        offset=50.0,
        samples=samples,
    )
    ground_seconds = time.perf_counter() - ground_start

    # The street graph is optional.  Sampling it against the ground samples
    # avoids keeping every ground point of the area in memory.
    graph_start = time.perf_counter()
    graph_updated_path, street_count = preprocess_graph(
        graph_path,
        samples if samples.size else hull,
        outname,
        target_crs=graph_target_crs or None,
    )
    graph_seconds = time.perf_counter() - graph_start

    print("Writing CityJSON ...")
    cityjson_start = time.perf_counter()
    cityJSONgen(
        str(temp_path),
        outname,
        str(graph_updated_path) if graph_updated_path else "",
        M_ori,
        final_id,
    )
    cityjson_seconds = time.perf_counter() - cityjson_start

    elapsed = time.perf_counter() - start
    result = {
        "success": not failed_tiles,
        "mode": "tiled",
        "outname": outname,
        "temp_folder": str(temp_path),
        "ground_polygon": f"{ground_base}.off",
        "graph_updated_path": str(graph_updated_path) if graph_updated_path else "",
        "tiles_selected": len(tile_cells),
        "tiles_processed": processed_tiles,
        "tiles_reused": reused_tiles,
        "tiles_with_buildings": len(buildings_by_cell),
        "buildings_total": total_buildings,
        "failed_tiles": failed_tiles,
        "street_count": street_count,
        "elapsed_seconds": round(elapsed, 2),
        "timings": {
            "footprint_preprocess": round(preprocess_seconds, 2),
            "tiles": round(tile_seconds, 2),
            "ground_polygon": round(ground_seconds, 2),
            "street_graph": round(graph_seconds, 2),
            "cityjson": round(cityjson_seconds, 2),
        },
    }

    print(f"{' Tiled run summary ':=^51}")
    print(f"{'Tiles selected:':32s} {len(tile_cells):8d}")
    print(f"{'Tiles processed:':32s} {processed_tiles:8d}")
    print(f"{'Tiles reused from cache:':32s} {reused_tiles:8d}")
    print(f"{'Footprints after pre-processing:':32s} {len(M):8d}")
    print(f"{'Buildings written:':32s} {total_buildings:8d}")
    print(f"{'Ground samples:':32s} {len(samples):8d}")
    if failed_tiles:
        print(f"{'Failed tiles:':32s} {len(failed_tiles):8d}")
        for cell, message in failed_tiles[:10]:
            print(f"  {cell[0]}-{cell[1]}: {message}")
    print(f"{' Timings ':=^51}")
    print(f"{'Footprint preprocessing:':32s} {preprocess_seconds:8.2f} s")
    print(f"{'Tile processing:':32s} {tile_seconds:8.2f} s")
    print(f"{'Ground polygon:':32s} {ground_seconds:8.2f} s")
    print(f"{'Street graph:':32s} {graph_seconds:8.2f} s")
    print(f"{'CityJSON:':32s} {cityjson_seconds:8.2f} s")
    print(f"{'Total:':32s} {elapsed:8.2f} s")
    print(f"{'=':=>51}")

    return result
