from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Source checkouts keep the pipeline modules in ``core``; frozen builds put
# them at the bundle root.  Add both locations before importing them.
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE / "core", _HERE):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from cityloder_main import run_cityloder_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete CityLODer pipeline on Windows."
    )
    parser.add_argument(
        "--las",
        required=True,
        help="LAS/LAZ file or folder; folders are searched recursively.",
    )
    parser.add_argument("--footprints", required=True, help="Building footprint SHP path.")
    parser.add_argument("--output", default="output", help="Output directory.")
    parser.add_argument("--name", default="mycity", help="Output model name.")
    parser.add_argument("--streets", default="", help="Optional street GeoJSON path.")
    parser.add_argument("--building-class", type=int, default=6)
    parser.add_argument("--ground-class", type=int, default=2)
    parser.add_argument("--temp", default="", help="Optional temporary directory.")
    parser.add_argument("--graph-target-crs", default="")
    parser.add_argument("--mesh-exe", default="")
    parser.add_argument("--mesh-backend", choices=["auto", "triangle", "cityjson"], default="auto",
                        help="auto: use C++ and recover via CityJSON/GEOS if it fails; cityjson: skip Triangle.")
    parser.add_argument(
        "--mode",
        choices=["auto", "single", "tiled"],
        default="auto",
        help="auto: folders run tiled, one file runs the classic pipeline.",
    )

    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="cuda: require CUDA for accelerated stages; auto: warn on CPU fallback.")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.8,
                        help="CuPy pool limit as a fraction of total VRAM (default: 0.8).")

    # Tiled-mode controls.
    parser.add_argument(
        "--dedupe",
        dest="dedupe",
        action="store_true",
        default=False,
        help="Opt in to selecting one product per identical source extent.",
    )
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false")
    parser.add_argument(
        "--include-ept",
        action="store_true",
        default=True,
        help="Also read files inside ept_* folders (included by default).",
    )
    parser.add_argument(
        "--include-merged",
        action="store_true",
        default=True,
        help="Also read full-area merged files such as 'Teste cidade.las' (included by default).",
    )
    parser.add_argument(
        "--only-tiles-with-buildings",
        action="store_true",
        help="Process only tiles that contain footprints (faster preview).",
    )
    parser.add_argument(
        "--tile-filter",
        default="",
        help="Comma-separated tile keys, e.g. 396000-6455000,395000-6454000.",
    )
    parser.add_argument(
        "--bbox",
        default="",
        help="Restrict tiles to a bounding box: minx,miny,maxx,maxy.",
    )
    parser.add_argument(
        "--tile-size",
        type=float,
        default=1000.0,
        help="Native tile size in map units (default: 1000).",
    )
    parser.add_argument(
        "--overhang",
        type=float,
        default=50.0,
        help="Extra metres loaded around footprints near tile borders (default: 50).",
    )
    parser.add_argument(
        "--ground-sample-step",
        type=float,
        default=25.0,
        help="Terrain sample spacing in metres (default: 25; use 0 to disable).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard cached tile results and rebuild from scratch.",
    )
    parser.add_argument(
        "--save-parts",
        action="store_true",
        help="Also write buildings_mesh.obj and ground_mesh.obj next to mesh.obj.",
    )
    return parser.parse_args()


def resolve_required(path_value: str, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def resolve_las_inputs(path_value: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in {".las", ".laz"}:
            raise ValueError(f"LAS input must be .las or .laz: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"LAS/LAZ file or folder not found: {path}")
    return path


def list_las_files(path: Path) -> list[Path]:
    files = sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in {".las", ".laz"}
    )
    if not files:
        raise FileNotFoundError(f"No .las or .laz files found in: {path}")
    return files


def resolve_mesh_exe(path_value: str) -> Path:
    if path_value:
        mesh_exe = Path(path_value).expanduser().resolve()
    else:
        updated = _HERE / "build" / "mesh-updated" / "Release" / "city_iconic_mesh.exe"
        if updated.is_file() and not getattr(sys, "frozen", False):
            return updated
        bundle_root = Path(getattr(sys, "_MEIPASS", Path.cwd()))
        mesh_exe = bundle_root / "mesh_gen" / "build" / "Release" / "city_iconic_mesh.exe"
        if not mesh_exe.is_file():
            mesh_exe = Path("mesh_gen/build/Release/city_iconic_mesh.exe").resolve()
    if not mesh_exe.is_file():
        alternative = mesh_exe.parent.parent / "city_iconic_mesh.exe"
        if alternative.is_file():
            mesh_exe = alternative
        else:
            raise FileNotFoundError(
                f"city_iconic_mesh.exe not found. Build mesh_gen first or pass --mesh-exe: {mesh_exe}"
            )
    return mesh_exe


def run_mesh(
    mesh_exe: Path,
    ground_polygon: str,
    temp_path: Path,
    graph_path: str,
    model_path: Path,
    save_parts: bool,
) -> None:
    mesh_graph = str(graph_path) if graph_path else ""
    command = [
        str(mesh_exe),
        ground_polygon,
        str(temp_path),
        mesh_graph,
        str(model_path),
    ]
    if save_parts:
        command.append("--save-parts")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"city_iconic_mesh failed with exit code {completed.returncode}."
        )


def generate_mesh(backend, mesh_exe, ground_polygon, temp_path, graph_for_mesh,
                  model_path, save_parts=False, tile_size=1000.):
    recovered = backend == "cityjson" or mesh_exe is None
    if not recovered:
        try:
            run_mesh(mesh_exe, ground_polygon, temp_path, graph_for_mesh,
                     model_path, save_parts)
        except RuntimeError:
            if backend == "triangle":
                raise
            print("C++ triangulation failed; recovering mesh from CityJSON and cached terrain.")
            recovered = True
    if recovered:
        from cityjson_to_mesh import export_cityjson
        export_cityjson(f"{model_path}.city.json", model_path / "mesh.obj",
                        ground_polygon, tile_size, save_parts)


def main() -> int:
    args = parse_args()
    from cityloder_gpu import configure_gpu, gpu_summary
    configure_gpu(args.device, args.gpu_memory_fraction)
    print(gpu_summary())
    las_input = resolve_las_inputs(args.las)
    footprints_path = resolve_required(args.footprints, "Footprint file")
    graph_path = (
        resolve_required(args.streets, "Street GeoJSON")
        if args.streets and args.streets != "."
        else ""
    )
    output_path = Path(args.output).expanduser().resolve()
    temp_path = (
        Path(args.temp).expanduser().resolve()
        if args.temp
        else output_path / f"{args.name}_tmp"
    )
    model_path = output_path / args.name
    mesh_exe = None
    if args.mesh_backend != "cityjson":
        try:
            mesh_exe = resolve_mesh_exe(args.mesh_exe)
        except FileNotFoundError:
            if args.mesh_backend == "triangle":
                raise
            print("C++ mesh generator unavailable; using CityJSON/GEOS recovery.")

    output_path.mkdir(parents=True, exist_ok=True)
    temp_path.mkdir(parents=True, exist_ok=True)
    total_timer = time.perf_counter()

    mode = args.mode
    if mode == "auto":
        mode = "tiled" if las_input.is_dir() else "single"
    if mode == "tiled" and not las_input.is_dir():
        raise ValueError("Tiled mode requires a folder passed to --las.")

    if mode == "single":
        las_files = [las_input] if las_input.is_file() else list_las_files(las_input)
        print(f"LAS/LAZ files found: {len(las_files)}")
        las_path = ",".join(str(file) for file in las_files)
        ground_polygon = temp_path / "ground_polygon"
        run_cityloder_pipeline(
            input2D_path=str(footprints_path),
            graph_path=str(graph_path),
            las_path=las_path,
            outname=str(model_path),
            class_building=args.building_class,
            class_ground=args.ground_class,
            temp_fold=str(temp_path),
            ground_polygon_name=str(ground_polygon),
            graph_target_crs=args.graph_target_crs,
        )
        ground_polygon_off = f"{ground_polygon}.off"
        graph_for_mesh = (
            str(
                graph_path.parent
                / f"{graph_path.stem}_updated{graph_path.suffix}"
            )
            if graph_path
            else ""
        )
    else:
        from cityloder_tiled import run_cityloder_tiled

        result = run_cityloder_tiled(
            input2D_path=str(footprints_path),
            graph_path=str(graph_path),
            las_path=str(las_input),
            outname=str(model_path),
            class_building=args.building_class,
            class_ground=args.ground_class,
            temp_fold=str(temp_path),
            ground_polygon_name=str(temp_path / "ground_polygon"),
            graph_target_crs=args.graph_target_crs,
            dedupe=args.dedupe,
            include_ept=args.include_ept,
            include_merged=args.include_merged,
            tile_size=args.tile_size,
            overhang=args.overhang,
            tile_filter=args.tile_filter,
            bbox=args.bbox,
            only_tiles_with_buildings=args.only_tiles_with_buildings,
            ground_sample_step=args.ground_sample_step,
            resume=not args.no_resume,
        )
        if not result.get("success", True):
            print(
                "WARNING: some tiles failed; the partial result was written so the "
                "run can be resumed without reprocessing completed tiles."
            )
        ground_polygon_off = result["ground_polygon"]
        graph_for_mesh = result.get("graph_updated_path", "")

    mesh_timer = time.perf_counter()
    generate_mesh(args.mesh_backend, mesh_exe, ground_polygon_off, temp_path,
                  graph_for_mesh, model_path, args.save_parts, args.tile_size)
    mesh_seconds = time.perf_counter() - mesh_timer
    total_seconds = time.perf_counter() - total_timer
    print("---------------------------------------")
    print(f"Mesh generation time: {mesh_seconds:.2f}s")
    print(f"Total pipeline time: {total_seconds:.2f}s")
    print(gpu_summary())
    print("Pipeline completed successfully")
    print("---------------------------------------")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
