import numpy as np
import shapefile
from shapely import STRtree
from shapely.geometry import GeometryCollection, Polygon
from shapely.ops import unary_union
from tqdm import tqdm

try:
    import fast_inpolygon
except ImportError:
    fast_inpolygon = None


def _inpolygon(points_x, points_y, polygon_x, polygon_y):
    if fast_inpolygon is not None:
        return fast_inpolygon.inpolygon(points_x, points_y, polygon_x, polygon_y)

    from matplotlib.path import Path

    points = np.column_stack((points_x, points_y))
    polygon = np.column_stack((polygon_x, polygon_y))
    inside = Path(polygon).contains_points(points)
    on = np.zeros(points.shape[0], dtype=bool)
    return inside, on

def polyarea(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def _join_parts(parts, trailing_nan=True):
    chunks = []
    for idx, part in enumerate(parts):
        part = np.asarray(part, dtype=float)
        if part.size == 0:
            continue
        chunks.append(part)
        if trailing_nan or idx < len(parts) - 1:
            chunks.append(np.array([[np.nan, np.nan]], dtype=float))

    if not chunks:
        return np.empty((0, 2), dtype=float)

    out = np.vstack(chunks)
    if not trailing_nan and np.isnan(out[-1, 0]):
        out = out[:-1]
    return out


def _split_parts(xy):
    xy = np.asarray(xy, dtype=float)
    if xy.size == 0:
        return []

    parts = []
    current = []
    for row in xy:
        if np.isnan(row[0]) or np.isnan(row[1]):
            if current:
                parts.append(np.asarray(current, dtype=float))
                current = []
        else:
            current.append(row)

    if current:
        parts.append(np.asarray(current, dtype=float))

    return parts


def _sample_boundary_points(poly):
    poly = np.asarray(poly, dtype=float)
    if len(poly) == 0:
        return np.empty((0, 2), dtype=float)

    t = np.arange(0.0, 1.0 + 1e-12, 0.1)[:, None]
    points = []

    for k in range(len(poly) - 1):
        points.append(poly[k] * t + poly[k + 1] * (1.0 - t))

    if np.linalg.norm(poly[0] - poly[-1]) > 0:
        points.append(poly[-1] * t + poly[0] * (1.0 - t))

    return np.vstack(points) if points else np.empty((0, 2), dtype=float)


def _safe_polygon(exterior, holes=None):
    holes = holes or []
    poly = Polygon(
        np.asarray(exterior, dtype=float),
        holes=[np.asarray(h, dtype=float).tolist() for h in holes if len(h) >= 3],
    )

    # MATLAB polyshape is fairly forgiving; buffer(0) is a common tidy-up step.
    if not poly.is_valid:
        fixed = poly.buffer(0)
        if not fixed.is_empty:
            poly = fixed

    return poly


def _xy_to_geometry(xy):
    parts = _split_parts(xy)
    if not parts:
        return GeometryCollection()

    exterior = parts[0]
    holes = parts[1:]
    return _safe_polygon(exterior, holes=holes)


def _geometry_to_xy(geom):
    if geom.is_empty:
        return np.array([[np.nan, np.nan]], dtype=float)

    if geom.geom_type == "Polygon":
        polygons = [geom]
    elif geom.geom_type == "MultiPolygon":
        polygons = list(geom.geoms)
    else:
        polygons = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
        if not polygons:
            return np.array([[np.nan, np.nan]], dtype=float)

    parts = []
    for poly in polygons:
        parts.append(np.asarray(poly.exterior.coords, dtype=float))
        for interior in poly.interiors:
            parts.append(np.asarray(interior.coords, dtype=float))

    return _join_parts(parts, trailing_nan=True)


def _read_shape_records(reader):
    """Read geometry records, allowing a geometry-only .shp without a .dbf."""
    try:
        return reader.shapeRecords()
    except shapefile.ShapefileException as exc:
        if "DbfReader requires" not in str(exc):
            raise
        return [(shape, {}) for shape in reader.iterShapes()]


def footprint_preprocess(fname, hole_thr=1, bounds=None):
    """
    Python translation of the MATLAB function footprintPreprocess,
    with an additional MATLAB-style shaperead-like output `M`.

    Parameters
    ----------
    fname : str
        Path to the .shp file.

    Returns
    -------
    final_building : list
        Processed building geometries.
    final_id : list
        IDs associated with merged/intersecting buildings.
    final_check_adj : list
        Adjacent/intersection information.
    M : list of dict
        MATLAB-style structure array, one entry per shape record, with:
            - 'Geometry'
            - 'X'
            - 'Y'
            - shapefile attributes
    """
    reader = shapefile.Reader(fname)
    try:
        field_names = [f[0] for f in reader.fields[1:]]
    except shapefile.ShapefileException as exc:
        if "DbfReader requires" not in str(exc):
            raise
        field_names = []

    # Keep the same behavior as your current function
    shape_records = _read_shape_records(reader)

    building = []
    id_building = []

    # New output
    original_buildings = []

    # for i, sr in enumerate(shape_records, start=1):
    for id_oggetto, sr in enumerate(tqdm(shape_records, desc="Processing footprints"), start=1):
        if isinstance(sr, tuple):
            shp, record = sr
        else:
            shp, record = sr.shape, sr.record
        rec = dict(zip(field_names, record))

        # ---------------------------------------------------------------
        # Build MATLAB-style structure entry M
        x_coords = []
        y_coords = []

        if shp.points:
            part_starts_full = list(shp.parts) + [len(shp.points)]
            for j in range(len(part_starts_full) - 1):
                start_idx = part_starts_full[j]
                end_idx = part_starts_full[j + 1]
                part = shp.points[start_idx:end_idx]
                if part:
                    xs, ys = zip(*part)
                    x_coords.extend(xs)
                    y_coords.extend(ys)
                    x_coords.append(np.nan)
                    y_coords.append(np.nan)

        shape_struct = {
            "Geometry": shp.shapeTypeName,
            "X": np.array(x_coords, dtype=float),
            "Y": np.array(y_coords, dtype=float),
        }
        shape_struct.update(rec)
        original_buildings.append(shape_struct)
        # ---------------------------------------------------------------

        if bounds is not None and shp.points:
            min_xy = np.min(shp.points, axis=0)[:2]
            max_xy = np.max(shp.points, axis=0)[:2]
            if np.any(max_xy < bounds[0]) or np.any(min_xy > bounds[1]):
                continue

        points = np.asarray(shp.points, dtype=float)
        if points.size == 0:
            continue

        part_starts = list(shp.parts) + [len(points)]
        rings = [
            points[part_starts[j]:part_starts[j + 1]]
            for j in range(len(part_starts) - 1)
        ]
        if not rings:
            continue

        poly_e = rings[0]

        if len(rings) > 1:
            lab_int = []
            lab_ext = []

            for ring_idx, poly_i in enumerate(rings[1:], start=2):
                inside, on = _inpolygon(
                    poly_i[:, 0], poly_i[:, 1],
                    poly_e[:, 0], poly_e[:, 1]
                )

                if np.count_nonzero(inside) == len(inside):
                    if len(poly_i) > 2 and polyarea(poly_i[:, 0], poly_i[:, 1]) < hole_thr:
                        lab_int.append(ring_idx)
                else:
                    if np.count_nonzero(~inside) == len(inside):
                        lab_ext.append((ring_idx, 1))
                    else:
                        lab_ext.append((ring_idx, 0))

            # Save first polygon + any interior polygons as one building
            main_parts = [poly_e] + [rings[idx - 1] for idx in lab_int]
            building.append(_join_parts(main_parts, trailing_nan=True))
            id_building.append(id_oggetto)

            # Preserve exact MATLAB behavior
            for ring_idx, _flag in lab_ext:
                poly_i = rings[ring_idx - 1]

                building.append(_join_parts([poly_i], trailing_nan=True))
                id_building.append(id_oggetto)

        else:
            building.append(poly_e.copy())
            id_building.append(id_oggetto)

    # ------------------------------------------------------------------
    # Check error "footprint has duplicate vertices"
    for i in range(len(building)):
        xy = building[i]
        parts = _split_parts(xy)

        cleaned_parts = []
        for p in parts:
            moltepl = np.ones(len(p), dtype=int)

            # Skip first and last, as in MATLAB
            for j in range(1, len(p) - 1):
                dist_row = np.linalg.norm(p - p[j], axis=1)
                moltepl[j] = np.count_nonzero(dist_row == 0)

            if len(moltepl) > 0 and moltepl.max() > 1:
                ind = np.flatnonzero(moltepl == moltepl.max())

                p1 = p[ind[0]:ind[-1] + 1]
                p2 = np.vstack([p[:ind[0] + 1], p[ind[-1]:]])

                p = p1 if len(p1) > len(p2) else p2

            cleaned_parts.append(p)

        building[i] = _join_parts(cleaned_parts, trailing_nan=True)

    # ------------------------------------------------------------------
    # Check intersecting polygons using a spatial index.  The previous
    # implementation allocated a dense N x N matrix (about 38 GB for the
    # full city) and compared every footprint against every other one.
    n_buildings = len(building)
    check_adj = [set() for _ in range(n_buildings)]

    if n_buildings == 0:
        return [], [], [], original_buildings

    geoms = []
    for xy in building:
        try:
            geom = _xy_to_geometry(xy)
        except Exception:
            geom = GeometryCollection()
        if not geom.is_empty and not geom.is_valid:
            try:
                fixed = geom.buffer(0)
                if not fixed.is_empty:
                    geom = fixed
            except Exception:
                pass
        geoms.append(geom)

    valid_idx = [
        i for i, geom in enumerate(geoms)
        if not geom.is_empty and geom.geom_type in ("Polygon", "MultiPolygon")
    ]

    merge_edges = []
    if len(valid_idx) > 1:
        valid_geoms = [geoms[i] for i in valid_idx]
        tree = STRtree(valid_geoms)
        pairs = tree.query(valid_geoms, predicate="intersects")
        for a, b in zip(pairs[0].tolist(), pairs[1].tolist()):
            if a >= b:
                continue
            i = valid_idx[a]
            j = valid_idx[b]
            try:
                overlap_area = geoms[i].intersection(geoms[j]).area
            except Exception:
                overlap_area = 0.0
            if overlap_area > 1e-6:
                merge_edges.append((i, j))
            else:
                # Buildings sharing a boundary but not overlapping.
                check_adj[i].add(id_building[j])
                check_adj[j].add(id_building[i])

    # Connected components of intersecting / overlapping buildings.
    parent = list(range(n_buildings))
    rank = [0] * n_buildings

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for i, j in merge_edges:
        union(i, j)

    components = {}
    for i in range(n_buildings):
        components.setdefault(find(i), []).append(i)
    components = list(components.values())

    # ------------------------------------------------------------------
    # Merge connected components
    final_building = []
    final_id = []
    final_check_adj = []

    for comp in components:
        idx = sorted(comp)

        if len(idx) == 1:
            j = idx[0]
            final_building.append(building[j])
            final_id.append(np.asarray([id_building[j]], dtype=object))

            if check_adj[j]:
                final_check_adj.append(np.asarray(sorted(check_adj[j]), dtype=object))
            else:
                final_check_adj.append(np.nan)

        else:
            geom = _xy_to_geometry(building[idx[0]])
            ids = [id_building[idx[0]]]
            adj_ids = set(check_adj[idx[0]])

            for j in idx[1:]:
                geom = unary_union([geom, _xy_to_geometry(building[j])])
                ids.append(id_building[j])
                adj_ids.update(check_adj[j])

            final_building.append(_geometry_to_xy(geom))
            final_id.append(np.asarray(ids, dtype=object))

            if adj_ids:
                final_check_adj.append(np.asarray(sorted(adj_ids), dtype=object))
            else:
                final_check_adj.append(np.nan)

    return final_building, final_id, final_check_adj, original_buildings
