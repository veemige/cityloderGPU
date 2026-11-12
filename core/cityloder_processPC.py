from cityloder_gpu import nearest_xy, filter_classes, prefetch_chunks
import laspy
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
import numpy as np

# TODO: classification step if no classification is detected

def read_las_file(file_path, class_building = 6, class_ground = 2):
    """
    Reads a LAS file and extracts XYZ coordinates and classification.

    Parameters:
        file_path (str): Path to the .las file.

    Returns:
        xyz_building (np.ndarray): Building-class XYZ points from all input files.
        xyz_ground (np.ndarray): Ground-class XYZ points from all input files.
        point_count (int): Total number of points across all files.
        bounds (tuple): XY minimum and maximum for the loaded LAS files.
    """
    if isinstance(file_path, (list, tuple)):
        paths = [Path(path) for path in file_path]
    else:
        paths = [Path(path.strip()) for path in str(file_path).split(",") if path.strip()]

    if not paths:
        raise ValueError("Informe ao menos um arquivo LAS.")

    building_chunks = []
    ground_chunks = []
    point_count = 0
    bounds_min = np.full(2, np.inf)
    bounds_max = np.full(2, -np.inf)

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo LAS não encontrado: {path}")

        with laspy.open(path) as reader:
            bounds_min = np.minimum(bounds_min, reader.header.mins[:2])
            bounds_max = np.maximum(bounds_max, reader.header.maxs[:2])
            for las in prefetch_chunks(reader):
                classification = np.asarray(las.classification)
                x, y, z = (np.asarray(values, dtype=np.float64) for values in (las.x, las.y, las.z))
                point_count += len(classification)
                result = filter_classes(x, y, z, classification, [-np.inf, -np.inf],
                                        [np.inf, np.inf], class_building, class_ground)
                if result is not None:
                    for points, chunks in zip(result, (building_chunks, ground_chunks)):
                        if points.size:
                            chunks.append(points)
                else:
                    for class_code, chunks in ((class_building, building_chunks), (class_ground, ground_chunks)):
                        mask = classification == class_code
                        if np.any(mask):
                            chunks.append(np.column_stack((x[mask], y[mask], z[mask])))

    if not ground_chunks:
        raise ValueError(f"Nenhum ponto da classe de terreno {class_ground} foi encontrado.")
    if not building_chunks:
        raise ValueError(f"Nenhum ponto da classe de edifício {class_building} foi encontrado.")

    xyz_building = np.concatenate(building_chunks, axis=0)
    xyz_ground = np.concatenate(ground_chunks, axis=0)

    return xyz_building, xyz_ground, point_count, (bounds_min, bounds_max)

def generate_ground_polygon(
    xyz,
    other,
    outname="ground_polygon",
    offset=50.0,
    samples=None,
):
    """
    Build a fast 2D convex hull from the XY coordinates of `xyz` and `other`,
    enlarge it by `offset`, assign Z from nearest XY point in `other`,
    then export a single-face OFF polygon as `<outname>.off`.

    If `samples` (an Nx3 array) is given, its points are written as
    unreferenced OFF vertices.  The C++ mesh generator passes those vertices
    to Triangle, so they constrain the terrain across the whole area instead
    of only along the outer boundary.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    if other.size == 0:
        raise ValueError("Não há pontos de terreno para gerar o limite do mapa.")
    aus = np.vstack((other, xyz))

    pts2d = aus[:, :2]

    # Remove exact duplicate XY points first: often a large speed win
    pts2d = np.unique(pts2d, axis=0)

    # -------------------
    # Fast 2D convex hull
    # -------------------
    pts2d = np.asarray(pts2d, dtype=float)

    hull2d = pts2d[ConvexHull(pts2d).vertices]

    # Ensure CCW orientation
    def signed_area(poly):
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

    if len(hull2d) >= 3 and signed_area(hull2d) < 0:
        hull2d = hull2d[::-1]

    # ----------------------------------------
    # Exact convex polygon outward XY offset
    # ----------------------------------------
    def line_intersection(p1, d1, p2, d2):
        # Solve p1 + t*d1 = p2 + s*d2
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-12:
            # Nearly parallel: fallback to average of shifted vertices
            return 0.5 * (p1 + p2)
        diff = p2 - p1
        t = (diff[0] * d2[1] - diff[1] * d2[0]) / denom
        return p1 + t * d1

    def offset_convex_polygon(poly, dist):
        n = len(poly)
        if dist == 0 or n < 3:
            return poly.copy()

        out = np.empty_like(poly)

        for i in range(n):
            p_prev = poly[i - 1]
            p_curr = poly[i]
            p_next = poly[(i + 1) % n]

            e1 = p_curr - p_prev
            e2 = p_next - p_curr

            n1 = np.array([e1[1], -e1[0]], dtype=np.float64)
            n2 = np.array([e2[1], -e2[0]], dtype=np.float64)

            l1 = np.linalg.norm(n1)
            l2 = np.linalg.norm(n2)
            if l1 < 1e-12 or l2 < 1e-12:
                out[i] = p_curr
                continue

            n1 /= l1
            n2 /= l2

            # Shift the two incident edges outward, then intersect them
            a1 = p_prev + dist * n1
            a2 = p_curr + dist * n1
            b1 = p_curr + dist * n2
            b2 = p_next + dist * n2

            d1 = a2 - a1
            d2 = b2 - b1

            out[i] = line_intersection(a1, d1, b1, d2)

        return out

    hull2d = offset_convex_polygon(hull2d, float(offset))

    # ----------------------------------------
    # Assign Z from nearest XY point in `other`
    # ----------------------------------------
    _, idx = nearest_xy(other, hull2d)
    hull3d = np.column_stack((hull2d, other[idx, 2]))

    # Write OFF
    off_path = outname if outname.lower().endswith(".off") else f"{outname}.off"
    extra = np.empty((0, 3), dtype=float)
    if samples is not None:
        extra = np.asarray(samples, dtype=float)
        if extra.size == 0:
            extra = np.empty((0, 3), dtype=float)
        else:
            extra = extra.reshape(-1, 3)
            # Hull vertices are already part of the polygon; avoid writing
            # exact duplicates as unreferenced vertices.
            hull_keys = {
                (round(float(x), 9), round(float(y), 9))
                for x, y in hull3d[:, :2]
            }
            keep = np.array(
                [
                    (round(float(x), 9), round(float(y), 9)) not in hull_keys
                    for x, y in extra[:, :2]
                ],
                dtype=bool,
            )
            extra = extra[keep]

    with open(off_path, "w") as f:
        f.write("OFF\n")
        f.write(f"{len(hull3d) + len(extra)} 1 0\n")
        for x, y, z in hull3d:
            f.write(f"{x:.12g} {y:.12g} {z:.12g}\n")
        for x, y, z in extra:
            f.write(f"{x:.12g} {y:.12g} {z:.12g}\n")
        face_idx = " ".join(map(str, range(len(hull3d))))
        f.write(f"{len(hull3d)} {face_idx}\n")

    return hull3d








if __name__ == "__main__":
    xyz_building, xyz_ground, coords = read_las_file("matera.las", class_building = 6, class_ground = 2)
    generate_ground_polygon(xyz_building, xyz_ground)
