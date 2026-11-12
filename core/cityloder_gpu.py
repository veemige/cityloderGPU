"""Optional CUDA kernels used by the CityLODer reconstruction pipeline."""

from __future__ import annotations

import numpy as np
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

try:
    import cupy as cp
except ImportError:  # CUDA support is an optional local extra.
    cp = None


_PIP_KERNEL = None
_GPU_PIP_CALLS = 0
_GPU_PIP_POINTS = 0
_GPU_ROOF_BATCHES = 0
_MODE = "auto"
_STATS = Counter()
_WARNED = set()


class CudaExecutionError(RuntimeError):
    pass


def configure_gpu(mode="auto", memory_fraction=0.8):
    """Select numerical backend and bound the CuPy memory pool."""
    global _MODE
    if mode not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be auto, cuda or cpu")
    if not 0 < memory_fraction <= 1:
        raise ValueError("GPU memory fraction must be in (0, 1]")
    _MODE = mode
    if mode == "cuda" and not gpu_available():
        raise CudaExecutionError("CUDA requested, but CuPy cannot access a CUDA device")
    if gpu_available():
        cp.get_default_memory_pool().set_limit(fraction=memory_fraction)
        # Force compilation now so strict mode fails before reading the dataset.
        try:
            cp.sum(cp.ones(32)).item()
        except Exception as exc:
            gpu_failure("CUDA initialization", exc)


def gpu_failure(operation, exc):
    _STATS["fallbacks"] += 1
    if _MODE == "cuda":
        raise CudaExecutionError(f"CUDA failed during {operation}: {exc}") from exc
    if operation not in _WARNED:
        warnings.warn(f"{operation}: CPU fallback ({exc})", RuntimeWarning)
        _WARNED.add(operation)


def to_host(array):
    return cp.asnumpy(array) if cp is not None and isinstance(array, cp.ndarray) else np.asarray(array)


def prefetch_chunks(reader, chunk_size=1_000_000):
    """Overlap LAS/LAZ decoding with CUDA work, keeping at most two chunks."""
    iterator = reader.chunk_iterator(chunk_size)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="las-decode") as executor:
        pending = executor.submit(next, iterator, None)
        while True:
            record = pending.result()
            if record is None:
                break
            pending = executor.submit(next, iterator, None)
            yield record


def array_module(array):
    return cp if cp is not None and isinstance(array, cp.ndarray) else np


class GPUPointIndex:
    """One tile resident in VRAM, indexed by sorted X for bounded bbox scans."""

    def __init__(self, points):
        data = cp.asarray(points, dtype=cp.float64)
        self.points = data[cp.argsort(data[:, 0])]
        self.x = cp.ascontiguousarray(self.points[:, 0])
        self.ranges = {}
        _STATS["resident_points"] += len(points)

    def bbox(self, xmin, xmax, ymin, ymax):
        key = (xmin, xmax)
        if key in self.ranges:
            lo, hi = self.ranges[key]
        else:
            lo = int(cp.searchsorted(self.x, xmin, side="left").item())
            hi = int(cp.searchsorted(self.x, xmax, side="right").item())
        candidates = self.points[int(lo):hi]
        _STATS["bbox_queries"] += 1
        return candidates[(candidates[:, 1] >= ymin) & (candidates[:, 1] <= ymax)]

    def prepare(self, footprints, indices, margin_divisor):
        """Resolve all X ranges in one batch instead of synchronizing per box."""
        bounds = []
        for i in indices:
            xy = footprints[i]
            xmin, xmax = np.nanmin(xy[:, 0]), np.nanmax(xy[:, 0])
            margin = (xmax - xmin) / margin_divisor
            bounds.append((xmin - margin, xmax + margin))
        if bounds:
            data = cp.asarray(bounds)
            low = cp.asnumpy(cp.searchsorted(self.x, data[:, 0], side="left"))
            high = cp.asnumpy(cp.searchsorted(self.x, data[:, 1], side="right"))
            self.ranges = {key: (int(lo), int(hi)) for key, lo, hi in zip(bounds, low, high)}


def make_point_index(points):
    if not gpu_available() or len(points) == 0:
        return None
    try:
        return GPUPointIndex(points)
    except Exception as exc:
        gpu_failure("resident tile index", exc)
        return None


def polygon_mask_device(points, polygon):
    """Run PIP without moving the selected cloud back to the CPU."""
    global _GPU_PIP_CALLS, _GPU_PIP_POINTS
    points = cp.ascontiguousarray(points[:, :2], dtype=cp.float64)
    polygon = cp.ascontiguousarray(cp.asarray(polygon, dtype=cp.float64))
    result = cp.empty(len(points), dtype=cp.uint8)
    if len(points):
        _pip_kernel()(((len(points) + 255) // 256,), (256,),
                      (points, polygon, np.int32(len(polygon)), result, np.int32(len(points))))
    _GPU_PIP_CALLS += 1
    _GPU_PIP_POINTS += len(points)
    return result.astype(bool)


def roof_statistics(building):
    """Quantiles, reductions and normal equations stay on the cloud's device."""
    xp = array_module(building)
    zmax = float(xp.max(building[:, 2]).item())
    q = xp.quantile(building[:, 2], xp.asarray([0.65, 0.995]))
    roof = building[building[:, 2] >= q[0]]
    center = normal = rhs = None
    if len(roof) >= 6:
        # Reduce in local coordinates to avoid cancellation at UTM magnitudes.
        origin = roof[0, :2]
        local = roof[:, :2] - origin
        mean_local = xp.mean(local, axis=0)
        center_device = origin + mean_local
        design = xp.column_stack((local - mean_local, xp.ones(len(roof))))
        matrix = design.T @ design
        if int(xp.linalg.matrix_rank(matrix).item()) == 3:
            center = to_host(center_device)
            normal = to_host(matrix)
            rhs = to_host(design.T @ roof[:, 2])
    if xp is not np:
        _STATS["roof_statistics"] += 1
    return zmax, float(q[1].item()), center, normal, rhs


def nearest_xy(points, queries, distance_upper_bound=np.inf):
    """Exact nearest neighbours, with bounded GPU query batches."""
    if gpu_available() and len(points) >= 4096 and len(queries) >= 128:
        try:
            from cupyx.scipy.spatial import KDTree
            tree = KDTree(cp.asarray(np.ascontiguousarray(points[:, :2])))
            distances, indices = [], []
            for start in range(0, len(queries), 65536):
                d, i = tree.query(cp.asarray(queries[start:start + 65536]),
                                  distance_upper_bound=distance_upper_bound)
                distances.append(cp.asnumpy(d))
                indices.append(cp.asnumpy(i))
            _STATS["nearest_queries"] += len(queries)
            return np.concatenate(distances), np.concatenate(indices)
        except Exception as exc:
            gpu_failure("nearest neighbours", exc)
    _STATS["cpu_nearest_queries"] += len(queries)
    from scipy.spatial import cKDTree
    return cKDTree(points[:, :2]).query(queries, distance_upper_bound=distance_upper_bound, workers=-1)


def filter_classes(x, y, z, classification, bounds_min, bounds_max, building_class, ground_class):
    """Filter large decompressed LAS chunks on CUDA; transfer only survivors."""
    if not gpu_available() or len(x) < 65536:
        return None
    try:
        dx, dy, dz, labels = (cp.asarray(a) for a in (x, y, z, classification))
        inside = ((dx >= bounds_min[0]) & (dx <= bounds_max[0]) &
                  (dy >= bounds_min[1]) & (dy <= bounds_max[1]))
        results = []
        for code in (building_class, ground_class):
            mask = inside & (labels == code)
            results.append(cp.asnumpy(cp.column_stack((dx[mask], dy[mask], dz[mask]))))
        _STATS["filtered_points"] += len(x)
        return tuple(results)
    except Exception as exc:
        gpu_failure("LAS class/bounds filter", exc)
        return None


def gpu_available() -> bool:
    if cp is None or _MODE == "cpu":
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception as exc:
        return False


def _pip_kernel():
    global _PIP_KERNEL
    if _PIP_KERNEL is None:
        _PIP_KERNEL = cp.RawKernel(
            r"""
            extern "C" __global__
            void points_in_polygon(const double* points,
                                   const double* polygon,
                                   const int polygon_size,
                                   unsigned char* result,
                                   const int point_count)
            {
                const int point_id = blockDim.x * blockIdx.x + threadIdx.x;
                if (point_id >= point_count) return;

                const double x = points[2 * point_id];
                const double y = points[2 * point_id + 1];
                bool inside = false;
                for (int i = 0, j = polygon_size - 1; i < polygon_size; j = i++) {
                    const double xi = polygon[2 * i];
                    const double yi = polygon[2 * i + 1];
                    const double xj = polygon[2 * j];
                    const double yj = polygon[2 * j + 1];
                    const bool crosses = ((yi > y) != (yj > y));
                    if (crosses) {
                        const double x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi;
                        if (x < x_cross) inside = !inside;
                    }
                }
                result[point_id] = inside ? 1 : 0;
            }
            """,
            "points_in_polygon",
        )
    return _PIP_KERNEL


def points_in_polygon(points, polygon, cpu_contains, gpu_threshold=4096):
    """Return an inside mask, using a batched CUDA ray-cast for dense candidates."""
    global _GPU_PIP_CALLS, _GPU_PIP_POINTS

    points = np.asarray(points, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    if points.shape[0] < gpu_threshold or not gpu_available():
        return cpu_contains(points, polygon), False

    try:
        point_array = cp.asarray(np.ascontiguousarray(points))
        polygon_array = cp.asarray(np.ascontiguousarray(polygon))
        result = cp.empty(points.shape[0], dtype=cp.uint8)
        block_size = 256
        _pip_kernel()(
            ((points.shape[0] + block_size - 1) // block_size,),
            (block_size,),
            (point_array, polygon_array, np.int32(polygon.shape[0]), result, np.int32(points.shape[0])),
        )
        mask = cp.asnumpy(result).astype(bool)
        _GPU_PIP_CALLS += 1
        _GPU_PIP_POINTS += points.shape[0]
        return mask, True
    except Exception as exc:
        # Keep CUDA optional and fail over cleanly if the runtime is unavailable.
        gpu_failure("point in polygon", exc)
        return cpu_contains(points, polygon), False


def fit_roof_planes(normal_matrices, right_hand_sides):
    """Solve all centered 3-parameter roof-plane fits in one GPU batch."""
    global _GPU_ROOF_BATCHES

    normals = np.asarray(normal_matrices, dtype=np.float64)
    rhs = np.asarray(right_hand_sides, dtype=np.float64)
    if normals.size == 0:
        return np.empty((0, 3), dtype=np.float64), False

    if gpu_available():
        try:
            matrix = cp.asarray(normals)
            vector = cp.asarray(rhs)
            # Centering keeps the fit well-conditioned; tiny ridge terms handle
            # near-collinear roof samples without rejecting an entire batch.
            matrix[:, 0, 0] += 1e-10
            matrix[:, 1, 1] += 1e-10
            matrix[:, 2, 2] += 1e-12
            # CuPy expects a matrix right-hand side for batched solves.
            fitted = cp.linalg.solve(matrix, vector[..., None])[..., 0]
            cp.cuda.Stream.null.synchronize()
            _GPU_ROOF_BATCHES += 1
            return cp.asnumpy(fitted), True
        except Exception as exc:
            gpu_failure("roof plane solve", exc)

    fitted = np.empty((normals.shape[0], 3), dtype=np.float64)
    for i, (matrix, vector) in enumerate(zip(normals, rhs)):
        matrix = matrix.copy()
        matrix[0, 0] += 1e-10
        matrix[1, 1] += 1e-10
        matrix[2, 2] += 1e-12
        fitted[i] = np.linalg.solve(matrix, vector)
    return fitted, False


def gpu_summary():
    if not gpu_available():
        return "CUDA GPU path: unavailable; CPU fallback used"
    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties["name"].decode(errors="replace")
    pool = cp.get_default_memory_pool()
    allocated_gib = pool.total_bytes() / 1024**3
    limit_gib = pool.get_limit() / 1024**3
    return (
        f"CUDA GPU: {name}; point-in-polygon calls: {_GPU_PIP_CALLS} "
        f"({_GPU_PIP_POINTS:,} points); batched roof fits: {_GPU_ROOF_BATCHES}"
        f"; CUDA operations: {dict(_STATS)}"
        f"; VRAM pool: {allocated_gib:.2f} GiB / limit {limit_gib:.2f} GiB"
    )
