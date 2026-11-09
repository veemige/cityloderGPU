"""Optional CUDA kernels used by the CityLODer reconstruction pipeline."""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ImportError:  # CUDA support is an optional local extra.
    cp = None


_PIP_KERNEL = None
_GPU_PIP_CALLS = 0
_GPU_PIP_POINTS = 0
_GPU_ROOF_BATCHES = 0


def gpu_available() -> bool:
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
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
    except Exception:
        # Keep CUDA optional and fail over cleanly if the runtime is unavailable.
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
        except Exception:
            pass

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
    return (
        f"CUDA GPU: {name}; point-in-polygon calls: {_GPU_PIP_CALLS} "
        f"({_GPU_PIP_POINTS:,} points); batched roof fits: {_GPU_ROOF_BATCHES}"
    )
