"""Numerical parity and CUDA integration checks (run with unittest discover)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
os.environ.setdefault('CUPY_CACHE_DIR', str(ROOT / '.cache' / 'cupy'))
import cityloder_gpu as gpu
from cityloder_compute_buildings import main as buildings


class CudaPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not gpu.gpu_available():
            raise unittest.SkipTest('CUDA device unavailable')
        gpu.configure_gpu('cuda')

    def setUp(self):
        gpu.configure_gpu('cuda')
        self.rng = np.random.default_rng(23)

    def test_resident_bbox_includes_edges(self):
        points = np.array([[0., 0., 1.], [1., 1., 3.], [1., 2., 5.], [2., 1., 7.]])
        index = gpu.make_point_index(points)
        index.prepare([np.array([[0., 0.], [1., 1.]])], [0], 500)
        actual = gpu.to_host(index.bbox(0., 1., 0., 1.))
        np.testing.assert_array_equal(actual, points[:2])
        self.assertEqual(len(index.bbox(3., 4., 0., 1.)), 0)

    def test_polygon_and_empty_selection(self):
        points = self.rng.random((20000, 3))
        polygon = np.array([[.1, .1], [.9, .1], [.4, .8]])
        actual = gpu.to_host(gpu.polygon_mask_device(gpu.cp.asarray(points), polygon))
        np.testing.assert_array_equal(actual, Polygon(polygon).contains_points(points[:, :2]))
        self.assertEqual(len(gpu.polygon_mask_device(gpu.cp.empty((0, 3)), polygon)), 0)

    def test_roof_statistics_and_solver_parity(self):
        points = self.rng.random((60000, 3)) + [300000., 7400000., 0.]
        points[:, 2] = (points[:, 0] - 300000) * .3 + (points[:, 1] - 7400000) * .2 + 10
        cpu = gpu.roof_statistics(points)
        cuda = gpu.roof_statistics(gpu.cp.asarray(points))
        for actual, expected in zip(cuda, cpu):
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
        fitted, used = gpu.fit_roof_planes([cuda[3]], [cuda[4]])
        self.assertTrue(used)
        np.testing.assert_allclose(fitted[0, :2], [.3, .2], atol=1e-7)

    def test_nearest_with_radius_and_no_neighbour(self):
        points = self.rng.random((10000, 3))
        queries = np.vstack((self.rng.random((300, 2)), [[10., 10.]]))
        distance, index = gpu.nearest_xy(points, queries, .1)
        expected, expected_index = cKDTree(points[:, :2]).query(queries, distance_upper_bound=.1)
        np.testing.assert_allclose(distance, expected, atol=1e-12)
        np.testing.assert_array_equal(index, expected_index)

    def test_class_filter_matches_cpu(self):
        points = self.rng.random((100000, 3))
        labels = self.rng.choice([2, 6, 1], len(points))
        actual = gpu.filter_classes(*points.T, labels, [0.2, 0.3], [0.8, 0.7], 6, 2)
        for result, code in zip(actual, [6, 2]):
            mask = ((points[:, 0] >= .2) & (points[:, 0] <= .8) &
                    (points[:, 1] >= .3) & (points[:, 1] <= .7) & (labels == code))
            np.testing.assert_array_equal(result, points[mask])

    def test_laz_prefetch_and_streamed_classification(self):
        import laspy
        from cityloder_processPC import read_las_file
        from cityloder_tiled import load_points
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / 'cloud.laz'
            las = laspy.LasData(laspy.LasHeader(point_format=3, version='1.2'))
            xyz = self.rng.uniform(0, 100, (70000, 3))
            las.x, las.y, las.z = xyz.T
            las.classification = np.where(np.arange(len(xyz)) % 2, 6, 2).astype(np.uint8)
            las.write(path)
            b, g, count, _ = read_las_file(path)
            self.assertEqual(count, 70000)
            self.assertEqual(len(b), 35000)
            self.assertEqual(len(g), 35000)
            bt, gt = load_points(path, np.array([0., 0.]), np.array([100., 100.]), 6, 2)
            np.testing.assert_array_equal(bt, b)
            np.testing.assert_array_equal(gt, g)

    def test_strict_mode_does_not_hide_failures(self):
        with self.assertRaises(gpu.CudaExecutionError):
            gpu.gpu_failure('test injected error', ValueError('test'))

    def test_building_export_cpu_cuda_parity(self):
        xy = self.rng.uniform(1, 9, (5000, 2))
        cloud = np.column_stack((xy, 10 + .2 * xy[:, 0] + .1 * xy[:, 1]))
        ground = np.column_stack((self.rng.uniform(-2, 12, (5000, 2)), np.ones(5000)))
        footprint = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.], [0., 0.], [np.nan, np.nan]])
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            for mode in ['cpu', 'cuda']:
                gpu.configure_gpu(mode)
                buildings(str(Path(folder) / mode), cloud, ground, [footprint], [[0]], [np.nan])
            for name in ['roof_polygon.off', 'pavement_polygon.off', 'facades.off']:
                cpu = np.loadtxt(Path(folder) / 'cpu' / 'building1' / name, skiprows=2, max_rows=4)
                cuda = np.loadtxt(Path(folder) / 'cuda' / 'building1' / name, skiprows=2, max_rows=4)
                np.testing.assert_allclose(cuda, cpu, atol=1e-7)
        gpu.configure_gpu('cuda')


if __name__ == '__main__':
    unittest.main()
