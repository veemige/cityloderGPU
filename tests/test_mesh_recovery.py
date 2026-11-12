"""Recovery regressions: ring repair, transformed CityJSON, holes and terrain cutouts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import unittest
from collections import Counter
import numpy as np
import shapely as sg
from cityjson_to_mesh import export_cityjson, triangulate_surface, ground_triangles


class MeshRecoveryTests(unittest.TestCase):
    def test_surface_with_duplicate_vertices(self):
        points = np.array([[0.,0.,2.],[2.,0.,2.],[2.,2.,2.],[0.,2.,2.]])
        stats = Counter()
        triangles = list(triangulate_surface(points, [[0,1,1,2,3,0]], stats))
        self.assertEqual(stats['duplicate_ring_vertices'], 1)
        self.assertEqual(len(triangles), 2)
        area = sum(np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0])) / 2 for t in triangles)
        self.assertAlmostEqual(area, 4)

    def test_vertical_wall_orientation(self):
        points = np.array([[1.,0.,0.],[1.,3.,0.],[1.,3.,5.],[1.,0.,5.]])
        triangles = list(triangulate_surface(points, [[0,1,2,3]], Counter()))
        for face in triangles:
            self.assertGreater(np.cross(face[1]-face[0],face[2]-face[0])[0], 0)
        self.assertAlmostEqual(sum(np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0])) / 2 for t in triangles),15)

    def test_roof_hole_is_preserved(self):
        vertices = np.array([[0,0,10],[10,0,10],[10,10,10],[0,10,10],
                             [3,3,10],[7,3,10],[7,7,10],[3,7,10]],dtype=float)
        triangles = list(triangulate_surface(vertices, [[0,1,2,3],[4,5,6,7]], Counter()))
        self.assertAlmostEqual(sum(sg.Polygon(t[:,:2]).area for t in triangles),84.)
        for triangle in triangles:
            self.assertLess(sg.intersection(sg.Polygon(triangle[:,:2]),sg.box(3,3,7,7)).area,1e-10)

    def test_transform_and_duplicate_ring_export(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            model={'type':'CityJSON','vertices':[[0,0,0],[2,0,0],[2,2,0],[0,2,0]],
                   'transform':{'scale':[.5,.5,.5],'translate':[300000,7400000,2]},
                   'CityObjects':{'b':{'type':'Building','geometry':[{'type':'MultiSurface','lod':'1','boundaries':[[[0,1,1,2,3]]]}]}}}
            source=folder/'city.json';source.write_text(json.dumps(model))
            target=folder/'mesh.obj'
            stats=export_cityjson(source,target)
            self.assertEqual(stats['objects_exported'],1)
            self.assertEqual(stats['triangles'],2)
            coords=np.array([[float(v) for v in line.split()[1:]] for line in target.read_text().splitlines() if line.startswith('v ')])
            np.testing.assert_array_equal(coords.min(axis=0),[300000,7400000,2])
            np.testing.assert_array_equal(coords.max(axis=0),[300001,7400001,2])
            self.assertFalse(target.with_suffix('.obj.partial').exists())

    def test_tiled_terrain_keeps_samples_and_excludes_overlapping_footprints(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ground.off'
            points=[[0,0,1],[10,0,1],[10,10,1],[0,10,1],[1,1,7],[9,9,3]]
            path.write_text('OFF\n6 1 0\n'+'\n'.join(' '.join(map(str,p)) for p in points)+'\n4 0 1 2 3\n')
            footprints=[sg.box(3,3,6,7),sg.box(5,3,7,7)]
            faces=list(ground_triangles(path,footprints,5,Counter()))
            self.assertAlmostEqual(sum(sg.Polygon(t[:,:2]).area for t in faces),84.)
            self.assertTrue(any(np.array_equal(vertex,[1,1,7]) for face in faces for vertex in face))
            for face in faces:
                self.assertLess(sg.intersection(sg.Polygon(face[:,:2]),sg.union_all(footprints)).area,1e-10)


class TerrainConnectionTests(unittest.TestCase):
    def test_linear_height_field_does_not_create_nearest_neighbour_steps(self):
        from cityloder_mesh_repair import TerrainHeightField
        samples = np.array([[0.,0.,0.], [10.,0.,10.], [0.,10.,20.], [10.,10.,30.]])
        field = TerrainHeightField(samples)
        xy = np.array([[4.9, 5.], [5., 5.], [5.1, 5.], [2., 3.]])
        np.testing.assert_allclose(field(xy), xy[:, 0] + 2 * xy[:, 1], atol=1e-10)

    def test_cuda_barycentric_evaluation_matches_cpu_in_projected_coordinates(self):
        from cityloder_mesh_repair import TerrainHeightField
        import cityloder_gpu as gpu
        if not gpu.gpu_available():
            self.skipTest('CUDA unavailable')
        origin=np.array([300000.,7400000.])
        samples=np.array([[0.,0.,0.],[10.,0.,10.],[0.,10.,20.],[10.,10.,30.]])
        samples[:,:2]+=origin
        field=TerrainHeightField(samples)
        xy=np.random.default_rng(17).uniform(0,10,(4096,2))+origin
        np.testing.assert_allclose(field.interpolate(xy),field.linear(xy),atol=1e-8)
        self.assertIsNotNone(field.gpu_tin)

    def test_base_edge_overrides_terrain_height_at_new_intersection(self):
        from cityloder_mesh_repair import TerrainHeightField
        samples = np.array([[0.,0.,0.], [10.,0.,0.], [0.,10.,0.], [10.,10.,0.]])
        ring = np.array([[2.,2.,3.], [8.,2.,9.], [8.,8.,9.], [2.,8.,3.]])
        field = TerrainHeightField(samples, [ring])
        np.testing.assert_allclose(field(np.array([[5.,2.], [3.,2.], [8.,5.]])), [6.,4.,9.], atol=1e-10)

    def test_ground_t_junction_splits_building_and_ground_edges(self):
        from cityloder_mesh_repair import SharedMesh
        mesh = SharedMesh()
        # Two ground triangles meet at an interior point of a single wall edge.
        mesh.add('buildings', np.array([[0.,0.,2.], [2.,0.,2.], [0.,0.,4.]]))
        mesh.add('ground', np.array([[0.,0.,2.], [1.,0.,2.], [0.,-1.,2.]]))
        mesh.add('ground', np.array([[1.,0.,2.], [2.,0.,2.], [2.,-1.,2.]]))
        stats = Counter()
        mesh.conform([], stats)
        self.assertGreater(stats['split_shared_edges'],0)
        b_edges = set()
        for face in mesh.parts['buildings']:
            for a,b in zip(face,np.roll(face,-1)):
                b_edges.add(tuple(sorted((a,b))))
        middle = mesh.vertex([1.,0.,2.])
        self.assertIn(tuple(sorted((mesh.vertex([0.,0.,2.]),middle))),b_edges)
        self.assertIn(tuple(sorted((mesh.vertex([2.,0.,2.]),middle))),b_edges)
        self.assertNotIn(tuple(sorted((mesh.vertex([0.,0.,2.]),mesh.vertex([2.,0.,2.])))),b_edges)

    def test_tile_border_t_junction_is_conformed(self):
        from cityloder_mesh_repair import SharedMesh
        mesh = SharedMesh()
        mesh.add('ground', np.array([[0.,0.,0.], [2.,0.,0.], [0.,1.,0.]]))
        mesh.add('ground', np.array([[0.,0.,0.], [1.,0.,1.], [0.,-1.,0.]]))
        mesh.add('ground', np.array([[1.,0.,1.], [2.,0.,0.], [2.,-1.,0.]]))
        stats=Counter(); mesh.conform([],stats)
        edges=Counter(tuple(sorted((a,b))) for face in mesh.parts['ground'] for a,b in zip(face,np.roll(face,-1)))
        middle=mesh.vertex([1.,0.,1.])
        self.assertEqual(edges[tuple(sorted((mesh.vertex([0.,0.,0.]),middle)))],2)
        self.assertEqual(edges[tuple(sorted((mesh.vertex([2.,0.,0.]),middle)))],2)

    def test_three_outputs_are_the_same_repaired_triangles(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            vertices=[[3,3,3],[7,3,3],[7,7,3],[3,7,3],[3,3,7],[7,3,7],[7,7,7],[3,7,7]]
            faces=[[[0,3,2,1]],[[4,5,6,7]],[[0,1,5,4]],[[1,2,6,5]],[[2,3,7,6]],[[3,0,4,7]]]
            model={'type':'CityJSON','vertices':vertices,'CityObjects':{'b':{'type':'Building','geometry':[{
                'type':'Solid','lod':'2','boundaries':[faces], 'semantics':{
                'surfaces':[{'type':'FloorSurface'},{'type':'RoofSurface'},{'type':'WallSurface'}],
                'values':[[0,1,2,2,2,2]]}}]}}}
            source=folder/'city.json';source.write_text(json.dumps(model))
            off=folder/'ground.off'
            off.write_text('OFF\n5 1 0\n0 0 0\n10 0 0\n10 10 0\n0 10 0\n2 2 0\n4 0 1 2 3\n')
            target=folder/'mesh.obj'
            export_cityjson(source,target,off,5,True)
            def load(path):
                points=[]; triangles=[]
                for line in path.read_text().splitlines():
                    if line.startswith('v '): points.append(tuple(map(float,line.split()[1:])))
                    if line.startswith('f '): triangles.append(tuple(points[int(i)-1] for i in line.split()[1:]))
                return Counter(triangles)
            self.assertEqual(load(target),load(folder/'buildings_mesh.obj')+load(folder/'ground_mesh.obj'))
            def materials(path):
                return {line.split()[1] for line in path.read_text().splitlines() if line.startswith('usemtl ')}
            self.assertEqual(materials(target),{'walls','roofs','terrain'})
            self.assertEqual(materials(folder/'buildings_mesh.obj'),{'walls','roofs'})
            self.assertEqual(materials(folder/'ground_mesh.obj'),{'terrain'})
            mtl=(folder/'city_materials.mtl').read_text()
            for name in ('walls','roofs','terrain'):
                self.assertIn('newmtl '+name,mtl)
            for path in (target,folder/'buildings_mesh.obj',folder/'ground_mesh.obj'):
                self.assertIn('mtllib city_materials.mtl',path.read_text())
            # Every terrain vertex on the footprint rim has the building base Z.
            for face in load(folder/'ground_mesh.obj'):
                for x,y,z in face:
                    if ((abs(x-3)<1e-8 or abs(x-7)<1e-8) and 3<=y<=7 or
                        (abs(y-3)<1e-8 or abs(y-7)<1e-8) and 3<=x<=7):
                        self.assertAlmostEqual(z,3.)


class RunnerRecoveryTests(unittest.TestCase):
    def test_native_failure_recovers_without_reprocessing_points(self):
        from unittest.mock import patch
        from cityloder_runner import generate_mesh
        with patch('cityloder_runner.run_mesh', side_effect=RuntimeError('Triangle failed')), patch('cityjson_to_mesh.export_cityjson') as recover:
            generate_mesh('auto', Path('mesh.exe'), 'ground.off', Path('temp'), '', Path('city'))
            recover.assert_called_once_with('city.city.json', Path('city/mesh.obj'), 'ground.off', 1000., False)

    def test_explicit_triangle_failure_remains_visible(self):
        from unittest.mock import patch
        from cityloder_runner import generate_mesh
        with patch('cityloder_runner.run_mesh', side_effect=RuntimeError('Triangle failed')):
            with self.assertRaises(RuntimeError):
                generate_mesh('triangle', Path('mesh.exe'), 'ground.off', Path('temp'), '', Path('city'))


if __name__=='__main__':
    unittest.main()
