"""Recover a triangle OBJ from CityJSON and optional cached terrain, without Triangle."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import shapely as sg
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from cityloder_gpu import configure_gpu
from cityloder_mesh_repair import TerrainHeightField, SharedMesh
from cityloder_cityJSONgen import read_off


def polygon_parts(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == 'Polygon':
        yield geometry
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def surface_entries(geometry):
    kind = geometry['type']
    bounds = geometry['boundaries']
    semantic = geometry.get('semantics', {})
    values = semantic.get('values')
    definitions = semantic.get('surfaces', [])
    if kind in {'MultiSurface', 'CompositeSurface'}:
        shells = [bounds]
        labels = [values]
    elif kind == 'Solid':
        shells = bounds
        labels = values or [None] * len(shells)
    elif kind in {'MultiSolid', 'CompositeSolid'}:
        shells = [shell for solid in bounds for shell in solid]
        labels = [shell for solid in values for shell in solid] if values else [None] * len(shells)
    else:
        raise ValueError(f'Unsupported CityJSON geometry: {kind}')
    for shell, shell_labels in zip(shells, labels):
        for i, rings in enumerate(shell):
            label = shell_labels[i] if shell_labels else None
            meaning = definitions[label].get('type') if label is not None else None
            yield rings, meaning


def triangulate_surface(vertices, rings, stats):
    """Project to the dominant plane; preserve original XYZ at existing vertices."""
    cleaned = []
    for ring in rings:
        coords = vertices[np.asarray(ring, dtype=np.int64)]
        keep = np.ones(len(coords), dtype=bool)
        if len(coords) > 1:
            keep[1:] = np.any(coords[1:] != coords[:-1], axis=1)
        stats['duplicate_ring_vertices'] += int((~keep).sum())
        coords = coords[keep]
        if len(coords) > 1 and np.array_equal(coords[0], coords[-1]):
            coords = coords[:-1]
        if len(coords) >= 3:
            cleaned.append(coords)
    if not cleaned:
        stats['degenerate_surfaces'] += 1
        return
    outer = cleaned[0]
    origin = outer[0]
    local = outer - origin
    normal = np.sum(np.cross(local, np.roll(local, -1, axis=0)), axis=0)
    if np.linalg.norm(normal) < 1e-12:
        stats['degenerate_surfaces'] += 1
        return
    drop = int(np.argmax(np.abs(normal)))
    axes = [a for a in range(3) if a != drop]
    projected = [coords[:, axes] - origin[axes] for coords in cleaned]
    polygon = sg.Polygon(projected[0], projected[1:])
    if not polygon.is_valid:
        polygon = sg.make_valid(polygon)
        stats['repaired_surfaces'] += 1
    # New intersection vertices use the face plane; unchanged vertices keep XYZ.
    lookup = {tuple(uv): xyz for ring, uv_ring in zip(cleaned, projected) for uv, xyz in zip(uv_ring, ring)}
    plane = None
    for part in polygon_parts(polygon):
        triangles = sg.get_parts(sg.constrained_delaunay_triangles(part))
        for triangle in triangles:
            uv = np.asarray(triangle.exterior.coords)[:3]
            xyz = np.empty((3, 3))
            for j, point in enumerate(uv):
                known = lookup.get(tuple(point))
                if known is not None:
                    xyz[j] = known
                else:
                    if plane is None:
                        design = np.column_stack((projected[0], np.ones(len(outer))))
                        plane = np.linalg.lstsq(design, outer[:, drop] - origin[drop], rcond=None)[0]
                    xyz[j, axes] = point + origin[axes]
                    xyz[j, drop] = point @ plane[:2] + plane[2] + origin[drop]
            if np.dot(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0]), normal) < 0:
                xyz = xyz[[0, 2, 1]]
            yield xyz


MATERIAL_COLORS = {
    'walls': (0.74, 0.70, 0.61),
    'roofs': (0.65, 0.20, 0.12),
    'terrain': (0.30, 0.48, 0.27),
    'roads': (0.28, 0.29, 0.31),
}


class ObjWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_suffix(self.path.suffix + '.partial')
        self.file = self.temporary.open('w', encoding='utf-8', buffering=1024 * 1024)
        self.file.write('# CityLODer recovery mesh; original CRS coordinates\nmtllib city_materials.mtl\no city\n')
        self.active_material = None
        self.vertices = {}
        self.faces = 0
        self.degenerate = 0

    def triangle(self, xyz, material='walls'):
        keys = [tuple(np.round(point, 8)) for point in xyz]
        if len(set(keys)) < 3 or np.linalg.norm(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])) < 1e-12:
            self.degenerate += 1
            return
        indices = []
        for key in keys:
            index = self.vertices.get(key)
            if index is None:
                index = len(self.vertices) + 1
                self.vertices[key] = index
                self.file.write(f'v {key[0]:.8f} {key[1]:.8f} {key[2]:.8f}\n')
            indices.append(index)
        if material != self.active_material:
            self.file.write(f'usemtl {material}\n')
            self.active_material = material
        self.file.write(f'f {indices[0]} {indices[1]} {indices[2]}\n')
        self.faces += 1

    def finish(self):
        self.file.close()
        mtl_path = self.path.parent / 'city_materials.mtl'
        contents = ['# CityLODer semantic surface colors']
        for material, rgb in MATERIAL_COLORS.items():
            contents.extend([f'newmtl {material}', 'Ka 0.12 0.12 0.12',
                             'Kd ' + ' '.join(f'{v:.4f}' for v in rgb),
                             'Ks 0.04 0.04 0.04', 'Ns 16', 'd 1', 'illum 2', ''])
        mtl_path.write_text('\n'.join(contents), encoding='utf-8')
        self.temporary.replace(self.path)


def ground_triangles(off_path, footprints, tile_size, stats, base_rings=()):
    """Tile, node/clip with GEOS, retaining cached terrain sample coordinates."""
    data = read_off(off_path)
    points = np.asarray(data['vertices'], dtype=float)
    outer = sg.Polygon(points[np.asarray(data['geometry'][0]), :2])
    outer = sg.make_valid(outer)
    terrain_tree = cKDTree(points[:, :2])
    heights = TerrainHeightField(points, base_rings)
    building_tree = sg.STRtree(footprints)
    x0, y0, x1, y1 = outer.bounds
    total = int(np.ceil((x1 - x0) / tile_size) * np.ceil((y1 - y0) / tile_size))
    with tqdm(total=total, desc='Recovering terrain tiles') as progress:
        for x in np.arange(x0, x1, tile_size):
            for y in np.arange(y0, y1, tile_size):
                tile = sg.box(x, y, min(x + tile_size, x1), min(y + tile_size, y1))
                terrain = sg.intersection(outer, tile)
                candidates = building_tree.query(tile, predicate='intersects')
                if len(candidates):
                    cutouts = sg.union_all([footprints[i] for i in candidates])
                    terrain = sg.difference(terrain, cutouts)
                if terrain.is_empty:
                    progress.update()
                    continue
                # The same cached heights and vertices are used on both sides of a tile border.
                center = [x + tile_size / 2, y + tile_size / 2]
                sample_ids = terrain_tree.query_ball_point(center, tile_size / np.sqrt(2) + 1e-8)
                samples = points[sample_ids, :2]
                inside = sg.intersects_xy(terrain, samples[:, 0], samples[:, 1])
                samples = samples[inside]
                combined = sg.GeometryCollection([terrain, sg.MultiPoint(samples)])
                raw = sg.get_parts(sg.delaunay_triangles(combined))
                sg.prepare(terrain)
                covered = sg.covers(terrain, raw)
                intersecting = (~covered) & sg.intersects(terrain, raw)
                triangles = list(raw[covered])
                for fragment in sg.intersection(raw[intersecting], terrain):
                    for part in polygon_parts(fragment):
                        if part.area > 1e-12:
                            triangles.extend(sg.get_parts(sg.constrained_delaunay_triangles(part)))
                if triangles:
                    xy = np.concatenate([np.asarray(t.exterior.coords)[:3] for t in triangles])
                    unique_xy, inverse = np.unique(xy, axis=0, return_inverse=True)
                    z = heights(unique_xy)[inverse]
                    xyz = np.column_stack((xy, z)).reshape(-1, 3, 3)
                    for face in xyz:
                        if np.cross(face[1] - face[0], face[2] - face[0])[2] < 0:
                            face = face[[0, 2, 1]]
                        yield face
                    stats['terrain_triangles'] += len(xyz)
                progress.update()


def export_cityjson(source, output, ground_off=None, tile_size=1000., save_parts=False):
    if not np.isfinite(tile_size) or tile_size <= 0:
        raise ValueError('tile-size must be positive')
    if not hasattr(sg, 'constrained_delaunay_triangles'):
        raise RuntimeError('Recovery requires Shapely >= 2.1 with GEOS >= 3.10')
    with Path(source).open(encoding='utf-8') as file:
        model = json.load(file)
    if model.get('type') != 'CityJSON':
        raise ValueError('Expected CityJSON')
    vertices = np.asarray(model['vertices'], dtype=float)
    transform = model.get('transform', {})
    vertices = vertices * transform.get('scale', [1, 1, 1]) + transform.get('translate', [0, 0, 0])
    writer = ObjWriter(output)
    stats = Counter()
    parts = {}
    if save_parts:
        parts['buildings'] = ObjWriter(Path(output).parent / 'buildings_mesh.obj')
        if ground_off:
            parts['ground'] = ObjWriter(Path(output).parent / 'ground_mesh.obj')
    footprints = []
    base_rings = []
    mesh = SharedMesh()
    if ground_off:
        base_indices = set()
        for obj in model['CityObjects'].values():
            if obj.get('type') not in {'Building', 'BuildingPart'} or not obj.get('geometry'):
                continue
            geometry = max(obj['geometry'], key=lambda g: float(g.get('lod', 0)))
            for rings, meaning in surface_entries(geometry):
                if meaning in {'FloorSurface', 'GroundSurface'}:
                    base_indices.update(i for ring in rings for i in ring)
        if base_indices:
            ids = np.array(sorted(base_indices), dtype=np.int64)
            xy = np.round(vertices[ids, :2], 6)
            _, groups = np.unique(xy, axis=0, return_inverse=True)
            means = np.bincount(groups, weights=vertices[ids, 2]) / np.bincount(groups)
            stats['reconciled_base_vertices'] = int(np.count_nonzero(np.abs(vertices[ids, 2] - means[groups]) > 1e-6))
            vertices[ids, 2] = means[groups]
    try:
        for name, obj in tqdm(model['CityObjects'].items(), desc='Recovering CityJSON surfaces'):
            before = len(mesh.parts['buildings'])
            geometries = obj.get('geometry', [])
            # Choose one LOD, preventing superposed copies when multiple LODs exist.
            if geometries:
                geometry = max(geometries, key=lambda g: float(g.get('lod', 0)))
                for rings, meaning in surface_entries(geometry):
                    if ground_off and obj['type'] in {'Building', 'BuildingPart'} and meaning in {'FloorSurface', 'GroundSurface'}:
                        coords = [vertices[np.asarray(r), :2] for r in rings if len(r) >= 3]
                        if coords:
                            footprint = sg.make_valid(sg.Polygon(coords[0], coords[1:]))
                            footprints.extend(polygon_parts(footprint))
                            base_rings.extend(vertices[np.asarray(r)] for r in rings if len(r) >= 3)
                    for triangle in triangulate_surface(vertices, rings, stats):
                        material = 'roofs' if meaning == 'RoofSurface' else 'roads' if obj.get('type') == 'Road' else 'walls'
                        mesh.add('buildings', triangle, material)
            if len(mesh.parts['buildings']) > before:
                stats['objects_exported'] += 1
            else:
                stats['objects_empty'] += 1
        # Release the parsed JSON before terrain triangulation.
        del model
        if ground_off:
            if not footprints:
                raise ValueError('No FloorSurface/GroundSurface found to cut terrain around buildings')
            for triangle in ground_triangles(ground_off, footprints, tile_size, stats, base_rings):
                mesh.add('ground', triangle)
            print('Conforming terrain seams and building base edges ...')
            mesh.conform(base_rings, stats)
        for part in ('buildings', 'ground'):
            for triangle, material in mesh.triangles(part, with_materials=True):
                writer.triangle(triangle, material)
                if part in parts:
                    parts[part].triangle(triangle, material)
        stats['buildings_triangles'] = len(mesh.parts['buildings'])
        stats['ground_triangles'] = len(mesh.parts['ground'])
        stats['vertices'] = len(writer.vertices)
        stats['triangles'] = writer.faces
        stats['degenerate_triangles_skipped'] = writer.degenerate
        writer.finish()
        for part in parts.values():
            part.finish()
    except BaseException:
        writer.file.close()
        for part in parts.values():
            part.file.close()
        raise
    report = Path(output).with_suffix('.report.json')
    report.write_text(json.dumps(dict(stats), indent=2), encoding='utf-8')
    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cityjson', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ground-off', help='Cached ground_polygon.off; omit for CityJSON objects only')
    parser.add_argument('--tile-size', type=float, default=1000.)
    parser.add_argument('--save-parts', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = parser.parse_args()
    configure_gpu(args.device)
    start = time.perf_counter()
    stats = export_cityjson(args.cityjson, args.output, args.ground_off, args.tile_size, args.save_parts)
    print(json.dumps(stats, indent=2))
    print(f'Mesh saved: {args.output} ({time.perf_counter() - start:.1f}s)')


if __name__ == '__main__':
    main()
