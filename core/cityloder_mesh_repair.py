"""Continuous terrain elevation and shared-edge repair for recovered city meshes."""
import numpy as np
import shapely as sg
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from cityloder_gpu import gpu_available, gpu_failure, cp


PRECISION = 6  # Micrometre welding in projected map units.
TOLERANCE = 2e-6


class TerrainHeightField:
    def __init__(self, samples, base_rings=()):
        self.segments = []
        anchors = []
        for ring in base_rings:
            ring = np.asarray(ring, dtype=float)
            anchors.extend(ring)
            for a, b in zip(ring, np.roll(ring, -1, axis=0)):
                if np.linalg.norm(b[:2] - a[:2]) > TOLERANCE:
                    self.segments.append((a, b))
        data = np.vstack((np.asarray(anchors).reshape(-1, 3), samples))
        # Base anchors take priority over a terrain sample at the same XY.
        _, first = np.unique(np.round(data[:, :2], PRECISION), axis=0, return_index=True)
        data = data[first]
        self.data = data
        self.linear = LinearNDInterpolator(data[:, :2], data[:, 2])
        self.gpu_tin = None
        self.nearest = cKDTree(data[:, :2])
        self.segments = np.asarray(self.segments, dtype=float).reshape(-1, 2, 3)
        self.tree = sg.STRtree(sg.linestrings(self.segments[:, :, :2]))

    def interpolate(self, xy):
        if gpu_available() and len(xy) >= 4096:
            try:
                tri = self.linear.tri
                if self.gpu_tin is None:
                    self.gpu_tin = (cp.asarray(tri.transform), cp.asarray(self.data[tri.simplices, 2]))
                transforms, elevations = self.gpu_tin
                simplex = tri.find_simplex(xy)
                valid = simplex >= 0
                output = np.full(len(xy), np.nan)
                indices = cp.asarray(simplex[valid])
                transform = transforms[indices]
                delta = cp.asarray(xy[valid]) - transform[:, 2]
                weights = cp.einsum('nij,nj->ni', transform[:, :2], delta)
                values = elevations[indices]
                z = cp.sum(weights * values[:, :2], axis=1) + (1 - weights.sum(axis=1)) * values[:, 2]
                output[valid] = cp.asnumpy(z)
                return output
            except Exception as exc:
                gpu_failure('terrain TIN interpolation', exc)
        return np.asarray(self.linear(xy)).reshape(-1)

    def __call__(self, xy):
        xy = np.asarray(xy, dtype=float)
        heights = self.interpolate(xy)
        missing = ~np.isfinite(heights)
        if missing.any():
            _, ids = self.nearest.query(xy[missing], workers=-1)
            heights[missing] = self.data[ids, 2]
        # Terrain vertices on a building edge use exactly that edge's interpolated
        # height, including new vertices introduced by tile clipping.
        if len(self.segments) and len(xy):
            query, segment = self.tree.query(sg.points(xy), predicate='dwithin', distance=TOLERANCE)
            if len(query):
                edges = self.segments[segment]
                delta = edges[:, 1, :2] - edges[:, 0, :2]
                t = np.clip(np.sum((xy[query] - edges[:, 0, :2]) * delta, axis=1) /
                            np.sum(delta * delta, axis=1), 0, 1)
                values = edges[:, 0, 2] + t * (edges[:, 1, 2] - edges[:, 0, 2])
                # Deterministic selection at vertices shared by several rings.
                order = np.lexsort((segment, query))
                q = query[order]
                unique = np.r_[True, q[1:] != q[:-1]]
                heights[q[unique]] = values[order[unique]]
        return heights


class SharedMesh:
    """One vertex pool for both parts; split T junctions before writing any OBJ."""
    def __init__(self):
        self.vertex_map = {}
        self.vertices = []
        self.parts = {'buildings': [], 'ground': []}
        self.materials = {'buildings': [], 'ground': []}

    def vertex(self, xyz):
        key = tuple(np.round(xyz, PRECISION))
        index = self.vertex_map.get(key)
        if index is None:
            index = len(self.vertices)
            self.vertex_map[key] = index
            self.vertices.append(key)
        return index

    def add(self, part, triangle, material=None):
        ids = [self.vertex(p) for p in triangle]
        if len(set(ids)) == 3:
            self.parts[part].append(ids)
            self.materials[part].append(material or ('terrain' if part == 'ground' else 'walls'))

    def conform(self, base_rings, stats):
        ground = np.asarray(self.parts['ground'], dtype=np.int64).reshape(-1, 3)
        if not len(ground):
            return
        vertices = np.asarray(self.vertices)
        edges = np.sort(np.concatenate((ground[:, [0, 1]], ground[:, [1, 2]], ground[:, [2, 0]])), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_ids = np.unique(unique[counts == 1])
        candidates = set(boundary_ids.tolist())
        for ring in base_rings:
            for p in ring:
                candidates.add(self.vertex(p))
        vertices = np.asarray(self.vertices)
        candidate_ids = np.array(sorted(candidates), dtype=np.int64)
        point_tree = sg.STRtree(sg.points(vertices[candidate_ids, :2]))
        del edges, unique, counts, ground
        splits = {}
        for part, faces in self.parts.items():
            faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
            edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
            edges = np.unique(edges, axis=0)
            for start in range(0, len(edges), 32768):
                batch = edges[start:start + 32768]
                coords = vertices[batch]
                xy_min = coords[:, :, :2].min(axis=1) - TOLERANCE
                xy_max = coords[:, :, :2].max(axis=1) + TOLERANCE
                edge_ids, point_ids = point_tree.query(sg.box(xy_min[:, 0], xy_min[:, 1], xy_max[:, 0], xy_max[:, 1]))
                if not len(edge_ids):
                    continue
                ids = candidate_ids[point_ids]
                a = coords[edge_ids, 0]
                delta = coords[edge_ids, 1] - a
                # A nonlinear terrain profile may bend between two endpoints.
                # Split its XY edge at the neighbouring tile's actual height;
                # requiring 3D collinearity would retain that vertical crack.
                dims = 2 if part == 'ground' else 3
                length2 = np.sum(delta[:, :dims] * delta[:, :dims], axis=1)
                safe = length2 > TOLERANCE**2
                t = np.zeros(len(ids))
                t[safe] = np.sum((vertices[ids[safe], :dims] - a[safe, :dims]) * delta[safe, :dims], axis=1) / length2[safe]
                projected = a[:, :dims] + t[:, None] * delta[:, :dims]
                valid = safe & (t > 1e-9) & (t < 1 - 1e-9)
                valid &= np.linalg.norm(projected - vertices[ids, :dims], axis=1) <= TOLERANCE
                for e, point, fraction in zip(edge_ids[valid], ids[valid], t[valid]):
                    key = tuple(batch[e])
                    splits.setdefault(key, {})[int(point)] = float(fraction)
            rebuilt = []
            rebuilt_materials = []
            for face, material in zip(faces, self.materials[part]):
                ring = []
                changed = False
                for a, b in zip(face, np.roll(face, -1)):
                    ring.append(int(a))
                    key = (min(a, b), max(a, b))
                    if key in splits:
                        added = sorted(splits[key], key=splits[key].get, reverse=a > b)
                        ring.extend(added)
                        changed = True
                if changed:
                    center = self.vertex(vertices[face].mean(axis=0))
                    for a, b in zip(ring, ring[1:] + ring[:1]):
                        if a != b:
                            rebuilt.append([a, b, center])
                            rebuilt_materials.append(material)
                    stats['conformed_triangles'] += 1
                else:
                    rebuilt.append(face.tolist())
                    rebuilt_materials.append(material)
            self.parts[part] = rebuilt
            self.materials[part] = rebuilt_materials
        stats['split_shared_edges'] = len(splits)
        stats['welded_vertex_pool'] = len(self.vertices)

    def triangles(self, part, with_materials=False):
        vertices = np.asarray(self.vertices)
        for face, material in zip(self.parts[part], self.materials[part]):
            yield (vertices[face], material) if with_materials else vertices[face]
