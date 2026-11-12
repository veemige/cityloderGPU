import os
import numpy as np
import shapefile

from matplotlib.path import Path as MatplotlibPath  # for CPU point-in-polygon
from scipy.spatial import cKDTree
from tqdm import tqdm

from cityloder_gpu import (fit_roof_planes, gpu_summary, points_in_polygon,
    make_point_index, polygon_mask_device, to_host, array_module, roof_statistics,
    nearest_xy, gpu_failure, GPUPointIndex, CudaExecutionError)


## -------------------------------------------------------------------------------
## -------------------- Subfunctions (Helper/Utility Functions) ------------------
## -------------------------------------------------------------------------------

def read_shapefile_to_dict(shapefile_path):
    """
    Reads a shapefile and returns a MATLAB-style structure array with NaNs
    separating parts of polygons/polylines, similar to MATLAB's shaperead.
    
    Parameters:
        shp_path (str): Path to the shapefile (without .shp extension required)
        
    Returns:
        M (list of dict): Each element corresponds to a shape record with:
            - 'Geometry': shape type ('Polygon', 'Polyline', etc.)
            - 'X': x-coordinates with NaN separators
            - 'Y': y-coordinates with NaN separators
            - other attributes from the shapefile record
    """
    sf = shapefile.Reader(shapefile_path)
    M = []
    
    try:
        shape_records = sf.shapeRecords()
    except shapefile.ShapefileException as exc:
        if "DbfReader requires" not in str(exc):
            raise
        shape_records = [(shape, {}) for shape in sf.iterShapes()]

    for shape_rec in shape_records:
        if isinstance(shape_rec, tuple):
            shape, rec = shape_rec
        else:
            shape = shape_rec.shape
            rec = shape_rec.record.as_dict() if hasattr(shape_rec.record, "as_dict") else dict(zip([field[0] for field in sf.fields[1:]], shape_rec.record))
        
        # Initialize output coordinates
        x_coords = []
        y_coords = []
        
        # Handle only shapes with points (ignore NullShapes)
        if shape.points:
            parts = list(shape.parts) + [len(shape.points)]  # add end index
            for i in range(len(shape.parts)):
                start_idx = parts[i]
                end_idx = parts[i+1]
                part = shape.points[start_idx:end_idx]
                if part:
                    xs, ys = zip(*part)
                    x_coords.extend(xs)
                    y_coords.extend(ys)
                    # Insert NaN between parts
                    x_coords.append(np.nan)
                    y_coords.append(np.nan)
        
        # Build MATLAB-style structure
        shape_struct = {
            'Geometry': shape.shapeTypeName,  # e.g., 'Polygon', 'Polyline'
            'X': np.array(x_coords),
            'Y': np.array(y_coords)
        }
        
        # Add attributes
        for k, v in rec.items():
            shape_struct[k] = v
        
        M.append(shape_struct)
    
    return M

def _points_in_bbox(points, tree, x_min, x_max, y_min, y_max):
    if isinstance(tree, GPUPointIndex):
        return tree.bbox(x_min, x_max, y_min, y_max)
    if tree is None:
        candidates = points
    else:
        center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
        radius = np.hypot((x_max - x_min) / 2.0, (y_max - y_min) / 2.0)
        indices = tree.query_ball_point(center, radius)
        if not indices:
            return np.empty((0, points.shape[1]), dtype=points.dtype)
        candidates = points[indices]

    mask = (
        (candidates[:, 0] >= x_min) & (candidates[:, 0] <= x_max) &
        (candidates[:, 1] >= y_min) & (candidates[:, 1] <= y_max)
    )
    return candidates[mask]


def PointSelectionAndDataStructure(xyz=None, M_i=None, other=None, xyz_tree=None, other_tree=None):
    """
    Combine las point data and a single shapefile data to indentify the 
    points of interest.
    
    Parameters:
        xyz (np.ndarray): ?x3 array of XYZ coordinates.
        M_i (dictionary): one of the dictionaries from read_shapefile_to_dict
        other (np.ndarray): ?x3 array of XYZ coordinates.
        
    Returns:
        other_aus (np.ndarray): ?x3 array of xyz coordinates of the points around the footprints
        X (np.array): ?-dimensional array with x-coordinates of the footprints
        Y (np.array): ?-dimensional array with y-coordinates of the footprints
        building (np.ndarray): refined ?x3 array of xyz coordinates of the points shaping the building
    """
    # building and numpy arrays
    X = M_i[:,0] 
    Y = M_i[:,1]


    # ----
    # Remove duplicate points (can happen in some footprint shapefiles)
    # use_these starts as all True
    use_these = np.ones_like(X, dtype=bool)

    # Loop from index 1 to end (since MATLAB is 1-based)
    for j in range(1, len(X)):
        if not np.isnan(X[j]) and X[j] == X[j-1] and Y[j] == Y[j-1]:
            use_these[j] = False

    X = X[use_these]
    Y = Y[use_these]
    # ----

    control = np.where(np.isnan(X))[0]
    xy = np.column_stack((X, Y))  # shape will be (n, 2)
    xMin = np.nanmin(X)
    xMax = np.nanmax(X)
    yMin = np.nanmin(Y)
    yMax = np.nanmax(Y)
    sogliaX = (xMax - xMin) / 500
    sogliaY = (yMax - yMin) / 500

    # Restrict xyz pointcloud to make elaboration quicker

    building = _points_in_bbox(
        xyz, xyz_tree,
        xMin - sogliaX, xMax + sogliaX,
        yMin - sogliaY, yMax + sogliaY,
    )

    sogliaX_aus = (xMax - xMin) / 4
    sogliaY_aus = (yMax - yMin) / 4
    other_aus = _points_in_bbox(
        other, other_tree,
        xMin - sogliaX_aus, xMax + sogliaX_aus,
        yMin - sogliaY_aus, yMax + sogliaY_aus,
    )

    # Initialize label array to properly assign polygons with holes.
    label = np.full(X.size, 1, dtype=float)  # i*ones(numel(X),2)
    lab = 0

    # Fill second column of label
    for k in range(X.size):  # Python indexing starts at 0
        if np.any(control == k): 
            lab += 1
        label[k] = lab

    # Select xy points where label's second column is 0
    pip = xy[label == 0, :]

    if pip.shape[0] > 1:  # Need at least 2 points for inpolygon
        # Dense polygons use a CUDA ray-cast kernel when CuPy is installed.
        polygon = pip[:-1, :]
        if array_module(building) is not np:
            in_mask = polygon_mask_device(building, polygon)
        else:
            in_mask, _ = points_in_polygon(
                building[:, :2], polygon,
                lambda pts, poly: MatplotlibPath(poly).contains_points(pts),
            )
        # An empty polygon selection is empty, not the entire bounding box.
        building = building[in_mask, :]



    return to_host(other_aus),X,Y,building

##

def DefineLabels(X, i):
    """
    Define labels for proper polygon-hole parsing.

    Parameters:
        X (array-like): Input array of numbers (can contain NaN)
        i (int/float): Label identifier
    
    Returns:
        np.ndarray: lab array with shape (N, 2)
    """
    X = np.array(X, dtype=float)
    lab = []
    l = 0

    for x in X:
        if np.isnan(x):
            if lab:  # only remove last row if lab is not empty
                lab.pop()
            l += 1
        else:
            lab.append([i, l])
    
    lab = np.array(lab)
    return lab

##

def DefineTR(lab, N_abs):
    """   
    Define triangles (OFF faces).

    Parameters:
        lab (np.ndarray): lab array of shape (N, 2)
        N_abs (int): offset to be added to indices
    
    Returns:
        np.ndarray: tr array (Nx4) with 0-based indexing
    """
    tr = []
    N = 0

    for k in range(lab[:, 1].max() + 1):  # loop over unique labels
        indices_k = np.where(lab[:, 1] == k)[0]
        N_aus = len(indices_k) - 1

        for s in range(1, N_aus+1):  # MATLAB 1:N_aus-1
            tr.append([N + s + N_abs, N + s + N_abs + 1, s + 1 + N, s + N])  # careful with offsets
            # Python is 0-based, adjust if needed

        # Append the extra row after the loop
        tr.append([N + 1 + N_abs, 1 + N, N + s + 1, N+s+1+N_abs])

        N += N_aus + 1

    tr = np.array(tr, dtype=int) - 1  # convert to 0-based indexing like MATLAB
    return tr

##

def exportFacades(tr, points, dir_name):
    """
    Export mesh facades to an OFF file.

    Parameters:
    - tr: numpy array of shape (n_faces, 4) containing face indices
    - points: numpy array of shape (n_vertices, 3) containing vertex coordinates
    - dir_name: string, directory path + filename prefix
    """
    
    # Create the array like in MATLAB: first column 4, rest are tr
    aus = np.ones((tr.shape[0], 5), dtype=int) * 4
    aus[:, 1:5] = tr
    
    name_file = f"{dir_name}/facades.off"
    
    with open(name_file, 'w') as f:
        # Write OFF header
        f.write("OFF\n")
        # Write number of vertices, faces, edges
        f.write(f"{points.shape[0]} {tr.shape[0]} 0\n")
        # Write vertex coordinates
        for p in points:
            f.write(f"{p[0]:12.8f} {p[1]:12.8f} {p[2]:12.8f}\n")
        # Write faces
        for face in aus:
            f.write(f"{face[0]:5d} {face[1]:5d} {face[2]:5d} {face[3]:5d} {face[4]:5d}\n")

##

def exportPavement(points_build_f, lab, dirNames):
    """
    Export mesh pavement to an OFF file.

    Parameters:
    - points_build_f: vertices of the OFF file
    - lab: lab from DefineLabels that define polygon-holes structure
    - dirNames: string, directory path + filename prefix
    """

    aus = np.arange(points_build_f.shape[0])
    nameFile = f"{dirNames}/pavement_polygon.off"

    with open(nameFile, 'w') as fileID:
        fileID.write('OFF\n')
        fileID.write(f"{points_build_f.shape[0]} {int(np.max(lab[:, 1])) + 1} 0\n")

        # Write vertex coordinates
        for point in points_build_f:
            fileID.write(f"{point[0]:12.8f} {point[1]:12.8f} {point[2]:12.8f}\n")

        # Write face indices
        max_label = int(np.max(lab[:, 1]))
        for k in range(max_label + 1):
            indices = aus[lab[:, 1] == k]
            fileID.write(f"{len(indices)} ")
            
            if k > 0:
                for idx in indices:
                    fileID.write(f"{idx} ")
            else:
                for idx in indices[::-1]:  # reverse for k=0
                    fileID.write(f"{idx} ")
            
            fileID.write('\n')
        fileID.write('\n')

##

def exportRoof(points_build_r, lab, dirNames):
    """
    Export mesh roof to an OFF file.

    Parameters:
    - points_build_f: vertices of the OFF file
    - lab: lab from DefineLabels that define polygon-holes structure
    - dirNames: string, directory path + filename prefix
    """

    aus = np.arange(points_build_r.shape[0])
    nameFile = f"{dirNames}/roof_polygon.off"
    
    with open(nameFile, 'w') as fileID:
        # Write the OFF header
        fileID.write('OFF\n')
        
        # Write counts: number of vertices, number of faces, 0
        fileID.write(f"{points_build_r.shape[0]} {int(np.max(lab[:,1]))+1} 0\n")
        
        # Write the vertex coordinates
        for point in points_build_r:
            fileID.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f}\n")
        
        # Write the face indices
        for k in range(int(np.max(lab[:,1]))+1):
            indices = aus[lab[:,1] == k]
            fileID.write(f"{len(indices)} ")
            
            if k > 0:
                for idx in indices:
                    fileID.write(f"{idx} ")
            else:
                for idx in indices[::-1]:
                    fileID.write(f"{idx} ")
            fileID.write('\n')

## -------------------------------------------------------------------------------
## ---------------------------------- MAIN -------------------------------
## -------------------------------------------------------------------------------

def main(
    temp_fold,
    xyz,
    other,
    M,
    final_id,
    final_check_adj,
    indices=None,
    ground_by_index=None,
    id_to_final=None,
    progress_prefix="",
):
    """Generate building OFF files.

    ``indices`` allows the caller to process only the buildings that belong
    to one tile/block.  Building directory names always use the global index
    (``building{i+1}``), so the files can be accumulated into one mesh.
    """

    ## Initialize further variables
    zMin_default = float(np.mean(other[:, 2])) if other.shape[0] else 0.0
    building_indices = list(range(len(M))) if indices is None else list(indices)
    if ground_by_index is None:
        ground_by_index = {}
    if id_to_final is None:
        id_to_final = {}
        for j, ids in enumerate(final_id):
            for original_id in np.ravel(ids):
                id_to_final[int(original_id)] = j

    buildings_outside_pc = 0
    xyz_tree = make_point_index(xyz)
    other_tree = make_point_index(other)
    if xyz_tree is None and xyz.shape[0]:
        xyz_tree = cKDTree(xyz[:, :2])
    if other_tree is None and other.shape[0]:
        other_tree = cKDTree(other[:, :2])
    try:
        for index, divisor in ((xyz_tree, 500), (other_tree, 4)):
            if isinstance(index, GPUPointIndex):
                index.prepare(M, building_indices, divisor)
    except Exception as exc:
        gpu_failure("batched spatial ranges", exc)
        xyz_tree = cKDTree(xyz[:, :2]) if len(xyz) else None
        other_tree = cKDTree(other[:, :2]) if len(other) else None
    roof_normals = []
    roof_rhs = []
    roof_centers = []
    pending_buildings = []

    for i in tqdm(building_indices, desc=f"{progress_prefix}Processing Buildings"):

        # Computing other_aus, control, X, Y, xy, zMin, building
        try:
            other_aus,X,Y,building = PointSelectionAndDataStructure(
                xyz, M[i], other, xyz_tree, other_tree
            )
        except CudaExecutionError:
            raise
        except Exception as exc:
            if not isinstance(xyz_tree, GPUPointIndex) and not isinstance(other_tree, GPUPointIndex):
                raise
            gpu_failure("building spatial selection", exc)
            xyz_tree = cKDTree(xyz[:, :2]) if len(xyz) else None
            other_tree = cKDTree(other[:, :2]) if len(other) else None
            other_aus,X,Y,building = PointSelectionAndDataStructure(
                xyz, M[i], other, xyz_tree, other_tree
            )

        if building.shape[0] > 0:
            try:
                z_max, z_ceiling, roof_center, normal, rhs = roof_statistics(building)
            except Exception as exc:
                gpu_failure("roof statistics", exc)
                z_max, z_ceiling, roof_center, normal, rhs = roof_statistics(to_host(building))

            xy = []
            for x, y in zip(X, Y):
                if np.isnan(x):
                    if xy:  # only remove last row if xy is not empty
                        xy.pop()
                else:
                    xy.append([x, y])
            
            xy = np.array(xy)  # convert list to numpy array


            # Step 1 Organize Data
            lab = DefineLabels(X,i)




            # Interpolate ground heights at the footprint vertices.  If
            # this building touches neighbours, also use the ground points
            # already computed for those neighbours instead of scanning the
            # entire (city-wide) point list for every building.
            val = final_check_adj[i]
            point_parts = [other_aus] if other_aus.size else []
            if not (np.isscalar(val) and np.isnan(val)):
                for original_id in np.ravel(val):
                    try:
                        neighbour = id_to_final.get(int(original_id))
                    except (TypeError, ValueError):
                        continue
                    if neighbour is None:
                        continue
                    neighbour_points = ground_by_index.get(neighbour)
                    if neighbour_points is not None and neighbour_points.size:
                        point_parts.append(neighbour_points)

            points = np.vstack(point_parts) if point_parts else other_aus
            if points.size != 0:
                _, I = nearest_xy(points, xy)
                z = points[I, 2]
            else:
                z = zMin_default * np.ones((xy.shape[0]))

            xyz_aus = np.hstack([xy, z[:, np.newaxis]])
            ground_by_index[i] = xyz_aus
            points_aus_ground = xyz_aus
            
            # Fit one planar roof to upper building-class points. The centered
            # normal equations are solved together on CUDA after point selection.
            roof_plane_id = None
            if normal is not None:
                roof_plane_id = len(roof_normals)
                roof_normals.append(normal)
                roof_rhs.append(rhs)
                roof_centers.append(roof_center)

            pending_buildings.append(
                (i, xy, lab, points_aus_ground[:, :3], z_max,
                 roof_plane_id, roof_center, z_ceiling)
            )

        else: 
            buildings_outside_pc+=1

    roof_coefficients, roof_gpu_used = fit_roof_planes(roof_normals, roof_rhs)

    for i, xy, lab, points_build_f, z_max, plane_id, roof_center, z_ceiling in tqdm(
        pending_buildings, desc="Writing LOD meshes"
    ):
        roof_lod = "1"
        roof_heights = np.full(xy.shape[0], z_max, dtype=float)
        if plane_id is not None:
            coeff = roof_coefficients[plane_id]
            roof_heights = (xy - roof_center) @ coeff[:2] + coeff[2]
            roof_heights = np.maximum(roof_heights, points_build_f[:, 2] + 1.5)
            roof_heights = np.minimum(roof_heights, z_ceiling + 0.5)
            roof_lod = "2"

        points_build_r = np.column_stack((xy, roof_heights))
        points = np.vstack((points_build_f, points_build_r))
        tr = DefineTR(lab, points_build_f.shape[0])

        dir_names = f"{temp_fold}/building{i+1}"
        os.makedirs(dir_names, exist_ok=True)
        exportFacades(tr, points, dir_names)
        exportPavement(points_build_f, lab, dir_names)
        exportRoof(points_build_r, lab, dir_names)
        with open(os.path.join(dir_names, "lod.txt"), "w", encoding="ascii") as lod_file:
            lod_file.write(roof_lod)

    if roof_gpu_used:
        print("Roof plane fits: solved as one CUDA batch")
    print(gpu_summary())
    
    print("Saved building polygons:", len(building_indices))

    return buildings_outside_pc

