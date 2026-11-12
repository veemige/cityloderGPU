from cityloder_gpu import nearest_xy
#!/usr/bin/env python3

import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def preprocess_graph(geojson_path, xyz_ground, out_folder, target_crs=None):
    """
    Reproject geographic GeoJSON coordinates when a target CRS is provided,
    then add Z values from the nearest ground-class LAS point.

    Parameters
    ----------
    geojson_path : str or Path
        Path to the input GeoJSON file. Supported geometries are:
        - LineString
        - MultiLineString

    xyz_ground : float array
        las coordinates of the ground points

    Returns
    -------
    Path to the output GeoJSON file, saved next to the input file with suffix "_updated".

    Notes
    -----
    Existing Z values, if present, are overwritten.
    Feature metadata is normalized so that:
    - "NO" is kept/set from existing "NO", else "osmid", else 0
    - properties with null values are removed
    """
    


    # Load the GeoJSON structure in memory. If "" or ".", load an empty graph
    if not geojson_path or str(geojson_path) == ".":
        geojson = {"type": "FeatureCollection", "features": []}
        out_path = ""
        street_count = 0

        return out_path, street_count
    
    else:
        # Load the GeoJSON structure in memory.
        geojson_path = Path(geojson_path)
        with geojson_path.open("r", encoding="utf-8") as f:
            geojson = json.load(f)



    # Initialize stuff
    nodes = []
    xy = []
    street_count = 0

    # Collect references to the actual coordinate lists in the GeoJSON.
    for feature in tqdm(geojson.get("features", []), desc = "Processing streetgraph"):
        geom = feature.get("geometry")
        if not geom:
            continue

        # Make sure NO field exists and remove NULL properties
        props = feature.setdefault("properties", {})
        props["NO"] = props.get("NO") or props.get("osmid") or 0 # To be updated
        feature["properties"] = {k: v for k, v in props.items() if v is not None}

        # Normalize both geometry types into an iterable of lines.
        if geom["type"] == "LineString":
            parts = [geom["coordinates"]]
        elif geom["type"] == "MultiLineString":
            parts = geom["coordinates"]
        else:
            raise ValueError(f"Unsupported geometry type: {geom['type']}")

        # Store each coordinate list itself, plus its XY values for the KDTree query.
        for line in parts:
            if len(line) >= 2:
                street_count += len(line) - 1

            for coord in line:
                if len(coord) >= 2:
                    nodes.append(coord)
                    xy.append(coord[:2])

    # Only run the nearest-neighbor step if at least one valid coordinate exists.
    if xy:
        xy = np.asarray(xy, dtype=float)

        # RFC 7946 GeoJSON coordinates are longitude/latitude. Reproject them
        # before querying the projected LAS points or writing the updated graph.
        if np.all(np.abs(xy[:, 0]) <= 180) and np.all(np.abs(xy[:, 1]) <= 90):
            if not target_crs:
                raise ValueError(
                    "O GeoJSON está em longitude/latitude, mas os pontos LAS estão "
                    "projetados. Informe o CRS de destino (por exemplo EPSG:31982)."
                )
            from pyproj import Transformer

            transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
            projected_x, projected_y = transformer.transform(xy[:, 0], xy[:, 1])
            xy = np.column_stack((projected_x, projected_y))
            for coord, projected in zip(nodes, xy):
                coord[0], coord[1] = float(projected[0]), float(projected[1])

        _, idx = nearest_xy(xyz_ground, xy)
        for coord, z in zip(nodes, xyz_ground[idx, 2]):
            coord[:] = [coord[0], coord[1], float(z)]

    # Save appending "_updated" to the filename.

    out_path = Path(out_folder).parent / f"{geojson_path.stem}_updated{geojson_path.suffix}"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)


    return out_path, street_count
