# CityLODer

![License](https://img.shields.io/badge/license-GPL%20v3-blue.svg)

**CityLODer** is an automated pipeline for generating **LOD1 and LOD2** urban models from airborne LiDAR point clouds and building footprints. It converts raw geospatial data into 3D city models suitable for visualization, analysis, and further processing.

The tool provides **OBJ meshes** and **CityJSON** file city models. It can also integrate street network data from GeoJSON files when available.

CityLODer runs locally through a Python environment and a compiled C++ mesh generator. Linux and WSL are supported by the included local runner.

## Requirements

CityLODer requires Python, a C++17 compiler, CMake, and the Python development headers. It is executed directly on the host system.


## Installation

### 1. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/EliaMTH/CityLODer.git
cd CityLODer
```

### 2. Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r core/requirements.txt
python -m pip install pybind11
```

For NVIDIA GPU acceleration, install the optional dependencies:

```bash
python -m pip install -r requirements-gpu.txt
```

### 3. Build the C++ mesh generator

```bash
cmake -S mesh_gen -B .build/cityloder -DCMAKE_BUILD_TYPE=Release
cmake --build .build/cityloder --parallel
```


## Input Data

The point cloud and building footprints must use the *same coordinate reference system*.

Street data must also be spatially aligned, or reprojected by the local runner by
passing the LAS/footprint CRS as its final argument.

### Point Cloud

Supported formats:

* `.las`
* `.laz`

**Ground** and **building** classes must be known (automatic classification included in the pipeline coming soon!).

Common ASPRS LAS classes include:

| Code | Description       |
| ---: | ----------------- |
|    0 | Never classified  |
|    1 | Unclassified      |
|    2 | **Ground**            |
|    3 | Low vegetation    |
|    4 | Medium vegetation |
|    5 | High vegetation   |
|    6 | **Building**          |
|    7 | Low point / noise |
|    8 | Model key-point   |
|    9 | Water             |

### Building Footprints

Supported format:

* ESRI Shapefile

Recommended shapefile components:

```text
footprints.shp
footprints.shx
footprints.dbf
```

The local pipeline can also read a geometry-only `.shp`. Without a `.dbf`,
building attributes are unavailable in the CityJSON output.

Optional but recommended:

```text
footprints.prj
```


### Street Network

Supported format:

* GeoJSON

Requirements:

* LineString geometries

The street network is optional. To skip street processing, omit the graph parameter or use `"."`.

## Usage

CityLODer expects the input data to be placed inside a single folder. This folder should contain the airborne LiDAR point cloud, the building footprints, and, optionally, the street network file.

For example:

```text
data/
├── point_cloud.las
├── footprints.shp
├── footprints.shx
├── footprints.dbf
├── footprints.prj
└── (street_network.geojson)
````

### Linux/WSL local runner

Make the local runner executable:

```bash
chmod +x wrapper_linux/cityloder_local
```

The runner accepts multiple LAS/LAZ files separated by commas:

```bash
./wrapper_linux/cityloder_local \
  "data/point_cloud.las" \
  "data/footprints.shp"
```

With output name and an optional street network:

```bash
./wrapper_linux/cityloder_local \
  "data/point_cloud.las" \
  "data/footprints.shp" \
  "output" \
  "mycity" \
  "data/street_network.geojson" \
  "EPSG:31982"
```


### Custom, Debug, or Batch Processing

For custom configurations, debugging, or batch processing, use the local
runner with the appropriate input and output paths:

```bash
./wrapper_linux/cityloder_local \
  "data/point_cloud.las" \
  "data/footprints.shp" \
  "output" \
  "mycity"
```


## Parameters

### Required Parameters

| Parameter      | Description                              | Example                |
| -------------- | ---------------------------------------- | ---------------------- |
| `LAS_PATH`     | Path to one or more LAS/LAZ files, comma-separated | `point_cloud.las` |
| `INPUT2D_PATH` | Path to the building footprint shapefile         | `footprints.shp`  |

### Optional Parameters

| Parameter          |  Default | Description                                      |
| ------------------ | -------: | ------------------------------------------------ |
| `OUTPUT_FILE_PATH` | `output` | Output directory                                 |
| `OUTPUT_FILE_NAME` | `mycity` | Base name for output model folder/files          |
| `GRAPH_PATH`       | empty | Optional street network GeoJSON. Leave empty to skip |
| `GRAPH_TARGET_CRS` | empty | CRS used to reproject the street network            |

## Output Files

After successful execution, the output directory contains the generated city model files.

Typical structure:

```text
output/
├── mycity.city.json
├── (street_graph_updated.geojson)
└── mycity/
    ├── buildings_mesh.obj
    ├── ground_mesh.obj
    └── city_mesh.obj
```

Depending on whether a street network is provided, `street_graph_updated.geojson` may or may not be generated.

### Output Description

| File                           | Description                                              |
| ------------------------------ | -------------------------------------------------------- |
| `<name>.city.json`             | CityJSON model containing reconstructed city geometry    |
| `buildings_mesh.obj`           | OBJ mesh containing reconstructed buildings              |
| `ground_mesh.obj`              | OBJ mesh containing the reconstructed ground surface     |
| `city_mesh.obj`                | Combined OBJ mesh containing buildings and ground       |
| `(street_graph_updated.geojson)` | Optional street graph updated with elevation information |

## Example Workflows

### Simple Processing

```bash
./wrapper_linux/cityloder_local \
  "data/matera.las" \
  "data/matera.shp"
```

This writes outputs using the default output folder and model name.

### Processing with Custom Output Name

```bash
./wrapper_linux/cityloder_local \
  "data/city.las" \
  "data/buildings.shp" \
  "results" \
  "matera_model"
```

### Processing with Street Network

```bash
./wrapper_linux/cityloder_local \
  "data/city.las" \
  "data/buildings.shp" \
  "results/processed_city" \
  "final_model" \
  "data/streets.geojson"
```

### Custom Classification Codes

Use this when the input LAS file uses non-standard class codes.

```bash
python core/cityloder_main.py \
  "data/footprints.shp" \
  "" \
  "data/custom_classification.las" \
  "output/custom_model" \
  7 \
  3 \
  "output/custom_model_tmp" \
  "output/custom_model_tmp/ground_polygon"
```

## Troubleshooting

### Permission Denied on Linux/macOS

Make the wrapper executable:

```bash
chmod +x wrapper_linux/cityloder_local
```

Run the command from the project root.

### PowerShell Script Execution Is Disabled

Run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Empty or Misaligned Output

Check that all input files use the same coordinate reference system:

* LAS/LAZ point cloud
* Building footprint shapefile
* Optional street network GeoJSON

Coordinate mismatch is one of the most common causes of empty or incorrect results.

### No Building Geometry Generated

Check that:

* the LAS file contains building-classified points,
* the building class code is correct,
* the footprints overlap the point cloud,
* the shapefile geometries are valid.

For example, if buildings are classified as class `7`, use:

```powershell
-CLASS_BUILDING 7
```

### No Ground Mesh Generated

Check that the LAS file contains ground-classified points and that the ground classification code is correct.

The default ground class is `2`.

## Local Development and Dependencies

CityLODer runs in a local Python/C++ environment.

### Run locally from WSL/Linux

The local runner supports multiple LAS files separated by commas. Build the
Python environment and C++ mesh generator once, then run the pipeline:

```bash
python3 -m venv ~/.venvs/cityloder
source ~/.venvs/cityloder/bin/activate
pip install -r core/requirements.txt pybind11
cmake -S mesh_gen -B ~/.cache/cityloder-build -DCMAKE_BUILD_TYPE=Release
cmake --build ~/.cache/cityloder-build --parallel
bash wrapper_linux/cityloder_local \
  "data/394000-6455000.las,data/394000-6456000.las,data/395000-6455000.las,data/395000-6456000.las" \
  "data/Edificacao.shp" output city_model "data/export (1).geojson" EPSG:31982
```

The street input is assumed to use RFC 7946 longitude/latitude when its
coordinates look geographic; pass the target CRS used by the LAS and footprints
as the final argument. Use an empty street path to omit the graph.

For NVIDIA GPU acceleration under WSL, install the optional CUDA dependencies
in the same environment:

```bash
pip install -r requirements-gpu.txt
```

When CuPy detects a CUDA device, dense point-in-polygon checks and the batch
roof-plane fit run on the GPU. Other inputs and unsupported systems use the CPU
path. Buildings with enough roof points receive a fitted sloped roof and LOD2
geometry; insufficient roof data falls back to LOD1.

### System Requirements

* Python 3.9+
* C++17-compatible compiler, such as:
  * `g++`
  * `clang`
  * MSVC on Windows
* CMake 3.10+
* Python development headers

### Python Dependencies

* [NumPy](https://numpy.org/) — numerical computing
* [SciPy](https://scipy.org/) — scientific computing
* [Shapely](https://shapely.readthedocs.io/) — geometric operations
* [Laspy](https://github.com/laspy/laspy) — LAS/LAZ point cloud handling
* [pyshp](https://github.com/karoly/pyshp) — shapefile handling
* [pybind11](https://github.com/pybind/pybind11) — Python/C++ bindings
* [pyproj](https://pyproj4.github.io/pyproj/stable/) — street coordinate reprojection
* [tqdm](https://github.com/tqdm/tqdm) — progress bars

### C++ Dependencies

CityLODer also relies on the following open-source C++ libraries:

* [cinolib](https://github.com/maxicino/cinolib) — mesh processing and geometry algorithms
* [nlohmann/json](https://github.com/nlohmann/json) — JSON serialization
* [Eigen](http://eigen.tuxfamily.org/) — linear algebra


## Publication
You can find the CityLODer paper [here](https://www.sciencedirect.com/science/article/pii/S1524070326000184).

### DOI:

```text
(https://doi.org/10.1016/j.gmod.2026.101337)
```

### BibTeX

```bibtex
@article{SORGENTE2026101337,
title = {CityLODer: City models from airborne point clouds},
journal = {Graphical Models},
volume = {147},
pages = {101337},
year = {2026},
issn = {1524-0703},
doi = {https://doi.org/10.1016/j.gmod.2026.101337},
url = {https://www.sciencedirect.com/science/article/pii/S1524070326000184},
author = {Tommaso Sorgente and Elia {Moscoso Thompson} and Chiara Romanengo}
}
```

If you use CityLODer in academic work, please cite the paper above.

## Datasets

A shareable subset of the comparison dataset used in the CityLODer tests (Genoa area) is available on [Zenodo](https://zenodo.org/records/21623262).

## Contributors

CityLODer is developed by:

**CNR — Istituto di Matematica Applicata e Tecnologie Informatiche**

* **Tommaso Sorgente**
  [tommaso.sorgente@cnr.it](mailto:tommaso.sorgente@cnr.it)

* **Elia Moscoso Thompson**
  [elia.moscosothompson@cnr.it](mailto:elia.moscosothompson@cnr.it)

* **Chiara Romanengo**
  [chiara.romanengo@cnr.it](mailto:chiara.romanengo@cnr.it)

## Contributing

Contributions are welcome.

You can contribute by:

* reporting bugs,
* suggesting features,
* improving documentation,
* sharing test data or usage examples.

Please use GitHub issues for discussions.

## Coming soon

Planned or ongoing work:

* [ ] Preprocessing step to conform street data and building footprint data
* [ ] Automatic classification support
