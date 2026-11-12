param(
    [Parameter(Mandatory = $true)]
    [string]$LAS_PATH,
    [Parameter(Mandatory = $true)]
    [string]$INPUT2D_PATH,
    [string]$OUTPUT_FILE_PATH = "output",
    [string]$OUTPUT_FILE_NAME = "mycity",
    [string]$GRAPH_PATH = "",
    [int]$CLASS_BUILDING = 6,
    [int]$CLASS_GROUND = 2,
    [string]$TEMP_FOLD = "",
    [string]$GRAPH_TARGET_CRS = "",
    [string]$PYTHON_BIN = "python",
    [string]$BUILD_DIR = "build\mesh-updated",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$DEVICE = "auto",
    [double]$GPU_MEMORY_FRACTION = 0.8
)

function Show-Usage {
@"
Usage:
  .\cityloder_app.ps1 -LAS_PATH <file> -INPUT2D_PATH <file> [-OUTPUT_FILE_PATH <path>] [-OUTPUT_FILE_NAME <name>] [-GRAPH_PATH <path>] [-CLASS_BUILDING <int>] [-CLASS_GROUND <int>] [-TEMP_FOLD <path>] [-GRAPH_TARGET_CRS <crs>]

Arguments:
  LAS_PATH          LAS/LAZ point cloud (absolute or relative to repository)
  INPUT2D_PATH      Building footprint SHP
  OUTPUT_FILE_PATH  Output directory. Default: output
  OUTPUT_FILE_NAME  Model name. Default: mycity
  GRAPH_PATH        Optional street GeoJSON; empty skips streets
  GRAPH_TARGET_CRS  Optional target CRS, for example EPSG:31982
  PYTHON_BIN        Python executable or virtual environment path
  BUILD_DIR         CMake build directory. Default: .build\cityloder

The script runs natively on Windows; it does not use Docker.
"@
}

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

function Invoke-Checked {
    param([string]$FilePath, [string[]]$Arguments)
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE."
    }
}

if (-not (Get-Command $PYTHON_BIN -ErrorAction SilentlyContinue)) {
    throw "Python not found: $PYTHON_BIN. Activate the virtual environment or pass -PYTHON_BIN."
}
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "CMake not found in PATH."
}

$lasPath = (Resolve-Path $LAS_PATH).Path
$footprintsPath = (Resolve-Path $INPUT2D_PATH).Path
$outputPath = [IO.Path]::GetFullPath($OUTPUT_FILE_PATH)
$tempPath = if ([string]::IsNullOrWhiteSpace($TEMP_FOLD)) {
    Join-Path $outputPath "${OUTPUT_FILE_NAME}_tmp"
} else {
    [IO.Path]::GetFullPath($TEMP_FOLD)
}
$graphPath = ""
if (-not [string]::IsNullOrWhiteSpace($GRAPH_PATH) -and $GRAPH_PATH -ne ".") {
    $graphPath = (Resolve-Path $GRAPH_PATH).Path
}

New-Item -ItemType Directory -Force -Path $outputPath, $tempPath | Out-Null
$buildPath = [IO.Path]::GetFullPath($BUILD_DIR)
$meshExe = Join-Path $buildPath "city_iconic_mesh.exe"
if (-not (Test-Path $meshExe)) {
    $meshExe = Join-Path $buildPath "Release\city_iconic_mesh.exe"
}
if (-not (Test-Path $meshExe)) {
    Invoke-Checked "cmake" @("-S", "mesh_gen", "-B", $BUILD_DIR, "-DCMAKE_BUILD_TYPE=Release")
    Invoke-Checked "cmake" @("--build", $BUILD_DIR, "--config", "Release", "--parallel")
    $meshExe = Join-Path $buildPath "city_iconic_mesh.exe"
    if (-not (Test-Path $meshExe)) {
        $meshExe = Join-Path $buildPath "Release\city_iconic_mesh.exe"
    }
    if (-not (Test-Path $meshExe)) {
        throw "city_iconic_mesh.exe was not produced in $buildPath."
    }
}

$runnerArgs = @(
    "cityloder_runner.py", "--las", $lasPath, "--footprints", $footprintsPath,
    "--output", $outputPath, "--name", $OUTPUT_FILE_NAME,
    "--building-class", $CLASS_BUILDING, "--ground-class", $CLASS_GROUND,
    "--temp", $tempPath, "--mesh-exe", $meshExe,
    "--device", $DEVICE, "--gpu-memory-fraction", $GPU_MEMORY_FRACTION.ToString([Globalization.CultureInfo]::InvariantCulture)
)
if ($graphPath) { $runnerArgs += @("--streets", $graphPath) }
if ($GRAPH_TARGET_CRS) { $runnerArgs += @("--graph-target-crs", $GRAPH_TARGET_CRS) }
Invoke-Checked $PYTHON_BIN $runnerArgs
