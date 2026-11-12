param(
    [string]$PythonBin = ".\.venv\Scripts\python.exe",
    [string]$BuildDir = ".build\cityloder",
    [switch]$KeepBuildOutput
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path $PSScriptRoot).Path
Set-Location $root

$pythonPath = if ([IO.Path]::IsPathRooted($PythonBin)) {
    $PythonBin
} else {
    Join-Path $root $PythonBin
}

if (-not (Test-Path $pythonPath -PathType Leaf)) {
    $basePython = Get-Command py -ErrorAction SilentlyContinue
    $basePythonArgs = @("-3")
    if (-not $basePython) {
        $basePython = Get-Command python -ErrorAction SilentlyContinue
        $basePythonArgs = @()
    }
    if (-not $basePython) {
        throw "Python was not found. Install Python 3.9+ and enable 'Add Python to PATH'."
    }

    $venvDir = Join-Path $root ".venv"
    Write-Host "Base Python: $($basePython.Source)"
    Write-Host "Creating Python environment: $venvDir"
    & $basePython.Source @basePythonArgs -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python virtual environment."
    }
    $pythonPath = Join-Path $venvDir "Scripts\python.exe"
}
if (-not (Test-Path $pythonPath -PathType Leaf)) {
    throw "Python environment was not created: $pythonPath"
}
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "CMake was not found in PATH. Install CMake and a C++17 compiler."
}

$cinolibPath = Join-Path $root "mesh_gen\cinolib"
$cinolibFiles = @(Get-ChildItem $cinolibPath -Recurse -File -ErrorAction SilentlyContinue)
if ($cinolibFiles.Count -eq 0) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "mesh_gen\cinolib is empty. Install Git and run: git submodule update --init --recursive"
    }
    Write-Host "Initializing the cinolib submodule..."
    git submodule update --init --recursive
    if ($LASTEXITCODE -ne 0) {
        throw "Could not initialize mesh_gen\cinolib. Run 'git submodule update --init --recursive' and check network/SSH access."
    }
    $cinolibFiles = @(Get-ChildItem $cinolibPath -Recurse -File -ErrorAction SilentlyContinue)
    if ($cinolibFiles.Count -eq 0) {
        throw "mesh_gen\cinolib is still empty after submodule initialization."
    }
}

Write-Host "Installing Python dependencies..."
& $pythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Could not upgrade pip."
}
& $pythonPath -m pip install -r core\requirements.txt -r requirements-gpu.txt
if ($LASTEXITCODE -ne 0) {
    throw "Could not install CityLODer dependencies."
}
& $pythonPath -m pip install pyinstaller
if ($LASTEXITCODE -ne 0) {
    throw "Could not install PyInstaller."
}

if (-not (Test-Path (Join-Path $BuildDir "city_iconic_mesh.exe")) -and
    -not (Test-Path (Join-Path $BuildDir "Release\city_iconic_mesh.exe"))) {
    Write-Host "Configuring C++ mesh generator..."
    cmake -S mesh_gen -B $BuildDir -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed."
    }
    Write-Host "Building C++ mesh generator..."
    cmake --build $BuildDir --config Release --parallel
    if ($LASTEXITCODE -ne 0) {
        throw "Mesh generator build failed."
    }
}

$meshExe = Join-Path $BuildDir "city_iconic_mesh.exe"
if (-not (Test-Path $meshExe)) {
    $meshExe = Join-Path $BuildDir "Release\city_iconic_mesh.exe"
}
if (-not (Test-Path $meshExe)) {
    $meshExe = Join-Path $BuildDir "Debug\city_iconic_mesh.exe"
}
if (-not (Test-Path $meshExe)) {
    $meshExe = Get-ChildItem -Path $BuildDir -Filter "city_iconic_mesh.exe" -Recurse -File |
        Select-Object -First 1 -ExpandProperty FullName
}
if ([string]::IsNullOrWhiteSpace($meshExe) -or -not (Test-Path $meshExe)) {
    throw "C++ mesh executable was not produced. Check the CMake/compiler error above."
}

Write-Host "Packaging executable with mesh generator: $meshExe"
& $pythonPath -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --noupx `
    --name cityloder `
    --paths core `
    --hidden-import cupy `
    --hidden-import cupy.cuda `
    --hidden-import cupy.linalg `
    --collect-all cupy `
    --collect-all cupyx `
    --hidden-import lazrs `
    --collect-all lazrs `
    --exclude-module cupy.cuda.cub `
    --add-binary "$meshExe;mesh_gen\build\Release" `
    cityloder_runner.py

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed."
}

if (-not (Test-Path "dist\cityloder.exe")) {
    throw "PyInstaller completed but dist\cityloder.exe was not created."
}
Write-Host "Executable created: dist\cityloder.exe"
