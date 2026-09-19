# PowerShell Build Script for AI Proctoring DatasetCollector on Windows 10/11
# Run: .\build_windows.ps1

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Building Standalone Windows DatasetCollector Package" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Verify Python availability
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "Python was not found in PATH. Please install Python 3.10+ on Windows."
    exit 1
}

Write-Host "[1/5] Checking build dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install pyinstaller opencv-python pyyaml numpy

# 2. Clean previous build artifacts
Write-Host "[2/5] Cleaning previous build folders..." -ForegroundColor Yellow
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist\DatasetCollector") { Remove-Item -Recurse -Force "dist\DatasetCollector" }

# 3. Execute PyInstaller packaging
Write-Host "[3/5] Compiling standalone executable via PyInstaller..." -ForegroundColor Yellow
python -m PyInstaller DatasetCollector.spec --noconfirm --clean

# 4. Verify output executable
$exePath = "dist\DatasetCollector\DatasetCollector.exe"
if (-not (Test-Path $exePath)) {
    Write-Error "Build failed! $exePath was not generated."
    exit 1
}

# 5. Copy configuration and end-user documentation
Write-Host "[4/5] Copying configuration and user instructions..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "dist\DatasetCollector\configs\capture_scenarios" | Out-Null
Copy-Item "configs\capture_scenarios\activities.yaml" "dist\DatasetCollector\configs\capture_scenarios\" -Force

if (Test-Path "README_WINDOWS.txt") {
    Copy-Item "README_WINDOWS.txt" "dist\DatasetCollector\README.txt" -Force
}

Write-Host "============================================================" -ForegroundColor Green
Write-Host "  BUILD SUCCESSFUL!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Portable distribution folder:"
Write-Host "  $((Resolve-Path 'dist\DatasetCollector').Path)" -ForegroundColor White
Write-Host ""
Write-Host "To distribute to users:"
Write-Host "  1. Zip the entire folder: dist\DatasetCollector"
Write-Host "  2. Share the zip file with participants."
Write-Host "  3. The recipient can extract and double-click DatasetCollector.exe"
Write-Host "============================================================" -ForegroundColor Green
