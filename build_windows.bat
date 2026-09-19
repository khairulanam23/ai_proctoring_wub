@echo off
REM Windows CMD Batch Build Script for AI Proctoring DatasetCollector
REM Usage: build_windows.bat

echo ============================================================
echo   Building Standalone Windows DatasetCollector Package (CMD)
echo ============================================================

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH. Please install Python on Windows.
    pause
    exit /b 1
)

echo [1/4] Installing build requirements...
python -m pip install --upgrade pip
python -m pip install pyinstaller opencv-python pyyaml numpy

echo [2/4] Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist\DatasetCollector rmdir /s /q dist\DatasetCollector

echo [3/4] Running PyInstaller...
python -m PyInstaller DatasetCollector.spec --noconfirm --clean
if %errorlevel% neq 0 (
    echo [ERROR] PyInstaller compilation failed!
    pause
    exit /b 1
)

echo [4/4] Copying configurations and documentation...
if not exist dist\DatasetCollector\configs\capture_scenarios mkdir dist\DatasetCollector\configs\capture_scenarios
copy configs\capture_scenarios\activities.yaml dist\DatasetCollector\configs\capture_scenarios\ >nul
if exist README_WINDOWS.txt copy README_WINDOWS.txt dist\DatasetCollector\README.txt >nul

echo ============================================================
echo   BUILD SUCCESSFUL!
echo ============================================================
echo Executable generated at: dist\DatasetCollector\DatasetCollector.exe
echo You can now distribute the dist\DatasetCollector folder.
echo ============================================================
pause
