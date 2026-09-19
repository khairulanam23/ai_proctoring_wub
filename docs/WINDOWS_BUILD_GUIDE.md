# Developer Build Guide: Standalone Windows `DatasetCollector.exe`

This guide explains how to package the **AI Proctoring Dataset Collector** into a portable, standalone Windows application (`DatasetCollector.exe`) that runs on Windows 10 and 11 without requiring Python, Git, virtual environments, or developer dependencies.

---

## 1. Architecture Overview

- **Source Code**:
  - `DatasetCollector.py` (Main entry point with logging and unhandled crash handler)
  - `tools/dataset_collector/collector_app.py` (OpenCV GUI, mouse/keyboard controller, 2-photo workflow)
  - `tools/dataset_collector/camera_utils.py` (DirectShow & Media Foundation camera enumeration)
  - `tools/dataset_collector/config_loader.py` (Resource resolution via `sys._MEIPASS` and filesystem)
  - `tools/dataset_collector/manifest.py` (Session manifest `2.0.0-still` schema, SHA-256 calculation)
- **Configuration**:
  - `configs/capture_scenarios/activities.yaml` (45 serialized activities, decoupled from code)
- **Packaging Mechanism**:
  - `DatasetCollector.spec` (PyInstaller specification configured for **One-Folder distribution**)
  - `build_windows.ps1` (PowerShell 1-click build script)
  - `build_windows.bat` (CMD batch 1-click build script)

---

## 2. Windows Build Requirements

To compile the native Windows binary, execute the build on a **Windows 10 or Windows 11 64-bit machine**:

1. **Python 3.10, 3.11, 3.12, 3.13, or 3.14 (64-bit)** installed on Windows.
2. Ensure `python` and `pip` are registered in the system `PATH`.
3. Minimal build dependencies:
   - `pyinstaller`
   - `opencv-python`
   - `pyyaml`
   - `numpy`

*Note: Heavy production AI dependencies (`torch`, `ultralytics`, `mediapipe`, `onnxruntime`, `fastapi`) are explicitly excluded in `DatasetCollector.spec` to keep the distribution under 60 MB and ensure 100% isolation from production AI.*

---

## 3. One-Click Build Instructions

### Method A: Using PowerShell (Recommended)
Open PowerShell in the repository root and run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\build_windows.ps1
```

### Method B: Using Windows Command Prompt (CMD)
Open CMD in the repository root and run:
```cmd
build_windows.bat
```

### Method C: Manual PyInstaller Invocation
If you prefer running PyInstaller manually:
```cmd
pip install pyinstaller opencv-python pyyaml numpy
pyinstaller DatasetCollector.spec --noconfirm --clean
mkdir dist\DatasetCollector\configs\capture_scenarios
copy configs\capture_scenarios\activities.yaml dist\DatasetCollector\configs\capture_scenarios\
copy README_WINDOWS.txt dist\DatasetCollector\README.txt
```

---

## 4. Generated Distribution Package Structure

Upon build completion, the distributable bundle is located in `dist/DatasetCollector/`:

```text
dist/DatasetCollector/
├── DatasetCollector.exe                 # Main double-clickable standalone executable
├── configs/
│   └── capture_scenarios/
│       └── activities.yaml             # Modifiable activity protocol
├── README.txt                           # Non-technical end-user guide
├── logs/                                # Automatically created at runtime
│   └── collector.log                   # Diagnostics log
├── _internal/                           # Bundled Python runtime, OpenCV DLLs, DirectShow filters
│   ├── python3*.dll
│   ├── cv2/
│   └── ...
```

---

## 5. Verification Checklist on Windows

Before sharing the package with participants, verify on a clean Windows machine:

1. **No Python Dependency**:
   - Double-click `dist\DatasetCollector\DatasetCollector.exe` on a test PC without Python installed.
   - The application window should open cleanly.
2. **Camera Enumeration**:
   - Verify that built-in laptop webcam or USB webcam is recognized.
   - If testing with **DroidCam OBS**:
     - Start DroidCam OBS, connect Android phone, toggle "Start Virtual Camera".
     - In `DatasetCollector.exe`, press `[C]` or click "[C] Switch Camera".
     - Confirm that the DroidCam OBS stream appears in the live preview.
3. **Capture Workflow**:
   - Complete 1 or 2 test activities (capturing 2 photos per activity with `[SPACE]`).
   - Click "[O] Open Folder" and confirm that `DatasetOutput/<PXXX>/<SXXX>/` opens in Windows Explorer.
   - Verify that `photo_01.jpg` and `photo_02.jpg` exist and decode cleanly.
   - Check `session_manifest.json` and `checksum.sha256`.

---

## 6. Maintenance & Versioning

### Updating the Application Version:
1. Update `APPLICATION_VERSION` in `tools/dataset_collector/manifest.py`.
2. Update `__version__` in `tools/dataset_collector/__init__.py`.
3. Re-run `build_windows.ps1`.

### Modifying or Adding Activities:
The activities are defined in `configs/capture_scenarios/activities.yaml`.
Because `activities.yaml` is copied next to `DatasetCollector.exe`, **you can modify the activities list or instructions without recompiling the executable**!

### Troubleshooting Camera Discovery on Windows:
- OpenCV utilizes `cv2.CAP_DSHOW` (DirectShow) as the preferred Windows backend.
- If a virtual camera like DroidCam OBS or OBS Virtual Camera does not appear, ensure that the virtual camera service was started *before* opening `DatasetCollector.exe`.
- All runtime exceptions and device warnings are captured in `logs/collector.log`.
