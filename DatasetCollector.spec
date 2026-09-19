# -*- mode: python ; coding: utf-8 -*-
# PyInstaller specification for AI Proctoring Dataset Collector (Standalone Windows / Linux)

import sys
from pathlib import Path

block_cipher = None

# Base directory
REPO_DIR = Path.cwd()

# Data files to bundle with executable
datas = [
    (str(REPO_DIR / "configs" / "capture_scenarios" / "activities.yaml"), "configs/capture_scenarios"),
]

# Exclude heavy unnecessary production dependencies to keep distribution small and isolated
excludes = [
    "torch",
    "torchvision",
    "torchaudio",
    "ultralytics",
    "mediapipe",
    "onnxruntime",
    "onnxruntime_gpu",
    "scipy",
    "sympy",
    "matplotlib",
    "fastapi",
    "uvicorn",
    "starlette",
    "pydantic",
    "polars",
    "jinja2",
    "pytest",
    "IPython",
]

hiddenimports = [
    "yaml",
    "cv2",
    "numpy",
]

a = Analysis(
    ['DatasetCollector.py'],
    pathex=[str(REPO_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# One-folder distribution for maximum stability with OpenCV camera backends and Windows DirectShow DLLs
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DatasetCollector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX false prevents antivirus false positives on Windows
    console=True,  # Keep console enabled so logs are visible if launched via CMD or for diagnostics
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='DatasetCollector',
)
