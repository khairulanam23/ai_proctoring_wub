"""Robust camera discovery, backend selection, and frame acquisition for Windows & Linux."""

from __future__ import annotations

import contextlib
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger("dataset_collector.camera_utils")


@dataclass
class DiscoveredCamera:
    """Represents an enumerated functional physical or virtual camera."""

    index: int
    backend_name: str
    backend_code: int
    width: int
    height: int
    fps: float
    is_valid: bool
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "backend_name": self.backend_name,
            "backend_code": self.backend_code,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "is_valid": self.is_valid,
            "label": self.label,
        }


def get_platform_backends() -> list[tuple[str, int]]:
    """Returns prioritized OpenCV camera backends for the current operating system."""
    if sys.platform.startswith("win"):
        # On Windows, DirectShow (CAP_DSHOW) is primary for webcams and virtual cameras (DroidCam OBS)
        return [
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("ANY", cv2.CAP_ANY),
        ]
    if sys.platform.startswith("linux"):
        return [
            ("V4L2", cv2.CAP_V4L2),
            ("ANY", cv2.CAP_ANY),
        ]
    if sys.platform.startswith("darwin"):
        return [
            ("AVFOUNDATION", cv2.CAP_AVFOUNDATION),
            ("ANY", cv2.CAP_ANY),
        ]
    return [("ANY", cv2.CAP_ANY)]


def probe_single_camera(
    index: int,
    backend_name: str,
    backend_code: int,
    target_width: int = 1280,
    target_height: int = 720,
) -> DiscoveredCamera | None:
    """Attempts to open and read a single camera index with the given backend."""
    cap = None
    try:
        cap = cv2.VideoCapture(index, backend_code)
        if not cap.isOpened():
            return None

        # Request target resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

        # Allow exposure/gain stabilization
        for _ in range(3):
            cap.read()

        ret, frame = cap.read()
        if not ret or frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            cap.release()
            return None

        h, w = frame.shape[:2]
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        actual_backend = cap.getBackendName() or backend_name
        label = f"Camera {index} ({actual_backend} {w}x{h})"

        cap.release()
        return DiscoveredCamera(
            index=index,
            backend_name=actual_backend,
            backend_code=backend_code,
            width=w,
            height=h,
            fps=fps,
            is_valid=True,
            label=label,
        )
    except Exception as e:
        LOGGER.debug(f"Exception probing camera index {index} with {backend_name}: {e}")
        if cap is not None:
            with contextlib.suppress(Exception):
                cap.release()
        return None


def enumerate_available_cameras(
    max_index: int = 4,
    target_width: int = 1280,
    target_height: int = 720,
) -> list[DiscoveredCamera]:
    """Enumerates functional camera devices across candidate indices.

    On Windows, probes DirectShow (DSHOW) first to properly discover DroidCam OBS
    and standard USB webcams.
    """
    discovered: list[DiscoveredCamera] = []
    seen_indices: set[int] = set()
    backends = get_platform_backends()

    for idx in range(max_index):
        for b_name, b_code in backends:
            if idx in seen_indices:
                continue
            cam = probe_single_camera(idx, b_name, b_code, target_width, target_height)
            if cam is not None and cam.is_valid:
                discovered.append(cam)
                seen_indices.add(idx)
                break  # Found working backend for this index

    LOGGER.info(f"Discovered {len(discovered)} camera(s): {[c.label for c in discovered]}")
    return discovered


def open_camera_stream(
    index: int,
    backend_code: int = cv2.CAP_ANY,
    target_width: int = 1280,
    target_height: int = 720,
) -> tuple[cv2.VideoCapture | None, dict[str, Any]]:
    """Opens a live camera stream with fallback to CAP_ANY if specific backend fails."""
    cap = cv2.VideoCapture(index, backend_code)
    if not cap.isOpened():
        time.sleep(0.2)
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)

    if not cap.isOpened():
        return None, {"error": f"Failed to open camera index {index}"}

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    backend_name = cap.getBackendName()

    info = {
        "device_index": index,
        "backend_name": backend_name,
        "actual_resolution": [actual_w, actual_h],
        "fps": round(fps, 2) if (fps > 0 and not np.isnan(fps)) else 30.0,
    }
    return cap, info
