"""Camera health monitoring, stream anomaly detection, and frozen frame diagnostics.

Distinguishes system and camera capture failures from candidate behavior:
- Frozen video frames (repeated identical or near-identical frames over time)
- Black screens / pitch-dark camera feeds
- Extreme glare / overexposed sensor conditions
- Interrupted frame delivery / large timestamp gaps
- Unexpected resolution switches mid-session

CRITICAL PRINCIPLE:
    Camera anomalies and hardware glitches are recorded as DIAGNOSTIC / SYSTEM events.
    They must NEVER be automatically classified as candidate misconduct or cheating.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np


class CameraAnomaly(str, Enum):
    """Categorical camera health states."""

    NONE = "NONE"
    FRAME_FROZEN = "FRAME_FROZEN"
    BLACK_SCREEN = "BLACK_SCREEN"
    EXTREME_GLARE = "EXTREME_GLARE"
    DELIVERY_GAP = "DELIVERY_GAP"
    RESOLUTION_CHANGED = "RESOLUTION_CHANGED"
    STREAM_DISCONNECTED = "STREAM_DISCONNECTED"


@dataclass
class CameraHealthStatus:
    """Detailed health assessment for an incoming camera frame."""

    anomaly: CameraAnomaly = CameraAnomaly.NONE
    is_healthy: bool = True
    is_frozen: bool = False
    freeze_duration_seconds: float = 0.0
    gap_duration_seconds: float = 0.0
    mean_luminance: float | None = None
    blur_variance: float | None = None
    resolution: tuple[int, int] | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly": self.anomaly.value,
            "is_healthy": self.is_healthy,
            "is_frozen": self.is_frozen,
            "freeze_duration_seconds": round(self.freeze_duration_seconds, 2),
            "gap_duration_seconds": round(self.gap_duration_seconds, 2),
            "mean_luminance": round(self.mean_luminance, 2)
            if self.mean_luminance is not None
            else None,
            "blur_variance": round(self.blur_variance, 2)
            if self.blur_variance is not None
            else None,
            "resolution": list(self.resolution) if self.resolution is not None else None,
            "details": self.details,
        }


class CameraHealthMonitor:
    """Stateful detector for webcam stream anomalies and hardware faults."""

    def __init__(
        self,
        frozen_frame_duration_seconds: float = 4.0,
        max_delivery_gap_seconds: float = 3.0,
        min_dark_luma: float = 5.0,
        max_glare_luma: float = 250.0,
        diff_threshold_mse: float = 0.8,
    ) -> None:
        self.frozen_frame_duration_seconds = float(frozen_frame_duration_seconds)
        self.max_delivery_gap_seconds = float(max_delivery_gap_seconds)
        self.min_dark_luma = float(min_dark_luma)
        self.max_glare_luma = float(max_glare_luma)
        self.diff_threshold_mse = float(diff_threshold_mse)

        # Stateful stream tracking
        self._last_timestamp: float | None = None
        self._last_thumbnail: np.ndarray | None = None
        self._freeze_start_timestamp: float | None = None
        self._initial_resolution: tuple[int, int] | None = None

    def reset(self) -> None:
        """Reset internal stream history."""
        self._last_timestamp = None
        self._last_thumbnail = None
        self._freeze_start_timestamp = None
        self._initial_resolution = None

    def assess(
        self,
        frame: np.ndarray | None,
        timestamp_seconds: float,
    ) -> CameraHealthStatus:
        """Evaluate stream continuity and frame health."""
        # 1. Null or empty frame buffer check (stream disconnected)
        if frame is None or frame.size == 0:
            return CameraHealthStatus(
                anomaly=CameraAnomaly.STREAM_DISCONNECTED,
                is_healthy=False,
                details={"reason": "Frame buffer is None or empty"},
            )

        h, w = frame.shape[:2]
        current_res = (w, h)

        # 2. Check for unexpected mid-session resolution renegotiation
        if self._initial_resolution is None:
            self._initial_resolution = current_res
        elif self._initial_resolution != current_res:
            res_change_details = {
                "initial_resolution": list(self._initial_resolution),
                "current_resolution": list(current_res),
            }
            # Update baseline resolution to avoid continuous failure
            self._initial_resolution = current_res
            return CameraHealthStatus(
                anomaly=CameraAnomaly.RESOLUTION_CHANGED,
                is_healthy=False,
                resolution=current_res,
                details=res_change_details,
            )

        # 3. Check for delivery interruption / gap in timestamp
        gap_sec = 0.0
        prev_ts = self._last_timestamp
        if self._last_timestamp is not None:
            gap_sec = timestamp_seconds - self._last_timestamp
            if gap_sec > self.max_delivery_gap_seconds:
                self._last_timestamp = timestamp_seconds
                return CameraHealthStatus(
                    anomaly=CameraAnomaly.DELIVERY_GAP,
                    is_healthy=False,
                    gap_duration_seconds=gap_sec,
                    resolution=current_res,
                    details={
                        "reason": f"Frame delivery gap of {gap_sec:.2f}s exceeded threshold {self.max_delivery_gap_seconds:.2f}s",
                    },
                )

        self._last_timestamp = timestamp_seconds

        # 4. Photometric checks (black screen / extreme glare)
        if len(frame.shape) == 3 and frame.shape[2] in (3, 4):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            gray = frame[:, :, 0]
        elif len(frame.shape) == 2:
            gray = frame
        else:
            return CameraHealthStatus(
                anomaly=CameraAnomaly.STREAM_DISCONNECTED,
                is_healthy=False,
                details={"reason": f"Unsupported frame dimensions: {frame.shape}"},
            )
        mean_luma = float(np.mean(gray))
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if mean_luma < self.min_dark_luma and blur_var < 5.0:
            return CameraHealthStatus(
                anomaly=CameraAnomaly.BLACK_SCREEN,
                is_healthy=False,
                mean_luminance=mean_luma,
                blur_variance=blur_var,
                resolution=current_res,
                details={"reason": "Frame is black/dark with near-zero texture variance"},
            )

        if mean_luma > self.max_glare_luma:
            return CameraHealthStatus(
                anomaly=CameraAnomaly.EXTREME_GLARE,
                is_healthy=False,
                mean_luminance=mean_luma,
                blur_variance=blur_var,
                resolution=current_res,
                details={"reason": "Frame is overexposed with extreme glare"},
            )

        # 5. Frame Freeze Detection (downsampled 64x48 thumbnail MSE)
        thumb = cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA)

        freeze_duration = 0.0
        is_frozen = False

        if self._last_thumbnail is not None:
            mse = float(
                np.mean((thumb.astype(np.float32) - self._last_thumbnail.astype(np.float32)) ** 2)
            )
            if mse <= self.diff_threshold_mse:
                if self._freeze_start_timestamp is None:
                    self._freeze_start_timestamp = (
                        prev_ts if prev_ts is not None else timestamp_seconds
                    )
                freeze_duration = timestamp_seconds - self._freeze_start_timestamp
                if freeze_duration >= self.frozen_frame_duration_seconds:
                    is_frozen = True
            else:
                self._freeze_start_timestamp = None
        else:
            self._freeze_start_timestamp = None

        self._last_thumbnail = thumb

        if is_frozen:
            return CameraHealthStatus(
                anomaly=CameraAnomaly.FRAME_FROZEN,
                is_healthy=False,
                is_frozen=True,
                freeze_duration_seconds=freeze_duration,
                mean_luminance=mean_luma,
                blur_variance=blur_var,
                resolution=current_res,
                details={
                    "reason": f"Webcam feed appears frozen for {freeze_duration:.2f}s",
                },
            )

        return CameraHealthStatus(
            anomaly=CameraAnomaly.NONE,
            is_healthy=True,
            is_frozen=False,
            freeze_duration_seconds=freeze_duration,
            gap_duration_seconds=gap_sec,
            mean_luminance=mean_luma,
            blur_variance=blur_var,
            resolution=current_res,
        )
