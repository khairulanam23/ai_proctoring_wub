"""Normalized gaze tracking, relative calibration, and viewing zone classification.

Webcam-based gaze estimation relies on relative iris displacement within the eye aperture
derived from the MediaPipe refined face mesh landmarks (indices 468-477).

TECHNICAL NOTICE & BOUNDARIES:
    Webcam iris tracking is a normalized *relative proxy* for attention direction.
    It does not possess the sub-degree precision or angular resolution of dedicated
    hardware eye trackers (e.g. infrared corneal reflection devices).
    Natural brief glances, saccades, and posture adjustments are expected and normal.
    The proctoring engine classifies normalized directional zones (CENTER, LEFT, RIGHT,
    UP, DOWN) and relies on temporal persistence and exam profiles (DIGITAL_SCREEN vs
    PHYSICAL_PAPER) to avoid false accusations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

# Landmark indices for eyes and irises in the MediaPipe 478-point canonical face mesh.
_LEFT_IRIS = (468, 469, 470, 471, 472)
_RIGHT_IRIS = (473, 474, 475, 476, 477)

_LEFT_EYE_INNER = 133
_LEFT_EYE_OUTER = 33
_LEFT_EYE_TOP = 159
_LEFT_EYE_BOTTOM = 145

_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263
_RIGHT_EYE_TOP = 386
_RIGHT_EYE_BOTTOM = 374


class GazeDirection(str, Enum):
    """Categorical gaze direction relative to the candidate's neutral baseline."""

    CENTER = "CENTER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


@dataclass
class GazeCalibration:
    """Neutral baseline horizontal and vertical gaze offsets."""

    baseline_horizontal: float = 0.0
    baseline_vertical: float = 0.0
    is_calibrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_horizontal": round(self.baseline_horizontal, 4),
            "baseline_vertical": round(self.baseline_vertical, 4),
            "is_calibrated": self.is_calibrated,
        }


@dataclass
class GazeObservation:
    """Normalized eye gaze measurement for one frame."""

    horizontal: float = 0.0
    """Normalized horizontal displacement (-1.0 to +1.0, negative = right, positive = left)."""

    vertical: float = 0.0
    """Normalized vertical displacement (-1.0 to +1.0, negative = down, positive = up)."""

    offset: float = 0.0
    """Euclidean magnitude of normalized displacement from neutral baseline."""

    direction: GazeDirection = GazeDirection.CENTER
    confidence: float = 0.0
    is_off_screen: bool = False
    raw_horizontal: float = 0.0
    raw_vertical: float = 0.0
    left_iris_ratio: tuple[float, float] | None = None
    right_iris_ratio: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizontal": round(self.horizontal, 4),
            "vertical": round(self.vertical, 4),
            "offset": round(self.offset, 4),
            "direction": self.direction.value,
            "confidence": round(self.confidence, 4),
            "is_off_screen": self.is_off_screen,
            "raw_horizontal": round(self.raw_horizontal, 4),
            "raw_vertical": round(self.raw_vertical, 4),
            "left_iris_ratio": (
                [round(self.left_iris_ratio[0], 3), round(self.left_iris_ratio[1], 3)]
                if self.left_iris_ratio
                else None
            ),
            "right_iris_ratio": (
                [round(self.right_iris_ratio[0], 3), round(self.right_iris_ratio[1], 3)]
                if self.right_iris_ratio
                else None
            ),
        }


class GazeTracker:
    """Extracts normalized 2D gaze vectors and categorical gaze directions from face landmarks."""

    def __init__(
        self,
        horizontal_threshold: float = 0.32,
        vertical_up_threshold: float = 0.28,
        vertical_down_threshold: float = 0.32,
        deadband: float = 0.12,
        smoothing_alpha: float = 0.65,
        hysteresis_margin: float = 0.04,
    ) -> None:
        self.horizontal_threshold = float(horizontal_threshold)
        self.vertical_up_threshold = float(vertical_up_threshold)
        self.vertical_down_threshold = float(vertical_down_threshold)
        self.deadband = float(deadband)
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.05, 1.0))
        self.hysteresis_margin = float(max(0.0, hysteresis_margin))

        self.calibration = GazeCalibration()
        self._smoothed_h: float | None = None
        self._smoothed_v: float | None = None
        self._current_direction: GazeDirection = GazeDirection.CENTER

    def reset(self) -> None:
        """Reset temporal smoothing filter, directional hysteresis state, and calibration."""
        self._smoothed_h = None
        self._smoothed_v = None
        self._current_direction = GazeDirection.CENTER
        self.calibration = GazeCalibration()

    def calibrate(self, raw_samples: list[tuple[float, float]]) -> dict[str, Any]:
        """Learn neutral baseline gaze from collected (horizontal, vertical) sample pairs."""
        if len(raw_samples) < 3:
            return {
                "calibrated": False,
                "reason": f"Only {len(raw_samples)} sample(s); need at least 3",
            }

        horizontals = [s[0] for s in raw_samples]
        verticals = [s[1] for s in raw_samples]

        h_std = float(np.std(horizontals))
        v_std = float(np.std(verticals))

        # Reject calibration if candidate moved around excessively
        if h_std > 0.18 or v_std > 0.18:
            return {
                "calibrated": False,
                "reason": f"Gaze varied too much during calibration (h_sd {h_std:.2f}, v_sd {v_std:.2f})",
            }

        self.calibration.baseline_horizontal = float(np.median(horizontals))
        self.calibration.baseline_vertical = float(np.median(verticals))
        self.calibration.is_calibrated = True

        return {
            "calibrated": True,
            "samples": len(raw_samples),
            "baseline_horizontal": round(self.calibration.baseline_horizontal, 4),
            "baseline_vertical": round(self.calibration.baseline_vertical, 4),
            "h_stability": round(h_std, 4),
            "v_stability": round(v_std, 4),
        }

    def reset_calibration(self) -> None:
        self.calibration = GazeCalibration()

    def measure(
        self,
        landmarks: np.ndarray,
        gaze_offset_limit: float | None = None,
        gaze_horizontal_limit: float | None = None,
        gaze_vertical_up_limit: float | None = None,
        gaze_vertical_down_limit: float | None = None,
    ) -> GazeObservation:
        """Compute normalized gaze coordinates from dense landmarks."""
        if landmarks is None or landmarks.shape[0] <= max(_RIGHT_IRIS):
            return GazeObservation(direction=GazeDirection.UNKNOWN, confidence=0.0)

        # Left eye measurements
        left_h, left_v, left_valid = self._measure_eye(
            landmarks=landmarks,
            iris_indices=_LEFT_IRIS,
            inner_idx=_LEFT_EYE_INNER,
            outer_idx=_LEFT_EYE_OUTER,
            top_idx=_LEFT_EYE_TOP,
            bottom_idx=_LEFT_EYE_BOTTOM,
        )

        # Right eye measurements
        right_h, right_v, right_valid = self._measure_eye(
            landmarks=landmarks,
            iris_indices=_RIGHT_IRIS,
            inner_idx=_RIGHT_EYE_INNER,
            outer_idx=_RIGHT_EYE_OUTER,
            top_idx=_RIGHT_EYE_TOP,
            bottom_idx=_RIGHT_EYE_BOTTOM,
        )

        if not left_valid and not right_valid:
            return GazeObservation(direction=GazeDirection.UNKNOWN, confidence=0.0)

        if left_valid and right_valid:
            raw_h = (left_h + right_h) / 2.0
            raw_v = (left_v + right_v) / 2.0
            confidence = 0.95
        elif left_valid:
            raw_h = left_h
            raw_v = left_v
            confidence = 0.70
        else:
            raw_h = right_h
            raw_v = right_v
            confidence = 0.70

        # Adjust relative to calibrated neutral baseline
        raw_h_norm = raw_h - self.calibration.baseline_horizontal
        raw_v_norm = raw_v - self.calibration.baseline_vertical

        # Apply exponential moving average (EMA) temporal smoothing with saccadic jump tracking
        if (
            self._smoothed_h is None
            or self.smoothing_alpha >= 1.0
            or abs(raw_h_norm - self._smoothed_h) > 0.25
            or abs(raw_v_norm - self._smoothed_v) > 0.25
        ):
            h_norm = raw_h_norm
            v_norm = raw_v_norm
        else:
            h_norm = (
                self.smoothing_alpha * raw_h_norm + (1.0 - self.smoothing_alpha) * self._smoothed_h
            )
            v_norm = (
                self.smoothing_alpha * raw_v_norm + (1.0 - self.smoothing_alpha) * self._smoothed_v
            )

        self._smoothed_h = h_norm
        self._smoothed_v = v_norm
        offset = float(np.sqrt(h_norm**2 + v_norm**2))

        # Determine direction with hysteresis
        direction = self._classify_direction(h_norm, v_norm)
        self._current_direction = direction

        # Evaluate off-screen condition against provided limits (or defaults)
        h_limit = gaze_horizontal_limit or self.horizontal_threshold
        v_up_limit = gaze_vertical_up_limit or self.vertical_up_threshold
        v_down_limit = gaze_vertical_down_limit or self.vertical_down_threshold
        scalar_limit = gaze_offset_limit or self.horizontal_threshold

        is_off_screen = (
            abs(h_norm) > h_limit
            or (v_norm > 0 and v_norm > v_up_limit)
            or (v_norm < 0 and abs(v_norm) > v_down_limit)
            or offset > scalar_limit
        )

        return GazeObservation(
            horizontal=h_norm,
            vertical=v_norm,
            offset=offset,
            direction=direction,
            confidence=confidence,
            is_off_screen=is_off_screen,
            raw_horizontal=raw_h_norm,
            raw_vertical=raw_v_norm,
            left_iris_ratio=(left_h, left_v) if left_valid else None,
            right_iris_ratio=(right_h, right_v) if right_valid else None,
        )

    def _classify_direction(self, h: float, v: float) -> GazeDirection:
        """Classify normalized (h, v) into a GazeDirection with directional hysteresis."""
        abs_h = abs(h)
        abs_v = abs(v)

        # Dynamic deadband based on current direction state
        effective_deadband = (
            self.deadband - self.hysteresis_margin
            if self._current_direction != GazeDirection.CENTER
            else self.deadband + self.hysteresis_margin
        )

        if abs_h <= effective_deadband and abs_v <= effective_deadband:
            return GazeDirection.CENTER

        # Dominant axis check
        if abs_h >= abs_v:
            if abs_h > effective_deadband:
                return GazeDirection.LEFT if h > 0 else GazeDirection.RIGHT
        else:
            if abs_v > effective_deadband:
                return GazeDirection.UP if v > 0 else GazeDirection.DOWN

        return GazeDirection.CENTER

    @staticmethod
    def _measure_eye(
        landmarks: np.ndarray,
        iris_indices: tuple[int, ...],
        inner_idx: int,
        outer_idx: int,
        top_idx: int,
        bottom_idx: int,
    ) -> tuple[float, float, bool]:
        """Compute horizontal and vertical iris position ratios within one eye aperture."""
        iris_points = landmarks[list(iris_indices)]
        iris_center = iris_points.mean(axis=0)

        inner = landmarks[inner_idx]
        outer = landmarks[outer_idx]
        top = landmarks[top_idx]
        bottom = landmarks[bottom_idx]

        eye_width = float(np.linalg.norm(outer - inner))
        eye_height = float(np.linalg.norm(bottom - top))

        if eye_width < 1e-4 or eye_height < 1e-4:
            return 0.0, 0.0, False

        eye_center_x = (inner[0] + outer[0]) / 2.0
        eye_center_y = (top[1] + bottom[1]) / 2.0

        # Normalized displacements (-1.0 to 1.0)
        h_ratio = float((iris_center[0] - eye_center_x) / (eye_width / 2.0))
        # Note: In pixel space, y increases downwards. So top_y < bottom_y.
        # Negative ratio = looking down (iris lower than eye center)
        v_ratio = float(-(iris_center[1] - eye_center_y) / (eye_height / 2.0))

        return (
            float(np.clip(h_ratio, -1.0, 1.0)),
            float(np.clip(v_ratio, -1.0, 1.0)),
            True,
        )
