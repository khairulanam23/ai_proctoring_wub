"""Adaptive image preprocessing, contrast-limited adaptive histogram equalization (CLAHE), and dynamic illumination normalization."""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class AdaptivePreprocessorConfig:
    """Configuration parameters for adaptive image enhancement."""

    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    low_light_threshold_luma: float = 65.0
    high_light_threshold_luma: float = 195.0
    min_blur_laplacian_var: float = 70.0
    target_width: int = 640
    target_height: int = 480

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_clahe": self.enable_clahe,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_tile_grid_size": list(self.clahe_tile_grid_size),
            "low_light_threshold_luma": self.low_light_threshold_luma,
            "high_light_threshold_luma": self.high_light_threshold_luma,
            "min_blur_laplacian_var": self.min_blur_laplacian_var,
            "target_resolution": [self.target_width, self.target_height],
        }


class AdaptiveImagePreprocessor:
    """Fast, adaptive image enhancement pipeline for extreme lighting conditions."""

    def __init__(self, config: AdaptivePreprocessorConfig | None = None) -> None:
        self.config = config or AdaptivePreprocessorConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size,
        )

    def assess_quality(self, image: np.ndarray) -> dict[str, Any]:
        """Measure frame luminance, contrast, and Laplacian blur variance."""
        if image is None or image.size == 0:
            return {"valid": False, "mean_luma": 0.0, "blur_var": 0.0, "is_blurred": True}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_luma = float(np.mean(gray))
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurred = blur_var < self.config.min_blur_laplacian_var

        return {
            "valid": True,
            "mean_luma": mean_luma,
            "blur_var": blur_var,
            "is_blurred": is_blurred,
            "is_low_light": mean_luma < self.config.low_light_threshold_luma,
            "is_high_light": mean_luma > self.config.high_light_threshold_luma,
        }

    def enhance(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply adaptive CLAHE enhancement to LAB lightness channel if illumination is extreme."""
        if image is None or image.size == 0:
            return image, {"enhanced": False, "reason": "empty_image"}

        quality = self.assess_quality(image)
        if not quality["valid"]:
            return image, {"enhanced": False, "reason": "invalid_image"}

        # Only apply CLAHE if low-light or backlighting detected to save CPU cycles
        if self.config.enable_clahe and (quality["is_low_light"] or quality["is_high_light"]):
            # Convert BGR to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # Apply CLAHE to L-channel
            cl = self._clahe.apply(l_channel)

            # Merge channels and convert back to BGR
            enhanced_lab = cv2.merge((cl, a_channel, b_channel))
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            return enhanced_bgr, {
                "enhanced": True,
                "condition": "clahe_applied",
                "quality": quality,
            }

        return image, {"enhanced": False, "condition": "normal_lighting", "quality": quality}


@dataclass
class FrameGateResult:
    """Outcome of running one frame through the stage-3 validation + enhancement gate."""

    accepted: bool
    frame: np.ndarray | None
    rejection_reason: str | None = None
    was_enhanced: bool = False
    mean_luminance: float | None = None
    blur_variance: float | None = None
    is_low_light: bool = False
    is_high_light: bool = False
    is_blurred: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "was_enhanced": self.was_enhanced,
            "mean_luminance": round(self.mean_luminance, 2)
            if self.mean_luminance is not None
            else None,
            "blur_variance": round(self.blur_variance, 2)
            if self.blur_variance is not None
            else None,
            "is_low_light": self.is_low_light,
            "is_high_light": self.is_high_light,
            "is_blurred": self.is_blurred,
        }


class FrameQualityGate:
    """Workflow stage 3: validate a frame, then normalise its illumination.

    Combines two checks that used to live apart — structural validation (was this
    buffer actually a usable image?) and photometric assessment (is it bright and
    sharp enough to detect a face in?) — so that every frame entering the pipeline,
    whether from a live webcam or a recorded file, passes the same gate.

    A rejected frame is *not* an error and *not* a proctoring event: it is a
    capture fault.  The caller records it as a skipped frame in telemetry and moves
    on, which is what keeps a flaky USB camera from generating spurious ``NO_FACE``
    incidents against the candidate.
    """

    def __init__(
        self,
        preprocessor: AdaptiveImagePreprocessor | None = None,
        min_width: int = 160,
        min_height: int = 120,
        enable_enhancement: bool = True,
    ) -> None:
        self.preprocessor = preprocessor or AdaptiveImagePreprocessor()
        self.min_width = int(min_width)
        self.min_height = int(min_height)
        self.enable_enhancement = bool(enable_enhancement)

    def validate(self, frame: Any) -> tuple[bool, str | None]:
        """Structural validation: is this buffer a usable BGR image of adequate size?"""
        if frame is None:
            return False, "Frame buffer is None (camera returned no data)"
        if not isinstance(frame, np.ndarray):
            return False, f"Frame is not a numpy array (got {type(frame).__name__})"
        if frame.size == 0:
            return False, "Frame has zero pixels"
        if frame.ndim != 3 or frame.shape[2] != 3:
            return False, f"Expected a 3-channel BGR frame, got shape {tuple(frame.shape)}"
        h, w = frame.shape[:2]
        if w < self.min_width or h < self.min_height:
            return (
                False,
                f"Frame resolution {w}x{h} below minimum {self.min_width}x{self.min_height}",
            )
        return True, None

    def process(self, frame: Any) -> FrameGateResult:
        """Validate the frame and, when illumination is extreme, return a CLAHE-enhanced copy.

        Enhancement is applied only under low-light or backlit conditions.  Running
        CLAHE unconditionally would cost roughly a millisecond per frame for no
        detection benefit on a well-lit desk, and can amplify sensor noise.
        """
        is_valid, reason = self.validate(frame)
        if not is_valid:
            return FrameGateResult(accepted=False, frame=None, rejection_reason=reason)

        quality = self.preprocessor.assess_quality(frame)
        if not quality.get("valid", False):
            return FrameGateResult(
                accepted=False,
                frame=None,
                rejection_reason="Frame failed photometric quality assessment",
            )

        out_frame = frame
        was_enhanced = False
        if self.enable_enhancement:
            out_frame, enhance_info = self.preprocessor.enhance(frame)
            was_enhanced = bool(enhance_info.get("enhanced", False))

        # A blurred frame is still processed — motion blur during a head turn is
        # normal — but the flag is carried into the timeline so a proctor reviewing
        # a marginal detection can see the frame was soft.
        return FrameGateResult(
            accepted=True,
            frame=out_frame,
            was_enhanced=was_enhanced,
            mean_luminance=quality.get("mean_luma"),
            blur_variance=quality.get("blur_var"),
            is_low_light=bool(quality.get("is_low_light", False)),
            is_high_light=bool(quality.get("is_high_light", False)),
            is_blurred=bool(quality.get("is_blurred", False)),
        )
