"""Evidence quality validation ensuring all visual artifacts are readable, non-corrupt, and cryptographically verified."""

import hashlib
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from proctoring.core.events import EventType


@dataclass
class EvidenceValidationResult:
    """Result of evidence image quality and integrity verification."""

    is_valid: bool
    error_reason: str | None = None
    frame_shape: tuple[int, int, int] | None = None
    mean_luminance: float | None = None
    sha256_checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "error_reason": self.error_reason,
            "frame_shape": list(self.frame_shape) if self.frame_shape else None,
            "mean_luminance": round(self.mean_luminance, 2)
            if self.mean_luminance is not None
            else None,
            "sha256_checksum": self.sha256_checksum,
        }


class EvidenceQualityValidator:
    """Rigorous pre-flight validator for visual evidence artifacts."""

    @classmethod
    def validate_frame(
        cls,
        image: np.ndarray | None,
        timestamp_seconds: float,
        event_type: EventType | None = None,  # noqa: ARG003  (reserved: per-event thresholds)
        min_width: int = 160,
        min_height: int = 120,
    ) -> EvidenceValidationResult:
        """Validate that an evidence image is well-formed, non-empty, and within acceptable dimensions."""
        if image is None:
            return EvidenceValidationResult(
                is_valid=False, error_reason="Image array is None (missing frame)."
            )

        if not isinstance(image, np.ndarray):
            return EvidenceValidationResult(
                is_valid=False, error_reason=f"Invalid image type: {type(image)}."
            )

        if image.size == 0:
            return EvidenceValidationResult(is_valid=False, error_reason="Image array has 0 size.")

        if len(image.shape) != 3 or image.shape[2] != 3:
            return EvidenceValidationResult(
                is_valid=False, error_reason=f"Image shape must be (H, W, 3), got {image.shape}."
            )

        h, w, c = image.shape
        if w < min_width or h < min_height:
            return EvidenceValidationResult(
                is_valid=False,
                error_reason=f"Frame dimensions ({w}x{h}) below minimum ({min_width}x{min_height}).",
            )

        if timestamp_seconds < 0.0:
            return EvidenceValidationResult(
                is_valid=False, error_reason=f"Invalid negative timestamp: {timestamp_seconds}."
            )

        # Check for completely blank/dead sensor frames (all 0 or all 255)
        mean_val = float(np.mean(image))
        std_val = float(np.std(image))
        if std_val < 0.001 and (mean_val < 1.0 or mean_val > 254.0):
            return EvidenceValidationResult(
                is_valid=False,
                error_reason=f"Blank/dead sensor frame detected (mean={mean_val:.1f}, std={std_val:.3f}).",
            )

        # Compute SHA-256 checksum
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            return EvidenceValidationResult(
                is_valid=False, error_reason="Failed to encode image to JPEG buffer."
            )

        sha256 = hashlib.sha256(encoded.tobytes()).hexdigest()

        return EvidenceValidationResult(
            is_valid=True,
            error_reason=None,
            frame_shape=(h, w, c),
            mean_luminance=mean_val,
            sha256_checksum=sha256,
        )
