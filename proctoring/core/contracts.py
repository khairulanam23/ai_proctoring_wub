"""Standardized detector contracts and typed interfaces for AI proctoring components.

Specifies input/output data types, coordinate conventions, confidence semantics,
provenance metadata, and failure behavior across all detectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np


class CoordinateConvention(str, Enum):
    """Supported bounding box coordinate conventions across detectors."""

    XYXY_PIXEL = "xyxy_pixel"  # (x1, y1, x2, y2) absolute pixel coordinates
    XYWH_PIXEL = "xywh_pixel"  # (x, y, w, h) absolute pixel coordinates (OpenCV native)
    NORMALIZED_XYXY = "normalized_xyxy"  # (x1, y1, x2, y2) in [0.0, 1.0] relative to frame width/height
    LANDMARK_PIXEL_2D = "landmark_pixel_2d"  # (x, y) absolute pixel coordinates
    LANDMARK_NORMALIZED_3D = "landmark_normalized_3d"  # (x, y, z) MediaPipe native normalized


class ConfidenceSemantics(str, Enum):
    """Semantic meaning of detector confidence scores."""

    PROBABILITY = "probability"  # Calibrated probability [0.0, 1.0] (e.g. YOLO softmax)
    COSINE_SIMILARITY = "cosine_similarity"  # Cosine similarity [-1.0, 1.0] (e.g. SFace embedding)
    DISTANCE_RATIO = "distance_ratio"  # Normalized ratio/distance (e.g. Gaze offset, blendshape)
    DETECTION_SCORE = "detection_score"  # Uncalibrated detector margin/confidence (e.g. YuNet score)


class FailureBehavior(str, Enum):
    """Standardized failure behavior when a detector encounters an unrecoverable fault."""

    EMIT_DIAGNOSTIC = "emit_diagnostic"  # Record TECHNICAL_DIAGNOSTIC, do not infer absence
    DEGRADE_OPTIONAL = "degrade_optional"  # Optional stage; mark unmeasured on observation


@dataclass(frozen=True)
class DetectorContract:
    """Explicit behavioral and interface contract for a pipeline detector or analyzer."""

    name: str
    version: str
    runtime_backend: str
    target_device: str  # "cpu" | "cuda:0"
    input_format: str  # e.g., "numpy.ndarray (H, W, 3) BGR uint8"
    output_type: type
    coordinate_convention: CoordinateConvention
    confidence_semantics: ConfidenceSemantics
    failure_behavior: FailureBehavior
    is_stateful: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "runtime_backend": self.runtime_backend,
            "target_device": self.target_device,
            "input_format": self.input_format,
            "output_type": self.output_type.__name__,
            "coordinate_convention": self.coordinate_convention.value,
            "confidence_semantics": self.confidence_semantics.value,
            "failure_behavior": self.failure_behavior.value,
            "is_stateful": self.is_stateful,
        }


@dataclass
class StandardDetection:
    """Standardized detection entity across all object and face detectors."""

    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    frame_index: int
    timestamp_seconds: float
    session_id: str | None = None
    track_id: str | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox_xyxy": list(self.bbox_xyxy),
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "session_id": self.session_id,
            "track_id": self.track_id,
            "metadata": self.metadata,
        }


# Standard coordinate transformation helpers
def xywh_to_xyxy(bbox_xywh: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Convert (x, y, w, h) to (x1, y1, x2, y2)."""
    x, y, w, h = bbox_xywh
    return int(x), int(y), int(x + w), int(y + h)


def xyxy_to_xywh(bbox_xyxy: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Convert (x1, y1, x2, y2) to (x, y, w, h)."""
    x1, y1, x2, y2 = bbox_xyxy
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def normalize_xyxy(
    bbox_xyxy: tuple[int, int, int, int], width: int, height: int
) -> tuple[float, float, float, float]:
    """Normalize (x1, y1, x2, y2) to [0.0, 1.0]."""
    w = max(1, width)
    h = max(1, height)
    return (
        round(bbox_xyxy[0] / w, 4),
        round(bbox_xyxy[1] / h, 4),
        round(bbox_xyxy[2] / w, 4),
        round(bbox_xyxy[3] / h, 4),
    )
