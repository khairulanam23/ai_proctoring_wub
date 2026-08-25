"""Workflow stage 3 — frame validation, quality assessment and illumination normalisation."""

from proctoring.preprocessing.face_preprocessing import (
    FacePreprocessor,
    PoseMetrics,
    PreprocessingResult,
    PreprocessingStatus,
    QualityMetrics,
)
from proctoring.preprocessing.frame_quality import (
    AdaptiveImagePreprocessor,
    AdaptivePreprocessorConfig,
    FrameGateResult,
    FrameQualityGate,
)

__all__ = [
    "AdaptiveImagePreprocessor",
    "AdaptivePreprocessorConfig",
    "FrameGateResult",
    "FrameQualityGate",
    "FacePreprocessor",
    "PoseMetrics",
    "PreprocessingResult",
    "PreprocessingStatus",
    "QualityMetrics",
]
