"""Face detection, preprocessing, presence analysis, and verification package."""

from src.face.detector import DetectionResult, FaceDetection, FaceDetector
from src.face.preprocessing import (
    FacePreprocessor,
    PoseMetrics,
    PreprocessingResult,
    PreprocessingStatus,
    QualityMetrics,
)
from src.face.presence import FacePresenceAnalyzer, FacePresenceResult, FacePresenceStatus
from src.face.verifier import FaceVerifier, MultiReferenceVerificationResult, VerificationResult
from src.face.video import PresenceEvent, VideoAnalysisSummary, VideoPresenceAnalyzer, format_timestamp

__all__ = [
    "FaceDetector",
    "FaceDetection",
    "DetectionResult",
    "FacePreprocessor",
    "PoseMetrics",
    "PreprocessingResult",
    "PreprocessingStatus",
    "QualityMetrics",
    "FacePresenceAnalyzer",
    "FacePresenceResult",
    "FacePresenceStatus",
    "FaceVerifier",
    "VerificationResult",
    "MultiReferenceVerificationResult",
    "VideoPresenceAnalyzer",
    "VideoAnalysisSummary",
    "PresenceEvent",
    "format_timestamp",
]
