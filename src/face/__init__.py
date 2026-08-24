"""Face detection, presence analysis, and verification package."""

from src.face.detector import DetectionResult, FaceDetection, FaceDetector
from src.face.presence import FacePresenceAnalyzer, FacePresenceResult, FacePresenceStatus
from src.face.verifier import FaceVerifier, VerificationResult
from src.face.video import PresenceEvent, VideoAnalysisSummary, VideoPresenceAnalyzer, format_timestamp

__all__ = [
    "FaceDetector",
    "FaceDetection",
    "DetectionResult",
    "FacePresenceAnalyzer",
    "FacePresenceResult",
    "FacePresenceStatus",
    "FaceVerifier",
    "VerificationResult",
    "VideoPresenceAnalyzer",
    "VideoAnalysisSummary",
    "PresenceEvent",
    "format_timestamp",
]
