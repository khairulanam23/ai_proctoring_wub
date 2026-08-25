"""Workflow stages 4-6 — face detection, identity verification and scene observation."""

from proctoring.detection.face_detector import (
    DetectionResult,
    FaceDetection,
    FaceDetector,
)
from proctoring.detection.face_presence import (
    FacePresenceAnalyzer,
    FacePresenceResult,
    FacePresenceStatus,
)
from proctoring.detection.face_verifier import (
    FaceVerifier,
    MultiReferenceVerificationResult,
    VerificationResult,
)
from proctoring.detection.object_detector import (
    DetectedObject,
    ObjectDetectionResult,
    ObjectDetector,
)
from proctoring.detection.object_relevance import (
    ObjectRelevanceFilter,
    ProctoringDetectionReport,
)

__all__ = [
    "DetectionResult",
    "FaceDetection",
    "FaceDetector",
    "FaceVerifier",
    "MultiReferenceVerificationResult",
    "VerificationResult",
    "FacePresenceAnalyzer",
    "FacePresenceResult",
    "FacePresenceStatus",
    "DetectedObject",
    "ObjectDetectionResult",
    "ObjectDetector",
    "ObjectRelevanceFilter",
    "ProctoringDetectionReport",
]
