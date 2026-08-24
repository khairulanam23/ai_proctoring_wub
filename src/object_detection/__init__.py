"""Object detection, relevance filtering, video sampling, temporal events, evidence packaging, and robustness package."""

from src.object_detection.detector import (
    DetectedObject,
    ObjectDetectionResult,
    ObjectDetector,
)
from src.object_detection.evidence import (
    EvidenceBuilder,
    EvidenceEvent,
    EvidenceFrame,
    EvidenceManifest,
    EvidenceObject,
    EvidencePackage,
    EvidenceSource,
    FaceEvidenceAdapter,
)
from src.object_detection.relevance import (
    DEFAULT_CLASS_THRESHOLDS,
    DEFAULT_PROCTORING_RELEVANT_CLASSES,
    ObjectRelevanceFilter,
    ProctoringDetectionReport,
)
from src.object_detection.robustness import (
    ConfusionMetrics,
    ImageAugmenter,
    ThresholdEvaluationResult,
    ThresholdEvaluator,
    VisualCondition,
    evaluate_detections,
    generate_visual_comparison_grid,
)
from src.object_detection.sampling import (
    FrameSample,
    VideoFrameSampler,
    VideoMetadata,
    format_timestamp,
)
from src.object_detection.temporal import (
    ObjectPresenceEvent,
    PersonCountChangeEvent,
    TemporalEventEngine,
    TemporalEventReport,
    render_temporal_gantt_chart,
)
from src.object_detection.video import (
    ObjectPresenceInterval,
    TimelineEntry,
    VideoAnalysisReport,
    VideoObjectAnalyzer,
)

__all__ = [
    "ObjectDetector",
    "DetectedObject",
    "ObjectDetectionResult",
    "ObjectRelevanceFilter",
    "ProctoringDetectionReport",
    "DEFAULT_PROCTORING_RELEVANT_CLASSES",
    "DEFAULT_CLASS_THRESHOLDS",
    "FrameSample",
    "VideoMetadata",
    "VideoFrameSampler",
    "format_timestamp",
    "TimelineEntry",
    "ObjectPresenceInterval",
    "VideoAnalysisReport",
    "VideoObjectAnalyzer",
    "ObjectPresenceEvent",
    "PersonCountChangeEvent",
    "TemporalEventEngine",
    "TemporalEventReport",
    "render_temporal_gantt_chart",
    "EvidenceObject",
    "EvidenceFrame",
    "EvidenceEvent",
    "EvidenceSource",
    "EvidenceManifest",
    "EvidencePackage",
    "FaceEvidenceAdapter",
    "EvidenceBuilder",
    "VisualCondition",
    "ImageAugmenter",
    "ConfusionMetrics",
    "evaluate_detections",
    "ThresholdEvaluationResult",
    "ThresholdEvaluator",
    "generate_visual_comparison_grid",
]
