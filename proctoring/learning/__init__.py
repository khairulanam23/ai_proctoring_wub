"""Proctoring continuous learning and dataset management system.

Provides structured, human-in-the-loop workflows for capturing runtime edge cases,
producing versioned datasets with session-grouped splits, evaluating candidate
models using champion-vs-challenger comparisons against a permanent regression suite,
and maintaining a traceable model registry.
"""

from proctoring.learning.annotations import (
    AnnotatedBBox,
    HumanAnnotationRecord,
    ReviewStatus,
    TargetObjectLabel,
    YOLO_LABEL_MAP,
    get_yolo_class_names,
)
from proctoring.learning.datasets import DatasetManager, DatasetManifest
from proctoring.learning.evaluation import (
    ChampionChallengerEvaluator,
    ComparativeEvaluationReport,
)
from proctoring.learning.inbox import FlagReason, InboxSample, TrainingInboxManager
from proctoring.learning.pipeline import (
    TrainingConfig,
    TrainingPipeline,
    TrainingRunSummary,
)
from proctoring.learning.registry import (
    ModelMetadata,
    ModelMetrics,
    ModelRegistry,
    ModelStatus,
)

__all__ = [
    "AnnotatedBBox",
    "HumanAnnotationRecord",
    "ReviewStatus",
    "TargetObjectLabel",
    "YOLO_LABEL_MAP",
    "get_yolo_class_names",
    "DatasetManager",
    "DatasetManifest",
    "ChampionChallengerEvaluator",
    "ComparativeEvaluationReport",
    "FlagReason",
    "InboxSample",
    "TrainingInboxManager",
    "TrainingConfig",
    "TrainingPipeline",
    "TrainingRunSummary",
    "ModelMetadata",
    "ModelMetrics",
    "ModelRegistry",
    "ModelStatus",
]
