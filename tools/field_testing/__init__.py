"""Real-World Dataset, Field Testing, and Long-Duration Session Validation Framework."""

from tools.field_testing.dataset_generator import (
    FieldTestDataset,
    FieldTestDatasetBuilder,
)
from tools.field_testing.evaluator import (
    ControlledVsFieldComparison,
    FieldBenchmarkEvaluator,
    FieldEvaluationMetrics,
)
from tools.field_testing.human_study import (
    HumanReviewerFeedback,
    HumanReviewStudyReport,
    HumanReviewStudySimulator,
)
from tools.field_testing.long_session import (
    LongDurationSessionSimulator,
    LongSessionProfileReport,
)
from tools.field_testing.runner import (
    ComprehensiveFieldTestReport,
    Phase6FieldTestRunner,
)
from tools.field_testing.schema import (
    AnnotatorAgreementMetrics,
    CameraMetadata,
    EnvironmentMetadata,
    FieldParticipant,
    FieldScenarioType,
    FieldSessionRecord,
    IndependentAnnotation,
    LightingCondition,
)

__all__ = [
    "LightingCondition",
    "CameraMetadata",
    "EnvironmentMetadata",
    "FieldParticipant",
    "FieldScenarioType",
    "FieldSessionRecord",
    "IndependentAnnotation",
    "AnnotatorAgreementMetrics",
    "FieldTestDatasetBuilder",
    "FieldTestDataset",
    "FieldEvaluationMetrics",
    "ControlledVsFieldComparison",
    "FieldBenchmarkEvaluator",
    "LongSessionProfileReport",
    "LongDurationSessionSimulator",
    "HumanReviewerFeedback",
    "HumanReviewStudyReport",
    "HumanReviewStudySimulator",
    "Phase6FieldTestRunner",
    "ComprehensiveFieldTestReport",
]
