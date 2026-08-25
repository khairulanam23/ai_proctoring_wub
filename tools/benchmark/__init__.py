"""AI Proctoring Model Validation, Benchmarking, and Quality Gate framework."""

from tools.benchmark.dataset import (
    EvaluationCategory,
    EvaluationDataset,
    EvaluationDatasetBuilder,
    EvaluationSample,
    GroundTruthLabel,
)
from tools.benchmark.evaluator import (
    BenchmarkEvaluator,
    ComponentMetrics,
    VerificationMetrics,
)
from tools.benchmark.evidence_auditor import EvidenceAuditReport, EvidencePackageAuditor
from tools.benchmark.profiler import LatencyBenchmarkReport, PipelineLatencyProfiler
from tools.benchmark.runner import ComprehensiveBenchmarkReport, Phase5BenchmarkRunner
from tools.benchmark.thresholds import ThresholdOptimizer, ThresholdSweepResult

__all__ = [
    "GroundTruthLabel",
    "EvaluationCategory",
    "EvaluationSample",
    "EvaluationDataset",
    "EvaluationDatasetBuilder",
    "ComponentMetrics",
    "VerificationMetrics",
    "BenchmarkEvaluator",
    "ThresholdSweepResult",
    "ThresholdOptimizer",
    "PipelineLatencyProfiler",
    "LatencyBenchmarkReport",
    "EvidencePackageAuditor",
    "EvidenceAuditReport",
    "Phase5BenchmarkRunner",
    "ComprehensiveBenchmarkReport",
]
