"""Champion vs Challenger comparative evaluation framework for proctoring models.

Executes side-by-side evaluation of the incumbent production model (Champion)
against a candidate model (Challenger) across identical evaluation datasets and
the permanent regression suite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from proctoring.learning.registry import ModelMetrics

LOGGER = logging.getLogger(__name__)


@dataclass
class ComparativeEvaluationReport:
    """Detailed differential evaluation comparing champion and challenger models."""

    champion_id: str
    challenger_id: str
    evaluation_dataset: str
    champion_metrics: ModelMetrics
    challenger_metrics: ModelMetrics
    delta_precision: float
    delta_recall: float
    delta_fpr: float
    delta_fnr: float
    delta_latency_ms: float
    delta_latency_percent: float
    recommendation: str  # "PROCEED_TO_APPROVAL", "REJECT", "MANUAL_REVIEW_REQUIRED"
    passed_regression_invariants: bool
    reasons: list[str] = field(default_factory=list)
    evaluated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "champion_id": self.champion_id,
            "challenger_id": self.challenger_id,
            "evaluation_dataset": self.evaluation_dataset,
            "champion_metrics": self.champion_metrics.to_dict(),
            "challenger_metrics": self.challenger_metrics.to_dict(),
            "delta_precision": round(self.delta_precision, 4),
            "delta_recall": round(self.delta_recall, 4),
            "delta_fpr": round(self.delta_fpr, 4),
            "delta_fnr": round(self.delta_fnr, 4),
            "delta_latency_ms": round(self.delta_latency_ms, 2),
            "delta_latency_percent": round(self.delta_latency_percent, 2),
            "recommendation": self.recommendation,
            "passed_regression_invariants": self.passed_regression_invariants,
            "reasons": self.reasons,
            "evaluated_at_utc": self.evaluated_at_utc,
        }


class ChampionChallengerEvaluator:
    """Evaluates candidate models against production models using differential metrics."""

    def __init__(
        self,
        max_recall_degradation: float = 0.01,
        max_fpr_increase: float = 0.02,
        max_latency_increase_percent: float = 20.0,
    ) -> None:
        self.max_recall_degradation = max_recall_degradation
        self.max_fpr_increase = max_fpr_increase
        self.max_latency_increase_percent = max_latency_increase_percent

    def compare(
        self,
        champion_id: str,
        challenger_id: str,
        champion_metrics: ModelMetrics,
        challenger_metrics: ModelMetrics,
        dataset_name: str,
        regression_passed: bool = True,
    ) -> ComparativeEvaluationReport:
        """Compute differential metrics and formulate promotion recommendation."""
        d_prec = challenger_metrics.precision - champion_metrics.precision
        d_rec = challenger_metrics.recall - champion_metrics.recall
        d_fpr = challenger_metrics.false_positive_rate - champion_metrics.false_positive_rate
        d_fnr = challenger_metrics.false_negative_rate - champion_metrics.false_negative_rate
        d_lat = challenger_metrics.cpu_latency_ms - champion_metrics.cpu_latency_ms

        base_lat = max(1.0, champion_metrics.cpu_latency_ms)
        d_lat_pct = (d_lat / base_lat) * 100.0

        reasons: list[str] = []
        rejected = False

        if not regression_passed:
            reasons.append("Challenger failed one or more permanent regression suite invariants.")
            rejected = True

        if d_rec < -self.max_recall_degradation:
            reasons.append(
                f"Recall degraded by {abs(d_rec):.3f} (exceeds allowed tolerance {self.max_recall_degradation:.3f})."
            )
            rejected = True

        if d_fpr > self.max_fpr_increase:
            reasons.append(
                f"False positive rate increased by {d_fpr:.3f} (exceeds allowed increase {self.max_fpr_increase:.3f})."
            )
            rejected = True

        if d_lat_pct > self.max_latency_increase_percent:
            reasons.append(
                f"CPU latency increased by {d_lat_pct:.1f}% (exceeds latency budget {self.max_latency_increase_percent:.1f}%)."
            )
            rejected = True

        if rejected:
            recommendation = "REJECT"
        elif d_prec >= 0.0 and d_rec >= 0.0 and d_fpr <= 0.0:
            recommendation = "PROCEED_TO_APPROVAL"
            reasons.append("Challenger meets or exceeds champion across all accuracy and latency metrics.")
        else:
            recommendation = "MANUAL_REVIEW_REQUIRED"
            reasons.append("Challenger exhibits trade-offs requiring invigilator/human-in-the-loop sign-off.")

        report = ComparativeEvaluationReport(
            champion_id=champion_id,
            challenger_id=challenger_id,
            evaluation_dataset=dataset_name,
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            delta_precision=d_prec,
            delta_recall=d_rec,
            delta_fpr=d_fpr,
            delta_fnr=d_fnr,
            delta_latency_ms=d_lat,
            delta_latency_percent=d_lat_pct,
            recommendation=recommendation,
            passed_regression_invariants=regression_passed,
            reasons=reasons,
        )

        LOGGER.info(
            "Champion vs Challenger (%s vs %s): %s",
            champion_id,
            challenger_id,
            recommendation,
        )
        return report
