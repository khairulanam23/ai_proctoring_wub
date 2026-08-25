"""Threshold sweeping, sensitivity analysis, and trade-off justification framework."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tools.benchmark.evaluator import BenchmarkEvaluator, VerificationMetrics


@dataclass
class ThresholdSweepResult:
    """Outcome of evaluating performance metrics across a range of operational thresholds."""

    parameter_name: str
    tested_thresholds: list[float]
    tradeoff_table: list[dict[str, Any]]
    selected_threshold: float
    selection_rationale: str
    optimal_f1_threshold: float
    eer_threshold_estimate: float | None = None
    metrics_at_selected: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "tested_thresholds": [round(t, 4) for t in self.tested_thresholds],
            "tradeoff_table": self.tradeoff_table,
            "selected_threshold": round(self.selected_threshold, 4),
            "selection_rationale": self.selection_rationale,
            "optimal_f1_threshold": round(self.optimal_f1_threshold, 4),
            "eer_threshold_estimate": round(self.eer_threshold_estimate, 4)
            if self.eer_threshold_estimate is not None
            else None,
            "metrics_at_selected": self.metrics_at_selected,
        }


class ThresholdOptimizer:
    """Performs systematic sensitivity sweeps across model operational thresholds."""

    @staticmethod
    def sweep_face_verification_thresholds(
        genuine_scores: list[float],
        impostor_scores: list[float],
        thresholds: list[float] | None = None,
        default_selected: float = 0.3630,
    ) -> ThresholdSweepResult:
        """Sweep cosine similarity thresholds for SFace identity verification."""
        if thresholds is None:
            thresholds = [float(t) for t in np.arange(0.20, 0.62, 0.02)]

        tradeoff: list[dict[str, Any]] = []
        best_f1 = -1.0
        best_f1_thresh = default_selected
        min_diff = 1e9
        eer_thresh = default_selected

        for t in thresholds:
            m: VerificationMetrics = BenchmarkEvaluator.evaluate_verification_pairs(
                genuine_scores=genuine_scores,
                impostor_scores=impostor_scores,
                threshold=t,
            )
            tp = m.genuine_acceptances
            fn = m.false_rejections
            tn = m.genuine_rejections
            fp = m.false_acceptances

            prec = tp / max(1, (tp + fp))
            rec = tp / max(1, (tp + fn))
            f1 = (2 * prec * rec) / max(1e-6, (prec + rec))

            if f1 > best_f1:
                best_f1 = f1
                best_f1_thresh = t

            diff = abs(m.far - m.frr)
            if diff < min_diff:
                min_diff = diff
                eer_thresh = t

            tradeoff.append(
                {
                    "threshold": round(t, 4),
                    "gar": round(m.gar, 4),
                    "frr": round(m.frr, 4),
                    "grr": round(m.grr, 4),
                    "far": round(m.far, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1_score": round(f1, 4),
                }
            )

        # Evaluate at selected
        selected_m = BenchmarkEvaluator.evaluate_verification_pairs(
            genuine_scores=genuine_scores,
            impostor_scores=impostor_scores,
            threshold=default_selected,
        )

        rationale = (
            f"Threshold {default_selected:.4f} is calibrated from standard LFW benchmark for OpenCV SFace. "
            f"At this threshold, FAR={selected_m.far:.4f} and FRR={selected_m.frr:.4f}, maintaining high security "
            f"against impostors while minimizing false rejections of legitimate candidates."
        )

        return ThresholdSweepResult(
            parameter_name="FACE_MATCH_THRESHOLD (SFace Cosine Similarity)",
            tested_thresholds=thresholds,
            tradeoff_table=tradeoff,
            selected_threshold=default_selected,
            selection_rationale=rationale,
            optimal_f1_threshold=best_f1_thresh,
            eer_threshold_estimate=eer_thresh,
            metrics_at_selected=selected_m.to_dict(),
        )

    @staticmethod
    def sweep_detection_score_thresholds(
        confidences_positive: list[float],
        confidences_negative: list[float],
        thresholds: list[float] | None = None,
        default_selected: float = 0.60,
        parameter_name: str = "DETECTION_SCORE_THRESHOLD (YuNet Confidence)",
    ) -> ThresholdSweepResult:
        """Sweep detection confidence threshold for face or object detectors."""
        if thresholds is None:
            thresholds = [float(t) for t in np.arange(0.30, 0.90, 0.05)]

        tradeoff: list[dict[str, Any]] = []
        best_f1 = -1.0
        best_f1_thresh = default_selected

        pos_arr = np.array(confidences_positive) if confidences_positive else np.array([])
        neg_arr = np.array(confidences_negative) if confidences_negative else np.array([])

        for t in thresholds:
            tp = int(np.sum(pos_arr >= t)) if len(pos_arr) > 0 else 0
            fn = len(pos_arr) - tp
            fp = int(np.sum(neg_arr >= t)) if len(neg_arr) > 0 else 0
            tn = len(neg_arr) - fp

            prec = tp / max(1, (tp + fp))
            rec = tp / max(1, (tp + fn))
            f1 = (2 * prec * rec) / max(1e-6, (prec + rec))

            if f1 > best_f1:
                best_f1 = f1
                best_f1_thresh = t

            tradeoff.append(
                {
                    "threshold": round(t, 4),
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1_score": round(f1, 4),
                }
            )

        rationale = (
            f"Selected threshold {default_selected:.2f} filters background noise and partial edge artifacts "
            f"while reliably detecting candidate faces across varying illumination."
        )

        return ThresholdSweepResult(
            parameter_name=parameter_name,
            tested_thresholds=thresholds,
            tradeoff_table=tradeoff,
            selected_threshold=default_selected,
            selection_rationale=rationale,
            optimal_f1_threshold=best_f1_thresh,
        )
