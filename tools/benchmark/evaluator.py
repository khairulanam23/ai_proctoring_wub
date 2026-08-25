"""Quantitative evaluation metrics, confusion matrices, and confidence analysis."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ComponentMetrics:
    """Standardized classification & detection performance metrics."""

    component_name: str
    total_samples: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    false_positive_rate: float
    confidence_distribution: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_name": self.component_name,
            "total_samples": self.total_samples,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "accuracy": round(self.accuracy, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "confidence_distribution": self.confidence_distribution,
        }


@dataclass
class VerificationMetrics:
    """Biometric verification metrics for face recognition systems."""

    total_pairs: int
    genuine_pairs: int
    impostor_pairs: int
    genuine_acceptances: int  # Genuine matches accepted (similarity >= T)
    false_rejections: int  # Genuine matches rejected (similarity < T)
    genuine_rejections: int  # Impostors rejected (similarity < T)
    false_acceptances: int  # Impostors accepted (similarity >= T)
    gar: float  # Genuine Acceptance Rate (TP / P)
    frr: float  # False Rejection Rate (FN / P)
    grr: float  # Genuine Rejection Rate (TN / N)
    far: float  # False Acceptance Rate (FP / N)
    threshold: float
    similarity_stats_genuine: dict[str, float] = field(default_factory=dict)
    similarity_stats_impostor: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pairs": self.total_pairs,
            "genuine_pairs": self.genuine_pairs,
            "impostor_pairs": self.impostor_pairs,
            "genuine_acceptances": self.genuine_acceptances,
            "false_rejections": self.false_rejections,
            "genuine_rejections": self.genuine_rejections,
            "false_acceptances": self.false_acceptances,
            "gar": round(self.gar, 4),
            "frr": round(self.frr, 4),
            "grr": round(self.grr, 4),
            "far": round(self.far, 4),
            "threshold": round(self.threshold, 4),
            "similarity_stats_genuine": {
                k: round(v, 4) for k, v in self.similarity_stats_genuine.items()
            },
            "similarity_stats_impostor": {
                k: round(v, 4) for k, v in self.similarity_stats_impostor.items()
            },
        }


class BenchmarkEvaluator:
    """Calculates factual quantitative metrics without artificial scoring."""

    @staticmethod
    def calculate_classification_metrics(
        component_name: str,
        tp: int,
        fp: int,
        tn: int,
        fn: int,
        confidences_tp: list[float] | None = None,
        confidences_fp: list[float] | None = None,
    ) -> ComponentMetrics:
        """Compute standard precision, recall, F1, accuracy, and confidence distributions."""
        total = tp + fp + tn + fn
        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = (2 * precision * recall) / max(1e-6, (precision + recall))
        accuracy = (tp + tn) / max(1, total)
        fpr = fp / max(1, (fp + tn))

        conf_tp = confidences_tp or []
        conf_fp = confidences_fp or []

        conf_dist = {
            "tp_mean_conf": float(np.mean(conf_tp)) if conf_tp else 0.0,
            "tp_min_conf": float(np.min(conf_tp)) if conf_tp else 0.0,
            "tp_max_conf": float(np.max(conf_tp)) if conf_tp else 0.0,
            "fp_mean_conf": float(np.mean(conf_fp)) if conf_fp else 0.0,
            "fp_max_conf": float(np.max(conf_fp)) if conf_fp else 0.0,
        }

        return ComponentMetrics(
            component_name=component_name,
            total_samples=total,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            false_positive_rate=fpr,
            confidence_distribution=conf_dist,
        )

    @staticmethod
    def evaluate_verification_pairs(
        genuine_scores: list[float],
        impostor_scores: list[float],
        threshold: float = 0.3630,
    ) -> VerificationMetrics:
        """Calculate GAR, FRR, GRR, FAR, and similarity distribution stats."""
        g_arr = np.array(genuine_scores, dtype=np.float64) if genuine_scores else np.array([])
        i_arr = np.array(impostor_scores, dtype=np.float64) if impostor_scores else np.array([])

        n_gen = len(g_arr)
        n_imp = len(i_arr)

        ga = int(np.sum(g_arr >= threshold)) if n_gen > 0 else 0
        fr = n_gen - ga

        gr = int(np.sum(i_arr < threshold)) if n_imp > 0 else 0
        fa = n_imp - gr

        gar = ga / max(1, n_gen)
        frr = fr / max(1, n_gen)
        grr = gr / max(1, n_imp)
        far = fa / max(1, n_imp)

        stats_gen = {
            "mean": float(np.mean(g_arr)) if n_gen > 0 else 0.0,
            "median": float(np.median(g_arr)) if n_gen > 0 else 0.0,
            "min": float(np.min(g_arr)) if n_gen > 0 else 0.0,
            "max": float(np.max(g_arr)) if n_gen > 0 else 0.0,
            "std": float(np.std(g_arr)) if n_gen > 0 else 0.0,
        }

        stats_imp = {
            "mean": float(np.mean(i_arr)) if n_imp > 0 else 0.0,
            "median": float(np.median(i_arr)) if n_imp > 0 else 0.0,
            "min": float(np.min(i_arr)) if n_imp > 0 else 0.0,
            "max": float(np.max(i_arr)) if n_imp > 0 else 0.0,
            "std": float(np.std(i_arr)) if n_imp > 0 else 0.0,
        }

        return VerificationMetrics(
            total_pairs=n_gen + n_imp,
            genuine_pairs=n_gen,
            impostor_pairs=n_imp,
            genuine_acceptances=ga,
            false_rejections=fr,
            genuine_rejections=gr,
            false_acceptances=fa,
            gar=gar,
            frr=frr,
            grr=grr,
            far=far,
            threshold=threshold,
            similarity_stats_genuine=stats_gen,
            similarity_stats_impostor=stats_imp,
        )
