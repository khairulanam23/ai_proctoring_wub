import numpy as np

from tools.benchmark.evaluator import BenchmarkEvaluator


def test_classification_metrics_calculation():
    """Verify precision, recall, F1, accuracy, and confidence metrics."""
    metrics = BenchmarkEvaluator.calculate_classification_metrics(
        component_name="Test Detector",
        tp=18,
        fp=2,
        tn=15,
        fn=0,
        confidences_tp=[0.9, 0.85, 0.95],
        confidences_fp=[0.4, 0.45],
    )

    assert metrics.component_name == "Test Detector"
    assert metrics.total_samples == 35
    assert metrics.precision == 18 / 20  # 0.90
    assert metrics.recall == 1.0
    assert metrics.accuracy == 33 / 35
    assert np.isclose(metrics.confidence_distribution["tp_mean_conf"], 0.90)
    assert np.isclose(metrics.confidence_distribution["fp_mean_conf"], 0.425)


def test_verification_pairs_evaluation():
    """Verify GAR, FAR, FRR, GRR calculation on genuine vs impostor score distributions."""
    genuine_scores = [0.85, 0.72, 0.65, 0.45, 0.38]  # all >= 0.3630 -> GAR = 1.0
    impostor_scores = [0.12, 0.18, 0.22, 0.15, 0.08]  # all < 0.3630 -> FAR = 0.0

    ver_m = BenchmarkEvaluator.evaluate_verification_pairs(
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        threshold=0.3630,
    )

    assert ver_m.total_pairs == 10
    assert ver_m.genuine_pairs == 5
    assert ver_m.impostor_pairs == 5
    assert ver_m.gar == 1.0
    assert ver_m.far == 0.0
    assert ver_m.frr == 0.0
    assert ver_m.grr == 1.0
