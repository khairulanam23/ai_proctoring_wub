"""Tests for threshold sensitivity optimizer and sweeping analysis."""

from tools.benchmark.thresholds import ThresholdOptimizer


def test_face_verification_threshold_sweep():
    """Verify threshold sweep generates tradeoff table and identifies optimal F1 and EER."""
    genuine = [0.85, 0.70, 0.60, 0.50, 0.40]
    impostor = [0.10, 0.15, 0.20, 0.25, 0.30]

    sweep_res = ThresholdOptimizer.sweep_face_verification_thresholds(
        genuine_scores=genuine,
        impostor_scores=impostor,
        thresholds=[0.20, 0.30, 0.3630, 0.40, 0.50],
        default_selected=0.3630,
    )

    assert len(sweep_res.tradeoff_table) == 5
    assert sweep_res.selected_threshold == 0.3630
    assert "FACE_MATCH_THRESHOLD" in sweep_res.parameter_name
    assert sweep_res.metrics_at_selected["gar"] == 1.0
    assert sweep_res.metrics_at_selected["far"] == 0.0
