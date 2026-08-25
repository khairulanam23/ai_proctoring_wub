"""Tests for workflow stage 3 — frame validation, quality assessment and CLAHE."""

import tempfile

import numpy as np

from proctoring.preprocessing.frame_quality import (
    AdaptiveImagePreprocessor,
    FrameQualityGate,
)
from tools.optimization.ablation import AblationStudyEvaluator
from tools.optimization.benchmark import Phase7OptimizationBenchmark


def test_adaptive_preprocessor_clahe_enhancement():
    """A dark frame is recognised as low-light and enhanced via CLAHE."""
    preprocessor = AdaptiveImagePreprocessor()
    dark_frame = np.full((100, 100, 3), 30, dtype=np.uint8)

    quality = preprocessor.assess_quality(dark_frame)
    assert quality["valid"] is True
    assert quality["is_low_light"] is True

    enhanced, meta = preprocessor.enhance(dark_frame)
    assert meta["enhanced"] is True
    assert meta["condition"] == "clahe_applied"
    assert enhanced.shape == dark_frame.shape


def test_preprocessor_leaves_well_lit_frames_untouched():
    """CLAHE is skipped under normal lighting, so ordinary frames cost nothing extra."""
    preprocessor = AdaptiveImagePreprocessor()
    normal_frame = np.full((100, 100, 3), 128, dtype=np.uint8)

    enhanced, meta = preprocessor.enhance(normal_frame)
    assert meta["enhanced"] is False
    assert meta["condition"] == "normal_lighting"
    assert np.array_equal(enhanced, normal_frame)


def test_quality_gate_rejects_unusable_frames():
    """Structurally invalid frames are rejected with a specific, actionable reason."""
    gate = FrameQualityGate()

    assert gate.process(None).accepted is False
    assert "None" in gate.process(None).rejection_reason

    too_small = gate.process(np.zeros((10, 10, 3), dtype=np.uint8))
    assert too_small.accepted is False
    assert "below minimum" in too_small.rejection_reason

    wrong_channels = gate.process(np.zeros((200, 200), dtype=np.uint8))
    assert wrong_channels.accepted is False
    assert "3-channel" in wrong_channels.rejection_reason


def test_quality_gate_accepts_and_reports_metrics():
    """An acceptable frame passes through carrying its measured photometric state."""
    gate = FrameQualityGate()
    result = gate.process(np.full((480, 640, 3), 40, dtype=np.uint8))

    assert result.accepted is True
    assert result.frame is not None
    assert result.is_low_light is True
    assert result.was_enhanced is True
    assert result.mean_luminance is not None
    assert result.blur_variance is not None


def test_quality_gate_enhancement_can_be_disabled():
    """With enhancement off the gate still validates but returns the original pixels."""
    gate = FrameQualityGate(enable_enhancement=False)
    dark = np.full((480, 640, 3), 40, dtype=np.uint8)

    result = gate.process(dark)
    assert result.accepted is True
    assert result.was_enhanced is False
    assert np.array_equal(result.frame, dark)


def test_ablation_and_optimization_benchmark():
    """The offline ablation study and optimisation benchmark still produce valid reports."""
    ablation_report = AblationStudyEvaluator.run_standard_ablation_suite()
    assert ablation_report.total_experiments >= 4
    assert len(ablation_report.summary_findings) > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        report = Phase7OptimizationBenchmark.run_optimization_benchmark(output_dir=tmpdir)
        assert report.report_id.startswith("OPT_PHASE7_")
        assert report.overall_status in ("PASS", "NEEDS_MORE_DATA")
        assert len(report.decisions_table) >= 3
        assert len(report.quality_gates) >= 12
