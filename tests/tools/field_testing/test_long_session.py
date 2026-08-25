"""Tests for long session simulator, human study simulator, and field test runner."""

import tempfile
from pathlib import Path

import numpy as np

from proctoring.config import SessionConfig
from proctoring.engine import ProctoringEngine
from tools.field_testing.human_study import HumanReviewStudySimulator
from tools.field_testing.long_session import LongDurationSessionSimulator
from tools.field_testing.runner import Phase6FieldTestRunner


def test_human_review_study_simulation():
    """Verify human reviewer study generates clarity, sufficiency, and concordance metrics."""
    report = HumanReviewStudySimulator.conduct_study()
    assert report.total_reviews_conducted > 0
    assert report.participating_reviewers_count == 2
    assert report.evidence_clarity_percentage >= 90.0
    assert report.mean_sufficiency_score_1_to_5 >= 4.0
    assert report.mean_review_time_per_event_seconds < 10.0


def test_long_session_simulator_execution():
    """Verify long-duration session simulator processes frames and tracks stability."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(
            session_id="test_long_sess",
            student_name="Test Student",
            output_dir=tmpdir,
        )
        engine = ProctoringEngine(config=config)
        sample_img = np.full((120, 160, 3), 150, dtype=np.uint8)

        rep = LongDurationSessionSimulator.run_extended_session(engine, sample_img, total_frames=20)
        assert rep.total_frames_processed == 20
        assert rep.stability_verdict in ("STABLE", "DEGRADED")
        assert rep.has_memory_leak is False


def test_phase6_field_test_runner_execution():
    """Verify Phase 6 master field test runner executes across sessions and outputs master report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = Phase6FieldTestRunner(
            samples_dir="data/samples",
            models_dir="models",
            output_dir=tmpdir,
        )
        report = runner.run_field_benchmark()

        assert report.report_id.startswith("FIELD_PHASE6_")
        assert report.overall_status in ("PASS", "NEEDS_MORE_DATA")
        assert len(report.quality_gates) >= 10
        assert (Path(tmpdir) / "phase6_comprehensive_field_report.json").exists()
