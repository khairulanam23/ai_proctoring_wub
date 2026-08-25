"""Tests for Phase 11 previous-phase audit, pipeline dataflow validation, and regression suite."""

import tempfile
from pathlib import Path

from tools.audit.audit_table import PhaseAuditStatus, PreviousPhasesAuditMatrix
from tools.audit.pipeline_validator import EndToEndPipelineValidator
from tools.audit.regression_suite import Phase11RegressionTestSuite
from tools.audit.runner import Phase11AuditRunner


def test_previous_phases_audit_matrix():
    """Verify audit matrix contains all 10 previous phases with valid status."""
    matrix = PreviousPhasesAuditMatrix.get_audit_matrix()
    assert len(matrix) == 10
    assert all(p.status == PhaseAuditStatus.PASS for p in matrix)
    assert all(p.data_compatibility_verified for p in matrix)


def test_end_to_end_pipeline_validator():
    """Verify 8-stage pipeline data flow contracts are valid."""
    res = EndToEndPipelineValidator.validate_full_pipeline_flow(
        samples_dir="data/samples",
        models_dir="models",
    )
    assert res.all_stages_valid is True
    assert len(res.stages) == 8
    assert res.zero_risk_score_enforced is True
    assert res.verdict == "PASS"


def test_phase11_regression_suite():
    """Verify all 13 regression tests pass cleanly."""
    res = Phase11RegressionTestSuite.run_13_regression_tests(
        samples_dir="data/samples",
        models_dir="models",
    )
    assert res.total_tests == 13
    assert res.passed_count == 13
    assert res.failed_count == 0
    assert res.overall_status == "PASS"


def test_phase11_audit_runner():
    """Verify master Phase 11 audit runner executes and produces valid report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = Phase11AuditRunner(
            samples_dir="data/samples",
            models_dir="models",
            output_dir=tmpdir,
        )
        report = runner.run_full_audit()
        assert report.report_id.startswith("AUDIT_PHASE11_")
        assert report.overall_audit_verdict == "PASS"
        assert report.phase_audit_summary["pass_count"] == 10
        assert (Path(tmpdir) / "phase11_audit_report.json").exists()
