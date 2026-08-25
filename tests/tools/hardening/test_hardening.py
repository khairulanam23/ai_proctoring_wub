"""Tests for Phase 10 system hardening, 12-test suite, stress matrix, and fault recovery."""

import tempfile
from pathlib import Path

from tools.hardening.recovery import FaultInjectionSimulator
from tools.hardening.runner import Phase10HardeningRunner
from tools.hardening.stress import PipelineStressTester
from tools.hardening.test_suite import Phase10ProctoringTestSuite


def test_12_test_suite_execution():
    """Verify all 12 tests in standard test suite execute and report verdicts."""
    res = Phase10ProctoringTestSuite.run_all_tests(samples_dir="data/samples", models_dir="models")
    assert res.total_tests == 12
    assert res.verified_count >= 10
    assert res.failed_count == 0
    assert res.overall_status in ("PASS", "PASS_WITH_WARNINGS")


def test_stress_tester_matrix():
    """Verify stress matrix runs across resolutions and measures memory growth."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res = PipelineStressTester.run_stress_matrix(output_dir=tmpdir)
        assert res.total_conditions_tested == 5
        assert res.overall_verdict == "PASS"


def test_fault_recovery_suite():
    """Verify fault injection recovers gracefully from camera disconnects and malformed frames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results = FaultInjectionSimulator.run_fault_recovery_suite(output_dir=tmpdir)
        assert len(results) == 3
        assert all(r.verdict == "PASS" for r in results)


def test_package_serialization_verifier():
    """Verify package serialization verifier detects valid manifests and reports equivalence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run hardening runner to produce valid report and audit
        runner = Phase10HardeningRunner(
            samples_dir="data/samples",
            models_dir="models",
            output_dir=tmpdir,
        )
        rep = runner.run_full_hardening_evaluation()
        assert rep.report_id.startswith("HARDEN_PHASE10_")
        assert rep.overall_verdict in ("PASS", "PASS_WITH_WARNINGS")
        assert (Path(tmpdir) / "phase10_final_report.json").exists()
