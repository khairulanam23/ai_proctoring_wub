"""Tests for Phase 5 benchmark runner."""

import tempfile
from pathlib import Path

from tools.benchmark.runner import Phase5BenchmarkRunner


def test_phase5_benchmark_runner_execution():
    """Verify Phase5BenchmarkRunner executes across all test components and produces a valid master report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = Phase5BenchmarkRunner(
            samples_dir="data/samples",
            models_dir="models",
            output_dir=tmpdir,
        )
        report = runner.run_full_benchmark()

        assert report.benchmark_id.startswith("BM_PHASE5_")
        assert report.overall_system_status in ("PASS", "NEEDS_MORE_DATA")
        assert len(report.quality_gates) >= 10
        assert (Path(tmpdir) / "phase5_comprehensive_benchmark_report.json").exists()
