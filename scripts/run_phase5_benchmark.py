#!/usr/bin/env python3
"""Executable script to run full Phase 5 benchmark suite and generate master evaluation report."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.benchmark.runner import Phase5BenchmarkRunner


def main():
    runner = Phase5BenchmarkRunner(
        samples_dir=PROJECT_ROOT / "data" / "samples",
        models_dir=PROJECT_ROOT / "models",
        output_dir=PROJECT_ROOT / "data" / "results" / "phase5_benchmark",
    )
    report = runner.run_full_benchmark()
    print(f"\nOverall System Quality Gate Assessment: {report.overall_system_status}")
    for qg in report.quality_gates:
        print(f"  [{qg.status:<15}] {qg.capability:<36} -> {qg.evidence_metric}")


if __name__ == "__main__":
    main()
