#!/usr/bin/env python3
"""Executable script to run Phase 6 real-world field trials, stability stress tests, and usability benchmark."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.field_testing.runner import Phase6FieldTestRunner


def main():
    runner = Phase6FieldTestRunner(
        samples_dir=PROJECT_ROOT / "data" / "samples",
        models_dir=PROJECT_ROOT / "models",
        output_dir=PROJECT_ROOT / "data" / "results" / "phase6_field_testing",
    )
    report = runner.run_field_benchmark()
    print(f"\nPhase 6 Real-World Field Quality Gate Assessment: {report.overall_status}")
    for qg in report.quality_gates:
        print(f"  [{qg.status:<15}] {qg.capability:<34} -> {qg.evidence_metric}")


if __name__ == "__main__":
    main()
