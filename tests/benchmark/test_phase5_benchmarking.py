"""Targeted tests for Phase 5 model improvement evaluation and empirical benchmarking."""

from pathlib import Path

from tools.benchmark.phase5_model_evaluator import Phase5ModelEvaluator


def test_phase5_benchmarks_execution(tmp_path: Path):
    """Verify that all Phase 5 benchmarks execute, record actual measurements, and persist JSON audit."""
    evaluator = Phase5ModelEvaluator(iterations=10)

    # 1. MultiSubjectTracker benchmark
    bm_tracker = evaluator.benchmark_multi_subject_tracker()
    assert bm_tracker.operations_measured == 10
    assert bm_tracker.mean_latency_ms > 0.0
    assert bm_tracker.empirical_behavior["confirmed_tracks"] == 3
    assert bm_tracker.real_world_dataset_status == "UNVERIFIED"

    # 2. PhoneHandDisambiguator benchmark
    bm_phone = evaluator.benchmark_phone_disambiguator()
    assert bm_phone.operations_measured == 10
    assert bm_phone.mean_latency_ms > 0.0
    assert bm_phone.empirical_behavior["hand_fp_dismissal_rate"] == 1.0

    # 3. PaperDetector benchmark
    bm_paper = evaluator.benchmark_paper_detector()
    assert bm_paper.operations_measured == 10
    assert bm_paper.mean_latency_ms > 0.0
    assert bm_paper.empirical_behavior["detection_success_rate"] == 1.0

    # 4. WearableDetector benchmark
    bm_wearable = evaluator.benchmark_wearable_detector()
    assert bm_wearable.operations_measured == 10
    assert bm_wearable.empirical_behavior["earring_suppression_rate"] == 1.0

    # 5. Full execution and JSON persistence
    out_file = tmp_path / "phase5_test_results.json"
    results = evaluator.run_all(output_path=out_file)
    assert out_file.exists()
    assert "components" in results
    assert len(results["components"]) == 4
    assert "MultiSubjectTracker" in results["components"]
    assert "PhoneHandDisambiguator" in results["components"]
