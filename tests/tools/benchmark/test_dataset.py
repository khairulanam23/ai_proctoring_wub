"""Tests for benchmark evaluation dataset builder and schema."""

from tools.benchmark.dataset import EvaluationCategory, EvaluationDatasetBuilder, GroundTruthLabel


def test_evaluation_dataset_builder_structure():
    """Verify benchmark dataset builds samples across standard test categories."""
    dataset = EvaluationDatasetBuilder.build_default_benchmark_dataset(samples_dir="data/samples")
    assert len(dataset.samples) > 0
    assert dataset.dataset_name == "Phase5_Comprehensive_Benchmark_Dataset"

    summary = dataset.summary()
    assert summary["total_samples"] == len(dataset.samples)
    assert len(summary["category_distribution"]) >= 5

    # Check presence of key categories
    norm_samples = dataset.get_by_category(EvaluationCategory.NORMAL_EXAM)
    assert len(norm_samples) > 0
    assert norm_samples[0].ground_truth_label == GroundTruthLabel.NORMAL

    absc_samples = dataset.get_by_category(EvaluationCategory.FACE_ABSENCE)
    assert len(absc_samples) > 0
    assert absc_samples[0].ground_truth_label == GroundTruthLabel.FACE_ABSENT
