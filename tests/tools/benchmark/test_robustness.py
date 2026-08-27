"""Unit tests for ImageAugmenter, evaluate_detections, ThresholdEvaluator, and visual grid generator."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from proctoring.detection.object_detector import (
    DetectedObject,
    ObjectDetectionResult,
    ObjectDetector,
)
from tools.benchmark.robustness import (
    ImageAugmenter,
    ThresholdEvaluator,
    VisualCondition,
    evaluate_detections,
    generate_visual_comparison_grid,
)


def test_image_augmenter_all_conditions() -> None:
    """Test ImageAugmenter applies all standard visual transformations safely."""
    dummy = np.full((120, 160, 3), 128, dtype=np.uint8)
    # Add a white box in middle
    dummy[40:80, 50:110] = 255

    conditions = [
        VisualCondition.ORIGINAL,
        VisualCondition.LOW_LIGHT,
        VisualCondition.HIGH_LIGHT,
        VisualCondition.LOW_CONTRAST,
        VisualCondition.GAUSSIAN_BLUR,
        VisualCondition.MOTION_BLUR,
        VisualCondition.PARTIAL_OCCLUSION,
        VisualCondition.LOW_RESOLUTION,
        VisualCondition.JPEG_COMPRESSION,
        VisualCondition.PERSPECTIVE_TILT,
    ]

    for cond in conditions:
        aug = ImageAugmenter.apply_condition(dummy, cond)
        assert aug.shape == dummy.shape
        assert aug.dtype == np.uint8
        assert aug.min() >= 0
        assert aug.max() <= 255

    # Invalid condition error
    with pytest.raises(ValueError):
        ImageAugmenter.apply_condition(dummy, "invalid_condition_xyz")

    # Invalid image error
    with pytest.raises(ValueError):
        ImageAugmenter.apply_condition(None, VisualCondition.LOW_LIGHT)  # type: ignore


def test_confusion_metrics_evaluation() -> None:
    """Test evaluate_detections precision, recall, and F1 calculations."""
    # 1. Perfect Match
    m1 = evaluate_detections(["person", "cell phone"], ["person", "cell phone"])
    assert m1.true_positives == 2
    assert m1.false_positives == 0
    assert m1.false_negatives == 0
    assert m1.precision == 1.0
    assert m1.recall == 1.0
    assert m1.f1 == 1.0

    # 2. Partial Match (1 TP, 1 FP, 1 FN)
    m2 = evaluate_detections(["person", "laptop"], ["person", "cell phone"])
    assert m2.true_positives == 1
    assert m2.false_positives == 1
    assert m2.false_negatives == 1
    assert m2.precision == 0.5
    assert m2.recall == 0.5
    assert m2.f1 == 0.5

    # 3. Empty Predictions vs Expected
    m3 = evaluate_detections([], ["person"])
    assert m3.true_positives == 0
    assert m3.false_positives == 0
    assert m3.false_negatives == 1
    assert m3.precision == 0.0
    assert m3.recall == 0.0

    # 4. Empty Both
    m4 = evaluate_detections([], [])
    assert m4.true_positives == 0
    assert m4.precision == 1.0
    assert m4.recall == 1.0


def test_threshold_evaluator_sweep_with_mock_detector() -> None:
    """Test ThresholdEvaluator over multiple confidence thresholds with mock detector."""
    mock_detector = MagicMock(spec=ObjectDetector)

    def mock_detect(image, confidence_threshold=0.25):
        # Return objects with confidence 0.45 and 0.75
        objs = [
            DetectedObject(0, "person", 0.75, (10, 10, 50, 50)),
            DetectedObject(67, "cell phone", 0.45, (60, 60, 90, 90)),
        ]
        filtered = [o for o in objs if o.confidence >= confidence_threshold]
        return ObjectDetectionResult(
            objects=filtered,
            count=len(filtered),
            image_shape=(100, 100, 3),
            image_width=100,
            image_height=100,
            inference_time_ms=5.0,
            model_name="mock_model",
            device="cpu",
        )

    mock_detector.detect.side_effect = mock_detect

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    images_gt = [(dummy_img, ["person", "cell phone"])]

    thresholds = [0.25, 0.50, 0.80]
    results = ThresholdEvaluator.evaluate_threshold_sweep(mock_detector, images_gt, thresholds)

    assert len(results) == 3

    # At 0.25: both pass (person 0.75, cell phone 0.45)
    assert results[0].threshold == 0.25
    assert results[0].total_detections == 2
    assert results[0].metrics is not None
    assert results[0].metrics.precision == 1.0

    # At 0.50: only person passes (0.75)
    assert results[1].threshold == 0.50
    assert results[1].total_detections == 1
    assert results[1].metrics is not None
    assert results[1].metrics.true_positives == 1
    assert results[1].metrics.false_negatives == 1

    # At 0.80: neither passes
    assert results[2].threshold == 0.80
    assert results[2].total_detections == 0


def test_generate_visual_comparison_grid(tmp_path: Path) -> None:
    """Test visual comparison grid generation."""
    img1 = np.full((100, 100, 3), 50, dtype=np.uint8)
    img2 = np.full((100, 100, 3), 150, dtype=np.uint8)

    cond_images = {
        "original": img1,
        "low_light": img2,
    }

    res1 = ObjectDetectionResult(
        objects=[DetectedObject(0, "person", 0.90, (10, 10, 50, 50))],
        count=1,
        image_shape=(100, 100, 3),
        image_width=100,
        image_height=100,
        inference_time_ms=5.0,
        model_name="mock_model",
        device="cpu",
    )
    res2 = ObjectDetectionResult(
        objects=[DetectedObject(0, "person", 0.85, (10, 10, 50, 50))],
        count=1,
        image_shape=(100, 100, 3),
        image_width=100,
        image_height=100,
        inference_time_ms=5.0,
        model_name="mock_model",
        device="cpu",
    )
    cond_results = {
        "original": res1,
        "low_light": res2,
    }

    out_grid = tmp_path / "test_grid.jpg"
    res_path = generate_visual_comparison_grid(cond_images, cond_results, out_grid, cols=2)

    assert res_path.exists()
    assert res_path.stat().st_size > 0
