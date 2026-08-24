"""Unit tests for ObjectDetector, DetectedObject, and ObjectDetectionResult."""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.object_detection.detector import (
    DetectedObject,
    ObjectDetectionResult,
    ObjectDetector,
)


def test_detected_object_dataclass_properties() -> None:
    """Test DetectedObject properties, coordinate accessors, and serialization."""
    obj = DetectedObject(
        class_id=0,
        class_name="person",
        confidence=0.9254,
        bbox=(50, 100, 200, 350),
    )

    assert obj.class_id == 0
    assert obj.class_name == "person"
    assert obj.confidence == 0.9254
    assert obj.bbox == (50, 100, 200, 350)
    assert obj.x1 == 50
    assert obj.y1 == 100
    assert obj.x2 == 200
    assert obj.y2 == 350
    assert obj.width == 150
    assert obj.height == 250

    data = obj.to_dict()
    assert data["class_id"] == 0
    assert data["class_name"] == "person"
    assert data["confidence"] == 0.9254
    assert data["bbox"] == [50, 100, 200, 350]
    assert data["width"] == 150
    assert data["height"] == 250


def test_object_detection_result_serialization() -> None:
    """Test ObjectDetectionResult structure and to_dict serialization."""
    objs = [
        DetectedObject(class_id=0, class_name="person", confidence=0.91, bbox=(10, 20, 100, 200)),
        DetectedObject(class_id=67, class_name="cell phone", confidence=0.84, bbox=(50, 60, 90, 120)),
    ]

    res = ObjectDetectionResult(
        objects=objs,
        count=2,
        image_shape=(480, 640, 3),
        image_width=640,
        image_height=480,
        inference_time_ms=12.45,
        model_name="yolo11n.pt",
        device="cpu",
    )

    assert res.count == 2
    assert res.image_width == 640
    assert res.image_height == 480
    assert res.inference_time_ms == 12.45
    assert res.device == "cpu"
    assert res.model_name == "yolo11n.pt"

    data = res.to_dict()
    assert data["count"] == 2
    assert data["image_width"] == 640
    assert len(data["objects"]) == 2
    assert data["objects"][1]["class_name"] == "cell phone"


def test_detector_initialization_defaults() -> None:
    """Test detector initialization without auto-loading weights."""
    detector = ObjectDetector(
        model_name="yolo11n.pt",
        confidence_threshold=0.35,
        device="cpu",
        auto_load=False,
    )

    assert detector.model_name == "yolo11n.pt"
    assert detector.confidence_threshold == 0.35
    assert detector.device == "cpu"
    assert detector.model is None


def test_detector_device_resolution() -> None:
    """Test explicit device override and fallback resolution."""
    det_cpu = ObjectDetector(device="cpu", auto_load=False)
    assert det_cpu.device == "cpu"

    det_cuda = ObjectDetector(device="cuda", auto_load=False)
    assert det_cuda.device == "cuda"


def test_detector_invalid_input_images() -> None:
    """Test clean validation and error raising for invalid input images."""
    detector = ObjectDetector(device="cpu", auto_load=False)
    detector.model = MagicMock()

    # None input
    with pytest.raises(ValueError, match="Invalid input image"):
        detector.detect(None)  # type: ignore

    # Empty array
    with pytest.raises(ValueError, match="Invalid input image"):
        detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))

    # 2D Grayscale instead of 3-channel
    with pytest.raises(ValueError, match="Expected 3-channel BGR"):
        detector.detect(np.zeros((100, 100), dtype=np.uint8))


def test_detector_mocked_inference_and_parsing() -> None:
    """Test detection parsing, bounding box clipping, and threshold handling with mock model."""
    detector = ObjectDetector(
        confidence_threshold=0.30,
        device="cpu",
        auto_load=False,
    )

    # Mock Ultralytics Results structure
    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu().numpy.return_value = np.array([
        [10.0, 20.0, 150.0, 300.0],
        [40.0, 50.0, 80.0, 120.0],
    ])
    mock_boxes.conf.cpu().numpy.return_value = np.array([0.94, 0.78])
    mock_boxes.cls.cpu().numpy.return_value = np.array([0, 67])

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    mock_result.names = {0: "person", 67: "cell phone"}

    mock_model = MagicMock()
    mock_model.return_value = [mock_result]
    detector.model = mock_model

    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(dummy_image)

    assert res.count == 2
    assert res.objects[0].class_name == "person"
    assert res.objects[0].confidence == 0.94
    assert res.objects[0].bbox == (10, 20, 150, 300)

    assert res.objects[1].class_name == "cell phone"
    assert res.objects[1].confidence == 0.78
    assert res.objects[1].bbox == (40, 50, 80, 120)

    # Verify model was called with correct parameters
    mock_model.assert_called_once()
    call_args, call_kwargs = mock_model.call_args
    assert call_kwargs["conf"] == 0.30
    assert call_kwargs["device"] == "cpu"


def test_detector_visualization() -> None:
    """Test visualization drawing on image."""
    detector = ObjectDetector(device="cpu", auto_load=False)
    dummy_image = np.zeros((300, 300, 3), dtype=np.uint8)

    objs = [
        DetectedObject(class_id=0, class_name="person", confidence=0.95, bbox=(20, 30, 150, 200)),
        DetectedObject(class_id=67, class_name="cell phone", confidence=0.82, bbox=(160, 40, 250, 120)),
    ]
    res = ObjectDetectionResult(
        objects=objs,
        count=2,
        image_shape=(300, 300, 3),
        image_width=300,
        image_height=300,
        inference_time_ms=5.0,
        model_name="yolo11n.pt",
        device="cpu",
    )

    vis = detector.visualize(dummy_image, res)
    assert vis is not None
    assert vis.shape == (300, 300, 3)
    assert vis.dtype == np.uint8
    # Visual image should have been modified (drawn boxes)
    assert not np.array_equal(vis, dummy_image)
