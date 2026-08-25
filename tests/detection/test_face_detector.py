"""Unit tests for FaceDetector wrapper."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from proctoring.detection.face_detector import DetectionResult, FaceDetection, FaceDetector


@pytest.fixture
def detector() -> FaceDetector:
    model_path = Path("models/face_detection_yunet_2023mar.onnx")
    if not model_path.exists():
        pytest.skip(f"YuNet model not found at {model_path}")
    return FaceDetector(model_path=model_path)


def test_detector_initialization(detector: FaceDetector) -> None:
    """Test that FaceDetector initializes correctly with valid model."""
    assert detector is not None
    assert detector.detector is not None


def test_detector_missing_model() -> None:
    """Test that FaceDetector raises FileNotFoundError if model is missing."""
    with pytest.raises(FileNotFoundError):
        FaceDetector(model_path="models/non_existent_model.onnx")


def test_detector_invalid_image(detector: FaceDetector) -> None:
    """Test that invalid images raise ValueError."""
    with pytest.raises(ValueError):
        detector.detect(None)  # type: ignore

    with pytest.raises(ValueError):
        detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))

    with pytest.raises(ValueError):
        # 1-channel grayscale when 3-channel BGR expected
        detector.detect(np.zeros((100, 100), dtype=np.uint8))


def test_detector_no_face_blank_image(detector: FaceDetector) -> None:
    """Test detection on a solid blank image (should find 0 faces)."""
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
    result = detector.detect(blank_img)

    assert isinstance(result, DetectionResult)
    assert result.count == 0
    assert len(result.faces) == 0
    assert result.image_shape == (300, 300, 3)


def test_detector_with_sample_image(detector: FaceDetector) -> None:
    """Test face detection on a sample image from data/samples if available."""
    sample_dir = Path("data/samples")
    sample_files = list(sample_dir.glob("*/*.jpg")) if sample_dir.exists() else []

    if not sample_files:
        pytest.skip("No sample images available yet in data/samples")

    img = cv2.imread(str(sample_files[0]))
    assert img is not None

    result = detector.detect(img)
    assert isinstance(result, DetectionResult)
    assert result.count >= 1
    face = result.faces[0]
    assert isinstance(face, FaceDetection)
    assert len(face.bbox) == 4
    assert len(face.landmarks) == 5
    assert 0.0 <= face.confidence <= 1.0
