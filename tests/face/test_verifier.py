"""Unit tests for FaceVerifier wrapper."""

from pathlib import Path
import numpy as np
import pytest
import cv2

from src.face.detector import FaceDetector
from src.face.verifier import FaceVerifier, VerificationResult


@pytest.fixture
def verifier() -> FaceVerifier:
    det_model = Path("models/face_detection_yunet_2023mar.onnx")
    rec_model = Path("models/face_recognition_sface_2021dec.onnx")

    if not det_model.exists() or not rec_model.exists():
        pytest.skip("YuNet or SFace model file missing")

    detector = FaceDetector(model_path=det_model)
    return FaceVerifier(detector=detector, recognizer_model_path=rec_model)


def test_verifier_initialization(verifier: FaceVerifier) -> None:
    """Test that FaceVerifier initializes correctly."""
    assert verifier is not None
    assert verifier.recognizer is not None
    assert verifier.detector is not None


def test_verifier_missing_model() -> None:
    """Test that FaceVerifier raises FileNotFoundError for missing model."""
    with pytest.raises(FileNotFoundError):
        FaceVerifier(recognizer_model_path="models/non_existent.onnx")


def test_verifier_no_face_handling(verifier: FaceVerifier) -> None:
    """Test verification rejection when images have no faces."""
    blank_a = np.zeros((200, 200, 3), dtype=np.uint8)
    blank_b = np.zeros((200, 200, 3), dtype=np.uint8)

    result = verifier.verify(blank_a, blank_b)

    assert isinstance(result, VerificationResult)
    assert not result.success
    assert not result.same_person
    assert result.similarity is None
    assert result.ref_face_count == 0
    assert result.test_face_count == 0
    assert "No face detected" in result.message


def test_verifier_metric_options(verifier: FaceVerifier) -> None:
    """Test metric validation."""
    with pytest.raises(ValueError):
        verifier.compute_similarity(np.zeros(128), np.zeros(128), metric="unsupported")


def test_verifier_with_same_image(verifier: FaceVerifier) -> None:
    """Test that verifying an image against itself yields maximum similarity (> 0.99 for cosine)."""
    sample_dir = Path("data/samples")
    sample_files = list(sample_dir.glob("*/*.jpg")) if sample_dir.exists() else []

    if not sample_files:
        pytest.skip("No sample images available yet in data/samples")

    img = cv2.imread(str(sample_files[0]))
    assert img is not None

    result = verifier.verify(img, img, metric="cosine")
    assert result.success
    assert result.same_person
    assert result.similarity is not None
    assert result.similarity > 0.95  # Identical image similarity
