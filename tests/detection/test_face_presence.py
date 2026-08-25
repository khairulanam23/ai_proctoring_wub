"""Unit tests for FacePresenceAnalyzer."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from proctoring.detection.face_detector import FaceDetector
from proctoring.detection.face_presence import (
    FacePresenceAnalyzer,
    FacePresenceResult,
    FacePresenceStatus,
)


@pytest.fixture
def presence_analyzer() -> FacePresenceAnalyzer:
    model_path = Path("models/face_detection_yunet_2023mar.onnx")
    if not model_path.exists():
        pytest.skip(f"YuNet model missing at {model_path}")
    detector = FaceDetector(model_path=model_path)
    return FacePresenceAnalyzer(detector=detector)


def test_presence_analyzer_initialization(presence_analyzer: FacePresenceAnalyzer) -> None:
    """Test analyzer initialization."""
    assert presence_analyzer is not None
    assert presence_analyzer.detector is not None


def test_presence_invalid_image(presence_analyzer: FacePresenceAnalyzer) -> None:
    """Test that invalid images raise ValueError."""
    with pytest.raises(ValueError):
        presence_analyzer.analyze(None)  # type: ignore

    with pytest.raises(ValueError):
        presence_analyzer.analyze(np.zeros((0, 0, 3), dtype=np.uint8))

    with pytest.raises(ValueError):
        presence_analyzer.analyze(np.zeros((100, 100), dtype=np.uint8))


def test_presence_no_face_blank_image(presence_analyzer: FacePresenceAnalyzer) -> None:
    """Test NO_FACE status on empty/solid canvas."""
    blank_img = np.full((360, 480, 3), 200, dtype=np.uint8)
    res = presence_analyzer.analyze(blank_img)

    assert isinstance(res, FacePresenceResult)
    assert res.status == FacePresenceStatus.NO_FACE
    assert res.face_count == 0
    assert len(res.faces) == 0
    assert res.inference_time_ms > 0

    # Test dictionary conversion
    d = res.to_dict()
    assert d["status"] == "NO_FACE"
    assert d["face_count"] == 0
    assert len(d["faces"]) == 0


def test_presence_single_face(presence_analyzer: FacePresenceAnalyzer) -> None:
    """Test SINGLE_FACE status on single face image."""
    single_img_path = Path("data/samples/synthetic/single_face.jpg")
    if not single_img_path.exists():
        # Fallback to LFW sample
        lfw_imgs = list(Path("data/samples").glob("*/*.jpg"))
        if not lfw_imgs:
            pytest.skip("No sample images available")
        single_img_path = lfw_imgs[0]

    img = cv2.imread(str(single_img_path))
    assert img is not None

    res = presence_analyzer.analyze(img, frame_index=1, timestamp_sec=0.033)

    assert isinstance(res, FacePresenceResult)
    assert res.status == FacePresenceStatus.SINGLE_FACE
    assert res.face_count == 1
    assert len(res.faces) == 1
    assert res.frame_index == 1
    assert res.timestamp_sec == 0.033

    face = res.faces[0]
    assert len(face.bbox) == 4
    assert len(face.landmarks) == 5
    assert face.confidence > 0.6

    d = res.to_dict()
    assert d["status"] == "SINGLE_FACE"
    assert d["face_count"] == 1
    assert len(d["faces"]) == 1


def test_presence_multiple_faces(presence_analyzer: FacePresenceAnalyzer) -> None:
    """Test MULTIPLE_FACES status on multi-face canvas."""
    multi_img_path = Path("data/samples/synthetic/multi_face_2.jpg")
    if not multi_img_path.exists():
        pytest.skip("Multi-face synthetic image missing")

    img = cv2.imread(str(multi_img_path))
    assert img is not None

    res = presence_analyzer.analyze(img)

    assert isinstance(res, FacePresenceResult)
    assert res.status == FacePresenceStatus.MULTIPLE_FACES
    assert res.face_count >= 2
    assert len(res.faces) >= 2

    d = res.to_dict()
    assert d["status"] == "MULTIPLE_FACES"
    assert d["face_count"] >= 2
