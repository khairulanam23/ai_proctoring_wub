"""Unit tests for FacePreprocessor, pose estimation, and landmark alignment module."""

from pathlib import Path
import numpy as np
import pytest
import cv2

from src.face.detector import FaceDetector
from src.face.preprocessing import FacePreprocessor, PreprocessingStatus

SAMPLE_DIR = Path("data/samples")
SYNTHETIC_DIR = Path("data/samples/synthetic")


@pytest.fixture
def detector() -> FaceDetector:
    return FaceDetector()


@pytest.fixture
def preprocessor(detector: FaceDetector) -> FacePreprocessor:
    return FacePreprocessor(detector=detector)


def test_preprocessor_initialization(preprocessor: FacePreprocessor) -> None:
    """Test default initialization and configuration."""
    assert preprocessor.output_size == (112, 112)
    assert preprocessor.min_face_size == 40
    assert preprocessor.min_blur_score == 15.0
    assert preprocessor.illumination_mode == "none"
    assert preprocessor.canonical_template.shape == (5, 2)


def test_preprocessor_invalid_image(preprocessor: FacePreprocessor) -> None:
    """Test handling of invalid, None, and empty images."""
    res_none = preprocessor.preprocess(None)  # type: ignore
    assert res_none.success is False
    assert res_none.status == PreprocessingStatus.INVALID_IMAGE
    assert res_none.aligned_face is None

    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    res_empty = preprocessor.preprocess(empty)
    assert res_empty.success is False
    assert res_empty.status == PreprocessingStatus.INVALID_IMAGE

    gray = np.zeros((100, 100), dtype=np.uint8)
    res_gray = preprocessor.preprocess(gray)
    assert res_gray.success is False
    assert res_gray.status == PreprocessingStatus.INVALID_IMAGE


def test_preprocessor_no_face_blank_image(preprocessor: FacePreprocessor) -> None:
    """Test preprocessing on a blank image with no faces."""
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    res = preprocessor.preprocess(blank)
    assert res.success is False
    assert res.status == PreprocessingStatus.NO_FACE
    assert res.face_count == 0
    assert res.aligned_face is None
    assert res.quality is None


def test_preprocessor_multiple_faces(preprocessor: FacePreprocessor) -> None:
    """Test preprocessing on an image with multiple faces when single face is required."""
    multi_path = SYNTHETIC_DIR / "multi_face_2.jpg"
    if not multi_path.exists():
        pytest.skip("Synthetic multi-face sample not found")

    image = cv2.imread(str(multi_path))
    res = preprocessor.preprocess(image, require_single_face=True)
    assert res.success is False
    assert res.status == PreprocessingStatus.MULTIPLE_FACES
    assert res.face_count >= 2
    assert res.aligned_face is None


def test_preprocessor_face_too_small(detector: FaceDetector) -> None:
    """Test rejection when detected face dimensions are below min_face_size."""
    strict_preprocessor = FacePreprocessor(detector=detector, min_face_size=500)
    sample_path = SAMPLE_DIR / "Colin_Powell" / "Colin_Powell_0001.jpg"
    if not sample_path.exists():
        pytest.skip("Sample image not found")

    image = cv2.imread(str(sample_path))
    res = strict_preprocessor.preprocess(image)
    assert res.success is False
    assert res.status in (PreprocessingStatus.LOW_RESOLUTION, PreprocessingStatus.FACE_TOO_SMALL)
    assert res.quality is not None
    assert res.quality.is_size_valid is False


def test_preprocessor_blur_detection(preprocessor: FacePreprocessor) -> None:
    """Test sharpness computation and blur warning flag."""
    sample_path = SAMPLE_DIR / "Colin_Powell" / "Colin_Powell_0001.jpg"
    if not sample_path.exists():
        pytest.skip("Sample image not found")

    image = cv2.imread(str(sample_path))

    # Sharp image
    res_sharp = preprocessor.preprocess(image)
    assert res_sharp.success is True
    assert res_sharp.quality is not None
    assert res_sharp.quality.blur_score > preprocessor.min_blur_score
    assert res_sharp.quality.is_sharp is True

    # Heavily blurred image
    blurred = cv2.GaussianBlur(image, (31, 31), 10.0)
    res_blur = preprocessor.preprocess(blurred)
    assert res_blur.success is True
    assert res_blur.status == PreprocessingStatus.BLUR_WARNING
    assert res_blur.quality is not None
    assert res_blur.quality.is_sharp is False


def test_preprocessor_pose_estimation(preprocessor: FacePreprocessor) -> None:
    """Test geometric pose and symmetry estimation from 5 landmarks."""
    # Frontal landmarks
    frontal_landmarks = [
        (38.0, 50.0),  # Right eye
        (74.0, 50.0),  # Left eye
        (56.0, 70.0),  # Nose tip (centered at 50%)
        (41.0, 90.0),  # Right mouth
        (71.0, 90.0),  # Left mouth
    ]
    pose_frontal = preprocessor.estimate_pose(frontal_landmarks)
    assert pose_frontal.is_frontal is True
    assert pose_frontal.is_extreme_pose is False
    assert 0.40 <= pose_frontal.yaw_ratio <= 0.60
    assert abs(pose_frontal.roll_angle_deg) < 5.0

    # Extreme yaw profile landmarks (nose strongly shifted to the right)
    profile_landmarks = [
        (30.0, 50.0),
        (80.0, 50.0),
        (75.0, 70.0),  # Nose heavily displaced
        (40.0, 90.0),
        (75.0, 90.0),
    ]
    pose_profile = preprocessor.estimate_pose(profile_landmarks)
    assert pose_profile.is_extreme_pose is True
    assert pose_profile.is_frontal is False


def test_preprocessor_landmark_alignment_and_output_dimensions(
    preprocessor: FacePreprocessor,
) -> None:
    """Test 5-point landmark extraction, transformation matrix, and aligned output dimensions."""
    sample_path = SAMPLE_DIR / "Colin_Powell" / "Colin_Powell_0001.jpg"
    if not sample_path.exists():
        pytest.skip("Sample image not found")

    image = cv2.imread(str(sample_path))
    res = preprocessor.preprocess(image)

    assert res.success is True
    assert res.aligned_face is not None
    assert res.aligned_face.shape == (112, 112, 3)
    assert res.aligned_face.dtype == np.uint8
    assert res.normalized_face is not None
    assert res.normalized_face.shape == (112, 112, 3)

    assert res.raw_face is not None
    assert len(res.raw_face.landmarks) == 5
    assert res.transform_matrix is not None
    assert res.transform_matrix.shape == (2, 3)

    # Check timing fields
    assert "detect_ms" in res.timing_ms
    assert "align_ms" in res.timing_ms
    assert "norm_ms" in res.timing_ms
    assert "total_ms" in res.timing_ms


def test_preprocessor_illumination_modes(detector: FaceDetector) -> None:
    """Test illumination normalization modes: none, mild_contrast, and clahe."""
    sample_path = SAMPLE_DIR / "Colin_Powell" / "Colin_Powell_0001.jpg"
    if not sample_path.exists():
        pytest.skip("Sample image not found")

    image = cv2.imread(str(sample_path))

    for mode in ["none", "mild_contrast", "clahe"]:
        p = FacePreprocessor(detector=detector, illumination_mode=mode)
        res = p.preprocess(image)
        assert res.success is True
        assert res.normalized_face is not None
        assert res.normalized_face.shape == (112, 112, 3)


def test_preprocessor_serialization(preprocessor: FacePreprocessor) -> None:
    """Test dictionary serialization of PreprocessingResult."""
    sample_path = SAMPLE_DIR / "Colin_Powell" / "Colin_Powell_0001.jpg"
    if not sample_path.exists():
        pytest.skip("Sample image not found")

    image = cv2.imread(str(sample_path))
    res = preprocessor.preprocess(image)
    data = res.to_dict()

    assert data["success"] is True
    assert "status" in data
    assert "face_count" in data
    assert "quality" in data
    assert data["aligned_shape"] == [112, 112, 3]
    assert "timing_ms" in data
