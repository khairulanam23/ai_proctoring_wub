"""Unit tests for FaceVerifier, embedding normalization, and multi-reference enrollment."""

from pathlib import Path
import numpy as np
import pytest
import cv2

from src.face.detector import FaceDetector
from src.face.preprocessing import FacePreprocessor
from src.face.verifier import FaceVerifier, VerificationResult, MultiReferenceVerificationResult

SAMPLE_DIR = Path("data/samples")


@pytest.fixture
def detector() -> FaceDetector:
    return FaceDetector()


@pytest.fixture
def preprocessor(detector: FaceDetector) -> FacePreprocessor:
    return FacePreprocessor(detector=detector)


@pytest.fixture
def verifier(detector: FaceDetector, preprocessor: FacePreprocessor) -> FaceVerifier:
    return FaceVerifier(detector=detector, preprocessor=preprocessor)


def test_verifier_initialization(verifier: FaceVerifier) -> None:
    """Test verifier initialization and default values."""
    assert verifier.recognizer is not None
    assert verifier.default_metric == "cosine"
    assert verifier.default_threshold == 0.3630


def test_verifier_missing_model() -> None:
    """Test error handling when SFace model path is invalid."""
    with pytest.raises(FileNotFoundError):
        FaceVerifier(recognizer_model_path="non_existent_sface.onnx")


def test_verifier_metric_options() -> None:
    """Test metric validation."""
    with pytest.raises(ValueError):
        FaceVerifier(default_metric="invalid_metric")


def test_verifier_embedding_l2_normalization(verifier: FaceVerifier) -> None:
    """Test that extracted embeddings are properly L2-normalized."""
    dummy_crop = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    feat = verifier.extract_feature(dummy_crop, normalize_l2=True)
    norm = np.linalg.norm(feat)
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_verifier_cosine_similarity_math(verifier: FaceVerifier) -> None:
    """Test mathematical correctness of cosine similarity computation."""
    # Orthogonal vectors -> cosine = 0
    v1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    v2 = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    assert np.isclose(verifier.compute_similarity(v1, v2, metric="cosine"), 0.0, atol=1e-5)

    # Identical vectors -> cosine = 1
    assert np.isclose(verifier.compute_similarity(v1, v1, metric="cosine"), 1.0, atol=1e-5)

    # Opposite vectors -> cosine = -1
    v3 = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)
    assert np.isclose(verifier.compute_similarity(v1, v3, metric="cosine"), -1.0, atol=1e-5)


def test_verifier_no_face_handling(verifier: FaceVerifier) -> None:
    """Test verification rejection when blank images are provided."""
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    res = verifier.verify(blank, blank)
    assert res.success is False
    assert res.same_person is False
    assert res.similarity is None


def test_verifier_with_same_image(verifier: FaceVerifier) -> None:
    """Test verification of an image against itself."""
    sample_path = SAMPLE_DIR / "Colin_Powell" / "Colin_Powell_0001.jpg"
    if not sample_path.exists():
        pytest.skip("Sample image not found")

    image = cv2.imread(str(sample_path))
    res = verifier.verify(image, image, metric="cosine")

    assert res.success is True
    assert res.same_person is True
    assert res.similarity is not None
    assert res.similarity >= 0.95


def test_verifier_positive_and_negative_pairs(verifier: FaceVerifier) -> None:
    """Test verification decisions on positive and negative image pairs."""
    cp1_path = SAMPLE_DIR / "Colin_Powell" / "Colin_Powell_0001.jpg"
    cp2_path = SAMPLE_DIR / "Colin_Powell" / "Colin_Powell_0002.jpg"
    gwb_path = SAMPLE_DIR / "George_W_Bush" / "George_W_Bush_0001.jpg"

    if not (cp1_path.exists() and cp2_path.exists() and gwb_path.exists()):
        pytest.skip("Sample benchmark images not found")

    img_cp1 = cv2.imread(str(cp1_path))
    img_cp2 = cv2.imread(str(cp2_path))
    img_gwb = cv2.imread(str(gwb_path))

    # Positive pair (Same person)
    pos_res = verifier.verify(img_cp1, img_cp2, metric="cosine")
    assert pos_res.success is True
    assert pos_res.same_person is True
    assert pos_res.similarity > verifier.default_threshold

    # Negative pair (Different person)
    neg_res = verifier.verify(img_cp1, img_gwb, metric="cosine")
    assert neg_res.success is True
    assert neg_res.same_person is False
    assert neg_res.similarity < verifier.default_threshold


def test_verifier_multi_reference_enrollment(verifier: FaceVerifier) -> None:
    """Test multi-image enrollment template aggregation and matching."""
    cp_imgs = [
        cv2.imread(str(SAMPLE_DIR / "Colin_Powell" / f"Colin_Powell_000{i}.jpg"))
        for i in [1, 2, 3]
        if (SAMPLE_DIR / "Colin_Powell" / f"Colin_Powell_000{i}.jpg").exists()
    ]
    test_cp = cv2.imread(str(SAMPLE_DIR / "Colin_Powell" / "Colin_Powell_0004.jpg"))
    test_gwb = cv2.imread(str(SAMPLE_DIR / "George_W_Bush" / "George_W_Bush_0001.jpg"))

    if len(cp_imgs) < 2 or test_cp is None or test_gwb is None:
        pytest.skip("Multi-reference sample images not found")

    # Multi-reference match with same person
    multi_pos = verifier.verify_multi_reference(cp_imgs, test_cp)
    assert multi_pos.success is True
    assert multi_pos.same_person is True
    assert multi_pos.similarity > verifier.default_threshold
    assert multi_pos.template_similarity is not None
    assert multi_pos.valid_reference_count == len(cp_imgs)

    # Multi-reference match with different person (impostor)
    multi_neg = verifier.verify_multi_reference(cp_imgs, test_gwb)
    assert multi_neg.success is True
    assert multi_neg.same_person is False
    assert multi_neg.similarity < verifier.default_threshold
