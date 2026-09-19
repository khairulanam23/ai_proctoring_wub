"""Targeted unit tests for ModelRegistry SHA-256 integrity verification."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from proctoring.detection.models import (
    ModelIntegrityError,
    ModelRegistry,
    compute_file_sha256,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry cache and trusted hashes are cleared before and after each test."""
    ModelRegistry.unload_all(clear_hashes=True)
    yield
    ModelRegistry.unload_all(clear_hashes=True)


def test_compute_file_sha256(tmp_path: Path) -> None:
    """Verify compute_file_sha256 accurately calculates checksums in blocks."""
    test_file = tmp_path / "sample_model.pt"
    content = b"MockNeuralWeightsData_BlockTest" * 1024
    test_file.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest().lower()
    computed = compute_file_sha256(test_file)

    assert computed == expected
    assert len(computed) == 64


def test_compute_file_sha256_missing_file(tmp_path: Path) -> None:
    """Verify FileNotFoundError is raised when file does not exist."""
    missing = tmp_path / "nonexistent.onnx"
    with pytest.raises(FileNotFoundError):
        compute_file_sha256(missing)


def test_correct_sha256_accepted(tmp_path: Path) -> None:
    """Verify that a model with a matching SHA-256 hash is accepted."""
    model_file = tmp_path / "test_model.onnx"
    content = b"ValidModelWeightsContent"
    model_file.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest().lower()

    # 1. Direct expected_sha256 parameter
    res = ModelRegistry.verify_model_integrity(model_file, expected_sha256=expected_hash)
    assert res == expected_hash

    # 2. Registered via set_trusted_hash by path
    ModelRegistry.set_trusted_hash(model_file, expected_hash)
    res_reg = ModelRegistry.verify_model_integrity(model_file)
    assert res_reg == expected_hash

    # 3. Registered by filename
    ModelRegistry.clear_trusted_hashes()
    ModelRegistry.set_trusted_hash(model_file.name, expected_hash)
    res_name = ModelRegistry.verify_model_integrity(model_file)
    assert res_name == expected_hash


def test_manifest_loading(tmp_path: Path) -> None:
    """Verify loading trusted hashes from a JSON manifest file."""
    model_file = tmp_path / "yolo11n_test.pt"
    content = b"Yolo11WeightsContent"
    model_file.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest().lower()

    manifest_file = tmp_path / "model_manifest.json"
    manifest_data = {
        model_file.name: expected_hash,
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    ModelRegistry.load_hash_manifest(manifest_file)
    assert ModelRegistry.get_expected_hash(model_file) == expected_hash

    verified = ModelRegistry.verify_model_integrity(model_file)
    assert verified == expected_hash


def test_modified_mismatched_weight_rejected_before_loading(tmp_path: Path) -> None:
    """Verify that modified/tampered model weights are rejected with ModelIntegrityError before loading."""
    model_file = tmp_path / "tampered_yolo.pt"
    original_content = b"AuthenticModelWeights"
    model_file.write_bytes(original_content)
    authentic_hash = hashlib.sha256(original_content).hexdigest().lower()

    # Register authentic hash
    ModelRegistry.set_trusted_hash(model_file, authentic_hash)

    # Tamper with file content
    model_file.write_bytes(b"TamperedMaliciousPayload")

    # verify_model_integrity fails closed
    with pytest.raises(ModelIntegrityError) as exc_info:
        ModelRegistry.verify_model_integrity(model_file)
    assert "SHA-256 integrity verification failed" in str(exc_info.value)

    # get_yolo fails closed before instantiating ultralytics.YOLO
    with patch("ultralytics.YOLO") as mock_yolo:
        with pytest.raises(ModelIntegrityError):
            ModelRegistry.get_yolo(model_file, device="cpu")
        mock_yolo.assert_not_called()

    # get_ort_session fails closed before instantiating ORTSessionWrapper
    with patch("proctoring.detection.backends.onnx_backend.ORTSessionWrapper") as mock_ort:
        with pytest.raises(ModelIntegrityError):
            ModelRegistry.get_ort_session(model_file, prefer_cuda=False)
        mock_ort.assert_not_called()


def test_unconfigured_verification_preserves_existing_behavior(tmp_path: Path) -> None:
    """Verify that without configured hashes, verification is skipped and existing dev behavior is preserved."""
    model_file = tmp_path / "untracked_dev_model.pt"
    model_file.write_bytes(b"DevWeights")

    # When no hash is registered, verify_model_integrity returns None and does not fail
    result = ModelRegistry.verify_model_integrity(model_file)
    assert result is None

    # Calling get_yolo loads the model via YOLO constructor without failing closed
    with patch("ultralytics.YOLO") as mock_yolo:
        mock_yolo.return_value = MagicMock()
        model = ModelRegistry.get_yolo(model_file, device="cpu")
        assert model is not None
        mock_yolo.assert_called_once_with(str(model_file))


def test_production_mode_requires_configured_hashes(tmp_path: Path, monkeypatch) -> None:
    """Verify that production enforcement policy rejects models missing a configured hash."""
    model_file = tmp_path / "production_model.pt"
    model_file.write_bytes(b"ProdWeights")

    # In default dev mode, unconfigured model passes
    assert ModelRegistry.verify_model_integrity(model_file) is None

    # When enforce_integrity is enabled programmatically
    ModelRegistry.set_enforce_integrity(True)
    with pytest.raises(ModelIntegrityError) as exc:
        ModelRegistry.verify_model_integrity(model_file)
    assert "Production integrity verification required" in str(exc.value)

    # Disable programmatic enforcement
    ModelRegistry.set_enforce_integrity(False)

    # When enforce_integrity is enabled via environment variable
    monkeypatch.setenv("PROCTORING_REQUIRE_MODEL_INTEGRITY", "true")
    with pytest.raises(ModelIntegrityError) as exc:
        ModelRegistry.verify_model_integrity(model_file)
    assert "Production integrity verification required" in str(exc.value)

    # With hash provided in production mode, verification succeeds
    valid_hash = hashlib.sha256(b"ProdWeights").hexdigest()
    ModelRegistry.set_trusted_hash(model_file, valid_hash)
    res = ModelRegistry.verify_model_integrity(model_file)
    assert res == valid_hash
