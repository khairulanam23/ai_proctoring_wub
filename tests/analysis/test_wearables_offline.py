"""Targeted tests for WearableDetector offline airgap enforcement and network isolation."""

import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from proctoring.analysis.wearables import WearableDetector
from proctoring.config import SessionConfig
from proctoring.engine import ProctoringEngine


def test_wearable_detector_missing_yolo_world_weights_no_download(tmp_path: Path, monkeypatch) -> None:
    """Verify missing YOLO-World weights transition to unavailable and trigger zero network calls."""
    # Ensure any attempt to open a network socket or HTTP request fails immediately
    def forbid_network(*args, **kwargs):
        raise AssertionError("Network access attempted during offline proctoring initialization!")

    monkeypatch.setattr(urllib.request, "urlopen", forbid_network)

    detector = WearableDetector(
        model_name="nonexistent_yolo_world_model_9999.pt",
        auto_load=True,
    )

    assert detector.is_available is False
    assert detector.model is None


def test_wearable_detector_missing_clip_weights_no_download(tmp_path: Path, monkeypatch) -> None:
    """Verify missing CLIP weights prevent network download and degrade gracefully to unavailable."""
    # Point candidate paths away from existing local clip weights
    def forbid_network(*args, **kwargs):
        raise AssertionError("Network access attempted for CLIP weights!")

    monkeypatch.setattr(urllib.request, "urlopen", forbid_network)

    # Patch clip_candidates search to simulate missing clip weights
    with patch("pathlib.Path.is_file") as mock_is_file:
        # Pretend yolo exists, but clip ViT-B-32 does not
        def fake_is_file(self):
            return "yolov8s-world.pt" in str(self)

        mock_is_file.side_effect = fake_is_file
        detector = WearableDetector(
            model_name="models/yolov8s-world.pt",
            auto_load=False,
        )
        loaded = detector.load_model()
        assert loaded is False
        assert detector.is_available is False


def test_wearable_detector_local_weights_initialize_offline(monkeypatch) -> None:
    """Verify local weights initialize successfully with zero network access."""
    def forbid_network(*args, **kwargs):
        raise AssertionError("Network access attempted during local initialization!")

    monkeypatch.setattr(urllib.request, "urlopen", forbid_network)

    local_model = Path("models/yolov8s-world.pt")
    local_clip = Path("weights/clip/ViT-B-32.pt")
    if not local_model.is_file() or not local_clip.is_file():
        pytest.skip("Local model or CLIP weights not present for offline load test")

    detector = WearableDetector(
        model_name="models/yolov8s-world.pt",
        auto_load=True,
    )
    assert detector.is_available is True
    assert detector.model is not None


def test_engine_startup_with_wearables_disabled(tmp_path: Path) -> None:
    """Verify normal engine startup remains completely functional when wearable detection is disabled."""
    config = SessionConfig(
        session_id="test_offline_startup",
        output_dir=tmp_path / "out",
        enable_wearable_detection=False,
    )
    engine = ProctoringEngine(config=config)
    assert engine.wearable_detector is None
    assert engine.is_active is False
