"""Unit tests for VideoPresenceAnalyzer and state transitions."""

import json
from pathlib import Path

import pytest

from proctoring.detection.face_detector import FaceDetector
from proctoring.detection.face_presence import FacePresenceAnalyzer, FacePresenceStatus
from tools.research.legacy_face_video import (
    VideoAnalysisSummary,
    VideoPresenceAnalyzer,
    format_timestamp,
)


@pytest.fixture
def video_analyzer() -> VideoPresenceAnalyzer:
    model_path = Path("models/face_detection_yunet_2023mar.onnx")
    if not model_path.exists():
        pytest.skip(f"YuNet model missing at {model_path}")
    detector = FaceDetector(model_path=model_path)
    presence_analyzer = FacePresenceAnalyzer(detector=detector)
    return VideoPresenceAnalyzer(presence_analyzer=presence_analyzer)


def test_format_timestamp() -> None:
    """Test timestamp formatting."""
    assert format_timestamp(0.0) == "00:00:00.000"
    assert format_timestamp(3.24) == "00:00:03.240"
    assert format_timestamp(65.5) == "00:01:05.500"
    assert format_timestamp(3665.123) == "01:01:05.123"


def test_video_analyzer_missing_file(video_analyzer: VideoPresenceAnalyzer) -> None:
    """Test error handling for non-existent video file."""
    with pytest.raises(FileNotFoundError):
        video_analyzer.analyze_video("non_existent_video.mp4")


def test_video_analyzer_synthetic_transitions(
    video_analyzer: VideoPresenceAnalyzer,
    tmp_path: Path,
) -> None:
    """Test video state transition detection on synthetic transition video."""
    video_path = Path("data/samples/synthetic/test_presence_transitions.mp4")
    if not video_path.exists():
        pytest.skip("Synthetic transition video missing")

    summary = video_analyzer.analyze_video(video_path=video_path, frame_step=1)

    assert isinstance(summary, VideoAnalysisSummary)
    assert summary.total_frames == 120
    assert summary.processed_frames == 120
    assert summary.fps == 30.0
    assert summary.duration_sec == 4.0
    assert len(summary.events) >= 4

    # Verify event types in sequence
    event_types = [e.event_type for e in summary.events]
    assert FacePresenceStatus.SINGLE_FACE.value in event_types
    assert FacePresenceStatus.MULTIPLE_FACES.value in event_types
    assert FacePresenceStatus.NO_FACE.value in event_types

    # Verify status distribution
    dist = summary.status_distribution
    assert dist[FacePresenceStatus.NO_FACE.value] > 0
    assert dist[FacePresenceStatus.SINGLE_FACE.value] > 0
    assert dist[FacePresenceStatus.MULTIPLE_FACES.value] > 0

    # Test saving evidence JSON
    json_path = tmp_path / "test_evidence.json"
    saved = summary.save_evidence_json(json_path)
    assert saved.exists()

    with open(saved, encoding="utf-8") as f:
        data = json.load(f)
    assert data["source_media"] == "test_presence_transitions.mp4"
    assert data["total_events"] == len(summary.events)
