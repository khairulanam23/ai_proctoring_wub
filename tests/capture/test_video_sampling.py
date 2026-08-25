"""Unit tests for VideoFrameSampler, VideoObjectAnalyzer, and timeline evidence logging."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from proctoring.capture.video import (
    FrameSample,
    VideoFrameSampler,
    VideoMetadata,
    format_timestamp,
)
from proctoring.detection.object_detector import (
    DetectedObject,
    ObjectDetectionResult,
    ObjectDetector,
)
from proctoring.detection.object_relevance import ObjectRelevanceFilter
from tools.research.legacy_object_video import (
    ObjectPresenceInterval,
    TimelineEntry,
    VideoObjectAnalyzer,
)

SYNTHETIC_VIDEO = Path("data/samples/synthetic/test_presence_transitions.mp4")


def test_format_timestamp() -> None:
    """Test MM:SS.mmm timestamp formatting helper."""
    assert format_timestamp(0.0) == "00:00.000"
    assert format_timestamp(12.5) == "00:12.500"
    assert format_timestamp(65.125) == "01:05.125"
    assert format_timestamp(3600.0) == "60:00.000"
    assert format_timestamp(-5.0) == "00:00.000"


def test_frame_sample_and_metadata_serialization() -> None:
    """Test FrameSample and VideoMetadata dataclasses and to_dict serialization."""
    dummy_frame = np.zeros((360, 480, 3), dtype=np.uint8)
    sample = FrameSample(
        frame=dummy_frame,
        frame_index=15,
        timestamp_seconds=0.500,
        formatted_timestamp="00:00.500",
        source_fps=30.0,
    )

    data = sample.to_dict()
    assert data["frame_index"] == 15
    assert data["timestamp_seconds"] == 0.500
    assert data["formatted_timestamp"] == "00:00.500"
    assert data["source_fps"] == 30.0
    assert data["height"] == 360
    assert data["width"] == 480

    meta = VideoMetadata(
        width=480, height=360, source_fps=30.0, total_frames=120, duration_seconds=4.0
    )
    meta_dict = meta.to_dict()
    assert meta_dict["width"] == 480
    assert meta_dict["height"] == 360
    assert meta_dict["total_frames"] == 120
    assert meta_dict["duration_seconds"] == 4.0


def test_video_frame_sampler_missing_file() -> None:
    """Test that missing video file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        VideoFrameSampler("non_existent_video.mp4")


def test_video_frame_sampler_synthetic_video() -> None:
    """Test sequential frame sampling on synthetic video stream."""
    if not SYNTHETIC_VIDEO.exists():
        pytest.skip(f"Synthetic video not found at: {SYNTHETIC_VIDEO}")

    # 4.00s video (120 frames at 30 FPS) sampled at 2.0 FPS -> sample every 15 frames -> 8 frames
    sampler = VideoFrameSampler(video_path=SYNTHETIC_VIDEO, target_sampling_fps=2.0)
    meta = sampler.get_metadata()

    assert meta.width == 480
    assert meta.height == 360
    assert meta.source_fps == 30.0
    assert meta.total_frames == 120
    assert sampler.step == 15

    samples = list(sampler.sample_frames())
    assert len(samples) == 8

    # Verify frame indices and timestamps
    expected_indices = [0, 15, 30, 45, 60, 75, 90, 105]
    for _idx, (s, expected_idx) in enumerate(zip(samples, expected_indices, strict=False)):
        assert s.frame_index == expected_idx
        assert np.isclose(s.timestamp_seconds, expected_idx / 30.0, atol=1e-3)
        assert s.frame.shape == (360, 480, 3)


def test_timeline_entry_and_presence_interval_serialization() -> None:
    """Test TimelineEntry and ObjectPresenceInterval serialization."""
    objs = [DetectedObject(0, "person", 0.94, (10, 20, 100, 200))]
    entry = TimelineEntry(
        frame_index=15,
        timestamp_seconds=0.500,
        formatted_timestamp="00:00.500",
        person_count=1,
        relevant_objects=objs,
        ignored_objects=[],
        relevant_count=1,
        total_detections=1,
        inference_time_ms=12.5,
        processing_time_ms=14.0,
        device="cpu",
    )

    data = entry.to_dict()
    assert data["frame_index"] == 15
    assert data["person_count"] == 1
    assert data["relevant_count"] == 1
    assert data["relevant_objects"][0]["class_name"] == "person"

    presence = ObjectPresenceInterval(
        class_name="cell phone",
        first_seen_seconds=1.0,
        last_seen_seconds=2.5,
        detection_count=4,
        max_confidence=0.88,
        timestamps=[1.0, 1.5, 2.0, 2.5],
    )
    p_dict = presence.to_dict()
    assert p_dict["class_name"] == "cell phone"
    assert p_dict["detection_count"] == 4
    assert len(p_dict["timestamps"]) == 4


def test_video_object_analyzer_pipeline_with_mock_detector(tmp_path: Path) -> None:
    """Test complete VideoObjectAnalyzer pipeline with mock detector and relevance filter."""
    if not SYNTHETIC_VIDEO.exists():
        pytest.skip(f"Synthetic video not found at: {SYNTHETIC_VIDEO}")

    # Build mock detector that alternates detections based on frame calls
    mock_detector = MagicMock(spec=ObjectDetector)
    mock_detector.device = "cpu"
    mock_detector.model_name = "yolo11n.pt"

    call_count = 0

    def mock_detect(image: np.ndarray, confidence_threshold=None) -> ObjectDetectionResult:
        nonlocal call_count
        call_count += 1
        h, w = image.shape[:2]

        if call_count in (1, 2):  # frames 0, 15: person
            objs = [DetectedObject(0, "person", 0.92, (50, 50, 200, 300))]
        elif call_count in (3, 4):  # frames 30, 45: person + phone + tie
            objs = [
                DetectedObject(0, "person", 0.94, (50, 50, 200, 300)),
                DetectedObject(67, "cell phone", 0.85, (100, 120, 150, 180)),
                DetectedObject(32, "tie", 0.88, (80, 90, 95, 140)),
            ]
        else:  # frames 60+: no relevant objects (empty or background)
            objs = [DetectedObject(56, "chair", 0.70, (200, 200, 300, 350))]

        return ObjectDetectionResult(
            objects=objs,
            count=len(objs),
            image_shape=image.shape,
            image_width=w,
            image_height=h,
            inference_time_ms=8.0,
            model_name="yolo11n.pt",
            device="cpu",
        )

    mock_detector.detect.side_effect = mock_detect

    relevance_filter = ObjectRelevanceFilter()
    analyzer = VideoObjectAnalyzer(
        detector=mock_detector,
        relevance_filter=relevance_filter,
        target_sampling_fps=2.0,
    )

    frames_output_dir = tmp_path / "frames"
    report = analyzer.analyze_video(
        video_path=SYNTHETIC_VIDEO,
        save_annotated_dir=frames_output_dir,
        show_all=True,
    )

    # 1. Verification of sampled count
    assert report.total_sampled_frames == 8
    assert len(report.timeline) == 8

    # 2. Timeline checks
    assert report.timeline[0].person_count == 1
    assert report.timeline[0].relevant_count == 1

    # Frames 30, 45 had person + cell phone (tie filtered to ignored)
    assert report.timeline[2].person_count == 1
    assert report.timeline[2].relevant_count == 2
    rel_names_frame3 = [o.class_name for o in report.timeline[2].relevant_objects]
    assert "person" in rel_names_frame3
    assert "cell phone" in rel_names_frame3
    assert "tie" not in rel_names_frame3
    assert len(report.timeline[2].ignored_objects) == 1

    # 3. Object presence occurrence map
    assert "person" in report.presence_summary
    assert "cell phone" in report.presence_summary
    assert report.presence_summary["person"].detection_count == 4
    assert report.presence_summary["cell phone"].detection_count == 2
    assert report.presence_summary["cell phone"].first_seen_seconds == 1.0
    assert report.presence_summary["cell phone"].last_seen_seconds == 1.5

    # 4. JSON Serialization check
    json_data = report.to_dict()
    assert json_data["total_sampled_frames"] == 8
    assert "presence_summary" in json_data
    assert "timeline" in json_data
    assert len(json_data["timeline"]) == 8

    # 5. Check annotated frames saved
    saved_frames = list(frames_output_dir.glob("*.jpg"))
    assert len(saved_frames) == 8
