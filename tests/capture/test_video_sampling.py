"""Unit tests for VideoFrameSampler and video capture utilities."""

from pathlib import Path

import numpy as np
import pytest

from proctoring.capture.video import (
    FrameSample,
    VideoFrameSampler,
    VideoMetadata,
    format_timestamp,
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
