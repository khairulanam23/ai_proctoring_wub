"""Video and webcam frame sampling module for periodic object detection inference."""

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS.mmm human-readable timestamp string."""
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    rem_seconds = seconds % 60
    return f"{minutes:02d}:{rem_seconds:06.3f}"


@dataclass
class FrameSample:
    """Individual sampled video frame with original stream metadata."""

    frame: np.ndarray
    frame_index: int
    timestamp_seconds: float
    formatted_timestamp: str
    source_fps: float

    def to_dict(self) -> dict[str, Any]:
        """Convert sample metadata to dictionary (excluding raw pixels)."""
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "formatted_timestamp": self.formatted_timestamp,
            "source_fps": round(self.source_fps, 2),
            "height": self.frame.shape[0] if self.frame is not None else 0,
            "width": self.frame.shape[1] if self.frame is not None else 0,
        }


@dataclass
class VideoMetadata:
    """Metadata extracted from a video source stream."""

    width: int
    height: int
    source_fps: float
    total_frames: int
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "source_fps": round(self.source_fps, 2),
            "total_frames": self.total_frames,
            "duration_seconds": round(self.duration_seconds, 2),
        }


class VideoFrameSampler:
    """Sequential frame sampler supporting configurable inference sampling rates."""

    def __init__(
        self,
        video_path: str | Path,
        target_sampling_fps: float = 2.0,
    ) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.target_sampling_fps = max(0.1, float(target_sampling_fps))
        self.metadata = self._probe_video()

        # Calculate sampling step interval (e.g. 30 FPS / 2 target FPS = step of 15)
        if self.metadata.source_fps > 0:
            self.step = max(1, round(self.metadata.source_fps / self.target_sampling_fps))
        else:
            self.step = 1

    def _probe_video(self) -> VideoMetadata:
        """Probe video header for resolution, FPS, and frame count."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0.0 or np.isnan(fps):
                fps = 30.0  # Fallback default FPS

            duration = (total_frames / fps) if (total_frames > 0 and fps > 0) else 0.0

            return VideoMetadata(
                width=width,
                height=height,
                source_fps=fps,
                total_frames=total_frames,
                duration_seconds=duration,
            )
        finally:
            cap.release()

    def get_metadata(self) -> VideoMetadata:
        """Return the probed video stream metadata."""
        return self.metadata

    def sample_frames(self) -> Generator[FrameSample, None, None]:
        """Iterate through the video and yield FrameSample on sampling intervals."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video stream: {self.video_path}")

        frame_idx = 0
        source_fps = self.metadata.source_fps if self.metadata.source_fps > 0 else 30.0

        try:
            while True:
                is_sample_idx = frame_idx % self.step == 0

                if is_sample_idx:
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        break

                    timestamp_sec = frame_idx / source_fps
                    yield FrameSample(
                        frame=frame,
                        frame_index=frame_idx,
                        timestamp_seconds=timestamp_sec,
                        formatted_timestamp=format_timestamp(timestamp_sec),
                        source_fps=source_fps,
                    )
                else:
                    # Fast-forward without full decompression
                    ret = cap.grab()
                    if not ret:
                        break

                frame_idx += 1
        finally:
            cap.release()
