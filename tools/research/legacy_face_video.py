"""Video face presence and state transition analyzer."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import cv2

from proctoring.detection.face_presence import FacePresenceAnalyzer, FacePresenceResult, FacePresenceStatus


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


@dataclass
class PresenceEvent:
    """Represents a state transition or significant face presence event."""
    timestamp_formatted: str
    timestamp_sec: float
    frame_index: int
    event_type: str
    previous_status: Optional[str]
    current_status: str
    face_count: int
    faces: List[Dict[str, Any]]
    source_media: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to serializable dictionary."""
        return {
            "timestamp_formatted": self.timestamp_formatted,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "frame_index": self.frame_index,
            "event_type": self.event_type,
            "previous_status": self.previous_status,
            "current_status": self.current_status,
            "face_count": self.face_count,
            "source_media": self.source_media,
            "faces": self.faces,
        }


@dataclass
class VideoAnalysisSummary:
    """Summary of face presence analysis across a video stream."""
    source_media: str
    total_frames: int
    processed_frames: int
    fps: float
    duration_sec: float
    events: List[PresenceEvent]
    status_distribution: Dict[str, int]
    avg_inference_time_ms: float
    min_inference_time_ms: float
    max_inference_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire summary to serializable dictionary."""
        return {
            "source_media": self.source_media,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "fps": round(self.fps, 2),
            "duration_sec": round(self.duration_sec, 2),
            "avg_inference_time_ms": round(self.avg_inference_time_ms, 2),
            "min_inference_time_ms": round(self.min_inference_time_ms, 2),
            "max_inference_time_ms": round(self.max_inference_time_ms, 2),
            "status_distribution": self.status_distribution,
            "total_events": len(self.events),
            "events": [event.to_dict() for event in self.events],
        }

    def save_evidence_json(self, output_path: Union[str, Path]) -> Path:
        """Save analysis summary and evidence events to JSON."""
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return dest


class VideoPresenceAnalyzer:
    """Processes video streams to detect face presence state transitions."""

    def __init__(
        self,
        presence_analyzer: Optional[FacePresenceAnalyzer] = None,
    ) -> None:
        self.presence_analyzer = (
            presence_analyzer if presence_analyzer is not None else FacePresenceAnalyzer()
        )

    def analyze_video(
        self,
        video_path: Union[str, Path],
        frame_step: int = 1,
        on_event_callback: Optional[Callable[[PresenceEvent], None]] = None,
    ) -> VideoAnalysisSummary:
        """Process a video file and track face presence state transitions.

        Args:
            video_path: Path to video file.
            frame_step: Process every N-th frame (default: 1, full analysis).
            on_event_callback: Optional callback invoked immediately when an event occurs.

        Returns:
            VideoAnalysisSummary with timeline events, latencies, and distributions.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found at: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_sec = total_frames / fps if fps > 0 else 0.0

        events: List[PresenceEvent] = []
        status_distribution: Dict[str, int] = {
            FacePresenceStatus.NO_FACE.value: 0,
            FacePresenceStatus.SINGLE_FACE.value: 0,
            FacePresenceStatus.MULTIPLE_FACES.value: 0,
        }
        latencies: List[float] = []

        previous_status: Optional[FacePresenceStatus] = None
        frame_idx = 0
        processed_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                if frame_step > 1 and (frame_idx - 1) % frame_step != 0:
                    continue

                timestamp_sec = (frame_idx - 1) / fps
                result = self.presence_analyzer.analyze(
                    image=frame,
                    frame_index=frame_idx,
                    timestamp_sec=timestamp_sec,
                )

                processed_count += 1
                latencies.append(result.inference_time_ms)
                status_distribution[result.status.value] += 1

                # Detect state transition
                if result.status != previous_status:
                    formatted_time = format_timestamp(timestamp_sec)
                    faces_data = [
                        {
                            "bbox": list(f.bbox),
                            "confidence": round(f.confidence, 4),
                            "landmarks": [list(lm) for lm in f.landmarks],
                        }
                        for f in result.faces
                    ]
                    event = PresenceEvent(
                        timestamp_formatted=formatted_time,
                        timestamp_sec=timestamp_sec,
                        frame_index=frame_idx,
                        event_type=result.status.value,
                        previous_status=previous_status.value if previous_status else None,
                        current_status=result.status.value,
                        face_count=result.face_count,
                        faces=faces_data,
                        source_media=path.name,
                    )
                    events.append(event)
                    if on_event_callback is not None:
                        on_event_callback(event)

                    previous_status = result.status
        finally:
            cap.release()

        avg_lat = float(np.mean(latencies)) if latencies else 0.0
        min_lat = float(np.min(latencies)) if latencies else 0.0
        max_lat = float(np.max(latencies)) if latencies else 0.0

        return VideoAnalysisSummary(
            source_media=path.name,
            total_frames=total_frames,
            processed_frames=processed_count,
            fps=fps,
            duration_sec=duration_sec,
            events=events,
            status_distribution=status_distribution,
            avg_inference_time_ms=avg_lat,
            min_inference_time_ms=min_lat,
            max_inference_time_ms=max_lat,
        )
