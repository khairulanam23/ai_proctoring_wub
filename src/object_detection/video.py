"""Video object detection pipeline, timeline evidence aggregation, and temporal tracking."""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import cv2

from src.object_detection.detector import DetectedObject, ObjectDetectionResult, ObjectDetector
from src.object_detection.relevance import (
    ObjectRelevanceFilter,
    ProctoringDetectionReport,
)
from src.object_detection.sampling import (
    FrameSample,
    VideoFrameSampler,
    VideoMetadata,
    format_timestamp,
)
from src.object_detection.temporal import (
    ObjectPresenceEvent,
    PersonCountChangeEvent,
    TemporalEventEngine,
    TemporalEventReport,
    render_temporal_gantt_chart,
)


@dataclass
class TimelineEntry:
    """Objective detection telemetry recorded at a specific video timestamp."""
    frame_index: int
    timestamp_seconds: float
    formatted_timestamp: str
    person_count: int
    relevant_objects: List[DetectedObject]
    ignored_objects: List[DetectedObject]
    relevant_count: int
    total_detections: int
    inference_time_ms: float
    processing_time_ms: float
    device: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert timeline entry to a JSON-serializable dictionary."""
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "formatted_timestamp": self.formatted_timestamp,
            "person_count": self.person_count,
            "relevant_count": self.relevant_count,
            "total_detections": self.total_detections,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "device": self.device,
            "relevant_objects": [obj.to_dict() for obj in self.relevant_objects],
            "ignored_objects": [obj.to_dict() for obj in self.ignored_objects],
        }


@dataclass
class ObjectPresenceInterval:
    """Summary of consecutive/periodic detections of a specific object class across time."""
    class_name: str
    first_seen_seconds: float
    last_seen_seconds: float
    detection_count: int
    max_confidence: float
    timestamps: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "first_seen_seconds": round(self.first_seen_seconds, 3),
            "last_seen_seconds": round(self.last_seen_seconds, 3),
            "detection_count": self.detection_count,
            "max_confidence": round(self.max_confidence, 4),
            "timestamps": [round(t, 3) for t in self.timestamps],
        }


@dataclass
class VideoAnalysisReport:
    """Comprehensive structured report of video object detection, raw timeline, and consolidated temporal events."""
    video_path: str
    metadata: VideoMetadata
    target_sampling_fps: float
    total_sampled_frames: int
    timeline: List[TimelineEntry]
    presence_summary: Dict[str, ObjectPresenceInterval]
    total_processing_time_ms: float
    average_inference_time_ms: float
    average_frame_processing_time_ms: float
    approximate_fps: float
    temporal_events: List[ObjectPresenceEvent] = field(default_factory=list)
    person_count_changes: List[PersonCountChangeEvent] = field(default_factory=list)
    temporal_report: Optional[TemporalEventReport] = None
    evidence_package: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert complete report to a JSON-serializable dictionary."""
        return {
            "video_path": self.video_path,
            "metadata": self.metadata.to_dict(),
            "target_sampling_fps": round(self.target_sampling_fps, 2),
            "total_sampled_frames": self.total_sampled_frames,
            "total_processing_time_ms": round(self.total_processing_time_ms, 2),
            "average_inference_time_ms": round(self.average_inference_time_ms, 2),
            "average_frame_processing_time_ms": round(self.average_frame_processing_time_ms, 2),
            "approximate_fps": round(self.approximate_fps, 2),
            "presence_summary": {k: v.to_dict() for k, v in self.presence_summary.items()},
            "temporal_events": [e.to_dict() for e in self.temporal_events],
            "person_count_changes": [c.to_dict() for c in self.person_count_changes],
            "timeline": [entry.to_dict() for entry in self.timeline],
        }


class VideoObjectAnalyzer:
    """Pipeline orchestrating frame sampling, detection, relevance filtering, and temporal state management."""

    def __init__(
        self,
        detector: Optional[ObjectDetector] = None,
        relevance_filter: Optional[ObjectRelevanceFilter] = None,
        target_sampling_fps: float = 2.0,
        absence_tolerance_seconds: float = 1.0,
        min_event_duration_seconds: float = 1.0,
        enable_temporal_events: bool = True,
    ) -> None:
        self.detector = detector if detector is not None else ObjectDetector()
        self.relevance_filter = (
            relevance_filter if relevance_filter is not None else ObjectRelevanceFilter()
        )
        self.target_sampling_fps = float(target_sampling_fps)
        self.absence_tolerance_seconds = float(absence_tolerance_seconds)
        self.min_event_duration_seconds = float(min_event_duration_seconds)
        self.enable_temporal_events = enable_temporal_events

    def analyze_video(
        self,
        video_path: Union[str, Path],
        target_sampling_fps: Optional[float] = None,
        save_annotated_dir: Optional[Union[str, Path]] = None,
        plot_timeline_path: Optional[Union[str, Path]] = None,
        save_evidence_dir: Optional[Union[str, Path]] = None,
        zip_evidence: bool = False,
        show_all: bool = False,
    ) -> VideoAnalysisReport:
        """Process a video stream through frame sampling, detection, and temporal event logging.

        Args:
            video_path: Path to video file.
            target_sampling_fps: Optional override for sampling FPS.
            save_annotated_dir: Optional directory to save annotated visual frames.
            plot_timeline_path: Optional path to save visual Gantt timeline chart image.
            save_evidence_dir: Optional path to build structured EvidencePackage.
            zip_evidence: If True, packages evidence directory into a .zip archive.
            show_all: If True, renders ignored background objects in visual output.

        Returns:
            VideoAnalysisReport containing raw timeline, temporal events, and optional EvidencePackage.
        """
        sampling_fps = target_sampling_fps if target_sampling_fps is not None else self.target_sampling_fps
        sampler = VideoFrameSampler(video_path=video_path, target_sampling_fps=sampling_fps)
        metadata = sampler.get_metadata()

        save_dir = Path(save_annotated_dir) if save_annotated_dir is not None else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        temporal_engine = TemporalEventEngine(
            absence_tolerance_seconds=self.absence_tolerance_seconds,
            min_event_duration_seconds=self.min_event_duration_seconds,
        )

        timeline: List[TimelineEntry] = []
        presence_map: Dict[str, ObjectPresenceInterval] = {}
        frame_cache: Dict[int, np.ndarray] = {}
        total_t0 = time.perf_counter()
        last_timestamp = 0.0

        for sample in sampler.sample_frames():
            t_frame0 = time.perf_counter()
            last_timestamp = sample.timestamp_seconds

            if save_evidence_dir is not None:
                frame_cache[sample.frame_index] = sample.frame.copy()

            # 1. Run detection
            raw_result = self.detector.detect(sample.frame)

            # 2. Run relevance filtering
            report = self.relevance_filter.filter(raw_result, timestamp=sample.timestamp_seconds)
            t_frame_ms = (time.perf_counter() - t_frame0) * 1000.0

            # 3. Ingest into temporal state engine
            if self.enable_temporal_events:
                temporal_engine.ingest_frame(
                    timestamp_seconds=sample.timestamp_seconds,
                    frame_index=sample.frame_index,
                    person_count=report.person_count,
                    detected_objects=report.relevant_objects,
                )

            # 4. Create Timeline Entry (raw frame-level telemetry)
            entry = TimelineEntry(
                frame_index=sample.frame_index,
                timestamp_seconds=sample.timestamp_seconds,
                formatted_timestamp=sample.formatted_timestamp,
                person_count=report.person_count,
                relevant_objects=report.relevant_objects,
                ignored_objects=report.ignored_objects,
                relevant_count=report.relevant_count,
                total_detections=report.total_detections,
                inference_time_ms=raw_result.inference_time_ms,
                processing_time_ms=t_frame_ms,
                device=raw_result.device,
            )
            timeline.append(entry)

            # 5. Update presence occurrence summary
            for obj in report.relevant_objects:
                cname = obj.class_name
                if cname not in presence_map:
                    presence_map[cname] = ObjectPresenceInterval(
                        class_name=cname,
                        first_seen_seconds=sample.timestamp_seconds,
                        last_seen_seconds=sample.timestamp_seconds,
                        detection_count=1,
                        max_confidence=obj.confidence,
                        timestamps=[sample.timestamp_seconds],
                    )
                else:
                    presence_map[cname].last_seen_seconds = sample.timestamp_seconds
                    presence_map[cname].detection_count += 1
                    presence_map[cname].max_confidence = max(presence_map[cname].max_confidence, obj.confidence)
                    presence_map[cname].timestamps.append(sample.timestamp_seconds)

            # 6. Optionally save annotated frame
            if save_dir is not None:
                annotated = self.relevance_filter.visualize_report(sample.frame, report, show_all=show_all)
                out_name = f"frame_{sample.frame_index:06d}_{sample.formatted_timestamp.replace(':', '_').replace('.', '_')}.jpg"
                cv2.imwrite(str(save_dir / out_name), annotated)

        # Finalize temporal engine state
        temporal_report = temporal_engine.finalize(last_timestamp) if self.enable_temporal_events else None
        temporal_events = temporal_report.events if temporal_report else []
        person_changes = temporal_report.person_count_changes if temporal_report else []

        total_time_ms = (time.perf_counter() - total_t0) * 1000.0
        sampled_count = len(timeline)

        avg_infer_ms = (
            float(np.mean([e.inference_time_ms for e in timeline])) if sampled_count > 0 else 0.0
        )
        avg_frame_proc_ms = (
            float(np.mean([e.processing_time_ms for e in timeline])) if sampled_count > 0 else 0.0
        )
        approx_fps = (1000.0 / avg_frame_proc_ms) if avg_frame_proc_ms > 0 else 0.0

        # 7. Optionally render temporal Gantt chart
        if plot_timeline_path is not None and temporal_report is not None:
            render_temporal_gantt_chart(
                report=temporal_report,
                total_duration_seconds=metadata.duration_seconds if metadata.duration_seconds > 0 else last_timestamp,
                output_path=Path(plot_timeline_path),
            )

        # 8. Build base analysis report
        analysis_report = VideoAnalysisReport(
            video_path=str(video_path),
            metadata=metadata,
            target_sampling_fps=sampling_fps,
            total_sampled_frames=sampled_count,
            timeline=timeline,
            presence_summary=presence_map,
            total_processing_time_ms=total_time_ms,
            average_inference_time_ms=avg_infer_ms,
            average_frame_processing_time_ms=avg_frame_proc_ms,
            approximate_fps=approx_fps,
            temporal_events=temporal_events,
            person_count_changes=person_changes,
            temporal_report=temporal_report,
        )

        # 9. Optionally build complete EvidencePackage
        if save_evidence_dir is not None:
            from src.object_detection.evidence import EvidenceBuilder
            builder = EvidenceBuilder()
            pkg = builder.build_package(
                report=analysis_report,
                output_dir=Path(save_evidence_dir),
                frame_cache=frame_cache,
                create_zip=zip_evidence,
            )
            analysis_report.evidence_package = pkg

        return analysis_report
