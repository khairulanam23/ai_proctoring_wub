"""Comprehensive performance telemetry, latency percentiles, and stage-by-stage profiling."""

import contextlib
import os
import platform
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class FrameTimingRecord:
    """Detailed stage-by-stage latency profile for an individual processed frame."""

    frame_index: int
    timestamp_seconds: float
    capture_decode_ms: float = 0.0
    preprocessing_ms: float = 0.0
    face_detector_ms: float = 0.0
    face_embedder_ms: float = 0.0
    object_detector_ms: float = 0.0
    behaviour_analysis_ms: float = 0.0
    temporal_postprocess_ms: float = 0.0
    evidence_io_ms: float = 0.0
    total_frame_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "capture_decode_ms": round(self.capture_decode_ms, 2),
            "preprocessing_ms": round(self.preprocessing_ms, 2),
            "face_detector_ms": round(self.face_detector_ms, 2),
            "face_embedder_ms": round(self.face_embedder_ms, 2),
            "object_detector_ms": round(self.object_detector_ms, 2),
            "behaviour_analysis_ms": round(self.behaviour_analysis_ms, 2),
            "temporal_postprocess_ms": round(self.temporal_postprocess_ms, 2),
            "evidence_io_ms": round(self.evidence_io_ms, 2),
            "total_frame_ms": round(self.total_frame_ms, 2),
        }


@dataclass
class LatencyStatistics:
    """Statistical summary of latency across all processed frames."""

    count: int
    mean_ms: float
    median_p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_dev_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 2),
            "median_p50_ms": round(self.median_p50_ms, 2),
            "p90_ms": round(self.p90_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "std_dev_ms": round(self.std_dev_ms, 2),
        }


@dataclass
class PerformanceReport:
    """Comprehensive examination session telemetry report."""

    session_id: str
    total_frames: int
    processed_frames: int
    skipped_frames: int
    session_duration_seconds: float
    effective_fps: float
    inference_fps: float
    latency_overall: LatencyStatistics
    stage_latencies_mean: dict[str, float]
    per_model_inference: dict[str, LatencyStatistics]
    resource_usage: dict[str, Any]
    environment: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "skipped_frames": self.skipped_frames,
            "session_duration_seconds": round(self.session_duration_seconds, 2),
            "effective_fps": round(self.effective_fps, 2),
            "inference_fps": round(self.inference_fps, 2),
            "latency_overall": self.latency_overall.to_dict(),
            "stage_latencies_mean_ms": {
                k: round(v, 2) for k, v in self.stage_latencies_mean.items()
            },
            "per_model_inference": {k: v.to_dict() for k, v in self.per_model_inference.items()},
            "resource_usage": self.resource_usage,
            "environment": self.environment,
        }


class PipelineTelemetryTracker:
    """Tracks latency percentiles, stage timings, and resource usage for proctoring sessions."""

    def __init__(self, session_id: str = "default_session") -> None:
        self.session_id = str(session_id)
        self.frame_timings: list[FrameTimingRecord] = []
        self.total_frames_count = 0
        self.processed_frames_count = 0
        self.skipped_frames_count = 0
        self.start_wall_time = time.time()
        self.start_process_time = time.process_time()

        # Per-model raw latencies
        self.model_latencies: dict[str, list[float]] = {}

        # Process monitor
        self._process = psutil.Process(os.getpid()) if HAS_PSUTIL else None
        self._initial_rss_mb = self._get_memory_rss_mb()
        self._peak_rss_mb = self._initial_rss_mb

    def _get_memory_rss_mb(self) -> float:
        if self._process is not None:
            try:
                return float(self._process.memory_info().rss / (1024 * 1024))
            except Exception:
                pass
        return 0.0

    def record_frame(
        self,
        timing: FrameTimingRecord,
        per_model_times: dict[str, float] | None = None,
    ) -> None:
        """Record stage timings for a processed frame."""
        self.frame_timings.append(timing)
        self.processed_frames_count += 1
        self.total_frames_count += 1

        # Track memory peak
        current_rss = self._get_memory_rss_mb()
        if current_rss > self._peak_rss_mb:
            self._peak_rss_mb = current_rss

        # Record per-model latencies
        if per_model_times:
            for model_name, ms in per_model_times.items():
                if model_name not in self.model_latencies:
                    self.model_latencies[model_name] = []
                self.model_latencies[model_name].append(ms)

    def record_skipped_frame(self) -> None:
        """Record a skipped or dropped frame."""
        self.skipped_frames_count += 1
        self.total_frames_count += 1

    def _compute_statistics(self, values: list[float]) -> LatencyStatistics:
        if not values:
            return LatencyStatistics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        arr = np.array(values, dtype=np.float64)
        return LatencyStatistics(
            count=len(arr),
            mean_ms=float(np.mean(arr)),
            median_p50_ms=float(np.median(arr)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            std_dev_ms=float(np.std(arr)),
        )

    def generate_report(self) -> PerformanceReport:
        """Generate complete telemetry and performance summary."""
        duration_sec = max(0.001, time.time() - self.start_wall_time)
        effective_fps = self.total_frames_count / duration_sec

        # Overall frame latencies
        total_frame_times = [ft.total_frame_ms for ft in self.frame_timings]
        overall_stats = self._compute_statistics(total_frame_times)

        mean_total_ms = overall_stats.mean_ms
        inference_fps = (1000.0 / mean_total_ms) if mean_total_ms > 0 else 0.0

        # Stage latency averages
        stage_averages = {
            "capture_decode": float(np.mean([ft.capture_decode_ms for ft in self.frame_timings]))
            if self.frame_timings
            else 0.0,
            "preprocessing": float(np.mean([ft.preprocessing_ms for ft in self.frame_timings]))
            if self.frame_timings
            else 0.0,
            "face_detector": float(np.mean([ft.face_detector_ms for ft in self.frame_timings]))
            if self.frame_timings
            else 0.0,
            "face_embedder": float(np.mean([ft.face_embedder_ms for ft in self.frame_timings]))
            if self.frame_timings
            else 0.0,
            "object_detector": float(np.mean([ft.object_detector_ms for ft in self.frame_timings]))
            if self.frame_timings
            else 0.0,
            "behaviour_analysis": float(
                np.mean([ft.behaviour_analysis_ms for ft in self.frame_timings])
            )
            if self.frame_timings
            else 0.0,
            "temporal_postprocess": float(
                np.mean([ft.temporal_postprocess_ms for ft in self.frame_timings])
            )
            if self.frame_timings
            else 0.0,
            "evidence_io": float(np.mean([ft.evidence_io_ms for ft in self.frame_timings]))
            if self.frame_timings
            else 0.0,
        }

        # Per-model inference statistics
        model_stats = {
            m_name: self._compute_statistics(times)
            for m_name, times in self.model_latencies.items()
        }

        # Resource usage
        cpu_pct = 0.0
        if self._process is not None:
            with contextlib.suppress(Exception):
                cpu_pct = float(self._process.cpu_percent(interval=None))

        resource_usage = {
            "initial_rss_mb": round(self._initial_rss_mb, 2),
            "peak_rss_mb": round(self._peak_rss_mb, 2),
            "current_rss_mb": round(self._get_memory_rss_mb(), 2),
            "process_cpu_percent": round(cpu_pct, 2),
        }

        environment = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "os": platform.system(),
        }

        return PerformanceReport(
            session_id=self.session_id,
            total_frames=self.total_frames_count,
            processed_frames=self.processed_frames_count,
            skipped_frames=self.skipped_frames_count,
            session_duration_seconds=duration_sec,
            effective_fps=effective_fps,
            inference_fps=inference_fps,
            latency_overall=overall_stats,
            stage_latencies_mean=stage_averages,
            per_model_inference=model_stats,
            resource_usage=resource_usage,
            environment=environment,
        )
