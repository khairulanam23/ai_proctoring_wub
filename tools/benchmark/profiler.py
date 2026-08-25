"""Granular latency profiling, stage-by-stage timing breakdowns, and hardware resource benchmarking."""

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

from proctoring.telemetry.performance import LatencyStatistics


@dataclass
class LatencyBenchmarkReport:
    """Rigorous benchmark report separating model inference latency from total pipeline overhead."""

    total_frames_profiled: int
    duration_seconds: float
    effective_throughput_fps: float
    pure_inference_fps: float
    model_inference_latency: LatencyStatistics  # Pure forward pass latency
    total_pipeline_latency: LatencyStatistics  # End-to-end frame latency
    stage_breakdown_means_ms: dict[str, float]
    per_model_latency_stats: dict[str, LatencyStatistics]
    resource_profile: dict[str, Any]
    environment: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_frames_profiled": self.total_frames_profiled,
            "duration_seconds": round(self.duration_seconds, 2),
            "effective_throughput_fps": round(self.effective_throughput_fps, 2),
            "pure_inference_fps": round(self.pure_inference_fps, 2),
            "model_inference_latency": self.model_inference_latency.to_dict(),
            "total_pipeline_latency": self.total_pipeline_latency.to_dict(),
            "stage_breakdown_means_ms": {
                k: round(v, 2) for k, v in self.stage_breakdown_means_ms.items()
            },
            "per_model_latency_stats": {
                k: v.to_dict() for k, v in self.per_model_latency_stats.items()
            },
            "resource_profile": self.resource_profile,
            "environment": self.environment,
        }


class PipelineLatencyProfiler:
    """High-precision profiler for measuring stage latencies and resource usage."""

    def __init__(self) -> None:
        self.raw_total_frame_times: list[float] = []
        self.raw_model_inference_times: list[float] = []
        self.stage_times: dict[str, list[float]] = {
            "capture_decode": [],
            "preprocessing": [],
            "face_detection": [],
            "face_verification": [],
            "object_detection": [],
            "temporal_aggregation": [],
            "evidence_capture_write": [],
        }
        self.model_times: dict[str, list[float]] = {}
        self.start_wall_time = time.perf_counter()

        self._process = psutil.Process(os.getpid()) if HAS_PSUTIL else None
        self._initial_rss_mb = self._get_rss_mb()
        self._peak_rss_mb = self._initial_rss_mb

    def _get_rss_mb(self) -> float:
        if self._process is not None:
            try:
                return float(self._process.memory_info().rss / (1024 * 1024))
            except Exception:
                pass
        return 0.0

    def record_stage_timing(
        self,
        capture_decode_ms: float = 0.0,
        preprocessing_ms: float = 0.0,
        face_detection_ms: float = 0.0,
        face_verification_ms: float = 0.0,
        object_detection_ms: float = 0.0,
        temporal_aggregation_ms: float = 0.0,
        evidence_capture_write_ms: float = 0.0,
        total_frame_ms: float | None = None,
    ) -> None:
        """Record stage timings for a single processed frame."""
        self.stage_times["capture_decode"].append(capture_decode_ms)
        self.stage_times["preprocessing"].append(preprocessing_ms)
        self.stage_times["face_detection"].append(face_detection_ms)
        self.stage_times["face_verification"].append(face_verification_ms)
        self.stage_times["object_detection"].append(object_detection_ms)
        self.stage_times["temporal_aggregation"].append(temporal_aggregation_ms)
        self.stage_times["evidence_capture_write"].append(evidence_capture_write_ms)

        # Pure model inference latency sum
        pure_model_ms = face_detection_ms + face_verification_ms + object_detection_ms
        self.raw_model_inference_times.append(pure_model_ms)

        if total_frame_ms is None:
            total_ms = (
                capture_decode_ms
                + preprocessing_ms
                + pure_model_ms
                + temporal_aggregation_ms
                + evidence_capture_write_ms
            )
        else:
            total_ms = total_frame_ms

        self.raw_total_frame_times.append(total_ms)

        # Update per-model buckets
        if face_detection_ms > 0:
            self.model_times.setdefault("face_detection_yunet", []).append(face_detection_ms)
        if face_verification_ms > 0:
            self.model_times.setdefault("face_verification_sface", []).append(face_verification_ms)
        if object_detection_ms > 0:
            self.model_times.setdefault("object_detection_yolo", []).append(object_detection_ms)

        # Update peak memory
        rss = self._get_rss_mb()
        if rss > self._peak_rss_mb:
            self._peak_rss_mb = rss

    def _calc_stats(self, data: list[float]) -> LatencyStatistics:
        if not data:
            return LatencyStatistics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        arr = np.array(data, dtype=np.float64)
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

    def generate_benchmark_report(self) -> LatencyBenchmarkReport:
        """Compute statistical latency percentiles and resource usage."""
        duration_sec = max(0.001, time.perf_counter() - self.start_wall_time)
        n_frames = len(self.raw_total_frame_times)
        throughput_fps = n_frames / duration_sec

        model_stats = self._calc_stats(self.raw_model_inference_times)
        total_stats = self._calc_stats(self.raw_total_frame_times)

        pure_inference_fps = (1000.0 / model_stats.mean_ms) if model_stats.mean_ms > 0 else 0.0

        stage_means = {k: float(np.mean(v)) if v else 0.0 for k, v in self.stage_times.items()}

        per_model_stats = {
            m_name: self._calc_stats(times) for m_name, times in self.model_times.items()
        }

        cpu_pct = 0.0
        if self._process is not None:
            try:
                cpu_pct = float(self._process.cpu_percent(interval=None))
            except Exception:
                pass

        resource_profile = {
            "initial_memory_rss_mb": round(self._initial_rss_mb, 2),
            "peak_memory_rss_mb": round(self._peak_rss_mb, 2),
            "current_memory_rss_mb": round(self._get_rss_mb(), 2),
            "process_cpu_percent": round(cpu_pct, 2),
        }

        env = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "os": platform.system(),
        }

        return LatencyBenchmarkReport(
            total_frames_profiled=n_frames,
            duration_seconds=duration_sec,
            effective_throughput_fps=throughput_fps,
            pure_inference_fps=pure_inference_fps,
            model_inference_latency=model_stats,
            total_pipeline_latency=total_stats,
            stage_breakdown_means_ms=stage_means,
            per_model_latency_stats=per_model_stats,
            resource_profile=resource_profile,
            environment=env,
        )
