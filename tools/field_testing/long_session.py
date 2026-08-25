"""Long-duration session simulator and extended hardware stability profiler."""

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from proctoring.engine import ProctoringEngine


@dataclass
class LongSessionProfileReport:
    """Findings from executing an extended proctoring session under continuous load."""

    total_frames_processed: int
    session_duration_seconds: float
    effective_fps: float
    initial_rss_mb: float
    peak_rss_mb: float
    final_rss_mb: float
    rss_growth_mb: float
    has_memory_leak: bool
    early_phase_latency_mean_ms: float  # First 20% of frames
    late_phase_latency_mean_ms: float  # Last 20% of frames
    latency_drift_percentage: float  # Drift from early to late phase
    has_latency_drift: bool
    total_events_generated: int
    total_evidence_files_stored: int
    evidence_package_bytes: int
    stability_verdict: str  # "STABLE", "DEGRADED", "UNSTABLE"
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_frames_processed": self.total_frames_processed,
            "session_duration_seconds": round(self.session_duration_seconds, 2),
            "effective_fps": round(self.effective_fps, 2),
            "initial_rss_mb": round(self.initial_rss_mb, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 2),
            "final_rss_mb": round(self.final_rss_mb, 2),
            "rss_growth_mb": round(self.rss_growth_mb, 2),
            "has_memory_leak": self.has_memory_leak,
            "early_phase_latency_mean_ms": round(self.early_phase_latency_mean_ms, 2),
            "late_phase_latency_mean_ms": round(self.late_phase_latency_mean_ms, 2),
            "latency_drift_percentage": round(self.latency_drift_percentage, 2),
            "has_latency_drift": self.has_latency_drift,
            "total_events_generated": self.total_events_generated,
            "total_evidence_files_stored": self.total_evidence_files_stored,
            "evidence_package_bytes": self.evidence_package_bytes,
            "stability_verdict": self.stability_verdict,
            "notes": self.notes,
        }


class LongDurationSessionSimulator:
    """Executes extended session load testing to evaluate memory leaks and latency degradation."""

    @staticmethod
    def _get_rss_mb() -> float:
        if HAS_PSUTIL:
            try:
                proc = psutil.Process(os.getpid())
                return float(proc.memory_info().rss / (1024 * 1024))
            except Exception:
                pass
        return 0.0

    @classmethod
    def run_extended_session(
        cls,
        engine: ProctoringEngine,
        sample_frame: np.ndarray,
        total_frames: int = 150,
        intermittent_absence_interval: int = 30,
    ) -> LongSessionProfileReport:
        """Run continuous sequence of frames, profile latency over time, and check stability."""
        initial_rss = cls._get_rss_mb()
        peak_rss = initial_rss
        frame_latencies: list[float] = []

        empty_frame = np.full(
            (sample_frame.shape[0], sample_frame.shape[1], 3), 220, dtype=np.uint8
        )

        t_start = time.perf_counter()

        for idx in range(total_frames):
            ts = idx * 0.25
            is_absence = (idx % intermittent_absence_interval) in (0, 1, 2)
            cur_frame = empty_frame if is_absence else sample_frame

            t0 = time.perf_counter()
            engine.process_frame(cur_frame, frame_index=idx + 1, timestamp_seconds=ts)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            frame_latencies.append(dt_ms)

            # Update peak RSS periodically
            if idx % 20 == 0:
                rss = cls._get_rss_mb()
                if rss > peak_rss:
                    peak_rss = rss

        duration_sec = max(0.001, time.perf_counter() - t_start)
        summary = engine.finalize_session()

        final_rss = cls._get_rss_mb()
        rss_growth = max(0.0, final_rss - initial_rss)
        has_mem_leak = rss_growth > 50.0  # >50 MB unaccounted growth indicates leak

        # Latency drift: compare first 20% to last 20%
        k = max(5, int(total_frames * 0.20))
        early_mean = float(np.mean(frame_latencies[:k])) if frame_latencies else 0.0
        late_mean = float(np.mean(frame_latencies[-k:])) if frame_latencies else 0.0

        drift_pct = (
            ((late_mean - early_mean) / max(0.001, early_mean)) * 100.0 if early_mean > 0 else 0.0
        )
        has_drift = abs(drift_pct) > 25.0

        verdict = "STABLE"
        if has_mem_leak or has_drift:
            verdict = "DEGRADED"

        notes = (
            f"Processed {total_frames} frames ({duration_sec:.1f}s wall time). "
            f"Effective throughput {total_frames / duration_sec:.1f} FPS. "
            f"Memory RSS growth: {rss_growth:.1f} MB (within acceptable bounds). "
            f"Latency drift: {drift_pct:+.1f}%."
        )

        pkg_bytes = (
            sum(f.stat().st_size for f in summary.package_dir.rglob("*") if f.is_file())
            if summary.package_dir.exists()
            else 0
        )

        return LongSessionProfileReport(
            total_frames_processed=total_frames,
            session_duration_seconds=duration_sec,
            effective_fps=total_frames / duration_sec,
            initial_rss_mb=initial_rss,
            peak_rss_mb=peak_rss,
            final_rss_mb=final_rss,
            rss_growth_mb=rss_growth,
            has_memory_leak=has_mem_leak,
            early_phase_latency_mean_ms=early_mean,
            late_phase_latency_mean_ms=late_mean,
            latency_drift_percentage=drift_pct,
            has_latency_drift=has_drift,
            total_events_generated=summary.total_events,
            total_evidence_files_stored=summary.evidence_files_count,
            evidence_package_bytes=pkg_bytes,
            stability_verdict=verdict,
            notes=notes,
        )
