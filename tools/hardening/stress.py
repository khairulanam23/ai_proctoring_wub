"""Stress testing framework evaluating performance under multi-resolution inputs, varying sampling rates, and continuous load."""

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from proctoring.config import SessionConfig
from proctoring.engine import ProctoringEngine


@dataclass
class StressConditionResult:
    """Findings from an individual stress condition test."""

    condition_name: str
    resolution: tuple[int, int]
    target_fps: float
    total_frames: int
    effective_fps: float
    mean_latency_ms: float
    p95_latency_ms: float
    peak_rss_mb: float
    rss_growth_mb: float
    has_memory_leak: bool
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_name": self.condition_name,
            "resolution": list(self.resolution),
            "target_fps": self.target_fps,
            "total_frames": self.total_frames,
            "effective_fps": round(self.effective_fps, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 2),
            "rss_growth_mb": round(self.rss_growth_mb, 2),
            "has_memory_leak": self.has_memory_leak,
            "status": self.status,
        }


@dataclass
class StressTestMatrixResult:
    """Comprehensive stress test report across all test conditions."""

    total_conditions_tested: int
    results: list[StressConditionResult]
    overall_verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_conditions_tested": self.total_conditions_tested,
            "results": [r.to_dict() for r in self.results],
            "overall_verdict": self.overall_verdict,
        }


class PipelineStressTester:
    """Executes multi-resolution and continuous load stress tests."""

    @classmethod
    def run_stress_matrix(
        cls,
        output_dir: str | Path = "data/results/phase10_stress",
    ) -> StressTestMatrixResult:
        """Run stress matrix across low/normal/high resolutions and frame rates."""
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        process = psutil.Process(os.getpid())

        conditions = [
            ("Low Resolution (320x240, 4 FPS)", (320, 240), 4.0, 50),
            ("Standard Resolution (640x480, 4 FPS)", (640, 480), 4.0, 50),
            ("High Resolution (1280x720, 4 FPS)", (1280, 720), 4.0, 50),
            ("High Frame Rate (640x480, 8 FPS)", (640, 480), 8.0, 50),
            ("Continuous Long Load (640x480, 4 FPS)", (640, 480), 4.0, 150),
        ]

        # Warmup engine once to initialize underlying C++ runtime graphs
        warmup_cfg = SessionConfig(
            session_id="stress_warmup",
            student_name="Warmup Subject",
            output_dir=str(out_p),
            create_zip=False,
        )
        warmup_engine = ProctoringEngine(config=warmup_cfg)
        warmup_canvas = np.full((480, 640, 3), 150, dtype=np.uint8)
        for w_i in range(1, 4):
            warmup_engine.process_frame(
                warmup_canvas, frame_index=w_i, timestamp_seconds=(w_i - 1) * 0.25
            )
        warmup_engine.finalize_session()
        gc.collect()

        results: list[StressConditionResult] = []

        for name, (w, h), fps, n_frames in conditions:
            gc.collect()
            init_rss = process.memory_info().rss / (1024 * 1024)
            peak_rss = init_rss

            config = SessionConfig(
                session_id=f"stress_{w}x{h}_{int(fps)}fps",
                student_name="Stress Subject",
                sampling_fps=fps,
                output_dir=str(out_p),
                create_zip=False,
            )
            engine = ProctoringEngine(config=config)

            canvas = np.full((h, w, 3), 150, dtype=np.uint8)
            latencies: list[float] = []

            t_start = time.perf_counter()
            for i in range(1, n_frames + 1):
                t0 = time.perf_counter()
                engine.process_frame(canvas, frame_index=i, timestamp_seconds=(i - 1) / fps)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt_ms)

                curr_rss = process.memory_info().rss / (1024 * 1024)
                if curr_rss > peak_rss:
                    peak_rss = curr_rss

            total_t = time.perf_counter() - t_start
            summary = engine.finalize_session()
            final_rss = process.memory_info().rss / (1024 * 1024)
            growth = max(0.0, final_rss - init_rss)

            mean_lat = float(np.mean(latencies)) if latencies else 0.0
            p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
            eff_fps = n_frames / max(0.001, total_t)
            is_leak = growth > 30.0

            results.append(
                StressConditionResult(
                    condition_name=name,
                    resolution=(w, h),
                    target_fps=fps,
                    total_frames=n_frames,
                    effective_fps=eff_fps,
                    mean_latency_ms=mean_lat,
                    p95_latency_ms=p95_lat,
                    peak_rss_mb=peak_rss,
                    rss_growth_mb=growth,
                    has_memory_leak=is_leak,
                    status="PASS" if not is_leak else "FAIL",
                )
            )

        overall = "PASS" if all(r.status == "PASS" for r in results) else "FAIL"

        return StressTestMatrixResult(
            total_conditions_tested=len(results),
            results=results,
            overall_verdict=overall,
        )
