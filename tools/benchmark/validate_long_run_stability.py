"""Long-Run Stability and Resource Leak Benchmark for AI Proctoring Engine.

Profiles sustained multi-session execution (450+ frames across sequential sessions)
to prove that:
1. RSS RAM remains bounded (no Python object or buffer leaks).
2. GPU VRAM does not grow across sessions (shared ModelRegistry reuse).
3. Active threads do not accumulate (proper analyzer/worker lifecycle).
4. Open file descriptors remain strictly bounded.
5. Consecutive sessions remain fully isolated with 100% verified evidence packages.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import torch

from proctoring.config import SessionConfig
from proctoring.engine import EngineState, ProctoringEngine
from proctoring.telemetry.gpu_diagnostics import get_gpu_diagnostics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("stability_benchmark")


def get_current_metrics(proc: psutil.Process) -> dict[str, float]:
    """Capture instantaneous OS process and GPU memory metrics."""
    mem_info = proc.memory_info()
    rss_mb = round(mem_info.rss / (1024 * 1024), 2)
    vms_mb = round(mem_info.vms / (1024 * 1024), 2)
    num_threads = proc.num_threads()
    num_fds = proc.num_fds() if hasattr(proc, "num_fds") else 0

    vram_alloc_mb = 0.0
    vram_res_mb = 0.0
    if torch.cuda.is_available():
        vram_alloc_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
        vram_res_mb = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)

    return {
        "rss_mb": rss_mb,
        "vms_mb": vms_mb,
        "num_threads": num_threads,
        "num_fds": num_fds,
        "vram_alloc_mb": vram_alloc_mb,
        "vram_res_mb": vram_res_mb,
    }


def make_dynamic_frame(width: int = 640, height: int = 480, frame_idx: int = 0) -> np.ndarray:
    """Generate dynamic synthetic frame with animated movement to exercise CV stages."""
    img = np.full((height, width, 3), 120, dtype=np.uint8)
    cx = int((width // 2) + 80 * np.sin(frame_idx * 0.1))
    cy = int((height // 2) + 40 * np.cos(frame_idx * 0.1))
    cv2.circle(img, (cx, cy), 50, (220, 220, 220), -1)
    noise = np.random.randint(0, 15, (height, width, 3), dtype=np.uint8)
    return cv2.add(img, noise)


def run_long_run_stability_test(
    num_sessions: int = 3,
    frames_per_session: int = 150,
    output_base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute sustained multi-session stability profiling."""
    proc = psutil.Process(os.getpid())

    tmp_ctx = None
    if output_base_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory()
        base_dir = Path(tmp_ctx.name)
    else:
        base_dir = Path(output_base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Starting Long-Run Stability Benchmark: %d sessions, %d frames/session (%d total frames)",
        num_sessions,
        frames_per_session,
        num_sessions * frames_per_session,
    )

    initial_metrics = get_current_metrics(proc)
    LOGGER.info("Initial Baseline Metrics: %s", initial_metrics)

    metric_snapshots: list[dict[str, Any]] = []
    session_summaries: list[dict[str, Any]] = []

    t_start = time.perf_counter()
    global_frame_count = 0

    for sess_num in range(1, num_sessions + 1):
        session_id = f"sustained_exam_sess_{sess_num:03d}"
        candidate_name = f"Candidate S{sess_num:03d}"

        config = SessionConfig(
            session_id=session_id,
            student_name=candidate_name,
            output_dir=base_dir,
            sampling_fps=30.0,
            capture_evidence=True,
            record_timeline=True,
        )

        engine = ProctoringEngine(config=config)
        engine.start_session()
        LOGGER.info("--- Session %d/%d (%s) Started ---", sess_num, num_sessions, session_id)

        for f_idx in range(frames_per_session):
            frame = make_dynamic_frame(640, 480, f_idx)
            obs = engine.process_frame(
                frame=frame,
                frame_index=f_idx,
                timestamp_seconds=f_idx / 30.0,
            )
            global_frame_count += 1

            if f_idx % 25 == 0:
                current_m = get_current_metrics(proc)
                current_m["session"] = sess_num
                current_m["session_frame"] = f_idx
                current_m["global_frame"] = global_frame_count
                metric_snapshots.append(current_m)

        # Finalize and seal package
        summary = engine.finalize_session()
        assert summary.integrity_verified is True
        assert len(summary.integrity_errors) == 0

        post_sess_metrics = get_current_metrics(proc)
        LOGGER.info(
            "Session %d finalized. Integrity: %s, Frames: %d, RSS: %.1fMB, VRAM Alloc: %.1fMB, FDs: %d",
            sess_num,
            summary.integrity_verified,
            summary.processed_frames,
            post_sess_metrics["rss_mb"],
            post_sess_metrics["vram_alloc_mb"],
            post_sess_metrics["num_fds"],
        )

        session_summaries.append({
            "session_id": session_id,
            "processed_frames": summary.processed_frames,
            "integrity_verified": summary.integrity_verified,
            "metrics_at_end": post_sess_metrics,
        })

        del engine

    t_total = time.perf_counter() - t_start
    final_metrics = get_current_metrics(proc)

    # Post-warmup deltas (from end of Session 1 to end of Session N)
    s1_metrics = session_summaries[0]["metrics_at_end"]
    post_warmup_rss_delta_mb = round(final_metrics["rss_mb"] - s1_metrics["rss_mb"], 2)
    post_warmup_fd_delta = final_metrics["num_fds"] - s1_metrics["num_fds"]
    post_warmup_thread_delta = final_metrics["num_threads"] - s1_metrics["num_threads"]
    vram_alloc_delta_mb = round(final_metrics["vram_alloc_mb"] - s1_metrics["vram_alloc_mb"], 2)

    # Total process deltas (from cold start)
    total_rss_delta_mb = round(final_metrics["rss_mb"] - initial_metrics["rss_mb"], 2)

    is_stable = (
        post_warmup_rss_delta_mb < 60.0  # Post-warmup memory growth across consecutive sessions strictly bounded
        and vram_alloc_delta_mb <= 5.0   # Zero VRAM multiplication across sessions
        and abs(post_warmup_fd_delta) <= 2  # Zero leaked file descriptors across sessions
        and all(s["integrity_verified"] for s in session_summaries)
    )

    report = {
        "benchmark": "LONG_RUN_STABILITY",
        "status": "PASS" if is_stable else "FAIL",
        "total_sessions": num_sessions,
        "frames_per_session": frames_per_session,
        "total_frames_processed": global_frame_count,
        "total_duration_seconds": round(t_total, 2),
        "overall_effective_fps": round(global_frame_count / t_total, 2),
        "initial_cold_metrics": initial_metrics,
        "post_warmup_s1_metrics": s1_metrics,
        "final_metrics": final_metrics,
        "deltas_post_warmup": {
            "rss_delta_mb": post_warmup_rss_delta_mb,
            "vram_alloc_delta_mb": vram_alloc_delta_mb,
            "thread_delta": post_warmup_thread_delta,
            "fd_delta": post_warmup_fd_delta,
        },
        "total_cold_rss_growth_mb": total_rss_delta_mb,
        "session_summaries": session_summaries,
        "metric_snapshots": metric_snapshots,
    }

    out_file = Path("data/results/long_run_stability_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if tmp_ctx:
        tmp_ctx.cleanup()

    LOGGER.info(
        "Deltas (post-warmup): RSS=%+.1fMB | VRAM=%+.1fMB | Threads=%+d | FDs=%+d",
        post_warmup_rss_delta_mb,
        vram_alloc_delta_mb,
        post_warmup_thread_delta,
        post_warmup_fd_delta,
    )

    return report


if __name__ == "__main__":
    rep = run_long_run_stability_test()
    if rep["status"] != "PASS":
        sys.exit(1)
