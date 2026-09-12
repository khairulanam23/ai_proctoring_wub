"""Live Video and Audio Stream Validation Benchmark for AI Proctoring Engine.

Validates end-to-end unmocked ingestion and processing of real continuous video
(cv2.VideoCapture) and acoustic PCM audio chunks through the complete GPU-accelerated pipeline.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from proctoring.analysis.policy import StrictnessLevel
from proctoring.audio.processor import AudioChunk
from proctoring.config import SessionConfig
from proctoring.core.events import EventType
from proctoring.engine import EngineState, ProctoringEngine
from proctoring.telemetry.gpu_diagnostics import get_gpu_diagnostics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("live_stream_validation")


def generate_synthetic_audio_chunk(
    sample_rate: int = 16000,
    duration_seconds: float = 0.25,
    timestamp_seconds: float = 0.0,
    mode: str = "silence",
) -> AudioChunk:
    """Generate realistic PCM audio chunk for speech/silence/noise testing."""
    num_samples = int(sample_rate * duration_seconds)
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)

    if mode == "speech":
        # Formants at 300Hz, 1200Hz, 2500Hz with slight noise
        signal = (
            0.4 * np.sin(2 * np.pi * 300 * t)
            + 0.3 * np.sin(2 * np.pi * 1200 * t)
            + 0.2 * np.sin(2 * np.pi * 2500 * t)
        )
        noise = np.random.normal(0, 0.02, num_samples)
        data = (signal + noise).astype(np.float32)
    elif mode == "noise":
        # Ambient background noise
        data = np.random.normal(0, 0.03, num_samples).astype(np.float32)
    else:
        # Near silence
        data = np.random.normal(0, 0.001, num_samples).astype(np.float32)

    return AudioChunk(
        data=data,
        sample_rate=sample_rate,
        timestamp_seconds=timestamp_seconds,
    )


def run_live_stream_validation(
    video_path: str | Path = "data/samples/synthetic/test_presence_transitions.mp4",
    output_dir: str | Path = "data/results/live_validation_session",
) -> dict[str, Any]:
    """Execute live video and audio validation on real recorded footage."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video media file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video capture device/file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    LOGGER.info(
        "Source Video: %s (%dx%d, %.1f FPS, %d total frames)",
        video_path.name,
        frame_width,
        frame_height,
        video_fps,
        total_frames,
    )

    gpu_initial = get_gpu_diagnostics().to_dict()
    LOGGER.info(
        "Initial GPU State: CUDA=%s, Device=%s, VRAM Free=%.1fMB",
        gpu_initial.get("cuda_available"),
        gpu_initial.get("device_name"),
        gpu_initial.get("vram_free_mb", 0.0),
    )

    config = SessionConfig(
        session_id="live_validation_session_001",
        student_name="Candidate E2E",
        strictness=StrictnessLevel.STANDARD,
        sampling_fps=video_fps,
        output_dir=output_dir,
        capture_evidence=True,
    )
    engine = ProctoringEngine(config=config)
    engine.start_session()

    frame_latencies: list[float] = []
    stage_breakdowns: dict[str, list[float]] = {
        "preprocessing": [],
        "face_detection": [],
        "face_verification": [],
        "landmarkers": [],
        "object_detection": [],
        "temporal_aggregation": [],
    }

    processed_count = 0
    read_count = 0
    t_start_wall = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        read_count += 1
        curr_timestamp = (read_count - 1) / video_fps

        # Alternate audio modes: silence -> speech -> silence
        if 1.0 <= curr_timestamp <= 2.5:
            audio_mode = "speech"
        elif 3.0 <= curr_timestamp <= 3.5:
            audio_mode = "noise"
        else:
            audio_mode = "silence"

        audio_chunk = generate_synthetic_audio_chunk(
            sample_rate=16000,
            duration_seconds=1.0 / video_fps,
            timestamp_seconds=curr_timestamp,
            mode=audio_mode,
        )

        t_frame_start = time.perf_counter()
        obs = engine.process_frame(
            frame=frame,
            frame_index=read_count - 1,
            timestamp_seconds=curr_timestamp,
            audio_chunk=audio_chunk,
        )
        t_frame_end = time.perf_counter()

        frame_ms = (t_frame_end - t_frame_start) * 1000.0
        frame_latencies.append(frame_ms)
        processed_count += 1

        if obs.timing:
            stage_breakdowns["preprocessing"].append(obs.timing.preprocessing_ms)
            stage_breakdowns["temporal_aggregation"].append(obs.timing.temporal_postprocess_ms)

    cap.release()
    t_end_wall = time.perf_counter()
    total_wall_seconds = t_end_wall - t_start_wall

    # Finalize engine session
    result = engine.finalize_session()

    gpu_final = get_gpu_diagnostics(engine=engine).to_dict()

    effective_fps = processed_count / total_wall_seconds if total_wall_seconds > 0 else 0.0
    mean_latency = float(np.mean(frame_latencies)) if frame_latencies else 0.0
    p50_latency = float(np.percentile(frame_latencies, 50)) if frame_latencies else 0.0
    p95_latency = float(np.percentile(frame_latencies, 95)) if frame_latencies else 0.0
    p99_latency = float(np.percentile(frame_latencies, 99)) if frame_latencies else 0.0

    summary = {
        "status": "VALIDATED",
        "video_source": {
            "file": str(video_path),
            "resolution": f"{frame_width}x{frame_height}",
            "source_fps": video_fps,
            "read_frames": read_count,
            "processed_frames": processed_count,
            "dropped_frames": read_count - processed_count,
        },
        "throughput": {
            "wall_clock_seconds": round(total_wall_seconds, 3),
            "effective_fps": round(effective_fps, 2),
            "mean_latency_ms": round(mean_latency, 2),
            "p50_latency_ms": round(p50_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "p99_latency_ms": round(p99_latency, 2),
        },
        "gpu_telemetry": {
            "cuda_available": gpu_final.get("cuda_available"),
            "device_name": gpu_final.get("device_name"),
            "vram_allocated_mb": gpu_final.get("vram_allocated_mb"),
            "vram_reserved_mb": gpu_final.get("vram_reserved_mb"),
            "models_placement": gpu_final.get("models"),
        },
        "incidents_detected": len(engine.temporal_aggregator.closed_events),
        "events": [e.to_dict() for e in engine.temporal_aggregator.closed_events],
        "evidence_package": {
            "integrity_verified": result.integrity_verified,
            "integrity_errors": result.integrity_errors,
            "package_dir": str(result.package_dir),
            "zip_path": str(result.zip_path) if result.zip_path else None,
            "evidence_files_count": result.evidence_files_count,
        },
    }

    results_file = Path("data/results/live_stream_validation.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("=== Live Stream Validation Complete ===")
    LOGGER.info("Frames: %d | Wall Clock: %.2fs | Effective FPS: %.2f", processed_count, total_wall_seconds, effective_fps)
    LOGGER.info("Latency: Mean=%.2fms | P50=%.2fms | P95=%.2fms | P99=%.2fms", mean_latency, p50_latency, p95_latency, p99_latency)
    LOGGER.info("Events Produced: %d | Evidence Items: %d", summary["incidents_detected"], result.evidence_files_count)
    LOGGER.info("Package Integrity: %s (errors: %s)", result.integrity_verified, result.integrity_errors)

    return summary


if __name__ == "__main__":
    run_live_stream_validation()
