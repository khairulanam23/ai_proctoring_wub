"""Progressive multi-session concurrency benchmark for Phase 7 GPU evaluation.

Measures throughput (FPS per session and aggregate FPS), latency percentiles
(p50, p95, p99), CPU %, and NVIDIA GPU VRAM scaling across 1, 2, 4, and 8 sessions.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from proctoring.config import SessionConfig
from proctoring.detection.face_detector import FaceDetector
from proctoring.detection.face_verifier import FaceVerifier
from proctoring.detection.object_detector import ObjectDetector
from proctoring.engine import ProctoringEngine


@dataclass
class SessionScalingResult:
    num_sessions: int
    total_frames_processed: int
    duration_seconds: float
    aggregate_fps: float
    fps_per_session: float
    mean_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    vram_allocated_mb: float
    vram_reserved_mb: float
    cpu_percent_avg: float
    errors_count: int


def run_single_session_worker(
    session_idx: int,
    frames: list[np.ndarray],
    face_detector: FaceDetector,
    face_verifier: FaceVerifier,
    object_detector: ObjectDetector,
    reference_templates: list[np.ndarray],
) -> tuple[list[float], int]:
    """Worker simulating a continuous candidate examination stream."""
    config = SessionConfig(
        session_id=f"concurrency_worker_{session_idx}",
        student_name=f"Candidate {session_idx}",
        enable_face_detection=True,
        enable_face_verification=bool(reference_templates),
        enable_object_detection=True,
        enable_facial_dynamics=True,
        enable_hand_analysis=True,
        reference_templates=reference_templates,
        capture_evidence=False,  # Isolate inference performance from disk write overhead
    )

    engine = ProctoringEngine(
        config=config,
        face_detector=face_detector,
        face_verifier=face_verifier,
        object_detector=object_detector,
    )

    latencies_ms: list[float] = []
    errors = 0

    try:
        engine.create_session()
        for idx, frame in enumerate(frames):
            ts = idx / 30.0
            t0 = time.perf_counter()
            try:
                engine.process_frame(frame, frame_index=idx, timestamp_seconds=ts)
            except Exception:
                errors += 1
            dt = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt)
        engine.destroy_session()
    except Exception:
        errors += 1

    return latencies_ms, errors


def benchmark_concurrency(
    session_counts: list[int] = [1, 2, 4, 8],
    frames_per_session: int = 60,
    warmup_frames: int = 15,
    device: str = "cuda",
    output_json: str = "audit/results/phase_7_concurrency.json",
) -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    cuda_available = HAS_TORCH and torch.cuda.is_available()
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None

    print("=" * 70)
    print("PHASE 7 MULTI-SESSION CONCURRENCY BENCHMARK")
    print("=" * 70)
    print(f"Platform:           {platform.platform()}")
    print(f"Python:             {platform.python_version()}")
    print(f"CUDA Hardware:      {cuda_device_name}")
    print(f"Target Device:      {device}")
    print(f"Concurrency Tiers:  {session_counts}")
    print(f"Frames Per Session: {frames_per_session}")
    print("-" * 70)

    # Initialize shared model detectors using ModelRegistry
    yunet_path = Path("models/face_detection_yunet_2023mar.onnx")
    sface_path = Path("models/face_recognition_sface_2021dec.onnx")
    yolo_path = Path("models/yolo11n.pt")

    face_detector = FaceDetector(model_path=yunet_path, device=device)
    face_verifier = FaceVerifier(
        detector=face_detector,
        recognizer_model_path=sface_path,
        device=device,
    )
    object_detector = ObjectDetector(model_path=yolo_path, device=device)

    # Build reference templates
    enrollment_dir = Path("data/samples/Colin_Powell")
    reference_templates: list[np.ndarray] = []
    if enrollment_dir.exists():
        for p in sorted(enrollment_dir.glob("*.jpg"))[:2]:
            img = cv2.imread(str(p))
            if img is not None:
                faces = face_detector.detect(img)
                if faces.faces:
                    emb = face_verifier.extract_feature(img, face=faces.faces[0])
                    reference_templates.append(emb)

    # Generate synthetic input stream
    test_face_p = Path("data/samples/Colin_Powell/Colin_Powell_0001.jpg")
    base_canvas = np.full((480, 640, 3), 200, dtype=np.uint8)
    if test_face_p.exists():
        face_img = cv2.imread(str(test_face_p))
        if face_img is not None:
            face_resized = cv2.resize(face_img, (200, 200))
            base_canvas[100:300, 220:420] = face_resized

    test_frames = [base_canvas.copy() for _ in range(frames_per_session)]

    results: list[dict[str, Any]] = []

    for num_sessions in session_counts:
        print(f"\nEvaluating Concurrency Tier: {num_sessions} Session(s)...")

        # Warmup pass
        if warmup_frames > 0:
            for i in range(warmup_frames):
                face_detector.detect(test_frames[0])

        if cuda_available:
            torch.cuda.empty_cache()

        cpu_samples: list[float] = []
        t_tier_start = time.perf_counter()

        all_latencies: list[float] = []
        total_errors = 0

        # Execute concurrent sessions in thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_sessions) as executor:
            futures = [
                executor.submit(
                    run_single_session_worker,
                    sess_idx,
                    test_frames,
                    face_detector,
                    face_verifier,
                    object_detector,
                    reference_templates,
                )
                for sess_idx in range(num_sessions)
            ]

            for fut in concurrent.futures.as_completed(futures):
                try:
                    lats, errs = fut.result()
                    all_latencies.extend(lats)
                    total_errors += errs
                except Exception as e:
                    print(f"Session execution error: {e}")
                    total_errors += 1

                cpu_samples.append(process.cpu_percent(interval=None))

        tier_duration = time.perf_counter() - t_tier_start
        total_frames = len(all_latencies)
        agg_fps = round(total_frames / tier_duration, 2) if tier_duration > 0 else 0.0
        fps_per_sess = round(agg_fps / num_sessions, 2) if num_sessions > 0 else 0.0

        lat_arr = np.array(all_latencies) if all_latencies else np.array([0.0])
        vram_alloc = (
            round(torch.cuda.memory_allocated() / (1024 * 1024), 2) if cuda_available else 0.0
        )
        vram_res = (
            round(torch.cuda.memory_reserved() / (1024 * 1024), 2) if cuda_available else 0.0
        )
        avg_cpu = round(float(np.mean(cpu_samples)), 1) if cpu_samples else 0.0

        res = SessionScalingResult(
            num_sessions=num_sessions,
            total_frames_processed=total_frames,
            duration_seconds=round(tier_duration, 2),
            aggregate_fps=agg_fps,
            fps_per_session=fps_per_sess,
            mean_latency_ms=round(float(np.mean(lat_arr)), 2),
            p50_latency_ms=round(float(np.median(lat_arr)), 2),
            p90_latency_ms=round(float(np.percentile(lat_arr, 90)), 2),
            p95_latency_ms=round(float(np.percentile(lat_arr, 95)), 2),
            p99_latency_ms=round(float(np.percentile(lat_arr, 99)), 2),
            vram_allocated_mb=vram_alloc,
            vram_reserved_mb=vram_res,
            cpu_percent_avg=avg_cpu,
            errors_count=total_errors,
        )
        results.append(asdict(res))

        print(
            f"  Aggregate FPS: {agg_fps:>6.2f} | FPS/Sess: {fps_per_sess:>5.2f} | "
            f"Mean Lat: {res.mean_latency_ms:>6.2f} ms | p95: {res.p95_latency_ms:>6.2f} ms | "
            f"VRAM: {vram_alloc:>6.2f} MB | CPU: {avg_cpu:>5.1f}%"
        )

    # Print summary table
    print("\n" + "=" * 85)
    print("CONCURRENCY SCALING SUMMARY (RTX 3060 12GB)")
    print("=" * 85)
    print(
        f"{'Sessions':<10} | {'Agg FPS':>9} | {'FPS/Sess':>9} | {'Mean (ms)':>10} | {'p95 (ms)':>9} | {'VRAM (MB)':>10} | {'Errors':>6}"
    )
    print("-" * 85)
    for r in results:
        print(
            f"{r['num_sessions']:<10} | {r['aggregate_fps']:>9.2f} | {r['fps_per_session']:>9.2f} | "
            f"{r['mean_latency_ms']:>10.2f} | {r['p95_latency_ms']:>9.2f} | {r['vram_allocated_mb']:>10.2f} | {r['errors_count']:>6}"
        )
    print("=" * 85)

    final_report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cuda_device": cuda_device_name,
            "target_device": device,
        },
        "concurrency_scaling": results,
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nReport saved to: {out_p.resolve()}")
    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7 Concurrency Scaling Benchmark")
    parser.add_argument("--sessions", default="1,2,4,8", help="Comma-separated session counts")
    parser.add_argument("--frames", type=int, default=60, help="Frames per session")
    parser.add_argument("--device", default="cuda", help="Target device")
    parser.add_argument("--output", default="audit/results/phase_7_concurrency.json", help="Output JSON path")
    args = parser.parse_args()

    session_counts = [int(s.strip()) for s in args.sessions.split(",") if s.strip()]
    benchmark_concurrency(
        session_counts=session_counts,
        frames_per_session=args.frames,
        device=args.device,
        output_json=args.output,
    )
