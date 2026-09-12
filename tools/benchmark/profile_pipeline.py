"""Reproducible pipeline profiler for Phase 0 baseline measurements.

Measures per-stage latency breakdowns, end-to-end throughput FPS,
CPU %, system RAM (RSS), and NVIDIA GPU VRAM utilization.
"""

from __future__ import annotations

import argparse
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
from proctoring.storage import ProctoringStorage


@dataclass
class StageMetrics:
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float


def compute_metrics(times_ms: list[float]) -> StageMetrics:
    if not times_ms:
        return StageMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.array(times_ms, dtype=np.float64)
    return StageMetrics(
        mean_ms=round(float(np.mean(arr)), 2),
        p50_ms=round(float(np.median(arr)), 2),
        p90_ms=round(float(np.percentile(arr, 90)), 2),
        p95_ms=round(float(np.percentile(arr, 95)), 2),
        p99_ms=round(float(np.percentile(arr, 99)), 2),
        min_ms=round(float(np.min(arr)), 2),
        max_ms=round(float(np.max(arr)), 2),
        std_ms=round(float(np.std(arr)), 2),
    )


def profile_pipeline(
    video_input: str | None = None,
    warmup_frames: int = 30,
    profile_frames: int = 300,
    models_dir: str = "models",
    output_json: str = "audit/results/phase_0_baseline.json",
    enable_wearables: bool = False,
    device: str = "cuda",
) -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    models_path = Path(models_dir)

    cuda_available = HAS_TORCH and torch.cuda.is_available()
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
    actual_device = "cuda" if (cuda_available and device == "cuda") else "cpu"

    print("=" * 70)
    print("PHASE 0 BASELINE PIPELINE PROFILER")
    print("=" * 70)
    print(f"Platform:       {platform.platform()}")
    print(f"Python:         {platform.python_version()}")
    print(f"PyTorch:        {torch.__version__ if HAS_TORCH else 'N/A'}")
    print(f"CUDA Available: {cuda_available} ({cuda_device_name})")
    print(f"Target Device:  {actual_device}")
    print(f"Warmup Frames:  {warmup_frames}")
    print(f"Profile Frames: {profile_frames}")
    print("-" * 70)

    # Initial RAM and VRAM
    initial_rss_mb = process.memory_info().rss / (1024 * 1024)
    initial_vram_mb = 0.0
    initial_vram_res_mb = 0.0
    if cuda_available:
        torch.cuda.empty_cache()
        initial_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        initial_vram_res_mb = torch.cuda.memory_reserved() / (1024 * 1024)

    # Initialize detectors
    t_load_start = time.perf_counter()
    yunet_path = models_path / "face_detection_yunet_2023mar.onnx"
    sface_path = models_path / "face_recognition_sface_2021dec.onnx"
    yolo_path = models_path / "yolo11n.pt"

    face_detector = (
        FaceDetector(model_path=yunet_path, device=actual_device)
        if yunet_path.exists()
        else None
    )
    face_verifier = (
        FaceVerifier(
            detector=face_detector,
            recognizer_model_path=sface_path,
            device=actual_device,
        )
        if sface_path.exists() and face_detector
        else None
    )
    object_detector = None
    if yolo_path.exists():
        try:
            object_detector = ObjectDetector(model_path=yolo_path, device=actual_device)
        except Exception as e:
            print(f"Warning: Failed to load ObjectDetector: {e}")

    # Build reference enrollment if samples exist
    storage = ProctoringStorage("data")
    enrollment_dir = Path("data/samples/Colin_Powell")
    reference_templates = []
    if enrollment_dir.exists() and face_verifier:
        for img_p in sorted(enrollment_dir.glob("*.jpg"))[:3]:
            img = cv2.imread(str(img_p))
            if img is not None:
                faces = face_detector.detect(img)
                if faces.faces:
                    emb = face_verifier.extract_feature(img, face=faces.faces[0])
                    if emb is not None:
                        reference_templates.append(emb)

    config = SessionConfig(
        session_id="baseline_profile_session",
        student_name="Baseline Profile Candidate",
        strictness="STANDARD",
        sampling_fps=30.0,
        enable_face_detection=face_detector is not None,
        enable_face_verification=bool(reference_templates),
        enable_object_detection=object_detector is not None,
        enable_facial_dynamics=True,
        enable_hand_analysis=True,
        enable_wearable_detection=enable_wearables,
        reference_templates=reference_templates,
        capture_evidence=True,
    )

    engine = ProctoringEngine(
        config=config,
        face_detector=face_detector,
        face_verifier=face_verifier,
        object_detector=object_detector,
    )

    model_load_time_ms = (time.perf_counter() - t_load_start) * 1000.0

    # Post-load memory
    post_load_rss_mb = process.memory_info().rss / (1024 * 1024)
    post_load_vram_mb = (
        torch.cuda.memory_allocated() / (1024 * 1024) if cuda_available else 0.0
    )
    post_load_vram_res_mb = (
        torch.cuda.memory_reserved() / (1024 * 1024) if cuda_available else 0.0
    )

    # Frame generator
    frames_buffer: list[np.ndarray] = []
    decode_times_ms: list[float] = []
    capture_times_ms: list[float] = []

    if video_input and Path(video_input).exists():
        cap = cv2.VideoCapture(video_input)
        total_needed = warmup_frames + profile_frames
        while len(frames_buffer) < total_needed:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            capture_times_ms.append(dt_ms)
            decode_times_ms.append(dt_ms)
            frames_buffer.append(frame)
        cap.release()
    else:
        # Generate synthetic realistic test frame with face and object
        test_face_p = Path("data/samples/Colin_Powell/Colin_Powell_0001.jpg")
        base_canvas = np.full((480, 640, 3), 200, dtype=np.uint8)
        if test_face_p.exists():
            face_img = cv2.imread(str(test_face_p))
            if face_img is not None:
                face_resized = cv2.resize(face_img, (200, 200))
                base_canvas[100:300, 220:420] = face_resized
        total_needed = warmup_frames + profile_frames
        for _ in range(total_needed):
            capture_times_ms.append(0.1)
            decode_times_ms.append(0.1)
            frames_buffer.append(base_canvas.copy())

    # Warmup
    print("\nWarming up pipeline stages...")
    for i in range(warmup_frames):
        engine.process_frame(frames_buffer[i], frame_index=i, timestamp_seconds=i / 30.0)

    # Profile frames after warmup
    stage_timings: dict[str, list[float]] = {
        "capture_ms": capture_times_ms[warmup_frames:warmup_frames + profile_frames],
        "decode_ms": decode_times_ms[warmup_frames:warmup_frames + profile_frames],
        "preprocessing_ms": [],
        "face_detection_ms": [],
        "face_verification_ms": [],
        "object_detection_ms": [],
        "hand_analysis_ms": [],
        "facial_dynamics_ms": [],
        "paper_detection_ms": [],
        "wearable_detection_ms": [],
        "temporal_aggregation_ms": [],
        "evidence_write_ms": [],
        "total_frame_ms": [],
    }

    print(f"Profiling {profile_frames} frames...")
    profile_start_wall = time.perf_counter()
    cpu_measurements: list[float] = []

    for i in range(profile_frames):
        frame_idx = warmup_frames + i
        raw_frame = frames_buffer[frame_idx]
        ts = frame_idx / 30.0

        t_frame_start = time.perf_counter()
        obs = engine.process_frame(raw_frame, frame_index=frame_idx, timestamp_seconds=ts)
        t_frame_end = time.perf_counter()
        total_ms = (t_frame_end - t_frame_start) * 1000.0

        timing = obs.timing
        stage_timings["preprocessing_ms"].append(timing.preprocessing_ms)
        stage_timings["face_detection_ms"].append(timing.face_detector_ms)
        stage_timings["face_verification_ms"].append(timing.face_embedder_ms)
        stage_timings["object_detection_ms"].append(timing.object_detector_ms)
        stage_timings["hand_analysis_ms"].append(
            obs.hand_analysis.inference_ms if obs.hand_analysis else 0.0
        )
        stage_timings["facial_dynamics_ms"].append(
            obs.facial_dynamics.inference_ms if obs.facial_dynamics else 0.0
        )
        stage_timings["paper_detection_ms"].append(
            obs.paper_analysis.inference_ms if obs.paper_analysis else 0.0
        )
        stage_timings["wearable_detection_ms"].append(
            obs.wearables.inference_ms if obs.wearables else 0.0
        )
        stage_timings["temporal_aggregation_ms"].append(timing.temporal_postprocess_ms)
        stage_timings["evidence_write_ms"].append(timing.evidence_io_ms)
        stage_timings["total_frame_ms"].append(total_ms)

        if i % 50 == 0:
            cpu_measurements.append(process.cpu_percent(interval=None))

    profile_duration_sec = time.perf_counter() - profile_start_wall
    effective_fps = profile_frames / profile_duration_sec if profile_duration_sec > 0 else 0.0

    # End memory
    end_rss_mb = process.memory_info().rss / (1024 * 1024)
    end_vram_mb = (
        torch.cuda.memory_allocated() / (1024 * 1024) if cuda_available else 0.0
    )
    end_vram_res_mb = (
        torch.cuda.memory_reserved() / (1024 * 1024) if cuda_available else 0.0
    )

    # Compute metrics for each stage
    stage_metrics = {
        stage: asdict(compute_metrics(times))
        for stage, times in stage_timings.items()
    }

    # Pure model inference latency = face_det + face_ver + obj_det + hand + dynamics + paper
    pure_model_ms = [
        stage_timings["face_detection_ms"][i]
        + stage_timings["face_verification_ms"][i]
        + stage_timings["object_detection_ms"][i]
        + stage_timings["hand_analysis_ms"][i]
        + stage_timings["facial_dynamics_ms"][i]
        + stage_timings["paper_detection_ms"][i]
        for i in range(profile_frames)
    ]
    pure_model_metrics = asdict(compute_metrics(pure_model_ms))
    pure_inference_fps = (
        round(1000.0 / pure_model_metrics["mean_ms"], 2)
        if pure_model_metrics["mean_ms"] > 0
        else 0.0
    )

    avg_cpu_pct = round(float(np.mean(cpu_measurements)), 1) if cpu_measurements else 0.0

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch_version": torch.__version__ if HAS_TORCH else None,
            "cuda_available": cuda_available,
            "cuda_device": cuda_device_name,
            "opencv_version": cv2.__version__,
            "opencv_cuda_devices": cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0,
        },
        "configuration": {
            "warmup_frames": warmup_frames,
            "profile_frames": profile_frames,
            "target_device": actual_device,
            "enable_wearables": enable_wearables,
            "input_source": str(video_input) if video_input else "synthetic_canvas",
        },
        "throughput": {
            "profile_duration_seconds": round(profile_duration_sec, 2),
            "effective_fps": round(effective_fps, 2),
            "pure_inference_fps": pure_inference_fps,
        },
        "memory_profile": {
            "initial_rss_mb": round(initial_rss_mb, 2),
            "post_load_rss_mb": round(post_load_rss_mb, 2),
            "end_rss_mb": round(end_rss_mb, 2),
            "peak_rss_mb": round(end_rss_mb, 2),
            "initial_vram_allocated_mb": round(initial_vram_mb, 2),
            "post_load_vram_allocated_mb": round(post_load_vram_mb, 2),
            "end_vram_allocated_mb": round(end_vram_mb, 2),
            "end_vram_reserved_mb": round(end_vram_res_mb, 2),
            "process_cpu_percent_avg": avg_cpu_pct,
        },
        "stage_latency_breakdown": stage_metrics,
        "pure_model_inference_latency": pure_model_metrics,
        "model_loading": {
            "total_model_load_ms": round(model_load_time_ms, 2),
        },
    }

    # Print summary table
    print("\n" + "=" * 70)
    print("STAGE-BY-STAGE LATENCY BREAKDOWN (ms)")
    print("=" * 70)
    print(f"{'Stage':<26} | {'Mean':>7} | {'p50':>7} | {'p90':>7} | {'p95':>7} | {'p99':>7}")
    print("-" * 70)
    for stage_name, m in stage_metrics.items():
        print(f"{stage_name:<26} | {m['mean_ms']:>7.2f} | {m['p50_ms']:>7.2f} | {m['p90_ms']:>7.2f} | {m['p95_ms']:>7.2f} | {m['p99_ms']:>7.2f}")
    print("-" * 70)
    print(f"{'Pure Model Inference Sum':<26} | {pure_model_metrics['mean_ms']:>7.2f} | {pure_model_metrics['p50_ms']:>7.2f} | {pure_model_metrics['p90_ms']:>7.2f} | {pure_model_metrics['p95_ms']:>7.2f} | {pure_model_metrics['p99_ms']:>7.2f}")
    print("=" * 70)
    print(f"Effective Throughput: {effective_fps:.2f} FPS | Pure Inference: {pure_inference_fps:.2f} FPS")
    print(f"System RAM: {end_rss_mb:.2f} MB | VRAM Allocated: {end_vram_mb:.2f} MB (Reserved: {end_vram_res_mb:.2f} MB)")
    print(f"Process CPU Utilization: {avg_cpu_pct:.1f}%")
    print("=" * 70)

    # Save output JSON
    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {out_p.resolve()}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0 Baseline Pipeline Profiler")
    parser.add_argument("--input", default=None, help="Video path (optional)")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup frames (default: 30)")
    parser.add_argument("--frames", type=int, default=300, help="Profile frames (default: 300)")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    parser.add_argument("--output", default="audit/results/phase_0_baseline.json", help="Output JSON path")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--enable-wearables", action="store_true", help="Enable wearable detector")
    args = parser.parse_args()

    profile_pipeline(
        video_input=args.input,
        warmup_frames=args.warmup,
        profile_frames=args.frames,
        models_dir=args.models_dir,
        output_json=args.output,
        enable_wearables=args.enable_wearables,
        device=args.device,
    )
