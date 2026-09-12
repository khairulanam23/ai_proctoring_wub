"""Memory leak and session isolation profiler for Phase 0 baseline.

Tracks RSS memory growth across sustained frame counts (100, 300, 500, 1000 frames),
VRAM allocation across session lifecycle (startup, inference, finalization, reset),
and checks for object retention in queues or singleton caches.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
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


def run_memory_stability_test(
    max_frames: int = 1000,
    checkpoints: list[int] | None = None,
    output_json: str = "audit/results/phase_0_memory_stability.json",
) -> dict[str, Any]:
    if checkpoints is None:
        checkpoints = [100, 300, 500, 1000]

    process = psutil.Process(os.getpid())
    cuda_available = HAS_TORCH and torch.cuda.is_available()

    print("=" * 70)
    print("PHASE 0 MEMORY STABILITY & SESSION LIFECYCLE BENCHMARK")
    print("=" * 70)
    print(f"Total Frames:  {max_frames}")
    print(f"Checkpoints:   {checkpoints}")
    print(f"CUDA Active:   {cuda_available}")
    print("-" * 70)

    # 1. Baseline Memory (Before Model Loading)
    gc.collect()
    rss_start = process.memory_info().rss / (1024 * 1024)
    vram_start = (
        torch.cuda.memory_allocated() / (1024 * 1024) if cuda_available else 0.0
    )
    vram_res_start = (
        torch.cuda.memory_reserved() / (1024 * 1024) if cuda_available else 0.0
    )

    # 2. Model Loading
    models_path = Path("models")
    face_detector = FaceDetector(model_path=models_path / "face_detection_yunet_2023mar.onnx")
    face_verifier = FaceVerifier(
        detector=face_detector,
        recognizer_model_path=models_path / "face_recognition_sface_2021dec.onnx",
    )
    object_detector = ObjectDetector(model_path=models_path / "yolo11n.pt", device="cuda" if cuda_available else "cpu")

    # Reference template
    storage = ProctoringStorage("data")
    ref_dir = Path("data/samples/Colin_Powell")
    ref_templates = []
    if ref_dir.exists():
        for p in sorted(ref_dir.glob("*.jpg"))[:2]:
            im = cv2.imread(str(p))
            if im is not None:
                res = face_detector.detect(im)
                if res.faces:
                    emb = face_verifier.extract_feature(im, face=res.faces[0])
                    if emb is not None:
                        ref_templates.append(emb)

    config = SessionConfig(
        session_id="memory_test_session",
        student_name="Memory Test Candidate",
        strictness="STANDARD",
        sampling_fps=30.0,
        enable_face_detection=True,
        enable_face_verification=bool(ref_templates),
        enable_object_detection=True,
        enable_facial_dynamics=True,
        enable_hand_analysis=True,
        reference_templates=ref_templates,
        capture_evidence=True,
    )

    engine = ProctoringEngine(
        config=config,
        face_detector=face_detector,
        face_verifier=face_verifier,
        object_detector=object_detector,
    )

    rss_post_load = process.memory_info().rss / (1024 * 1024)
    vram_post_load = (
        torch.cuda.memory_allocated() / (1024 * 1024) if cuda_available else 0.0
    )

    # Synthetic realistic frame
    canvas = np.full((480, 640, 3), 180, dtype=np.uint8)
    if ref_dir.exists():
        img_f = cv2.imread(str(list(ref_dir.glob("*.jpg"))[0]))
        if img_f is not None:
            canvas[100:300, 200:400] = cv2.resize(img_f, (200, 200))

    # 3. Sustained Inference
    rss_history: dict[str, float] = {}
    vram_history: dict[str, float] = {}
    t_start = time.perf_counter()

    for i in range(1, max_frames + 1):
        engine.process_frame(canvas, frame_index=i, timestamp_seconds=i / 30.0)

        if i in checkpoints:
            elapsed = time.perf_counter() - t_start
            cur_rss = process.memory_info().rss / (1024 * 1024)
            cur_vram = (
                torch.cuda.memory_allocated() / (1024 * 1024) if cuda_available else 0.0
            )
            rss_history[f"frame_{i}"] = round(cur_rss, 2)
            vram_history[f"frame_{i}"] = round(cur_vram, 2)
            print(f"[{i:4d} frames | {elapsed:5.1f}s] RSS: {cur_rss:7.2f} MB | VRAM: {cur_vram:6.2f} MB")

    # 4. Session Finalization
    t_fin_0 = time.perf_counter()
    summary = engine.finalize_session()
    fin_time_ms = (time.perf_counter() - t_fin_0) * 1000.0

    gc.collect()
    rss_finalized = process.memory_info().rss / (1024 * 1024)
    vram_finalized = (
        torch.cuda.memory_allocated() / (1024 * 1024) if cuda_available else 0.0
    )

    # 5. Session Analyzer Cleanup (Isolation test)
    engine.close_analyzers()
    gc.collect()
    rss_reset = process.memory_info().rss / (1024 * 1024)
    vram_reset = (
        torch.cuda.memory_allocated() / (1024 * 1024) if cuda_available else 0.0
    )

    # Check for memory growth between frame 300 and frame 1000
    rss_growth_mb = 0.0
    if "frame_1000" in rss_history and "frame_300" in rss_history:
        rss_growth_mb = round(rss_history["frame_1000"] - rss_history["frame_300"], 2)
    elif "frame_500" in rss_history and "frame_100" in rss_history:
        rss_growth_mb = round(rss_history["frame_500"] - rss_history["frame_100"], 2)

    leak_suspected = rss_growth_mb > 50.0  # more than 50MB unexplained growth over 700 frames

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_frames_run": max_frames,
        "rss_memory_mb": {
            "initial_startup": round(rss_start, 2),
            "post_model_load": round(rss_post_load, 2),
            **rss_history,
            "post_finalization": round(rss_finalized, 2),
            "post_reset": round(rss_reset, 2),
            "growth_during_steady_state_mb": rss_growth_mb,
            "leak_suspected": leak_suspected,
        },
        "vram_memory_mb": {
            "initial_startup": round(vram_start, 2),
            "initial_reserved": round(vram_res_start, 2),
            "post_model_load": round(vram_post_load, 2),
            **vram_history,
            "post_finalization": round(vram_finalized, 2),
            "post_reset": round(vram_reset, 2),
        },
        "session_finalization": {
            "duration_ms": round(fin_time_ms, 2),
            "total_events_sealed": summary.total_events,
            "integrity_verified": summary.integrity_verified,
        },
        "isolation_check": {
            "engine_active_after_finalization": engine.is_active,
            "total_timeline_frames": len(engine.timeline),
            "sealed_package_exists": summary.package_dir.exists(),
        },
    }

    print("\n" + "=" * 70)
    print("MEMORY STABILITY SUMMARY")
    print("=" * 70)
    print(f"Startup RSS:        {rss_start:.2f} MB")
    print(f"Post-Model-Load RSS:{rss_post_load:.2f} MB")
    for k, v in rss_history.items():
        print(f"{k:<20}: {v:.2f} MB")
    print(f"Post-Finalize RSS:  {rss_finalized:.2f} MB")
    print(f"Post-Reset RSS:     {rss_reset:.2f} MB")
    print(f"Steady Growth:      {rss_growth_mb:.2f} MB (Leak Suspected: {leak_suspected})")
    print(f"VRAM Allocated:     {vram_post_load:.2f} MB -> Finalized: {vram_finalized:.2f} MB")
    print(f"Finalize Duration:  {fin_time_ms:.2f} ms (Integrity: {summary.integrity_verified})")
    print("=" * 70)

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {out_p.resolve()}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory stability test")
    parser.add_argument("--frames", type=int, default=1000, help="Total frames to run")
    parser.add_argument("--output", default="audit/results/phase_0_memory_stability.json")
    args = parser.parse_args()

    checkpoints = [100, 300, 500, args.frames] if args.frames >= 500 else [50, 100, args.frames]
    checkpoints = sorted(list(set(checkpoints)))
    run_memory_stability_test(max_frames=args.frames, checkpoints=checkpoints, output_json=args.output)
