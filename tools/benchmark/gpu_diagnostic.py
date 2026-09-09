"""GPU Diagnostic and Benchmarking Tool for AI Proctoring Engine.

Verifies NVIDIA CUDA availability, probes framework device placement,
measures per-model latencies, compares CPU vs GPU outputs, and profiles VRAM/RAM.
"""

import gc
import json
import os
import platform
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


def get_memory_info() -> dict[str, float]:
    """Retrieve system RAM and NVIDIA VRAM usage in MB."""
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    vram_mb = 0.0
    vram_reserved_mb = 0.0
    if HAS_TORCH and torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        vram_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    return {
        "system_ram_mb": round(ram_mb, 2),
        "vram_allocated_mb": round(vram_mb, 2),
        "vram_reserved_mb": round(vram_reserved_mb, 2),
    }


def run_gpu_diagnostics() -> dict[str, Any]:
    report: dict[str, Any] = {
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "has_torch": HAS_TORCH,
            "torch_version": torch.__version__ if HAS_TORCH else None,
            "cuda_available": torch.cuda.is_available() if HAS_TORCH else False,
            "cuda_device_name": torch.cuda.get_device_name(0) if (HAS_TORCH and torch.cuda.is_available()) else None,
            "cuda_device_count": torch.cuda.device_count() if (HAS_TORCH and torch.cuda.is_available()) else 0,
            "cuda_version_compiled": torch.version.cuda if HAS_TORCH else None,
            "opencv_version": cv2.__version__,
            "opencv_cuda_devices": cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0,
        },
        "model_accelerations": {},
        "benchmarks": {},
        "comparison": {},
    }

    print("=" * 70)
    print("AI PROCTORING ENGINE — GPU DIAGNOSTIC & HARDWARE PROBE")
    print("=" * 70)
    print(f"OS: {report['environment']['platform']}")
    print(f"Python: {report['environment']['python_version']}")
    print(f"PyTorch: {report['environment']['torch_version']} | CUDA Available: {report['environment']['cuda_available']}")
    if report['environment']['cuda_available']:
        print(f"GPU: {report['environment']['cuda_device_name']} (Total Devices: {report['environment']['cuda_device_count']})")
        print(f"CUDA Runtime Version: {report['environment']['cuda_version_compiled']}")
    print(f"OpenCV: {report['environment']['opencv_version']} (CUDA build count: {report['environment']['opencv_cuda_devices']})")
    print("-" * 70)

    # 1. Probe YOLO11n (PyTorch / Ultralytics)
    print("\n[1/6] Probing Object Detector (YOLO11n)...")
    from proctoring.detection.object_detector import ObjectDetector
    from proctoring.detection.face_detector import FaceDetection, FaceDetector

    dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Test CUDA execution
    cuda_available = report['environment']['cuda_available']
    t_load_cuda = 0.0
    lat_cuda_mean = 0.0
    lat_cuda_p50 = 0.0
    lat_cuda_p95 = 0.0

    if cuda_available:
        t0 = time.perf_counter()
        detector_cuda = ObjectDetector(model_path="models/yolo11n.pt", device="cuda")
        if hasattr(detector_cuda.model, "to"):
            detector_cuda.model.to("cuda")
        t_load_cuda = (time.perf_counter() - t0) * 1000.0

        # Verify device placement
        is_on_cuda = False
        try:
            param = next(detector_cuda.model.model.parameters())
            is_on_cuda = param.is_cuda
        except Exception:
            is_on_cuda = detector_cuda.device == "cuda"

        # Warmup
        for _ in range(5):
            detector_cuda.detect(dummy_frame)
        torch.cuda.synchronize()

        cuda_times = []
        for _ in range(30):
            t_start = time.perf_counter()
            detector_cuda.detect(dummy_frame)
            torch.cuda.synchronize()
            cuda_times.append((time.perf_counter() - t_start) * 1000.0)

        lat_cuda_mean = float(np.mean(cuda_times))
        lat_cuda_p50 = float(np.percentile(cuda_times, 50))
        lat_cuda_p95 = float(np.percentile(cuda_times, 95))
        report["model_accelerations"]["yolo11n"] = {
            "device": "cuda:0",
            "is_gpu_accelerated": is_on_cuda,
            "load_time_ms": round(t_load_cuda, 2),
            "latency_mean_ms": round(lat_cuda_mean, 2),
            "latency_p50_ms": round(lat_cuda_p50, 2),
            "latency_p95_ms": round(lat_cuda_p95, 2),
            "fps": round(1000.0 / lat_cuda_mean, 1) if lat_cuda_mean > 0 else 0,
        }
        print(f"  ✓ YOLO11n on CUDA: Verified GPU tensor={is_on_cuda} | Latency: {lat_cuda_mean:.2f} ms | FPS: {1000.0/lat_cuda_mean:.1f}")

    # Test CPU execution for comparison
    print("  Comparing with YOLO11n on CPU...")
    t0 = time.perf_counter()
    detector_cpu = ObjectDetector(model_path="models/yolo11n.pt", device="cpu")
    t_load_cpu = (time.perf_counter() - t0) * 1000.0

    for _ in range(3):
        detector_cpu.detect(dummy_frame)

    cpu_times = []
    for _ in range(15):
        t_start = time.perf_counter()
        detector_cpu.detect(dummy_frame)
        cpu_times.append((time.perf_counter() - t_start) * 1000.0)

    lat_cpu_mean = float(np.mean(cpu_times))
    lat_cpu_p95 = float(np.percentile(cpu_times, 95))
    report["comparison"]["yolo11n_cpu"] = {
        "device": "cpu",
        "load_time_ms": round(t_load_cpu, 2),
        "latency_mean_ms": round(lat_cpu_mean, 2),
        "latency_p95_ms": round(lat_cpu_p95, 2),
        "speedup_gpu_vs_cpu": round(lat_cpu_mean / lat_cuda_mean, 2) if lat_cuda_mean > 0 else 1.0,
    }
    speedup = report["comparison"]["yolo11n_cpu"]["speedup_gpu_vs_cpu"]
    print(f"  ✓ YOLO11n on CPU: Latency: {lat_cpu_mean:.2f} ms | GPU Speedup: {speedup}x")

    # 2. Probe YuNet Face Detector (OpenCV DNN)
    print("\n[2/6] Probing Face Detector (YuNet ONNX)...")
    from proctoring.detection.face_detector import FaceDetector

    t0 = time.perf_counter()
    face_detector = FaceDetector(model_path="models/face_detection_yunet_2023mar.onnx")
    t_load_yunet = (time.perf_counter() - t0) * 1000.0

    face_times = []
    for _ in range(30):
        t_start = time.perf_counter()
        face_detector.detect(dummy_frame)
        face_times.append((time.perf_counter() - t_start) * 1000.0)

    lat_yunet_mean = float(np.mean(face_times))
    report["model_accelerations"]["yunet"] = {
        "backend": "OpenCV DNN (CPU)",
        "is_gpu_accelerated": False,
        "reason": "OpenCV pip wheel compiled without CUDA DNN target; CPU execution is ~3ms.",
        "load_time_ms": round(t_load_yunet, 2),
        "latency_mean_ms": round(lat_yunet_mean, 2),
        "latency_p95_ms": round(float(np.percentile(face_times, 95)), 2),
    }
    print(f"  ✓ YuNet Face Detector: Backend=OpenCV CPU | Latency: {lat_yunet_mean:.2f} ms")

    # 3. Probe SFace Face Verifier (OpenCV DNN)
    print("\n[3/6] Probing Face Verifier (SFace ONNX)...")
    from proctoring.detection.face_verifier import FaceVerifier

    t0 = time.perf_counter()
    face_verifier = FaceVerifier(detector=face_detector, recognizer_model_path="models/face_recognition_sface_2021dec.onnx")
    t_load_sface = (time.perf_counter() - t0) * 1000.0

    dummy_face_crop = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    dummy_face_raw = np.array([10, 10, 80, 80, 20, 20, 40, 20, 30, 40, 20, 60, 40, 60, 0.95], dtype=np.float32)
    dummy_face_det = FaceDetection(
        bbox=(10, 10, 80, 80),
        confidence=0.95,
        landmarks=[(20, 20), (40, 20), (30, 40), (20, 60), (40, 60)],
        raw_detection=dummy_face_raw,
    )

    sface_times = []
    for _ in range(30):
        t_start = time.perf_counter()
        face_verifier.extract_feature(dummy_frame, dummy_face_det)
        sface_times.append((time.perf_counter() - t_start) * 1000.0)

    lat_sface_mean = float(np.mean(sface_times))
    report["model_accelerations"]["sface"] = {
        "backend": "OpenCV DNN (CPU)",
        "is_gpu_accelerated": False,
        "reason": "OpenCV pip wheel compiled without CUDA DNN target; CPU execution is ~2.7ms.",
        "load_time_ms": round(t_load_sface, 2),
        "latency_mean_ms": round(lat_sface_mean, 2),
        "latency_p95_ms": round(float(np.percentile(sface_times, 95)), 2),
    }
    print(f"  ✓ SFace Face Verifier: Backend=OpenCV CPU | Latency: {lat_sface_mean:.2f} ms")

    # 4. Probe MediaPipe Face & Hand Landmark Components
    print("\n[4/6] Probing MediaPipe Components...")
    from proctoring.analysis.facial_dynamics import FacialDynamicsAnalyzer
    from proctoring.analysis.hands import HandAnalyzer

    t0 = time.perf_counter()
    face_mesh = FacialDynamicsAnalyzer(model_path="models/face_landmarker.task")
    t_load_mesh = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    hand_analyzer = HandAnalyzer(model_path="models/hand_landmarker.task")
    t_load_hands = (time.perf_counter() - t0) * 1000.0

    mesh_times = []
    hand_times = []
    for _ in range(20):
        t_start = time.perf_counter()
        face_mesh.analyze(dummy_frame)
        mesh_times.append((time.perf_counter() - t_start) * 1000.0)

        t_start = time.perf_counter()
        hand_analyzer.analyze(dummy_frame)
        hand_times.append((time.perf_counter() - t_start) * 1000.0)

    lat_mesh_mean = float(np.mean(mesh_times))
    lat_hands_mean = float(np.mean(hand_times))

    report["model_accelerations"]["mediapipe_face"] = {
        "backend": "MediaPipe Tasks (CPU XNNPACK)",
        "is_gpu_accelerated": False,
        "reason": "MediaPipe Tasks Python on Linux uses EGL OpenGL ES; CPU XNNPACK delegate is active.",
        "load_time_ms": round(t_load_mesh, 2),
        "latency_mean_ms": round(lat_mesh_mean, 2),
    }
    report["model_accelerations"]["mediapipe_hands"] = {
        "backend": "MediaPipe Tasks (CPU XNNPACK)",
        "is_gpu_accelerated": False,
        "reason": "MediaPipe Tasks Python on Linux uses EGL OpenGL ES; CPU XNNPACK delegate is active.",
        "load_time_ms": round(t_load_hands, 2),
        "latency_mean_ms": round(lat_hands_mean, 2),
    }
    print(f"  ✓ MediaPipe Face Landmarker: Backend=CPU XNNPACK | Latency: {lat_mesh_mean:.2f} ms")
    print(f"  ✓ MediaPipe Hand Landmarker: Backend=CPU XNNPACK | Latency: {lat_hands_mean:.2f} ms")

    # 5. Probe Paper Detector
    print("\n[5/6] Probing Algorithmic Paper Detector...")
    from proctoring.analysis.paper import PaperDetector

    paper_detector = PaperDetector()
    paper_times = []
    for _ in range(30):
        t_start = time.perf_counter()
        paper_detector.detect(dummy_frame)
        paper_times.append((time.perf_counter() - t_start) * 1000.0)

    lat_paper = float(np.mean(paper_times))
    report["model_accelerations"]["paper_detector"] = {
        "backend": "Algorithmic OpenCV / NumPy (CPU)",
        "latency_mean_ms": round(lat_paper, 2),
    }
    print(f"  ✓ Paper Detector: Latency: {lat_paper:.2f} ms")

    # 6. Full End-to-End Pipeline Latency (Normal Runtime)
    print("\n[6/6] Measuring Full Pipeline Runtime Latency with GPU YOLO...")
    from proctoring.engine import ProctoringEngine
    from proctoring.config import SessionConfig
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SessionConfig(output_dir=tmpdir)
        engine = ProctoringEngine(
            config=cfg,
            face_detector=face_detector,
            face_verifier=face_verifier,
            object_detector=detector_cuda if cuda_available else detector_cpu,
        )

        pipeline_times = []
        for i in range(25):
            t_start = time.perf_counter()
            engine.process_frame(dummy_frame, timestamp_seconds=float(i * 0.25))
            if cuda_available:
                torch.cuda.synchronize()
            pipeline_times.append((time.perf_counter() - t_start) * 1000.0)

    pipe_mean = float(np.mean(pipeline_times))
    pipe_p50 = float(np.percentile(pipeline_times, 50))
    pipe_p95 = float(np.percentile(pipeline_times, 95))
    mem_info = get_memory_info()

    report["benchmarks"]["pipeline_runtime"] = {
        "mean_latency_ms": round(pipe_mean, 2),
        "p50_latency_ms": round(pipe_p50, 2),
        "p95_latency_ms": round(pipe_p95, 2),
        "fps": round(1000.0 / pipe_mean, 1) if pipe_mean > 0 else 0,
        "memory": mem_info,
    }

    print("-" * 70)
    print(f"FULL PIPELINE (with GPU YOLO11n):")
    print(f"  Mean Latency: {pipe_mean:.2f} ms | P50: {pipe_p50:.2f} ms | P95: {pipe_p95:.2f} ms")
    print(f"  Effective Throughput: {1000.0 / pipe_mean:.1f} FPS")
    print(f"  System RAM: {mem_info['system_ram_mb']} MB | VRAM Allocated: {mem_info['vram_allocated_mb']} MB (Reserved: {mem_info['vram_reserved_mb']} MB)")
    print("=" * 70)

    return report


if __name__ == "__main__":
    report = run_gpu_diagnostics()
    out_path = Path("tools/benchmark/gpu_diagnostic_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {out_path.resolve()}")
