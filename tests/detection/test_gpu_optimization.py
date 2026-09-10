"""Tests for AI-005: GPU optimization, device placement, and safe fallback across detectors."""

import numpy as np
import pytest

from proctoring.analysis.facial_dynamics import FacialDynamicsAnalyzer
from proctoring.analysis.hands import HandAnalyzer
from proctoring.config import SessionConfig
from proctoring.detection.face_detector import FaceDetector
from proctoring.detection.face_verifier import FaceVerifier
from proctoring.detection.object_detector import ObjectDetector
from proctoring.engine import ProctoringEngine


def test_cuda_hardware_availability() -> None:
    """Verify NVIDIA CUDA availability in the host environment when torch is present."""
    import torch

    assert torch.cuda.is_available() is True, "CUDA must be available on RTX 3060 development system"
    device_name = torch.cuda.get_device_name(0)
    assert "RTX 3060" in device_name or "NVIDIA" in device_name


def test_yolo11_gpu_inference_and_parameter_residency() -> None:
    """Verify YOLO11 parameters reside in GPU memory and actual inference runs on CUDA."""
    import torch

    detector = ObjectDetector(device="auto")
    assert detector.device == "cuda"
    assert detector.is_gpu_accelerated is True

    # Verify actual PyTorch model parameters are on CUDA VRAM
    param = next(detector.model.model.parameters())
    assert param.is_cuda is True, "Model parameters must reside in CUDA VRAM"

    # Execute actual inference
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    result = detector.detect(dummy_frame)
    torch.cuda.synchronize()

    assert result.device == "cuda"
    assert result.count >= 0
    assert result.inference_time_ms > 0
    # Latency should be sub-20ms on RTX 3060
    assert result.inference_time_ms < 50.0


def test_yolo11_explicit_cpu_execution() -> None:
    """Verify YOLO11 runs safely on CPU when explicitly configured."""
    detector = ObjectDetector(device="cpu")
    assert detector.device == "cpu"
    assert detector.is_gpu_accelerated is False

    param = next(detector.model.model.parameters())
    assert param.is_cuda is False, "Model parameters must reside on CPU"

    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    result = detector.detect(dummy_frame)
    assert result.device == "cpu"
    assert result.count >= 0


def test_yunet_cpu_residency_and_graceful_handling() -> None:
    """Verify YuNet face detector operates safely on CPU and handles CUDA request gracefully."""
    # When requesting CUDA or auto on OpenCV build without CUDA DNN target,
    # it must safely resolve to CPU (MLAS SGEMM) without crashing.
    detector_auto = FaceDetector(device="auto")
    assert detector_auto.device == "cpu"
    assert detector_auto.is_gpu_accelerated is False

    detector_cuda = FaceDetector(device="cuda")
    assert detector_cuda.device == "cpu"
    assert detector_cuda.is_gpu_accelerated is False

    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    res = detector_auto.detect(dummy_frame)
    assert res.count == 0


def test_sface_cpu_residency_and_verification_invariance() -> None:
    """Verify SFace face verifier operates on CPU and preserves exact similarity semantics."""
    verifier = FaceVerifier(device="auto")
    assert verifier.device == "cpu"
    assert verifier.is_gpu_accelerated is False

    # Verify same synthetic face against itself has cosine similarity 1.0 (or > 0.99)
    dummy_aligned = np.random.RandomState(42).randint(0, 255, (112, 112, 3), dtype=np.uint8)
    feat1 = verifier.extract_feature(dummy_aligned)
    feat2 = verifier.extract_feature(dummy_aligned)

    sim = verifier.compute_similarity(feat1, feat2)
    assert sim is not None
    assert sim > 0.99, "Self-similarity must be ~1.0"
    assert verifier.default_threshold == 0.3630


def test_mediapipe_cpu_xnnpack_residency() -> None:
    """Verify MediaPipe landmarker pipelines operate on CPU via XNNPACK delegate."""
    face_mesh = FacialDynamicsAnalyzer(device="auto")
    assert face_mesh.device == "cpu"
    assert face_mesh.is_gpu_accelerated is False

    hand_analyzer = HandAnalyzer(device="auto")
    assert hand_analyzer.device == "cpu"
    assert hand_analyzer.is_gpu_accelerated is False


def test_end_to_end_pipeline_with_optimized_backends(tmp_path) -> None:
    """Verify full proctoring pipeline execution with GPU YOLO and CPU YuNet/SFace/MediaPipe."""
    cfg = SessionConfig(
        output_dir=tmp_path,
        device="auto",
        enable_wearable_detection=False,
    )

    detector = ObjectDetector(device="auto")
    engine = ProctoringEngine(config=cfg, object_detector=detector)
    assert engine.object_detector_info.device in ("cuda", "cuda:0")

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    obs = engine.process_frame(dummy_frame, timestamp_seconds=0.25)

    assert obs is not None
    assert obs.frame_index == 0
    assert obs.accepted is True
    engine.close_analyzers()
