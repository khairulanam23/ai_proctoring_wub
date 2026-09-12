"""Tests for GPU acceleration, backend parity, and model registry caching."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from proctoring.core.model_registry import ModelRegistry
from proctoring.detection.face_detector import FaceDetector, _OpenCVYuNetBackend, _ORTYuNetBackend
from proctoring.detection.face_verifier import FaceVerifier, _OpenCVSFaceBackend, _ORTSFaceBackend
from proctoring.telemetry.gpu_diagnostics import get_gpu_diagnostics

SAMPLE_IMAGE = Path("data/samples/Colin_Powell/Colin_Powell_0001.jpg")


def test_ort_cuda_backend_initialization() -> None:
    """Test that ORT backends successfully initialize with CUDAExecutionProvider."""
    yunet_path = Path("models/face_detection_yunet_2023mar.onnx")
    sface_path = Path("models/face_recognition_sface_2021dec.onnx")

    if not yunet_path.exists() or not sface_path.exists():
        pytest.skip("Model files missing")

    ort_yunet = _ORTYuNetBackend(
        model_path=yunet_path,
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
        prefer_cuda=True,
    )
    assert ort_yunet.is_cuda is True

    ort_sface = _ORTSFaceBackend(
        model_path=sface_path,
        prefer_cuda=True,
    )
    assert ort_sface.is_cuda is True


def test_yunet_cpu_vs_gpu_parity() -> None:
    """Test that CPU and GPU YuNet outputs achieve high IoU and score parity."""
    if not SAMPLE_IMAGE.exists():
        pytest.skip("Sample image missing")

    img = cv2.imread(str(SAMPLE_IMAGE))
    assert img is not None

    detector_cpu = FaceDetector(device="cpu")
    detector_gpu = FaceDetector(device="cuda")

    assert detector_cpu.backend_name == "opencv_cpu"
    assert detector_gpu.backend_name == "ort_cuda"
    assert detector_gpu.is_gpu_accelerated is True

    res_cpu = detector_cpu.detect(img)
    res_gpu = detector_gpu.detect(img)

    assert res_cpu.count >= 1
    assert res_gpu.count >= 1

    box_cpu = res_cpu.faces[0].bbox
    box_gpu = res_gpu.faces[0].bbox

    xA = max(box_cpu[0], box_gpu[0])
    yA = max(box_cpu[1], box_gpu[1])
    xB = min(box_cpu[0] + box_cpu[2], box_gpu[0] + box_gpu[2])
    yB = min(box_cpu[1] + box_cpu[3], box_gpu[1] + box_gpu[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = box_cpu[2] * box_cpu[3] + box_gpu[2] * box_gpu[3] - inter
    iou = inter / union if union > 0 else 0.0

    assert iou >= 0.90, f"YuNet CPU vs GPU IoU too low: {iou:.4f}"
    assert abs(res_cpu.faces[0].confidence - res_gpu.faces[0].confidence) < 0.05


def test_sface_cpu_vs_gpu_parity() -> None:
    """Test that CPU and GPU SFace extract virtually identical embeddings (cosine sim >= 0.999)."""
    if not SAMPLE_IMAGE.exists():
        pytest.skip("Sample image missing")

    img = cv2.imread(str(SAMPLE_IMAGE))
    assert img is not None

    verifier_cpu = FaceVerifier(device="cpu")
    verifier_gpu = FaceVerifier(device="cuda")

    assert verifier_cpu.backend_name == "opencv_cpu"
    assert verifier_gpu.backend_name == "ort_cuda"
    assert verifier_gpu.is_gpu_accelerated is True

    detector = FaceDetector(device="cuda")
    det_res = detector.detect(img)
    assert det_res.count >= 1
    face = det_res.faces[0]

    emb_cpu = verifier_cpu.extract_feature(img, face=face)
    emb_gpu = verifier_gpu.extract_feature(img, face=face)

    assert emb_cpu.squeeze().shape == (128,)
    assert emb_gpu.squeeze().shape == (128,)

    cos_sim = verifier_cpu.compute_similarity(emb_cpu, emb_gpu, metric="cosine")
    assert cos_sim >= 0.999, f"SFace CPU vs GPU cosine similarity below 0.999: {cos_sim:.6f}"


def test_model_registry_caching() -> None:
    """Verify that ModelRegistry caches model instances across repeated acquisitions."""
    sface_path = Path("models/face_recognition_sface_2021dec.onnx")
    if not sface_path.exists():
        pytest.skip("Model missing")

    sess1 = ModelRegistry.get_ort_session(sface_path, prefer_cuda=True)
    sess2 = ModelRegistry.get_ort_session(sface_path, prefer_cuda=True)
    assert sess1 is sess2

    summary = ModelRegistry.get_resident_models_summary()
    assert len(summary["cached_ort_sessions"]) >= 1


def test_gpu_diagnostics_report() -> None:
    """Verify that GPUDiagnosticsReport properly identifies active GPU hardware and model placement."""
    face_det = FaceDetector(device="cuda")
    face_ver = FaceVerifier(device="cuda")
    report = get_gpu_diagnostics(face_detector=face_det, face_verifier=face_ver)

    assert report.cuda_available is True
    assert "RTX" in (report.gpu_name or "")
    assert report.vram_total_mb > 0

    hw_map = {m.model_name: m.target_hardware for m in report.models}
    assert hw_map.get("YuNet (Face Detection)") == "GPU"
    assert hw_map.get("SFace (Face Recognition / Verification)") == "GPU"
    assert hw_map.get("HandLandmarker (Hand Kinematics)") == "CPU"
    assert hw_map.get("FaceLandmarker (Facial Dynamics)") == "CPU"
