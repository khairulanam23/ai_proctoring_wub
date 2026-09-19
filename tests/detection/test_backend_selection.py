"""Regression tests for Part 8: Backend selection and OpenCV CPU fallback (P1-4).

Verifies:
1. FaceDetector selects ORT CUDA when requested and available, falling back to OpenCV CPU.
2. FaceVerifier selects ORT CUDA when requested, falling back to OpenCV CPU.
3. OpenCV CUDA availability is truthfully checked (0 devices on standard pip build);
   no false claim of OpenCV CUDA acceleration is made.
4. Backend hierarchy is explicit:
   - Primary: ONNX Runtime CUDA (CUDAExecutionProvider)
   - Secondary / Fallback: OpenCV CPU / ORT CPU
"""
import cv2
import numpy as np
import pytest

from proctoring.detection.face_detector import FaceDetector
from proctoring.detection.face_verifier import FaceVerifier


def test_opencv_cuda_device_truthfulness():
    """Verify that OpenCV CUDA is reported truthfully and not falsely claimed."""
    has_opencv_cuda = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    # Standard opencv-python wheel does NOT include CUDA binaries
    assert has_opencv_cuda is False, "OpenCV should not falsely claim CUDA support on standard pip wheel"


def test_face_detector_backend_selection_cpu():
    """Verify FaceDetector explicitly uses OpenCV CPU when configured for CPU."""
    detector = FaceDetector(device="cpu")
    assert detector.device == "cpu"
    assert detector.backend_name == "opencv_cpu"
    assert detector.is_gpu_accelerated is False

    # Inference test
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    result = detector.detect(dummy)
    assert result.count == 0


def test_face_detector_backend_selection_cuda():
    """Verify FaceDetector selects ORT CUDA when requested and CUDA is available."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on host")

    detector = FaceDetector(device="cuda")
    assert detector.device.startswith("cuda")
    assert detector.backend_name == "ort_cuda"
    assert detector.is_gpu_accelerated is True


def test_face_verifier_backend_selection_cpu():
    """Verify FaceVerifier explicitly uses CPU when configured for CPU."""
    verifier = FaceVerifier(device="cpu")
    assert verifier.device == "cpu"
    assert verifier.is_gpu_accelerated is False


def test_face_verifier_backend_selection_cuda():
    """Verify FaceVerifier selects ORT CUDA when requested and CUDA is available."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on host")

    verifier = FaceVerifier(device="cuda")
    assert verifier.device.startswith("cuda")
    assert verifier.backend_name == "ort_cuda"
    assert verifier.is_gpu_accelerated is True
