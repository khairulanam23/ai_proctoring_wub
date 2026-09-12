"""GPU health and runtime acceleration diagnostics for production telemetry."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class DeviceModelPlacement:
    """Device placement and backend execution metadata for a neural model."""

    model_name: str
    target_hardware: str  # "GPU" or "CPU"
    backend_framework: str  # "PyTorch CUDA", "ORT CUDA", "OpenCV CPU", "MediaPipe XNNPACK"
    device_name: str  # "cuda:0", "cpu"
    status: str  # "ACTIVE", "FALLBACK", "UNAVAILABLE"


@dataclass
class GPUDiagnosticsReport:
    """Structured report of GPU availability, VRAM memory, and model placement."""

    cuda_available: bool
    gpu_name: str | None = None
    cuda_version: str | None = None
    compute_capability: str | None = None
    vram_total_mb: float = 0.0
    vram_free_mb: float = 0.0
    vram_allocated_mb: float = 0.0
    vram_reserved_mb: float = 0.0
    models: list[DeviceModelPlacement] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_gpu_diagnostics(
    engine: Any | None = None,
    face_detector: Any | None = None,
    face_verifier: Any | None = None,
    object_detector: Any | None = None,
) -> GPUDiagnosticsReport:
    """Collect runtime GPU hardware state and model backend placement."""
    cuda_available = False
    gpu_name = None
    cuda_version = None
    compute_cap = None
    total_mb = 0.0
    free_mb = 0.0
    alloc_mb = 0.0
    res_mb = 0.0

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            cap = torch.cuda.get_device_capability(0)
            compute_cap = f"{cap[0]}.{cap[1]}"

            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            total_mb = round(total_bytes / (1024 * 1024), 2)
            free_mb = round(free_bytes / (1024 * 1024), 2)
            alloc_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
            res_mb = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
    except Exception as exc:
        LOGGER.debug("GPU diagnostics probe warning: %s", exc)

    # Resolve detectors from engine if not passed directly
    if engine is not None:
        face_detector = face_detector or getattr(engine, "face_detector", None)
        face_verifier = face_verifier or getattr(engine, "face_verifier", None)
        object_detector = object_detector or getattr(engine, "object_detector", None)

    models: list[DeviceModelPlacement] = []

    # 1. Object Detector (YOLO)
    if object_detector is not None:
        is_gpu = getattr(object_detector, "is_gpu_accelerated", False)
        dev = getattr(object_detector, "device", "cpu")
        models.append(
            DeviceModelPlacement(
                model_name="YOLO11n (Object / Person Detection)",
                target_hardware="GPU" if is_gpu else "CPU",
                backend_framework="PyTorch CUDA" if is_gpu else "PyTorch CPU",
                device_name=dev,
                status="ACTIVE",
            )
        )

    # 2. Face Detector (YuNet)
    if face_detector is not None:
        is_gpu = getattr(face_detector, "is_gpu_accelerated", False)
        b_name = getattr(face_detector, "backend_name", "opencv_cpu")
        framework = "ORT CUDAExecutionProvider" if b_name == "ort_cuda" else "OpenCV DNN (CPU)"
        models.append(
            DeviceModelPlacement(
                model_name="YuNet (Face Detection)",
                target_hardware="GPU" if is_gpu else "CPU",
                backend_framework=framework,
                device_name=getattr(face_detector, "device", "cpu"),
                status="ACTIVE",
            )
        )

    # 3. Face Verifier (SFace)
    if face_verifier is not None:
        is_gpu = getattr(face_verifier, "is_gpu_accelerated", False)
        b_name = getattr(face_verifier, "backend_name", "opencv_cpu")
        framework = "ORT CUDAExecutionProvider" if b_name == "ort_cuda" else "OpenCV DNN (CPU)"
        models.append(
            DeviceModelPlacement(
                model_name="SFace (Face Recognition / Verification)",
                target_hardware="GPU" if is_gpu else "CPU",
                backend_framework=framework,
                device_name=getattr(face_verifier, "device", "cpu"),
                status="ACTIVE",
            )
        )

    # 4. MediaPipe HandLandmarker
    models.append(
        DeviceModelPlacement(
            model_name="HandLandmarker (Hand Kinematics)",
            target_hardware="CPU",
            backend_framework="MediaPipe Tasks (XNNPACK multithreaded)",
            device_name="cpu",
            status="ACTIVE",
        )
    )

    # 5. MediaPipe FaceLandmarker
    models.append(
        DeviceModelPlacement(
            model_name="FaceLandmarker (Facial Dynamics)",
            target_hardware="CPU",
            backend_framework="MediaPipe Tasks (XNNPACK multithreaded)",
            device_name="cpu",
            status="ACTIVE",
        )
    )

    return GPUDiagnosticsReport(
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        cuda_version=cuda_version,
        compute_capability=compute_cap,
        vram_total_mb=total_mb,
        vram_free_mb=free_mb,
        vram_allocated_mb=alloc_mb,
        vram_reserved_mb=res_mb,
        models=models,
    )


def log_gpu_diagnostics(report: GPUDiagnosticsReport) -> None:
    """Emit formatted diagnostic report to application logs."""
    LOGGER.info("=" * 65)
    LOGGER.info("AI RUNTIME ACCELERATION DIAGNOSTICS")
    LOGGER.info("=" * 65)
    LOGGER.info("CUDA Available:      %s", report.cuda_available)
    if report.cuda_available:
        LOGGER.info("GPU Hardware:        %s", report.gpu_name)
        LOGGER.info("CUDA / Compute:      %s (sm_%s)", report.cuda_version, report.compute_capability)
        LOGGER.info(
            "VRAM Total:          %.2f MB (Free: %.2f MB, Alloc: %.2f MB)",
            report.vram_total_mb,
            report.vram_free_mb,
            report.vram_allocated_mb,
        )
    LOGGER.info("-" * 65)
    LOGGER.info("%-24s | %-6s | %-12s | %s", "Model / Component", "HW", "Device", "Inference Backend")
    LOGGER.info("-" * 65)
    for m in report.models:
        LOGGER.info(
            "%-24s | %-6s | %-12s | %s",
            m.model_name[:24],
            m.target_hardware,
            m.device_name,
            m.backend_framework,
        )
    LOGGER.info("=" * 65)
