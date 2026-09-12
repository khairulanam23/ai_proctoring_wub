"""Model Registry for centralized, thread-safe model lifecycle and VRAM optimization.

Ensures heavyweight neural models (YOLO11n, YuNet, SFace) are instantiated once and
safely shared across concurrent sessions without multiplying GPU memory allocation.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class ModelRegistry:
    """Thread-safe registry and cache for shared model instances."""

    _lock = threading.Lock()
    _yolo_instances: dict[str, Any] = {}
    _ort_sessions: dict[str, Any] = {}

    @classmethod
    def get_yolo(cls, model_path: str | Path, device: str = "cuda") -> Any:
        """Get or load a shared YOLO model instance on the specified device."""
        key = f"{Path(model_path).resolve()}_{device}"
        with cls._lock:
            if key not in cls._yolo_instances:
                from ultralytics import YOLO

                LOGGER.info("ModelRegistry: Loading shared YOLO model %s on %s", model_path, device)
                model = YOLO(str(model_path))
                cls._yolo_instances[key] = model
            return cls._yolo_instances[key]

    @classmethod
    def get_ort_session(
        cls,
        model_path: str | Path,
        prefer_cuda: bool = True,
        device_id: int = 0,
    ) -> Any:
        """Get or load a shared ORTSessionWrapper for the specified ONNX model."""
        key = f"{Path(model_path).resolve()}_cuda_{prefer_cuda}_dev_{device_id}"
        with cls._lock:
            if key not in cls._ort_sessions:
                from proctoring.detection.backends.onnx_backend import ORTSessionWrapper

                LOGGER.info(
                    "ModelRegistry: Loading shared ORT session %s (prefer_cuda=%s, dev=%s)",
                    model_path,
                    prefer_cuda,
                    device_id,
                )
                session = ORTSessionWrapper(
                    model_path=model_path,
                    prefer_cuda=prefer_cuda,
                    device_id=device_id,
                )
                cls._ort_sessions[key] = session
            return cls._ort_sessions[key]

    @classmethod
    def unload_all(cls) -> None:
        """Unload all cached models and release GPU VRAM."""
        with cls._lock:
            cls._yolo_instances.clear()
            cls._ort_sessions.clear()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            LOGGER.info("ModelRegistry: All shared models unloaded and VRAM cache cleared.")

    @classmethod
    def get_resident_models_summary(cls) -> dict[str, Any]:
        """Return summary of currently cached models and GPU VRAM footprint."""
        vram_allocated_mb = 0.0
        vram_reserved_mb = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                vram_allocated_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
                vram_reserved_mb = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
        except ImportError:
            pass

        return {
            "cached_yolo_models": list(cls._yolo_instances.keys()),
            "cached_ort_sessions": list(cls._ort_sessions.keys()),
            "vram_allocated_mb": vram_allocated_mb,
            "vram_reserved_mb": vram_reserved_mb,
        }
