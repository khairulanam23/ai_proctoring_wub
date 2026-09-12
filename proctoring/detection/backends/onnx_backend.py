"""ONNX Runtime backend management for GPU-accelerated inference.

Provides automatic detection and pre-loading of bundled NVIDIA CUDA 13 / cuDNN 9
shared libraries, and manages ONNX Runtime InferenceSessions configured for
CUDAExecutionProvider with graceful fallback to CPUExecutionProvider.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

_CUDA_LIBS_PRELOADED = False


def ensure_cuda_libraries() -> bool:
    """Pre-load bundled NVIDIA CUDA and cuDNN shared libraries into global symbol space.

    Python wheels installed from pip (such as nvidia-cublas-cu13 and nvidia-cudnn-cu13)
    reside in site-packages/nvidia/*/lib. On Linux systems, these directories are not
    automatically searched by the OS dynamic linker unless added to LD_LIBRARY_PATH
    or loaded via ctypes with RTLD_GLOBAL prior to onnxruntime initialization.

    Returns:
        True if CUDA shared libraries were successfully located and loaded.
    """
    global _CUDA_LIBS_PRELOADED
    if _CUDA_LIBS_PRELOADED:
        return True

    try:
        # Search Python environment site-packages
        site_packages_dirs: list[Path] = []
        for p in sys.path:
            p_path = Path(p)
            if p_path.is_dir() and (p_path / "nvidia").is_dir():
                site_packages_dirs.append(p_path)

        libs_to_load = [
            ("nvidia/cu13/lib", "libcudart.so.13"),
            ("nvidia/cu13/lib", "libcublasLt.so.13"),
            ("nvidia/cu13/lib", "libcublas.so.13"),
            ("nvidia/cu13/lib", "libcurand.so.10"),
            ("nvidia/cudnn/lib", "libcudnn.so.9"),
        ]

        loaded_count = 0
        added_paths: list[str] = []

        for site_pkg in site_packages_dirs:
            for rel_dir, lib_name in libs_to_load:
                lib_path = site_pkg / rel_dir / lib_name
                if lib_path.exists():
                    try:
                        ctypes.CDLL(str(lib_path.resolve()), mode=ctypes.RTLD_GLOBAL)
                        loaded_count += 1
                        dir_str = str(lib_path.parent.resolve())
                        if dir_str not in added_paths:
                            added_paths.append(dir_str)
                    except Exception as e:
                        LOGGER.debug("Could not pre-load %s: %s", lib_path, e)

        if added_paths:
            cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
            prefix = ":".join(added_paths)
            os.environ["LD_LIBRARY_PATH"] = f"{prefix}:{cur_ld}" if cur_ld else prefix

        _CUDA_LIBS_PRELOADED = loaded_count > 0
        return _CUDA_LIBS_PRELOADED
    except Exception as exc:
        LOGGER.debug("ensure_cuda_libraries encounter: %s", exc)
        return False


class ORTSessionWrapper:
    """Encapsulates an onnxruntime.InferenceSession with CUDA prioritization."""

    def __init__(
        self,
        model_path: str | Path,
        prefer_cuda: bool = True,
        device_id: int = 0,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")

        import onnxruntime as ort

        # Ensure CUDA libraries are preloaded before session creation
        if prefer_cuda:
            ensure_cuda_libraries()

        available_providers = ort.get_available_providers()
        providers: list[tuple[str, dict[str, Any]] | str] = []

        cuda_requested = prefer_cuda and ("CUDAExecutionProvider" in available_providers)
        if cuda_requested:
            providers.append("CUDAExecutionProvider")

        providers.append("CPUExecutionProvider")

        # Session configuration
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3  # 3 = Error, suppress harmless warnings

        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers,
            )
        except Exception as exc:
            if cuda_requested:
                LOGGER.warning(
                    "CUDAExecutionProvider failed for %s (%s). Falling back to CPUExecutionProvider.",
                    self.model_path.name,
                    exc,
                )
                self.session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )
            else:
                raise

        active_providers = self.session.get_providers()
        self.active_provider = active_providers[0] if active_providers else "CPUExecutionProvider"
        self.is_cuda = self.active_provider == "CUDAExecutionProvider"

        LOGGER.info(
            "Loaded %s with provider %s (requested CUDA: %s)",
            self.model_path.name,
            self.active_provider,
            prefer_cuda,
        )

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        """Execute model forward pass."""
        return self.session.run(output_names, input_feed)

    def get_inputs(self) -> list[Any]:
        """Get model inputs descriptor."""
        return self.session.get_inputs()

    def get_outputs(self) -> list[Any]:
        """Get model outputs descriptor."""
        return self.session.get_outputs()
