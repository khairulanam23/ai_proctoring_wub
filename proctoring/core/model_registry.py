"""Model Registry for centralized, thread-safe model lifecycle and VRAM optimization.

Ensures heavyweight neural models (YOLO11n, YuNet, SFace) are instantiated once and
safely shared across concurrent sessions without multiplying GPU memory allocation.
Provides cryptographic SHA-256 weight integrity verification before model instantiation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class ModelIntegrityError(RuntimeError):
    """Raised when model weight cryptographic integrity verification fails."""


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file in 64KB chunks.

    Args:
        file_path: Path to the target binary file.

    Returns:
        Lowercase hexadecimal SHA-256 string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found for checksum computation: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


class ModelRegistry:
    """Thread-safe registry and cache for shared model instances with integrity verification."""

    _lock = threading.Lock()
    _yolo_instances: dict[str, Any] = {}
    _ort_sessions: dict[str, Any] = {}
    _trusted_hashes: dict[str, str] = {}
    _enforce_integrity: bool = False

    @classmethod
    def set_enforce_integrity(cls, enforce: bool) -> None:
        """Enable or disable mandatory integrity verification for all loaded models."""
        with cls._lock:
            cls._enforce_integrity = bool(enforce)

    @classmethod
    def is_enforce_integrity(cls) -> bool:
        """Whether model integrity verification is mandatory (production mode)."""
        with cls._lock:
            if cls._enforce_integrity:
                return True
        env_val = os.environ.get("PROCTORING_REQUIRE_MODEL_INTEGRITY", "").lower().strip()
        return env_val in ("1", "true", "yes", "on", "required")

    @classmethod
    def set_trusted_hash(cls, model_identifier: str | Path, sha256: str) -> None:
        """Register expected SHA-256 hash for a model path or filename."""
        with cls._lock:
            key = str(model_identifier).strip()
            hash_val = sha256.lower().strip()
            cls._trusted_hashes[key] = hash_val
            try:
                p = Path(model_identifier)
                cls._trusted_hashes[p.name] = hash_val
                if p.is_file():
                    cls._trusted_hashes[str(p.resolve())] = hash_val
            except Exception:
                pass

    @classmethod
    def set_trusted_hashes(cls, hashes: dict[str, str]) -> None:
        """Register multiple expected SHA-256 hashes."""
        for model_id, sha in hashes.items():
            cls.set_trusted_hash(model_id, sha)

    @classmethod
    def load_hash_manifest(cls, manifest_path: str | Path) -> None:
        """Load trusted SHA-256 hashes from a JSON manifest file."""
        path = Path(manifest_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Integrity manifest not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Model manifest must be a JSON object mapping model names to hashes")
        cls.set_trusted_hashes(data)

    @classmethod
    def clear_trusted_hashes(cls) -> None:
        """Clear all registered trusted hashes."""
        with cls._lock:
            cls._trusted_hashes.clear()

    @classmethod
    def get_expected_hash(cls, model_path: str | Path) -> str | None:
        """Retrieve expected SHA-256 hash if configured."""
        path = Path(model_path)
        with cls._lock:
            if str(model_path) in cls._trusted_hashes:
                return cls._trusted_hashes[str(model_path)]
            if path.name in cls._trusted_hashes:
                return cls._trusted_hashes[path.name]
            try:
                resolved_str = str(path.resolve())
                if resolved_str in cls._trusted_hashes:
                    return cls._trusted_hashes[resolved_str]
            except Exception:
                pass

        env_manifest = os.environ.get("PROCTORING_MODEL_MANIFEST")
        if env_manifest and Path(env_manifest).is_file():
            cls.load_hash_manifest(env_manifest)
            with cls._lock:
                if str(model_path) in cls._trusted_hashes:
                    return cls._trusted_hashes[str(model_path)]
                if path.name in cls._trusted_hashes:
                    return cls._trusted_hashes[path.name]

        return None

    @classmethod
    def verify_model_integrity(
        cls,
        model_path: str | Path,
        expected_sha256: str | None = None,
    ) -> str | None:
        """Verify SHA-256 checksum of a model file before loading.

        If an expected hash is provided or configured in the registry manifest,
        computes the file's SHA-256 and asserts equality. Fails closed with
        ModelIntegrityError if hashes mismatch.

        If no expected hash is configured:
        - In production mode (is_enforce_integrity() == True), raises ModelIntegrityError.
        - In development/testing mode (is_enforce_integrity() == False), returns None
          and allows unverified loading.

        Returns:
            The computed SHA-256 hex digest if verified, or None in unconfigured dev mode.
        """
        expected = expected_sha256 or cls.get_expected_hash(model_path)
        if expected is not None:
            expected = expected.lower().strip()

        # If verification is not configured for this model:
        if expected is None:
            if cls.is_enforce_integrity():
                raise ModelIntegrityError(
                    f"Production integrity verification required, but no trusted SHA-256 hash "
                    f"is configured for model '{model_path}'."
                )
            return None

        # Verification IS configured: resolve file and verify
        path = Path(model_path)
        if not path.is_file():
            candidate = Path("models") / path
            if candidate.is_file():
                path = candidate
            else:
                raise FileNotFoundError(
                    f"Model file not found for integrity verification: {model_path}"
                )

        computed = compute_file_sha256(path)
        if computed != expected:
            raise ModelIntegrityError(
                f"SHA-256 integrity verification failed for '{model_path}': "
                f"expected '{expected}', computed '{computed}'"
            )

        LOGGER.info(
            "ModelRegistry: Verified SHA-256 integrity for %s (%s)",
            model_path,
            computed[:12] + "...",
        )
        return computed

    @classmethod
    def get_yolo(
        cls,
        model_path: str | Path,
        device: str = "cuda",
        expected_sha256: str | None = None,
    ) -> Any:
        """Get or load a shared YOLO model instance on the specified device.

        Integrity verification is performed before loading the model into memory.
        Enforces local file resolution to prevent unintended dynamic network downloads.
        """
        resolved_path = Path(model_path)
        if not resolved_path.is_file():
            candidate = Path("models") / model_path
            if candidate.is_file():
                resolved_path = candidate
            else:
                raise FileNotFoundError(
                    f"Model weights file '{model_path}' not found locally. "
                    "Automatic network downloads are blocked for offline airgap compliance."
                )

        cls.verify_model_integrity(resolved_path, expected_sha256=expected_sha256)

        key = f"{resolved_path.resolve()}_{device}"
        with cls._lock:
            if key not in cls._yolo_instances:
                from ultralytics import YOLO

                LOGGER.info("ModelRegistry: Loading shared YOLO model %s on %s", resolved_path, device)
                model = YOLO(str(resolved_path))
                cls._yolo_instances[key] = model
            return cls._yolo_instances[key]

    @classmethod
    def get_ort_session(
        cls,
        model_path: str | Path,
        prefer_cuda: bool = True,
        device_id: int = 0,
        expected_sha256: str | None = None,
    ) -> Any:
        """Get or load a shared ORTSessionWrapper for the specified ONNX model.

        Integrity verification is performed before initializing the ORT session.
        """
        cls.verify_model_integrity(model_path, expected_sha256=expected_sha256)

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
    def unload_all(cls, clear_hashes: bool = False) -> None:
        """Unload all cached models and release GPU VRAM."""
        with cls._lock:
            cls._yolo_instances.clear()
            cls._ort_sessions.clear()
            if clear_hashes:
                cls._trusted_hashes.clear()
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
            "configured_trusted_hashes_count": len(cls._trusted_hashes),
            "vram_allocated_mb": vram_allocated_mb,
            "vram_reserved_mb": vram_reserved_mb,
        }

