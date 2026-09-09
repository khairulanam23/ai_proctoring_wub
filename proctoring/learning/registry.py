"""Production model registry with audit traceability, promotion rules, and rollback.

Ensures no candidate model is promoted to active production runtime without meeting
explicit latency, precision/recall, and hard-negative regression criteria.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Lifecycle status of a model in the registry."""

    EXPERIMENTAL = "experimental"
    CANDIDATE = "candidate"
    APPROVED = "approved"
    ACTIVE = "active"
    RETIRED = "retired"
    REJECTED = "rejected"


@dataclass
class ModelMetrics:
    """Benchmark performance metrics for model evaluation."""

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    cpu_latency_ms: float = 0.0
    peak_memory_mb: float = 0.0
    hard_negative_accuracy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "cpu_latency_ms": round(self.cpu_latency_ms, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "hard_negative_accuracy": round(self.hard_negative_accuracy, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetrics":
        return cls(
            precision=float(data.get("precision", 0.0)),
            recall=float(data.get("recall", 0.0)),
            f1_score=float(data.get("f1_score", 0.0)),
            false_positive_rate=float(data.get("false_positive_rate", 0.0)),
            false_negative_rate=float(data.get("false_negative_rate", 0.0)),
            cpu_latency_ms=float(data.get("cpu_latency_ms", 0.0)),
            peak_memory_mb=float(data.get("peak_memory_mb", 0.0)),
            hard_negative_accuracy=float(data.get("hard_negative_accuracy", 0.0)),
        )


@dataclass
class ModelMetadata:
    """Comprehensive metadata for a registered model version."""

    model_id: str
    model_name: str
    version: str
    model_type: str  # "object_detector", "face_verifier", "wearable_detector", etc.
    base_model: str
    dataset_version: str
    metrics: ModelMetrics
    status: ModelStatus = ModelStatus.EXPERIMENTAL
    thresholds: dict[str, float] = field(default_factory=dict)
    artifact_rel_path: str = ""
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    promoted_at_utc: str | None = None
    retired_at_utc: str | None = None
    git_commit: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "model_type": self.model_type,
            "base_model": self.base_model,
            "dataset_version": self.dataset_version,
            "metrics": self.metrics.to_dict(),
            "status": self.status.value,
            "thresholds": self.thresholds,
            "artifact_rel_path": self.artifact_rel_path,
            "created_at_utc": self.created_at_utc,
            "promoted_at_utc": self.promoted_at_utc,
            "retired_at_utc": self.retired_at_utc,
            "git_commit": self.git_commit,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        return cls(
            model_id=data["model_id"],
            model_name=data["model_name"],
            version=data["version"],
            model_type=data["model_type"],
            base_model=data.get("base_model", ""),
            dataset_version=data.get("dataset_version", ""),
            metrics=ModelMetrics.from_dict(data.get("metrics", {})),
            status=ModelStatus(data.get("status", "experimental")),
            thresholds=data.get("thresholds", {}),
            artifact_rel_path=data.get("artifact_rel_path", ""),
            created_at_utc=data.get("created_at_utc", ""),
            promoted_at_utc=data.get("promoted_at_utc"),
            retired_at_utc=data.get("retired_at_utc"),
            git_commit=data.get("git_commit", ""),
            notes=data.get("notes", ""),
        )


class ModelRegistry:
    """Manages model registration, promotion criteria evaluation, and rollback."""

    def __init__(self, registry_root: str | Path = "training/registry") -> None:
        self.registry_root = Path(registry_root)
        self.models_dir = self.registry_root / "models"
        self.metadata_file = self.registry_root / "registry_index.json"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_root.mkdir(parents=True, exist_ok=True)

    def register_model(
        self,
        model_name: str,
        version: str,
        model_type: str,
        artifact_path: str | Path,
        metrics: ModelMetrics,
        base_model: str = "",
        dataset_version: str = "",
        thresholds: dict[str, float] | None = None,
        git_commit: str = "",
        notes: str = "",
    ) -> ModelMetadata:
        """Register a new candidate model artifact into the registry."""
        model_id = f"{model_name}_{version.replace('.', '_')}"
        dest_filename = f"{model_id}{Path(artifact_path).suffix}"
        dest_path = self.models_dir / dest_filename

        if Path(artifact_path).exists():
            shutil.copy2(artifact_path, dest_path)

        meta = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            version=version,
            model_type=model_type,
            base_model=base_model,
            dataset_version=dataset_version,
            metrics=metrics,
            status=ModelStatus.CANDIDATE,
            thresholds=thresholds or {},
            artifact_rel_path=f"models/{dest_filename}",
            git_commit=git_commit,
            notes=notes,
        )

        all_models = self.list_models()
        all_models[model_id] = meta
        self._save_index(all_models)
        LOGGER.info("Registered model %s as CANDIDATE", model_id)
        return meta

    def list_models(self) -> dict[str, ModelMetadata]:
        if not self.metadata_file.exists():
            return {}
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
            return {k: ModelMetadata.from_dict(v) for k, v in data.items()}

    def get_active_model(self, model_type: str) -> ModelMetadata | None:
        """Retrieve currently active production model for a given type."""
        for meta in self.list_models().values():
            if meta.model_type == model_type and meta.status == ModelStatus.ACTIVE:
                return meta
        return None

    def evaluate_and_promote(
        self,
        candidate_model_id: str,
        max_latency_ms: float = 60.0,
        min_precision: float = 0.85,
        min_recall: float = 0.85,
        min_hard_negative_acc: float = 0.90,
    ) -> tuple[bool, str]:
        """Evaluate candidate against promotion criteria and promote with rollback safety."""
        models = self.list_models()
        candidate = models.get(candidate_model_id)
        if not candidate:
            return False, f"Model {candidate_model_id} not found in registry."

        m = candidate.metrics
        reasons: list[str] = []

        if m.cpu_latency_ms > max_latency_ms:
            reasons.append(f"Latency {m.cpu_latency_ms:.1f}ms exceeds budget {max_latency_ms:.1f}ms")
        if m.precision < min_precision:
            reasons.append(f"Precision {m.precision:.3f} below floor {min_precision:.3f}")
        if m.recall < min_recall:
            reasons.append(f"Recall {m.recall:.3f} below floor {min_recall:.3f}")
        if m.hard_negative_accuracy < min_hard_negative_acc:
            reasons.append(
                f"Hard-negative accuracy {m.hard_negative_accuracy:.3f} below floor {min_hard_negative_acc:.3f}"
            )

        if reasons:
            candidate.status = ModelStatus.REJECTED
            self._save_index(models)
            msg = "Promotion rejected: " + "; ".join(reasons)
            LOGGER.warning(msg)
            return False, msg

        # Retire previously active model of same type
        now_iso = datetime.now(timezone.utc).isoformat()
        for m_id, m_meta in models.items():
            if (
                m_meta.model_type == candidate.model_type
                and m_meta.status == ModelStatus.ACTIVE
                and m_id != candidate_model_id
            ):
                m_meta.status = ModelStatus.RETIRED
                m_meta.retired_at_utc = now_iso
                LOGGER.info("Retired previously active model %s", m_id)

        candidate.status = ModelStatus.ACTIVE
        candidate.promoted_at_utc = now_iso
        self._save_index(models)
        LOGGER.info("Successfully promoted %s to ACTIVE", candidate_model_id)
        return True, f"Model {candidate_model_id} successfully promoted to ACTIVE."

    def rollback(self, model_type: str, target_model_id: str) -> bool:
        """Rollback active model to a specified previously retired or approved model."""
        models = self.list_models()
        target = models.get(target_model_id)
        if not target or target.model_type != model_type:
            return False

        now_iso = datetime.now(timezone.utc).isoformat()
        for m_meta in models.values():
            if m_meta.model_type == model_type and m_meta.status == ModelStatus.ACTIVE:
                m_meta.status = ModelStatus.RETIRED
                m_meta.retired_at_utc = now_iso

        target.status = ModelStatus.ACTIVE
        target.promoted_at_utc = now_iso
        self._save_index(models)
        LOGGER.info("Rolled back %s to active model %s", model_type, target_model_id)
        return True

    def _save_index(self, models: dict[str, ModelMetadata]) -> None:
        tmp = self.registry_root / "registry_index.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps({k: v.to_dict() for k, v in models.items()}, indent=2))
        os.replace(tmp, self.metadata_file)
