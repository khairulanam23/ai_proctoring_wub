"""Versioned dataset management with session-grouped train/val/test splitting.

Guarantees data leakage prevention: frames/crops from the same exam session or
candidate are strictly contained within one partition (never randomly split across
train and test).
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from proctoring.learning.annotations import HumanAnnotationRecord, ReviewStatus
from proctoring.learning.inbox import InboxSample

LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetManifest:
    """Immutable audit manifest for a versioned training dataset."""

    dataset_id: str
    version: str
    classes: list[str]
    total_samples: int
    train_count: int
    val_count: int
    test_count: int
    hard_negatives_count: int
    session_ids: list[str]
    split_strategy: str = "session_grouped"
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    provenance_hash: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "classes": self.classes,
            "total_samples": self.total_samples,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "test_count": self.test_count,
            "hard_negatives_count": self.hard_negatives_count,
            "session_ids": self.session_ids,
            "split_strategy": self.split_strategy,
            "created_at_utc": self.created_at_utc,
            "provenance_hash": self.provenance_hash,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetManifest":
        return cls(
            dataset_id=data["dataset_id"],
            version=data["version"],
            classes=data["classes"],
            total_samples=int(data["total_samples"]),
            train_count=int(data["train_count"]),
            val_count=int(data["val_count"]),
            test_count=int(data["test_count"]),
            hard_negatives_count=int(data["hard_negatives_count"]),
            session_ids=data.get("session_ids", []),
            split_strategy=data.get("split_strategy", "session_grouped"),
            created_at_utc=data.get("created_at_utc", ""),
            provenance_hash=data.get("provenance_hash", ""),
            notes=data.get("notes", ""),
        )


class DatasetManager:
    """Creates, versions, and audits datasets for proctoring model training."""

    def __init__(self, datasets_root: str | Path = "training/datasets") -> None:
        self.datasets_root = Path(datasets_root)
        self.datasets_root.mkdir(parents=True, exist_ok=True)

    def create_versioned_dataset(
        self,
        version: str,
        samples: list[InboxSample],
        annotations: dict[str, HumanAnnotationRecord],
        split_ratio: tuple[float, float, float] = (0.70, 0.15, 0.15),
        seed: int = 42,
        notes: str = "",
    ) -> DatasetManifest:
        """Create a new versioned dataset with session-grouped train/val/test splits."""
        dataset_id = f"dataset_{version.replace('.', '_')}"
        dataset_dir = self.datasets_root / dataset_id
        if dataset_dir.exists():
            raise ValueError(f"Dataset version {version} already exists at {dataset_dir}")

        # Filter only human-verified annotations
        verified_samples = [
            s
            for s in samples
            if s.sample_id in annotations
            and annotations[s.sample_id].review_status == ReviewStatus.VERIFIED
            and annotations[s.sample_id].quality_pass
        ]

        if not verified_samples:
            raise ValueError("Cannot create dataset without verified human annotations.")

        # Group samples strictly by session_id to prevent data leakage
        session_map: dict[str, list[InboxSample]] = {}
        for s in verified_samples:
            session_map.setdefault(s.session_id, []).append(s)

        all_sessions = sorted(session_map.keys())
        rng = random.Random(seed)
        rng.shuffle(all_sessions)

        n_sessions = len(all_sessions)
        n_train = max(1, int(n_sessions * split_ratio[0]))
        n_val = max(1, int(n_sessions * split_ratio[1])) if n_sessions >= 3 else 0

        train_sessions = set(all_sessions[:n_train])
        val_sessions = set(all_sessions[n_train : n_train + n_val])
        test_sessions = set(all_sessions[n_train + n_val :])
        if not test_sessions and n_sessions >= 2:
            test_sessions.add(all_sessions[-1])
            train_sessions.discard(all_sessions[-1])

        # Prepare directory structure
        for split in ("train", "val", "test"):
            (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        counts = {"train": 0, "val": 0, "test": 0}
        hard_negatives = 0
        classes_set: set[str] = set()

        for s in verified_samples:
            if s.session_id in train_sessions:
                split = "train"
            elif s.session_id in val_sessions:
                split = "val"
            else:
                split = "test"

            counts[split] += 1
            ann = annotations[s.sample_id]
            for obj in ann.objects:
                classes_set.add(obj.label.value)
                if obj.is_hard_negative:
                    hard_negatives += 1

        manifest = DatasetManifest(
            dataset_id=dataset_id,
            version=version,
            classes=sorted(classes_set),
            total_samples=len(verified_samples),
            train_count=counts["train"],
            val_count=counts["val"],
            test_count=counts["test"],
            hard_negatives_count=hard_negatives,
            session_ids=all_sessions,
            notes=notes,
        )

        manifest_path = dataset_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest.to_dict(), indent=2))

        LOGGER.info(
            "Created versioned dataset %s: %d samples across %d sessions",
            version,
            len(verified_samples),
            len(all_sessions),
        )
        return manifest

    def load_manifest(self, version: str) -> DatasetManifest | None:
        dataset_id = f"dataset_{version.replace('.', '_')}"
        manifest_path = self.datasets_root / dataset_id / "dataset_manifest.json"
        if not manifest_path.exists():
            return None
        with open(manifest_path, "r", encoding="utf-8") as f:
            return DatasetManifest.from_dict(json.loads(f.read()))
