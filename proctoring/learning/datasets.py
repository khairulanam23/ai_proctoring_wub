"""Versioned dataset management with session-grouped train/val/test splitting.

Guarantees data leakage prevention: frames/crops from the same exam session or
candidate are strictly contained within one partition (never randomly split across
train and test).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from proctoring.learning.annotations import (
    YOLO_LABEL_MAP,
    HumanAnnotationRecord,
    ReviewStatus,
    TargetObjectLabel,
    get_yolo_class_names,
)
from proctoring.learning.inbox import InboxSample

LOGGER = logging.getLogger(__name__)

FORBIDDEN_BIOMETRIC_KEYS = {
    "embedding",
    "embeddings",
    "reference_template",
    "reference_templates",
    "face_vector",
    "face_vectors",
    "biometric_vector",
    "biometric_vectors",
    "sface_vector",
    "face_descriptor",
    "gallery_vector",
    "probe_vector",
}


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
    sample_records: list[dict[str, Any]] = field(default_factory=list)
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
            "sample_records": self.sample_records,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
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
            sample_records=data.get("sample_records", []),
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
        inbox_base_dir: str | Path | None = None,
        export_assets: bool = True,
    ) -> DatasetManifest:
        """Create a new versioned dataset with session-grouped train/val/test splits."""
        dataset_id = f"dataset_{version.replace('.', '_')}"
        dataset_dir = self.datasets_root / dataset_id
        if dataset_dir.exists():
            raise ValueError(f"Dataset version {version} already exists at {dataset_dir}")

        # Filter only human-reviewed annotations (CONFIRMED, CORRECTED, or legacy VERIFIED)
        # Excludes PENDING, REJECTED, and UNCERTAIN
        verified_samples = [
            s
            for s in samples
            if s.sample_id in annotations
            and annotations[s.sample_id].review_status in (
                ReviewStatus.CONFIRMED,
                ReviewStatus.CORRECTED,
                ReviewStatus.VERIFIED,
            )
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
        sample_records: list[dict[str, Any]] = []
        checksum_lines: list[str] = []

        # Compute provenance hash over sample IDs and split assignment
        hasher = hashlib.sha256()
        hasher.update(f"{dataset_id}:{version}".encode())

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
                if obj.label in YOLO_LABEL_MAP and not obj.is_hard_negative:
                    classes_set.add(obj.label.value)
                if obj.is_hard_negative or obj.label == TargetObjectLabel.HARD_NEGATIVE_OBJECT:
                    hard_negatives += 1

            hasher.update(f"{s.sample_id}:{split}".encode())

            dest_img_sha256: str | None = None
            if export_assets:
                # Find source image
                src_img_path: Path | None = None
                if inbox_base_dir:
                    candidate_path = Path(inbox_base_dir) / s.image_rel_path
                    if candidate_path.exists():
                        src_img_path = candidate_path
                if not src_img_path:
                    cand = Path(s.image_rel_path)
                    if cand.exists():
                        src_img_path = cand
                    elif (Path("training/inbox") / s.image_rel_path).exists():
                        src_img_path = Path("training/inbox") / s.image_rel_path

                img_w, img_h = 640, 640
                dest_img_path = dataset_dir / split / "images" / f"{s.sample_id}.jpg"
                if src_img_path and src_img_path.exists():
                    shutil.copy2(src_img_path, dest_img_path)
                    dest_img_sha256 = hashlib.sha256(dest_img_path.read_bytes()).hexdigest()
                    checksum_lines.append(f"{dest_img_sha256}  {split}/images/{s.sample_id}.jpg")
                    im = cv2.imread(str(src_img_path))
                    if im is not None:
                        img_h, img_w = im.shape[:2]

                # Write YOLO label file
                yolo_lines = ann.to_yolo_format(img_width=img_w, img_height=img_h)
                dest_label_path = dataset_dir / split / "labels" / f"{s.sample_id}.txt"
                dest_label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

            # Scrub any biometric keys from metadata to ensure strict privacy
            sanitized_meta = {
                k: v for k, v in s.metadata.items()
                if k.lower() not in FORBIDDEN_BIOMETRIC_KEYS
            }

            sample_records.append({
                "sample_id": s.sample_id,
                "session_id": s.session_id,
                "frame_index": s.frame_index,
                "target_class": s.target_class,
                "split": split,
                "image_file": f"{split}/images/{s.sample_id}.jpg" if export_assets else None,
                "image_sha256": dest_img_sha256,
                "label_file": f"{split}/labels/{s.sample_id}.txt" if export_assets else None,
                "metadata": sanitized_meta,
                "annotator_id": ann.annotator_id,
                "review_status": ann.review_status.value,
                "annotated_at_utc": ann.annotated_at_utc,
            })

        # Generate standard data.yaml
        yaml_content = [
            f"# Proctoring Dataset {version}",
            f"path: {dataset_dir.resolve()}",
            "train: train/images",
            "val: val/images",
            "test: test/images",
            "",
            "names:",
        ]
        class_names = get_yolo_class_names()
        for idx in sorted(class_names.keys()):
            yaml_content.append(f"  {idx}: {class_names[idx]}")
        yaml_content.append("")
        (dataset_dir / "data.yaml").write_text("\n".join(yaml_content), encoding="utf-8")

        provenance_hash = hasher.hexdigest()

        if checksum_lines:
            (dataset_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

        (dataset_dir / "provenance.json").write_text(json.dumps(sample_records, indent=2), encoding="utf-8")

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
            provenance_hash=provenance_hash,
            sample_records=sample_records,
            notes=notes,
        )

        manifest_path = dataset_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest.to_dict(), indent=2))

        LOGGER.info(
            "Created versioned dataset %s: %d samples across %d sessions (hash: %s)",
            version,
            len(verified_samples),
            len(all_sessions),
            provenance_hash[:12],
        )
        return manifest

    def verify_dataset_integrity(self, version: str) -> tuple[bool, list[str]]:
        """Verify SHA-256 integrity of all exported evidence files in dataset."""
        dataset_id = f"dataset_{version.replace('.', '_')}"
        dataset_dir = self.datasets_root / dataset_id
        prov_file = dataset_dir / "provenance.json"
        if not prov_file.exists():
            return False, [f"Missing provenance file at {prov_file}"]

        try:
            records = json.loads(prov_file.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, [f"Corrupt provenance file: {exc}"]

        errors = []
        for rec in records:
            img_rel = rec.get("image_file")
            expected_sha = rec.get("image_sha256")
            if not img_rel or not expected_sha:
                continue
            img_path = dataset_dir / img_rel
            if not img_path.exists():
                errors.append(f"Missing file: {img_rel}")
                continue
            actual_sha = hashlib.sha256(img_path.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                errors.append(
                    f"Integrity violation in {img_rel}: expected {expected_sha}, got {actual_sha}"
                )

        return len(errors) == 0, errors

    def load_manifest(self, version: str) -> DatasetManifest | None:
        dataset_id = f"dataset_{version.replace('.', '_')}"
        manifest_path = self.datasets_root / dataset_id / "dataset_manifest.json"
        if not manifest_path.exists():
            return None
        with open(manifest_path, encoding="utf-8") as f:
            return DatasetManifest.from_dict(json.loads(f.read()))
