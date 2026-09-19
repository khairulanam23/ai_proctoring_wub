"""Human verification and labeling system for proctoring training datasets.

Provides structured schemas for bounding box objects (phones, earbuds, headphones, paper),
hand kinematics states, and hard negatives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class ReviewStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"
    VERIFIED = "verified"  # backward compatibility
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


class TargetObjectLabel(str, Enum):
    # Core Phase 3 controlled object taxonomy
    PHONE = "phone"
    SCIENTIFIC_CALCULATOR = "scientific_calculator"
    POWER_BANK = "power_bank"
    NOTEBOOK = "notebook"
    BOOK = "book"
    PENCIL_CASE = "pencil_case"
    ID_CARD = "id_card"
    EARBUDS = "earbuds"
    HEADPHONES = "headphones"
    PEN = "pen"
    PENCIL = "pencil"
    PAPER = "paper"
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    OTHER = "other"

    # Disqualification / negative categories (EXCLUDED from positive object classes)
    UNCERTAIN = "uncertain"
    NOT_A_RELEVANT_OBJECT = "not_a_relevant_object"
    HARD_NEGATIVE_OBJECT = "hard_negative_object"

    # Backward compatibility aliases
    EARBUD = "earbud"
    OVER_EAR_HEADPHONE = "over_ear_headphone"
    TABLET = "tablet"
    LAPTOP = "laptop"


@dataclass
class AnnotatedBBox:
    """Ground-truth bounding box created or confirmed by human annotator."""

    label: TargetObjectLabel
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    is_hard_negative: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.value,
            "bbox": list(self.bbox),
            "is_hard_negative": self.is_hard_negative,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnnotatedBBox":
        return cls(
            label=TargetObjectLabel(data["label"]),
            bbox=tuple(data["bbox"]),
            is_hard_negative=bool(data.get("is_hard_negative", False)),
            notes=data.get("notes", ""),
        )


# Positive object classes for YOLO training (strictly excludes uncertain and not_a_relevant_object)
YOLO_LABEL_MAP: dict[TargetObjectLabel, int] = {
    TargetObjectLabel.PHONE: 0,
    TargetObjectLabel.SCIENTIFIC_CALCULATOR: 1,
    TargetObjectLabel.POWER_BANK: 2,
    TargetObjectLabel.NOTEBOOK: 3,
    TargetObjectLabel.BOOK: 4,
    TargetObjectLabel.PENCIL_CASE: 5,
    TargetObjectLabel.ID_CARD: 6,
    TargetObjectLabel.EARBUDS: 7,
    TargetObjectLabel.HEADPHONES: 8,
    TargetObjectLabel.PEN: 9,
    TargetObjectLabel.PENCIL: 10,
    TargetObjectLabel.PAPER: 11,
    TargetObjectLabel.KEYBOARD: 12,
    TargetObjectLabel.MOUSE: 13,
    TargetObjectLabel.OTHER: 14,
}

# Non-trainable labels (excluded from positive detector training)
NON_TRAINABLE_LABELS: set[TargetObjectLabel] = {
    TargetObjectLabel.UNCERTAIN,
    TargetObjectLabel.NOT_A_RELEVANT_OBJECT,
    TargetObjectLabel.HARD_NEGATIVE_OBJECT,
}


def get_yolo_class_names() -> dict[int, str]:
    """Return map of class ID to class string name for dataset YAML configs."""
    return {idx: label.value for label, idx in YOLO_LABEL_MAP.items()}



@dataclass
class HumanAnnotationRecord:
    """Authoritative human ground truth for a training sample."""

    sample_id: str
    annotator_id: str
    review_status: ReviewStatus
    objects: list[AnnotatedBBox] = field(default_factory=list)
    hand_behavior_label: str | None = None
    quality_pass: bool = True
    rejection_reason: str | None = None
    annotated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "annotator_id": self.annotator_id,
            "review_status": self.review_status.value,
            "objects": [obj.to_dict() for obj in self.objects],
            "hand_behavior_label": self.hand_behavior_label,
            "quality_pass": self.quality_pass,
            "rejection_reason": self.rejection_reason,
            "annotated_at_utc": self.annotated_at_utc,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HumanAnnotationRecord":
        return cls(
            sample_id=data["sample_id"],
            annotator_id=data["annotator_id"],
            review_status=ReviewStatus(data["review_status"]),
            objects=[AnnotatedBBox.from_dict(o) for o in data.get("objects", [])],
            hand_behavior_label=data.get("hand_behavior_label"),
            quality_pass=bool(data.get("quality_pass", True)),
            rejection_reason=data.get("rejection_reason"),
            annotated_at_utc=data.get("annotated_at_utc", ""),
            metadata=data.get("metadata", {}),
        )

    def to_yolo_format(self, img_width: int, img_height: int) -> list[str]:
        """Convert bounding boxes to standard YOLO normalized txt lines: class_id cx cy w h."""
        lines: list[str] = []
        for obj in self.objects:
            if obj.label in YOLO_LABEL_MAP and not obj.is_hard_negative:
                cid = YOLO_LABEL_MAP[obj.label]
                x1, y1, x2, y2 = obj.bbox
                cx = ((x1 + x2) / 2.0) / max(1, img_width)
                cy = ((y1 + y2) / 2.0) / max(1, img_height)
                w = (x2 - x1) / max(1, img_width)
                h = (y2 - y1) / max(1, img_height)
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return lines
