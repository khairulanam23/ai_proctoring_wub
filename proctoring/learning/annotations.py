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
    VERIFIED = "verified"
    REJECTED = "rejected"


class TargetObjectLabel(str, Enum):
    PHONE = "phone"
    EARBUD = "earbud"
    OVER_EAR_HEADPHONE = "over_ear_headphone"
    PAPER = "paper"
    TABLET = "tablet"
    LAPTOP = "laptop"
    BOOK = "book"
    HARD_NEGATIVE_OBJECT = "hard_negative_object"


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


YOLO_LABEL_MAP: dict[TargetObjectLabel, int] = {
    TargetObjectLabel.PHONE: 0,
    TargetObjectLabel.EARBUD: 1,
    TargetObjectLabel.OVER_EAR_HEADPHONE: 2,
    TargetObjectLabel.PAPER: 3,
    TargetObjectLabel.TABLET: 4,
    TargetObjectLabel.LAPTOP: 5,
    TargetObjectLabel.BOOK: 6,
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
