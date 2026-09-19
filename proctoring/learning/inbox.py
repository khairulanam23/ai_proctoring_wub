"""Training data inbox manager for controlled human-in-the-loop learning.

Captures difficult, marginal, false-positive, and human-disputed runtime observations
into a structured staging inbox for offline human review and labeling.
Never performs automated self-training.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class FlagReason(str, Enum):
    """Reason why an inference sample was flagged for the training inbox."""

    UNCERTAIN_DETECTION = "uncertain_detection"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    PROCTOR_DISPUTED = "proctor_disputed"
    HARD_NEGATIVE_CANDIDATE = "hard_negative_candidate"
    ANOMALOUS_GEOMETRY = "anomalous_geometry"


@dataclass
class InboxSample:
    """A staged training candidate sample awaiting human annotation."""

    sample_id: str
    session_id: str
    frame_index: int
    flag_reason: FlagReason
    image_rel_path: str
    target_class: str
    model_predictions: list[dict[str, Any]]
    confidence: float
    bbox: tuple[int, int, int, int] | None = None
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    is_hard_negative: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "session_id": self.session_id,
            "frame_index": self.frame_index,
            "flag_reason": self.flag_reason.value,
            "image_rel_path": self.image_rel_path,
            "target_class": self.target_class,
            "model_predictions": self.model_predictions,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox) if self.bbox else None,
            "created_at_utc": self.created_at_utc,
            "is_hard_negative": self.is_hard_negative,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InboxSample":
        return cls(
            sample_id=data["sample_id"],
            session_id=data["session_id"],
            frame_index=int(data["frame_index"]),
            flag_reason=FlagReason(data["flag_reason"]),
            image_rel_path=data["image_rel_path"],
            target_class=data["target_class"],
            model_predictions=data.get("model_predictions", []),
            confidence=float(data.get("confidence", 0.0)),
            bbox=tuple(data["bbox"]) if data.get("bbox") else None,
            created_at_utc=data.get("created_at_utc", ""),
            is_hard_negative=bool(data.get("is_hard_negative", False)),
            metadata=data.get("metadata", {}),
        )


class TrainingInboxManager:
    """Manages the training data inbox directory and queue."""

    def __init__(self, base_dir: str | Path = "training/inbox") -> None:
        self.base_dir = Path(base_dir)
        self.samples_dir = self.base_dir / "samples"
        self.queue_file = self.base_dir / "queue.jsonl"
        self._counter = 0

        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def capture_sample(
        self,
        image: np.ndarray,
        session_id: str,
        frame_index: int,
        target_class: str,
        flag_reason: FlagReason,
        confidence: float,
        model_predictions: list[dict[str, Any]] | None = None,
        bbox: tuple[int, int, int, int] | None = None,
        is_hard_negative: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> InboxSample:
        """Atomically persist a sample frame/crop and record it in the inbox queue."""
        self._counter += 1
        safe_session = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
        safe_class = target_class.replace(" ", "_").lower()
        sample_id = (
            f"samp_{int(time.time() * 1000)}_{self._counter:04d}_{safe_session}_{frame_index}_{safe_class}"
        )
        img_filename = f"{sample_id}.jpg"
        img_path = self.samples_dir / img_filename
        rel_path = f"samples/{img_filename}"

        # Write image atomically
        tmp_img = self.samples_dir / f"{sample_id}_tmp.jpg"
        cv2.imwrite(str(tmp_img), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        os.replace(tmp_img, img_path)

        sample = InboxSample(
            sample_id=sample_id,
            session_id=session_id,
            frame_index=frame_index,
            flag_reason=flag_reason,
            image_rel_path=rel_path,
            target_class=target_class,
            model_predictions=model_predictions or [],
            confidence=confidence,
            bbox=bbox,
            is_hard_negative=is_hard_negative,
            metadata=metadata or {},
        )

        # Append to queue.jsonl
        line = json.dumps(sample.to_dict()) + "\n"
        with open(self.queue_file, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

        LOGGER.info("Captured training inbox sample: %s (reason: %s)", sample_id, flag_reason.value)
        return sample

    def list_pending_samples(self) -> list[InboxSample]:
        """Return all samples awaiting annotation."""
        if not self.queue_file.exists():
            return []

        samples: list[InboxSample] = []
        with open(self.queue_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(InboxSample.from_dict(json.loads(line)))
                    except Exception as exc:
                        LOGGER.warning("Could not parse inbox record: %s", exc)
        return samples

    def delete_sample(self, sample_id: str) -> bool:
        """Support candidate privacy / data deletion requests."""
        if not self.queue_file.exists():
            return False

        samples = self.list_pending_samples()
        remaining = [s for s in samples if s.sample_id != sample_id]
        if len(remaining) == len(samples):
            return False

        # Remove image file
        for s in samples:
            if s.sample_id == sample_id:
                target_img = self.base_dir / s.image_rel_path
                if target_img.exists():
                    target_img.unlink()

        # Rewrite queue file
        tmp_queue = self.base_dir / "queue.jsonl.tmp"
        with open(tmp_queue, "w", encoding="utf-8") as f:
            for s in remaining:
                f.write(json.dumps(s.to_dict()) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_queue, self.queue_file)
        return True
