"""Detection of audio and wearable devices: headphones, earpieces, smart watches.

None of these classes exist in the COCO label set that standard YOLO weights are
trained on, so this analyzer uses **YOLO-World**, an open-vocabulary detector that
accepts free-text class prompts.

### Honest accuracy expectations

These three targets are not equally detectable, and the pipeline says so rather
than presenting one confidence number for all of them:

* **Over-ear headphones** — large, high-contrast, usually unoccluded.  Detected
  reliably. Reported as ``HEADPHONES_DETECTED``.
* **Smart watches** — small but distinctly shaped on a wrist. Detected moderately
  well when the wrist is in frame.
* **In-ear earbuds** — a few dozen pixels at typical webcam distance, frequently
  hidden by hair, and easily confused with an earring, a mole, or an ear canal
  shadow.  **This is not a reliable detection.**  It is reported as
  ``EARBUDS_SUSPECTED`` at ``MEDIUM`` severity, never as a confirmed finding, and
  always with a snapshot so a proctor can look for themselves.

To cut the largest source of earbud false positives, any earpiece-class detection
is geometrically cross-checked against the ear regions derived from the face
landmarks. A candidate box that is nowhere near an ear is discarded outright.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from proctoring.core.events import EventType

LOGGER = logging.getLogger(__name__)

# Text prompts given to the open-vocabulary detector, and the event each maps to.
# Several phrasings per target: open-vocabulary recall is sensitive to wording.
WEARABLE_PROMPTS: dict[str, tuple[EventType, str]] = {
    "headphones": (EventType.HEADPHONES_DETECTED, "headphones"),
    "over-ear headphones": (EventType.HEADPHONES_DETECTED, "headphones"),
    "headset with microphone": (EventType.HEADPHONES_DETECTED, "headphones"),
    "earphones": (EventType.EARBUDS_SUSPECTED, "earbuds"),
    "wireless earbud in ear": (EventType.EARBUDS_SUSPECTED, "earbuds"),
    "bluetooth earpiece": (EventType.EARBUDS_SUSPECTED, "earbuds"),
    "smart watch": (EventType.SMARTWATCH_DETECTED, "smartwatch"),
}

# Targets whose detections must sit near an ear to be believable.
_EAR_ANCHORED = {"earbuds", "headphones"}


@dataclass
class WearableDetection:
    """One detected device."""

    target: str  # "headphones" | "earbuds" | "smartwatch"
    prompt: str  # text prompt that fired
    event_type: EventType
    confidence: float
    bbox: tuple[int, int, int, int]
    ear_anchored: bool = False  # confirmed to overlap an ear region
    reliability: str = "medium"  # "high" | "medium" | "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "prompt": self.prompt,
            "event_type": self.event_type.value,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "ear_anchored": self.ear_anchored,
            "reliability": self.reliability,
        }


@dataclass
class WearableAnalysisResult:
    """Per-invocation wearable detection outcome."""

    ran: bool = False
    """False when the detector was skipped this frame by the cadence scheduler."""

    detections: list[WearableDetection] = field(default_factory=list)
    rejected_count: int = 0
    """Candidates discarded for sitting nowhere near an ear."""

    inference_ms: float = 0.0

    @property
    def target_names(self) -> list[str]:
        return [d.target for d in self.detections]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "detections": [d.to_dict() for d in self.detections],
            "rejected_count": self.rejected_count,
            "inference_ms": round(self.inference_ms, 2),
        }


class WearableDetector:
    """Open-vocabulary detector for audio devices and wearables.

    Roughly an order of magnitude slower than the landmark models (about 250 ms per
    frame on CPU versus 15 ms), so the engine runs it on a decimated cadence rather
    than every frame.  A device worn during an exam stays on for minutes, so
    sampling every second or two loses nothing that matters.
    """

    def __init__(
        self,
        model_name: str = "yolov8s-world.pt",
        confidence_threshold: float = 0.20,
        earbud_confidence_threshold: float = 0.28,
        prompts: Sequence[str] | None = None,
        device: str | None = None,
        auto_load: bool = True,
    ) -> None:
        """
        Args:
            confidence_threshold: Floor for headphones and smart watches.
            earbud_confidence_threshold: Deliberately higher floor for earpieces.
                Small, ambiguous targets produce the most false positives, and a
                false accusation of wearing an earpiece is expensive for a candidate.
        """
        self.model_name = model_name
        self.confidence_threshold = float(confidence_threshold)
        self.earbud_confidence_threshold = float(earbud_confidence_threshold)
        self.prompts = list(prompts) if prompts else list(WEARABLE_PROMPTS.keys())
        self.device = device
        self.model = None
        self.is_available = False

        if auto_load:
            self.load_model()

    def load_model(self) -> bool:
        """Load YOLO-World and bind the text prompts, degrading gracefully on failure."""
        try:
            from ultralytics import YOLOWorld

            self.model = YOLOWorld(self.model_name)
            self.model.set_classes(self.prompts)
            if self.device:
                self.model.to(self.device)
            self.is_available = True
        except Exception as exc:
            LOGGER.info("Wearable detector unavailable: %s", exc)
            self.model = None
            self.is_available = False
        return self.is_available

    def detect(
        self,
        frame: np.ndarray,
        ear_regions: list[tuple[int, int, int, int]] | None = None,
        face_bbox: tuple[int, int, int, int] | None = None,
    ) -> WearableAnalysisResult:
        """Run open-vocabulary detection and filter the results geometrically.

        ``ear_regions`` come from the facial landmarks.  When supplied, ear-anchored
        targets must overlap one of them; this removes the bulk of spurious earbud
        boxes fired by earrings, hair clips and background clutter.
        """
        import time

        result = WearableAnalysisResult()
        if not self.is_available or frame is None or frame.size == 0:
            return result

        t0 = time.perf_counter()
        try:
            prediction = self.model.predict(
                frame,
                conf=min(self.confidence_threshold, self.earbud_confidence_threshold),
                verbose=False,
            )[0]
        except Exception as exc:
            LOGGER.warning("Wearable detection failed: %s", exc)
            return result
        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        result.ran = True

        names = self.model.model.names
        for box in prediction.boxes:
            prompt = names[int(box.cls)]
            mapping = WEARABLE_PROMPTS.get(prompt)
            if mapping is None:
                continue
            event_type, target = mapping

            confidence = float(box.conf)
            floor = (
                self.earbud_confidence_threshold
                if target == "earbuds"
                else self.confidence_threshold
            )
            if confidence < floor:
                continue

            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            bbox = (x1, y1, x2, y2)

            anchored = self._overlaps_any(bbox, ear_regions or [])
            if target in _EAR_ANCHORED:
                if ear_regions and not anchored:
                    # Fires nowhere near an ear — not a believable earpiece.
                    result.rejected_count += 1
                    continue
                if (
                    not ear_regions
                    and face_bbox is not None
                    and not self._overlaps_any(bbox, [self._expand(face_bbox, 0.4)])
                ):
                    # No landmarks this frame, so no ear regions. The face box is a
                    # coarser but still meaningful check: an earpiece detected on the
                    # far side of the room is not one.
                    result.rejected_count += 1
                    continue

            result.detections.append(
                WearableDetection(
                    target=target,
                    prompt=prompt,
                    event_type=event_type,
                    confidence=confidence,
                    bbox=bbox,
                    ear_anchored=anchored,
                    reliability=self._rate_reliability(target, confidence, anchored),
                )
            )

        return self._deduplicate(result)

    @staticmethod
    def _rate_reliability(target: str, confidence: float, ear_anchored: bool) -> str:
        """Rate how much weight a proctor should give this detection.

        Earbuds never rate "high" regardless of the model's confidence — the target
        is too small and too easily confused at webcam resolution for a confidence
        score alone to be trustworthy.
        """
        if target == "earbuds":
            return "medium" if (ear_anchored and confidence >= 0.45) else "low"
        if target == "headphones":
            if confidence >= 0.45:
                return "high"
            return "medium" if confidence >= 0.30 else "low"
        return "high" if confidence >= 0.50 else "medium"

    @staticmethod
    def _expand(box: tuple[int, int, int, int], ratio: float) -> tuple[int, int, int, int]:
        """Grow a box by a proportional margin on every side."""
        x1, y1, x2, y2 = box
        pad_x, pad_y = int((x2 - x1) * ratio), int((y2 - y1) * ratio)
        return (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)

    @staticmethod
    def _overlaps_any(
        bbox: tuple[int, int, int, int], regions: list[tuple[int, int, int, int]]
    ) -> bool:
        x1, y1, x2, y2 = bbox
        for rx1, ry1, rx2, ry2 in regions:
            if not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2):
                return True
        return False

    @staticmethod
    def _deduplicate(result: WearableAnalysisResult) -> WearableAnalysisResult:
        """Keep only the strongest detection per target.

        Several prompts describe the same object ("headphones", "over-ear
        headphones", "headset with microphone"), so one physical device commonly
        fires more than one box. Reporting it once keeps the event record honest.
        """
        best: dict[str, WearableDetection] = {}
        for detection in result.detections:
            existing = best.get(detection.target)
            if existing is None or detection.confidence > existing.confidence:
                best[detection.target] = detection
        result.detections = sorted(best.values(), key=lambda d: -d.confidence)
        return result
