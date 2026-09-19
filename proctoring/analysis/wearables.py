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

### Why small earpieces need a second pass

At a typical webcam framing an in-ear bud occupies on the order of fifteen pixels
on a 640x480 frame. Open-vocabulary detectors do not resolve objects that small,
so a full-frame sweep alone reports headphones well and earbuds essentially never
— which is exactly what real-world testing showed.

The detector therefore runs a **second pass over upscaled crops of each ear
region**, where the same bud occupies a couple of hundred pixels and is within the
model's working range. Boxes found in a crop are mapped back into frame
coordinates, and are ear-anchored by construction. This costs one extra small
inference per ear on the frames the cadence scheduler already selected, and it is
the single largest recall improvement available without changing models.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from proctoring.core.events import EventType

LOGGER = logging.getLogger(__name__)


class WearableCategory(str, Enum):
    """Refined classification of wearable / ear devices."""

    NO_WEARABLE = "no_wearable"
    OVER_EAR_HEADPHONES = "over_ear_headphone"
    EARBUD_AIRPOD = "earbud_airpod"
    OTHER_EAR_OBJECT = "other_ear_object"
    UNCERTAIN = "uncertain"


# Text prompts given to the open-vocabulary detector, and the event each maps to.
# Several phrasings per target: open-vocabulary recall is sensitive to wording.
WEARABLE_PROMPTS: dict[str, tuple[EventType, str]] = {
    # Over-ear and on-ear: large, high contrast, reliably detected on the full frame.
    "headphones": (EventType.HEADPHONES_DETECTED, "headphones"),
    "over-ear headphones": (EventType.HEADPHONES_DETECTED, "headphones"),
    "headset with microphone": (EventType.HEADPHONES_DETECTED, "headphones"),
    "gaming headset": (EventType.HEADPHONES_DETECTED, "headphones"),
    # In-ear and wired: several phrasings, because open-vocabulary recall depends
    # heavily on wording and no single phrase covers the whole category. Wired
    # earphones and their cable were previously not asked for at all.
    "earphones": (EventType.EARBUDS_SUSPECTED, "earbuds"),
    "wireless earbud in ear": (EventType.EARBUDS_SUSPECTED, "earbuds"),
    "bluetooth earpiece": (EventType.EARBUDS_SUSPECTED, "earbuds"),
    "wired earphone in ear": (EventType.EARBUDS_SUSPECTED, "earbuds"),
    "earphone cable next to face": (EventType.EARBUDS_SUSPECTED, "earbuds"),
    "small white earbud": (EventType.EARBUDS_SUSPECTED, "earbuds"),
    "earbud": (EventType.EARBUDS_SUSPECTED, "earbuds"),
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
    category: WearableCategory = WearableCategory.UNCERTAIN
    category_reason: str = ""

    source: str = "frame"
    """``frame`` for the full-frame sweep, ``ear_zoom`` for the upscaled ear crop.
    Recorded because the two passes have genuinely different characteristics and a
    proctor reviewing a marginal earbud call should be able to tell them apart."""

    hand_corroborated: bool = False
    """A hand was at the ear region in the same frame. Independent of the visual
    detection, and the gesture that accompanies fitting or adjusting an earpiece."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "prompt": self.prompt,
            "event_type": self.event_type.value,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "ear_anchored": self.ear_anchored,
            "reliability": self.reliability,
            "category": self.category.value,
            "category_reason": self.category_reason,
            "source": self.source,
            "hand_corroborated": self.hand_corroborated,
        }


@dataclass
class WearableAnalysisResult:
    """Per-invocation wearable detection outcome."""

    ran: bool = False
    """False when the detector was skipped this frame by the cadence scheduler."""

    detections: list[WearableDetection] = field(default_factory=list)
    overall_category: WearableCategory = WearableCategory.NO_WEARABLE
    rejected_count: int = 0
    """Candidates discarded for sitting nowhere near an ear."""

    ear_regions_scanned: int = 0
    """Ear crops the zoom pass examined this sweep. Zero means the pass did not run —
    either it is disabled, or no face landmarks were available to locate the ears."""

    inference_ms: float = 0.0

    @property
    def target_names(self) -> list[str]:
        return [d.target for d in self.detections]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "overall_category": self.overall_category.value,
            "detections": [d.to_dict() for d in self.detections],
            "rejected_count": self.rejected_count,
            "ear_regions_scanned": self.ear_regions_scanned,
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
        enable_ear_region_zoom: bool = True,
        ear_roi_target_px: int = 320,
        auto_load: bool = True,
    ) -> None:
        """
        Args:
            confidence_threshold: Floor for headphones and smart watches.
            earbud_confidence_threshold: Deliberately higher floor for earpieces.
                Small, ambiguous targets produce the most false positives, and a
                false accusation of wearing an earpiece is expensive for a candidate.
            enable_ear_region_zoom: Run a second pass over upscaled ear crops. An
                in-ear bud is below the model's resolving power at native frame
                scale, so without this pass earbuds are effectively undetectable.
            ear_roi_target_px: Longest side each ear crop is upscaled to.
        """
        self.model_name = model_name
        self.confidence_threshold = float(confidence_threshold)
        self.earbud_confidence_threshold = float(earbud_confidence_threshold)
        self.enable_ear_region_zoom = bool(enable_ear_region_zoom)
        self.ear_roi_target_px = max(64, int(ear_roi_target_px))
        self.prompts = list(prompts) if prompts else list(WEARABLE_PROMPTS.keys())
        self.device = device
        self.model = None
        self.is_available = False

        if auto_load:
            self.load_model()

    def load_model(self) -> bool:
        """Load YOLO-World and bind text prompts with strict offline airgap enforcement."""
        try:
            import warnings
            from pathlib import Path
            from ultralytics import YOLOWorld
            from proctoring.core.model_registry import ModelRegistry

            # 1. Enforce local YOLO-World model existence (zero network downloads)
            model_target = None
            candidates = [
                Path(self.model_name),
                Path("models") / self.model_name,
                Path(__file__).resolve().parent.parent.parent / "models" / self.model_name,
            ]
            for c in candidates:
                if c.is_file():
                    model_target = str(c.resolve())
                    break

            if model_target is None:
                LOGGER.warning(
                    "WearableDetector: Local weights not found for '%s'. "
                    "Dynamic network downloads are strictly disabled for offline airgap compliance. "
                    "Wearable detector marked unavailable.",
                    self.model_name,
                )
                self.model = None
                self.is_available = False
                return False

            # 2. Enforce local CLIP weights existence (prevent dynamic ViT-B/32 download)
            clip_candidates = [
                Path("weights/clip/ViT-B-32.pt"),
                Path(__file__).resolve().parent.parent.parent / "weights" / "clip" / "ViT-B-32.pt",
                Path.home() / ".cache" / "clip" / "ViT-B-32.pt",
            ]
            clip_found = any(p.is_file() for p in clip_candidates)
            if not clip_found:
                LOGGER.warning(
                    "WearableDetector: Local CLIP weights not found ('weights/clip/ViT-B-32.pt'). "
                    "Dynamic network downloads are strictly disabled for offline airgap compliance. "
                    "Wearable detector marked unavailable.",
                )
                self.model = None
                self.is_available = False
                return False

            # 3. Verify cryptographic integrity before loading
            ModelRegistry.verify_model_integrity(model_target)

            # 4. Load local YOLO-World model
            self.model = YOLOWorld(model_target)

            # 5. Bind text prompts with targeted Python 3.14 TorchScript deprecation filter
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message=r".*torch\.jit\.load is not supported in Python 3\.14.*",
                )
                self.model.set_classes(self.prompts)

            if self.device:
                self.model.to(self.device)
            self.is_available = True
        except Exception as exc:
            LOGGER.warning("Wearable detector unavailable: %s", exc)
            self.model = None
            self.is_available = False
        return self.is_available

    def detect(
        self,
        frame: np.ndarray,
        ear_regions: list[tuple[int, int, int, int]] | None = None,
        face_bbox: tuple[int, int, int, int] | None = None,
        hand_at_ear: bool = False,
    ) -> WearableAnalysisResult:
        """Run open-vocabulary detection and filter the results geometrically.

        Two passes are made. The first covers the whole frame and finds anything
        large enough to resolve there — headphones, headsets, a watch on a wrist.
        The second, when ``ear_regions`` are available, re-examines upscaled crops of
        each ear, where a small earpiece finally occupies enough pixels to be found.

        ``ear_regions`` come from the facial landmarks. When supplied, ear-anchored
        targets from the full-frame pass must overlap one of them; this removes the
        bulk of spurious earbud boxes fired by earrings, hair clips and background
        clutter. Detections from the zoom pass are ear-anchored by construction.

        ``hand_at_ear`` is the hand analyzer's independent view of the same moment.
        It never creates a detection on its own — it only strengthens one that the
        image already supports, and is recorded on the detection so a proctor can
        see which signals agreed.
        """
        import time

        result = WearableAnalysisResult()
        if not self.is_available or frame is None or frame.size == 0:
            return result

        regions = list(ear_regions or [])
        t0 = time.perf_counter()

        detections = self._scan_frame(frame, regions, face_bbox, result)

        if self.enable_ear_region_zoom and regions:
            for region in regions:
                found = self._scan_ear_region(frame, region)
                result.ear_regions_scanned += 1
                detections.extend(found)

        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        result.ran = True

        for detection in detections:
            detection.hand_corroborated = bool(hand_at_ear) and detection.target in _EAR_ANCHORED
            detection.reliability = self._rate_reliability(
                detection.target,
                detection.confidence,
                detection.ear_anchored,
                hand_corroborated=detection.hand_corroborated,
            )
            cat, reason = self._classify_category(
                detection=detection,
                ear_regions=regions,
                hand_corroborated=detection.hand_corroborated,
            )
            detection.category = cat
            detection.category_reason = reason

        result.detections = detections
        deduped = self._deduplicate(result)

        cats = [d.category for d in deduped.detections]
        if WearableCategory.OVER_EAR_HEADPHONES in cats:
            deduped.overall_category = WearableCategory.OVER_EAR_HEADPHONES
        elif WearableCategory.EARBUD_AIRPOD in cats:
            deduped.overall_category = WearableCategory.EARBUD_AIRPOD
        elif WearableCategory.OTHER_EAR_OBJECT in cats:
            deduped.overall_category = WearableCategory.OTHER_EAR_OBJECT
        elif WearableCategory.UNCERTAIN in cats:
            deduped.overall_category = WearableCategory.UNCERTAIN
        else:
            deduped.overall_category = WearableCategory.NO_WEARABLE

        return deduped

    def _classify_category(
        self,
        detection: WearableDetection,
        ear_regions: list[tuple[int, int, int, int]],
        hand_corroborated: bool = False,
    ) -> tuple[WearableCategory, str]:
        """Refine detection into explicit categorical classification."""
        target = detection.target
        conf = detection.confidence
        bx1, by1, bx2, by2 = detection.bbox

        if target == "headphones":
            if conf >= 0.35:
                return WearableCategory.OVER_EAR_HEADPHONES, "Over-ear/on-ear headphone confirmed"
            return WearableCategory.UNCERTAIN, "Low-confidence headphone candidate"

        if target == "earbuds":
            matched_region = None
            for er in ear_regions:
                ex1, ey1, ex2, ey2 = er
                if not (bx2 < ex1 or bx1 > ex2 or by2 < ey1 or by1 > ey2):
                    matched_region = er
                    break

            if matched_region is not None:
                ex1, ey1, ex2, ey2 = matched_region
                ear_h = max(1, ey2 - ey1)
                ear_w = max(1, ex2 - ex1)
                cy = (by1 + by2) / 2.0
                rel_y = (cy - ey1) / float(ear_h)
                box_h = by2 - by1
                box_w = bx2 - bx1
                rel_h = box_h / float(ear_h)
                rel_w = box_w / float(ear_w)

                # Lobule placement (bottom 25% of ear, small punctate size)
                if rel_y > 0.75 and rel_h < 0.30 and rel_w < 0.35:
                    return (
                        WearableCategory.OTHER_EAR_OBJECT,
                        "Lobule-positioned candidate consistent with earring or jewelry",
                    )

                # Superior helix / upper rim
                if rel_y < 0.18 and rel_w > 0.60:
                    return (
                        WearableCategory.OTHER_EAR_OBJECT,
                        "Superior helix candidate consistent with glasses temple or hair accessory",
                    )

                if (
                    hand_corroborated
                    or conf >= 0.45
                    or "airpod" in detection.prompt
                    or "wireless earbud" in detection.prompt
                ):
                    return (
                        WearableCategory.EARBUD_AIRPOD,
                        "Concha-positioned in-ear bud or AirPod profile",
                    )
                elif conf < 0.32:
                    return (
                        WearableCategory.UNCERTAIN,
                        "Marginal earbud detection lacking distinct acoustic stem or concha confirmation",
                    )
                return (
                    WearableCategory.EARBUD_AIRPOD,
                    "Candidate earbud detected in ear region",
                )

            return WearableCategory.UNCERTAIN, "Candidate not anchored within calibrated ear region"

        return WearableCategory.UNCERTAIN, f"Device target '{target}' outside audio category"

    def _scan_frame(
        self,
        frame: np.ndarray,
        ear_regions: list[tuple[int, int, int, int]],
        face_bbox: tuple[int, int, int, int] | None,
        result: WearableAnalysisResult,
    ) -> list[WearableDetection]:
        """Full-frame pass, with ear-anchored targets cross-checked geometrically."""
        boxes = self._predict(frame)
        detections: list[WearableDetection] = []

        for prompt, confidence, bbox in boxes:
            mapping = WEARABLE_PROMPTS.get(prompt)
            if mapping is None:
                continue
            event_type, target = mapping
            if confidence < self._floor_for(target):
                continue

            anchored = self._overlaps_any(bbox, ear_regions)
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

            detections.append(
                WearableDetection(
                    target=target,
                    prompt=prompt,
                    event_type=event_type,
                    confidence=confidence,
                    bbox=bbox,
                    ear_anchored=anchored,
                    source="frame",
                )
            )
        return detections

    def _scan_ear_region(
        self,
        frame: np.ndarray,
        region: tuple[int, int, int, int],
    ) -> list[WearableDetection]:
        """Second pass over one ear, upscaled so a small earpiece is resolvable.

        Boxes are mapped back into frame coordinates before being returned, so every
        detection this analyzer emits is in the same coordinate space regardless of
        which pass produced it — the evidence annotator and the crop stage must not
        have to know which pass ran.
        """
        crop, origin, scale = self._ear_crop(frame, region)
        if crop is None:
            return []

        detections: list[WearableDetection] = []
        for prompt, confidence, bbox in self._predict(crop):
            mapping = WEARABLE_PROMPTS.get(prompt)
            if mapping is None:
                continue
            event_type, target = mapping
            if target not in _EAR_ANCHORED:
                # A watch found inside an ear crop is a misfire, not a wrist.
                continue
            if confidence < self._floor_for(target):
                continue

            x1, y1, x2, y2 = bbox
            detections.append(
                WearableDetection(
                    target=target,
                    prompt=prompt,
                    event_type=event_type,
                    confidence=confidence,
                    bbox=(
                        origin[0] + int(x1 / scale),
                        origin[1] + int(y1 / scale),
                        origin[0] + int(x2 / scale),
                        origin[1] + int(y2 / scale),
                    ),
                    ear_anchored=True,
                    source="ear_zoom",
                )
            )
        return detections

    def _ear_crop(
        self,
        frame: np.ndarray,
        region: tuple[int, int, int, int],
    ) -> tuple[np.ndarray | None, tuple[int, int], float]:
        """Clip an ear region to the frame and upscale it, returning the mapping back."""
        height, width = frame.shape[:2]
        x1 = max(0, min(int(region[0]), width - 1))
        y1 = max(0, min(int(region[1]), height - 1))
        x2 = max(x1 + 1, min(int(region[2]), width))
        y2 = max(y1 + 1, min(int(region[3]), height))

        patch = frame[y1:y2, x1:x2]
        if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
            return None, (x1, y1), 1.0

        longest = max(patch.shape[0], patch.shape[1])
        scale = self.ear_roi_target_px / float(longest)
        if scale <= 1.0:
            # Already large enough; upscaling further adds cost and no information.
            return patch, (x1, y1), 1.0

        try:
            import cv2

            resized = cv2.resize(
                patch,
                (max(1, int(patch.shape[1] * scale)), max(1, int(patch.shape[0] * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        except Exception as exc:
            LOGGER.debug("Ear region upscale failed: %s", exc)
            return patch, (x1, y1), 1.0

        return resized, (x1, y1), scale

    def _predict(self, image: np.ndarray) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        """Run the model over one image, returning (prompt, confidence, xyxy) triples."""
        try:
            prediction = self.model.predict(
                image,
                conf=min(self.confidence_threshold, self.earbud_confidence_threshold),
                verbose=False,
            )[0]
        except Exception as exc:
            LOGGER.warning("Wearable detection failed: %s", exc)
            return []

        names = self.model.model.names
        boxes: list[tuple[str, float, tuple[int, int, int, int]]] = []
        for box in prediction.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            boxes.append((names[int(box.cls)], float(box.conf), (x1, y1, x2, y2)))
        return boxes

    def _floor_for(self, target: str) -> float:
        """Confidence floor for one target class."""
        return (
            self.earbud_confidence_threshold if target == "earbuds" else self.confidence_threshold
        )

    @staticmethod
    def _rate_reliability(
        target: str,
        confidence: float,
        ear_anchored: bool,
        hand_corroborated: bool = False,
    ) -> str:
        """Rate how much weight a proctor should give this detection.

        Earbuds never rate "high" regardless of the model's confidence — the target
        is too small and too easily confused at webcam resolution for a confidence
        score alone to be trustworthy. A hand at the ear in the same frame is an
        independent signal and lifts a marginal call one step, but it cannot lift an
        earbud past "medium" either: two weak signals agreeing is still not proof.
        """
        if target == "earbuds":
            if ear_anchored and (confidence >= 0.45 or hand_corroborated):
                return "medium"
            return "low"
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

        def rank(detection: WearableDetection) -> tuple[int, float]:
            # An ear-anchored detection outranks a more confident unanchored one:
            # confidence from a box that is nowhere near an ear is confidence in the
            # wrong thing.
            return (1 if detection.ear_anchored else 0, detection.confidence)

        best: dict[str, WearableDetection] = {}
        for detection in result.detections:
            existing = best.get(detection.target)
            if existing is None or rank(detection) > rank(existing):
                best[detection.target] = detection
        result.detections = sorted(best.values(), key=lambda d: -d.confidence)
        return result
