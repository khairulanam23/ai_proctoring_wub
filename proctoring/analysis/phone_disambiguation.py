"""Contextual phone vs hand disambiguation module.

Differentiates genuine mobile phones from hand-only false positives (empty hands,
cupped hands, fingers) using bounding box geometry, hand landmark spatial relations,
grip heuristics, and multi-frame temporal confirmation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from proctoring.analysis.hands import HandAnalysisResult, HandObservation
from proctoring.detection.object_detector import DetectedObject
from proctoring.detection.object_relevance import ExamObjectCategory

LOGGER = logging.getLogger(__name__)


class PhoneClassification(str, Enum):
    """Categorization of phone detection candidate."""

    CONFIRMED_PHONE = "confirmed_phone"
    POSSIBLE_PHONE = "possible_phone"
    UNCERTAIN_CANDIDATE = "phone_candidate_uncertain"
    HAND_OBJECT_AMBIGUITY = "hand_object_ambiguity"
    HAND_FALSE_POSITIVE = "hand_false_positive"
    DISMISSED = "dismissed"


@dataclass
class PhoneDisambiguationResult:
    """Outcome of contextual phone-vs-hand analysis."""

    classification: PhoneClassification
    confidence: float
    raw_confidence: float
    reason: str
    aspect_ratio: float
    hand_iou: float = 0.0
    hand_gripping: bool = False
    temporal_confirmations: int = 1
    bbox: tuple[int, int, int, int] | None = None
    exam_category: ExamObjectCategory = ExamObjectCategory.PHONE
    domain_validation_status: str = "NOT_VALIDATED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.value,
            "confidence": round(self.confidence, 4),
            "raw_confidence": round(self.raw_confidence, 4),
            "reason": self.reason,
            "aspect_ratio": round(self.aspect_ratio, 3),
            "hand_iou": round(self.hand_iou, 3),
            "hand_gripping": self.hand_gripping,
            "temporal_confirmations": self.temporal_confirmations,
            "bbox": list(self.bbox) if self.bbox else None,
            "exam_category": self.exam_category.value,
            "domain_validation_status": self.domain_validation_status,
        }


class PhoneHandDisambiguator:
    """Disambiguates cell phone detections from hand-induced false positives."""

    def __init__(
        self,
        min_phone_aspect_ratio: float = 1.3,
        max_phone_aspect_ratio: float = 2.6,
        high_confidence_bypass: float = 0.85,
        hand_overlap_threshold: float = 0.65,
        min_temporal_frames: int = 2,
    ) -> None:
        self.min_phone_aspect_ratio = min_phone_aspect_ratio
        self.max_phone_aspect_ratio = max_phone_aspect_ratio
        # high_confidence_bypass retained for backwards compatibility; raw detector confidence
        # alone no longer bypasses contextual hand/geometry/temporal verification.
        self.high_confidence_bypass = high_confidence_bypass
        self.hand_overlap_threshold = hand_overlap_threshold
        self.min_temporal_frames = min_temporal_frames

        # Track history for temporal confirmation: list of (centroid, frame_index, count)
        self._tracks: list[dict[str, Any]] = []

    def disambiguate(
        self,
        phone_obj: DetectedObject | dict[str, Any],
        hand_analysis: HandAnalysisResult | None = None,
        frame_index: int = 0,
    ) -> PhoneDisambiguationResult:
        """Evaluate a phone detection against hand geometry and context."""
        if isinstance(phone_obj, dict):
            bbox = tuple(phone_obj["bbox"])
            raw_conf = float(phone_obj.get("confidence", 0.5))
        else:
            bbox = phone_obj.bbox
            raw_conf = float(phone_obj.confidence)

        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        long_side = max(width, height)
        short_side = min(width, height)
        aspect_ratio = long_side / short_side

        # 1. Contextual hand interaction and landmark analysis (Authoritative over raw detector confidence)
        max_iou = 0.0
        is_gripping = False
        empty_hand_false_positive = False

        if hand_analysis and hand_analysis.hands:
            for hand in hand_analysis.hands:
                iou = self._calculate_iou(bbox, hand.bbox)
                if iou > max_iou:
                    max_iou = iou

                # Check if hand surrounds or grips the phone candidate
                if iou > 0.20:
                    gripping, empty = self._evaluate_hand_grip(bbox, hand)
                    if gripping:
                        is_gripping = True
                    if empty:
                        empty_hand_false_positive = True

        # If box aligns with an empty open hand -> Contextual False Positive dismissal
        if empty_hand_false_positive:
            if max_iou > self.hand_overlap_threshold:
                return PhoneDisambiguationResult(
                    classification=PhoneClassification.HAND_FALSE_POSITIVE,
                    confidence=raw_conf * 0.3,
                    raw_confidence=raw_conf,
                    reason=f"Detection aligns with empty hand geometry without physical phone contrast (IoU {max_iou:.2f})",
                    aspect_ratio=aspect_ratio,
                    hand_iou=max_iou,
                    hand_gripping=False,
                    bbox=bbox,
                )
            elif max_iou > 0.35:
                return PhoneDisambiguationResult(
                    classification=PhoneClassification.HAND_OBJECT_AMBIGUITY,
                    confidence=raw_conf * 0.5,
                    raw_confidence=raw_conf,
                    reason=f"Ambiguous hand-object overlap without definitive device contrast (IoU {max_iou:.2f})",
                    aspect_ratio=aspect_ratio,
                    hand_iou=max_iou,
                    hand_gripping=False,
                    bbox=bbox,
                )

        # 3. Aspect ratio and stationery ambiguity check
        # Phones are rectangular slabs (~16:9 to 21:9, ratio ~1.6 - 2.3).
        # Squarish or moderately wide rectangular profiles (< 1.45) frequently conflate
        # scientific calculators, power banks, and small notebooks on the candidate desk.
        if aspect_ratio < 1.45 and not is_gripping:
            return PhoneDisambiguationResult(
                classification=PhoneClassification.UNCERTAIN_CANDIDATE,
                confidence=raw_conf * 0.55,
                raw_confidence=raw_conf,
                reason=(
                    f"Candidate aspect ratio ({aspect_ratio:.2f}) conflates smartphone with "
                    f"scientific calculator or rectangular stationery on desk; domain quality NOT_VALIDATED"
                ),
                aspect_ratio=aspect_ratio,
                hand_iou=max_iou,
                hand_gripping=False,
                temporal_confirmations=self._update_temporal_track(bbox, frame_index),
                bbox=bbox,
                exam_category=ExamObjectCategory.CALCULATOR,
                domain_validation_status="NOT_VALIDATED",
            )

        aspect_ok = self.min_phone_aspect_ratio <= aspect_ratio <= self.max_phone_aspect_ratio
        if not aspect_ok and raw_conf < 0.70:
            return PhoneDisambiguationResult(
                classification=PhoneClassification.UNCERTAIN_CANDIDATE,
                confidence=raw_conf * 0.6,
                raw_confidence=raw_conf,
                reason=f"Irregular aspect ratio ({aspect_ratio:.2f}) for standard smartphone",
                aspect_ratio=aspect_ratio,
                bbox=bbox,
                exam_category=ExamObjectCategory.OTHER,
                domain_validation_status="NOT_VALIDATED",
            )

        # 4. Temporal confirmation tracking
        confs = self._update_temporal_track(bbox, frame_index)

        # If hand is gripping an object or confirmed over multiple frames
        if is_gripping or confs >= self.min_temporal_frames:
            return PhoneDisambiguationResult(
                classification=PhoneClassification.CONFIRMED_PHONE,
                confidence=min(1.0, raw_conf * (1.1 if is_gripping else 1.0)),
                raw_confidence=raw_conf,
                reason=(
                    "Confirmed phone with hand grip interaction"
                    if is_gripping
                    else "Confirmed phone with multi-frame persistence"
                ),
                aspect_ratio=aspect_ratio,
                hand_iou=max_iou,
                hand_gripping=is_gripping,
                temporal_confirmations=confs,
                bbox=bbox,
            )

        # Single-frame candidate with valid aspect ratio -> POSSIBLE_PHONE
        if aspect_ok and raw_conf >= 0.50:
            return PhoneDisambiguationResult(
                classification=PhoneClassification.POSSIBLE_PHONE,
                confidence=raw_conf * 0.85,
                raw_confidence=raw_conf,
                reason="Single-frame phone candidate with standard rectangular geometry awaiting temporal confirmation",
                aspect_ratio=aspect_ratio,
                hand_iou=max_iou,
                hand_gripping=is_gripping,
                temporal_confirmations=confs,
                bbox=bbox,
            )

        # Otherwise marginal candidate
        return PhoneDisambiguationResult(
            classification=PhoneClassification.UNCERTAIN_CANDIDATE,
            confidence=raw_conf,
            raw_confidence=raw_conf,
            reason="Phone candidate lacks multi-frame confirmation or definitive grip evidence",
            aspect_ratio=aspect_ratio,
            hand_iou=max_iou,
            hand_gripping=is_gripping,
            temporal_confirmations=confs,
            bbox=bbox,
        )

    def _evaluate_hand_grip(
        self, phone_bbox: tuple[int, int, int, int], hand: HandObservation
    ) -> tuple[bool, bool]:
        """Analyze landmarks to determine if hand is gripping an object vs empty open hand."""
        if hand.landmarks is None or len(hand.landmarks) < 21:
            return False, False

        px1, py1, px2, py2 = phone_bbox
        phone_centre = np.array([(px1 + px2) / 2.0, (py1 + py2) / 2.0])

        # Palm center approx (wrist 0, index_mcp 5, pinky_mcp 17)
        wrist = hand.landmarks[0]
        index_mcp = hand.landmarks[5]
        pinky_mcp = hand.landmarks[17]
        palm_centre = (wrist + index_mcp + pinky_mcp) / 3.0

        # Measure distances from fingertips to palm
        # When gripping an object, fingers curl around it (shorter tip-to-palm than fully extended)
        fingertip_indices = [4, 8, 12, 16, 20]
        tip_distances = [
            float(np.linalg.norm(hand.landmarks[idx] - palm_centre))
            for idx in fingertip_indices
        ]
        avg_tip_dist = float(np.mean(tip_distances))
        hand_scale = float(np.linalg.norm(index_mcp - wrist))

        # Relative curl ratio
        curl_ratio = avg_tip_dist / max(1.0, hand_scale)

        # Distance between palm centre and phone centre
        phone_dist = float(np.linalg.norm(palm_centre - phone_centre)) / max(1.0, hand_scale)

        # Gripping: phone centre close to palm (<1.6 hand scale) and fingers curled around object (0.25 - 1.25)
        is_gripping = phone_dist < 1.6 and (0.25 <= curl_ratio <= 1.25)

        # Empty open/splayed hand: fingers extended (curl_ratio > 1.30)
        is_empty = curl_ratio >= 1.30 and phone_dist < 1.0

        return is_gripping, is_empty

    def _calculate_iou(
        self, box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
    ) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection = iw * ih
        if intersection == 0:
            return 0.0

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - intersection
        return float(intersection / max(1, union))

    def _update_temporal_track(
        self, bbox: tuple[int, int, int, int], frame_index: int
    ) -> int:
        """Track candidate across frames; prune dead tracks."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        matched_track = None
        for track in self._tracks:
            tcx, tcy = track["centroid"]
            dist = np.hypot(cx - tcx, cy - tcy)
            if dist < 60.0 and (frame_index - track["last_frame"]) <= 4:
                matched_track = track
                break

        if matched_track:
            matched_track["count"] += 1
            matched_track["centroid"] = (cx, cy)
            matched_track["last_frame"] = frame_index
            count = matched_track["count"]
        else:
            self._tracks.append(
                {"centroid": (cx, cy), "last_frame": frame_index, "count": 1}
            )
            count = 1

        # Prune old tracks (> 10 frames ago)
        self._tracks = [
            t for t in self._tracks if (frame_index - t["last_frame"]) <= 10
        ]
        return count

    def reset(self) -> None:
        self._tracks.clear()
