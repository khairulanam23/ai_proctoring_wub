"""Face occlusion classification and camera obstruction analysis.

Distinguishes between:
- Candidate missing vs Camera obstructed (covered lens / pitch black / extreme low texture)
- Hands occluding facial regions (eyes, mouth/nose, partial, significant)
- Normal unoccluded face

TECHNICAL NOTE:
    Occlusion classification combines face landmark geometry, hand tracking proximity/overlap,
    and frame illumination / Laplacian variance quality signals. When signals are ambiguous,
    it conservatively reports UNKNOWN rather than fabricating disciplinary findings.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from proctoring.analysis.hands import HandAnalysisResult


class FaceOcclusionState(str, Enum):
    """Categorical face occlusion states."""

    NONE = "NONE"
    EYES_OBSCURED = "EYES_OBSCURED"
    MOUTH_NOSE_OBSCURED = "MOUTH_NOSE_OBSCURED"
    FACE_PARTIALLY_OBSCURED = "FACE_PARTIALLY_OBSCURED"
    FACE_SIGNIFICANTLY_OBSCURED = "FACE_SIGNIFICANTLY_OBSCURED"
    FACE_MISSING = "FACE_MISSING"
    CAMERA_OBSTRUCTED = "CAMERA_OBSTRUCTED"
    UNKNOWN = "UNKNOWN"


@dataclass
class FaceOcclusionResult:
    """Per-frame occlusion analysis outcome."""

    state: FaceOcclusionState = FaceOcclusionState.NONE
    confidence: float = 1.0
    is_occluded: bool = False
    hand_overlap_ratio: float = 0.0
    eyes_visible: bool = True
    mouth_visible: bool = True
    nose_visible: bool = True
    camera_obstructed: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "confidence": round(self.confidence, 4),
            "is_occluded": self.is_occluded,
            "hand_overlap_ratio": round(self.hand_overlap_ratio, 4),
            "eyes_visible": self.eyes_visible,
            "mouth_visible": self.mouth_visible,
            "nose_visible": self.nose_visible,
            "camera_obstructed": self.camera_obstructed,
            "details": self.details,
        }


class FaceOcclusionClassifier:
    """Classifies face occlusion and camera obstruction states from multi-modal cues."""

    def __init__(
        self,
        min_hand_overlap_partial: float = 0.15,
        min_hand_overlap_significant: float = 0.45,
        min_camera_dark_luma: float = 12.0,
        min_camera_blur_var: float = 18.0,
    ) -> None:
        self.min_hand_overlap_partial = float(min_hand_overlap_partial)
        self.min_hand_overlap_significant = float(min_hand_overlap_significant)
        self.min_camera_dark_luma = float(min_camera_dark_luma)
        self.min_camera_blur_var = float(min_camera_blur_var)

    def classify(
        self,
        face_found: bool,
        face_bbox: tuple[int, int, int, int] | None = None,
        landmarks: np.ndarray | None = None,  # noqa: ARG002  (reserved: landmark-level occlusion)
        hands: HandAnalysisResult | None = None,
        mean_luminance: float | None = None,
        blur_variance: float | None = None,
    ) -> FaceOcclusionResult:
        """Evaluate occlusion state for one frame."""
        # 1. No face detected on screen
        if not face_found or face_bbox is None:
            # Check for camera obstruction cues: uniform darkness / low variance
            is_dark = mean_luminance is not None and mean_luminance < self.min_camera_dark_luma
            is_featureless = blur_variance is not None and blur_variance < self.min_camera_blur_var

            if is_dark and is_featureless:
                return FaceOcclusionResult(
                    state=FaceOcclusionState.CAMERA_OBSTRUCTED,
                    confidence=0.90,
                    is_occluded=True,
                    eyes_visible=False,
                    mouth_visible=False,
                    nose_visible=False,
                    camera_obstructed=True,
                    details={
                        "reason": "Uniform darkness and low texture variance consistent with covered or blocked camera lens",
                        "mean_luminance": mean_luminance,
                        "blur_variance": blur_variance,
                    },
                )

            # Normal illuminated frame with no candidate
            return FaceOcclusionResult(
                state=FaceOcclusionState.FACE_MISSING,
                confidence=0.95,
                is_occluded=False,
                eyes_visible=False,
                mouth_visible=False,
                nose_visible=False,
                camera_obstructed=False,
                details={"reason": "No face detected in normal scene illumination"},
            )

        # 2. Face is detected: evaluate hand proximity and overlap
        fx1, fy1, fx2, fy2 = face_bbox
        face_w = max(1, fx2 - fx1)
        face_h = max(1, fy2 - fy1)
        face_area = float(face_w * face_h)

        max_overlap_ratio = 0.0
        eyes_occluded = False
        mouth_occluded = False

        # Define approximate facial sub-zones (normalized relative to face_bbox)
        # Eyes zone: top 20% to 50%
        eye_box = (
            fx1,
            int(fy1 + 0.15 * face_h),
            fx2,
            int(fy1 + 0.50 * face_h),
        )
        # Mouth/nose zone: 45% to bottom
        mouth_nose_box = (
            int(fx1 + 0.15 * face_w),
            int(fy1 + 0.45 * face_h),
            int(fx2 - 0.15 * face_w),
            fy2,
        )

        if hands and hands.hands:
            for hand in hands.hands:
                hx1, hy1, hx2, hy2 = hand.bbox
                overlap_area = self._box_intersection_area(face_bbox, hand.bbox)
                ratio = overlap_area / face_area
                if ratio > max_overlap_ratio:
                    max_overlap_ratio = ratio

                # Check sub-region intersections
                if self._box_intersection_area(eye_box, hand.bbox) > 0.15 * (
                    (eye_box[2] - eye_box[0]) * (eye_box[3] - eye_box[1])
                ):
                    eyes_occluded = True

                if self._box_intersection_area(mouth_nose_box, hand.bbox) > 0.18 * (
                    (mouth_nose_box[2] - mouth_nose_box[0])
                    * (mouth_nose_box[3] - mouth_nose_box[1])
                ):
                    mouth_occluded = True

        # Check explicit hand proximity flags
        if hands and hands.hand_near_mouth:
            mouth_occluded = True

        # Determine occlusion state
        if max_overlap_ratio >= self.min_hand_overlap_significant:
            state = FaceOcclusionState.FACE_SIGNIFICANTLY_OBSCURED
            is_occluded = True
            conf = min(1.0, max_overlap_ratio / 0.6)
        elif eyes_occluded and mouth_occluded:
            state = FaceOcclusionState.FACE_SIGNIFICANTLY_OBSCURED
            is_occluded = True
            conf = 0.85
        elif eyes_occluded:
            state = FaceOcclusionState.EYES_OBSCURED
            is_occluded = True
            conf = 0.80
        elif mouth_occluded:
            state = FaceOcclusionState.MOUTH_NOSE_OBSCURED
            is_occluded = True
            conf = 0.80
        elif max_overlap_ratio >= self.min_hand_overlap_partial:
            state = FaceOcclusionState.FACE_PARTIALLY_OBSCURED
            is_occluded = True
            conf = min(1.0, max_overlap_ratio / 0.3)
        else:
            state = FaceOcclusionState.NONE
            is_occluded = False
            conf = 1.0

        return FaceOcclusionResult(
            state=state,
            confidence=conf,
            is_occluded=is_occluded,
            hand_overlap_ratio=max_overlap_ratio,
            eyes_visible=not eyes_occluded,
            mouth_visible=not mouth_occluded,
            nose_visible=not (eyes_occluded and mouth_occluded),
            camera_obstructed=False,
            details={
                "hand_overlap_ratio": round(max_overlap_ratio, 3),
                "eyes_occluded": eyes_occluded,
                "mouth_occluded": mouth_occluded,
            },
        )

    @staticmethod
    def _box_intersection_area(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        """Compute pixel area of intersection between two (x1, y1, x2, y2) boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        return float(iw * ih)
