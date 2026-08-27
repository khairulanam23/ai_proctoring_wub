"""Hand presence and position analysis via the MediaPipe Hand Landmarker.

Detects up to two hands as 21 landmarks each and describes where they are
relative to the candidate's face.  Three situations matter in an exam:

* **Hand at the ear** — consistent with holding a phone or adjusting an earpiece.
* **Hand covering the mouth** — consistent with speaking to someone off-camera.
* **No hand visible at all** — the candidate's hands are below the desk or out of
  frame, so nothing about what they are doing can be observed.

Every one of those has an innocent explanation: resting a head on a hand, scratching
an ear, or simply sitting with hands in the lap because the camera is angled high.
These are recorded as observations for a proctor with a snapshot attached, and the
strictness policy decides which are worth surfacing at all.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

# MediaPipe hand landmark indices.
_WRIST = 0
_INDEX_TIP = 8
_MIDDLE_TIP = 12
_FINGERTIPS = (4, 8, 12, 16, 20)


@dataclass
class HandObservation:
    """One detected hand."""

    handedness: str  # "Left" / "Right" / "Unknown"
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: tuple[int, int]
    landmarks: np.ndarray | None = None  # (21, 2) pixel coordinates
    fingertip_points: list[tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "handedness": self.handedness,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "centroid": list(self.centroid),
        }


@dataclass
class HandAnalysisResult:
    """Per-frame hand observation and its spatial relation to the face."""

    hands_detected: int | None = None
    hands: list[HandObservation] = field(default_factory=list)

    hand_near_face: bool = False
    hand_near_ear: bool = False
    hand_near_mouth: bool = False
    hands_visible: bool | None = None

    hand_in_writing_area: bool = False
    """True when at least one hand is positioned in the lower desk/writing quadrant."""
    writing_posture_detected: bool = False
    """True when hands are resting/active in writing area without obscuring face."""
    unusual_movement: bool = False
    """True when rapid hand movement dynamics are detected across consecutive frames."""

    nearest_hand_distance_ratio: float | None = None
    """Distance from the closest hand to the face centre, in face widths."""
    distance_to_mouth_ratio: float | None = None
    """Distance from the closest hand point to the mouth center, in face widths."""
    distance_to_ear_ratio: float | None = None
    """Distance from the closest hand point to the nearest ear region, in face widths."""

    inference_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hands_detected": self.hands_detected,
            "hands": [h.to_dict() for h in self.hands],
            "hand_near_face": self.hand_near_face,
            "hand_near_ear": self.hand_near_ear,
            "hand_near_mouth": self.hand_near_mouth,
            "hands_visible": self.hands_visible,
            "hand_in_writing_area": self.hand_in_writing_area,
            "writing_posture_detected": self.writing_posture_detected,
            "unusual_movement": self.unusual_movement,
            "nearest_hand_distance_ratio": (
                round(self.nearest_hand_distance_ratio, 3)
                if self.nearest_hand_distance_ratio is not None
                else None
            ),
            "distance_to_mouth_ratio": (
                round(self.distance_to_mouth_ratio, 3)
                if self.distance_to_mouth_ratio is not None
                else None
            ),
            "distance_to_ear_ratio": (
                round(self.distance_to_ear_ratio, 3)
                if self.distance_to_ear_ratio is not None
                else None
            ),
            "inference_ms": round(self.inference_ms, 2),
        }


class HandAnalyzer:
    """Locates hands and relates them to the face region."""

    DEFAULT_MODEL = "models/hand_landmarker.task"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        face_proximity_ratio: float = 1.0,
        ear_proximity_ratio: float = 0.55,
        writing_area_top_ratio: float = 0.45,
    ) -> None:
        """
        Args:
            face_proximity_ratio: A hand within this many face-widths of the face
                centre counts as "near the face".
            ear_proximity_ratio: Tighter radius, measured from each ear region, for
                the more specific "hand at the ear" observation.
            writing_area_top_ratio: Normalized vertical threshold (from frame top)
                below which hands are considered in the desk/writing workspace.
        """
        self.model_path = Path(model_path)
        self.max_hands = int(max_hands)
        self.min_detection_confidence = float(min_detection_confidence)
        self.face_proximity_ratio = float(face_proximity_ratio)
        self.ear_proximity_ratio = float(ear_proximity_ratio)
        self.writing_area_top_ratio = float(writing_area_top_ratio)

        self._last_centroids: list[tuple[int, int]] = []
        self._landmarker = None
        self._mp = None
        self.is_available = self._load()

    def _load(self) -> bool:
        """Load the hand landmarker, degrading to unavailable rather than raising."""
        if not self.model_path.exists():
            return False
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            self._mp = mp
            self._landmarker = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(self.model_path)),
                    num_hands=self.max_hands,
                    min_hand_detection_confidence=self.min_detection_confidence,
                )
            )
            return True
        except Exception as exc:
            LOGGER.info("Hand landmarker unavailable: %s", exc)
            return False

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception as exc:
                LOGGER.debug("Hand landmarker close failed: %s", exc)
            self._landmarker = None

    def analyze(
        self,
        frame: np.ndarray,
        face_bbox: tuple[int, int, int, int] | None = None,
        ear_regions: list[tuple[int, int, int, int]] | None = None,
        mouth_region: tuple[int, int, int, int] | None = None,
    ) -> HandAnalysisResult:
        """Detect hands and, when face geometry is supplied, describe their position.

        Without ``face_bbox`` the analyzer still reports how many hands it saw but
        makes no proximity claims, since "near the face" is meaningless with no
        face to measure against.
        """
        import time

        result = HandAnalysisResult()
        if not self.is_available or frame is None or frame.size == 0:
            return result

        t0 = time.perf_counter()
        try:
            import cv2

            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            detection = self._landmarker.detect(mp_image)
        except Exception as exc:
            LOGGER.warning("Hand detection failed: %s", exc)
            return result
        result.inference_ms = (time.perf_counter() - t0) * 1000.0

        height, width = frame.shape[:2]
        for index, hand in enumerate(detection.hand_landmarks or []):
            points = np.array([[lm.x * width, lm.y * height] for lm in hand], dtype=np.float32)
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)

            handedness, score = "Unknown", 1.0
            if detection.handedness and index < len(detection.handedness):
                category = detection.handedness[index][0]
                handedness, score = category.category_name, float(category.score)

            result.hands.append(
                HandObservation(
                    handedness=handedness,
                    confidence=score,
                    bbox=(
                        max(0, int(x1)),
                        max(0, int(y1)),
                        min(width, int(x2)),
                        min(height, int(y2)),
                    ),
                    centroid=(int(points[:, 0].mean()), int(points[:, 1].mean())),
                    landmarks=points,
                    fingertip_points=[
                        (int(points[i][0]), int(points[i][1]))
                        for i in _FINGERTIPS
                        if i < len(points)
                    ],
                )
            )

        result.hands_detected = len(result.hands)
        result.hands_visible = len(result.hands) > 0

        # Check writing area and hand motion dynamics
        current_centroids: list[tuple[int, int]] = []
        for hand in result.hands:
            current_centroids.append(hand.centroid)
            # Desk/writing workspace vertical boundary
            if hand.centroid[1] >= int(height * self.writing_area_top_ratio):
                result.hand_in_writing_area = True

        if self._last_centroids and current_centroids:
            # Check maximum single-frame displacement across matched hands
            max_disp = 0.0
            for curr_c in current_centroids:
                min_dist = min(
                    float(np.hypot(curr_c[0] - prev_c[0], curr_c[1] - prev_c[1]))
                    for prev_c in self._last_centroids
                )
                max_disp = max(max_disp, min_dist)
            if max_disp > (0.35 * width):
                result.unusual_movement = True

        self._last_centroids = current_centroids

        if face_bbox is not None and result.hands:
            self._relate_to_face(result, face_bbox, ear_regions or [], mouth_region)
            # Writing posture: hand in desk/writing zone without covering face
            if result.hand_in_writing_area and not result.hand_near_face:
                result.writing_posture_detected = True
        elif result.hand_in_writing_area:
            result.writing_posture_detected = True

        return result

    def reset(self) -> None:
        """Reset temporal state between exam sessions."""
        self._last_centroids = []

    def _relate_to_face(
        self,
        result: HandAnalysisResult,
        face_bbox: tuple[int, int, int, int],
        ear_regions: list[tuple[int, int, int, int]],
        mouth_region: tuple[int, int, int, int] | None,
    ) -> None:
        """Measure hand position against face, ear and mouth regions.

        Distances are normalised by face width so the result does not depend on how
        close the candidate sits to the camera.
        """
        fx1, fy1, fx2, fy2 = face_bbox
        face_centre = np.array([(fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0])
        face_width = max(1.0, float(fx2 - fx1))

        face_distances: list[float] = []
        mouth_distances: list[float] = []
        ear_distances: list[float] = []

        mouth_centre = (
            np.array(
                [
                    (mouth_region[0] + mouth_region[2]) / 2.0,
                    (mouth_region[1] + mouth_region[3]) / 2.0,
                ]
            )
            if mouth_region
            else None
        )
        ear_centres = [np.array([(e[0] + e[2]) / 2.0, (e[1] + e[3]) / 2.0]) for e in ear_regions]

        for hand in result.hands:
            # Use the closest point of the hand, not its centroid: a hand reaching
            # toward the ear touches it with a fingertip long before its centre.
            candidate_points = [np.array(hand.centroid, dtype=np.float32)]
            candidate_points.extend(np.array(p, dtype=np.float32) for p in hand.fingertip_points)

            face_distances.append(
                min(
                    float(np.linalg.norm(point - face_centre)) / face_width
                    for point in candidate_points
                )
            )

            if mouth_centre is not None:
                mouth_distances.append(
                    min(
                        float(np.linalg.norm(point - mouth_centre)) / face_width
                        for point in candidate_points
                    )
                )

            if ear_centres:
                for ec in ear_centres:
                    ear_distances.append(
                        min(
                            float(np.linalg.norm(point - ec)) / face_width
                            for point in candidate_points
                        )
                    )

            for point in candidate_points:
                if self._point_in_box(point, face_bbox, pad_ratio=0.15):
                    result.hand_near_face = True
                if mouth_region and self._point_in_box(point, mouth_region, pad_ratio=0.4):
                    result.hand_near_mouth = True
                for ear in ear_regions:
                    if self._point_in_box(point, ear, pad_ratio=0.5):
                        result.hand_near_ear = True

        if face_distances:
            result.nearest_hand_distance_ratio = min(face_distances)
            if result.nearest_hand_distance_ratio <= self.face_proximity_ratio:
                result.hand_near_face = True

        if mouth_distances:
            result.distance_to_mouth_ratio = min(mouth_distances)

        if ear_distances:
            result.distance_to_ear_ratio = min(ear_distances)

    @staticmethod
    def _point_in_box(
        point: np.ndarray,
        box: tuple[int, int, int, int],
        pad_ratio: float = 0.0,
    ) -> bool:
        """Is the point inside the box, expanded by a proportional margin?"""
        x1, y1, x2, y2 = box
        pad_x = (x2 - x1) * pad_ratio
        pad_y = (y2 - y1) * pad_ratio
        return (x1 - pad_x) <= point[0] <= (x2 + pad_x) and (y1 - pad_y) <= point[1] <= (y2 + pad_y)
