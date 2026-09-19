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

from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

# MediaPipe hand landmark indices.
_WRIST = 0
_INDEX_TIP = 8
_MIDDLE_TIP = 12
_FINGERTIPS = (4, 8, 12, 16, 20)


class HandState(str, Enum):
    """Categorical semantic state of candidate hands."""

    HAND_RESTING = "hand_resting"
    HAND_WRITING = "hand_writing"
    HAND_MOVING_ACROSS_PAPER = "hand_moving_across_paper"
    HAND_LIFTED_FROM_PAPER = "hand_lifted_from_paper"
    HAND_NEAR_FACE = "hand_near_face"
    HAND_NEAR_EAR = "hand_near_ear"
    HAND_LEAVING_WRITING_AREA = "hand_leaving_writing_area"
    PAPER_MANIPULATION = "paper_manipulation"
    UNKNOWN = "unknown"


@dataclass
class HandKinematics:
    """Temporal kinematics and trajectory features for a hand."""

    velocity: tuple[float, float] = (0.0, 0.0)  # (vx, vy) in px/s
    speed: float = 0.0  # px/s
    acceleration: float = 0.0  # px/s^2
    direction_deg: float = 0.0
    trajectory_length: int = 0
    writing_micro_oscillation: bool = False
    state: HandState = HandState.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "velocity": [round(self.velocity[0], 1), round(self.velocity[1], 1)],
            "speed": round(self.speed, 1),
            "acceleration": round(self.acceleration, 1),
            "direction_deg": round(self.direction_deg, 1),
            "trajectory_length": self.trajectory_length,
            "writing_micro_oscillation": self.writing_micro_oscillation,
            "state": self.state.value,
        }


@dataclass
class HandObservation:
    """One detected hand."""

    handedness: str  # "Left" / "Right" / "Unknown"
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: tuple[int, int]
    landmarks: np.ndarray | None = None  # (21, 2) pixel coordinates
    fingertip_points: list[tuple[int, int]] = field(default_factory=list)
    kinematics: HandKinematics = field(default_factory=HandKinematics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "handedness": self.handedness,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "centroid": list(self.centroid),
            "kinematics": self.kinematics.to_dict(),
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
    primary_hand_state: HandState = HandState.UNKNOWN
    """High-level semantic categorization of hand behavior (writing, resting, etc.)."""

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
            "primary_hand_state": self.primary_hand_state.value,
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
        device: str | None = None,
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
        self._requested_device = str(device or "cpu").lower()
        self._is_gpu_active = False
        self.device = "cpu"

        self.max_hands = int(max_hands)
        self.min_detection_confidence = float(min_detection_confidence)
        self.face_proximity_ratio = float(face_proximity_ratio)
        self.ear_proximity_ratio = float(ear_proximity_ratio)
        self.writing_area_top_ratio = float(writing_area_top_ratio)

        self._last_centroids: list[tuple[int, int]] = []
        self._kinematics_tracks: dict[str, list[dict[str, Any]]] = {}
        self._landmarker = None
        self._mp = None
        self.is_available = self._load()

    @property
    def is_gpu_accelerated(self) -> bool:
        """Whether the analyzer is currently executing on a GPU device."""
        return self._is_gpu_active

    def _load(self) -> bool:
        """Load the hand landmarker, degrading to unavailable rather than raising."""
        if not self.model_path.exists():
            return False
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            self._mp = mp

            # Try GPU delegate if requested and supported
            if self._requested_device in ("cuda", "cuda:0", "gpu"):
                try:
                    self._landmarker = vision.HandLandmarker.create_from_options(
                        vision.HandLandmarkerOptions(
                            base_options=BaseOptions(
                                model_asset_path=str(self.model_path),
                                delegate=BaseOptions.Delegate.GPU,
                            ),
                            num_hands=self.max_hands,
                            min_hand_detection_confidence=self.min_detection_confidence,
                        )
                    )
                    self._is_gpu_active = True
                    self.device = "cuda"
                    return True
                except Exception as gpu_exc:
                    LOGGER.info(
                        "MediaPipe HandLandmarker GPU delegate unavailable (%s); falling back to CPU XNNPACK.",
                        gpu_exc,
                    )

            # Standard CPU path (Google TFLite XNNPACK)
            self._landmarker = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(self.model_path),
                        delegate=BaseOptions.Delegate.CPU,
                    ),
                    num_hands=self.max_hands,
                    min_hand_detection_confidence=self.min_detection_confidence,
                )
            )
            self._is_gpu_active = False
            self.device = "cpu"
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
        paper_bbox: tuple[int, int, int, int] | None = None,
        timestamp_seconds: float | None = None,
        rgb_frame: np.ndarray | None = None,
    ) -> HandAnalysisResult:
        """Detect hands and, when face geometry is supplied, describe their position.

        Without ``face_bbox`` the analyzer still reports how many hands it saw but
        makes no proximity claims, since "near the face" is meaningless with no
        face to measure against.
        ``rgb_frame`` eliminates redundant color conversions across pipeline stages.
        """
        import time

        result = HandAnalysisResult()
        if not self.is_available or frame is None or frame.size == 0:
            return result

        t0 = time.perf_counter()
        try:
            image_data = rgb_frame if rgb_frame is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=image_data,
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

        # Temporal kinematics and behavioral state tracking
        hand_states: list[HandState] = []
        for hand in result.hands:
            self._update_kinematics(
                hand=hand,
                result=result,
                height=height,
                width=width,
                timestamp_seconds=timestamp_seconds,
                paper_bbox=paper_bbox,
            )
            hand_states.append(hand.kinematics.state)

        if hand_states:
            priority = [
                HandState.HAND_NEAR_EAR,
                HandState.HAND_NEAR_FACE,
                HandState.PAPER_MANIPULATION,
                HandState.HAND_LEAVING_WRITING_AREA,
                HandState.HAND_WRITING,
                HandState.HAND_MOVING_ACROSS_PAPER,
                HandState.HAND_RESTING,
                HandState.HAND_LIFTED_FROM_PAPER,
                HandState.UNKNOWN,
            ]
            for p in priority:
                if p in hand_states:
                    result.primary_hand_state = p
                    break
        else:
            result.primary_hand_state = HandState.UNKNOWN

        return result

    def _update_kinematics(
        self,
        hand: HandObservation,
        result: HandAnalysisResult,
        height: int,
        width: int,
        timestamp_seconds: float | None = None,
        paper_bbox: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Compute trajectory speed, acceleration, micro-oscillation and semantic state."""
        import time

        t_now = timestamp_seconds if timestamp_seconds is not None else time.time()
        track_key = hand.handedness if hand.handedness in ("Left", "Right") else "Primary"
        history = self._kinematics_tracks.setdefault(track_key, [])

        cx, cy = hand.centroid
        tip_pt = hand.fingertip_points[0] if hand.fingertip_points else (cx, cy)

        history.append({
            "timestamp": t_now,
            "centroid": (cx, cy),
            "tip": tip_pt,
        })
        if len(history) > 15:
            history.pop(0)

        vx, vy = 0.0, 0.0
        speed = 0.0
        accel = 0.0
        direction_deg = 0.0
        writing_micro_oscillation = False

        if len(history) >= 2:
            prev = history[-2]
            dt = max(0.01, t_now - prev["timestamp"])
            vx = (cx - prev["centroid"][0]) / dt
            vy = (cy - prev["centroid"][1]) / dt
            speed = float(np.hypot(vx, vy))
            direction_deg = float(np.degrees(np.arctan2(vy, vx)))

            if len(history) >= 3:
                prev2 = history[-3]
                dt_prev = max(0.01, prev["timestamp"] - prev2["timestamp"])
                prev_speed = float(np.hypot(
                    (prev["centroid"][0] - prev2["centroid"][0]) / dt_prev,
                    (prev["centroid"][1] - prev2["centroid"][1]) / dt_prev,
                ))
                accel = (speed - prev_speed) / dt

            # Analyze fingertip micro-movements for handwriting:
            # Fingertip oscillating back-and-forth while wrist/palm speed is low
            if len(history) >= 4 and speed < 40.0:
                tip_dys = [history[i]["tip"][1] - history[i - 1]["tip"][1] for i in range(1, len(history))]
                reversals = sum(1 for i in range(len(tip_dys) - 1) if tip_dys[i] * tip_dys[i + 1] < 0)
                if reversals >= 2:
                    writing_micro_oscillation = True

        # Semantic classification
        if result.hand_near_ear:
            state = HandState.HAND_NEAR_EAR
        elif result.hand_near_face:
            state = HandState.HAND_NEAR_FACE
        elif paper_bbox is not None and self._box_overlap(hand.bbox, paper_bbox) and speed > 100.0:
            state = HandState.PAPER_MANIPULATION
        elif hand.centroid[1] >= int(height * self.writing_area_top_ratio):
            if writing_micro_oscillation or result.writing_posture_detected:
                state = HandState.HAND_WRITING
            elif speed < 12.0:
                state = HandState.HAND_RESTING
            else:
                state = HandState.HAND_MOVING_ACROSS_PAPER
        else:
            if vy < -30.0:
                state = HandState.HAND_LEAVING_WRITING_AREA
            else:
                state = HandState.UNKNOWN

        hand.kinematics = HandKinematics(
            velocity=(vx, vy),
            speed=speed,
            acceleration=accel,
            direction_deg=direction_deg,
            trajectory_length=len(history),
            writing_micro_oscillation=writing_micro_oscillation,
            state=state,
        )

    def _box_overlap(
        self, box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
    ) -> bool:
        return not (box_a[2] < box_b[0] or box_a[0] > box_b[2] or box_a[3] < box_b[1] or box_a[1] > box_b[3])

    def reset(self) -> None:
        """Reset temporal state between exam sessions."""
        self._last_centroids = []
        self._kinematics_tracks.clear()

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
