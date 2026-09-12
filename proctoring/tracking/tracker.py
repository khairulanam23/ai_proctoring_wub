"""Multi-subject persistent tracker for faces, objects, and hands.

Ensures persistent frame-to-frame identities, eliminating the fragile assumption that
detections[0] always represents the same candidate across frames. Handles track creation,
updates, temporal coasting, disappearance, timeouts, and multi-object association.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

import numpy as np

from proctoring.detection.face_detector import FaceDetection


class TrackState(str, Enum):
    """Lifecycle states of a tracked entity."""

    TENTATIVE = "TENTATIVE"  # Newly observed, awaiting confirmation
    CONFIRMED = "CONFIRMED"  # Stable, established track
    COASTING = "COASTING"  # Temporarily missed, predicted/held in window
    DELETED = "DELETED"  # Timed out, pending garbage collection


@dataclass
class TrackedSubject:
    """Persistent subject (person/face) tracked across video frames."""

    track_id: int
    is_enrolled: bool
    similarity_score: float | None
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    first_seen: float
    last_seen: float
    frames_seen: int = 1
    consecutive_misses: int = 0
    state: TrackState = TrackState.TENTATIVE
    embedding: np.ndarray | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    associated_objects: list[int] = field(default_factory=list)
    associated_hands: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "is_enrolled": self.is_enrolled,
            "similarity_score": round(self.similarity_score, 4)
            if self.similarity_score is not None
            else None,
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "first_seen": round(self.first_seen, 3),
            "last_seen": round(self.last_seen, 3),
            "frames_seen": self.frames_seen,
            "state": self.state.value,
            "associated_objects": list(self.associated_objects),
            "associated_hands": list(self.associated_hands),
        }


@dataclass
class TrackedObject:
    """Persistent physical object tracked across video frames."""

    track_id: int
    class_name: str
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    first_seen: float
    last_seen: float
    frames_seen: int = 1
    consecutive_misses: int = 0
    state: TrackState = TrackState.TENTATIVE
    associated_subject_id: int | None = None
    interaction_type: str = "unassociated"  # "in_hand", "near_subject", "desk", "unassociated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "first_seen": round(self.first_seen, 3),
            "last_seen": round(self.last_seen, 3),
            "frames_seen": self.frames_seen,
            "state": self.state.value,
            "associated_subject_id": self.associated_subject_id,
            "interaction_type": self.interaction_type,
        }


@dataclass
class TrackedHand:
    """Persistent hand tracked across video frames."""

    track_id: int
    handedness: str  # "Left", "Right", "Unknown"
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    state: str  # "writing", "resting", "moving", "unknown"
    first_seen: float
    last_seen: float
    frames_seen: int = 1
    consecutive_misses: int = 0
    associated_subject_id: int | None = None


class MultiSubjectTracker:
    """Frame-to-frame persistent multi-subject and object association tracker.

    Eliminates arbitrary ordering of detections across frames. Associates detected
    phones, papers, and hands to specific subject tracks so that an object held by
    a secondary person is not falsely attributed to the candidate.
    """

    def __init__(
        self,
        face_match_threshold: float = 0.50,
        face_iou_threshold: float = 0.25,
        object_iou_threshold: float = 0.20,
        max_face_misses: int = 8,  # ~2.0s at 4 fps
        max_object_misses: int = 6,  # ~1.5s at 4 fps
        confirmation_frames: int = 2,
    ) -> None:
        self.face_match_threshold = float(face_match_threshold)
        self.face_iou_threshold = float(face_iou_threshold)
        self.object_iou_threshold = float(object_iou_threshold)
        self.max_face_misses = int(max_face_misses)
        self.max_object_misses = int(max_object_misses)
        self.confirmation_frames = int(confirmation_frames)

        # Monotonically increasing track ID counters
        self._next_subject_id: int = 1
        self._next_object_id: int = 1
        self._next_hand_id: int = 1

        # Active tracking tables
        self.subjects: dict[int, TrackedSubject] = {}
        self.objects: dict[int, TrackedObject] = {}
        self.hands: dict[int, TrackedHand] = {}

    def reset(self) -> None:
        """Clear all tracks and reset ID counters for clean session isolation."""
        self.subjects.clear()
        self.objects.clear()
        self.hands.clear()
        self._next_subject_id = 1
        self._next_object_id = 1
        self._next_hand_id = 1

    @staticmethod
    def box_iou(
        box_a: tuple[int, int, int, int] | None,
        box_b: tuple[int, int, int, int] | None,
    ) -> float:
        """Compute Intersection over Union between two (x1, y1, x2, y2) pixel boxes."""
        if box_a is None or box_b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection = float(iw * ih)

        area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
        area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def box_distance(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        """Euclidean distance between bounding box centroids."""
        c_ax = (box_a[0] + box_a[2]) / 2.0
        c_ay = (box_a[1] + box_a[3]) / 2.0
        c_bx = (box_b[0] + box_b[2]) / 2.0
        c_by = (box_b[1] + box_b[3]) / 2.0
        return math.hypot(c_ax - c_bx, c_ay - c_by)

    @staticmethod
    def cosine_similarity(v1: np.ndarray | None, v2: np.ndarray | None) -> float:
        """Compute cosine similarity between two 1D feature vectors."""
        if v1 is None or v2 is None:
            return 0.0
        dot = float(np.dot(v1.flatten(), v2.flatten()))
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm1 * norm2)))

    # ------------------------------------------------------------------
    # Subject (Face) Tracking
    # ------------------------------------------------------------------

    def update_faces(
        self,
        faces: list[FaceDetection],
        embeddings: list[np.ndarray | None] | None = None,
        similarities: list[float | None] | None = None,
        timestamp: float = 0.0,
        frame_index: int = 0,
    ) -> list[TrackedSubject]:
        """Update subject tracks using spatial overlap and facial embedding matching."""
        if embeddings is None:
            embeddings = [None] * len(faces)
        if similarities is None:
            similarities = [None] * len(faces)

        active_track_ids = [
            tid
            for tid, subj in self.subjects.items()
            if subj.state in (TrackState.CONFIRMED, TrackState.TENTATIVE, TrackState.COASTING)
        ]

        # Cost matrix: pair existing tracks with detections
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        if active_track_ids and faces:
            pairs: list[tuple[float, int, int]] = []
            for d_idx, face in enumerate(faces):
                f_box = face.bbox_xyxy if hasattr(face, "bbox_xyxy") else (
                    face.bbox[0],
                    face.bbox[1],
                    face.bbox[0] + face.bbox[2],
                    face.bbox[1] + face.bbox[3],
                )
                emb = embeddings[d_idx]

                for t_id in active_track_ids:
                    track = self.subjects[t_id]
                    iou = self.box_iou(f_box, track.bbox)

                    # Biometric similarity if embeddings exist
                    bio_sim = 0.0
                    if emb is not None and track.embedding is not None:
                        bio_sim = self.cosine_similarity(emb, track.embedding)

                    # Score combination: IoU (spatial) + Biometric similarity
                    if bio_sim > 0.0:
                        affinity = 0.5 * iou + 0.5 * bio_sim
                    else:
                        affinity = iou

                    # Distance penalty if too far
                    if affinity > 0.15 or iou >= self.face_iou_threshold or bio_sim >= 0.55:
                        pairs.append((affinity, t_id, d_idx))

            # Greedy matching from highest affinity
            pairs.sort(key=lambda p: p[0], reverse=True)
            for score, t_id, d_idx in pairs:
                if t_id not in matched_tracks and d_idx not in matched_detections:
                    matched_tracks.add(t_id)
                    matched_detections.add(d_idx)
                    # Update matched track
                    track = self.subjects[t_id]
                    face = faces[d_idx]
                    f_box = face.bbox_xyxy if hasattr(face, "bbox_xyxy") else (
                        face.bbox[0],
                        face.bbox[1],
                        face.bbox[0] + face.bbox[2],
                        face.bbox[1] + face.bbox[3],
                    )
                    track.bbox = f_box
                    track.confidence = face.confidence
                    track.last_seen = timestamp
                    track.frames_seen += 1
                    track.consecutive_misses = 0
                    track.history.append((timestamp, f_box))

                    sim = similarities[d_idx]
                    if sim is not None:
                        track.similarity_score = sim
                        if sim >= self.face_match_threshold:
                            track.is_enrolled = True

                    emb = embeddings[d_idx]
                    if emb is not None:
                        if track.embedding is None:
                            track.embedding = emb
                        else:
                            # Moving average embedding update
                            track.embedding = 0.8 * track.embedding + 0.2 * emb

                    if track.frames_seen >= self.confirmation_frames:
                        track.state = TrackState.CONFIRMED

        # Handle unmatched detections -> create new tracks
        for d_idx, face in enumerate(faces):
            if d_idx not in matched_detections:
                f_box = face.bbox_xyxy if hasattr(face, "bbox_xyxy") else (
                    face.bbox[0],
                    face.bbox[1],
                    face.bbox[0] + face.bbox[2],
                    face.bbox[1] + face.bbox[3],
                )
                sim = similarities[d_idx]
                is_enrolled = bool(sim is not None and sim >= self.face_match_threshold)
                tid = self._next_subject_id
                self._next_subject_id += 1

                new_track = TrackedSubject(
                    track_id=tid,
                    is_enrolled=is_enrolled,
                    similarity_score=sim,
                    bbox=f_box,
                    confidence=face.confidence,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    frames_seen=1,
                    consecutive_misses=0,
                    state=TrackState.CONFIRMED if is_enrolled else TrackState.TENTATIVE,
                    embedding=embeddings[d_idx],
                )
                new_track.history.append((timestamp, f_box))
                self.subjects[tid] = new_track

        # Handle unmatched active tracks -> coast or delete
        for t_id in active_track_ids:
            if t_id not in matched_tracks:
                track = self.subjects[t_id]
                track.consecutive_misses += 1
                if track.consecutive_misses > self.max_face_misses:
                    track.state = TrackState.DELETED
                else:
                    track.state = TrackState.COASTING

        # Prune deleted tracks
        deleted_keys = [tid for tid, s in self.subjects.items() if s.state == TrackState.DELETED]
        for dk in deleted_keys:
            del self.subjects[dk]

        return [
            s
            for s in self.subjects.values()
            if s.state in (TrackState.CONFIRMED, TrackState.TENTATIVE, TrackState.COASTING)
        ]

    # ------------------------------------------------------------------
    # Object Tracking & Spatial Association
    # ------------------------------------------------------------------

    def update_objects(
        self,
        detected_objects: list[dict[str, Any]],
        timestamp: float,
        frame_index: int,
        subjects: list[TrackedSubject] | None = None,
        hands: list[TrackedHand] | None = None,
    ) -> list[TrackedObject]:
        """Update object tracks across frames and associate them to subjects/hands."""
        if subjects is None:
            subjects = list(self.subjects.values())
        if hands is None:
            hands = list(self.hands.values())

        active_obj_ids = [
            oid
            for oid, obj in self.objects.items()
            if obj.state in (TrackState.CONFIRMED, TrackState.TENTATIVE, TrackState.COASTING)
        ]

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        if active_obj_ids and detected_objects:
            pairs: list[tuple[float, int, int]] = []
            for d_idx, det in enumerate(detected_objects):
                raw_bbox = det.get("bbox")
                if raw_bbox is None:
                    continue
                d_box = tuple(int(v) for v in raw_bbox)
                c_name = str(det.get("class_name", "")).strip().lower()

                for o_id in active_obj_ids:
                    track = self.objects[o_id]
                    if track.class_name.lower() == c_name:
                        iou = self.box_iou(d_box, track.bbox)
                        if iou >= self.object_iou_threshold:
                            pairs.append((iou, o_id, d_idx))

            pairs.sort(key=lambda p: p[0], reverse=True)
            for score, o_id, d_idx in pairs:
                if o_id not in matched_tracks and d_idx not in matched_detections:
                    matched_tracks.add(o_id)
                    matched_detections.add(d_idx)
                    track = self.objects[o_id]
                    det = detected_objects[d_idx]
                    track.bbox = tuple(int(v) for v in det["bbox"])
                    track.confidence = float(det.get("confidence", 0.0))
                    track.last_seen = timestamp
                    track.frames_seen += 1
                    track.consecutive_misses = 0
                    if track.frames_seen >= self.confirmation_frames:
                        track.state = TrackState.CONFIRMED

        # Handle unmatched detections -> create new object tracks
        for d_idx, det in enumerate(detected_objects):
            if d_idx not in matched_detections:
                raw_bbox = det.get("bbox")
                if raw_bbox is None:
                    continue
                d_box = tuple(int(v) for v in raw_bbox)
                c_name = str(det.get("class_name", "")).strip().lower()
                oid = self._next_object_id
                self._next_object_id += 1

                new_track = TrackedObject(
                    track_id=oid,
                    class_name=c_name,
                    bbox=d_box,
                    confidence=float(det.get("confidence", 0.0)),
                    first_seen=timestamp,
                    last_seen=timestamp,
                    frames_seen=1,
                    consecutive_misses=0,
                    state=TrackState.TENTATIVE,
                )
                self.objects[oid] = new_track

        # Handle unmatched object tracks -> coast or delete
        for o_id in active_obj_ids:
            if o_id not in matched_tracks:
                track = self.objects[o_id]
                track.consecutive_misses += 1
                if track.consecutive_misses > self.max_object_misses:
                    track.state = TrackState.DELETED
                else:
                    track.state = TrackState.COASTING

        # Prune deleted tracks
        deleted_keys = [oid for oid, o in self.objects.items() if o.state == TrackState.DELETED]
        for dk in deleted_keys:
            del self.objects[dk]

        # Associate objects with hands and subjects
        active_objects = [
            o
            for o in self.objects.values()
            if o.state in (TrackState.CONFIRMED, TrackState.TENTATIVE, TrackState.COASTING)
        ]
        self._associate_objects(active_objects, subjects, hands)

        return active_objects

    def _associate_objects(
        self,
        objects: list[TrackedObject],
        subjects: list[TrackedSubject],
        hands: list[TrackedHand],
    ) -> None:
        """Determine which subject/hand an object belongs to."""
        # Reset subject associated object lists
        for subj in subjects:
            subj.associated_objects.clear()

        for obj in objects:
            assigned_subj_id: int | None = None
            interaction = "unassociated"

            # 1. Check hand overlap / spatial proximity
            best_hand_dist = float("inf")
            best_hand: TrackedHand | None = None
            for hand in hands:
                iou = self.box_iou(obj.bbox, hand.bbox)
                dist = self.box_distance(obj.bbox, hand.bbox)
                if iou > 0.05 or dist < 120.0:
                    if dist < best_hand_dist:
                        best_hand_dist = dist
                        best_hand = hand

            if best_hand is not None:
                assigned_subj_id = best_hand.associated_subject_id
                interaction = "in_hand"

            # 2. If not associated via hands, associate via nearest subject head/torso
            if assigned_subj_id is None and subjects:
                best_dist = float("inf")
                nearest_subj: TrackedSubject | None = None
                for subj in subjects:
                    dist = self.box_distance(obj.bbox, subj.bbox)
                    if dist < best_dist:
                        best_dist = dist
                        nearest_subj = subj

                if nearest_subj is not None and best_dist < 400.0:
                    assigned_subj_id = nearest_subj.track_id
                    interaction = "near_subject"

            obj.associated_subject_id = assigned_subj_id
            obj.interaction_type = interaction

            # Link back to subject
            if assigned_subj_id is not None and assigned_subj_id in self.subjects:
                self.subjects[assigned_subj_id].associated_objects.append(obj.track_id)

    # ------------------------------------------------------------------
    # Hand Tracking & Association
    # ------------------------------------------------------------------

    def update_hands(
        self,
        hand_bboxes: list[tuple[int, int, int, int]],
        timestamp: float,
        frame_index: int,
        subjects: list[TrackedSubject] | None = None,
        states: list[str] | None = None,
    ) -> list[TrackedHand]:
        """Update hand tracks and associate them to subjects."""
        if subjects is None:
            subjects = list(self.subjects.values())
        if states is None:
            states = ["unknown"] * len(hand_bboxes)

        active_hand_ids = [
            hid
            for hid, h in self.hands.items()
            if h.consecutive_misses <= 4
        ]

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        if active_hand_ids and hand_bboxes:
            pairs: list[tuple[float, int, int]] = []
            for d_idx, h_box in enumerate(hand_bboxes):
                for h_id in active_hand_ids:
                    track = self.hands[h_id]
                    iou = self.box_iou(h_box, track.bbox)
                    dist = self.box_distance(h_box, track.bbox)
                    if iou > 0.15 or dist < 80.0:
                        score = iou + 1.0 / (1.0 + dist)
                        pairs.append((score, h_id, d_idx))

            pairs.sort(key=lambda p: p[0], reverse=True)
            for score, h_id, d_idx in pairs:
                if h_id not in matched_tracks and d_idx not in matched_detections:
                    matched_tracks.add(h_id)
                    matched_detections.add(d_idx)
                    track = self.hands[h_id]
                    track.bbox = hand_bboxes[d_idx]
                    track.state = states[d_idx]
                    track.last_seen = timestamp
                    track.frames_seen += 1
                    track.consecutive_misses = 0

        # Handle unmatched detections
        for d_idx, h_box in enumerate(hand_bboxes):
            if d_idx not in matched_detections:
                hid = self._next_hand_id
                self._next_hand_id += 1

                new_hand = TrackedHand(
                    track_id=hid,
                    handedness="Unknown",
                    bbox=h_box,
                    state=states[d_idx],
                    first_seen=timestamp,
                    last_seen=timestamp,
                    frames_seen=1,
                    consecutive_misses=0,
                )
                self.hands[hid] = new_hand

        # Handle unmatched hand tracks
        for h_id in active_hand_ids:
            if h_id not in matched_tracks:
                track = self.hands[h_id]
                track.consecutive_misses += 1
                if track.consecutive_misses > 6:
                    del self.hands[h_id]

        # Associate hands with nearest subject
        for hand in self.hands.values():
            if subjects:
                best_dist = float("inf")
                best_subj = None
                for subj in subjects:
                    dist = self.box_distance(hand.bbox, subj.bbox)
                    if dist < best_dist:
                        best_dist = dist
                        best_subj = subj
                hand.associated_subject_id = best_subj.track_id if best_subj else None

        return list(self.hands.values())

    def get_enrolled_subject(self) -> TrackedSubject | None:
        """Find the subject track confirmed as the enrolled candidate."""
        enrolled = [s for s in self.subjects.values() if s.is_enrolled and s.state != TrackState.DELETED]
        if not enrolled:
            return None
        # Highest similarity score takes precedence
        return max(enrolled, key=lambda s: s.similarity_score or 0.0)
