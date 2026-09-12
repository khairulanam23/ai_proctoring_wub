"""Targeted tests for MultiSubjectTracker, persistent identity, and object association."""

import numpy as np
import pytest

from proctoring.detection.face_detector import FaceDetection
from proctoring.tracking.tracker import (
    MultiSubjectTracker,
    TrackedHand,
    TrackedObject,
    TrackedSubject,
    TrackState,
)


def _make_face(x: int, y: int, w: int, h: int, conf: float = 0.9) -> FaceDetection:
    raw = np.zeros((15,), dtype=np.float32)
    raw[0:4] = [x, y, w, h]
    raw[-1] = conf
    return FaceDetection(
        bbox=(x, y, w, h),
        confidence=conf,
        landmarks=[(x, y), (x + w, y), (x + w // 2, y + h // 2), (x, y + h), (x + w, y + h)],
        raw_detection=raw,
    )


def test_persistent_identity_and_track_stability():
    """Verify that a subject maintains the same track_id across consecutive frames."""
    tracker = MultiSubjectTracker()

    # Frame 1: Single face at (100, 100, 80, 80)
    f1 = _make_face(100, 100, 80, 80)
    emb1 = np.ones((128,), dtype=np.float32)
    subjects1 = tracker.update_faces(faces=[f1], embeddings=[emb1], timestamp=0.0, frame_index=1)
    assert len(subjects1) == 1
    track_id = subjects1[0].track_id
    assert subjects1[0].state == TrackState.TENTATIVE

    # Frame 2: Same face slightly shifted to (105, 102, 80, 80)
    f2 = _make_face(105, 102, 80, 80)
    emb2 = np.ones((128,), dtype=np.float32) * 0.98
    subjects2 = tracker.update_faces(faces=[f2], embeddings=[emb2], timestamp=0.25, frame_index=2)
    assert len(subjects2) == 1
    assert subjects2[0].track_id == track_id  # Stable ID
    assert subjects2[0].state == TrackState.CONFIRMED  # Confirmed after 2 frames
    assert subjects2[0].frames_seen == 2


def test_multiple_subjects_order_invariance():
    """Verify that swap in detection order does NOT swap persistent track IDs."""
    tracker = MultiSubjectTracker()

    # Frame 1: Candidate on left (100, 100), Bystander on right (400, 100)
    face_left = _make_face(100, 100, 80, 80)
    face_right = _make_face(400, 100, 80, 80)
    emb_cand = np.array([1.0] * 64 + [0.0] * 64, dtype=np.float32)
    emb_byst = np.array([0.0] * 64 + [1.0] * 64, dtype=np.float32)

    subjs1 = tracker.update_faces(
        faces=[face_left, face_right],
        embeddings=[emb_cand, emb_byst],
        similarities=[0.85, 0.20],  # Left is enrolled candidate
        timestamp=0.0,
        frame_index=1,
    )
    assert len(subjs1) == 2
    cand_track = next(s for s in subjs1 if s.is_enrolled)
    byst_track = next(s for s in subjs1 if not s.is_enrolled)
    cand_id = cand_track.track_id
    byst_id = byst_track.track_id

    # Frame 2: Detector returns bystander FIRST, candidate SECOND
    subjs2 = tracker.update_faces(
        faces=[face_right, face_left],  # Swapped detection order!
        embeddings=[emb_byst, emb_cand],
        similarities=[0.20, 0.85],
        timestamp=0.25,
        frame_index=2,
    )
    assert len(subjs2) == 2
    cand_track_2 = next(s for s in subjs2 if s.is_enrolled)
    byst_track_2 = next(s for s in subjs2 if not s.is_enrolled)

    # Track IDs must remain consistent with subjects, NOT detection order
    assert cand_track_2.track_id == cand_id
    assert byst_track_2.track_id == byst_id
    assert cand_track_2.bbox[0] < byst_track_2.bbox[0]  # Candidate is still on left


def test_track_timeout_and_removal():
    """Verify that absent tracks coast and then get deleted after max_misses."""
    tracker = MultiSubjectTracker(max_face_misses=3)

    # Frame 1: Face detected
    f1 = _make_face(100, 100, 80, 80)
    subjs = tracker.update_faces(faces=[f1], timestamp=0.0, frame_index=1)
    assert len(subjs) == 1
    t_id = subjs[0].track_id

    # Frames 2-4: Face missing (coasting)
    for i in range(1, 4):
        coasting = tracker.update_faces(faces=[], timestamp=i * 0.25, frame_index=i + 1)
        assert len(coasting) == 1
        assert coasting[0].state == TrackState.COASTING
        assert coasting[0].track_id == t_id

    # Frame 5: Exceeds max_face_misses -> Deleted
    active = tracker.update_faces(faces=[], timestamp=1.0, frame_index=5)
    assert len(active) == 0
    assert t_id not in tracker.subjects


def test_object_and_person_association():
    """Verify that a phone held near a hand is associated with the corresponding subject."""
    tracker = MultiSubjectTracker()

    # Track Candidate (track 1) on left and Secondary Person (track 2) on right
    cand_face = _make_face(100, 80, 80, 80)
    byst_face = _make_face(450, 80, 80, 80)
    tracker.update_faces(
        faces=[cand_face, bystander_face := byst_face],
        similarities=[0.88, 0.15],
        timestamp=0.0,
        frame_index=1,
    )

    # Candidate hand on left: (120, 250, 60, 60), Bystander hand on right: (460, 250, 60, 60)
    hands = tracker.update_hands(
        hand_bboxes=[(120, 250, 180, 310), (460, 250, 520, 310)],
        timestamp=0.0,
        frame_index=1,
    )
    assert len(hands) == 2

    # A cell phone is detected near the bystander's hand on the right: (470, 260, 510, 300)
    objects = tracker.update_objects(
        detected_objects=[
            {"class_name": "cell phone", "confidence": 0.92, "bbox": (470, 260, 510, 300)}
        ],
        timestamp=0.0,
        frame_index=1,
        hands=hands,
    )
    assert len(objects) == 1
    phone = objects[0]
    assert phone.interaction_type == "in_hand"
    # Must be associated with the bystander, NOT the candidate!
    assert phone.associated_subject_id == 2


def test_session_isolation_resets_all_tracks():
    """Verify that tracker.reset() clears all active tracks and counter state."""
    tracker = MultiSubjectTracker()

    f = _make_face(100, 100, 80, 80)
    tracker.update_faces([f], similarities=[0.85], timestamp=0.0, frame_index=1)
    tracker.update_objects(
        [{"class_name": "cell phone", "confidence": 0.8, "bbox": (100, 100, 150, 150)}],
        timestamp=0.0,
        frame_index=1,
    )
    assert len(tracker.subjects) == 1
    assert len(tracker.objects) == 1
    assert tracker._next_subject_id > 1

    tracker.reset()

    assert len(tracker.subjects) == 0
    assert len(tracker.objects) == 0
    assert len(tracker.hands) == 0
    assert tracker._next_subject_id == 1
    assert tracker._next_object_id == 1
    assert tracker._next_hand_id == 1
