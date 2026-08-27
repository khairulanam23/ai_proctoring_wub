"""Unit tests for identity resolution hardening and uncertainty differentiation."""

from unittest.mock import MagicMock

import numpy as np

from proctoring.config import SessionConfig
from proctoring.detection.face_detector import FaceDetection
from proctoring.engine_stages import StageCoordinator
from proctoring.observation import FaceStatus, FrameObservation
from proctoring.telemetry.performance import FrameTimingRecord


def test_identity_resolution_states():
    """Verify StageCoordinator distinguishes ENROLLED, IDENTITY_UNCERTAIN, IDENTITY_MISMATCH."""
    template = np.ones((128,), dtype=np.float32)
    config = SessionConfig(
        session_id="test_id",
        student_name="Candidate",
        enable_face_verification=True,
        face_match_threshold=0.363,
        reference_templates=[template],
    )

    coordinator = StageCoordinator(config=config)
    mock_verifier = MagicMock()
    coordinator.face_verifier = mock_verifier

    dummy_face = FaceDetection(
        bbox=(10, 10, 100, 100),
        confidence=0.95,
        landmarks=np.zeros((5, 2)),
        raw_detection=np.zeros((15,)),
    )
    timing = FrameTimingRecord(frame_index=0, timestamp_seconds=0.0)
    per_model = {}

    # Case 1: High similarity (0.60) >= 0.363 -> ENROLLED
    mock_verifier.extract_embedding.return_value = np.ones((128,), dtype=np.float32)
    mock_verifier.compute_similarity.return_value = 0.60
    obs1 = FrameObservation(frame_index=0, timestamp_seconds=0.0, iso_timestamp="")
    coordinator.resolve_identity(
        np.zeros((100, 100, 3), dtype=np.uint8), [dummy_face], obs1, 0.0, 0, timing, per_model
    )
    assert obs1.face_status == FaceStatus.ENROLLED
    assert obs1.identity_verified is True

    # Case 2: Borderline similarity (0.32) -> IDENTITY_UNCERTAIN
    mock_verifier.compute_similarity.return_value = 0.32
    obs2 = FrameObservation(frame_index=1, timestamp_seconds=1.0, iso_timestamp="")
    coordinator.resolve_identity(
        np.zeros((100, 100, 3), dtype=np.uint8), [dummy_face], obs2, 1.0, 1, timing, per_model
    )
    assert obs2.face_status == FaceStatus.IDENTITY_UNCERTAIN
    assert obs2.identity_verified is False

    # Case 3: Very low similarity (0.10) -> IDENTITY_MISMATCH
    mock_verifier.compute_similarity.return_value = 0.10
    obs3 = FrameObservation(frame_index=2, timestamp_seconds=2.0, iso_timestamp="")
    coordinator.resolve_identity(
        np.zeros((100, 100, 3), dtype=np.uint8), [dummy_face], obs3, 2.0, 2, timing, per_model
    )
    assert obs3.face_status == FaceStatus.IDENTITY_MISMATCH
    assert obs3.identity_verified is False
