"""Tests for Phase 9 event validation layer, candidate lifecycle state machine, and evidence quality validation."""

import tempfile
from pathlib import Path

import numpy as np

from proctoring.config import SessionConfig
from proctoring.core.events import EventType
from proctoring.evidence.quality import EvidenceQualityValidator
from proctoring.temporal.lifecycle import (
    CandidateEventRecord,
    CandidateObservation,
    EventLifecycleState,
    ValidationConfig,
)
from tools.harness import ValidatedSessionHarness


def test_lifecycle_state_transitions_to_validated_and_closed():
    """Verify candidate transitions from OBSERVED -> CANDIDATE -> VALIDATED -> CLOSED."""
    cand = CandidateEventRecord(
        candidate_id="c001",
        session_id="sess_001",
        event_type=EventType.NO_FACE,
    )
    config = ValidationConfig(
        min_consecutive_frames={EventType.NO_FACE.value: 2},
        min_duration_seconds={EventType.NO_FACE.value: 0.5},
    )

    # Obs 1
    cand.add_observation(CandidateObservation(timestamp=0.0, frame_index=1, confidence=0.9))
    assert cand.state == EventLifecycleState.OBSERVED

    # Obs 2
    cand.add_observation(CandidateObservation(timestamp=0.25, frame_index=2, confidence=0.95))
    assert cand.state == EventLifecycleState.CANDIDATE

    # Evaluate at t=0.25 -> Meets 2 frames criteria -> VALIDATED
    st = cand.evaluate_state(0.25, config)
    assert st == EventLifecycleState.VALIDATED

    # Idle beyond absence tolerance -> CLOSED
    st2 = cand.evaluate_state(1.5, config)
    assert st2 == EventLifecycleState.CLOSED


def test_transient_candidate_discarded():
    """Verify single-frame transient anomaly is discarded when it expires without qualifying."""
    cand = CandidateEventRecord(
        candidate_id="c002",
        session_id="sess_002",
        event_type=EventType.NO_FACE,
    )
    config = ValidationConfig(
        min_consecutive_frames={EventType.NO_FACE.value: 3},
        min_duration_seconds={EventType.NO_FACE.value: 1.0},
    )

    # Single transient observation
    cand.add_observation(CandidateObservation(timestamp=0.0, frame_index=1, confidence=0.85))

    # Evaluate after absence tolerance expires
    st = cand.evaluate_state(1.2, config)
    assert st == EventLifecycleState.DISCARDED
    assert "Transient observation" in cand.validation_reason


def test_evidence_quality_validator_rules():
    """Verify evidence validator accepts valid images and rejects invalid/dead sensor images."""
    valid_img = np.full((240, 320, 3), 128, dtype=np.uint8)
    res_valid = EvidenceQualityValidator.validate_frame(valid_img, timestamp_seconds=1.0)
    assert res_valid.is_valid is True
    assert res_valid.sha256_checksum is not None

    # Too small
    small_img = np.full((50, 50, 3), 128, dtype=np.uint8)
    res_small = EvidenceQualityValidator.validate_frame(small_img, timestamp_seconds=1.0)
    assert res_small.is_valid is False
    assert "below minimum" in (res_small.error_reason or "")

    # None image
    res_none = EvidenceQualityValidator.validate_frame(None, timestamp_seconds=1.0)
    assert res_none.is_valid is False


def test_advanced_validated_engine_execution():
    """Verify advanced validated engine processes frames and exports validation statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(
            session_id="test_p9_val_engine",
            student_name="Test Student",
            output_dir=tmpdir,
            create_zip=False,
        )
        engine = ValidatedSessionHarness(config=config)
        test_img = np.full((240, 320, 3), 150, dtype=np.uint8)

        engine.process_frame(test_img, frame_index=1, timestamp_seconds=0.0)
        engine.process_frame(test_img, frame_index=2, timestamp_seconds=0.25)
        summary, stats = engine.finalize_session()

        assert summary.processed_frames == 2
        assert stats.total_frames_processed == 2
        assert (Path(tmpdir) / summary.session_id / "validation_stats.json").exists()
