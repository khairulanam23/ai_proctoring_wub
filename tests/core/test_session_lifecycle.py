"""Targeted unit tests for ProctoringEngine session lifecycle and state management."""

import numpy as np

from proctoring.config import SessionConfig
from proctoring.engine import EngineState, ProctoringEngine


def test_session_lifecycle_states(tmp_path) -> None:
    """Test session state transitions: CREATED -> RUNNING -> PAUSED -> RESUMED -> FINALIZED."""
    config = SessionConfig(session_id="test_lifecycle", output_dir=tmp_path / "out")
    engine = ProctoringEngine(config=config)

    assert engine.state == EngineState.CREATED
    assert engine.is_active is False

    # Process first frame triggers start_session -> RUNNING
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    engine.process_frame(frame, frame_index=0, timestamp_seconds=0.0)

    assert engine.state == EngineState.RUNNING
    assert engine.is_active is True

    # Pause session
    engine.pause(reason="Invigilator requested pause")
    assert engine.state == EngineState.PAUSED

    # While paused, frames are skipped with rejection_reason
    obs_paused = engine.process_frame(frame, frame_index=1, timestamp_seconds=1.0)
    assert obs_paused.accepted is False
    assert obs_paused.rejection_reason == "SESSION_PAUSED"

    # Resume session
    engine.resume(reason="Invigilator resumed")
    assert engine.state == EngineState.RUNNING

    # Finalize session
    summary = engine.finalize_session()
    assert engine.state == EngineState.FINALIZED
    assert engine.is_active is False
    assert summary.session_id == "test_lifecycle"


def test_session_cancellation(tmp_path) -> None:
    """Test aborting a session via cancel()."""
    config = SessionConfig(session_id="test_cancel", output_dir=tmp_path / "out")
    engine = ProctoringEngine(config=config)

    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    engine.process_frame(frame, frame_index=0, timestamp_seconds=0.0)
    assert engine.state == EngineState.RUNNING

    engine.cancel(reason="Candidate aborted exam")
    assert engine.state == EngineState.CANCELLED
    assert engine.is_active is False
