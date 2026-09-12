"""Unit and integration tests for progressive durable persistence and crash recovery."""

import tempfile
from pathlib import Path

import numpy as np

from proctoring.config import SessionConfig
from proctoring.core.events import EventType
from proctoring.engine import EngineState, ProctoringEngine


def test_progressive_persistence_and_crash_recovery():
    """Verify events spooled to disk mid-session and session cleanly recovers after crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        session_id = "test_recovery_sess_01"

        config = SessionConfig(
            session_id=session_id,
            student_name="Recovery Candidate",
            output_dir=output_dir,
            enable_face_detection=False,
            enable_face_verification=False,
            enable_object_detection=False,
            enable_facial_dynamics=False,
            enable_hand_analysis=False,
            enable_wearable_detection=False,
            record_timeline=True,
            capture_evidence=False,
        )

        # 1. Start live engine
        engine = ProctoringEngine(config=config)
        engine.start_session()
        assert engine.state == EngineState.RUNNING

        session_pkg_dir = output_dir / session_id
        checkpoint_file = session_pkg_dir / "session_checkpoint.json"
        events_jsonl = session_pkg_dir / "events.jsonl"
        timeline_jsonl = session_pkg_dir / "timeline.jsonl"

        assert checkpoint_file.exists()

        # 2. Process frames
        blank_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        for i in range(1, 11):
            engine.process_frame(blank_frame, frame_index=i, timestamp_seconds=i * 0.25)

        # 3. Emit client events (e.g. browser tab switch, window blur)
        engine.record_browser_event(
            event_type=EventType.BROWSER_TAB_SWITCH,
            timestamp_seconds=2.5,
            description="Candidate switched tabs",
            frame_index=10,
        )
        engine.pause("Network latency check")
        engine.resume("Network back to normal")

        # Verify durable files were written mid-session
        assert events_jsonl.exists()
        assert timeline_jsonl.exists()

        # Read checkpoint before finalization
        checkpoint = engine.journal.load_checkpoint()
        assert checkpoint is not None
        assert checkpoint.state == EngineState.RUNNING.value
        assert checkpoint.total_events >= 3  # tab hidden + pause + resume
        assert checkpoint.sequence_number >= 3

        # 4. SIMULATE SUDDEN CRASH / PROCESS TERMINATION
        # We deliberately DO NOT call finalize_session(), simulating process death
        del engine

        # 5. RECOVER SESSION FROM DISK
        recovered_engine = ProctoringEngine.recover_session(session_pkg_dir)
        assert recovered_engine.state in (EngineState.ACTIVE, EngineState.RUNNING, EngineState.RECOVERY_REQUIRED)
        assert recovered_engine.session_id == session_id
        assert recovered_engine.student_name == "Recovery Candidate"
        assert len(recovered_engine.temporal_aggregator.closed_events) >= 3
        assert len(recovered_engine.timeline.entries) == 10
        assert recovered_engine._sequence_number >= 3

        # 6. SEAL RECOVERED SESSION
        summary = recovered_engine.finalize_recovered_session()
        assert summary.session_id == session_id
        assert summary.integrity_verified is True
        assert len(summary.integrity_errors) == 0
        assert (session_pkg_dir / "manifest.json").exists()
        assert (session_pkg_dir / "manifest.sha256").exists()
        assert (session_pkg_dir / "events.json").exists()
        assert recovered_engine.state == EngineState.FINALIZED


def test_resume_recovered_session_and_continue_processing():
    """Verify an interrupted session can be resumed, process new frames, and then finalize."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        session_id = "test_recovery_sess_02"

        config = SessionConfig(
            session_id=session_id,
            student_name="Resume Candidate",
            output_dir=output_dir,
            enable_face_detection=False,
            enable_face_verification=False,
            enable_object_detection=False,
            enable_facial_dynamics=False,
            enable_hand_analysis=False,
            enable_wearable_detection=False,
            record_timeline=True,
            capture_evidence=False,
        )

        engine = ProctoringEngine(config=config)
        engine.start_session()

        blank_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        for i in range(1, 6):
            engine.process_frame(blank_frame, frame_index=i, timestamp_seconds=i * 0.25)

        engine.record_browser_event(
            event_type=EventType.BROWSER_TAB_SWITCH,
            timestamp_seconds=1.25,
            description="Tab switch before crash",
            frame_index=5,
        )

        session_pkg_dir = output_dir / session_id

        # Simulate crash
        del engine

        # Recover
        recovered_engine = ProctoringEngine.recover_session(session_pkg_dir)
        assert recovered_engine.state in (EngineState.ACTIVE, EngineState.RUNNING, EngineState.RECOVERY_REQUIRED)

        # Resume (idempotent transition to RUNNING/ACTIVE)
        recovered_engine.resume("Candidate reconnected after machine reboot")
        assert recovered_engine.state in (EngineState.RUNNING, EngineState.ACTIVE)

        # Process further frames
        for i in range(6, 11):
            recovered_engine.process_frame(blank_frame, frame_index=i, timestamp_seconds=i * 0.25)

        # Finalize
        summary = recovered_engine.finalize_session()
        assert summary.integrity_verified is True
        assert len(summary.events) >= 2  # browser tab switch + session resumed


def test_recovery_isolation_across_multiple_sessions():
    """Verify recovering multiple crashed sessions maintains strict session isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        sess_a_id = "session_recovery_cand_a"
        sess_b_id = "session_recovery_cand_b"

        # Session A
        cfg_a = SessionConfig(
            session_id=sess_a_id,
            student_name="Candidate Alice",
            output_dir=output_dir,
            enable_face_detection=False,
            enable_face_verification=False,
            enable_object_detection=False,
            enable_facial_dynamics=False,
            enable_hand_analysis=False,
            enable_wearable_detection=False,
            record_timeline=True,
        )
        eng_a = ProctoringEngine(config=cfg_a)
        eng_a.start_session()

        # Session B
        cfg_b = SessionConfig(
            session_id=sess_b_id,
            student_name="Candidate Bob",
            output_dir=output_dir,
            enable_face_detection=False,
            enable_face_verification=False,
            enable_object_detection=False,
            enable_facial_dynamics=False,
            enable_hand_analysis=False,
            enable_wearable_detection=False,
            record_timeline=True,
        )
        eng_b = ProctoringEngine(config=cfg_b)
        eng_b.start_session()

        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        # Process 5 frames for A, 8 frames for B
        for i in range(1, 6):
            eng_a.process_frame(frame, frame_index=i, timestamp_seconds=i * 0.25)
        for i in range(1, 9):
            eng_b.process_frame(frame, frame_index=i, timestamp_seconds=i * 0.25)

        eng_a.record_browser_event(
            event_type=EventType.BROWSER_TAB_SWITCH,
            timestamp_seconds=1.25,
            description="Alice switched tabs",
            frame_index=5,
        )

        # Simulate concurrent process termination
        del eng_a
        del eng_b

        # Recover both
        rec_a = ProctoringEngine.recover_session(output_dir / sess_a_id)
        rec_b = ProctoringEngine.recover_session(output_dir / sess_b_id)

        assert rec_a.session_id == sess_a_id
        assert rec_a.student_name == "Candidate Alice"
        assert rec_a._frame_counter == 6
        assert len(rec_a.temporal_aggregator.closed_events) == 1
        assert rec_a.temporal_aggregator.closed_events[0].observation.description == "Alice switched tabs"

        assert rec_b.session_id == sess_b_id
        assert rec_b.student_name == "Candidate Bob"
        assert rec_b._frame_counter == 9
        assert len(rec_b.temporal_aggregator.closed_events) == 0  # No events for Bob


def test_repeated_recovery_idempotence():
    """Verify calling recover_session multiple times is deterministic and non-destructive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        session_id = "test_repeated_recovery"

        cfg = SessionConfig(
            session_id=session_id,
            student_name="Candidate Repeat",
            output_dir=output_dir,
            enable_face_detection=False,
            enable_face_verification=False,
            enable_object_detection=False,
            enable_facial_dynamics=False,
            enable_hand_analysis=False,
            enable_wearable_detection=False,
            record_timeline=True,
        )
        eng = ProctoringEngine(config=cfg)
        eng.start_session()

        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        for i in range(1, 6):
            eng.process_frame(frame, frame_index=i, timestamp_seconds=i * 0.25)

        del eng
        session_dir = output_dir / session_id

        # First recovery
        rec1 = ProctoringEngine.recover_session(session_dir)
        assert rec1._frame_counter == 6

        # Second recovery without modifying files
        rec2 = ProctoringEngine.recover_session(session_dir)
        assert rec2._frame_counter == 6
        assert rec2.session_id == rec1.session_id
        assert len(rec2.timeline.entries) == len(rec1.timeline.entries)

