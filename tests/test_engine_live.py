"""Tests for the engine's live-session path: timeline fidelity and fault tolerance.

These cover the behaviour the real-time session manager used to own — continuous
frame feeding, browser event injection, corrupt-frame handling and packaging —
now that a single engine serves both the live and offline paths.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from proctoring.config import SessionConfig
from proctoring.core.events import EventType
from proctoring.engine import FaceStatus, ProctoringEngine


def _frame(value: int = 150, size=(240, 320)) -> np.ndarray:
    return np.full((size[0], size[1], 3), value, dtype=np.uint8)


def test_live_session_lifecycle_and_packaging():
    """A live session logs a timeline, records browser events and seals a valid package."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(
            session_id="test_live_001",
            student_name="Candidate_Tester",
            sampling_fps=4.0,
            absence_tolerance_seconds=0.5,
            min_event_duration_seconds=1.0,
            output_dir=tmpdir,
            create_zip=False,
        )
        engine = ProctoringEngine(config=config)
        engine.start_session()
        assert engine.is_active is True

        first = engine.process_frame(_frame(), frame_index=0, timestamp_seconds=0.0)
        assert first.frame_index == 0
        assert first.accepted is True
        assert first.timing.total_frame_ms >= 0.0

        engine.record_browser_event(
            event_type=EventType.BROWSER_FULLSCREEN_EXIT,
            timestamp_seconds=0.5,
            description="Student exited full screen mode.",
        )

        second = engine.process_frame(_frame(), frame_index=1, timestamp_seconds=0.75)
        assert second.frame_index == 1

        summary = engine.finalize_session()
        assert engine.is_active is False
        assert summary.total_frames == 2
        assert summary.processed_frames == 2
        assert summary.integrity_verified is True

        package = Path(summary.package_dir)
        for artefact in ("timeline.json", "manifest.json", "events.json", "telemetry.json"):
            assert (package / artefact).exists(), f"missing {artefact}"

        # The browser event must survive into the sealed package.
        events = json.loads((package / "events.json").read_text())
        assert any(e["event_type"] == "BROWSER_FULLSCREEN_EXIT" for e in events)


def test_timeline_records_measured_values_not_placeholders():
    """Timeline rows carry real detector output; unmeasured stages stay null.

    Guards the regression where the live path emitted invented similarity scores
    and object lists whenever any event happened to be open.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(
            session_id="test_live_timeline",
            student_name="Candidate",
            output_dir=tmpdir,
            enable_face_detection=False,
            enable_facial_dynamics=False,
            enable_hand_analysis=False,
            enable_object_detection=False,
            enable_face_verification=False,
        )
        engine = ProctoringEngine(config=config)
        engine.start_session()
        for i in range(4):
            engine.process_frame(_frame(), frame_index=i, timestamp_seconds=i * 0.25)
        summary = engine.finalize_session()

        rows = json.loads((Path(summary.package_dir) / "timeline.json").read_text())
        assert len(rows) == 4
        for row in rows:
            # No detector ran, so nothing may be asserted about faces or objects.
            assert row["face_count"] is None
            assert row["cosine_similarity"] is None
            assert row["identity_verified"] is None
            assert row["prohibited_objects"] == []
            assert row["hands_detected"] is None
            assert row["is_speaking"] is None
            assert row["head_yaw"] is None
            assert row["frame_accepted"] is True


def test_unmeasured_face_never_becomes_a_no_face_event():
    """With no face stage running at all, the engine asserts nothing about presence.

    Regression test: the engine defaulted ``face_status`` to ``NO_FACE``, so a
    deployment with face detection switched off produced a continuous NO_FACE
    incident against a candidate who was sitting there the whole time.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(
            session_id="test_not_measured",
            student_name="Candidate",
            output_dir=tmpdir,
            enable_face_detection=False,
            enable_facial_dynamics=False,
            enable_hand_analysis=False,
            enable_object_detection=False,
        )
        engine = ProctoringEngine(config=config)
        engine.start_session()
        for i in range(10):
            observation = engine.process_frame(_frame(), frame_index=i, timestamp_seconds=i * 0.25)
            assert observation.face_status == FaceStatus.NOT_MEASURED

        summary = engine.finalize_session()
        assert summary.total_events == 0


def test_corrupt_frames_are_skipped_without_creating_events():
    """A camera fault is recorded as a diagnostic, never as an event against the candidate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(
            session_id="test_live_corrupt",
            student_name="Candidate_Tester",
            output_dir=tmpdir,
            create_zip=False,
            enable_face_detection=False,
            enable_object_detection=False,
        )
        engine = ProctoringEngine(config=config)
        engine.start_session()

        observation = engine.process_frame(None, frame_index=0, timestamp_seconds=0.0)
        assert observation.accepted is False
        assert observation.rejection_reason is not None
        assert observation.active_event_types == []

        summary = engine.finalize_session()
        assert summary.skipped_frames == 1
        assert summary.integrity_verified is True

        diagnostics = json.loads((Path(summary.package_dir) / "diagnostics.json").read_text())
        assert any(d["category"] == "CORRUPT_FRAME" for d in diagnostics)

        events = json.loads((Path(summary.package_dir) / "events.json").read_text())
        assert events == []


def test_adaptive_sampling_raises_rate_when_an_incident_is_open():
    """Sampling idles low while the scene is quiet and steps up once something opens."""

    class NoFaceDetector:
        def detect(self, frame, score_threshold=None):
            class Result:
                faces, count = [], 0

            return Result()

    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(
            session_id="test_live_adaptive",
            student_name="Candidate",
            output_dir=tmpdir,
            enable_adaptive_sampling=True,
            idle_fps=2.0,
            active_fps=8.0,
            enable_object_detection=False,
            enable_face_verification=False,
        )
        engine = ProctoringEngine(config=config, face_detector=NoFaceDetector())
        engine.start_session()

        assert engine.target_fps == 2.0  # nothing open yet
        engine.process_frame(_frame(), frame_index=0, timestamp_seconds=0.0)
        assert engine.target_fps == 8.0  # NO_FACE incident now open
        engine.finalize_session()
