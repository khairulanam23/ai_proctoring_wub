"""Tests for strict session state isolation and explicit session lifecycle in ProctoringEngine."""

import numpy as np
import pytest

from proctoring.analysis.hands import HandAnalysisResult, HandKinematics, HandObservation, HandState
from proctoring.analysis.paper import PaperAnalysisResult, PaperSheet, PaperState
from proctoring.config import SessionConfig
from proctoring.core.events import EventType
from proctoring.engine import EngineState, FaceStatus, FrameObservation, ProctoringEngine


@pytest.fixture
def clean_engine_factory(tmp_path):
    def _create(session_id: str, student_name: str = "Test Student"):
        config = SessionConfig(
            session_id=session_id,
            student_name=student_name,
            output_dir=tmp_path / session_id,
            enable_face_detection=False,
            enable_face_verification=False,
            enable_object_detection=False,
            enable_facial_dynamics=True,
            enable_hand_analysis=True,
            capture_evidence=True,
        )
        return ProctoringEngine(config=config)

    return _create


def test_explicit_session_lifecycle(clean_engine_factory):
    engine = clean_engine_factory("test_lifecycle_001")
    assert engine.state == EngineState.CREATED
    assert not engine.is_active

    # 1. create_session()
    engine.create_session()
    assert engine.state == EngineState.RUNNING
    assert engine.is_active

    # 2. process_frame()
    dummy_frame = np.full((360, 480, 3), 128, dtype=np.uint8)
    obs = engine.process_frame(dummy_frame, frame_index=0, timestamp_seconds=0.0)
    assert obs.frame_index == 0
    assert len(engine.timeline) == 1

    # 3. reset_session()
    engine.reset_session()
    assert engine.state == EngineState.CREATED
    assert not engine.is_active
    assert engine._frame_counter == 0
    assert len(engine.timeline) == 0

    # 4. destroy_session()
    engine.destroy_session()
    assert engine.state == EngineState.FINALIZED
    assert not engine.is_active


def test_session_isolation_calibration_and_kinematics(clean_engine_factory):
    """Verify that calibration and kinematics state do not leak from Session A to Session B."""
    engine_a = clean_engine_factory("session_alpha", "Student Alpha")
    engine_a.create_session()

    # Manually set a candidate calibration on Session A
    if engine_a._stages.facial_dynamics:
        engine_a._stages.facial_dynamics.baseline_yaw = 18.5
        engine_a._stages.facial_dynamics.baseline_pitch = -8.2
        engine_a._stages.facial_dynamics.is_calibrated = True

    # Manually populate hand kinematics history on Session A
    if engine_a._stages.hand_analyzer:
        engine_a._stages.hand_analyzer._last_centroids = [(200, 200)]
        engine_a._stages.hand_analyzer._kinematics_tracks["hand_0"] = [{"speed": 45.0}]

    # Manually populate paper detector state on Session A
    if engine_a._stages.paper_detector:
        engine_a._stages.paper_detector._last_centroid = (300, 300)
        engine_a._stages.paper_detector._last_area = 50000.0

    # Reset Session A
    engine_a.reset_session()

    # Assert Session A's analyzers are completely reset to baseline defaults
    if engine_a._stages.facial_dynamics:
        assert engine_a._stages.facial_dynamics.baseline_yaw == 0.0
        assert engine_a._stages.facial_dynamics.baseline_pitch == 0.0
        assert not engine_a._stages.facial_dynamics.is_calibrated

    if engine_a._stages.hand_analyzer:
        assert engine_a._stages.hand_analyzer._last_centroids == []
        assert len(engine_a._stages.hand_analyzer._kinematics_tracks) == 0

    if engine_a._stages.paper_detector:
        assert engine_a._stages.paper_detector._last_centroid is None
        assert engine_a._stages.paper_detector._last_area is None

    # Now create Session B using fresh engine
    engine_b = clean_engine_factory("session_beta", "Student Beta")
    engine_b.create_session()

    if engine_b._stages.facial_dynamics:
        assert engine_b._stages.facial_dynamics.baseline_yaw == 0.0
        assert not engine_b._stages.facial_dynamics.is_calibrated

    if engine_b._stages.hand_analyzer:
        assert engine_b._stages.hand_analyzer._last_centroids == []


def test_interleaved_concurrent_sessions(clean_engine_factory):
    """Verify that two concurrently executing engines do not cross-pollinate observations."""
    engine_1 = clean_engine_factory("session_1", "Candidate 1")
    engine_2 = clean_engine_factory("session_2", "Candidate 2")

    engine_1.create_session()
    engine_2.create_session()

    frame_1 = np.full((360, 480, 3), 100, dtype=np.uint8)
    frame_2 = np.full((360, 480, 3), 200, dtype=np.uint8)

    # Interleaved frame processing
    obs_1_0 = engine_1.process_frame(frame_1, frame_index=0, timestamp_seconds=0.0)
    obs_2_0 = engine_2.process_frame(frame_2, frame_index=0, timestamp_seconds=0.0)
    obs_1_1 = engine_1.process_frame(frame_1, frame_index=1, timestamp_seconds=0.25)
    obs_2_1 = engine_2.process_frame(frame_2, frame_index=1, timestamp_seconds=0.25)

    assert obs_1_0.frame_index == 0
    assert obs_2_0.frame_index == 0
    assert obs_1_1.frame_index == 1
    assert obs_2_1.frame_index == 1

    assert engine_1.session_id == "session_1"
    assert engine_2.session_id == "session_2"
    assert len(engine_1.timeline) == 2
    assert len(engine_2.timeline) == 2

    # Clean finalization
    res_1 = engine_1.finalize_session()
    res_2 = engine_2.finalize_session()

    assert res_1.session_id == "session_1"
    assert res_2.session_id == "session_2"
    assert res_1.package_dir != res_2.package_dir
    assert res_1.integrity_verified
    assert res_2.integrity_verified
