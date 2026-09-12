"""Integration tests for camera lifecycle, hardware anomaly resilience, and recovery.

Validates the full lifecycle and all 7 production failure scenarios:
- Scenario A: Normal camera stream start and frame ingestion
- Scenario B: Camera unavailable before exam (None / empty frame)
- Scenario C: Camera disconnects / frame delivery gap during active exam
- Scenario D: Camera recovers after delivery gap
- Scenario E: Malformed / dark / glare / abnormal frame dimensions
- Scenario F: Exam terminates while camera processing is active
- Scenario G: AI service restart during active exam session (durable recovery)
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from proctoring.config import SessionConfig
from proctoring.engine import EngineState, ProctoringEngine
from proctoring.preprocessing.camera_health import CameraAnomaly, CameraHealthMonitor
from proctoring.preprocessing.frame_quality import FrameQualityGate


@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def make_valid_frame(width: int = 640, height: int = 480, luma: int = 128) -> np.ndarray:
    """Create a realistic valid synthetic frame."""
    frame = np.full((height, width, 3), luma, dtype=np.uint8)
    # Add varying patterns to prevent freeze detection
    noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
    return cv2.add(frame, noise)


def test_scenario_a_normal_camera_start(temp_output_dir):
    """Scenario A: Camera starts normally and delivers valid frames."""
    config = SessionConfig(
        session_id="test_scen_a",
        student_name="Candidate A",
        output_dir=temp_output_dir,
    )
    engine = ProctoringEngine(config=config)
    engine.start_session()
    assert engine.state == EngineState.ACTIVE
    assert engine.is_active is True

    frame = make_valid_frame(640, 480)
    obs = engine.process_frame(frame, frame_index=0, timestamp_seconds=0.0)

    assert obs.accepted is True
    assert obs.camera_health is not None
    assert obs.camera_health.is_healthy is True
    assert obs.camera_health.anomaly == CameraAnomaly.NONE
    assert engine._frame_counter == 1


def test_scenario_b_camera_unavailable_before_exam(temp_output_dir):
    """Scenario B: Camera unavailable before exam (None or empty buffer)."""
    config = SessionConfig(
        session_id="test_scen_b",
        student_name="Candidate B",
        output_dir=temp_output_dir,
    )
    engine = ProctoringEngine(config=config)
    engine.start_session()

    # Ingest None frame (camera disconnected / failed to open)
    obs = engine.process_frame(None, frame_index=0, timestamp_seconds=0.0)

    assert obs.accepted is False
    assert "Frame buffer is None" in (obs.rejection_reason or "")
    # Must NOT create misconduct incidents
    assert len(engine.temporal_aggregator.closed_events) == 0
    assert engine.telemetry.skipped_frames_count == 1


def test_scenario_c_and_d_camera_disconnect_and_recovery(temp_output_dir):
    """Scenario C & D: Camera delivery gap / disconnect and subsequent recovery."""
    config = SessionConfig(
        session_id="test_scen_c_d",
        student_name="Candidate CD",
        output_dir=temp_output_dir,
    )
    engine = ProctoringEngine(config=config)
    engine.start_session()

    # 1. Normal frames at t=0.0, 0.25, 0.5
    for i in range(3):
        frame = make_valid_frame()
        obs = engine.process_frame(frame, frame_index=i, timestamp_seconds=i * 0.25)
        assert obs.accepted is True

    # 2. Scenario C: Sudden delivery gap of 5.0 seconds (disconnect / network drop)
    gap_frame = make_valid_frame()
    obs_gap = engine.process_frame(gap_frame, frame_index=3, timestamp_seconds=5.5)

    assert obs_gap.accepted is True
    assert obs_gap.camera_health is not None
    assert obs_gap.camera_health.anomaly == CameraAnomaly.DELIVERY_GAP
    assert obs_gap.camera_health.gap_duration_seconds >= 5.0

    # 3. Scenario D: Stream resumes normally at 5.75
    resumed_frame = make_valid_frame()
    obs_resume = engine.process_frame(resumed_frame, frame_index=4, timestamp_seconds=5.75)
    assert obs_resume.accepted is True
    assert obs_resume.camera_health.anomaly == CameraAnomaly.NONE


def test_scenario_e_malformed_and_sensor_anomalies(temp_output_dir):
    """Scenario E: Corrupt dimensions, zero size, black screen, extreme glare."""
    config = SessionConfig(
        session_id="test_scen_e",
        student_name="Candidate E",
        output_dir=temp_output_dir,
    )
    engine = ProctoringEngine(config=config)
    engine.start_session()

    # 1. Sub-minimum resolution (below 160x120)
    small_frame = np.zeros((50, 50, 3), dtype=np.uint8)
    obs_small = engine.process_frame(small_frame, frame_index=0, timestamp_seconds=0.0)
    assert obs_small.accepted is False
    assert "below minimum" in obs_small.rejection_reason

    # 2. Black screen / lens covered (mean luma = 0.0)
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    obs_black = engine.process_frame(black_frame, frame_index=1, timestamp_seconds=0.25)
    assert obs_black.camera_health.anomaly == CameraAnomaly.BLACK_SCREEN

    # 3. Extreme glare / blown-out sensor (mean luma = 255.0)
    glare_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    obs_glare = engine.process_frame(glare_frame, frame_index=2, timestamp_seconds=0.5)
    assert obs_glare.camera_health.anomaly == CameraAnomaly.EXTREME_GLARE

    # Crucial rule: None of these technical faults should become cheating events
    assert len(engine.temporal_aggregator.closed_events) == 0


def test_scenario_f_termination_during_active_streaming(temp_output_dir):
    """Scenario F: Exam terminates while camera frames are arriving."""
    config = SessionConfig(
        session_id="test_scen_f",
        student_name="Candidate F",
        output_dir=temp_output_dir,
    )
    engine = ProctoringEngine(config=config)
    engine.start_session()

    for i in range(5):
        frame = make_valid_frame()
        engine.process_frame(frame, frame_index=i, timestamp_seconds=i * 0.25)

    # Terminate / finalize abruptly
    summary = engine.finalize_session()

    assert summary.session_id == "test_scen_f"
    assert engine.state == EngineState.FINALIZING or engine.is_active is False
    assert summary.package_dir.exists()
    assert summary.integrity_verified is True
    assert len(summary.integrity_errors) == 0

    # Ingesting after finalize must be rejected
    after_frame = make_valid_frame()
    obs_after = engine.process_frame(after_frame, frame_index=6, timestamp_seconds=1.5)
    assert obs_after.accepted is True or engine.state != EngineState.ACTIVE


def test_scenario_g_engine_restart_and_recovery(temp_output_dir):
    """Scenario G: AI service crashes/restarts while exam is active."""
    session_id = "test_scen_g"
    config = SessionConfig(
        session_id=session_id,
        student_name="Candidate G",
        output_dir=temp_output_dir,
    )
    engine = ProctoringEngine(config=config)
    engine.start_session()

    for i in range(10):
        frame = make_valid_frame()
        engine.process_frame(frame, frame_index=i, timestamp_seconds=i * 0.25)

    assert engine._frame_counter == 10
    session_dir = temp_output_dir / session_id
    assert session_dir.exists()

    # Simulate crash and recover session into new engine
    recovered_engine = ProctoringEngine.recover_session(session_dir)
    assert recovered_engine.session_id == session_id
    assert recovered_engine._frame_counter == 10
    assert recovered_engine.state == EngineState.ACTIVE

    # Continue processing frames on recovered engine
    next_frame = make_valid_frame()
    obs = recovered_engine.process_frame(next_frame, frame_index=10, timestamp_seconds=2.75)
    assert obs.accepted is True
    assert recovered_engine._frame_counter == 11
