"""Unit tests for multi-modal correlation in BehaviourObserver."""

from proctoring.analysis.facial_dynamics import FacialDynamicsResult, HeadPose
from proctoring.analysis.gaze import GazeDirection, GazeObservation
from proctoring.analysis.observer import BehaviourObserver
from proctoring.analysis.policy import ExamMode, ExamPolicy, StrictnessLevel
from proctoring.core.events import EventType
from proctoring.observation import FrameObservation
from proctoring.preprocessing.camera_health import CameraAnomaly, CameraHealthStatus


def test_looking_away_with_gaze_correlation():
    """Verify LOOKING_AWAY incorporates gaze direction into the descriptive observation."""
    policy = ExamPolicy.for_level(StrictnessLevel.STRICT, ExamMode.DIGITAL_SCREEN)
    observer = BehaviourObserver(policy=policy)

    dynamics = FacialDynamicsResult(
        face_found=True,
        head_pose=HeadPose(yaw=42.0, pitch=10.0, roll=0.0),
        gaze=GazeObservation(
            horizontal=0.65, vertical=0.1, direction=GazeDirection.RIGHT, confidence=0.88
        ),
    )

    obs = FrameObservation(
        frame_index=0,
        timestamp_seconds=0.0,
        iso_timestamp="2026-08-27T10:00:00Z",
        facial_dynamics=dynamics,
        gaze=dynamics.gaze,
    )

    active = observer.map_to_events(obs)
    assert EventType.LOOKING_AWAY in active
    desc = active[EventType.LOOKING_AWAY]["description"]
    assert "yaw 42°" in desc
    assert "right zone" in desc


def test_camera_frozen_event_mapping():
    """Verify frozen camera stream triggers EventType.CAMERA_FRAME_FROZEN."""
    policy = ExamPolicy.for_level(StrictnessLevel.STANDARD)
    observer = BehaviourObserver(policy=policy)

    health = CameraHealthStatus(
        anomaly=CameraAnomaly.FRAME_FROZEN,
        is_healthy=False,
        is_frozen=True,
        freeze_duration_seconds=4.5,
    )

    obs = FrameObservation(
        frame_index=10,
        timestamp_seconds=5.0,
        iso_timestamp="2026-08-27T10:00:05Z",
        camera_health=health,
    )

    active = observer.map_to_events(obs)
    assert EventType.CAMERA_FRAME_FROZEN in active
    assert "Camera stream frozen for 4.5s" in active[EventType.CAMERA_FRAME_FROZEN]["description"]
