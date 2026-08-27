"""Tests for exam mode behavior (DIGITAL_SCREEN vs PHYSICAL_PAPER) and policy rules."""

from proctoring.analysis.facial_dynamics import FacialDynamicsResult, HeadPose
from proctoring.analysis.gaze import GazeDirection, GazeObservation
from proctoring.analysis.observer import BehaviourObserver
from proctoring.analysis.policy import ExamMode, ExamPolicy, StrictnessLevel
from proctoring.core.events import EventType
from proctoring.observation import FrameObservation


def test_digital_screen_pitch_and_gaze_boundaries() -> None:
    policy = ExamPolicy.for_mode(ExamMode.DIGITAL_SCREEN, StrictnessLevel.STRICT)

    assert policy.mode == ExamMode.DIGITAL_SCREEN
    assert policy.pitch_down_limit_degrees <= 32.0

    # Looking straight ahead (0, 0) is valid
    assert not policy.is_pose_deviated(yaw=0.0, pitch=0.0)
    assert not policy.is_gaze_deviated(horizontal=0.0, vertical=0.0)

    # Moderate downward pitch (-35 deg) is outside digital screen limits (looking away)
    assert policy.is_pose_deviated(yaw=0.0, pitch=-35.0)

    # Moderate downward gaze (-0.45) is outside digital screen limits
    assert policy.is_gaze_deviated(horizontal=0.0, vertical=-0.45)


def test_physical_paper_pitch_and_gaze_boundaries() -> None:
    policy = ExamPolicy.for_mode(ExamMode.PHYSICAL_PAPER, StrictnessLevel.STRICT)

    assert policy.mode == ExamMode.PHYSICAL_PAPER
    assert policy.pitch_down_limit_degrees >= 50.0
    assert policy.gaze_vertical_down_limit >= 0.60

    # Looking down at paper desk (-35 deg pitch) is ALLOWED in physical paper mode
    assert not policy.is_pose_deviated(yaw=0.0, pitch=-35.0)

    # Downward gaze (-0.45) is ALLOWED in physical paper mode
    assert not policy.is_gaze_deviated(horizontal=0.0, vertical=-0.45)

    # Extreme downward pitch (-70 deg) is still flagged
    assert policy.is_pose_deviated(yaw=0.0, pitch=-70.0)

    # Looking up (+30 deg) is flagged in physical paper mode
    assert policy.is_pose_deviated(yaw=0.0, pitch=30.0)

    # Looking sideways (+45 deg yaw) is flagged in physical paper mode
    assert policy.is_pose_deviated(yaw=45.0, pitch=-20.0)


def test_behaviour_observer_respects_exam_mode() -> None:
    policy_digital = ExamPolicy.for_mode(ExamMode.DIGITAL_SCREEN, StrictnessLevel.STRICT)
    observer_digital = BehaviourObserver(policy_digital)

    policy_paper = ExamPolicy.for_mode(ExamMode.PHYSICAL_PAPER, StrictnessLevel.STRICT)
    observer_paper = BehaviourObserver(policy_paper)

    # Candidate looking down at desk: pitch=-35°, yaw=5°
    dynamics = FacialDynamicsResult(
        face_found=True,
        face_bbox=(100, 100, 300, 300),
        head_pose=HeadPose(yaw=5.0, pitch=-35.0),
        gaze=GazeObservation(
            horizontal=0.05,
            vertical=-0.45,
            direction=GazeDirection.DOWN,
            confidence=0.9,
        ),
    )

    obs = FrameObservation(
        frame_index=1,
        timestamp_seconds=1.0,
        iso_timestamp="2026-08-27T00:00:01Z",
        facial_dynamics=dynamics,
    )

    # In digital screen mode, looking down at desk raises LOOKING_AWAY / GAZE_OFF_SCREEN
    events_digital = observer_digital.map_to_events(obs)
    assert EventType.LOOKING_AWAY in events_digital

    # In physical paper mode, looking down at desk does NOT raise LOOKING_AWAY
    events_paper = observer_paper.map_to_events(obs)
    assert EventType.LOOKING_AWAY not in events_paper
