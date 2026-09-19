"""Regression tests for Part 9: Gaze / Head-Pose Calibration & Exam-Mode Policy (P2-2).

Verifies:
1. Normal digital-exam gaze produces no deviation.
2. Temporary gaze deviation does not qualify (hysteresis / debounce).
3. Sustained gaze deviation qualifies after duration threshold.
4. Written / PDF exam downward gaze is permitted as normal paper interaction.
5. Temporal qualification distinguishes transient vs sustained deviation.
6. Configuration differences between DIGITAL_SCREEN and PHYSICAL_PAPER / WRITTEN_PDF_EXAM.
"""
from proctoring.analysis.facial_dynamics import FacialDynamicsResult, HeadPose
from proctoring.analysis.gaze import GazeDirection, GazeObservation
from proctoring.analysis.observer import BehaviourObserver
from proctoring.analysis.policy import ExamMode, ExamPolicy, StrictnessLevel
from proctoring.core.events import EventType, DetectorInfo, EventStatus
from proctoring.observation import FaceStatus, FrameObservation
from proctoring.temporal.aggregator import UnifiedTemporalAggregator


def test_normal_digital_exam_gaze():
    """Verify centered gaze and head pose produce no deviation in digital mode."""
    policy = ExamPolicy.for_level(StrictnessLevel.STRICT, ExamMode.DIGITAL_SCREEN)
    assert not policy.is_pose_deviated(yaw=2.0, pitch=-3.0)
    assert not policy.is_gaze_deviated(horizontal=0.05, vertical=-0.05)

    observer = BehaviourObserver(policy)
    dynamics = FacialDynamicsResult(
        face_found=True,
        head_pose=HeadPose(yaw=2.0, pitch=-3.0, roll=0.0),
        gaze=GazeObservation(direction=GazeDirection.CENTER, horizontal=0.05, vertical=-0.05),
    )
    obs = FrameObservation(
        frame_index=1,
        timestamp_seconds=1.0,
        iso_timestamp="2026-09-18T10:00:00Z",
        face_count=1,
        face_boxes=[(50, 50, 300, 400)],
        face_status=FaceStatus.ENROLLED,
        facial_dynamics=dynamics,
    )
    events = observer.map_to_events(obs)
    assert EventType.LOOKING_AWAY not in events
    assert EventType.GAZE_OFF_SCREEN not in events


def test_written_exam_downward_gaze_permitted():
    """Verify downward head pitch and gaze are permitted during written/PDF exams."""
    paper_policy = ExamPolicy.for_level(StrictnessLevel.STRICT, ExamMode.PHYSICAL_PAPER)
    digital_policy = ExamPolicy.for_level(StrictnessLevel.STRICT, ExamMode.DIGITAL_SCREEN)

    # Looking down at paper: pitch = -35 deg, vertical gaze = -0.45
    # In digital mode, this violates screen-facing expectation:
    assert digital_policy.is_pose_deviated(yaw=0.0, pitch=-35.0) is True
    assert digital_policy.is_gaze_deviated(horizontal=0.0, vertical=-0.45) is True

    # In written/paper mode, downward posture is expected and permitted:
    assert paper_policy.is_pose_deviated(yaw=0.0, pitch=-35.0) is False
    assert paper_policy.is_gaze_deviated(horizontal=0.0, vertical=-0.45) is False

    # Observer in paper mode does NOT flag normal downward writing:
    observer = BehaviourObserver(paper_policy)
    dynamics = FacialDynamicsResult(
        face_found=True,
        head_pose=HeadPose(yaw=0.0, pitch=-35.0, roll=0.0),
        gaze=GazeObservation(direction=GazeDirection.DOWN, horizontal=0.0, vertical=-0.45),
    )
    obs = FrameObservation(
        frame_index=1,
        timestamp_seconds=1.0,
        iso_timestamp="2026-09-18T10:00:00Z",
        face_count=1,
        face_boxes=[(50, 50, 300, 400)],
        face_status=FaceStatus.ENROLLED,
        facial_dynamics=dynamics,
    )
    events = observer.map_to_events(obs)
    assert EventType.LOOKING_AWAY not in events
    assert EventType.GAZE_OFF_SCREEN not in events


def test_temporary_vs_sustained_gaze_temporal_qualification():
    """Verify temporary gaze shift does not qualify, whereas sustained gaze does."""
    policy = ExamPolicy.for_level(StrictnessLevel.STRICT, ExamMode.DIGITAL_SCREEN)
    aggregator = UnifiedTemporalAggregator(
        session_id="test_gaze_session",
        absence_tolerance_seconds=1.0,
        min_event_duration_seconds=policy.duration_for(EventType.LOOKING_AWAY),
    )

    req_duration = policy.duration_for(EventType.LOOKING_AWAY)
    assert req_duration >= 2.0
    detector = DetectorInfo(name="face_dynamics", version="1.0")

    # 1. Brief deviation (1.0 second): below threshold
    aggregator.update_behaviour_observations(
        active_behaviours={EventType.LOOKING_AWAY: {"confidence": 0.8}},
        timestamp=1.0,
        frame_index=10,
        detector=detector,
    )
    aggregator.update_behaviour_observations(
        active_behaviours={EventType.LOOKING_AWAY: {"confidence": 0.8}},
        timestamp=2.0,
        frame_index=20,
        detector=detector,
    )
    # Return to normal at 2.2s
    aggregator.update_behaviour_observations(
        active_behaviours={},
        timestamp=2.2,
        frame_index=22,
        detector=detector,
    )
    # At 3.5s, tolerance expires and event is closed
    aggregator.update_behaviour_observations(
        active_behaviours={},
        timestamp=3.5,
        frame_index=35,
        detector=detector,
    )

    # Verify brief deviation did NOT produce a qualified event
    closed = aggregator.flush()
    qualified = [e for e in closed if e.status == EventStatus.QUALIFIED]
    assert len(qualified) == 0, "Temporary glance must not produce a qualified violation"

    # 2. Sustained deviation (lasting longer than req_duration):
    t = 10.0
    aggregator.update_behaviour_observations(
        active_behaviours={EventType.LOOKING_AWAY: {"confidence": 0.9}},
        timestamp=t,
        frame_index=100,
        detector=detector,
    )
    t += req_duration + 0.5
    aggregator.update_behaviour_observations(
        active_behaviours={EventType.LOOKING_AWAY: {"confidence": 0.9}},
        timestamp=t,
        frame_index=115,
        detector=detector,
    )
    t += 1.5
    aggregator.update_behaviour_observations(
        active_behaviours={},
        timestamp=t,
        frame_index=130,
        detector=detector,
    )
    closed = aggregator.flush()
    sustained_qualified = [e for e in closed if e.status == EventStatus.QUALIFIED]
    assert len(sustained_qualified) == 1, "Sustained looking away must qualify"


def test_configuration_differences_digital_vs_written():
    """Verify configuration parameters differ appropriately between exam modes."""
    digital = ExamPolicy.for_level(StrictnessLevel.STRICT, ExamMode.DIGITAL_SCREEN)
    paper = ExamPolicy.for_level(StrictnessLevel.STRICT, ExamMode.WRITTEN_PDF_EXAM)

    assert paper.pitch_down_limit_degrees == 60.0
    assert digital.pitch_down_limit_degrees == 26.0

    assert paper.gaze_vertical_down_limit == 0.70
    assert digital.gaze_vertical_down_limit == 0.32

    assert paper.yaw_limit_degrees >= 38.0
    assert digital.yaw_limit_degrees == 28.0

    assert EventType.PAPER_PRESENT in paper.enabled_events
    assert EventType.HAND_WRITING in paper.enabled_events
    assert EventType.PAPER_PRESENT not in digital.enabled_events
