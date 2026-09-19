"""Regression tests for Part 6: Visual mouth movement event semantics (P1-2).

Verifies:
1. EventType.MOUTH_MOVEMENT_DETECTED is defined.
2. Observer emits MOUTH_MOVEMENT_DETECTED with explicit sensor metadata:
   - sensor: "visual_blendshapes"
   - audio_recorded: False
3. Observation text clearly states no audio is recorded.
4. Legacy CANDIDATE_SPEAKING alias compatibility is preserved.
"""
from proctoring.core.events import EventType
from proctoring.analysis.policy import ExamPolicy, StrictnessLevel, ExamMode
from proctoring.analysis.observer import BehaviourObserver
from proctoring.analysis.facial_dynamics import FacialDynamicsResult
from proctoring.observation import FrameObservation, FaceStatus


def test_mouth_movement_event_definition():
    assert EventType.MOUTH_MOVEMENT_DETECTED.value == "MOUTH_MOVEMENT_DETECTED"
    assert EventType.CANDIDATE_SPEAKING.value in ("CANDIDATE_SPEAKING", "MOUTH_MOVEMENT_DETECTED")


def test_observer_emits_mouth_movement_with_visual_sensor_metadata():
    policy = ExamPolicy.for_level(StrictnessLevel.STRICT, ExamMode.DIGITAL_SCREEN)
    observer = BehaviourObserver(policy)

    dynamics = FacialDynamicsResult(
        face_found=True,
        is_speaking=True,
        speech_activity=0.85,
        mouth_region=(100, 200, 150, 250),
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

    assert EventType.MOUTH_MOVEMENT_DETECTED in events
    event_data = events[EventType.MOUTH_MOVEMENT_DETECTED]
    assert event_data["sensor"] == "visual_blendshapes"
    assert event_data["audio_recorded"] is False
    assert "No audio is recorded" in event_data["description"]
    assert "purely visual blendshape observation" in event_data["description"]


def test_observer_legacy_candidate_speaking_fallback():
    # Create custom policy enabling ONLY legacy CANDIDATE_SPEAKING
    policy = ExamPolicy(
        level=StrictnessLevel.STRICT,
        mode=ExamMode.DIGITAL_SCREEN,
        enabled_events=frozenset({EventType.CANDIDATE_SPEAKING}),
    )
    observer = BehaviourObserver(policy)

    dynamics = FacialDynamicsResult(
        face_found=True,
        is_speaking=True,
        speech_activity=0.75,
        mouth_region=(100, 200, 150, 250),
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

    assert EventType.CANDIDATE_SPEAKING in events
    event_data = events[EventType.CANDIDATE_SPEAKING]
    assert event_data["sensor"] == "visual_blendshapes"
    assert event_data["audio_recorded"] is False
