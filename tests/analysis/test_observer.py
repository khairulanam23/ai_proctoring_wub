"""Tests for the behaviour observer: policy gating, debouncing and annotation context."""

import numpy as np

from proctoring.analysis.facial_dynamics import FacialDynamicsResult, HeadPose
from proctoring.analysis.hands import HandAnalysisResult, HandObservation
from proctoring.analysis.observer import BehaviourObserver
from proctoring.analysis.policy import ExamPolicy, StrictnessLevel
from proctoring.analysis.wearables import WearableAnalysisResult, WearableDetection
from proctoring.core.events import EventType
from proctoring.observation import FrameObservation


def _observation(**overrides) -> FrameObservation:
    base = {"frame_index": 0, "timestamp_seconds": 0.0, "iso_timestamp": "1970-01-01T00:00:00Z"}
    base.update(overrides)
    return FrameObservation(**base)


def _dynamics(**overrides) -> FacialDynamicsResult:
    result = FacialDynamicsResult(face_found=True, face_bbox=(100, 100, 200, 220))
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


def _observer(level=StrictnessLevel.MAXIMUM, window=1) -> BehaviourObserver:
    policy = ExamPolicy.for_level(level)
    policy.smoothing_window_frames = window
    return BehaviourObserver(policy)


# ---------------------------------------------------------------------------
# Policy gating
# ---------------------------------------------------------------------------


def test_observations_below_the_strictness_level_are_not_reported():
    """STANDARD must not surface speaking, however clearly it was measured."""
    observer = _observer(StrictnessLevel.STANDARD)
    events = observer.map_to_events(
        _observation(facial_dynamics=_dynamics(is_speaking=True, speech_activity=0.9))
    )
    assert EventType.CANDIDATE_SPEAKING not in events


def test_strict_level_reports_speaking():
    observer = _observer(StrictnessLevel.STRICT)
    events = observer.map_to_events(
        _observation(facial_dynamics=_dynamics(is_speaking=True, speech_activity=0.9))
    )
    assert EventType.CANDIDATE_SPEAKING in events
    assert "articulation" in events[EventType.CANDIDATE_SPEAKING]["description"]


def test_behavioural_observations_require_a_face_in_frame():
    """With no face, ``NO_FACE`` already describes the scene.

    Reporting that a candidate's hands are missing while the candidate themselves
    is missing is noise, and would double-count one situation as two observations.
    """
    observer = _observer(StrictnessLevel.MAXIMUM)
    events = observer.map_to_events(
        _observation(
            facial_dynamics=FacialDynamicsResult(face_found=False, is_speaking=True),
            hand_analysis=HandAnalysisResult(hands_detected=0, hands_visible=False),
        )
    )
    assert events == {}


def test_hand_at_ear_supersedes_the_generic_hand_near_face():
    """One gesture must not produce two overlapping incidents."""
    observer = _observer(StrictnessLevel.MAXIMUM)
    hands = HandAnalysisResult(
        hands_detected=1,
        hands_visible=True,
        hand_near_face=True,
        hand_near_ear=True,
        hands=[HandObservation("Right", 0.9, (10, 10, 50, 50), (30, 30))],
    )
    events = observer.map_to_events(_observation(facial_dynamics=_dynamics(), hand_analysis=hands))
    assert EventType.HAND_NEAR_EAR in events
    assert EventType.HAND_NEAR_FACE not in events


def test_looking_away_description_carries_the_measured_angles():
    observer = _observer(StrictnessLevel.STRICT)
    events = observer.map_to_events(
        _observation(
            facial_dynamics=_dynamics(
                is_looking_away=True, head_pose=HeadPose(yaw=41.0, pitch=-8.0)
            )
        )
    )
    description = events[EventType.LOOKING_AWAY]["description"]
    assert "41" in description and "yaw" in description


def test_liveness_failure_is_reported_from_strict_upward():
    observer = _observer(StrictnessLevel.STRICT)
    events = observer.map_to_events(
        _observation(
            facial_dynamics=_dynamics(liveness_state="NO_BLINK_DETECTED", observed_seconds=62.0)
        )
    )
    assert EventType.POSSIBLE_PRESENTATION_ATTACK in events
    description = events[EventType.POSSIBLE_PRESENTATION_ATTACK]["description"]
    assert "blinks rarely" in description, "the innocent explanation must travel with the finding"


def test_wearable_detections_become_events_with_their_reliability():
    observer = _observer(StrictnessLevel.STRICT)
    wearables = WearableAnalysisResult(
        ran=True,
        detections=[
            WearableDetection(
                "earbuds",
                "bluetooth earpiece",
                EventType.EARBUDS_SUSPECTED,
                0.33,
                (10, 10, 20, 20),
                ear_anchored=True,
                reliability="low",
            ),
        ],
    )
    events = observer.map_to_events(_observation(facial_dynamics=_dynamics(), wearables=wearables))
    assert EventType.EARBUDS_SUSPECTED in events
    assert "reliability low" in events[EventType.EARBUDS_SUSPECTED]["description"]


# ---------------------------------------------------------------------------
# Debouncing
# ---------------------------------------------------------------------------


def _run_sequence(observer, present_flags, event_type=EventType.LOOKING_AWAY):
    out = []
    for present in present_flags:
        active = {event_type: {"confidence": 1.0}} if present else {}
        out.append(event_type in observer._debounce(active, observer.policy))
    return out


def test_single_frame_dropouts_are_bridged():
    """A tracker losing one frame must not split one behaviour into several incidents."""
    observer = _observer(window=3)
    result = _run_sequence(observer, [True, True, False, True, True, False, True, True])
    assert all(result), f"dropouts were not bridged: {result}"


def test_isolated_single_frame_blips_are_suppressed():
    """One noisy frame must not manufacture an observation."""
    observer = _observer(window=3)
    result = _run_sequence(observer, [False, False, True, False, False, False])
    assert not any(result), f"an isolated blip survived: {result}"


def test_smoothing_can_be_disabled():
    observer = _observer(window=1)
    flags = [True, False, True, False]
    assert _run_sequence(observer, flags) == flags


def test_sustained_behaviour_is_reported_promptly():
    """Debouncing must not delay a genuine observation by more than a frame or two."""
    observer = _observer(window=3)
    result = _run_sequence(observer, [True] * 5)
    assert result[0] is True, "a behaviour present from the first frame should report immediately"


def test_wearables_bypass_the_debounce():
    """Wearables already run on a decimated cadence; a second smoothing layer adds only lag."""
    observer = _observer(StrictnessLevel.STRICT, window=5)
    active = {EventType.HEADPHONES_DETECTED: {"confidence": 0.6}}
    assert EventType.HEADPHONES_DETECTED in observer._debounce(active, observer.policy)


def test_reset_clears_debounce_history():
    observer = _observer(window=3)
    _run_sequence(observer, [True, True, True])
    observer.reset()
    assert _run_sequence(observer, [True]) == [True]
    assert observer._hands_absent_since is None


# ---------------------------------------------------------------------------
# Hands-absent grace
# ---------------------------------------------------------------------------


def test_hands_leaving_view_is_only_reported_after_the_grace_period():
    """Hands leave frame constantly and innocently; only a sustained absence counts."""
    observer = _observer(StrictnessLevel.MAXIMUM)
    observer.policy.hands_absent_grace_seconds = 5.0
    hands = HandAnalysisResult(hands_detected=0, hands_visible=False)

    early = observer.map_to_events(
        _observation(timestamp_seconds=0.0, facial_dynamics=_dynamics(), hand_analysis=hands)
    )
    assert EventType.HANDS_NOT_VISIBLE not in early

    late = observer.map_to_events(
        _observation(timestamp_seconds=9.0, facial_dynamics=_dynamics(), hand_analysis=hands)
    )
    assert EventType.HANDS_NOT_VISIBLE in late


def test_hands_reappearing_restarts_the_grace_period():
    observer = _observer(StrictnessLevel.MAXIMUM)
    observer.policy.hands_absent_grace_seconds = 5.0
    absent = HandAnalysisResult(hands_detected=0, hands_visible=False)
    present = HandAnalysisResult(hands_detected=1, hands_visible=True)

    observer.map_to_events(
        _observation(timestamp_seconds=0.0, facial_dynamics=_dynamics(), hand_analysis=absent)
    )
    observer.map_to_events(
        _observation(timestamp_seconds=3.0, facial_dynamics=_dynamics(), hand_analysis=present)
    )
    events = observer.map_to_events(
        _observation(timestamp_seconds=6.0, facial_dynamics=_dynamics(), hand_analysis=absent)
    )
    assert EventType.HANDS_NOT_VISIBLE not in events


# ---------------------------------------------------------------------------
# Annotation context
# ---------------------------------------------------------------------------


def test_annotation_context_collects_everything_drawable():
    observer = _observer(StrictnessLevel.MAXIMUM)
    hands = HandAnalysisResult(
        hands_detected=1,
        hands_visible=True,
        hands=[
            HandObservation(
                "Left", 0.8, (10, 10, 60, 60), (35, 35), landmarks=np.zeros((21, 2), np.float32)
            )
        ],
    )
    observation = _observation(
        face_boxes=[(100, 100, 80, 80)],
        prohibited_objects=[
            {"class_name": "cell phone", "confidence": 0.7, "bbox": (5, 5, 40, 40)}
        ],
        facial_dynamics=_dynamics(
            head_pose=HeadPose(yaw=10.0, pitch=2.0),
            mouth_open_ratio=0.1,
            ear_regions=[(1, 1, 5, 5)],
        ),
        hand_analysis=hands,
        similarity=0.71,
    )
    context = observer.build_annotation_context(observation)

    assert context.face_boxes == [(100, 100, 180, 180)], "face box must be converted to x1y1x2y2"
    assert len(context.hand_landmarks) == 1
    assert context.object_boxes[0][0] == "cell phone"
    assert context.measurements["yaw"] == 10.0
    assert context.measurements["identity sim"] == 0.71


def test_annotation_context_omits_unmeasured_stages():
    """A stage that did not run contributes nothing rather than a placeholder."""
    observer = _observer()
    context = observer.build_annotation_context(_observation())
    assert context.hand_landmarks == []
    assert context.object_boxes == []
    assert context.ear_regions == []
