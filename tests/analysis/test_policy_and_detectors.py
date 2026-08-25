"""Tests for the exam strictness policy, hand analysis and wearable detection."""

import numpy as np
import pytest

from proctoring.analysis.hands import HandAnalysisResult, HandAnalyzer, HandObservation
from proctoring.analysis.policy import ExamPolicy, StrictnessLevel
from proctoring.analysis.wearables import (
    WearableAnalysisResult,
    WearableDetection,
    WearableDetector,
)
from proctoring.core.events import EventType

# ---------------------------------------------------------------------------
# Strictness policy
# ---------------------------------------------------------------------------


def test_strictness_levels_are_strictly_nested():
    """Each level must report everything the level below it reports, and more."""
    standard = ExamPolicy.for_level(StrictnessLevel.STANDARD)
    strict = ExamPolicy.for_level(StrictnessLevel.STRICT)
    maximum = ExamPolicy.for_level(StrictnessLevel.MAXIMUM)

    assert standard.enabled_events < strict.enabled_events
    assert strict.enabled_events < maximum.enabled_events


def test_higher_strictness_qualifies_events_sooner():
    """Raising strictness must shorten, never lengthen, the qualification durations."""
    durations = [
        ExamPolicy.for_level(level).min_event_duration_seconds
        for level in (StrictnessLevel.STANDARD, StrictnessLevel.STRICT, StrictnessLevel.MAXIMUM)
    ]
    assert durations == sorted(durations, reverse=True)


def test_behavioural_observations_are_off_at_standard_strictness():
    """Ordinary coursework must not flag speaking, gaze or hand posture."""
    policy = ExamPolicy.for_level(StrictnessLevel.STANDARD)
    for event_type in (
        EventType.CANDIDATE_SPEAKING,
        EventType.GAZE_OFF_SCREEN,
        EventType.HAND_NEAR_FACE,
        EventType.HANDS_NOT_VISIBLE,
        EventType.EARBUDS_SUSPECTED,
    ):
        assert policy.allows(event_type) is False


def test_unambiguous_observations_are_reported_at_every_level():
    """Nobody present, someone else present, or a phone is always worth recording."""
    for level in StrictnessLevel:
        policy = ExamPolicy.for_level(level)
        for event_type in (
            EventType.NO_FACE,
            EventType.MULTIPLE_FACES,
            EventType.UNKNOWN_FACE,
            EventType.PHONE_DETECTED,
            EventType.HEADPHONES_DETECTED,
        ):
            assert policy.allows(event_type), f"{event_type.value} disabled at {level.value}"


def test_weak_signals_require_longer_persistence_than_strong_ones():
    """A weak signal must persist longer before it qualifies than a strong one.

    An earbud detection is far less trustworthy than a phone detection, so it has to
    survive longer before it is presented to a proctor as qualified.
    """
    policy = ExamPolicy.for_level(StrictnessLevel.STRICT)
    assert policy.duration_for(EventType.EARBUDS_SUSPECTED) > policy.duration_for(
        EventType.PHONE_DETECTED
    )
    assert policy.duration_for(EventType.CANDIDATE_SPEAKING) > policy.duration_for(
        EventType.PHONE_DETECTED
    )


def test_policy_serialises_for_the_manifest():
    """The policy must round-trip into the manifest so a review is reproducible."""
    data = ExamPolicy.for_level(StrictnessLevel.MAXIMUM).to_dict()
    assert data["level"] == "MAXIMUM"
    assert "CANDIDATE_SPEAKING" in data["enabled_events"]
    assert data["behavioural_thresholds"]["yaw_limit_degrees"] > 0
    assert "earbuds" in data["confidence_floors"]


# ---------------------------------------------------------------------------
# Hand analysis geometry
# ---------------------------------------------------------------------------


def _analyzer() -> HandAnalyzer:
    """A HandAnalyzer with no model loaded — geometry only."""
    analyzer = HandAnalyzer.__new__(HandAnalyzer)
    analyzer.face_proximity_ratio = 1.0
    analyzer.ear_proximity_ratio = 0.55
    return analyzer


def _hand(centroid, fingertips=()) -> HandObservation:
    return HandObservation(
        handedness="Right",
        confidence=0.9,
        bbox=(centroid[0] - 20, centroid[1] - 20, centroid[0] + 20, centroid[1] + 20),
        centroid=centroid,
        fingertip_points=list(fingertips),
    )


def test_hand_far_from_face_is_not_near_it():
    result = HandAnalysisResult(hands=[_hand((500, 400))])
    _analyzer()._relate_to_face(
        result, face_bbox=(100, 100, 200, 220), ear_regions=[], mouth_region=None
    )
    assert result.hand_near_face is False
    assert result.hand_near_ear is False


def test_hand_overlapping_the_face_is_flagged():
    result = HandAnalysisResult(hands=[_hand((150, 160))])
    _analyzer()._relate_to_face(
        result, face_bbox=(100, 100, 200, 220), ear_regions=[], mouth_region=None
    )
    assert result.hand_near_face is True


def test_fingertip_reaching_the_ear_is_detected_before_the_whole_hand_arrives():
    """A hand whose centroid is outside the ear region still counts if a finger is in it.

    A candidate adjusting an earpiece reaches with a fingertip; requiring the hand's
    centre to land on the ear would miss exactly the gesture that matters.
    """
    ear = (90, 130, 130, 170)
    result = HandAnalysisResult(hands=[_hand((260, 300), fingertips=[(110, 150)])])
    _analyzer()._relate_to_face(
        result, face_bbox=(100, 100, 200, 220), ear_regions=[ear], mouth_region=None
    )
    assert result.hand_near_ear is True


def test_proximity_is_normalised_by_face_size():
    """The same pixel distance is near for a large face and far for a small one."""
    analyzer = _analyzer()

    near = HandAnalysisResult(hands=[_hand((300, 160))])
    analyzer._relate_to_face(
        near, face_bbox=(100, 100, 300, 300), ear_regions=[], mouth_region=None
    )

    far = HandAnalysisResult(hands=[_hand((300, 160))])
    analyzer._relate_to_face(far, face_bbox=(100, 100, 140, 140), ear_regions=[], mouth_region=None)

    assert near.nearest_hand_distance_ratio < far.nearest_hand_distance_ratio


def test_hand_analyzer_without_model_is_unavailable():
    analyzer = HandAnalyzer(model_path="models/no_such_hand_model.task")
    assert analyzer.is_available is False
    result = analyzer.analyze(np.full((480, 640, 3), 120, np.uint8))
    assert result.hands_detected is None


# ---------------------------------------------------------------------------
# Wearable detection
# ---------------------------------------------------------------------------


def test_earbuds_never_rate_as_high_reliability():
    """However confident the model is, an in-ear device stays a weak signal.

    The target is a few dozen pixels at webcam distance and is routinely confused
    with earrings and ear shadow, so a raw confidence score must not be allowed to
    present it as a firm finding.
    """
    assert WearableDetector._rate_reliability("earbuds", 0.99, ear_anchored=True) != "high"
    assert WearableDetector._rate_reliability("earbuds", 0.50, ear_anchored=False) == "low"
    assert WearableDetector._rate_reliability("headphones", 0.60, ear_anchored=True) == "high"


def test_overlapping_boxes_are_detected_for_ear_anchoring():
    assert WearableDetector._overlaps_any((100, 100, 140, 140), [(120, 120, 200, 200)]) is True
    assert WearableDetector._overlaps_any((100, 100, 140, 140), [(300, 300, 400, 400)]) is False
    assert WearableDetector._overlaps_any((100, 100, 140, 140), []) is False


def test_duplicate_prompts_collapse_to_one_detection_per_device():
    """Several prompts describing one object must not produce several observations."""
    result = WearableAnalysisResult(
        ran=True,
        detections=[
            WearableDetection(
                "headphones", "headphones", EventType.HEADPHONES_DETECTED, 0.42, (0, 0, 10, 10)
            ),
            WearableDetection(
                "headphones",
                "over-ear headphones",
                EventType.HEADPHONES_DETECTED,
                0.61,
                (0, 0, 10, 10),
            ),
            WearableDetection(
                "earbuds", "bluetooth earpiece", EventType.EARBUDS_SUSPECTED, 0.33, (5, 5, 12, 12)
            ),
        ],
    )
    collapsed = WearableDetector._deduplicate(result)
    assert len(collapsed.detections) == 2
    headphones = next(d for d in collapsed.detections if d.target == "headphones")
    assert headphones.confidence == pytest.approx(0.61)


def test_wearable_detector_without_model_is_unavailable():
    detector = WearableDetector(model_name="definitely-not-a-model.pt", auto_load=True)
    assert detector.is_available is False
    result = detector.detect(np.full((480, 640, 3), 120, np.uint8))
    assert result.ran is False
    assert result.detections == []
