"""Unit and regression tests for UnifiedTemporalAggregator incident lifecycle and spatial persistence."""

import pytest

from proctoring.core.events import (
    DetectorInfo,
    EventSeverity,
    EventStatus,
    EventType,
)
from proctoring.temporal.aggregator import ActiveIncident, UnifiedTemporalAggregator


@pytest.fixture
def dummy_detector():
    return DetectorInfo(name="TestDetector", version="1.0")


def test_active_incident_lifecycle_progression(dummy_detector):
    """Verify ActiveIncident progresses through OPEN -> ACTIVE -> QUALIFIED."""
    inc = ActiveIncident(
        event_type=EventType.PHONE_DETECTED,
        severity=EventSeverity.HIGH,
        detector=dummy_detector,
        start_timestamp=0.0,
        last_seen_timestamp=0.0,
    )
    assert inc.status == EventStatus.OPEN

    # 1. Add first observation
    inc.add_observation(
        timestamp=0.0,
        frame_index=0,
        confidence=0.8,
        bbox=(10, 10, 100, 100),
        blur_variance=80.0,
        required_duration=1.0,
    )
    assert inc.status == EventStatus.OPEN

    # 2. Add second observation at 0.5s (ACTIVE, but not yet qualified)
    inc.add_observation(
        timestamp=0.5,
        frame_index=5,
        confidence=0.85,
        bbox=(12, 12, 102, 102),
        blur_variance=120.0,
        required_duration=1.0,
    )
    assert inc.status == EventStatus.ACTIVE

    # 3. Add third observation at 1.2s (duration >= 1.0s -> QUALIFIED)
    inc.add_observation(
        timestamp=1.2,
        frame_index=12,
        confidence=0.9,
        bbox=(15, 15, 105, 105),
        blur_variance=150.0,
        required_duration=1.0,
    )
    assert inc.status == EventStatus.QUALIFIED


def test_best_evidence_quality_selection(dummy_detector):
    """Verify that the frame with highest composite quality (confidence + sharpness) is selected."""
    inc = ActiveIncident(
        event_type=EventType.PHONE_DETECTED,
        severity=EventSeverity.HIGH,
        detector=dummy_detector,
        start_timestamp=0.0,
        last_seen_timestamp=0.0,
    )

    # Frame 0: High confidence (0.95), but very blurry (blur_var = 10.0)
    inc.add_observation(
        timestamp=0.0,
        frame_index=0,
        confidence=0.95,
        bbox=(10, 10, 50, 50),
        blur_variance=10.0,
    )

    # Frame 5: Moderate confidence (0.92), but extremely sharp (blur_var = 180.0)
    inc.add_observation(
        timestamp=0.5,
        frame_index=5,
        confidence=0.92,
        bbox=(15, 15, 55, 55),
        blur_variance=180.0,
    )

    # Best frame index should be 5 because of sharpness
    assert inc.best_frame_index == 5
    assert inc.best_timestamp == 0.5


def test_dropout_bridging_prevents_fragmentation(dummy_detector):
    """Verify intermittent 1-frame dropout within tolerance does not split incident."""
    aggregator = UnifiedTemporalAggregator(
        session_id="test_session",
        absence_tolerance_seconds=1.0,
        min_event_duration_seconds=1.0,
    )

    # Frame 0: Phone visible at t=0.0s
    aggregator.update_object_observations(
        detected_objects=[
            {"class_name": "cell phone", "confidence": 0.9, "bbox": (50, 50, 150, 150)}
        ],
        timestamp=0.0,
        frame_index=0,
        detector=dummy_detector,
    )

    # Frame 1: Phone temporarily dropped at t=0.2s (dropout)
    aggregator.update_object_observations(
        detected_objects=[],
        timestamp=0.2,
        frame_index=1,
        detector=dummy_detector,
    )

    # Frame 2: Phone reappears at t=0.4s
    aggregator.update_object_observations(
        detected_objects=[
            {"class_name": "cell phone", "confidence": 0.92, "bbox": (52, 52, 152, 152)}
        ],
        timestamp=0.4,
        frame_index=2,
        detector=dummy_detector,
    )

    # Frame 3: Phone visible at t=1.2s
    aggregator.update_object_observations(
        detected_objects=[
            {"class_name": "cell phone", "confidence": 0.95, "bbox": (55, 55, 155, 155)}
        ],
        timestamp=1.2,
        frame_index=3,
        detector=dummy_detector,
    )

    # Close stream at t=2.5s (dropout > 1.0s)
    aggregator.update_object_observations(
        detected_objects=[],
        timestamp=2.5,
        frame_index=4,
        detector=dummy_detector,
    )

    events = aggregator.get_all_events()
    # Should produce exactly 1 continuous qualified event, NOT multiple fragmented events
    assert len(events) == 1
    assert events[0].event_type == EventType.PHONE_DETECTED
    assert events[0].status == EventStatus.QUALIFIED
    assert events[0].duration >= 1.2


def test_spatial_iou_tracking_multi_objects(dummy_detector):
    """Verify distinct spatial bounding boxes track separate instances."""
    aggregator = UnifiedTemporalAggregator(
        session_id="test_session",
        absence_tolerance_seconds=0.5,
        min_event_duration_seconds=0.5,
    )

    # Two distinct phones on screen simultaneously
    obj_left = {"class_name": "cell phone", "confidence": 0.88, "bbox": (10, 10, 80, 80)}
    obj_right = {"class_name": "cell phone", "confidence": 0.91, "bbox": (400, 300, 500, 400)}

    aggregator.update_object_observations(
        detected_objects=[obj_left, obj_right],
        timestamp=0.0,
        frame_index=0,
        detector=dummy_detector,
    )

    aggregator.update_object_observations(
        detected_objects=[obj_left, obj_right],
        timestamp=0.6,
        frame_index=6,
        detector=dummy_detector,
    )

    # Flush session
    events = aggregator.flush()
    phone_events = [e for e in events if e.event_type == EventType.PHONE_DETECTED]
    assert len(phone_events) == 2
