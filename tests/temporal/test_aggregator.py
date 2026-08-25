"""Tests for Unified Temporal Aggregator."""

from proctoring.core.events import DetectorInfo, EventSeverity, EventStatus, EventType
from proctoring.temporal.aggregator import UnifiedTemporalAggregator


def test_continuous_no_face_aggregation():
    """Verify consecutive NO_FACE frames merge into one continuous EventRecord."""
    agg = UnifiedTemporalAggregator(
        session_id="test_sess",
        absence_tolerance_seconds=1.0,
        min_event_duration_seconds=1.0,
    )
    detector = DetectorInfo(name="YuNet", version="2023mar")

    # Frames 1-10: NO_FACE at 4 FPS (t = 0.0 to 2.25s)
    for i in range(10):
        t = i * 0.25
        agg.update_face_observation(
            face_status="NO_FACE",
            timestamp=t,
            frame_index=i + 1,
            face_count=0,
            detector=detector,
        )

    # Frame 11: SINGLE_FACE at t = 2.50s (candidate returns)
    agg.update_face_observation(
        face_status="SINGLE_FACE",
        timestamp=2.50,
        frame_index=11,
        face_count=1,
        detector=detector,
    )

    # Move time past absence tolerance (e.g. t = 4.0s)
    agg.update_face_observation(
        face_status="SINGLE_FACE",
        timestamp=4.00,
        frame_index=17,
        face_count=1,
        detector=detector,
    )

    events = agg.get_all_events()
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == EventType.NO_FACE
    assert ev.timestamp == 0.0
    assert ev.end_timestamp == 2.25
    assert ev.duration == 2.25
    assert ev.status == EventStatus.QUALIFIED
    assert len(ev.observation.frame_indices) == 10


def test_multiple_faces_aggregation():
    """Verify MULTIPLE_FACES frames merge into a single incident."""
    agg = UnifiedTemporalAggregator(
        session_id="test_sess",
        absence_tolerance_seconds=0.5,
        min_event_duration_seconds=1.0,
    )
    detector = DetectorInfo(name="YuNet", version="2023mar")

    # Frames 1-6: MULTIPLE_FACES at t = 10.0 to 11.25s (1.25s duration)
    for i in range(6):
        t = 10.0 + i * 0.25
        agg.update_face_observation(
            face_status="MULTIPLE_FACES",
            timestamp=t,
            frame_index=i + 1,
            face_count=2,
            detector=detector,
            bboxes=[(50, 50, 100, 100), (250, 50, 100, 100)],
        )

    events = agg.flush()
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == EventType.MULTIPLE_FACES
    assert ev.duration == 1.25
    assert ev.status == EventStatus.QUALIFIED


def test_object_detection_temporal_aggregation():
    """Verify sustained cell phone presence groups into one continuous event."""
    agg = UnifiedTemporalAggregator(
        session_id="test_sess",
        absence_tolerance_seconds=1.0,
        min_event_duration_seconds=1.0,
    )
    detector = DetectorInfo(name="YOLO11", version="11.0")

    # Frames 1-8: cell phone detected at t = 0.0 to 1.75s
    for i in range(8):
        t = i * 0.25
        agg.update_object_observations(
            detected_objects=[
                {"class_name": "cell phone", "confidence": 0.85, "bbox": (100, 100, 150, 200)}
            ],
            timestamp=t,
            frame_index=i + 1,
            detector=detector,
        )

    events = agg.flush()
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == EventType.PHONE_DETECTED
    assert ev.observation.object_class == "cell phone"
    assert ev.duration == 1.75
    assert ev.status == EventStatus.QUALIFIED


def test_sub_duration_event_recorded_not_dropped():
    """Verify sub-duration events are marked RECORDED but NOT silently dropped."""
    agg = UnifiedTemporalAggregator(
        session_id="test_sess",
        absence_tolerance_seconds=0.5,
        min_event_duration_seconds=2.0,  # Strict 2s threshold
    )
    detector = DetectorInfo(name="YuNet", version="2023mar")

    # Only 2 frames (0.25s duration < 2.0s min)
    agg.update_face_observation(
        face_status="NO_FACE",
        timestamp=1.0,
        frame_index=1,
        face_count=0,
        detector=detector,
    )
    agg.update_face_observation(
        face_status="NO_FACE",
        timestamp=1.25,
        frame_index=2,
        face_count=0,
        detector=detector,
    )

    events = agg.flush()
    assert len(events) == 1
    ev = events[0]
    assert ev.status == EventStatus.RECORDED
    assert ev.metadata["is_duration_qualified"] is False
    assert ev.duration == 0.25


def test_instant_browser_event():
    """Verify browser tab switch is recorded as immediate qualified event."""
    agg = UnifiedTemporalAggregator(session_id="test_sess")
    detector = DetectorInfo(name="BrowserBridge", version="1.0")

    ev = agg.record_instant_event(
        event_type=EventType.BROWSER_TAB_SWITCH,
        timestamp=15.3,
        frame_index=61,
        description="Candidate navigated away from examination window (tab switch detected)",
        detector=detector,
    )

    assert ev.event_type == EventType.BROWSER_TAB_SWITCH
    assert ev.severity == EventSeverity.CRITICAL
    assert ev.status == EventStatus.QUALIFIED
    assert ev.duration == 0.0
