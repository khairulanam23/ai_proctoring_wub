"""Unit tests for TemporalEventEngine, ObjectPresenceEvent, and PersonCountChangeEvent."""

import json
from pathlib import Path
import pytest

from src.object_detection.detector import DetectedObject
from src.object_detection.temporal import (
    ObjectPresenceEvent,
    PersonCountChangeEvent,
    TemporalEventEngine,
    TemporalEventReport,
    render_temporal_gantt_chart,
)


def make_detected_obj(name: str, conf: float = 0.90, bbox=(10, 10, 100, 100)) -> DetectedObject:
    return DetectedObject(class_id=0, class_name=name, confidence=conf, bbox=bbox)


def test_consecutive_detections_consolidated_into_single_event() -> None:
    """Test that consecutive frame detections of the same class merge into a single event."""
    engine = TemporalEventEngine(absence_tolerance_seconds=1.0, min_event_duration_seconds=1.0)

    # Ingest 4 consecutive frames (0.0s, 0.5s, 1.0s, 1.5s)
    engine.ingest_frame(0.0, 0, 1, [make_detected_obj("cell phone", 0.85)])
    engine.ingest_frame(0.5, 15, 1, [make_detected_obj("cell phone", 0.92)])
    engine.ingest_frame(1.0, 30, 1, [make_detected_obj("cell phone", 0.88)])
    engine.ingest_frame(1.5, 45, 1, [make_detected_obj("cell phone", 0.90)])

    report = engine.finalize(final_timestamp=2.0)

    assert report.total_events == 1
    event = report.events[0]
    assert event.object_class == "cell phone"
    assert event.start_seconds == 0.0
    assert event.end_seconds == 1.5
    assert event.duration_seconds == 1.5
    assert event.detection_count == 4
    assert event.max_confidence == 0.92
    assert event.is_duration_qualified is True
    assert len(event.observation_timestamps) == 4


def test_absence_tolerance_bridging() -> None:
    """Test bridging over a 1-frame gap when gap <= absence_tolerance."""
    engine = TemporalEventEngine(absence_tolerance_seconds=1.0, min_event_duration_seconds=1.0)

    # Frame 0: phone present (t=0.0)
    engine.ingest_frame(0.0, 0, 1, [make_detected_obj("cell phone", 0.88)])
    # Frame 15: phone MISSING (t=0.5) -> gap is 0.5s <= 1.0s
    engine.ingest_frame(0.5, 15, 1, [])
    # Frame 30: phone present (t=1.0) -> bridged!
    engine.ingest_frame(1.0, 30, 1, [make_detected_obj("cell phone", 0.91)])

    report = engine.finalize(final_timestamp=1.5)

    assert report.total_events == 1
    assert report.events[0].start_seconds == 0.0
    assert report.events[0].end_seconds == 1.0
    assert report.events[0].duration_seconds == 1.0
    assert report.events[0].detection_count == 2


def test_absence_tolerance_exceeded_splits_events() -> None:
    """Test that a gap exceeding absence_tolerance creates two separate events."""
    engine = TemporalEventEngine(absence_tolerance_seconds=1.0, min_event_duration_seconds=0.5)

    # Event 1: t=0.0 to t=0.5
    engine.ingest_frame(0.0, 0, 1, [make_detected_obj("cell phone", 0.85)])
    engine.ingest_frame(0.5, 15, 1, [make_detected_obj("cell phone", 0.87)])

    # Gap of 2.0 seconds (t=0.5 to t=2.5) -> exceeds tolerance (1.0s)
    engine.ingest_frame(1.0, 30, 1, [])
    engine.ingest_frame(1.5, 45, 1, [])
    engine.ingest_frame(2.0, 60, 1, [])

    # Event 2: t=2.5 to t=3.0
    engine.ingest_frame(2.5, 75, 1, [make_detected_obj("cell phone", 0.90)])
    engine.ingest_frame(3.0, 90, 1, [make_detected_obj("cell phone", 0.93)])

    report = engine.finalize(final_timestamp=3.5)

    assert report.total_events == 2
    ev1, ev2 = report.events[0], report.events[1]

    assert ev1.start_seconds == 0.0
    assert ev1.end_seconds == 0.5
    assert ev1.detection_count == 2

    assert ev2.start_seconds == 2.5
    assert ev2.end_seconds == 3.0
    assert ev2.detection_count == 2


def test_minimum_event_duration_qualification() -> None:
    """Test that short events are preserved but marked as not duration-qualified."""
    engine = TemporalEventEngine(absence_tolerance_seconds=0.5, min_event_duration_seconds=2.0)

    # Event lasting 0.5s (0.0 to 0.5) < 2.0s min duration
    engine.ingest_frame(0.0, 0, 1, [make_detected_obj("book", 0.80)])
    engine.ingest_frame(0.5, 15, 1, [make_detected_obj("book", 0.82)])
    engine.ingest_frame(1.5, 45, 1, [])  # gap exceeds 0.5s -> closes event

    report = engine.finalize(final_timestamp=2.0)

    assert report.total_events == 1
    assert report.qualified_events_count == 0
    assert report.events[0].is_duration_qualified is False
    assert report.events[0].duration_seconds == 0.5


def test_person_count_transition_tracking() -> None:
    """Test tracking 1 -> 2 -> 1 -> 0 person transitions."""
    engine = TemporalEventEngine()

    engine.ingest_frame(0.0, 0, 1, [make_detected_obj("person")])
    engine.ingest_frame(0.5, 15, 1, [make_detected_obj("person")])
    engine.ingest_frame(1.0, 30, 2, [make_detected_obj("person"), make_detected_obj("person")])
    engine.ingest_frame(1.5, 45, 2, [make_detected_obj("person"), make_detected_obj("person")])
    engine.ingest_frame(2.0, 60, 1, [make_detected_obj("person")])
    engine.ingest_frame(2.5, 75, 0, [])

    report = engine.finalize(final_timestamp=3.0)

    assert len(report.person_count_changes) == 3

    c1 = report.person_count_changes[0]
    assert c1.timestamp_seconds == 1.0
    assert c1.previous_count == 1
    assert c1.new_count == 2

    c2 = report.person_count_changes[1]
    assert c2.timestamp_seconds == 2.0
    assert c2.previous_count == 2
    assert c2.new_count == 1

    c3 = report.person_count_changes[2]
    assert c3.timestamp_seconds == 2.5
    assert c3.previous_count == 1
    assert c3.new_count == 0


def test_multiple_simultaneous_object_classes() -> None:
    """Test independent temporal tracking for multiple classes."""
    engine = TemporalEventEngine(absence_tolerance_seconds=1.0)

    # Person present from 0.0 to 3.0
    # Phone present from 1.0 to 2.0
    # Laptop present from 1.5 to 3.0
    engine.ingest_frame(0.0, 0, 1, [make_detected_obj("person")])
    engine.ingest_frame(0.5, 15, 1, [make_detected_obj("person")])
    engine.ingest_frame(1.0, 30, 1, [make_detected_obj("person"), make_detected_obj("cell phone")])
    engine.ingest_frame(1.5, 45, 1, [make_detected_obj("person"), make_detected_obj("cell phone"), make_detected_obj("laptop")])
    engine.ingest_frame(2.0, 60, 1, [make_detected_obj("person"), make_detected_obj("cell phone"), make_detected_obj("laptop")])
    engine.ingest_frame(2.5, 75, 1, [make_detected_obj("person"), make_detected_obj("laptop")])
    engine.ingest_frame(3.0, 90, 1, [make_detected_obj("person"), make_detected_obj("laptop")])

    report = engine.finalize(final_timestamp=3.5)

    assert report.total_events == 3
    classes_tracked = {e.object_class for e in report.events}
    assert classes_tracked == {"person", "cell phone", "laptop"}

    phone_event = next(e for e in report.events if e.object_class == "cell phone")
    assert phone_event.start_seconds == 1.0
    assert phone_event.end_seconds == 2.0
    assert phone_event.duration_seconds == 1.0


def test_temporal_event_report_json_serialization() -> None:
    """Test JSON serialization of TemporalEventReport."""
    engine = TemporalEventEngine()
    engine.ingest_frame(0.0, 0, 1, [make_detected_obj("person", 0.95, (10, 10, 100, 100))])
    engine.ingest_frame(1.0, 30, 2, [make_detected_obj("person", 0.92), make_detected_obj("cell phone", 0.88)])
    report = engine.finalize(final_timestamp=1.5)

    data = report.to_dict()
    assert "temporal_events" in data
    assert "person_count_changes" in data
    assert len(data["temporal_events"]) == 2
    assert len(data["person_count_changes"]) == 1

    json_str = json.dumps(data)
    assert "cell phone" in json_str
    assert "person" in json_str


def test_render_temporal_gantt_chart(tmp_path: Path) -> None:
    """Test rendering of Gantt timeline chart."""
    engine = TemporalEventEngine()
    engine.ingest_frame(0.0, 0, 1, [make_detected_obj("person")])
    engine.ingest_frame(1.0, 30, 2, [make_detected_obj("person"), make_detected_obj("cell phone")])
    engine.ingest_frame(2.0, 60, 1, [make_detected_obj("person")])
    report = engine.finalize(final_timestamp=3.0)

    out_file = tmp_path / "gantt.png"
    result_path = render_temporal_gantt_chart(report, total_duration_seconds=3.0, output_path=out_file)

    assert result_path.exists()
    assert result_path.stat().st_size > 0
