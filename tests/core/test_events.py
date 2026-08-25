"""Tests for unified event and evidence schema."""

import json

from proctoring.core.events import (
    DetectorInfo,
    EventRecord,
    EventSeverity,
    EventStatus,
    EventType,
    EvidenceReference,
    ObservationDetail,
    format_seconds_to_timestamp,
)


def test_event_schema_serialization_and_deserialization():
    """Verify EventRecord converts to/from dictionary and JSON cleanly."""
    detector = DetectorInfo(
        name="OpenCV YuNet",
        version="2023mar",
        model_file="face_detection_yunet_2023mar.onnx",
        device="cpu",
        confidence_threshold=0.60,
    )
    obs = ObservationDetail(
        description="Face not detected for 5.25s across 21 frames",
        frame_indices=[10, 11, 12],
        timestamps=[2.5, 2.75, 3.0],
        raw_confidences=[1.0, 1.0, 1.0],
    )
    ev_ref = EvidenceReference(
        evidence_id="ev_001",
        media_type="frame",
        file_path="evidence/frames/frame_000010.jpg",
        timestamp_seconds=2.5,
        formatted_timestamp="00:00:02.500",
        frame_index=10,
        sha256="abcdef123456",
        file_size_bytes=1024,
        is_validated=True,
    )

    event = EventRecord(
        event_id="evt_test_001",
        session_id="session_101",
        timestamp=2.5,
        end_timestamp=7.75,
        duration=5.25,
        formatted_start="00:00:02.500",
        formatted_end="00:00:07.750",
        event_type=EventType.NO_FACE,
        severity=EventSeverity.HIGH,
        confidence=1.0,
        average_confidence=1.0,
        detector=detector,
        observation=obs,
        evidence=[ev_ref],
        metadata={"custom_key": "custom_val"},
        status=EventStatus.QUALIFIED,
    )

    d = event.to_dict()
    assert d["event_id"] == "evt_test_001"
    assert d["event_type"] == "NO_FACE"
    assert d["severity"] == "HIGH"
    assert d["duration"] == 5.25
    assert len(d["evidence"]) == 1

    # JSON roundtrip
    json_str = json.dumps(d)
    parsed_dict = json.loads(json_str)
    reconstructed = EventRecord.from_dict(parsed_dict)

    assert reconstructed.event_id == event.event_id
    assert reconstructed.event_type == EventType.NO_FACE
    assert reconstructed.severity == EventSeverity.HIGH
    assert reconstructed.duration == event.duration
    assert len(reconstructed.evidence) == 1
    assert reconstructed.evidence[0].evidence_id == "ev_001"
    assert reconstructed.observation.description == event.observation.description


def test_timestamp_formatting():
    """Verify seconds format into standard HH:MM:SS.mmm format."""
    assert format_seconds_to_timestamp(0.0) == "00:00:00.000"
    assert format_seconds_to_timestamp(12.345) == "00:00:12.345"
    assert format_seconds_to_timestamp(65.1) == "00:01:05.100"
    assert format_seconds_to_timestamp(3661.05) == "01:01:01.050"
