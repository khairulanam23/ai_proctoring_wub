"""Unit tests for EvidenceBuilder, EvidenceManifest, key-frame selection, side panels, and ZIP packaging."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from proctoring.capture.video import VideoMetadata
from proctoring.detection.object_detector import DetectedObject
from tools.research.legacy_object_evidence import (
    EvidenceBuilder,
    EvidenceFrame,
    EvidenceObject,
    FaceEvidenceAdapter,
)
from tools.research.legacy_object_temporal import (
    ObjectPresenceEvent,
)
from tools.research.legacy_object_video import TimelineEntry, VideoAnalysisReport


def test_evidence_object_and_frame_serialization() -> None:
    """Test EvidenceObject and EvidenceFrame serialization and factual reason fields."""
    obj = EvidenceObject(
        object_id="frame_000015_phone_01",
        instance_id="phone_01",
        class_name="cell phone",
        confidence=0.8876,
        bbox=(50, 60, 120, 180),
        reason="Detected object class: cell phone",
        timestamp_seconds=0.500,
        frame_index=15,
        crop_relative_path="crops/frame_000015_phone_01.jpg",
    )
    data = obj.to_dict()
    assert data["object_id"] == "frame_000015_phone_01"
    assert data["instance_id"] == "phone_01"
    assert data["class_name"] == "cell phone"
    assert data["confidence"] == 0.8876
    assert data["reason"] == "Detected object class: cell phone"
    assert data["timestamp_seconds"] == 0.500
    assert data["frame_index"] == 15
    assert data["crop_relative_path"] == "crops/frame_000015_phone_01.jpg"

    frame = EvidenceFrame(
        frame_id="frame_000015",
        frame_index=15,
        timestamp_seconds=0.500,
        formatted_timestamp="00:00.500",
        person_count=1,
        objects=[obj],
        frame_relative_path="frames/frame_000015_00_00_500.jpg",
        width=640,
        height=480,
        selection_reason="PEAK CONFIDENCE",
        explanation="Key evidence frame captured due to peak confidence.",
        video_position_pct=50.0,
        event_context={"event_id": "obj_event_001", "duration_seconds": 1.5},
    )
    f_data = frame.to_dict()
    assert f_data["frame_id"] == "frame_000015"
    assert len(f_data["objects"]) == 1
    assert "explanation" in f_data
    assert f_data["video_position_pct"] == 50.0
    assert f_data["event_context"]["event_id"] == "obj_event_001"


def test_factual_detection_reasons() -> None:
    """Test generating factual, non-subjective reasons for detection markings."""
    # Single person
    r_p1 = EvidenceBuilder.get_factual_detection_reason("person", instance_index=1)
    assert "Detected object class: person" in r_p1

    # Additional person
    r_p2 = EvidenceBuilder.get_factual_detection_reason("person", instance_index=2, person_count=2)
    assert "person (#02 in scene)" in r_p2

    # Phone
    r_phone = EvidenceBuilder.get_factual_detection_reason("cell phone")
    assert r_phone == "Detected object class: cell phone"

    # Laptop
    r_laptop = EvidenceBuilder.get_factual_detection_reason("laptop")
    assert r_laptop == "Detected object class: laptop"

    # Book
    r_book = EvidenceBuilder.get_factual_detection_reason("book")
    assert r_book == "Detected object class: book"

    # Person transition
    r_trans = EvidenceBuilder.get_factual_detection_reason(
        "person", is_person_transition=True, prev_count=1, new_count=2
    )
    assert "1 to 2" in r_trans


def test_crop_object_bounds_clamping_and_header() -> None:
    """Test bounding box clamping to image boundaries with padding and header banner."""
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_img[20:40, 20:40] = 255  # white square

    # Normal inside bbox with header
    crop1 = EvidenceBuilder.crop_object(
        image=dummy_img,
        bbox=(20, 20, 40, 40),
        class_name="cell phone",
        confidence=0.88,
        instance_id="phone_01",
        frame_index=10,
        timestamp_seconds=0.33,
        reason="Detected object class: cell phone",
        padding_ratio=0.10,
        add_header=True,
    )
    assert crop1.shape[0] > 50  # Header adds banner height
    assert crop1.shape[1] >= 340  # Minimum banner width
    assert crop1.shape[2] == 3

    # Boundary crossing bbox (negative & oversized coordinates)
    crop2 = EvidenceBuilder.crop_object(
        image=dummy_img,
        bbox=(-10, -5, 120, 150),
        padding_ratio=0.0,
        add_header=False,
    )
    assert crop2.shape[0] == 100
    assert crop2.shape[1] == 100

    # Invalid image raises ValueError
    with pytest.raises(ValueError):
        EvidenceBuilder.crop_object(None, (10, 10, 20, 20))  # type: ignore


def test_render_annotated_evidence_frame_side_panel() -> None:
    """Test rendering visual evidence frame with side explanation panel."""
    dummy = np.full((360, 480, 3), 100, dtype=np.uint8)
    objs = [
        DetectedObject(0, "person", 0.95, (50, 50, 200, 300)),
        DetectedObject(67, "cell phone", 0.87, (220, 200, 280, 290)),
    ]

    annotated = EvidenceBuilder.render_annotated_evidence_frame(
        image=dummy,
        frame_index=15,
        timestamp_seconds=0.5,
        person_count=1,
        objects=objs,
        selection_reason="PEAK CONFIDENCE",
        video_filename="test_video.mp4",
        total_video_frames=30,
        total_video_duration_seconds=1.0,
        event_context={
            "event_id": "obj_event_001",
            "duration_seconds": 1.5,
            "max_confidence": 0.87,
        },
    )

    # Annotated width includes frame width (480) + side panel (>=420)
    assert annotated.shape[1] > 480
    assert annotated.shape[0] > 360  # includes top & bottom banners
    assert annotated.shape[2] == 3


def test_render_portrait_phone_video_evidence_frame() -> None:
    """Test rendering self-contained evidence frame on 1080x1920 portrait phone video."""
    portrait_dummy = np.full((1920, 1080, 3), 80, dtype=np.uint8)
    objs = [
        DetectedObject(0, "person", 0.96, (200, 300, 900, 1700)),
        DetectedObject(67, "cell phone", 0.89, (742, 1012, 901, 1284)),
    ]

    annotated = EvidenceBuilder.render_annotated_evidence_frame(
        image=portrait_dummy,
        frame_index=219,
        timestamp_seconds=7.500,
        person_count=2,
        objects=objs,
        selection_reason="PEAK CONFIDENCE",
        video_filename="VID_20260824_155225.mp4",
        total_video_frames=412,
        total_video_duration_seconds=14.06,
        event_context={
            "event_id": "obj_event_003",
            "first_seen": "00:05.000",
            "last_seen": "00:07.500",
            "duration_seconds": 2.500,
            "detection_count": 6,
            "max_confidence": 0.91,
        },
    )

    # Original height preserved plus top/bottom banners
    assert annotated.shape[0] > 1920
    # Original width plus side panel
    assert annotated.shape[1] > 1080
    assert annotated.shape[2] == 3


def test_face_evidence_adapter_contract() -> None:
    """Test FaceEvidenceAdapter mapping into standardized EvidenceEvent."""
    event = FaceEvidenceAdapter.create_face_presence_event(
        event_id="face_event_001",
        status="SINGLE_FACE",
        start_seconds=0.0,
        end_seconds=4.0,
        frame_indices=[0, 15, 30, 45, 60],
        face_count=1,
        key_frame_ids=["frame_000000"],
    )

    assert event.event_id == "face_event_001"
    assert event.modality == "face"
    assert event.event_type == "face_observation"
    assert event.duration_seconds == 4.0
    assert event.key_frame_ids == ["frame_000000"]
    assert event.metadata["presence_status"] == "SINGLE_FACE"
    assert hasattr(event, "reason")

    data = event.to_dict()
    assert data["modality"] == "face"
    assert "presence_status" in data["metadata"]
    assert "reason" in data


def test_evidence_builder_complete_package_and_zip(tmp_path: Path) -> None:
    """Test EvidenceBuilder package generation, manifest creation, and zip archiving."""
    meta = VideoMetadata(
        width=640, height=480, source_fps=30.0, total_frames=120, duration_seconds=4.0
    )

    # Build dummy timeline entries
    objs_f0 = [DetectedObject(0, "person", 0.94, (100, 100, 300, 400))]
    objs_f15 = [
        DetectedObject(0, "person", 0.95, (100, 100, 300, 400)),
        DetectedObject(67, "cell phone", 0.88, (200, 250, 250, 320)),
    ]
    objs_f30 = [
        DetectedObject(0, "person", 0.94, (100, 100, 300, 400)),
        DetectedObject(67, "cell phone", 0.93, (200, 250, 250, 320)),
    ]
    objs_f45 = [
        DetectedObject(0, "person", 0.92, (100, 100, 300, 400)),
    ]

    timeline = [
        TimelineEntry(0, 0.0, "00:00.000", 1, objs_f0, [], 1, 1, 10.0, 11.0, "cpu"),
        TimelineEntry(15, 0.5, "00:00.500", 1, objs_f15, [], 2, 2, 10.0, 11.0, "cpu"),
        TimelineEntry(30, 1.0, "00:01.000", 1, objs_f30, [], 2, 2, 10.0, 11.0, "cpu"),
        TimelineEntry(45, 1.5, "00:01.500", 1, objs_f45, [], 1, 1, 10.0, 11.0, "cpu"),
    ]

    temp_events = [
        ObjectPresenceEvent(
            event_id="obj_event_001",
            object_class="person",
            start_seconds=0.0,
            end_seconds=1.5,
            duration_seconds=1.5,
            formatted_start="00:00.000",
            formatted_end="00:01.500",
            detection_count=4,
            max_confidence=0.95,
            average_confidence=0.9375,
            is_duration_qualified=True,
            observation_timestamps=[0.0, 0.5, 1.0, 1.5],
            representative_bbox=(100, 100, 300, 400),
        ),
        ObjectPresenceEvent(
            event_id="obj_event_002",
            object_class="cell phone",
            start_seconds=0.5,
            end_seconds=1.0,
            duration_seconds=0.5,
            formatted_start="00:00.500",
            formatted_end="00:01.000",
            detection_count=2,
            max_confidence=0.93,
            average_confidence=0.905,
            is_duration_qualified=False,
            observation_timestamps=[0.5, 1.0],
            representative_bbox=(200, 250, 250, 320),
        ),
    ]

    report = VideoAnalysisReport(
        video_path="test_video.mp4",
        metadata=meta,
        target_sampling_fps=2.0,
        total_sampled_frames=4,
        timeline=timeline,
        presence_summary={},
        total_processing_time_ms=50.0,
        average_inference_time_ms=10.0,
        average_frame_processing_time_ms=11.0,
        approximate_fps=90.0,
        temporal_events=temp_events,
        person_count_changes=[],
    )

    # Frame cache with synthetic test images
    frame_cache = {
        0: np.full((480, 640, 3), 50, dtype=np.uint8),
        15: np.full((480, 640, 3), 100, dtype=np.uint8),
        30: np.full((480, 640, 3), 150, dtype=np.uint8),
        45: np.full((480, 640, 3), 200, dtype=np.uint8),
    }

    output_dir = tmp_path / "evidence_package"
    builder = EvidenceBuilder(crop_padding_ratio=0.10)
    pkg = builder.build_package(
        report=report,
        output_dir=output_dir,
        frame_cache=frame_cache,
        create_zip=True,
    )

    # 1. Directory Structure Checks
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "frames").is_dir()
    assert (output_dir / "crops").is_dir()
    assert (output_dir / "events").is_dir()
    assert (output_dir / "timeline").is_dir()

    # 2. Manifest Checks
    with open(output_dir / "manifest.json", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert manifest_data["schema_version"] == "1.0"
    assert "package_id" in manifest_data
    assert len(manifest_data["events"]) == 2
    assert len(manifest_data["frames"]) > 0

    # Ensure relative paths and reason fields
    for ev in manifest_data["events"]:
        assert "reason" in ev
        assert "Detected object class" in ev["reason"] or "Person" in ev["reason"]

    for fr in manifest_data["frames"]:
        assert not fr["frame_relative_path"].startswith("/")
        assert "explanation" in fr
        for ob in fr["objects"]:
            assert "reason" in ob
            assert "instance_id" in ob
            if ob["crop_relative_path"]:
                assert not ob["crop_relative_path"].startswith("/")

    # 3. Crops verification
    crop_files = list((output_dir / "crops").glob("*.jpg"))
    assert len(crop_files) > 0

    # 4. ZIP verification
    assert pkg.zip_path is not None
    assert pkg.zip_path.exists()
    with zipfile.ZipFile(pkg.zip_path, "r") as zf:
        namelist = zf.namelist()
        assert "manifest.json" in namelist
        assert any(name.startswith("frames/") for name in namelist)
        assert any(name.startswith("crops/") for name in namelist)
        assert any(name.startswith("events/") for name in namelist)
