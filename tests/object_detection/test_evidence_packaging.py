"""Unit tests for EvidenceBuilder, EvidenceManifest, key-frame selection, and ZIP packaging."""

import json
from pathlib import Path
import zipfile
import numpy as np
import pytest

from src.object_detection.detector import DetectedObject
from src.object_detection.evidence import (
    EvidenceBuilder,
    EvidenceEvent,
    EvidenceFrame,
    EvidenceManifest,
    EvidenceObject,
    EvidencePackage,
    EvidenceSource,
    FaceEvidenceAdapter,
)
from src.object_detection.sampling import VideoMetadata
from src.object_detection.temporal import ObjectPresenceEvent, PersonCountChangeEvent, TemporalEventReport
from src.object_detection.video import TimelineEntry, VideoAnalysisReport


def test_evidence_object_and_frame_serialization() -> None:
    """Test EvidenceObject and EvidenceFrame serialization and relative path handling."""
    obj = EvidenceObject(
        object_id="frame_000015_obj_01",
        class_name="cell phone",
        confidence=0.8876,
        bbox=(50, 60, 120, 180),
        crop_relative_path="crops/frame_000015_cell_phone_01.jpg",
    )
    data = obj.to_dict()
    assert data["object_id"] == "frame_000015_obj_01"
    assert data["class_name"] == "cell phone"
    assert data["confidence"] == 0.8876
    assert data["crop_relative_path"] == "crops/frame_000015_cell_phone_01.jpg"

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
        selection_reason="event_peak (cell phone)",
    )
    f_data = frame.to_dict()
    assert f_data["frame_id"] == "frame_000015"
    assert len(f_data["objects"]) == 1
    assert f_data["selection_reason"] == "event_peak (cell phone)"


def test_crop_object_bounds_clamping_and_padding() -> None:
    """Test bounding box clamping to image boundaries with padding."""
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_img[20:40, 20:40] = 255  # white square

    # Normal inside bbox
    crop1 = EvidenceBuilder.crop_object(dummy_img, (20, 20, 40, 40), padding_ratio=0.10)
    assert crop1.shape[0] > 20
    assert crop1.shape[1] > 20
    assert crop1.shape[2] == 3

    # Boundary crossing bbox (negative & oversized coordinates)
    crop2 = EvidenceBuilder.crop_object(dummy_img, (-10, -5, 120, 150), padding_ratio=0.0)
    assert crop2.shape[0] == 100
    assert crop2.shape[1] == 100

    # Invalid image raises ValueError
    with pytest.raises(ValueError):
        EvidenceBuilder.crop_object(None, (10, 10, 20, 20))  # type: ignore


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

    data = event.to_dict()
    assert data["modality"] == "face"
    assert "presence_status" in data["metadata"]


def test_evidence_builder_complete_package_and_zip(tmp_path: Path) -> None:
    """Test EvidenceBuilder package generation, manifest creation, and zip archiving."""
    meta = VideoMetadata(width=640, height=480, source_fps=30.0, total_frames=120, duration_seconds=4.0)

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
    with open(output_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert manifest_data["schema_version"] == "1.0"
    assert "package_id" in manifest_data
    assert len(manifest_data["events"]) == 2
    assert len(manifest_data["frames"]) > 0

    # Ensure relative paths only
    for fr in manifest_data["frames"]:
        assert not fr["frame_relative_path"].startswith("/")
        for ob in fr["objects"]:
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
