"""Automated test suite for Standalone DatasetCollector still-image acquisition system."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from tools.dataset_collector.camera_utils import (
    DiscoveredCamera,
    get_platform_backends,
)
from tools.dataset_collector.collector_app import DatasetCollectorApp
from tools.dataset_collector.config_loader import (
    ActivitiesConfig,
    load_activities_config,
    resolve_resource_path,
)
from tools.dataset_collector.manifest import (
    APPLICATION_VERSION,
    SCHEMA_VERSION,
    ActivityRecord,
    PhotoRecord,
    SessionManifest,
    calculate_sha256,
    check_disk_space,
    generate_session_readme,
    sanitize_identifier,
    write_manifest_and_checksums,
)


def test_activities_yaml_loading():
    """Verifies that configs/capture_scenarios/activities.yaml loads correctly with 45 activities."""
    config = load_activities_config()
    assert isinstance(config, ActivitiesConfig)
    assert config.photos_per_activity == 2
    assert len(config.activities) == 45

    # Check categories and fields on all activities
    valid_categories = {
        "FACE_POSTURE",
        "HANDS_DESK",
        "PHONE_POSITIVE",
        "PHONE_HARD_NEGATIVE",
        "PAPER_POSITIVE",
        "PAPER_HARD_NEGATIVE",
        "EAR_POSITIVE",
        "EAR_HARD_NEGATIVE",
        "BOUNDARY_FRAMING",
    }
    seen_ids = set()
    for act in config.activities:
        assert act.id.startswith("ACT")
        assert act.id not in seen_ids
        seen_ids.add(act.id)
        assert act.category in valid_categories
        assert act.name
        assert act.instructed_condition
        assert act.physical_action
        assert act.visible_state
        assert act.photo_1
        assert act.photo_2
        assert act.purpose


def test_resolve_resource_path(tmp_path: Path):
    """Verifies resource resolution logic including PyInstaller _MEIPASS and filesystem fallback."""
    # Existing relative file
    resolved = resolve_resource_path("configs/capture_scenarios/activities.yaml")
    assert resolved.is_file()

    # Mock PyInstaller _MEIPASS environment
    fake_bundle = tmp_path / "fake_meipass"
    fake_bundle.mkdir()
    fake_config = fake_bundle / "configs" / "test.yaml"
    fake_config.parent.mkdir(parents=True)
    fake_config.write_text("test: true", encoding="utf-8")

    with patch("sys._MEIPASS", str(fake_bundle), create=True):
        meipass_resolved = resolve_resource_path("configs/test.yaml")
        assert meipass_resolved.resolve() == fake_config.resolve()


def test_platform_camera_backends():
    """Verifies camera backend prioritization on Windows vs Linux."""
    backends = get_platform_backends()
    assert len(backends) >= 1

    # Test simulated Windows platform
    with patch("sys.platform", "win32"):
        win_backends = get_platform_backends()
        assert win_backends[0][0] == "DSHOW"
        assert win_backends[1][0] == "MSMF"


def test_identifier_sanitization():
    """Verifies that participant and session IDs are properly validated and sanitized."""
    assert sanitize_identifier("P001") == "P001"
    assert sanitize_identifier("session_01-test") == "session_01-test"

    with pytest.raises(ValueError, match="contains invalid characters"):
        sanitize_identifier("P001/evil")

    with pytest.raises(ValueError, match="contains invalid characters"):
        sanitize_identifier("P001; rm -rf")

    with pytest.raises(ValueError, match="contains invalid characters"):
        sanitize_identifier("P 001 with spaces")


def test_disk_space_guard(tmp_path: Path):
    """Verifies disk space inspection."""
    ok, free_bytes, total_bytes = check_disk_space(tmp_path, min_bytes=1024)
    assert ok is True
    assert free_bytes > 0
    assert total_bytes > 0


def test_manifest_and_checksum_generation(tmp_path: Path):
    """Verifies session manifest serialization, checksum generation, and README output."""
    session_dir = tmp_path / "P001" / "S001"
    images_dir = session_dir / "images" / "ACT001"
    images_dir.mkdir(parents=True)

    # Create dummy photo files
    img1 = np.full((100, 100, 3), 120, dtype=np.uint8)
    img2 = np.full((100, 100, 3), 200, dtype=np.uint8)
    p1_path = images_dir / "photo_01.jpg"
    p2_path = images_dir / "photo_02.jpg"
    cv2.imwrite(str(p1_path), img1)
    cv2.imwrite(str(p2_path), img2)

    p1_rec = PhotoRecord(
        photo_number=1,
        filename=p1_path.name,
        relative_path="images/ACT001/photo_01.jpg",
        width=100,
        height=100,
        file_size_bytes=p1_path.stat().st_size,
        sha256=calculate_sha256(p1_path),
        captured_at="2026-09-16T12:00:00Z",
    )
    p2_rec = PhotoRecord(
        photo_number=2,
        filename=p2_path.name,
        relative_path="images/ACT001/photo_02.jpg",
        width=100,
        height=100,
        file_size_bytes=p2_path.stat().st_size,
        sha256=calculate_sha256(p2_path),
        captured_at="2026-09-16T12:00:05Z",
    )

    act_rec = ActivityRecord(
        activity_id="ACT001",
        activity_name="Frontal screen reading",
        category="FACE_POSTURE",
        instructed_condition="frontal_screen_reading",
        purpose="Positive baseline",
        status="completed",
        photos=[p1_rec, p2_rec],
    )

    manifest = SessionManifest(
        participant_id="P001",
        session_id="S001",
        camera={"backend": "DSHOW", "resolution": [1280, 720]},
        activities=[act_rec],
        session_status="completed",
    )

    m_path, c_path = write_manifest_and_checksums(session_dir, manifest)
    readme_path = generate_session_readme(session_dir, "P001", "S001")

    assert m_path.is_file()
    assert c_path.is_file()
    assert readme_path.is_file()

    # Verify manifest JSON content
    with open(m_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["application_version"] == APPLICATION_VERSION
    assert data["participant_id"] == "P001"
    assert data["session_id"] == "S001"
    assert data["session_status"] == "completed"
    assert data["summary"]["total_activities"] == 1
    assert data["summary"]["completed_activities"] == 1
    assert data["summary"]["total_photos_captured"] == 2

    # Verify checksum.sha256 content
    checksum_text = c_path.read_text(encoding="utf-8")
    assert "images/ACT001/photo_01.jpg" in checksum_text
    assert "images/ACT001/photo_02.jpg" in checksum_text
    assert "session_manifest.json" in checksum_text


def test_two_photo_capture_workflow_and_retake(tmp_path: Path):
    """Verifies that exactly 2 photos can be captured per activity, retake works, and duplicate captures are blocked."""
    app = DatasetCollectorApp(
        participant_id="PTEST",
        session_id="STEST",
        output_dir=tmp_path / "output",
        dry_run=True,
    )
    app.init_session_records()

    act0 = app.activities[0]
    rec0 = app.activity_records[act0.id]
    assert len(rec0.photos) == 0
    assert rec0.status == "pending"

    # Capture Photo 1
    frame1 = np.full((100, 100, 3), 50, dtype=np.uint8)
    ok1 = app.capture_current_photo(frame1)
    assert ok1 is True
    assert len(rec0.photos) == 1
    assert rec0.photos[0].photo_number == 1
    assert rec0.status != "completed"
    p1_file = app.session_dir / rec0.photos[0].relative_path
    assert p1_file.is_file()

    # Capture Photo 2
    frame2 = np.full((100, 100, 3), 150, dtype=np.uint8)
    ok2 = app.capture_current_photo(frame2)
    assert ok2 is True
    assert len(rec0.photos) == 2
    assert rec0.photos[1].photo_number == 2
    assert rec0.status == "completed"
    p2_file = app.session_dir / rec0.photos[1].relative_path
    assert p2_file.is_file()

    # Attempting to capture a 3rd photo must be safely rejected
    frame3 = np.full((100, 100, 3), 200, dtype=np.uint8)
    ok3 = app.capture_current_photo(frame3)
    assert ok3 is False
    assert len(rec0.photos) == 2

    # Test Retake: should remove Photo 2
    app.retake_photo()
    assert len(rec0.photos) == 1
    assert not p2_file.exists()
    assert rec0.status == "incomplete"

    # Re-capture Photo 2
    ok2_retake = app.capture_current_photo(frame2)
    assert ok2_retake is True
    assert len(rec0.photos) == 2
    assert p2_file.is_file()
    assert rec0.status == "completed"


def test_session_resumption(tmp_path: Path):
    """Verifies that an interrupted session can be resumed from disk without re-taking completed activities."""
    out_dir = tmp_path / "resume_test"

    # Session 1: capture 2 photos for ACT001
    app1 = DatasetCollectorApp(
        participant_id="PRESUME",
        session_id="S001",
        output_dir=out_dir,
        dry_run=True,
    )
    app1.init_session_records()
    frame = np.full((100, 100, 3), 100, dtype=np.uint8)
    app1.capture_current_photo(frame)
    app1.capture_current_photo(frame)
    app1.finalize_session(aborted=True)

    # Session 2: open same participant & session
    app2 = DatasetCollectorApp(
        participant_id="PRESUME",
        session_id="S001",
        output_dir=out_dir,
        dry_run=True,
    )
    app2.init_session_records()

    # ACT001 must have 2 photos already loaded
    rec0 = app2.activity_records[app2.activities[0].id]
    assert len(rec0.photos) == 2
    assert rec0.status == "completed"

    # Current activity index should point to ACT002 (index 1)
    assert app2.current_activity_idx == 1


def test_end_to_end_dataset_handoff_simulation(tmp_path: Path):
    """Simulates end-to-end collection, directory transfer to another location, and independent checksum verification."""
    source_dir = tmp_path / "source_collector"
    app = DatasetCollectorApp(
        participant_id="PHANDOFF",
        session_id="S001",
        output_dir=source_dir,
        dry_run=True,
    )
    app.init_session_records()

    # Capture 2 activities
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    app.capture_current_photo(frame)
    app.capture_current_photo(frame)
    app.current_activity_idx = 1
    app.capture_current_photo(frame)
    app.capture_current_photo(frame)

    session_dir = app.finalize_session(aborted=False)
    assert (session_dir / "session_manifest.json").is_file()
    assert (session_dir / "checksum.sha256").is_file()
    assert (session_dir / "README.txt").is_file()

    # Simulate handoff: copy session_dir to destination
    destination_dir = tmp_path / "handoff_recipient" / "PHANDOFF" / "S001"
    shutil.copytree(session_dir, destination_dir)

    # In destination directory, verify all checksums match
    checksum_file = destination_dir / "checksum.sha256"
    assert checksum_file.is_file()

    with open(checksum_file, encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) >= 5  # 4 photos + 1 manifest
    for line in lines:
        expected_hash, rel_path = line.strip().split("  ", 1)
        target_file = destination_dir / rel_path
        assert target_file.is_file(), f"Missing file: {target_file}"
        actual_hash = calculate_sha256(target_file)
        assert actual_hash == expected_hash, f"Hash mismatch for {rel_path}"

        # If it's an image, verify OpenCV can decode it
        if rel_path.endswith(".jpg"):
            dec = cv2.imread(str(target_file))
            assert dec is not None and dec.shape == (100, 100, 3)


def test_paths_with_spaces_and_special_locations(tmp_path: Path):
    """Verifies that the collector operates safely from directories containing spaces."""
    spaced_dir = tmp_path / "My Folder With Spaces" / "Dataset Collector Output"
    app = DatasetCollectorApp(
        participant_id="P_SPACE",
        session_id="S_SPACE",
        output_dir=spaced_dir,
        dry_run=True,
    )
    app.init_session_records()

    frame = np.full((120, 120, 3), 80, dtype=np.uint8)
    ok1 = app.capture_current_photo(frame)
    ok2 = app.capture_current_photo(frame)
    assert ok1 is True
    assert ok2 is True

    session_dir = app.finalize_session(aborted=False)
    assert session_dir.is_dir()
    assert (session_dir / "session_manifest.json").is_file()
    assert (session_dir / "checksum.sha256").is_file()

    p1 = session_dir / "images" / "ACT001" / "photo_01.jpg"
    assert p1.is_file()
    read_back = cv2.imread(str(p1))
    assert read_back is not None and read_back.shape == (120, 120, 3)


def test_camera_disconnected_prevents_capture(tmp_path: Path):
    """Verifies that capturing a photo while camera is disconnected is blocked."""
    app = DatasetCollectorApp(
        participant_id="P_DISCONN",
        session_id="S_DISCONN",
        output_dir=tmp_path / "disconn_output",
        dry_run=False,
    )
    app.init_session_records()
    # Explicitly set camera offline
    app.camera_online = False
    app.cap = None

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    ok = app.capture_current_photo(frame)
    assert ok is False
    assert "ERROR: Cannot capture photo while camera is disconnected!" in app.status_message
    rec = app.activity_records[app.activities[0].id]
    assert len(rec.photos) == 0


def test_full_45_activity_90_photo_session_and_handoff(tmp_path: Path):
    """Executes a full 45-activity session (90 photos), verifying schema, checksums, and dataset handoff."""
    output_base = tmp_path / "Full_Session_Test"
    app = DatasetCollectorApp(
        participant_id="PTEST",
        session_id="STEST",
        output_dir=output_base,
        dry_run=True,
    )
    app.init_session_records()

    total_activities = len(app.activities)
    assert total_activities == 45

    # Simulate capturing 2 photos for each of the 45 activities
    for idx in range(total_activities):
        app.current_activity_idx = idx
        # Unique test pixel intensity per activity to verify content uniqueness
        val1 = (idx * 2) % 255
        val2 = (idx * 2 + 1) % 255
        frame1 = np.full((120, 160, 3), val1, dtype=np.uint8)
        frame2 = np.full((120, 160, 3), val2, dtype=np.uint8)

        ok1 = app.capture_current_photo(frame1)
        ok2 = app.capture_current_photo(frame2)
        assert ok1 is True, f"Failed capturing photo 1 for activity index {idx}"
        assert ok2 is True, f"Failed capturing photo 2 for activity index {idx}"

    session_dir = app.finalize_session(aborted=False)
    assert session_dir.is_dir()

    # Verify 1 manifest, 1 checksum file, 1 README
    manifest_file = session_dir / "session_manifest.json"
    checksum_file = session_dir / "checksum.sha256"
    readme_file = session_dir / "README.txt"

    assert manifest_file.is_file()
    assert checksum_file.is_file()
    assert readme_file.is_file()

    # Parse and validate manifest
    with open(manifest_file, encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert manifest_data["participant_id"] == "PTEST"
    assert manifest_data["session_id"] == "STEST"
    assert manifest_data["session_status"] == "completed"
    assert manifest_data["summary"]["total_activities"] == 45
    assert manifest_data["summary"]["completed_activities"] == 45
    assert manifest_data["summary"]["total_photos_captured"] == 90
    assert len(manifest_data["activities"]) == 45

    # Verify exactly 90 JPEG files exist in images/
    image_files = sorted((session_dir / "images").rglob("*.jpg"))
    assert len(image_files) == 90

    # Verify checksum.sha256 contains 91 entries (90 images + 1 manifest)
    with open(checksum_file, encoding="utf-8") as f:
        checksum_lines = [ln.strip() for ln in f if ln.strip()]
    assert len(checksum_lines) == 91

    # Simulate handoff: copy session folder to an external "remote/handoff" directory
    handoff_dest = tmp_path / "External_Admin_Storage" / "PTEST" / "STEST"
    shutil.copytree(session_dir, handoff_dest)

    # In handoff destination, verify every checksum matches exactly
    dest_checksum = handoff_dest / "checksum.sha256"
    assert dest_checksum.is_file()

    with open(dest_checksum, encoding="utf-8") as f:
        for line in f:
            expected_hash, rel_path = line.strip().split("  ", 1)
            target = handoff_dest / rel_path
            assert target.is_file(), f"Target missing in handoff: {rel_path}"
            computed_hash = calculate_sha256(target)
            assert computed_hash == expected_hash, f"Checksum mismatch in handoff: {rel_path}"

            # Verify every image decodes with OpenCV
            if rel_path.endswith(".jpg"):
                img = cv2.imread(str(target))
                assert img is not None and img.shape == (120, 160, 3)


def test_camera_switch_lock_mid_activity(tmp_path: Path):
    """Verifies that camera switching is safely blocked while an activity has 1 photo captured."""
    app = DatasetCollectorApp(
        participant_id="P_CAMLOCK",
        session_id="S_CAMLOCK",
        output_dir=tmp_path / "camlock_output",
        dry_run=True,
    )
    app.init_session_records()
    app.available_cameras = [
        DiscoveredCamera(0, "MOCK1", cv2.CAP_ANY, 640, 480, 30.0, True, "Cam 0"),
        DiscoveredCamera(1, "MOCK2", cv2.CAP_ANY, 640, 480, 30.0, True, "Cam 1"),
    ]
    app.selected_camera_idx = 0

    # Capture Photo 1
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    ok = app.capture_current_photo(frame)
    assert ok is True
    assert len(app.activity_records[app.activities[0].id].photos) == 1

    # Attempt to switch camera mid-activity
    app.switch_to_next_camera()
    # Camera index must remain 0 and message must indicate lock
    assert app.selected_camera_idx == 0
    assert "Camera locked" in app.status_message


def test_session_abort_and_incomplete_status(tmp_path: Path):
    """Verifies that exiting early sets session_status='incomplete' or 'aborted' without claiming completion."""
    app = DatasetCollectorApp(
        participant_id="P_ABORT",
        session_id="S_ABORT",
        output_dir=tmp_path / "abort_test",
        dry_run=True,
    )
    app.init_session_records()
    frame = np.full((100, 100, 3), 75, dtype=np.uint8)
    app.capture_current_photo(frame)
    app.capture_current_photo(frame)

    # Partial session finalized normally: should be incomplete
    session_dir = app.finalize_session(aborted=False)
    manifest_path = session_dir / "session_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["session_status"] == "incomplete"
    assert data["summary"]["completed_activities"] == 1
    assert data["summary"]["total_photos_captured"] == 2
    assert (session_dir / "images" / "ACT001" / "photo_01.jpg").is_file()
    assert (session_dir / "images" / "ACT001" / "photo_02.jpg").is_file()

    # Session finalized with aborted=True flag: should be aborted
    session_dir_aborted = app.finalize_session(aborted=True)
    with open(manifest_path, encoding="utf-8") as f:
        data_aborted = json.load(f)

    assert data_aborted["session_status"] == "aborted"
    assert data_aborted["summary"]["completed_activities"] == 1



