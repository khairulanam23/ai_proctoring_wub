"""Tests for EvidenceManager, ROI cropping, and evidence validation."""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from proctoring.core.events import (
    DetectorInfo,
    EventRecord,
    EventSeverity,
    EventStatus,
    EventType,
    ObservationDetail,
)
from proctoring.evidence.manager import EvidenceManager


def test_evidence_capture_and_cropping():
    """Verify full frame and ROI crop saving with valid checksums."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = EvidenceManager(base_dir=tmpdir, session_id="sess_ev_test")

        # Create synthetic test frame with a distinct colored box
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (200, 250), (0, 0, 255), -1)

        # 1. Capture frame
        ev_frame = mgr.capture_frame(frame, frame_index=1, timestamp_seconds=0.25)
        assert ev_frame is not None
        assert ev_frame.media_type == "frame"
        assert ev_frame.sha256 is not None
        assert Path(tmpdir, "sess_ev_test", ev_frame.file_path).exists()

        # 2. Crop ROI
        ev_crop = mgr.crop_region_of_interest(
            frame=frame,
            bbox=(100, 100, 200, 250),
            event_id="evt_01",
            frame_index=1,
            timestamp_seconds=0.25,
            label="cell_phone",
        )
        assert ev_crop is not None
        assert ev_crop.media_type == "crop"
        assert Path(tmpdir, "sess_ev_test", ev_crop.file_path).exists()

        # 3. Validate
        val_res_frame = mgr.validate_evidence(ev_frame)
        assert val_res_frame.is_valid is True
        assert val_res_frame.checksum_match is True
        assert val_res_frame.can_open is True

        val_res_crop = mgr.validate_evidence(ev_crop)
        assert val_res_crop.is_valid is True
        assert val_res_crop.can_open is True


def test_evidence_validation_failure_modes():
    """Verify validator catches missing, corrupted, or 0-byte files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = EvidenceManager(base_dir=tmpdir, session_id="sess_fail_test")

        # Create dummy event
        ev_rec = EventRecord(
            event_id="evt_001",
            session_id="sess_fail_test",
            timestamp=1.0,
            end_timestamp=2.0,
            duration=1.0,
            formatted_start="00:00:01.000",
            formatted_end="00:00:02.000",
            event_type=EventType.PHONE_DETECTED,
            severity=EventSeverity.HIGH,
            confidence=0.9,
            average_confidence=0.9,
            detector=DetectorInfo(name="YOLO"),
            observation=ObservationDetail(description="phone"),
        )

        # Attach with empty frame -> should fail gracefully
        res = mgr.attach_evidence_to_event(ev_rec, frame=None, frame_index=1, timestamp_seconds=1.0)
        assert res is False
        assert ev_rec.status == EventStatus.EVIDENCE_FAILED
        assert ev_rec.metadata.get("evidence_failed") is True
