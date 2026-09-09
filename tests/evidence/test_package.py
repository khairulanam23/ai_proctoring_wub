"""Tests for session evidence package generation and manifest verification."""

import tempfile
from pathlib import Path

from proctoring.core.events import (
    DetectorInfo,
    EventRecord,
    EventSeverity,
    EventType,
    ObservationDetail,
)
from proctoring.evidence.package import SessionEvidencePackage
from proctoring.telemetry.performance import FrameTimingRecord, PipelineTelemetryTracker


def test_package_creation_and_integrity_verification():
    """Verify package creates manifest.json, events.json, telemetry.json, and passes checksum verification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        packager = SessionEvidencePackage(
            base_dir=tmpdir,
            session_id="pkg_test_sess",
            student_name="Alice Smith",
        )

        detector = DetectorInfo(name="YuNet")
        obs = ObservationDetail(description="No face detected for 2.0s")
        event = EventRecord(
            event_id="evt_01",
            session_id="pkg_test_sess",
            timestamp=1.0,
            end_timestamp=3.0,
            duration=2.0,
            formatted_start="00:00:01.000",
            formatted_end="00:00:03.000",
            event_type=EventType.NO_FACE,
            severity=EventSeverity.HIGH,
            confidence=1.0,
            average_confidence=1.0,
            detector=detector,
            observation=obs,
        )

        tracker = PipelineTelemetryTracker(session_id="pkg_test_sess")
        tracker.record_frame(
            FrameTimingRecord(frame_index=1, timestamp_seconds=0.25, total_frame_ms=15.0)
        )
        telemetry_rep = tracker.generate_report()

        pkg_path = packager.build_package(
            events=[event],
            telemetry_report=telemetry_rep,
            diagnostics=[],
            processing_config={"sampling_fps": 4.0},
            models_info={"face_detector": "yunet"},
            create_zip=True,
        )

        assert (pkg_path / "manifest.json").exists()
        assert (pkg_path / "events.json").exists()
        assert (pkg_path / "telemetry.json").exists()
        assert (pkg_path / "diagnostics.json").exists()
        assert Path(tmpdir, f"{packager.package_id}.zip").exists()

        # Check integrity
        is_ok, errors = packager.verify_package_integrity()
        assert is_ok is True
        assert len(errors) == 0

        # Tamper with manifest.json directly and verify it catches manifest alteration
        manifest_file = pkg_path / "manifest.json"
        manifest_data = manifest_file.read_text(encoding="utf-8")
        manifest_file.write_text(manifest_data.replace("Alice Smith", "Mallory Impostor"))
        is_ok_manifest_tampered, manifest_tampered_errors = packager.verify_package_integrity()
        assert is_ok_manifest_tampered is False
        assert any("manifest.json" in err for err in manifest_tampered_errors)

        # Restore manifest.json
        manifest_file.write_text(manifest_data)
        is_ok_restored, errors_restored = packager.verify_package_integrity()
        assert is_ok_restored is True
        assert len(errors_restored) == 0

        # Tamper with an evidence/event file and verify it catches corruption
        events_file = pkg_path / "events.json"
        events_file.write_text("tampered content")
        is_ok_tampered, tampered_errors = packager.verify_package_integrity()
        assert is_ok_tampered is False
        assert len(tampered_errors) > 0
