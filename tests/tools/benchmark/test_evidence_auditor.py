"""Tests for evidence package auditor and traceability validation."""

import json
import tempfile
from pathlib import Path

from tools.benchmark.evidence_auditor import EvidencePackageAuditor


def test_evidence_package_auditor_on_valid_and_corrupt_packages():
    """Verify auditor passes valid packages and detects missing or corrupted files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "test_pkg"
        pkg_path.mkdir()

        # Create valid package structure
        (pkg_path / "events.json").write_text("[]")
        (pkg_path / "telemetry.json").write_text("{}")
        (pkg_path / "diagnostics.json").write_text("[]")

        import hashlib

        events_sha = hashlib.sha256((pkg_path / "events.json").read_bytes()).hexdigest()
        manifest = {
            "schema_version": "1.0.0",
            "package_id": "pkg_01",
            "integrity_checksums": {
                "events.json": events_sha,
            },
        }
        (pkg_path / "manifest.json").write_text(json.dumps(manifest))

        report = EvidencePackageAuditor.audit_package(pkg_path)
        assert report.manifest_present is True
        assert report.events_present is True
        assert report.manifest_integrity_verified is True
        assert report.audit_passed is True

        # Tamper with file
        (pkg_path / "events.json").write_text("tampered")
        tampered_report = EvidencePackageAuditor.audit_package(pkg_path)
        assert tampered_report.manifest_integrity_verified is False
        assert tampered_report.audit_passed is False
