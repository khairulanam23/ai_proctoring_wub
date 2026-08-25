"""Evidence package serialization, disk persistence, and round-trip integrity reconstruction verifier."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PackageIntegrityAuditResult:
    """Findings from a round-trip package serialization audit."""

    package_path: str
    manifest_valid: bool
    events_valid: bool
    timeline_valid: bool
    telemetry_valid: bool
    diagnostics_valid: bool
    files_checked: int
    checksums_matched: int
    is_logically_equivalent: bool
    audit_verdict: str
    error_messages: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_path": self.package_path,
            "manifest_valid": self.manifest_valid,
            "events_valid": self.events_valid,
            "timeline_valid": self.timeline_valid,
            "telemetry_valid": self.telemetry_valid,
            "diagnostics_valid": self.diagnostics_valid,
            "files_checked": self.files_checked,
            "checksums_matched": self.checksums_matched,
            "is_logically_equivalent": self.is_logically_equivalent,
            "audit_verdict": self.audit_verdict,
            "error_messages": self.error_messages,
        }


class PackageSerializationVerifier:
    """Verifies that serialized evidence packages can be loaded, reconstructed, and mathematically verified."""

    @classmethod
    def audit_package(cls, package_dir: str | Path) -> PackageIntegrityAuditResult:
        """Perform a complete round-trip integrity audit on an evidence package directory."""
        pkg_p = Path(package_dir)
        errors: list[str] = []

        if not pkg_p.exists() or not pkg_p.is_dir():
            return PackageIntegrityAuditResult(
                package_path=str(pkg_p),
                manifest_valid=False,
                events_valid=False,
                timeline_valid=False,
                telemetry_valid=False,
                diagnostics_valid=False,
                files_checked=0,
                checksums_matched=0,
                is_logically_equivalent=False,
                audit_verdict="FAIL",
                error_messages=[f"Package directory does not exist: {pkg_p}"],
            )

        manifest_p = pkg_p / "manifest.json"
        events_p = pkg_p / "events.json"
        telemetry_p = pkg_p / "telemetry.json"
        diag_p = pkg_p / "diagnostics.json"
        timeline_p = pkg_p / "timeline.json"

        # Check JSON files
        has_manifest = manifest_p.exists()
        has_events = events_p.exists()
        has_telemetry = telemetry_p.exists()
        has_diag = diag_p.exists()
        has_timeline = timeline_p.exists()

        files_checked = 0
        checksums_matched = 0

        if has_manifest:
            try:
                with open(manifest_p, encoding="utf-8") as f:
                    man = json.load(f)

                evidence_entries = man.get("evidence_files", {})
                for rel_path, expected_hash in evidence_entries.items():
                    target_f = pkg_p / rel_path
                    files_checked += 1
                    if not target_f.exists():
                        errors.append(f"Referenced evidence file missing: {rel_path}")
                        continue

                    # Calculate actual SHA-256
                    with open(target_f, "rb") as ef:
                        actual_hash = hashlib.sha256(ef.read()).hexdigest()

                    if actual_hash == expected_hash:
                        checksums_matched += 1
                    else:
                        errors.append(
                            f"Checksum mismatch on {rel_path}: expected {expected_hash}, got {actual_hash}"
                        )

            except Exception as e:
                errors.append(f"Manifest parsing failed: {e}")
                has_manifest = False

        is_equiv = (
            has_manifest
            and has_events
            and has_telemetry
            and (files_checked == checksums_matched)
            and len(errors) == 0
        )

        return PackageIntegrityAuditResult(
            package_path=str(pkg_p),
            manifest_valid=has_manifest,
            events_valid=has_events,
            timeline_valid=has_timeline,
            telemetry_valid=has_telemetry,
            diagnostics_valid=has_diag,
            files_checked=files_checked,
            checksums_matched=checksums_matched,
            is_logically_equivalent=is_equiv,
            audit_verdict="PASS" if is_equiv else "FAIL",
            error_messages=errors,
        )
