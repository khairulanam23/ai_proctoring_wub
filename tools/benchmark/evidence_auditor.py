"""Systematic evidence pipeline auditor verifying metadata completeness, image validity, and cryptographic integrity."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2


@dataclass
class EvidenceAuditReport:
    """Outcome of auditing an entire examination evidence package."""

    package_dir: str
    manifest_present: bool
    events_present: bool
    telemetry_present: bool
    diagnostics_present: bool
    total_events: int
    total_evidence_references: int
    valid_evidence_files: int
    corrupted_or_missing_files: int
    manifest_integrity_verified: bool
    audit_passed: bool
    reconstruction_traceability_passed: bool
    audit_errors: list[str] = field(default_factory=list)
    reconstruction_sample: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_dir": self.package_dir,
            "manifest_present": self.manifest_present,
            "events_present": self.events_present,
            "telemetry_present": self.telemetry_present,
            "diagnostics_present": self.diagnostics_present,
            "total_events": self.total_events,
            "total_evidence_references": self.total_evidence_references,
            "valid_evidence_files": self.valid_evidence_files,
            "corrupted_or_missing_files": self.corrupted_or_missing_files,
            "manifest_integrity_verified": self.manifest_integrity_verified,
            "audit_passed": self.audit_passed,
            "reconstruction_traceability_passed": self.reconstruction_traceability_passed,
            "audit_errors": self.audit_errors,
            "reconstruction_sample": self.reconstruction_sample,
        }


class EvidencePackageAuditor:
    """Audits examination evidence packages for completeness, integrity, and investigator traceability."""

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def audit_package(cls, package_path: str | Path) -> EvidenceAuditReport:
        """Perform comprehensive audit on a generated evidence package."""
        pkg_dir = Path(package_path)
        errors: list[str] = []

        if not pkg_dir.exists() or not pkg_dir.is_dir():
            return EvidenceAuditReport(
                package_dir=str(pkg_dir),
                manifest_present=False,
                events_present=False,
                telemetry_present=False,
                diagnostics_present=False,
                total_events=0,
                total_evidence_references=0,
                valid_evidence_files=0,
                corrupted_or_missing_files=0,
                manifest_integrity_verified=False,
                audit_passed=False,
                reconstruction_traceability_passed=False,
                audit_errors=[f"Package directory does not exist: {pkg_dir}"],
            )

        manifest_file = pkg_dir / "manifest.json"
        events_file = pkg_dir / "events.json"
        telemetry_file = pkg_dir / "telemetry.json"
        diagnostics_file = pkg_dir / "diagnostics.json"

        has_manifest = manifest_file.exists()
        has_events = events_file.exists()
        has_telemetry = telemetry_file.exists()
        has_diagnostics = diagnostics_file.exists()

        if not has_manifest:
            errors.append("Missing manifest.json")
        if not has_events:
            errors.append("Missing events.json")
        if not has_telemetry:
            errors.append("Missing telemetry.json")
        if not has_diagnostics:
            errors.append("Missing diagnostics.json")

        # Read manifest & verify SHA256 checksums
        manifest_integrity = False
        if has_manifest:
            try:
                with open(manifest_file, encoding="utf-8") as f:
                    manifest_data = json.load(f)
                checksums = manifest_data.get("integrity_checksums", {})
                checksum_failures = 0
                for rel_path, exp_sha in checksums.items():
                    f_path = pkg_dir / rel_path
                    if not f_path.exists():
                        errors.append(f"Referenced file missing: {rel_path}")
                        checksum_failures += 1
                        continue
                    calc_sha = cls._compute_sha256(f_path)
                    if calc_sha != exp_sha:
                        errors.append(f"Checksum mismatch for {rel_path}")
                        checksum_failures += 1
                manifest_integrity = checksum_failures == 0
            except Exception as e:
                errors.append(f"Error parsing manifest.json: {e}")

        # Read events and verify evidence image decodability
        total_events = 0
        total_ev_refs = 0
        valid_ev_files = 0
        corrupt_or_missing = 0
        reconstruction_sample = None
        reconstruction_ok = True

        if has_events:
            try:
                with open(events_file, encoding="utf-8") as f:
                    events_list = json.load(f)
                total_events = len(events_list)

                for ev in events_list:
                    # Test investigator reconstruction questions
                    req_keys = [
                        "event_id",
                        "timestamp",
                        "duration",
                        "event_type",
                        "detector",
                        "observation",
                        "evidence",
                    ]
                    if not all(k in ev for k in req_keys):
                        reconstruction_ok = False
                        errors.append(
                            f"Event missing required traceability fields: {ev.get('event_id')}"
                        )

                    ev_refs = ev.get("evidence", [])
                    total_ev_refs += len(ev_refs)

                    for r in ev_refs:
                        rel_file = r.get("file_path", "")
                        abs_file = pkg_dir / rel_file
                        if not abs_file.exists() or abs_file.stat().st_size == 0:
                            corrupt_or_missing += 1
                            errors.append(f"Evidence file missing or 0-bytes: {rel_file}")
                            continue

                        # Image decoding check
                        img = cv2.imread(str(abs_file))
                        if img is None or img.size == 0:
                            corrupt_or_missing += 1
                            errors.append(f"Evidence image corrupted / cannot decode: {rel_file}")
                        else:
                            valid_ev_files += 1

                # Sample reconstruction entry for human proctor report
                if events_list:
                    sample_ev = events_list[0]
                    reconstruction_sample = {
                        "WHAT": sample_ev.get("event_type"),
                        "WHEN": sample_ev.get("formatted_start"),
                        "HOW_LONG": f"{sample_ev.get('duration', 0):.2f} seconds",
                        "WHY": sample_ev.get("observation", {}).get("description"),
                        "DETECTOR": sample_ev.get("detector", {}).get("name"),
                        "EVIDENCE_FILES": [
                            r.get("file_path") for r in sample_ev.get("evidence", [])
                        ],
                    }

            except Exception as e:
                errors.append(f"Error reading events.json: {e}")

        audit_passed = (
            has_manifest
            and has_events
            and has_telemetry
            and has_diagnostics
            and manifest_integrity
            and (corrupt_or_missing == 0)
            and len(errors) == 0
        )

        return EvidenceAuditReport(
            package_dir=str(pkg_dir),
            manifest_present=has_manifest,
            events_present=has_events,
            telemetry_present=has_telemetry,
            diagnostics_present=has_diagnostics,
            total_events=total_events,
            total_evidence_references=total_ev_refs,
            valid_evidence_files=valid_ev_files,
            corrupted_or_missing_files=corrupt_or_missing,
            manifest_integrity_verified=manifest_integrity,
            audit_passed=audit_passed,
            reconstruction_traceability_passed=reconstruction_ok,
            audit_errors=errors,
            reconstruction_sample=reconstruction_sample,
        )
