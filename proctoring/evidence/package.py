"""Session evidence package creation, manifest generation, checksum integrity, and zip archiving."""

import hashlib
import json
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from proctoring.core.errors import DiagnosticRecord
from proctoring.core.events import EventRecord
from proctoring.core.paths import sanitise_identifier
from proctoring.telemetry.performance import PerformanceReport

LOGGER = logging.getLogger(__name__)


@dataclass
class PackageManifest:
    """Standardized manifest summarizing examination evidence package assets and integrity."""

    schema_version: str
    package_id: str
    session_id: str
    student_name: str
    created_at_utc: str
    processing_config: dict[str, Any]
    models: dict[str, Any]
    event_summary: dict[str, Any]
    evidence_summary: dict[str, Any]
    telemetry_summary: dict[str, Any]
    diagnostics_summary: dict[str, Any]
    timeline_summary: dict[str, Any] = field(default_factory=dict)
    integrity_checksums: dict[str, str] = field(default_factory=dict)
    manifest_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "package_id": self.package_id,
            "session_id": self.session_id,
            "student_name": self.student_name,
            "created_at_utc": self.created_at_utc,
            "processing_config": self.processing_config,
            "models": self.models,
            "event_summary": self.event_summary,
            "evidence_summary": self.evidence_summary,
            "telemetry_summary": self.telemetry_summary,
            "diagnostics_summary": self.diagnostics_summary,
            "timeline_summary": self.timeline_summary,
            "integrity_checksums": self.integrity_checksums,
            "manifest_sha256": self.manifest_sha256,
        }


class SessionEvidencePackage:
    """Compiles, structures, and validates complete examination session evidence packages."""

    SCHEMA_VERSION = "1.0.0"

    def __init__(
        self,
        base_dir: str | Path = "data/evidence_packages",
        session_id: str = "default_session",
        student_name: str = "Candidate",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.session_id = sanitise_identifier(session_id, fallback="session")
        self.student_name = str(student_name)
        self.package_dir = self.base_dir / self.session_id
        self.package_id = (
            f"pkg_{self.session_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        # Populated by build_package() when create_zip is requested.  Callers must
        # read the archive location from here rather than deriving it from
        # package_dir: the archive is named after package_id, not the session.
        self.zip_path: Path | None = None

    def _compute_sha256(self, file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def build_package(
        self,
        events: list[EventRecord],
        telemetry_report: PerformanceReport,
        diagnostics: list[DiagnosticRecord],
        processing_config: dict[str, Any],
        models_info: dict[str, Any],
        create_zip: bool = False,
        timeline_entries: list[dict[str, Any]] | None = None,
        timeline_summary: dict[str, Any] | None = None,
    ) -> Path:
        """Write every package artefact, then seal the package with a signed manifest.

        Ordering matters and is load-bearing: all content files (events, telemetry,
        diagnostics, timeline and every evidence image) must be on disk *before*
        the checksum sweep runs, otherwise they are absent from
        ``integrity_checksums`` and silently fall outside the tamper-evident
        boundary.  The manifest is written last, and the archive last of all.
        """
        self.package_dir.mkdir(parents=True, exist_ok=True)

        # 1. Write events.json
        events_json_path = self.package_dir / "events.json"
        with open(events_json_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in events], f, indent=2)

        # 2. Write telemetry.json
        telemetry_json_path = self.package_dir / "telemetry.json"
        with open(telemetry_json_path, "w", encoding="utf-8") as f:
            json.dump(telemetry_report.to_dict(), f, indent=2)

        # 3. Write diagnostics.json
        diagnostics_json_path = self.package_dir / "diagnostics.json"
        with open(diagnostics_json_path, "w", encoding="utf-8") as f:
            json.dump([d.to_dict() for d in diagnostics], f, indent=2)

        # 4. Write timeline.json — the per-frame observation record.  Written here,
        #    before the checksum sweep below, so it is covered by the manifest.
        if timeline_entries is not None:
            timeline_json_path = self.package_dir / "timeline.json"
            with open(timeline_json_path, "w", encoding="utf-8") as f:
                json.dump(timeline_entries, f, indent=2)

        # 5. Summarize events
        event_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {}
        for e in events:
            event_counts[e.event_type.value] = event_counts.get(e.event_type.value, 0) + 1
            severity_counts[e.severity.value] = severity_counts.get(e.severity.value, 0) + 1

        event_summary = {
            "total_events": len(events),
            "qualified_events": sum(
                1 for e in events if e.metadata.get("is_duration_qualified", True)
            ),
            "counts_by_type": event_counts,
            "counts_by_severity": severity_counts,
        }

        # 6. Summarize evidence assets & calculate file checksums
        checksums: dict[str, str] = {}
        total_evidence_files = 0
        total_evidence_bytes = 0

        for file_p in sorted(self.package_dir.rglob("*")):
            if file_p.is_file() and file_p.name != "manifest.json":
                rel_path = file_p.relative_to(self.package_dir).as_posix()
                sha = self._compute_sha256(file_p)
                checksums[rel_path] = sha
                if "evidence/" in rel_path:
                    total_evidence_files += 1
                    total_evidence_bytes += file_p.stat().st_size

        evidence_summary = {
            "total_files": total_evidence_files,
            "total_bytes": total_evidence_bytes,
            "frames_dir": "evidence/frames",
            "crops_dir": "evidence/crops",
        }

        telemetry_summary = {
            "total_frames": telemetry_report.total_frames,
            "processed_frames": telemetry_report.processed_frames,
            "skipped_frames": telemetry_report.skipped_frames,
            "effective_fps": round(telemetry_report.effective_fps, 2),
            "latency_median_p50_ms": round(telemetry_report.latency_overall.median_p50_ms, 2),
            "latency_p95_ms": round(telemetry_report.latency_overall.p95_ms, 2),
            "latency_max_ms": round(telemetry_report.latency_overall.max_ms, 2),
        }

        diagnostics_summary = {
            "total_system_errors": len(diagnostics),
            "fatal_errors": sum(1 for d in diagnostics if d.is_fatal),
        }

        manifest = PackageManifest(
            schema_version=self.SCHEMA_VERSION,
            package_id=self.package_id,
            session_id=self.session_id,
            student_name=self.student_name,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            processing_config=processing_config,
            models=models_info,
            event_summary=event_summary,
            evidence_summary=evidence_summary,
            telemetry_summary=telemetry_summary,
            diagnostics_summary=diagnostics_summary,
            timeline_summary=timeline_summary or {},
            integrity_checksums=checksums,
        )

        manifest_path = self.package_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        # Calculate final manifest SHA256 and update
        manifest.manifest_sha256 = self._compute_sha256(manifest_path)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        # Optional zip packaging
        if create_zip:
            zip_out = self.base_dir / f"{self.package_id}.zip"
            with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
                for f_path in sorted(self.package_dir.rglob("*")):
                    if f_path.is_file():
                        zf.write(f_path, f_path.relative_to(self.package_dir))
            self.zip_path = zip_out

        return self.package_dir

    def verify_package_integrity(self) -> tuple[bool, list[str]]:
        """Validate all files in package directory against manifest checksums."""
        manifest_path = self.package_dir / "manifest.json"
        if not manifest_path.exists():
            return False, ["Manifest file manifest.json does not exist"]

        with open(manifest_path, encoding="utf-8") as f:
            manifest_data = json.load(f)

        errors: list[str] = []
        checksums = manifest_data.get("integrity_checksums", {})

        for rel_path, expected_sha in checksums.items():
            file_path = self.package_dir / rel_path
            if not file_path.exists():
                errors.append(f"Missing file: {rel_path}")
                continue
            calc_sha = self._compute_sha256(file_path)
            if calc_sha != expected_sha:
                errors.append(
                    f"Checksum mismatch for {rel_path} (expected {expected_sha[:8]}, got {calc_sha[:8]})"
                )

        return len(errors) == 0, errors
