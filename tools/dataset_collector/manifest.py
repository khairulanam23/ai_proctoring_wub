"""Session manifest creation, checksum generation, and directory management for still-image datasets."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("dataset_collector.manifest")

SCHEMA_VERSION = "2.0.0-still"
APPLICATION_NAME = "AI Proctoring Dataset Collector"
APPLICATION_VERSION = "1.0.0"
MIN_FREE_DISK_BYTES = 500 * 1024 * 1024  # 500 MB safety guard


def sanitize_identifier(value: str, name: str = "Identifier") -> str:
    """Validates and cleans participant/session identifiers."""
    cleaned = value.strip()
    if not re.match(r"^[A-Za-z0-9_-]+$", cleaned):
        raise ValueError(
            f"{name} '{value}' contains invalid characters. Use alphanumeric characters, dashes, or underscores only."
        )
    return cleaned


def calculate_sha256(filepath: Path) -> str:
    """Computes SHA-256 digest of a local file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_disk_space(target_dir: Path, min_bytes: int = MIN_FREE_DISK_BYTES) -> tuple[bool, int, int]:
    """Ensures destination directory exists and has sufficient free disk space."""
    target_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target_dir)
    return (usage.free >= min_bytes), usage.free, usage.total


@dataclass
class PhotoRecord:
    """Metadata for an individual captured still photograph."""

    photo_number: int
    filename: str
    relative_path: str
    width: int
    height: int
    file_size_bytes: int
    sha256: str
    captured_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActivityRecord:
    """Record of a collection activity within a session."""

    activity_id: str
    activity_name: str
    category: str
    instructed_condition: str
    purpose: str
    status: str = "incomplete"  # "completed", "incomplete", "skipped"
    photos: list[PhotoRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "activity_id": self.activity_id,
            "activity_name": self.activity_name,
            "category": self.category,
            "instructed_condition": self.instructed_condition,
            "purpose": self.purpose,
            "status": self.status,
            "photos": [p.to_dict() for p in self.photos],
        }


@dataclass
class SessionManifest:
    """Authoritative top-level session manifest."""

    participant_id: str
    session_id: str
    camera: dict[str, Any]
    activities: list[ActivityRecord]
    session_status: str = "in_progress"  # "completed", "incomplete", "aborted"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str = ""
    schema_version: str = SCHEMA_VERSION
    application_name: str = APPLICATION_NAME
    application_version: str = APPLICATION_VERSION
    platform_system: str = field(default_factory=lambda: platform.system())
    platform_release: str = field(default_factory=lambda: platform.release())

    def to_dict(self) -> dict[str, Any]:
        total_photos = sum(len(a.photos) for a in self.activities)
        completed_activities = sum(1 for a in self.activities if a.status == "completed")

        return {
            "schema_version": self.schema_version,
            "application_name": self.application_name,
            "application_version": self.application_version,
            "platform": {
                "system": self.platform_system,
                "release": self.platform_release,
            },
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "session_status": self.session_status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "camera": self.camera,
            "summary": {
                "total_activities": len(self.activities),
                "completed_activities": completed_activities,
                "total_photos_captured": total_photos,
            },
            "activities": [a.to_dict() for a in self.activities],
        }


def write_manifest_and_checksums(session_dir: Path, manifest: SessionManifest) -> tuple[Path, Path]:
    """Writes session_manifest.json and detached checksum.sha256 in the session directory."""
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = session_dir / "session_manifest.json"
    checksum_path = session_dir / "checksum.sha256"

    # Set completion timestamp if empty
    if not manifest.completed_at:
        manifest.completed_at = datetime.now(timezone.utc).isoformat()

    manifest_dict = manifest.to_dict()
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)

    # Compute checksums for all image files and the manifest itself
    checksum_lines: list[str] = []
    images_dir = session_dir / "images"
    if images_dir.exists():
        for img_path in sorted(images_dir.rglob("*.jpg")):
            rel_path = img_path.relative_to(session_dir).as_posix()
            digest = calculate_sha256(img_path)
            checksum_lines.append(f"{digest}  {rel_path}\n")

    manifest_digest = calculate_sha256(manifest_path)
    checksum_lines.append(f"{manifest_digest}  session_manifest.json\n")

    with open(checksum_path, "w", encoding="utf-8") as f:
        f.writelines(checksum_lines)

    LOGGER.info(f"Wrote manifest ({manifest_path.name}) and {len(checksum_lines)} checksums.")
    return manifest_path, checksum_path


def generate_session_readme(session_dir: Path, participant_id: str, session_id: str) -> Path:
    """Generates an end-user / recipient README.txt inside the delivered dataset folder."""
    readme_path = session_dir / "README.txt"
    content = f"""AI Proctoring Engine — Research Still-Image Dataset Package
================================================================================
Participant ID : {participant_id}
Session ID     : {session_id}
Generated By   : {APPLICATION_NAME} v{APPLICATION_VERSION}
Platform       : {platform.system()} {platform.release()}
Date (UTC)     : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
================================================================================

DELIVERABLE CONTENTS:
---------------------
1. images/
   Subdirectories per activity (e.g. images/ACT001/photo_01.jpg, photo_02.jpg).
   Contains raw, uncompressed/high-quality captured photographs.

2. session_manifest.json
   Authoritative machine-readable metadata including exact activity specifications,
   instructed conditions, camera technical properties, and individual file hashes.

3. checksum.sha256
   Detached cryptographic SHA-256 integrity verification file for all images.
   To verify integrity on Linux/macOS:
     sha256sum -c checksum.sha256
   To verify on Windows PowerShell:
     Get-FileHash -Algorithm SHA256 images\\*\\*.jpg

IMPORTANT NOTES:
----------------
- DO NOT rename, modify, or delete any image files.
- DO NOT edit session_manifest.json or checksum.sha256.
- 'instructed_condition' specifies the condition the participant was guided to perform.
  It is NOT verified ground truth; downstream human verification is required.

To hand off this dataset:
Zip or copy this entire session folder and send it to the dataset administrator.
================================================================================
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    return readme_path
