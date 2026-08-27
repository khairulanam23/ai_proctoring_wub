"""Where the engine keeps what it must keep, and what it must throw away.

Two kinds of image live on disk, and conflating them is how a proctoring system
loses a candidate's trust:

``PERMANENT``
    A student's enrolment reference photographs, and the sealed evidence of a
    finished examination. Both are records. Neither is ever removed by cleanup.

``TEMPORARY``
    Frames captured while an enrolment is still in progress, previews, and any
    other disposable intermediate. These are staging artefacts, and an enrolment
    that succeeds, fails or is abandoned must leave none of them behind.

:class:`ProctoringStorage` owns that distinction. It is the single place that
knows the on-disk layout, so paths are not reinvented at each call site and
cleanup can be reasoned about in one file rather than audited across the tree::

    data/
    ├── students/
    │   └── <student>/
    │       └── enrollment/
    │           ├── image_01.jpg … image_05.jpg   permanent references
    │           ├── templates.npz                 embeddings derived from them
    │           ├── enrollment.json               manifest + SHA-256 per image
    │           └── .staging/                     temporary; never survives a call
    └── sessions/
        └── <student>_<YYYY-MM-DD_HH-MM-SS-mmm>/  one examination, never reused

Enrolment embeddings sit beside the images they were derived from rather than in
a separate store: one identity mechanism, one directory to delete when a student
exercises their right to erasure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from proctoring.core.paths import resolve_within, sanitise_identifier

LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path("data")
ENROLLMENT_IMAGE_COUNT = 5
_IMAGE_PATTERN = "image_{index:02d}.jpg"
_STAGING_DIRNAME = ".staging"
_MANIFEST_NAME = "enrollment.json"
_TEMPLATES_NAME = "templates.npz"
_SCHEMA_VERSION = "1.0"


class EnrollmentState(str, Enum):
    """Whether a stored enrolment can be trusted for identity verification."""

    MISSING = "MISSING"
    """No enrolment has been recorded for this student."""

    VALID = "VALID"
    """Manifest, images and templates are all present and internally consistent."""

    INCOMPLETE = "INCOMPLETE"
    """Something the manifest promises is absent — fewer images than recorded,
    or missing templates. Safe to re-enrol; unsafe to verify against."""

    CORRUPT = "CORRUPT"
    """Present but unreadable: an undecodable manifest, an unreadable image, or a
    checksum that no longer matches the file it describes."""


@dataclass
class EnrollmentRecord:
    """A student's stored enrolment and the verdict on its usability."""

    student: str
    state: EnrollmentState
    directory: Path
    image_paths: list[Path] = field(default_factory=list)
    templates: list[np.ndarray] = field(default_factory=list)
    created_at_utc: str = ""
    model: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """True only when the enrolment may be used to verify identity."""
        return self.state is EnrollmentState.VALID and bool(self.templates)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for reporting. Embeddings are biometric data and never included."""
        return {
            "student": self.student,
            "state": self.state.value,
            "directory": str(self.directory),
            "image_count": len(self.image_paths),
            "template_count": len(self.templates),
            "created_at_utc": self.created_at_utc,
            "model": self.model,
            "problems": self.problems,
        }


class ProctoringStorage:
    """Filesystem layout for enrolment references and examination sessions.

    Every path this class returns is resolved inside ``data_root``; a student name
    that tries to escape it is sanitised, and the resolved path is checked again
    before anything is written. Deletion is confined to directories this class
    constructed, so a bad identifier cannot direct a removal elsewhere.
    """

    def __init__(self, data_root: str | Path = DEFAULT_DATA_ROOT) -> None:
        self.data_root = Path(data_root)
        self.students_root = self.data_root / "students"
        self.sessions_root = self.data_root / "sessions"

    # ------------------------------------------------------------------
    # Path construction
    # ------------------------------------------------------------------

    @staticmethod
    def student_key(student: str) -> str:
        """Directory-safe key for a student name.

        Names arrive from a CLI flag or an LMS payload and become directory names,
        so they are reduced to a safe form before touching the filesystem.
        """
        return sanitise_identifier(student, fallback="student")

    def student_dir(self, student: str) -> Path:
        """Root directory for one student, guaranteed inside the data root."""
        return resolve_within(self.students_root, self.student_key(student))

    def enrollment_dir(self, student: str) -> Path:
        """Directory holding a student's permanent enrolment references."""
        return resolve_within(self.students_root, self.student_key(student), "enrollment")

    def staging_dir(self, student: str) -> Path:
        """Scratch directory for an enrolment still in progress.

        Kept inside the enrolment directory but dot-prefixed and always removed, so
        a partially captured enrolment can never be mistaken for a complete one.
        """
        return resolve_within(
            self.students_root, self.student_key(student), "enrollment", _STAGING_DIRNAME
        )

    def session_dir(self, student: str, started_at: datetime | None = None) -> Path:
        """Allocate a fresh, unique directory for one examination session.

        The name carries the student and the real session start time to millisecond
        precision. Should a directory somehow already exist — a clock adjustment, or
        two sessions opened in the same millisecond — a numeric suffix is appended
        rather than reusing it. A session must never write into another's evidence.
        """
        started_at = started_at or datetime.now()
        stamp = started_at.strftime("%Y-%m-%d_%H-%M-%S-") + f"{started_at.microsecond // 1000:03d}"
        base_name = f"{self.student_key(student)}_{stamp}"

        candidate = resolve_within(self.sessions_root, base_name)
        suffix = 1
        while candidate.exists():
            candidate = resolve_within(self.sessions_root, f"{base_name}_{suffix:02d}")
            suffix += 1

        candidate.mkdir(parents=True, exist_ok=False)
        LOGGER.info("Session directory allocated: %s", candidate)
        return candidate

    def session_id(self, student: str, started_at: datetime | None = None) -> str:
        """The directory name a session will use, without creating it."""
        started_at = started_at or datetime.now()
        stamp = started_at.strftime("%Y-%m-%d_%H-%M-%S-") + f"{started_at.microsecond // 1000:03d}"
        return f"{self.student_key(student)}_{stamp}"

    # ------------------------------------------------------------------
    # Enrolment: reading
    # ------------------------------------------------------------------

    def load_enrollment(self, student: str) -> EnrollmentRecord:
        """Load and validate a student's enrolment.

        Validation is deliberately strict. Verifying a candidate's identity against
        a half-written or altered reference set is worse than not verifying at all:
        it produces confident-looking mismatches. Anything short of intact is
        reported as such so the caller can re-enrol instead.
        """
        directory = self.enrollment_dir(student)
        manifest_path = directory / _MANIFEST_NAME

        if not manifest_path.exists():
            return EnrollmentRecord(
                student=student, state=EnrollmentState.MISSING, directory=directory
            )

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return EnrollmentRecord(
                student=student,
                state=EnrollmentState.CORRUPT,
                directory=directory,
                problems=[f"enrollment.json could not be read: {exc}"],
            )

        problems: list[str] = []
        image_paths: list[Path] = []

        for entry in manifest.get("images", []):
            path = directory / str(entry.get("file", ""))
            if not path.exists():
                problems.append(f"missing reference image: {entry.get('file')}")
                continue
            expected = entry.get("sha256")
            if expected and self._sha256(path) != expected:
                problems.append(f"reference image altered since enrolment: {entry.get('file')}")
                continue
            image_paths.append(path)

        templates = self._load_templates(directory, problems)

        recorded = int(manifest.get("image_count", len(manifest.get("images", []))))
        if len(image_paths) < recorded:
            problems.append(f"expected {recorded} reference image(s), found {len(image_paths)}")

        state = EnrollmentState.VALID
        if any("altered" in p or "could not" in p for p in problems):
            state = EnrollmentState.CORRUPT
        elif problems or not templates:
            state = EnrollmentState.INCOMPLETE

        return EnrollmentRecord(
            student=student,
            state=state,
            directory=directory,
            image_paths=image_paths,
            templates=templates,
            created_at_utc=manifest.get("created_at_utc", ""),
            model=manifest.get("model", {}),
            problems=problems,
        )

    def is_enrolled(self, student: str) -> bool:
        """True when a usable enrolment already exists for this student."""
        return self.load_enrollment(student).is_usable

    def load_templates(self, student: str) -> list[np.ndarray]:
        """Enrolment embeddings for identity verification, or empty if unusable."""
        record = self.load_enrollment(student)
        return record.templates if record.is_usable else []

    # ------------------------------------------------------------------
    # Enrolment: writing
    # ------------------------------------------------------------------

    def save_enrollment(
        self,
        student: str,
        images: list[np.ndarray] | None,
        templates: list[np.ndarray],
        model: dict[str, Any] | None = None,
    ) -> EnrollmentRecord:
        """Persist a student's reference images and the embeddings derived from them.

        Writes replace any previous enrolment wholesale rather than merging, so a
        re-enrolment cannot leave a mixture of old and new references behind. The
        staging directory is cleared on the way out whether or not the write
        succeeded.
        """
        if not images and not templates:
            raise ValueError("Cannot save an enrolment with neither images nor templates")

        import cv2

        images = images or []

        directory = self.enrollment_dir(student)
        self._clear_enrollment_dir(directory)
        directory.mkdir(parents=True, exist_ok=True)

        entries: list[dict[str, Any]] = []
        written: list[Path] = []
        try:
            for index, image in enumerate(images, start=1):
                filename = _IMAGE_PATTERN.format(index=index)
                path = directory / filename
                if not cv2.imwrite(str(path), image):
                    raise OSError(f"could not write reference image {filename}")
                written.append(path)
                entries.append(
                    {
                        "file": filename,
                        "sha256": self._sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                )

            if templates:
                np.savez_compressed(directory / _TEMPLATES_NAME, templates=np.stack(templates))

            manifest = {
                "schema_version": _SCHEMA_VERSION,
                "student": student,
                "student_key": self.student_key(student),
                "created_at_utc": datetime.utcnow().isoformat() + "Z",
                "image_count": len(written),
                "template_count": len(templates),
                "images": entries,
                "model": model or {},
            }
            (directory / _MANIFEST_NAME).write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        except Exception:
            # A half-written enrolment is worse than none: it would verify against
            # an incomplete reference set. Remove it and let the caller re-enrol.
            LOGGER.exception("Enrolment write failed for %s; removing partial data", student)
            self._clear_enrollment_dir(directory)
            raise
        finally:
            self.cleanup_staging(student)

        LOGGER.info("Enrolment stored for %s: %d image(s) in %s", student, len(written), directory)
        return self.load_enrollment(student)

    def delete_enrollment(self, student: str) -> bool:
        """Erase a student's enrolment entirely.

        Present so a withdrawal of consent or an expired retention period can
        actually be honoured: reference photographs and embeddings are biometric
        data, and an erasure path nobody can call is not an erasure path.
        """
        directory = self.enrollment_dir(student)
        if not directory.exists():
            return False
        self._assert_inside_data_root(directory)
        shutil.rmtree(directory, ignore_errors=True)
        LOGGER.info("Enrolment deleted for %s", student)
        return True

    # ------------------------------------------------------------------
    # Temporary artefacts
    # ------------------------------------------------------------------

    def stage_image(self, student: str, index: int, image: np.ndarray) -> Path:
        """Write one in-progress capture to the staging area.

        Staged frames exist only so an interrupted enrolment can be inspected while
        it runs; they are never the record. :meth:`cleanup_staging` removes them.
        """
        import cv2

        staging = self.staging_dir(student)
        staging.mkdir(parents=True, exist_ok=True)
        path = staging / _IMAGE_PATTERN.format(index=index)
        cv2.imwrite(str(path), image)
        return path

    def cleanup_staging(self, student: str) -> int:
        """Remove a student's staging directory. Returns the number of files removed.

        Safe to call at any time, including when nothing was staged. Only the
        dot-prefixed staging directory is touched, so permanent references sitting
        beside it are never at risk.
        """
        staging = self.staging_dir(student)
        if not staging.exists():
            return 0
        self._assert_inside_data_root(staging)
        if staging.name != _STAGING_DIRNAME:
            raise ValueError(f"Refusing to remove a non-staging directory: {staging}")

        removed = sum(1 for p in staging.rglob("*") if p.is_file())
        shutil.rmtree(staging, ignore_errors=True)
        if removed:
            LOGGER.debug("Removed %d staged file(s) for %s", removed, student)
        return removed

    def purge_all_staging(self) -> int:
        """Clear staging left behind by interrupted enrolments across all students.

        Intended for start-up. It walks only directories named ``.staging`` beneath
        the students root — never a recursive delete of anything broader.
        """
        if not self.students_root.exists():
            return 0
        removed = 0
        for staging in self.students_root.glob(f"*/enrollment/{_STAGING_DIRNAME}"):
            if not staging.is_dir():
                continue
            self._assert_inside_data_root(staging)
            removed += sum(1 for p in staging.rglob("*") if p.is_file())
            shutil.rmtree(staging, ignore_errors=True)
        if removed:
            LOGGER.info("Purged %d stale staged file(s)", removed)
        return removed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assert_inside_data_root(self, path: Path) -> None:
        """Refuse to delete anything outside the configured data root."""
        root = self.data_root.resolve()
        target = path.resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"Refusing to remove a path outside {root}: {target}")

    def _clear_enrollment_dir(self, directory: Path) -> None:
        """Remove a previous enrolment before writing a replacement."""
        if not directory.exists():
            return
        self._assert_inside_data_root(directory)
        shutil.rmtree(directory, ignore_errors=True)

    @staticmethod
    def _load_templates(directory: Path, problems: list[str]) -> list[np.ndarray]:
        path = directory / _TEMPLATES_NAME
        if not path.exists():
            problems.append("enrolment templates are missing")
            return []
        try:
            with np.load(path) as archive:
                if "templates" in archive.files:
                    return [np.asarray(row) for row in archive["templates"]]
                return [archive[key] for key in sorted(archive.files)]
        except Exception as exc:
            problems.append(f"enrolment templates could not be read: {exc}")
            return []

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while chunk := handle.read(65536):
                digest.update(chunk)
        return digest.hexdigest()
