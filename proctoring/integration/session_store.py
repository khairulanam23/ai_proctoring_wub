"""Persistence for session records and candidate enrolment templates.

The default implementation keeps records in memory and mirrors them to JSON on
disk, which is enough for a single-process deployment and for tests.  A Moodle
installation running several PHP-FPM workers behind one proctoring service will
want the same interface backed by its own database; :class:`SessionStore` is
written to be subclassed for exactly that.

Enrolment templates are handled separately from session records and never leave
the server.  They are face embeddings — biometric data — so they are stored under
an opaque identifier, kept out of every response, and given an explicit deletion
path so a candidate's right to erasure can actually be honoured.
"""

import builtins
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from proctoring.core.paths import sanitise_identifier
from proctoring.integration.schemas import SessionState
from proctoring.storage import ProctoringStorage

LOGGER = logging.getLogger(__name__)


@dataclass
class SessionRecord:
    """Everything retained about one proctored attempt."""

    session_id: str
    attempt_id: str
    user_id: str
    candidate_name: str = "Candidate"
    course_id: str | None = None
    quiz_id: str | None = None
    state: SessionState = SessionState.CREATED
    strictness: str = "STANDARD"

    started_at_utc: str = ""
    ended_at_utc: str = ""
    package_dir: str = ""
    error: str | None = None

    result: dict[str, Any] | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    timeline_summary: dict[str, Any] = field(default_factory=dict)
    telemetry_summary: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionRecord":
        data = dict(data)
        data["state"] = SessionState(data.get("state", SessionState.CREATED))
        return cls(**data)


class SessionStore:
    """In-memory session registry with JSON mirroring and enrolment storage.

    Subclass and override :meth:`put`, :meth:`get`, :meth:`list` and the enrolment
    methods to back this with a real database.
    """

    def __init__(
        self,
        storage_dir: str | Path = "data/proctoring_sessions",
        enrolment_storage_dir: str | Path | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.sessions_dir = self.storage_dir / "sessions"
        # Enrolment is delegated rather than duplicated: ProctoringStorage owns the
        # single identity store, so reference images and the embeddings derived from
        # them live together and are erased together.
        if enrolment_storage_dir is not None:
            enrol_dir = Path(enrolment_storage_dir)
        elif self.storage_dir == Path("data/proctoring_sessions") and (Path("data") / "students").exists():
            enrol_dir = Path("data")
        else:
            enrol_dir = self.storage_dir
        self.storage = ProctoringStorage(enrol_dir)
        self.enrolment_dir = self.storage.students_root
        self._records: dict[str, SessionRecord] = {}
        self._lock = threading.RLock()

    def _ensure_dirs(self) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_path(directory: Path, identifier: str, suffix: str) -> Path:
        """Build a path inside ``directory`` from an untrusted identifier.

        Session and enrolment ids come from the host. Sanitising them keeps a
        traversal sequence from writing records — or a candidate's biometric
        template — outside the store.
        """
        return directory / f"{sanitise_identifier(identifier, fallback='record')}{suffix}"

    # ------------------------------------------------------------------
    # Session records
    # ------------------------------------------------------------------

    def put(self, record: SessionRecord) -> SessionRecord:
        """Insert or update a session record."""
        with self._lock:
            self._records[record.session_id] = record
            try:
                self._ensure_dirs()
                path = self._safe_path(self.sessions_dir, record.session_id, ".json")
                path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
            except Exception as exc:
                # Disk mirroring is a convenience for inspection and restart; losing
                # it must not fail an in-progress exam.
                LOGGER.warning("Session record mirror failed for %s: %s", record.session_id, exc)
        return record

    def get(self, session_id: str) -> SessionRecord | None:
        """Fetch a record, falling back to the on-disk mirror after a restart."""
        with self._lock:
            record = self._records.get(session_id)
            if record is not None:
                return record
            for rec in self._records.values():
                if rec.attempt_id == session_id:
                    return rec
            path = self._safe_path(self.sessions_dir, session_id, ".json")
            if path.exists():
                try:
                    record = SessionRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
                    self._records[session_id] = record
                    return record
                except Exception as exc:
                    LOGGER.warning("Unreadable session mirror %s: %s", path, exc)
                    return None
        return None

    def list(
        self,
        attempt_id: str | None = None,
        user_id: str | None = None,
    ) -> list[SessionRecord]:
        """List records, newest first, optionally filtered."""
        with self._lock:
            records = list(self._records.values())
        if attempt_id is not None:
            records = [r for r in records if r.attempt_id == attempt_id]
        if user_id is not None:
            records = [r for r in records if r.user_id == user_id]
        return sorted(records, key=lambda r: r.started_at_utc, reverse=True)

    def delete(self, session_id: str) -> bool:
        """Remove a session record and its on-disk mirror."""
        with self._lock:
            existed = self._records.pop(session_id, None) is not None
            path = self._safe_path(self.sessions_dir, session_id, ".json")
            if path.exists():
                path.unlink()
                existed = True
        return existed

    # ------------------------------------------------------------------
    # Enrolment templates (biometric data — server-side only)
    # ------------------------------------------------------------------

    def save_enrolment(
        self,
        enrolment_id: str,
        templates: builtins.list[np.ndarray],
        images: builtins.list[np.ndarray] | None = None,
    ) -> bool:
        """Store a candidate's reference embeddings, and their images when supplied.

        Delegates to :class:`ProctoringStorage` so the service and the CLI share one
        identity store rather than writing to two places that can disagree.
        """
        if not templates and not images:
            return False
        try:
            self.storage.save_enrollment(enrolment_id, images, templates)
            return True
        except Exception as exc:
            LOGGER.error("Enrolment save failed for %s: %s", enrolment_id, exc)
            return False

    def load_enrolment(self, enrolment_id: str) -> builtins.list[np.ndarray]:
        """Load a candidate's reference embeddings, or an empty list if unusable."""
        keys_to_try = [enrolment_id]
        if isinstance(enrolment_id, str):
            clean = enrolment_id.strip()
            hyphenated = clean.replace(" ", "-").replace("_", "-")
            underscored = clean.replace(" ", "_").replace("-", "_")
            for k in (hyphenated, underscored, clean.lower(), hyphenated.lower()):
                if k and k not in keys_to_try:
                    keys_to_try.append(k)

        for key in keys_to_try:
            templates = self.storage.load_templates(key)
            if not templates and (Path("data") / "students").exists():
                try:
                    templates = ProctoringStorage(Path("data")).load_templates(key)
                except Exception:
                    pass
            if templates:
                return templates
        return []

    def delete_enrolment(self, enrolment_id: str) -> bool:
        """Erase a candidate's stored reference images and embeddings.

        Present so an erasure request can be honoured without hunting through the
        filesystem; hosts should call it when a candidate withdraws consent or when
        a retention period expires.
        """
        return self.storage.delete_enrollment(enrolment_id)
