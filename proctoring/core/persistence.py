"""Progressive durable session persistence, append-only journals, and crash checkpoints."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from proctoring.core.events import EventRecord

LOGGER = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = "1.0"


@dataclass
class SessionCheckpoint:
    """Metadata describing the last durable state of an in-progress session."""

    schema_version: str
    session_id: str
    student_name: str
    state: str
    started_at_iso: str
    last_checkpoint_utc: str
    last_frame_index: int
    last_timestamp_seconds: float
    processed_frames: int
    total_events: int
    sequence_number: int
    exam_id: str = "default_exam"
    candidate_id: str = "default_candidate"
    models_info: dict[str, Any] = field(default_factory=dict)
    processing_config: dict[str, Any] = field(default_factory=dict)
    is_recovered: bool = False
    recovery_notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionCheckpoint":
        return cls(
            schema_version=data.get("schema_version", CHECKPOINT_SCHEMA_VERSION),
            session_id=data["session_id"],
            student_name=data.get("student_name", "Unknown"),
            state=data["state"],
            started_at_iso=data.get("started_at_iso", ""),
            last_checkpoint_utc=data.get("last_checkpoint_utc", ""),
            last_frame_index=int(data.get("last_frame_index", 0)),
            last_timestamp_seconds=float(data.get("last_timestamp_seconds", 0.0)),
            processed_frames=int(data.get("processed_frames", 0)),
            total_events=int(data.get("total_events", 0)),
            sequence_number=int(data.get("sequence_number", 0)),
            exam_id=data.get("exam_id", "default_exam"),
            candidate_id=data.get("candidate_id", "default_candidate"),
            models_info=data.get("models_info", {}),
            processing_config=data.get("processing_config", {}),
            is_recovered=bool(data.get("is_recovered", False)),
            recovery_notes=data.get("recovery_notes"),
        )


class SessionJournalManager:
    """Manages append-only JSONL files and atomic checkpoints for mid-session durability."""

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.events_jsonl_path = self.session_dir / "events.jsonl"
        self.timeline_jsonl_path = self.session_dir / "timeline.jsonl"
        self.diagnostics_jsonl_path = self.session_dir / "diagnostics.jsonl"
        self.checkpoint_path = self.session_dir / "session_checkpoint.json"

    def append_event(self, event: EventRecord) -> None:
        """Atomically append an EventRecord to events.jsonl with immediate flush."""
        try:
            line = json.dumps(event.to_dict()) + "\n"
            with open(self.events_jsonl_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except Exception as exc:
            LOGGER.error("Failed to append event %s to journal: %s", event.event_id, exc)

    def append_timeline_entry(self, entry: dict[str, Any]) -> None:
        """Append a timeline entry to timeline.jsonl with immediate flush."""
        try:
            line = json.dumps(entry) + "\n"
            with open(self.timeline_jsonl_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        except Exception as exc:
            LOGGER.error("Failed to append timeline entry to journal: %s", exc)

    def append_diagnostic(self, diag: dict[str, Any]) -> None:
        """Append a diagnostic entry to diagnostics.jsonl with immediate flush."""
        try:
            line = json.dumps(diag) + "\n"
            with open(self.diagnostics_jsonl_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        except Exception as exc:
            LOGGER.error("Failed to append diagnostic to journal: %s", exc)

    def write_checkpoint(self, checkpoint: SessionCheckpoint) -> None:
        """Write session checkpoint using atomic file replacement (temp file + rename)."""
        tmp_path = self.session_dir / "session_checkpoint.json.tmp"
        try:
            data = checkpoint.to_dict()
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.checkpoint_path)
        except Exception as exc:
            LOGGER.error("Failed to write session checkpoint atomically: %s", exc)
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def load_checkpoint(self) -> SessionCheckpoint | None:
        """Load the session checkpoint if it exists."""
        if not self.checkpoint_path.exists():
            return None
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return SessionCheckpoint.from_dict(data)
        except Exception as exc:
            LOGGER.error("Failed to read session checkpoint from %s: %s", self.checkpoint_path, exc)
            return None

    def read_events(self) -> list[EventRecord]:
        """Read all persisted events from events.jsonl."""
        if not self.events_jsonl_path.exists():
            return []
        events: list[EventRecord] = []
        with open(self.events_jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    events.append(EventRecord.from_dict(data))
                except Exception as exc:
                    LOGGER.warning("Malformed JSON on line %d of %s: %s", line_no, self.events_jsonl_path, exc)
        return events

    def read_timeline(self) -> list[dict[str, Any]]:
        """Read all timeline entries from timeline.jsonl."""
        if not self.timeline_jsonl_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with open(self.timeline_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
        return entries

    def read_diagnostics(self) -> list[dict[str, Any]]:
        """Read all diagnostic entries from diagnostics.jsonl."""
        if not self.diagnostics_jsonl_path.exists():
            return []
        diagnostics: list[dict[str, Any]] = []
        with open(self.diagnostics_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        diagnostics.append(json.loads(line))
                    except Exception:
                        pass
        return diagnostics
