"""Offline synchronization outbox for durable event queuing, retry, and idempotent transport."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from proctoring.core.events import EventRecord
from proctoring.integration.schemas import OutboxEventRecord, SyncStatus

LOGGER = logging.getLogger(__name__)


@dataclass
class SyncBatchReport:
    """Summary of an outbox sync attempt to a remote Exam Controller endpoint."""

    total_pending: int
    synced_count: int
    duplicate_count: int
    failed_count: int
    synced_sequence_numbers: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pending": self.total_pending,
            "synced_count": self.synced_count,
            "duplicate_count": self.duplicate_count,
            "failed_count": self.failed_count,
            "synced_sequence_numbers": self.synced_sequence_numbers,
            "errors": self.errors,
        }


class OfflineOutboxManager:
    """Manages durable local event queuing, retry backoff, and idempotent synchronization."""

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.outbox_file = self.session_dir / "sync_outbox.jsonl"
        self._lock = os.path.join(self.session_dir, ".sync_outbox.lock")
        self._processed_idempotency_keys: set[str] = set()

    def enqueue(self, event: EventRecord) -> OutboxEventRecord:
        """Add a locally generated EventRecord to the durable offline sync queue."""
        outbox_entry = OutboxEventRecord(
            event_id=event.event_id,
            session_id=event.session_id,
            sequence_number=event.sequence_number,
            event_type=event.event_type.value,
            timestamp=event.timestamp,
            created_at_utc=event.created_at_utc,
            payload=event.to_dict(),
            sync_status=SyncStatus.PENDING_SYNC,
            retry_count=0,
            last_error=None,
        )

        try:
            line = json.dumps(outbox_entry.to_dict()) + "\n"
            with open(self.outbox_file, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except Exception as exc:
            LOGGER.error("Failed to enqueue event %s to outbox: %s", event.event_id, exc)

        return outbox_entry

    def get_pending_records(self) -> list[OutboxEventRecord]:
        """Retrieve all unsynchronized outbox records ordered by sequence_number."""
        if not self.outbox_file.exists():
            return []

        all_records: dict[str, OutboxEventRecord] = {}
        try:
            with open(self.outbox_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = OutboxEventRecord.from_dict(json.loads(line))
                        all_records[record.idempotency_key] = record
                    except Exception:
                        pass
        except Exception as exc:
            LOGGER.error("Failed reading outbox file %s: %s", self.outbox_file, exc)

        pending = [
            rec
            for rec in all_records.values()
            if rec.sync_status in (SyncStatus.PENDING_SYNC, SyncStatus.SYNC_FAILED, SyncStatus.LOCAL_DURABLE)
        ]
        return sorted(pending, key=lambda r: r.sequence_number)

    def mark_synchronized(self, idempotency_keys: list[str]) -> None:
        """Update synchronized records to SYNCHRONIZED state."""
        if not self.outbox_file.exists() or not idempotency_keys:
            return

        key_set = set(idempotency_keys)
        records: list[OutboxEventRecord] = []
        with open(self.outbox_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = OutboxEventRecord.from_dict(json.loads(line))
                    if record.idempotency_key in key_set:
                        record.sync_status = SyncStatus.SYNCHRONIZED
                    records.append(record)
                except Exception:
                    pass

        # Rewrite atomically
        tmp_file = self.session_dir / "sync_outbox.jsonl.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict()) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, self.outbox_file)

    def record_failure(self, idempotency_key: str, error: str) -> None:
        """Mark record as SYNC_FAILED with incremented retry count and last error message."""
        if not self.outbox_file.exists():
            return

        records: list[OutboxEventRecord] = []
        with open(self.outbox_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = OutboxEventRecord.from_dict(json.loads(line))
                    if record.idempotency_key == idempotency_key:
                        record.sync_status = SyncStatus.SYNC_FAILED
                        record.retry_count += 1
                        record.last_error = error
                    records.append(record)
                except Exception:
                    pass

        tmp_file = self.session_dir / "sync_outbox.jsonl.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict()) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, self.outbox_file)

    def sync_to_remote(
        self,
        sender_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
        max_batch_size: int = 50,
    ) -> SyncBatchReport:
        """Execute a sync cycle sending pending batches through a transport function."""
        pending = self.get_pending_records()
        if not pending:
            return SyncBatchReport(
                total_pending=0, synced_count=0, duplicate_count=0, failed_count=0
            )

        batch = pending[:max_batch_size]
        payloads = [r.to_dict() for r in batch]
        synced_keys: list[str] = []
        synced_seqs: list[int] = []
        errors: list[str] = []

        try:
            ack = sender_fn(payloads)
            ack_synced = set(ack.get("synced_sequence_numbers", []))
            ack_dups = set(ack.get("duplicate_sequence_numbers", []))
            ack_event_ids = set(ack.get("acknowledged_event_ids", [])) | set(ack.get("duplicate_event_ids", []))

            for rec in batch:
                if (
                    rec.sequence_number in ack_synced
                    or rec.sequence_number in ack_dups
                    or rec.event_id in ack_event_ids
                    or rec.idempotency_key in ack_event_ids
                ):
                    synced_keys.append(rec.idempotency_key)
                    synced_seqs.append(rec.sequence_number)
                else:
                    self.record_failure(rec.idempotency_key, "Unacknowledged sequence number")

            self.mark_synchronized(synced_keys)

            return SyncBatchReport(
                total_pending=len(pending),
                synced_count=len(synced_keys),
                duplicate_count=len(ack_dups),
                failed_count=len(batch) - len(synced_keys),
                synced_sequence_numbers=synced_seqs,
                errors=errors,
            )
        except Exception as exc:
            error_msg = str(exc)
            LOGGER.warning("Sync batch delivery failed: %s", error_msg)
            for rec in batch:
                self.record_failure(rec.idempotency_key, error_msg)
            return SyncBatchReport(
                total_pending=len(pending),
                synced_count=0,
                duplicate_count=0,
                failed_count=len(batch),
                synced_sequence_numbers=[],
                errors=[error_msg],
            )
