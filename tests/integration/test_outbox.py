"""Integration tests for the offline outbox manager and synchronization protocol."""

from __future__ import annotations

import time
from pathlib import Path

from proctoring.core.events import (
    DetectorInfo,
    EventRecord,
    EventSeverity,
    EventType,
    ObservationDetail,
    format_seconds_to_timestamp,
)
from proctoring.integration.outbox import OfflineOutboxManager
from proctoring.integration.schemas import (
    SyncStatus,
)


def _make_event(
    event_id: str, session_id: str, seq: int, event_type: EventType, timestamp: float = 1.0, metadata: dict | None = None
) -> EventRecord:
    return EventRecord(
        event_id=event_id,
        session_id=session_id,
        sequence_number=seq,
        timestamp=timestamp,
        end_timestamp=timestamp + 1.0,
        duration=1.0,
        formatted_start=format_seconds_to_timestamp(timestamp),
        formatted_end=format_seconds_to_timestamp(timestamp + 1.0),
        event_type=event_type,
        severity=EventSeverity.LOW,
        confidence=0.9,
        average_confidence=0.9,
        detector=DetectorInfo(name="test_detector"),
        observation=ObservationDetail(description="test observation"),
        metadata=metadata or {},
    )


def test_outbox_queue_and_batching(tmp_path: Path):
    outbox = OfflineOutboxManager(session_dir=tmp_path)

    # Queue 3 events
    evt1 = _make_event("evt_001", "sess_abc", 1, EventType.PERSON_LEFT_FRAME, 10.0, {"confidence": 0.95})
    evt2 = _make_event("evt_002", "sess_abc", 2, EventType.PERSON_ENTERED_FRAME, 15.0, {"confidence": 0.99})
    evt3 = _make_event("evt_003", "sess_abc", 3, EventType.PHONE_DETECTED, 20.0, {"object": "phone", "confidence": 0.88})

    outbox.enqueue(evt1)
    outbox.enqueue(evt2)
    outbox.enqueue(evt3)

    pending = outbox.get_pending_records()
    assert len(pending) == 3
    assert [p.sequence_number for p in pending] == [1, 2, 3]

    # Synchronize first event
    outbox.mark_synchronized([pending[0].idempotency_key])

    remaining = outbox.get_pending_records()
    assert len(remaining) == 2
    assert [r.sequence_number for r in remaining] == [2, 3]


def test_outbox_record_failure_retry(tmp_path: Path):
    outbox = OfflineOutboxManager(session_dir=tmp_path)
    evt = _make_event("evt_retry_1", "sess_xyz", 1, EventType.LOOKING_AWAY, 5.0)
    rec = outbox.enqueue(evt)
    assert rec.retry_count == 0

    outbox.record_failure(rec.idempotency_key, error="HTTP 503 Service Unavailable")
    pending = outbox.get_pending_records()
    assert len(pending) == 1
    assert pending[0].sync_status == SyncStatus.SYNC_FAILED
    assert pending[0].retry_count == 1
    assert pending[0].last_error == "HTTP 503 Service Unavailable"


def test_outbox_sync_to_remote(tmp_path: Path):
    outbox = OfflineOutboxManager(session_dir=tmp_path)
    for i in range(5):
        outbox.enqueue(
            _make_event(
                event_id=f"evt_{i}",
                session_id="sess_batch",
                seq=i + 1,
                event_type=EventType.SUSPICIOUS_HEAD_POSE,
                timestamp=float(i),
            )
        )

    assert len(outbox.get_pending_records()) == 5

    def mock_remote_transport(batch: list[dict]):
        # Acknowledge first 3
        return {
            "acknowledged_event_ids": [b["idempotency_key"] for b in batch[:3]],
            "duplicate_event_ids": [],
        }

    report = outbox.sync_to_remote(mock_remote_transport, max_batch_size=5)
    assert report.synced_count == 3
    assert len(outbox.get_pending_records()) == 2


def test_outbox_persistence_across_restarts(tmp_path: Path):
    outbox1 = OfflineOutboxManager(session_dir=tmp_path)
    evt = _make_event(
        event_id="evt_durable_1",
        session_id="sess_persistent",
        seq=1,
        event_type=EventType.SUSPICIOUS_HEAD_POSE,
        timestamp=42.0,
        metadata={"yaw": 35.0},
    )
    outbox1.enqueue(evt)

    # Reload from disk
    outbox2 = OfflineOutboxManager(session_dir=tmp_path)
    pending = outbox2.get_pending_records()
    assert len(pending) == 1
    assert pending[0].event_id == "evt_durable_1"
    assert pending[0].sequence_number == 1
    assert pending[0].payload["metadata"]["yaw"] == 35.0
