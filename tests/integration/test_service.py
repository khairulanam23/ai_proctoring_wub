"""Tests for the LMS integration boundary a Moodle plugin will call."""

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from proctoring.integration import (
    CONTRACT_VERSION,
    ProctoringService,
    ProctoringServiceError,
    SessionHandle,
    SessionState,
    SessionStore,
    StartSessionRequest,
)


def _frame(value: int = 130) -> np.ndarray:
    return np.full((480, 640, 3), value, dtype=np.uint8)


def _data_url(frame: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", frame)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode()


@pytest.fixture
def service(tmp_path):
    """A service with no detectors, so tests exercise the boundary not the models."""
    return ProctoringService(
        output_dir=tmp_path / "packages",
        store=SessionStore(tmp_path / "store"),
        detector_factory=lambda: {},
    )


def _start(service, **overrides) -> "SessionHandle":
    request = StartSessionRequest(
        attempt_id=overrides.pop("attempt_id", "4471"),
        user_id=overrides.pop("user_id", "82"),
        candidate_name="A. Candidate",
        **overrides,
    )
    return service.start_session(request)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_full_attempt_lifecycle(service):
    """Start, ingest, record a client event, finalise, review."""
    handle = _start(service)
    assert handle.state is SessionState.ACTIVE
    assert handle.contract_version == CONTRACT_VERSION
    assert service.get_state(handle.session_id) is SessionState.ACTIVE

    for index in range(6):
        ack = service.ingest_frame(
            handle.session_id, _data_url(_frame()), timestamp_seconds=index * 0.25
        )
        assert ack.session_id == handle.session_id
        assert ack.accepted is True
        assert ack.next_frame_due_in_seconds > 0

    service.record_client_event(handle.session_id, "BROWSER_TAB_SWITCH", 1.5, "Tab hidden")

    result = service.finalize_session(handle.session_id)
    assert result.state is SessionState.COMPLETED
    assert result.frames_processed == 6
    assert result.integrity_verified is True
    assert Path(result.package_dir).exists()
    assert result.manifest_sha256

    review = service.get_review(handle.session_id)
    assert review.attempt_id == "4471"
    assert any(o.event_type == "BROWSER_TAB_SWITCH" for o in review.observations)
    assert review.reviewer_guidance, "review payload must carry framing guidance"


def test_session_reports_what_it_cannot_observe(service):
    """A session with no detectors must say so rather than implying full coverage."""
    handle = _start(service)
    assert "face_detection" in handle.unavailable_detectors
    assert handle.identity_verification_enabled is False

    service.ingest_frame(handle.session_id, _frame())
    result = service.finalize_session(handle.session_id)

    assert result.coverage_warnings
    assert any("Not observed" in w for w in result.coverage_warnings)


def test_frames_rejected_after_finalisation(service):
    """A finalised attempt must not silently accept more frames."""
    handle = _start(service)
    service.ingest_frame(handle.session_id, _frame())
    service.finalize_session(handle.session_id)

    with pytest.raises(ProctoringServiceError, match="COMPLETED"):
        service.ingest_frame(handle.session_id, _frame())


def test_unknown_session_is_rejected(service):
    with pytest.raises(ProctoringServiceError, match="Unknown session"):
        service.ingest_frame("no-such-session", _frame())


def test_aborted_attempt_still_seals_its_evidence(service):
    """A crashed or withdrawn attempt keeps the record of what happened before it."""
    handle = _start(service)
    for index in range(3):
        service.ingest_frame(handle.session_id, _frame(), timestamp_seconds=index * 0.25)

    result = service.abort_session(handle.session_id, reason="browser closed")
    assert result.state is SessionState.FAILED
    assert Path(result.package_dir).exists()
    assert any("before normal completion" in w for w in result.coverage_warnings)


# ---------------------------------------------------------------------------
# Frame decoding
# ---------------------------------------------------------------------------


def test_frame_accepted_in_every_transport_form(service):
    """Data URL, raw bytes and decoded array must all be accepted."""
    handle = _start(service)
    ok, buffer = cv2.imencode(".jpg", _frame())

    for payload in (_data_url(_frame()), buffer.tobytes(), _frame()):
        ack = service.ingest_frame(handle.session_id, payload)
        assert ack.accepted is True
    service.finalize_session(handle.session_id)


def test_corrupt_payload_does_not_abort_the_attempt(service):
    """A bad POST is a capture fault: recorded and skipped, never fatal.

    An exam must survive a truncated upload; raising here would end a candidate's
    attempt over a dropped packet.
    """
    handle = _start(service)
    ack = service.ingest_frame(handle.session_id, "data:image/jpeg;base64,!!!not-base64!!!")
    assert ack.accepted is False
    assert ack.rejection_reason is not None

    # The session must still be usable afterwards.
    assert service.ingest_frame(handle.session_id, _frame()).accepted is True
    result = service.finalize_session(handle.session_id)
    assert result.frames_rejected >= 1
    assert result.integrity_verified is True


def test_unknown_client_event_is_recorded_not_dropped(service):
    """A newer client sending an unrecognised event still gets it on the timeline."""
    handle = _start(service)
    response = service.record_client_event(handle.session_id, "SOME_FUTURE_EVENT", 1.0, "unknown")
    assert response["recorded"] is True
    assert response["event_type"] == "OTHER_SUSPICIOUS_ACTIVITY"
    service.finalize_session(handle.session_id)


# ---------------------------------------------------------------------------
# Review contract
# ---------------------------------------------------------------------------


def test_weak_observations_carry_a_reliability_note(service, tmp_path):
    """Observations built on weak signals must warn the proctor inline."""
    from proctoring.core.events import (
        DetectorInfo,
        EventRecord,
        EventSeverity,
        EventStatus,
        EventType,
        ObservationDetail,
    )

    event = EventRecord(
        event_id="e1",
        session_id="s",
        timestamp=1.0,
        end_timestamp=3.0,
        duration=2.0,
        formatted_start="00:00:01.000",
        formatted_end="00:00:03.000",
        event_type=EventType.EARBUDS_SUSPECTED,
        severity=EventSeverity.MEDIUM,
        confidence=0.4,
        average_confidence=0.4,
        detector=DetectorInfo(name="test"),
        observation=ObservationDetail(description="Earbuds detected"),
        status=EventStatus.QUALIFIED,
    )
    summary = ProctoringService._summarise(event)
    assert summary.reliability_note is not None
    assert "earring" in summary.reliability_note.lower()


def test_review_payload_marks_derived_images(service):
    """Annotated review images must be distinguishable from source evidence."""
    from proctoring.core.events import (
        DetectorInfo,
        EventRecord,
        EventSeverity,
        EventType,
        EvidenceReference,
        ObservationDetail,
    )

    event = EventRecord(
        event_id="e1",
        session_id="s",
        timestamp=0.0,
        end_timestamp=1.0,
        duration=1.0,
        formatted_start="a",
        formatted_end="b",
        event_type=EventType.NO_FACE,
        severity=EventSeverity.HIGH,
        confidence=1.0,
        average_confidence=1.0,
        detector=DetectorInfo(name="t"),
        observation=ObservationDetail(description="d"),
        evidence=[
            EvidenceReference("ev1", "frame", "evidence/frames/a.jpg", 0.0, "a", 0),
            EvidenceReference("ev2", "review", "evidence/review/a.jpg", 0.0, "a", 0),
        ],
    )
    summary = ProctoringService._summarise(event)
    by_type = {e["media_type"]: e["is_derived"] for e in summary.evidence}
    assert by_type["frame"] is False
    assert by_type["review"] is True


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------


def test_store_persists_and_reloads_records(tmp_path):
    """Records survive a process restart via the on-disk mirror."""
    store = SessionStore(tmp_path / "store")
    handle = ProctoringService(
        output_dir=tmp_path / "pkg", store=store, detector_factory=lambda: {}
    ).start_session(StartSessionRequest(attempt_id="1", user_id="2"))

    reloaded = SessionStore(tmp_path / "store").get(handle.session_id)
    assert reloaded is not None
    assert reloaded.attempt_id == "1"


def test_enrolment_templates_can_be_erased(tmp_path):
    """Stored biometric templates must have a working deletion path."""
    store = SessionStore(tmp_path / "store")
    assert store.save_enrolment("user9", [np.ones(128, dtype=np.float32)]) is True
    assert len(store.load_enrolment("user9")) == 1

    assert store.delete_enrolment("user9") is True
    assert store.load_enrolment("user9") == []
    assert store.delete_enrolment("user9") is False


def test_sessions_can_be_filtered_by_attempt_and_user(tmp_path):
    service = ProctoringService(
        output_dir=tmp_path / "pkg",
        store=SessionStore(tmp_path / "store"),
        detector_factory=lambda: {},
    )
    service.start_session(StartSessionRequest(attempt_id="A", user_id="1"))
    service.start_session(StartSessionRequest(attempt_id="B", user_id="2"))

    assert len(service.list_sessions(attempt_id="A")) == 1
    assert len(service.list_sessions(user_id="2")) == 1
    assert len(service.list_sessions()) == 2


def test_start_request_round_trips_through_json():
    """The contract must survive the wire without loss."""
    request = StartSessionRequest(
        attempt_id="4471",
        user_id="82",
        course_id="12",
        strictness="STRICT",
        enable_wearable_detection=True,
        metadata={"ip": "10.0.0.1"},
    )
    restored = StartSessionRequest.from_dict(json.loads(json.dumps(request.to_dict())))
    assert restored == request
