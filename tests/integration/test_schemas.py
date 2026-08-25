"""Tests for the LMS data contracts.

These shapes are what a Moodle plugin depends on, so they change only by bumping
``CONTRACT_VERSION``. The tests assert the guarantees the contract makes rather
than merely that the dataclasses construct.
"""

import json

import pytest

from proctoring.core.events import EventType
from proctoring.integration.schemas import (
    CONTRACT_VERSION,
    DEFAULT_REVIEWER_GUIDANCE,
    RELIABILITY_NOTES,
    FrameAck,
    ObservationSummary,
    ReviewPayload,
    SessionHandle,
    SessionResult,
    SessionState,
    StartSessionRequest,
)


def _json_round_trip(payload: dict) -> dict:
    """Prove a payload survives an actual JSON encode/decode, as it would on the wire."""
    return json.loads(json.dumps(payload))


# ---------------------------------------------------------------------------
# Wire safety
# ---------------------------------------------------------------------------


def test_start_request_round_trips():
    request = StartSessionRequest(
        attempt_id="4471",
        user_id="82",
        course_id="12",
        quiz_id="99",
        candidate_name="A. Candidate",
        strictness="STRICT",
        sampling_fps=2.5,
        enable_wearable_detection=True,
        enrolment_id="user_82",
        metadata={"ip": "10.0.0.1", "browser": "firefox"},
    )
    assert StartSessionRequest.from_dict(_json_round_trip(request.to_dict())) == request


def test_start_request_tolerates_a_minimal_payload():
    """Only attempt and user are required; everything else has a usable default."""
    request = StartSessionRequest.from_dict({"attempt_id": "1", "user_id": "2"})
    assert request.strictness == "STANDARD"
    assert request.sampling_fps == 4.0
    assert request.enable_wearable_detection is False


def test_start_request_coerces_numeric_ids_to_strings():
    """Moodle sends integer ids; the contract stores them as strings consistently."""
    request = StartSessionRequest.from_dict({"attempt_id": 4471, "user_id": 82})
    assert request.attempt_id == "4471"
    assert request.user_id == "82"


@pytest.mark.parametrize(
    "payload",
    [
        SessionHandle(session_id="s", attempt_id="a", user_id="u", state=SessionState.ACTIVE),
        FrameAck(session_id="s", frame_index=0, timestamp_seconds=0.0, accepted=True),
        SessionResult(session_id="s", attempt_id="a", user_id="u", state=SessionState.COMPLETED),
    ],
)
def test_every_response_is_json_serialisable(payload):
    """No numpy arrays, enums or Paths may leak across the boundary."""
    encoded = _json_round_trip(payload.to_dict())
    assert isinstance(encoded, dict)
    assert isinstance(encoded["session_id"], str)


def test_states_serialise_as_plain_strings():
    handle = SessionHandle(
        session_id="s", attempt_id="a", user_id="u", state=SessionState.FINALIZING
    )
    assert handle.to_dict()["state"] == "FINALIZING"


def test_contract_version_is_stamped_on_every_response():
    """A host must be able to detect a server it does not understand."""
    for payload in (
        SessionHandle(session_id="s", attempt_id="a", user_id="u", state=SessionState.ACTIVE),
        SessionResult(session_id="s", attempt_id="a", user_id="u", state=SessionState.COMPLETED),
        ReviewPayload(session_id="s", attempt_id="a", user_id="u", candidate_name="c"),
    ):
        assert payload.to_dict()["contract_version"] == CONTRACT_VERSION


# ---------------------------------------------------------------------------
# Guarantees the contract makes
# ---------------------------------------------------------------------------


def test_no_response_carries_a_risk_or_suspicion_score():
    """The contract must not expose anything a host could render as a verdict.

    The system reports observations; the academic judgement is the invigilator's.
    A field named like a score would invite exactly the misuse the design avoids.
    """
    banned = ("score", "risk", "suspicion", "cheat", "guilt", "probability")
    payloads = [
        SessionHandle(
            session_id="s", attempt_id="a", user_id="u", state=SessionState.ACTIVE
        ).to_dict(),
        FrameAck(session_id="s", frame_index=0, timestamp_seconds=0.0, accepted=True).to_dict(),
        SessionResult(
            session_id="s", attempt_id="a", user_id="u", state=SessionState.COMPLETED
        ).to_dict(),
        ReviewPayload(session_id="s", attempt_id="a", user_id="u", candidate_name="c").to_dict(),
    ]
    for payload in payloads:
        for key in payload:
            assert not any(word in key.lower() for word in banned), f"suspicious field: {key}"


def test_frame_ack_paces_the_client():
    """The client follows server pacing, so the interval must always be positive."""
    ack = FrameAck(
        session_id="s",
        frame_index=0,
        timestamp_seconds=0.0,
        accepted=True,
        next_frame_due_in_seconds=0.5,
    )
    assert ack.to_dict()["next_frame_due_in_seconds"] > 0


def test_unmeasured_frame_fields_stay_null():
    """A stage that did not run must serialise as null, never as a neutral value."""
    encoded = FrameAck(
        session_id="s", frame_index=0, timestamp_seconds=0.0, accepted=True
    ).to_dict()
    for field in (
        "face_count",
        "face_status",
        "identity_verified",
        "hands_detected",
        "is_speaking",
        "is_looking_away",
    ):
        assert encoded[field] is None


def test_weak_signals_all_carry_a_reliability_note():
    """Every observation the guide rates low-confidence must warn inline."""
    for event_type in (
        EventType.EARBUDS_SUSPECTED,
        EventType.CANDIDATE_SPEAKING,
        EventType.LOOKING_AWAY,
        EventType.GAZE_OFF_SCREEN,
        EventType.HANDS_NOT_VISIBLE,
        EventType.HAND_NEAR_FACE,
    ):
        note = RELIABILITY_NOTES.get(event_type.value)
        assert note, f"{event_type.value} has no reliability note"
        assert len(note) > 40, f"{event_type.value} note is too terse to be useful"


def test_strong_signals_do_not_carry_a_caveat():
    """Over-warning is its own failure: a caveat on everything is a caveat on nothing."""
    for event_type in (EventType.NO_FACE, EventType.MULTIPLE_FACES, EventType.PHONE_DETECTED):
        assert event_type.value not in RELIABILITY_NOTES


def test_reviewer_guidance_frames_the_evidence():
    """The review screen must state that these are observations, not findings."""
    joined = " ".join(DEFAULT_REVIEWER_GUIDANCE).lower()
    assert "not findings" in joined or "observations, not" in joined
    assert "score" in joined, "guidance must say no score is produced"
    assert "unmodified" in joined, "guidance must distinguish source from annotated evidence"


def test_observation_summary_marks_derived_evidence():
    summary = ObservationSummary(
        event_id="e",
        event_type="NO_FACE",
        severity="HIGH",
        status="VALIDATED",
        start_timestamp=1.0,
        end_timestamp=2.0,
        formatted_start="a",
        formatted_end="b",
        duration_seconds=1.0,
        description="d",
        qualified=True,
        confidence=1.0,
        evidence=[
            {"media_type": "frame", "is_derived": False},
            {"media_type": "review", "is_derived": True},
        ],
    )
    encoded = summary.to_dict()
    assert encoded["evidence"][0]["is_derived"] is False
    assert encoded["evidence"][1]["is_derived"] is True


def test_session_result_reports_coverage_gaps():
    """Absence of observations must never be presentable as absence of events."""
    result = SessionResult(
        session_id="s",
        attempt_id="a",
        user_id="u",
        state=SessionState.COMPLETED,
        coverage_warnings=["Not observed this session: wearable_detection"],
    )
    assert result.to_dict()["coverage_warnings"]
