"""Versioned data contracts between the proctoring engine and a host LMS.

These are the only shapes a Moodle plugin (or any other consumer) should depend
on.  Everything inside ``src/proctoring`` is free to change; this module is not,
except by bumping :data:`CONTRACT_VERSION`.

The contract is deliberately plain: dataclasses that serialise to JSON with no
numpy arrays, no file handles and no engine objects.  A Moodle
``quizaccess_proctoring`` plugin speaks to this over whatever transport suits —
web service, REST endpoint, or a queue — without importing the pipeline.

Two rules shape every payload here:

* **No scores.** Nothing in the contract ranks or grades a candidate.  The host
  receives observations and evidence; the academic judgement is the invigilator's.
* **Identity data stays server-side.** Face embeddings never cross this boundary.
  A session references its enrolment by identifier only.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

CONTRACT_VERSION = "1.0"


class SessionState(str, Enum):
    """Lifecycle of a proctored attempt, mirroring a Moodle quiz attempt."""

    CREATED = "CREATED"  # Registered, no frames yet
    ENROLLING = "ENROLLING"  # Capturing the candidate's reference face
    ACTIVE = "ACTIVE"  # Attempt in progress, frames arriving
    PAUSED = "PAUSED"  # Temporarily suspended; frames are rejected, not inferred
    FINALIZING = "FINALIZING"  # Attempt submitted, package being sealed
    COMPLETED = "COMPLETED"  # Package sealed and verified
    FAILED = "FAILED"  # Aborted; any partial package is marked as such


@dataclass
class StartSessionRequest:
    """Open a proctoring session for one quiz attempt.

    ``attempt_id`` and ``user_id`` are the host's own identifiers, echoed back on
    every response so the plugin can correlate without keeping its own map.
    """

    attempt_id: str
    user_id: str
    course_id: str | None = None
    quiz_id: str | None = None
    candidate_name: str = "Candidate"
    strictness: str = "STANDARD"
    sampling_fps: float = 4.0
    enable_wearable_detection: bool = False
    enrolment_id: str | None = None
    """Reference to a previously stored enrolment template, if the candidate has one."""

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "user_id": self.user_id,
            "course_id": self.course_id,
            "quiz_id": self.quiz_id,
            "candidate_name": self.candidate_name,
            "strictness": self.strictness,
            "sampling_fps": self.sampling_fps,
            "enable_wearable_detection": self.enable_wearable_detection,
            "enrolment_id": self.enrolment_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StartSessionRequest":
        return cls(
            attempt_id=str(data["attempt_id"]),
            user_id=str(data["user_id"]),
            course_id=data.get("course_id"),
            quiz_id=data.get("quiz_id"),
            candidate_name=data.get("candidate_name", "Candidate"),
            strictness=data.get("strictness", "STANDARD"),
            sampling_fps=float(data.get("sampling_fps", 4.0)),
            enable_wearable_detection=bool(data.get("enable_wearable_detection", False)),
            enrolment_id=data.get("enrolment_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SessionHandle:
    """Identifies an open session and reports what it is actually able to observe.

    ``active_detectors`` and ``unavailable_detectors`` matter: a deployment missing
    the MediaPipe models silently loses speech and hand analysis, and the host must
    be able to tell a proctor that rather than implying full coverage.
    """

    session_id: str
    attempt_id: str
    user_id: str
    state: SessionState
    contract_version: str = CONTRACT_VERSION
    started_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    strictness: str = "STANDARD"
    active_detectors: list[str] = field(default_factory=list)
    unavailable_detectors: list[str] = field(default_factory=list)
    identity_verification_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "attempt_id": self.attempt_id,
            "user_id": self.user_id,
            "state": self.state.value,
            "contract_version": self.contract_version,
            "started_at_utc": self.started_at_utc,
            "strictness": self.strictness,
            "active_detectors": self.active_detectors,
            "unavailable_detectors": self.unavailable_detectors,
            "identity_verification_enabled": self.identity_verification_enabled,
        }


@dataclass
class FrameAck:
    """Response to one ingested frame — the live state a proctor dashboard renders."""

    session_id: str
    frame_index: int
    timestamp_seconds: float
    accepted: bool
    rejection_reason: str | None = None

    face_count: int | None = None
    face_status: str | None = None
    identity_verified: bool | None = None

    hands_detected: int | None = None
    is_speaking: bool | None = None
    is_looking_away: bool | None = None
    detected_objects: list[str] = field(default_factory=list)
    detected_wearables: list[str] = field(default_factory=list)

    active_observations: list[str] = field(default_factory=list)
    """Event types currently open. Not a verdict — conditions being watched."""

    engine_state: str | None = None
    """The engine's processing state for this frame (RUNNING, PAUSED, ...). Lets a
    client tell a rejected-because-paused frame from a rejected-because-corrupt one."""

    processing_latency_ms: float = 0.0
    next_frame_due_in_seconds: float = 0.25
    """How long the client should wait before sending the next frame; follows the
    engine's adaptive sampling so a quiet session costs less bandwidth and CPU."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "face_count": self.face_count,
            "face_status": self.face_status,
            "identity_verified": self.identity_verified,
            "hands_detected": self.hands_detected,
            "is_speaking": self.is_speaking,
            "is_looking_away": self.is_looking_away,
            "detected_objects": self.detected_objects,
            "detected_wearables": self.detected_wearables,
            "active_observations": self.active_observations,
            "engine_state": self.engine_state,
            "processing_latency_ms": round(self.processing_latency_ms, 2),
            "next_frame_due_in_seconds": round(self.next_frame_due_in_seconds, 3),
        }


@dataclass
class ObservationSummary:
    """One qualified observation, as the review UI lists it."""

    event_id: str
    event_type: str
    severity: str
    status: str
    start_timestamp: float
    end_timestamp: float
    formatted_start: str
    formatted_end: str
    duration_seconds: float
    description: str
    qualified: bool
    confidence: float

    category: str = "CANDIDATE_OBSERVATION"
    """``CANDIDATE_OBSERVATION`` or ``TECHNICAL_DIAGNOSTIC``. A host must never
    present a technical diagnostic as something the candidate did."""

    reliability_note: str | None = None
    """Set for observations a proctor must treat with extra caution — see
    :data:`RELIABILITY_NOTES`."""

    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "status": self.status,
            "category": self.category,
            "is_technical": self.category == "TECHNICAL_DIAGNOSTIC",
            "start_timestamp": round(self.start_timestamp, 3),
            "end_timestamp": round(self.end_timestamp, 3),
            "formatted_start": self.formatted_start,
            "formatted_end": self.formatted_end,
            "duration_seconds": round(self.duration_seconds, 3),
            "description": self.description,
            "qualified": self.qualified,
            "confidence": round(self.confidence, 4),
            "reliability_note": self.reliability_note,
            "evidence": self.evidence,
        }


# Caveats attached to observations whose underlying signal is weak. Surfaced in the
# review payload so a proctor sees the limitation next to the finding, not buried
# in documentation they will not read.
RELIABILITY_NOTES: dict[str, str] = {
    "EARBUDS_SUSPECTED": (
        "In-ear devices are only a few dozen pixels at webcam distance and are easily "
        "confused with earrings, hair or ear shadow. Treat as a prompt to look at the "
        "snapshot, never as a finding on its own."
    ),
    "CANDIDATE_SPEAKING": (
        "Possible talking, detected from mouth movement only; there is no audio and "
        "nothing here shows that the candidate spoke. Reading aloud, muttering while "
        "thinking, chewing and permitted accommodations can all register."
    ),
    "LOOKING_AWAY": (
        "Head orientation relative to the camera. A camera mounted off-centre, or a "
        "second monitor, produces sustained readings with no misconduct involved."
    ),
    "SUSPICIOUS_HEAD_POSE": (
        "A pattern of repeated or rapid head movement, not a single turn. Fidgeting, "
        "a noisy room, and a candidate working between a screen and a keyboard all "
        "produce it. Watch the stretch of session it points at before drawing any "
        "conclusion from it."
    ),
    "GAZE_OFF_SCREEN": (
        "Estimated from iris position without per-candidate calibration. Indicative only."
    ),
    "HANDS_NOT_VISIBLE": (
        "Depends heavily on camera framing. A laptop camera angled at the face may never "
        "see the candidate's hands at all."
    ),
    "HAND_NEAR_FACE": ("Resting a head on a hand is a common, innocent posture."),
}


@dataclass
class SessionResult:
    """Outcome handed back when an attempt is finalised."""

    session_id: str
    attempt_id: str
    user_id: str
    state: SessionState
    contract_version: str = CONTRACT_VERSION

    started_at_utc: str = ""
    ended_at_utc: str = ""
    duration_seconds: float = 0.0

    frames_sampled: int = 0
    frames_processed: int = 0
    frames_rejected: int = 0

    total_observations: int = 0
    qualified_observations: int = 0
    candidate_observations: int = 0
    technical_diagnostics: int = 0
    """Equipment faults recorded during the session. Never misconduct — reported
    separately so a host cannot total them together with candidate observations."""
    observations_by_type: dict[str, int] = field(default_factory=dict)

    package_dir: str = ""
    package_archive: str | None = None
    manifest_sha256: str | None = None
    integrity_verified: bool = False
    integrity_errors: list[str] = field(default_factory=list)

    coverage_warnings: list[str] = field(default_factory=list)
    """Anything that limited what this session could observe — a detector that was
    unavailable, a camera that dropped out. The host should show these to the
    proctor so absence of evidence is not read as evidence of absence."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "attempt_id": self.attempt_id,
            "user_id": self.user_id,
            "state": self.state.value,
            "contract_version": self.contract_version,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "duration_seconds": round(self.duration_seconds, 2),
            "frames_sampled": self.frames_sampled,
            "frames_processed": self.frames_processed,
            "frames_rejected": self.frames_rejected,
            "total_observations": self.total_observations,
            "qualified_observations": self.qualified_observations,
            "candidate_observations": self.candidate_observations,
            "technical_diagnostics": self.technical_diagnostics,
            "observations_by_type": self.observations_by_type,
            "package_dir": self.package_dir,
            "package_archive": self.package_archive,
            "manifest_sha256": self.manifest_sha256,
            "integrity_verified": self.integrity_verified,
            "integrity_errors": self.integrity_errors,
            "coverage_warnings": self.coverage_warnings,
        }


@dataclass
class ReviewPayload:
    """Everything an invigilator's review screen needs for one attempt."""

    session_id: str
    attempt_id: str
    user_id: str
    candidate_name: str
    contract_version: str = CONTRACT_VERSION

    result: dict[str, Any] | None = None
    observations: list[ObservationSummary] = field(default_factory=list)
    timeline_summary: dict[str, Any] = field(default_factory=dict)
    telemetry_summary: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)

    reviewer_guidance: list[str] = field(default_factory=list)
    """Plain-language notes framing how the evidence should be read."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "attempt_id": self.attempt_id,
            "user_id": self.user_id,
            "candidate_name": self.candidate_name,
            "contract_version": self.contract_version,
            "result": self.result,
            "observations": [o.to_dict() for o in self.observations],
            "timeline_summary": self.timeline_summary,
            "telemetry_summary": self.telemetry_summary,
            "policy": self.policy,
            "reviewer_guidance": self.reviewer_guidance,
        }


# Framing shown at the top of every review screen. Kept in the contract so a host
# cannot present the evidence without the caveats that make it fair to read.
DEFAULT_REVIEWER_GUIDANCE: list[str] = [
    "These are observations, not findings. The system does not decide whether "
    "misconduct occurred and produces no risk or suspicion score.",
    "Every observation has innocent explanations. Review the snapshot and the "
    "surrounding timeline before drawing any conclusion.",
    "Detection quality depends on camera, lighting and framing. Check the coverage "
    "warnings before treating an absence of observations as an absence of events.",
    "Source frames under evidence/frames/ are unmodified. Images under "
    "evidence/review/ are annotated copies produced by the system.",
]
