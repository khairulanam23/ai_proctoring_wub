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
    """Lifecycle of a proctored attempt, mirroring a host examination platform."""

    CREATED = "CREATED"  # Registered, no frames yet
    ENROLLING = "ENROLLING"  # Capturing the candidate's reference face
    ACTIVE = "ACTIVE"  # Attempt in progress, frames arriving
    PAUSED = "PAUSED"  # Temporarily suspended; frames are rejected, not inferred
    FINALIZING = "FINALIZING"  # Attempt submitted, package being sealed
    COMPLETED = "COMPLETED"  # Package sealed and verified
    FAILED = "FAILED"  # Aborted; any partial package is marked as such


class SyncStatus(str, Enum):
    """Synchronization status between engine and remote Exam Controller."""

    LOCAL_DURABLE = "LOCAL_DURABLE"
    PENDING_SYNC = "PENDING_SYNC"
    SYNCING = "SYNCING"
    SYNCHRONIZED = "SYNCHRONIZED"
    SYNC_FAILED = "SYNC_FAILED"


@dataclass
class StartSessionRequest:
    """Open a proctoring session for one exam candidate.

    Supports platform-neutral concepts (exam_id, candidate_id, session_id,
    organization_id) while preserving legacy LMS aliases (attempt_id, user_id,
    quiz_id, course_id).
    """

    attempt_id: str | None = None
    user_id: str | None = None
    exam_id: str | None = None
    candidate_id: str | None = None
    session_id: str | None = None
    organization_id: str | None = None
    course_id: str | None = None
    quiz_id: str | None = None
    candidate_name: str = "Candidate"
    strictness: str = "STANDARD"
    sampling_fps: float = 4.0
    enable_wearable_detection: bool = False
    enrolment_id: str | None = None
    reference_templates: list[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.attempt_id is not None:
            self.attempt_id = str(self.attempt_id)
        if self.session_id is not None:
            self.session_id = str(self.session_id)
        if self.user_id is not None:
            self.user_id = str(self.user_id)
        if self.candidate_id is not None:
            self.candidate_id = str(self.candidate_id)
        if self.quiz_id is not None:
            self.quiz_id = str(self.quiz_id)
        if self.exam_id is not None:
            self.exam_id = str(self.exam_id)

        if not self.attempt_id and self.session_id:
            self.attempt_id = self.session_id
        if not self.session_id and self.attempt_id:
            self.session_id = self.attempt_id
        if not self.session_id:
            self.session_id = "sess_" + str(int(datetime.now(timezone.utc).timestamp()))
            self.attempt_id = self.session_id

        if not self.user_id and self.candidate_id:
            self.user_id = self.candidate_id
        if not self.candidate_id and self.user_id:
            self.candidate_id = self.user_id
        if not self.candidate_id:
            self.candidate_id = "default_candidate"
            self.user_id = self.candidate_id

        if not self.quiz_id and self.exam_id:
            self.quiz_id = self.exam_id
        if not self.exam_id and self.quiz_id:
            self.exam_id = self.quiz_id
        if not self.exam_id:
            self.exam_id = "default_exam"
            self.quiz_id = self.exam_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "attempt_id": self.attempt_id,
            "candidate_id": self.candidate_id,
            "user_id": self.user_id,
            "exam_id": self.exam_id,
            "quiz_id": self.quiz_id,
            "course_id": self.course_id,
            "organization_id": self.organization_id,
            "candidate_name": self.candidate_name,
            "strictness": self.strictness,
            "sampling_fps": self.sampling_fps,
            "enable_wearable_detection": self.enable_wearable_detection,
            "enrolment_id": self.enrolment_id,
            "reference_templates": self.reference_templates,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StartSessionRequest":
        return cls(
            session_id=data.get("session_id") or data.get("attempt_id"),
            attempt_id=data.get("attempt_id") or data.get("session_id"),
            candidate_id=data.get("candidate_id") or data.get("user_id"),
            user_id=data.get("user_id") or data.get("candidate_id"),
            exam_id=data.get("exam_id") or data.get("quiz_id"),
            quiz_id=data.get("quiz_id") or data.get("exam_id"),
            course_id=data.get("course_id"),
            organization_id=data.get("organization_id"),
            candidate_name=data.get("candidate_name", "Candidate"),
            strictness=data.get("strictness", "STANDARD"),
            sampling_fps=float(data.get("sampling_fps", 4.0)),
            enable_wearable_detection=bool(data.get("enable_wearable_detection", False)),
            enrolment_id=data.get("enrolment_id"),
            reference_templates=data.get("reference_templates"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SessionHandle:
    """Identifies an open session and reports active capabilities."""

    session_id: str
    attempt_id: str
    user_id: str
    state: SessionState
    exam_id: str = "default_exam"
    candidate_id: str = "default_candidate"
    organization_id: str | None = None
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
            "exam_id": self.exam_id,
            "candidate_id": self.candidate_id,
            "organization_id": self.organization_id,
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

    active_incidents: list[str] = field(default_factory=list)
    """All qualified conditions currently active on screen."""

    emitted_alerts: list[dict[str, Any]] = field(default_factory=list)
    """Structured alerts emitted on this specific frame under repeat cooldown rules."""

    engine_state: str | None = None
    """The engine's processing state for this frame (RUNNING, PAUSED, ...). Lets a
    client tell a rejected-because-paused frame from a rejected-because-corrupt one."""

    processing_latency_ms: float = 0.0
    next_frame_due_in_seconds: float = 0.25
    """How long the client should wait before sending the next frame; follows the
    engine's adaptive sampling so a quiet session costs less bandwidth and CPU."""

    evidence_id: str | None = None
    evidence_sha256: str | None = None
    evidence_file_path: str | None = None

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
            "active_incidents": self.active_incidents,
            "emitted_alerts": self.emitted_alerts,
            "engine_state": self.engine_state,
            "processing_latency_ms": round(self.processing_latency_ms, 2),
            "next_frame_due_in_seconds": round(self.next_frame_due_in_seconds, 3),
            "evidence_id": self.evidence_id,
            "evidence_sha256": self.evidence_sha256,
            "evidence_file_path": self.evidence_file_path,
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

    sequence_number: int = 0
    engine_version: str = "1.0.0"
    sync_status: str = "LOCAL_DURABLE"

    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "status": self.status,
            "category": self.category,
            "is_technical": self.category == "TECHNICAL_DIAGNOSTIC",
            "sequence_number": self.sequence_number,
            "engine_version": self.engine_version,
            "sync_status": self.sync_status,
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


@dataclass
class OutboxEventRecord:
    """An event held in the offline synchronization outbox."""

    event_id: str
    session_id: str
    sequence_number: int
    event_type: str
    timestamp: float
    created_at_utc: str
    payload: dict[str, Any]
    sync_status: SyncStatus = SyncStatus.LOCAL_DURABLE
    retry_count: int = 0
    last_error: str | None = None
    idempotency_key: str = ""

    def __post_init__(self) -> None:
        if not self.idempotency_key:
            self.idempotency_key = f"{self.session_id}:{self.sequence_number}:{self.event_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "sequence_number": self.sequence_number,
            "event_type": self.event_type,
            "timestamp": round(self.timestamp, 3),
            "created_at_utc": self.created_at_utc,
            "payload": self.payload,
            "sync_status": self.sync_status.value if hasattr(self.sync_status, "value") else str(self.sync_status),
            "retry_count": self.retry_count,
            "last_error": self.last_error,
            "idempotency_key": self.idempotency_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutboxEventRecord":
        return cls(
            event_id=data["event_id"],
            session_id=data["session_id"],
            sequence_number=int(data["sequence_number"]),
            event_type=data["event_type"],
            timestamp=float(data["timestamp"]),
            created_at_utc=data.get("created_at_utc", ""),
            payload=data.get("payload", {}),
            sync_status=SyncStatus(data.get("sync_status", SyncStatus.LOCAL_DURABLE.value)),
            retry_count=int(data.get("retry_count", 0)),
            last_error=data.get("last_error"),
            idempotency_key=data.get("idempotency_key", ""),
        )


@dataclass
class SyncBatchAck:
    """Response acknowledging idempotent ingestion of events by remote endpoint."""

    session_id: str
    synced_sequence_numbers: list[int]
    duplicate_sequence_numbers: list[int]
    failed_sequence_numbers: list[int] = field(default_factory=list)
    server_timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "synced_sequence_numbers": self.synced_sequence_numbers,
            "duplicate_sequence_numbers": self.duplicate_sequence_numbers,
            "failed_sequence_numbers": self.failed_sequence_numbers,
            "server_timestamp_utc": self.server_timestamp_utc,
        }


@dataclass
class HealthResponse:
    """Engine health, readiness, and model presence status."""

    status: str
    engine_version: str
    models_loaded: dict[str, bool]
    active_sessions_count: int
    hardware: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status,
            "engine_version": self.engine_version,
            "models_loaded": self.models_loaded,
            "active_sessions_count": self.active_sessions_count,
            "timestamp_utc": self.timestamp_utc,
        }
        if self.hardware:
            result["hardware"] = self.hardware
        return result


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


@dataclass
class IncidentReviewRecord:
    """Human review record for an AI observation incident."""

    observation_id: str
    session_id: str
    original_event_type: str
    original_ai_label: str
    original_detector: str
    original_detector_version: str
    original_confidence: float
    original_timestamp: str
    review_status: str  # "PENDING", "CONFIRMED", "CORRECTED", "REJECTED", "UNCERTAIN"
    reviewed_label: str | None = None
    reviewer_id: str | None = None
    reviewed_at_utc: str | None = None
    review_notes: str | None = None
    bbox: list[int] | None = None
    evidence_ref: str | None = None
    image_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "session_id": self.session_id,
            "original_event_type": self.original_event_type,
            "original_ai_label": self.original_ai_label,
            "original_detector": self.original_detector,
            "original_detector_version": self.original_detector_version,
            "original_confidence": self.original_confidence,
            "original_timestamp": self.original_timestamp,
            "review_status": self.review_status,
            "reviewed_label": self.reviewed_label,
            "reviewer_id": self.reviewer_id,
            "reviewed_at_utc": self.reviewed_at_utc,
            "review_notes": self.review_notes,
            "bbox": self.bbox,
            "evidence_ref": self.evidence_ref,
            "image_sha256": self.image_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IncidentReviewRecord":
        return cls(
            observation_id=data["observation_id"],
            session_id=data["session_id"],
            original_event_type=data["original_event_type"],
            original_ai_label=data.get("original_ai_label", data["original_event_type"]),
            original_detector=data.get("original_detector", "unknown"),
            original_detector_version=data.get("original_detector_version", "unknown"),
            original_confidence=float(data.get("original_confidence", 0.0)),
            original_timestamp=data.get("original_timestamp", ""),
            review_status=data.get("review_status", "PENDING"),
            reviewed_label=data.get("reviewed_label"),
            reviewer_id=data.get("reviewer_id"),
            reviewed_at_utc=data.get("reviewed_at_utc"),
            review_notes=data.get("review_notes"),
            bbox=data.get("bbox"),
            evidence_ref=data.get("evidence_ref"),
            image_sha256=data.get("image_sha256"),
        )

