"""Unified event and evidence schema for AI proctoring observation records."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Supported proctoring event classifications."""

    # Face presence & identity events
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    UNKNOWN_FACE = "UNKNOWN_FACE"
    FACE_MISMATCH = "FACE_MISMATCH"
    POSSIBLE_PRESENTATION_ATTACK = "POSSIBLE_PRESENTATION_ATTACK"  # face never blinks
    PERSON_ENTERED_FRAME = "PERSON_ENTERED_FRAME"
    PERSON_LEFT_FRAME = "PERSON_LEFT_FRAME"

    # Object detection events
    PHONE_DETECTED = "PHONE_DETECTED"
    PROHIBITED_OBJECT = "PROHIBITED_OBJECT"

    # Browser / Application monitoring events
    BROWSER_TAB_SWITCH = "BROWSER_TAB_SWITCH"
    BROWSER_FULLSCREEN_EXIT = "BROWSER_FULLSCREEN_EXIT"
    BROWSER_WINDOW_BLUR = "BROWSER_WINDOW_BLUR"
    BROWSER_COPY_PASTE = "BROWSER_COPY_PASTE"

    # Head pose, gaze & attention events
    LOOKING_AWAY = "LOOKING_AWAY"
    SUSPICIOUS_HEAD_POSE = "SUSPICIOUS_HEAD_POSE"
    GAZE_OFF_SCREEN = "GAZE_OFF_SCREEN"

    # Hand behaviour events
    HAND_NEAR_FACE = "HAND_NEAR_FACE"  # A hand entered the face region
    HAND_NEAR_EAR = "HAND_NEAR_EAR"  # Hand at the ear - consistent with an earpiece or a call
    HANDS_NOT_VISIBLE = "HANDS_NOT_VISIBLE"  # No hand visible while the candidate is present

    # Speech / vocalisation events
    CANDIDATE_SPEAKING = "CANDIDATE_SPEAKING"  # Sustained mouth articulation consistent with speech

    # Wearable device events
    HEADPHONES_DETECTED = "HEADPHONES_DETECTED"
    EARBUDS_SUSPECTED = "EARBUDS_SUSPECTED"  # Low-confidence by nature - see the accuracy guide
    SMARTWATCH_DETECTED = "SMARTWATCH_DETECTED"

    # System and failure diagnostics
    SYSTEM_ERROR = "SYSTEM_ERROR"
    DETECTOR_ERROR = "DETECTOR_ERROR"
    OTHER_SUSPICIOUS_ACTIVITY = "OTHER_SUSPICIOUS_ACTIVITY"


class EventSeverity(str, Enum):
    """Categorical event severity for proctor review triage (NOT a risk score)."""

    INFO = "INFO"  # Normal state transition or diagnostic
    LOW = "LOW"  # Minor anomaly (e.g. momentary occlusion)
    MEDIUM = "MEDIUM"  # Moderate event (e.g. book detected, looking away)
    HIGH = "HIGH"  # Significant observation (e.g. phone detected, multiple faces)
    CRITICAL = (
        "CRITICAL"  # Major observation (e.g. unknown person during entire session, tab switch)
    )


class EventStatus(str, Enum):
    """Lifecycle status of an observation event."""

    RECORDED = "RECORDED"  # Raw observation recorded
    QUALIFIED = "QUALIFIED"  # Passed duration qualification threshold
    EVIDENCE_CAPTURED = "EVIDENCE_CAPTURED"  # Visual evidence captured on disk
    EVIDENCE_FAILED = "EVIDENCE_FAILED"  # Evidence capture was attempted but failed
    VALIDATED = "VALIDATED"  # Evidence verified on disk with valid checksums


def format_seconds_to_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm string."""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


@dataclass
class DetectorInfo:
    """Metadata identifying the detector model and configuration."""

    name: str
    version: str = "1.0"
    model_file: str = ""
    device: str = "cpu"
    confidence_threshold: float = 0.0
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "model_file": self.model_file,
            "device": self.device,
            "confidence_threshold": round(self.confidence_threshold, 4),
            "extra_params": self.extra_params,
        }


@dataclass
class ObservationDetail:
    """Factual, descriptive observations recorded by detectors."""

    description: str
    object_class: str | None = None
    face_count: int | None = None
    similarity_score: float | None = None
    bounding_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)  # [(x1,y1,x2,y2)]
    frame_indices: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    raw_confidences: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "object_class": self.object_class,
            "face_count": self.face_count,
            "similarity_score": round(self.similarity_score, 4)
            if self.similarity_score is not None
            else None,
            "bounding_boxes": [list(b) for b in self.bounding_boxes],
            "frame_indices": self.frame_indices,
            "timestamps": [round(t, 3) for t in self.timestamps],
            "raw_confidences": [round(c, 4) for c in self.raw_confidences],
        }


@dataclass
class EvidenceReference:
    """Reference to visual evidence artifacts stored on disk."""

    evidence_id: str
    media_type: str  # "frame", "crop", "clip", "browser_log"
    file_path: str  # Relative path within session evidence package
    timestamp_seconds: float
    formatted_timestamp: str
    frame_index: int
    bbox: tuple[int, int, int, int] | None = None
    sha256: str | None = None
    file_size_bytes: int = 0
    is_validated: bool = False
    validation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "media_type": self.media_type,
            "file_path": self.file_path,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "formatted_timestamp": self.formatted_timestamp,
            "frame_index": self.frame_index,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "sha256": self.sha256,
            "file_size_bytes": self.file_size_bytes,
            "is_validated": self.is_validated,
            "validation_error": self.validation_error,
        }


@dataclass
class EventRecord:
    """Unified, extensible proctoring observation event record."""

    event_id: str
    session_id: str
    timestamp: float
    end_timestamp: float
    duration: float
    formatted_start: str
    formatted_end: str
    event_type: EventType
    severity: EventSeverity
    confidence: float
    average_confidence: float
    detector: DetectorInfo
    observation: ObservationDetail
    evidence: list[EvidenceReference] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: EventStatus = EventStatus.RECORDED
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if isinstance(self.event_type, str) and not isinstance(self.event_type, EventType):
            self.event_type = EventType(self.event_type)
        if isinstance(self.severity, str) and not isinstance(self.severity, EventSeverity):
            self.severity = EventSeverity(self.severity)
        if isinstance(self.status, str) and not isinstance(self.status, EventStatus):
            self.status = EventStatus(self.status)

    def to_dict(self) -> dict[str, Any]:
        """Convert EventRecord to a structured JSON-serializable dictionary."""
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "timestamp": round(self.timestamp, 3),
            "end_timestamp": round(self.end_timestamp, 3),
            "duration": round(self.duration, 3),
            "formatted_start": self.formatted_start,
            "formatted_end": self.formatted_end,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "confidence": round(self.confidence, 4),
            "average_confidence": round(self.average_confidence, 4),
            "detector": self.detector.to_dict(),
            "observation": self.observation.to_dict(),
            "evidence": [ev.to_dict() for ev in self.evidence],
            "metadata": self.metadata,
            "status": self.status.value,
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventRecord":
        """Reconstruct EventRecord from dictionary."""
        det_data = data.get("detector", {})
        detector = DetectorInfo(
            name=det_data.get("name", "unknown"),
            version=det_data.get("version", "1.0"),
            model_file=det_data.get("model_file", ""),
            device=det_data.get("device", "cpu"),
            confidence_threshold=float(det_data.get("confidence_threshold", 0.0)),
            extra_params=det_data.get("extra_params", {}),
        )

        obs_data = data.get("observation", {})
        bboxes = [tuple(b) for b in obs_data.get("bounding_boxes", [])]
        observation = ObservationDetail(
            description=obs_data.get("description", ""),
            object_class=obs_data.get("object_class"),
            face_count=obs_data.get("face_count"),
            similarity_score=obs_data.get("similarity_score"),
            bounding_boxes=bboxes,
            frame_indices=obs_data.get("frame_indices", []),
            timestamps=obs_data.get("timestamps", []),
            raw_confidences=obs_data.get("raw_confidences", []),
        )

        evidence_list = []
        for ev in data.get("evidence", []):
            bbox = tuple(ev["bbox"]) if ev.get("bbox") is not None else None
            evidence_list.append(
                EvidenceReference(
                    evidence_id=ev["evidence_id"],
                    media_type=ev.get("media_type", "frame"),
                    file_path=ev["file_path"],
                    timestamp_seconds=float(ev["timestamp_seconds"]),
                    formatted_timestamp=ev.get("formatted_timestamp", ""),
                    frame_index=int(ev.get("frame_index", 0)),
                    bbox=bbox,
                    sha256=ev.get("sha256"),
                    file_size_bytes=int(ev.get("file_size_bytes", 0)),
                    is_validated=bool(ev.get("is_validated", False)),
                    validation_error=ev.get("validation_error"),
                )
            )

        return cls(
            event_id=data["event_id"],
            session_id=data["session_id"],
            timestamp=float(data["timestamp"]),
            end_timestamp=float(data["end_timestamp"]),
            duration=float(data["duration"]),
            formatted_start=data["formatted_start"],
            formatted_end=data["formatted_end"],
            event_type=EventType(data["event_type"]),
            severity=EventSeverity(data["severity"]),
            confidence=float(data["confidence"]),
            average_confidence=float(data["average_confidence"]),
            detector=detector,
            observation=observation,
            evidence=evidence_list,
            metadata=data.get("metadata", {}),
            status=EventStatus(data.get("status", EventStatus.RECORDED)),
            created_at_utc=data.get("created_at_utc", ""),
        )


def coerce_event_type(value: EventType | str) -> EventType:
    """Normalise a caller-supplied event type into a real :class:`EventType` member.

    Callers at the edge of the system (the browser bridge, CLI flags, JSON payloads)
    naturally pass plain strings.  Anything downstream that reads ``.value`` would
    raise ``AttributeError`` on a bare ``str``, so every external entry point funnels
    through here first.  Unrecognised names degrade to ``OTHER_SUSPICIOUS_ACTIVITY``
    rather than aborting the session: an unclassifiable observation is still worth
    showing the proctor.
    """
    if isinstance(value, EventType):
        return value
    try:
        return EventType(str(value).strip().upper())
    except ValueError:
        return EventType.OTHER_SUSPICIOUS_ACTIVITY


def coerce_severity(value: EventSeverity | str) -> EventSeverity:
    """Normalise a caller-supplied severity into a real :class:`EventSeverity` member."""
    if isinstance(value, EventSeverity):
        return value
    try:
        return EventSeverity(str(value).strip().upper())
    except ValueError:
        return EventSeverity.MEDIUM
