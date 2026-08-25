"""Event lifecycle state machine tracking transitions from raw observations to validated evidence records."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from proctoring.core.events import EventType


class EventLifecycleState(str, Enum):
    """Formal states in the proctoring event validation lifecycle."""

    OBSERVED = "OBSERVED"  # Single raw frame detection
    CANDIDATE = "CANDIDATE"  # Accumulating evidence toward qualification
    VALIDATED = "VALIDATED"  # Formally passed temporal/confidence criteria
    EVIDENCE_CAPTURED = "EVIDENCE_CAPTURED"  # Visual artifacts validated and stored on disk
    CLOSED = "CLOSED"  # Incident resolved (condition ended)
    DISCARDED = "DISCARDED"  # Transient anomaly dropped (false positive suppressed)


@dataclass
class ValidationConfig:
    """Configurable temporal and frame thresholds per event category."""

    min_consecutive_frames: dict[str, int] = field(
        default_factory=lambda: {
            EventType.NO_FACE.value: 3,
            EventType.MULTIPLE_FACES.value: 2,
            EventType.UNKNOWN_FACE.value: 3,
            EventType.PHONE_DETECTED.value: 1,  # Immediate capture for critical items
            EventType.PROHIBITED_OBJECT.value: 2,
            EventType.BROWSER_FULLSCREEN_EXIT.value: 1,
            EventType.BROWSER_TAB_SWITCH.value: 1,
        }
    )
    min_duration_seconds: dict[str, float] = field(
        default_factory=lambda: {
            EventType.NO_FACE.value: 1.0,
            EventType.MULTIPLE_FACES.value: 0.5,
            EventType.UNKNOWN_FACE.value: 1.0,
            EventType.PHONE_DETECTED.value: 0.25,
            EventType.PROHIBITED_OBJECT.value: 0.5,
            EventType.BROWSER_FULLSCREEN_EXIT.value: 0.0,
            EventType.BROWSER_TAB_SWITCH.value: 0.0,
        }
    )
    absence_tolerance_seconds: float = 0.5
    max_candidate_idle_seconds: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_consecutive_frames": self.min_consecutive_frames,
            "min_duration_seconds": self.min_duration_seconds,
            "absence_tolerance_seconds": self.absence_tolerance_seconds,
            "max_candidate_idle_seconds": self.max_candidate_idle_seconds,
        }


@dataclass
class CandidateObservation:
    """Lightweight single-frame observation record."""

    timestamp: float
    frame_index: int
    confidence: float
    bbox: tuple[int, int, int, int] | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "frame_index": self.frame_index,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "description": self.description,
        }


@dataclass
class CandidateEventRecord:
    """State-tracked candidate incident accumulating observations."""

    candidate_id: str
    session_id: str
    event_type: EventType
    state: EventLifecycleState = EventLifecycleState.OBSERVED
    start_timestamp: float = 0.0
    last_seen_timestamp: float = 0.0
    observations: list[CandidateObservation] = field(default_factory=list)
    validation_reason: str = ""
    evidence_references: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen_timestamp - self.start_timestamp)

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def average_confidence(self) -> float:
        if not self.observations:
            return 0.0
        return float(np.mean([o.confidence for o in self.observations]))

    def add_observation(self, obs: CandidateObservation) -> None:
        """Append an observation and update temporal bounds."""
        if not self.observations:
            self.start_timestamp = obs.timestamp
            self.state = EventLifecycleState.OBSERVED
        else:
            if self.state == EventLifecycleState.OBSERVED:
                self.state = EventLifecycleState.CANDIDATE

        self.last_seen_timestamp = obs.timestamp
        self.observations.append(obs)

    def evaluate_state(self, current_time: float, config: ValidationConfig) -> EventLifecycleState:
        """Evaluate if the candidate event qualifies for VALIDATED or DISCARDED."""
        ev_key = self.event_type.value
        req_frames = config.min_consecutive_frames.get(ev_key, 2)
        req_dur = config.min_duration_seconds.get(ev_key, 0.5)

        # Check if idle threshold exceeded
        time_since_last = current_time - self.last_seen_timestamp
        if time_since_last > config.absence_tolerance_seconds:
            if self.state in (EventLifecycleState.OBSERVED, EventLifecycleState.CANDIDATE):
                # Did not meet validation criteria before expiring -> DISCARD
                if self.observation_count < req_frames and self.duration < req_dur:
                    self.state = EventLifecycleState.DISCARDED
                    self.validation_reason = (
                        f"Transient observation ({self.observation_count} frames, "
                        f"{self.duration:.2f}s) discarded below threshold ({req_frames} frames / {req_dur}s)."
                    )
                    return self.state
            elif self.state in (
                EventLifecycleState.VALIDATED,
                EventLifecycleState.EVIDENCE_CAPTURED,
            ):
                self.state = EventLifecycleState.CLOSED
                return self.state

        # Check if candidate meets validation criteria
        if self.state in (EventLifecycleState.OBSERVED, EventLifecycleState.CANDIDATE) and (
            self.observation_count >= req_frames
            or (req_dur > 0 and self.duration >= req_dur)
            or req_dur == 0
        ):
            self.state = EventLifecycleState.VALIDATED
            self.validation_reason = (
                f"Validated after {self.observation_count} frames ({self.duration:.2f}s duration). "
                f"Avg confidence: {self.average_confidence:.2f}."
            )

        return self.state

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "session_id": self.session_id,
            "event_type": self.event_type.value,
            "state": self.state.value,
            "start_timestamp": round(self.start_timestamp, 3),
            "last_seen_timestamp": round(self.last_seen_timestamp, 3),
            "duration_seconds": round(self.duration, 3),
            "observation_count": self.observation_count,
            "average_confidence": round(self.average_confidence, 4),
            "validation_reason": self.validation_reason,
            "evidence_references": self.evidence_references,
        }
