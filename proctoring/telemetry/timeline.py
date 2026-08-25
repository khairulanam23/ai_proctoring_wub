"""Chronological session timeline recording what was actually observed per frame.

The timeline is the workflow's audit trail between raw inference and the evidence
package: one row per sampled frame, carrying the real detector output for that
frame.  It is deliberately separate from ``events.json`` — events summarise
*qualified incidents*, whereas the timeline shows the continuous record a proctor
can scrub through to see the context around any event.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimelineEntry:
    """One sampled frame's observation, as measured — never inferred or defaulted.

    Every field here is populated from actual detector output.  When a stage did
    not run (disabled, or the frame was rejected by the quality gate) the value is
    ``None`` rather than a stand-in, so downstream consumers can distinguish
    "not measured" from "measured as zero".
    """

    frame_index: int
    timestamp_seconds: float
    iso_timestamp: str

    # Stage 3 — frame validation & preprocessing
    frame_accepted: bool = True
    rejection_reason: str | None = None
    was_enhanced: bool = False
    mean_luminance: float | None = None
    blur_variance: float | None = None

    # Stage 4 — face detection
    face_count: int | None = None
    face_boxes: list[list[int]] = field(default_factory=list)

    # Stage 5 — identity verification
    identity_verified: bool | None = None
    cosine_similarity: float | None = None

    # Stage 6 — scene / behavioural observation
    face_status: str | None = None
    prohibited_objects: list[str] = field(default_factory=list)

    # Stage 6b — behavioural analysis
    hands_detected: int | None = None
    hand_near_face: bool | None = None
    hand_near_ear: bool | None = None
    is_speaking: bool | None = None
    speech_activity: float | None = None
    head_yaw: float | None = None
    head_pitch: float | None = None
    is_looking_away: bool | None = None
    gaze_offset: float | None = None
    detected_wearables: list[str] = field(default_factory=list)

    # Stage 7 — temporal qualification state at this instant
    active_event_types: list[str] = field(default_factory=list)
    is_anomalous_state: bool = False

    # Telemetry
    processing_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "iso_timestamp": self.iso_timestamp,
            "frame_accepted": self.frame_accepted,
            "rejection_reason": self.rejection_reason,
            "was_enhanced": self.was_enhanced,
            "mean_luminance": round(self.mean_luminance, 2)
            if self.mean_luminance is not None
            else None,
            "blur_variance": round(self.blur_variance, 2)
            if self.blur_variance is not None
            else None,
            "face_count": self.face_count,
            "face_boxes": self.face_boxes,
            "identity_verified": self.identity_verified,
            "cosine_similarity": round(self.cosine_similarity, 4)
            if self.cosine_similarity is not None
            else None,
            "face_status": self.face_status,
            "prohibited_objects": self.prohibited_objects,
            "hands_detected": self.hands_detected,
            "hand_near_face": self.hand_near_face,
            "hand_near_ear": self.hand_near_ear,
            "is_speaking": self.is_speaking,
            "speech_activity": round(self.speech_activity, 4)
            if self.speech_activity is not None
            else None,
            "head_yaw": round(self.head_yaw, 2) if self.head_yaw is not None else None,
            "head_pitch": round(self.head_pitch, 2) if self.head_pitch is not None else None,
            "is_looking_away": self.is_looking_away,
            "gaze_offset": round(self.gaze_offset, 4) if self.gaze_offset is not None else None,
            "detected_wearables": self.detected_wearables,
            "active_event_types": self.active_event_types,
            "is_anomalous_state": self.is_anomalous_state,
            "processing_latency_ms": round(self.processing_latency_ms, 2),
        }


class SessionTimeline:
    """Accumulates :class:`TimelineEntry` rows and summarises session continuity."""

    def __init__(self, session_id: str = "default_session") -> None:
        self.session_id = str(session_id)
        self.entries: list[TimelineEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def append(self, entry: TimelineEntry) -> TimelineEntry:
        """Record one frame's observation."""
        self.entries.append(entry)
        return entry

    def to_list(self) -> list[dict[str, Any]]:
        """Serialise every row, in capture order."""
        return [e.to_dict() for e in self.entries]

    def summarise(self) -> dict[str, Any]:
        """Aggregate continuity statistics for the package manifest.

        ``anomalous_frame_ratio`` is the fraction of *accepted* frames in which at
        least one incident was open.  It is a coverage statistic for the proctor's
        benefit, not a suspicion score — the system never grades a candidate.
        """
        total = len(self.entries)
        accepted = [e for e in self.entries if e.frame_accepted]
        rejected = total - len(accepted)
        anomalous = sum(1 for e in accepted if e.is_anomalous_state)
        enhanced = sum(1 for e in self.entries if e.was_enhanced)

        latencies = [e.processing_latency_ms for e in self.entries if e.processing_latency_ms > 0]
        mean_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

        span = 0.0
        if total >= 2:
            span = max(0.0, self.entries[-1].timestamp_seconds - self.entries[0].timestamp_seconds)

        speaking_frames = sum(1 for e in accepted if e.is_speaking)
        looking_away_frames = sum(1 for e in accepted if e.is_looking_away)
        hands_measured = [e for e in accepted if e.hands_detected is not None]
        hands_absent_frames = sum(1 for e in hands_measured if e.hands_detected == 0)

        return {
            "session_id": self.session_id,
            "total_entries": total,
            "speaking_frames": speaking_frames,
            "looking_away_frames": looking_away_frames,
            "hands_absent_frames": hands_absent_frames,
            "hands_measured_frames": len(hands_measured),
            "accepted_frames": len(accepted),
            "rejected_frames": rejected,
            "enhanced_frames": enhanced,
            "anomalous_frames": anomalous,
            "anomalous_frame_ratio": round(anomalous / len(accepted), 4) if accepted else 0.0,
            "timeline_span_seconds": round(span, 3),
            "mean_processing_latency_ms": round(mean_latency, 2),
        }
