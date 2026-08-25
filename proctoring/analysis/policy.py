"""Exam strictness policy: which observations become events, and how they are rated.

The detectors measure; this module decides what a given examination *cares* about.
Separating the two matters because the same observation carries very different
weight in different settings — a hand near the face is unremarkable in an open-book
practice quiz and worth a proctor's glance in a high-stakes certification exam.

Three preset levels are provided.  Each controls both the detection thresholds and
the set of observations that are allowed to raise an event at all.

``STANDARD``
    Ordinary coursework.  Only unambiguous observations are raised: nobody present,
    someone else present, an unrecognised face, a phone, headphones.

``STRICT``
    Invigilated and higher-stakes exams.  Adds speaking, sustained looking away,
    hands at the ear and suspected earpieces, and shortens the durations required
    to qualify an event.

``MAXIMUM``
    Certification and remote high-stakes testing.  Adds hands leaving the frame and
    gaze leaving the screen, and qualifies events after roughly a second.

Raising the strictness raises the false-positive rate. That is a deliberate
trade, and it is the reason every event carries a snapshot: at ``MAXIMUM`` a
proctor should expect to dismiss a meaningful share of what is flagged.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from proctoring.core.events import EventSeverity, EventType


class StrictnessLevel(str, Enum):
    """Preset examination strictness profiles."""

    STANDARD = "STANDARD"
    STRICT = "STRICT"
    MAXIMUM = "MAXIMUM"


# Observations that every level reports. These are the unambiguous ones: their
# meaning does not depend on how strict the exam is.
_BASELINE_EVENTS: frozenset[EventType] = frozenset(
    {
        EventType.NO_FACE,
        EventType.MULTIPLE_FACES,
        EventType.UNKNOWN_FACE,
        EventType.PHONE_DETECTED,
        EventType.PROHIBITED_OBJECT,
        EventType.HEADPHONES_DETECTED,
        EventType.BROWSER_TAB_SWITCH,
        EventType.BROWSER_FULLSCREEN_EXIT,
        EventType.BROWSER_WINDOW_BLUR,
    }
)

_STRICT_ADDITIONS: frozenset[EventType] = frozenset(
    {
        EventType.CANDIDATE_SPEAKING,
        EventType.LOOKING_AWAY,
        EventType.HAND_NEAR_EAR,
        EventType.EARBUDS_SUSPECTED,
        EventType.SMARTWATCH_DETECTED,
        EventType.POSSIBLE_PRESENTATION_ATTACK,
    }
)

_MAXIMUM_ADDITIONS: frozenset[EventType] = frozenset(
    {
        EventType.HAND_NEAR_FACE,
        EventType.HANDS_NOT_VISIBLE,
        EventType.GAZE_OFF_SCREEN,
    }
)


@dataclass
class ExamPolicy:
    """Thresholds and enabled observations for one strictness level.

    Construct via :meth:`for_level`, then override individual fields if a specific
    examination needs something the presets do not cover.
    """

    level: StrictnessLevel = StrictnessLevel.STANDARD

    # Which observations may raise an event.
    enabled_events: frozenset[EventType] = _BASELINE_EVENTS

    # --- Temporal qualification -------------------------------------------
    min_event_duration_seconds: float = 1.5
    absence_tolerance_seconds: float = 1.0

    # Per-event duration overrides. A phone needs almost no persistence to be
    # worth a look; a hand near the face needs a lot before it means anything.
    event_min_duration: dict[EventType, float] = field(default_factory=dict)

    # --- Behavioural thresholds -------------------------------------------
    speech_movement_threshold: float = 0.055
    speech_min_oscillations: int = 2
    yaw_limit_degrees: float = 35.0
    pitch_limit_degrees: float = 28.0
    gaze_offset_limit: float = 0.35
    blink_threshold: float = 0.45
    liveness_grace_seconds: float = 45.0
    """Seconds a face may be watched with no blink before liveness is questioned."""
    smoothing_window_frames: int = 3
    """Frames voting on each behavioural boolean before it is reported. 1 disables
    smoothing; 3 removes single-frame flicker at a cost of at most one frame's lag."""

    hands_absent_grace_seconds: float = 8.0
    """How long hands may be out of view before it is recorded at all."""

    # --- Detector confidence floors ---------------------------------------
    phone_confidence_threshold: float = 0.40
    headphone_confidence_threshold: float = 0.25
    earbud_confidence_threshold: float = 0.30

    # --- Evidence ----------------------------------------------------------
    capture_annotated_snapshots: bool = True
    snapshot_on_every_event: bool = True

    def allows(self, event_type: EventType) -> bool:
        """Is this observation reportable under the current policy?"""
        return event_type in self.enabled_events

    def duration_for(self, event_type: EventType) -> float:
        """Seconds this observation must persist before it is marked QUALIFIED."""
        return self.event_min_duration.get(event_type, self.min_event_duration_seconds)

    @classmethod
    def for_level(cls, level: StrictnessLevel) -> "ExamPolicy":
        """Build the preset policy for a strictness level."""
        level = StrictnessLevel(level)

        if level is StrictnessLevel.STANDARD:
            return cls(
                level=level,
                enabled_events=_BASELINE_EVENTS,
                min_event_duration_seconds=2.0,
                absence_tolerance_seconds=1.5,
                event_min_duration={
                    EventType.PHONE_DETECTED: 0.5,
                    EventType.HEADPHONES_DETECTED: 1.0,
                    EventType.MULTIPLE_FACES: 1.5,
                },
                yaw_limit_degrees=40.0,
                pitch_limit_degrees=32.0,
            )

        if level is StrictnessLevel.STRICT:
            return cls(
                level=level,
                enabled_events=_BASELINE_EVENTS | _STRICT_ADDITIONS,
                min_event_duration_seconds=1.5,
                absence_tolerance_seconds=1.0,
                event_min_duration={
                    EventType.PHONE_DETECTED: 0.25,
                    EventType.HEADPHONES_DETECTED: 0.75,
                    EventType.EARBUDS_SUSPECTED: 2.0,  # weak signal: demand persistence
                    EventType.MULTIPLE_FACES: 1.0,
                    EventType.CANDIDATE_SPEAKING: 2.0,  # a word or two is not "speaking"
                    EventType.LOOKING_AWAY: 3.0,  # glancing away is normal
                    EventType.HAND_NEAR_EAR: 2.0,
                    # A weak signal about a serious allegation: demand real persistence.
                    EventType.POSSIBLE_PRESENTATION_ATTACK: 10.0,
                },
                speech_movement_threshold=0.050,
                yaw_limit_degrees=32.0,
                pitch_limit_degrees=26.0,
                gaze_offset_limit=0.32,
                earbud_confidence_threshold=0.28,
            )

        return cls(
            level=StrictnessLevel.MAXIMUM,
            enabled_events=_BASELINE_EVENTS | _STRICT_ADDITIONS | _MAXIMUM_ADDITIONS,
            min_event_duration_seconds=1.0,
            absence_tolerance_seconds=0.75,
            event_min_duration={
                EventType.PHONE_DETECTED: 0.25,
                EventType.HEADPHONES_DETECTED: 0.5,
                EventType.EARBUDS_SUSPECTED: 1.5,
                EventType.MULTIPLE_FACES: 0.5,
                EventType.CANDIDATE_SPEAKING: 1.5,
                EventType.LOOKING_AWAY: 2.0,
                EventType.GAZE_OFF_SCREEN: 2.5,
                EventType.HAND_NEAR_EAR: 1.0,
                EventType.HAND_NEAR_FACE: 4.0,
                EventType.HANDS_NOT_VISIBLE: 10.0,
                EventType.POSSIBLE_PRESENTATION_ATTACK: 10.0,
            },
            speech_movement_threshold=0.045,
            speech_min_oscillations=2,
            yaw_limit_degrees=28.0,
            pitch_limit_degrees=22.0,
            gaze_offset_limit=0.28,
            hands_absent_grace_seconds=6.0,
            smoothing_window_frames=3,
            phone_confidence_threshold=0.35,
            headphone_confidence_threshold=0.22,
            earbud_confidence_threshold=0.25,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the policy into the evidence manifest.

        Recording the policy alongside the events is what lets a reviewer months
        later understand why a session flagged what it did — the same footage under
        a different policy produces a different event list.
        """
        return {
            "level": self.level.value,
            "enabled_events": sorted(e.value for e in self.enabled_events),
            "min_event_duration_seconds": self.min_event_duration_seconds,
            "absence_tolerance_seconds": self.absence_tolerance_seconds,
            "event_min_duration": {
                k.value: v
                for k, v in sorted(self.event_min_duration.items(), key=lambda kv: kv[0].value)
            },
            "behavioural_thresholds": {
                "speech_movement_threshold": self.speech_movement_threshold,
                "speech_min_oscillations": self.speech_min_oscillations,
                "yaw_limit_degrees": self.yaw_limit_degrees,
                "pitch_limit_degrees": self.pitch_limit_degrees,
                "gaze_offset_limit": self.gaze_offset_limit,
                "blink_threshold": self.blink_threshold,
                "liveness_grace_seconds": self.liveness_grace_seconds,
                "hands_absent_grace_seconds": self.hands_absent_grace_seconds,
                "smoothing_window_frames": self.smoothing_window_frames,
            },
            "confidence_floors": {
                "phone": self.phone_confidence_threshold,
                "headphones": self.headphone_confidence_threshold,
                "earbuds": self.earbud_confidence_threshold,
            },
        }


# Severity assigned to each new observation type. Severity orders a proctor's
# review queue; it is never summed into a score.
BEHAVIOURAL_SEVERITY: dict[EventType, EventSeverity] = {
    EventType.CANDIDATE_SPEAKING: EventSeverity.HIGH,
    EventType.POSSIBLE_PRESENTATION_ATTACK: EventSeverity.MEDIUM,  # weak signal
    EventType.HEADPHONES_DETECTED: EventSeverity.HIGH,
    EventType.EARBUDS_SUSPECTED: EventSeverity.MEDIUM,  # weak signal, never HIGH
    EventType.SMARTWATCH_DETECTED: EventSeverity.MEDIUM,
    EventType.HAND_NEAR_EAR: EventSeverity.MEDIUM,
    EventType.HAND_NEAR_FACE: EventSeverity.LOW,
    EventType.HANDS_NOT_VISIBLE: EventSeverity.LOW,
    EventType.LOOKING_AWAY: EventSeverity.MEDIUM,
    EventType.GAZE_OFF_SCREEN: EventSeverity.LOW,
}
