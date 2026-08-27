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

from proctoring.core.events import EventSeverity, EventType, is_technical


class StrictnessLevel(str, Enum):
    """Preset examination strictness profiles."""

    STANDARD = "STANDARD"
    STRICT = "STRICT"
    MAXIMUM = "MAXIMUM"


class ExamMode(str, Enum):
    """Exam format profile governing behavioural interpretation boundaries.

    ``DIGITAL_SCREEN``
        The candidate is expected to look primarily at the monitor. Sustained looking
        downward or to the sides is interpreted as looking away.

    ``PHYSICAL_PAPER``
        The candidate is writing or reading from physical paper on their desk.
        Normal downward head tilt and downward gaze are expected and permitted;
        only extreme lateral deviation or looking straight up/away are flagged.
    """

    DIGITAL_SCREEN = "DIGITAL_SCREEN"
    PHYSICAL_PAPER = "PHYSICAL_PAPER"


# Observations that every level reports. These are the unambiguous ones: their
# meaning does not depend on how strict the exam is.
_BASELINE_EVENTS: frozenset[EventType] = frozenset(
    {
        EventType.NO_FACE,
        EventType.MULTIPLE_FACES,
        EventType.UNKNOWN_FACE,
        EventType.CAMERA_OBSTRUCTED,
        EventType.CAMERA_FRAME_FROZEN,
        EventType.CAMERA_DISCONNECTED,
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
        EventType.SUSPICIOUS_HEAD_POSE,
        EventType.FACE_OCCLUDED,
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
    """Thresholds and enabled observations for one strictness level and exam mode.

    Construct via :meth:`for_level` or :meth:`for_mode`, then override individual
    fields if a specific examination needs custom parameters.
    """

    level: StrictnessLevel = StrictnessLevel.STANDARD
    mode: ExamMode = ExamMode.DIGITAL_SCREEN

    # Which observations may raise an event.
    enabled_events: frozenset[EventType] = _BASELINE_EVENTS

    # --- Temporal qualification -------------------------------------------
    min_event_duration_seconds: float = 1.5
    absence_tolerance_seconds: float = 1.0

    # Per-event duration overrides. A phone needs almost no persistence to be
    # worth a look; a hand near the face needs a lot before it means anything.
    event_min_duration: dict[EventType, float] = field(default_factory=dict)

    # --- Speech-like mouth activity ---------------------------------------
    # Webcam proctoring has no audio. What is measured is *articulation*: the
    # mouth repeatedly opening and closing over a window. Three conditions must
    # hold together, because each alone has a common innocent cause.
    speech_articulation_amplitude: float = 0.18
    """Robust peak-to-trough swing (p90 - p10) of the articulation index required
    across the window. A resting or neutrally-open mouth barely varies; speech
    swings widely. Chosen well above landmark jitter and below a yawn's excursion."""

    speech_min_crossings: int = 3
    """Times the articulation index must cross its own mid-level within the window.
    This is what separates speech from a yawn: a yawn crosses the midpoint twice
    (once opening, once closing) however large it is, while continuous articulation
    crosses repeatedly. Counting crossings rather than frame-to-frame sign changes
    is deliberate — at 4 fps the 3-6 Hz syllable rate is aliased, and per-frame
    deltas are then close to noise."""

    speech_closed_ratio: float = 0.45
    """The mouth must return below this fraction of the window's amplitude at least
    once. A yawn, a smile and a resting open mouth all hold a level; speech keeps
    returning to near-closed."""

    speech_window_seconds: float = 3.0
    """Length of the articulation window in *seconds*, not frames, so the measure
    does not silently change meaning when the sampling rate changes."""

    speech_min_sampling_fps: float = 6.0
    """Sampling rate below which speech-like activity is **not measurable at all**.

    Ordinary speech articulates at roughly 3-5 Hz. Sampling below twice that rate
    aliases the mouth signal into a slow wave whose shape is indistinguishable from
    a single yawn — measured at 4 fps, a talking mouth and a yawning mouth both
    produce one mid-level crossing across a three-second window. No threshold can
    separate them, because the information is gone before any threshold sees it.

    Below this floor the analyzer therefore reports *not measured* rather than
    *not speaking*: a rate that makes the measurement impossible must not be
    reported as a candidate who was silent. Sessions that report speaking raise
    their sampling rate to this floor automatically — see
    :meth:`~proctoring.config.SessionConfig.__post_init__`."""

    # --- Head pose and movement -------------------------------------------
    yaw_limit_degrees: float = 35.0
    pitch_limit_degrees: float = 28.0  # Legacy symmetrical pitch limit
    pitch_up_limit_degrees: float = 25.0
    pitch_down_limit_degrees: float = 28.0
    gaze_offset_limit: float = 0.35  # Legacy scalar gaze limit
    gaze_horizontal_limit: float = 0.35
    gaze_vertical_up_limit: float = 0.30
    gaze_vertical_down_limit: float = 0.35
    blink_threshold: float = 0.45
    liveness_grace_seconds: float = 45.0
    """Seconds a face may be watched with no blink before liveness is questioned."""
    smoothing_window_frames: int = 3
    """Frames voting on each behavioural boolean before it is reported. 1 disables
    smoothing; 3 removes single-frame flicker at a cost of at most one frame's lag."""

    head_pattern_window_seconds: float = 20.0
    """Window over which repeated head-movement patterns are looked for. Long enough
    that several glances fall inside it, short enough that a proctor can watch the
    corresponding stretch of the session."""

    head_episode_min_seconds: float = 0.5
    """A deviation must last at least this long to count as one *episode*. Filters
    the single-frame threshold crossings that a head merely passing through a wide
    angle produces."""

    head_repeat_episode_count: int = 3
    """Separate look-away episodes within the pattern window before the movement is
    reported as a pattern rather than as isolated glances. Three is the point at
    which "glanced at something" stops being a reasonable single explanation."""

    head_reversal_velocity_deg_per_s: float = 45.0
    """Angular speed above which a yaw movement counts as a rapid directional move
    rather than ordinary postural drift."""

    head_min_reversals: int = 4
    """Rapid direction reversals within the pattern window before repeated
    fast head movement is reported."""

    gaze_corroboration_margin: float = 0.6
    """Fraction of the gaze limit that gaze must exceed, in the same direction as
    the head, for the two signals to be treated as corroborating each other."""

    hands_absent_grace_seconds: float = 8.0
    """How long hands may be out of view before it is recorded at all."""

    # --- Detector confidence floors ---------------------------------------
    phone_confidence_threshold: float = 0.40
    headphone_confidence_threshold: float = 0.25
    earbud_confidence_threshold: float = 0.30

    # --- Worn audio devices -----------------------------------------------
    ear_region_padding_ratio: float = 0.55
    """How far each ear box is grown beyond the ear landmarks, as a fraction of its
    own size. The ear perimeter landmarks trace the *attachment* line of the ear;
    a bud sitting in the canal, and anything hooked over the top, fall outside that
    line. Too tight a box was silently discarding genuine detections."""

    enable_ear_region_zoom: bool = True
    """Run a second detection pass over upscaled crops of each ear region. An in-ear
    bud occupies roughly 15 px on a 640x480 frame — below what the detector resolves
    at native scale. Zooming the ear to a few hundred pixels is what makes small
    earpieces detectable at all, and it is the single largest recall improvement
    available here."""

    ear_roi_target_px: int = 320
    """Longest side each ear crop is upscaled to before the second pass."""

    wearable_confirmation_sweeps: int = 2
    """Detection sweeps that must agree before a worn device is reported. The
    detector runs on a decimated cadence and its result is carried forward between
    sweeps, so without this a single weak frame could qualify a whole incident."""

    wearable_confirmation_window_sweeps: int = 3
    """Sweeps the confirmation votes are counted over."""

    hand_at_ear_corroboration: bool = True
    """Let a hand at the ear lower the confirmation requirement for an earpiece by
    one sweep. Reaching to an ear is exactly the gesture that accompanies inserting
    or adjusting a bud, and the two signals are independent."""

    # --- Evidence ----------------------------------------------------------
    capture_annotated_snapshots: bool = True
    snapshot_on_every_event: bool = True

    def allows(self, event_type: EventType) -> bool:
        """Is this observation reportable under the current policy?

        Technical diagnostics always are. Strictness governs how closely the
        *candidate* is scrutinised; it must never decide whether the operator is
        told that the camera froze or a detector died. Suppressing those would let a
        lenient profile hide the very faults that explain missing observations.
        """
        if is_technical(event_type):
            return True
        return event_type in self.enabled_events

    def duration_for(self, event_type: EventType) -> float:
        """Seconds this observation must persist before it is marked QUALIFIED."""
        return self.event_min_duration.get(event_type, self.min_event_duration_seconds)

    def is_pose_deviated(
        self,
        yaw: float,
        pitch: float,
        baseline_yaw: float = 0.0,
        baseline_pitch: float = 0.0,
    ) -> bool:
        """Evaluate whether head rotation exceeds the allowable viewing zone for this exam mode."""
        dyaw = abs(yaw - baseline_yaw)
        dpitch = pitch - baseline_pitch

        if dyaw > self.yaw_limit_degrees:
            return True

        if dpitch > 0:  # Looking upwards
            return dpitch > self.pitch_up_limit_degrees
        # Looking downwards (negative pitch)
        return abs(dpitch) > self.pitch_down_limit_degrees

    def is_gaze_deviated(
        self,
        horizontal: float,
        vertical: float,
        baseline_horizontal: float = 0.0,
        baseline_vertical: float = 0.0,
    ) -> bool:
        """Evaluate whether eye gaze direction exceeds the expected zone for this exam mode."""
        dh = abs(horizontal - baseline_horizontal)
        dv = vertical - baseline_vertical

        if dh > self.gaze_horizontal_limit:
            return True

        if dv > 0:  # Gaze upwards
            return dv > self.gaze_vertical_up_limit
        # Gaze downwards (negative vertical)
        return abs(dv) > self.gaze_vertical_down_limit

    @classmethod
    def for_level(
        cls,
        level: StrictnessLevel,
        mode: ExamMode = ExamMode.DIGITAL_SCREEN,
    ) -> "ExamPolicy":
        """Build the preset policy for a strictness level and exam mode."""
        level = StrictnessLevel(level)
        mode = ExamMode(mode)

        # Base parameters per strictness level
        if level is StrictnessLevel.STANDARD:
            policy = cls(
                level=level,
                mode=mode,
                enabled_events=_BASELINE_EVENTS,
                min_event_duration_seconds=2.0,
                absence_tolerance_seconds=1.5,
                event_min_duration={
                    EventType.PHONE_DETECTED: 0.5,
                    EventType.HEADPHONES_DETECTED: 1.0,
                    EventType.MULTIPLE_FACES: 1.5,
                    EventType.CAMERA_OBSTRUCTED: 2.0,
                },
                yaw_limit_degrees=40.0,
                pitch_limit_degrees=32.0,
                pitch_up_limit_degrees=28.0,
                pitch_down_limit_degrees=32.0,
                gaze_offset_limit=0.38,
                gaze_horizontal_limit=0.38,
                gaze_vertical_up_limit=0.32,
                gaze_vertical_down_limit=0.38,
            )
        elif level is StrictnessLevel.STRICT:
            policy = cls(
                level=level,
                mode=mode,
                enabled_events=_BASELINE_EVENTS | _STRICT_ADDITIONS,
                min_event_duration_seconds=1.5,
                absence_tolerance_seconds=1.0,
                event_min_duration={
                    EventType.PHONE_DETECTED: 0.25,
                    EventType.HEADPHONES_DETECTED: 0.75,
                    EventType.EARBUDS_SUSPECTED: 2.0,
                    EventType.MULTIPLE_FACES: 1.0,
                    EventType.CANDIDATE_SPEAKING: 2.0,
                    EventType.LOOKING_AWAY: 3.0,
                    EventType.SUSPICIOUS_HEAD_POSE: 1.0,
                    EventType.FACE_OCCLUDED: 2.5,
                    EventType.CAMERA_OBSTRUCTED: 1.5,
                    EventType.HAND_NEAR_EAR: 2.0,
                    EventType.POSSIBLE_PRESENTATION_ATTACK: 10.0,
                },
                speech_articulation_amplitude=0.16,
                speech_min_crossings=3,
                yaw_limit_degrees=28.0,
                pitch_limit_degrees=26.0,
                pitch_up_limit_degrees=24.0,
                pitch_down_limit_degrees=26.0,
                gaze_offset_limit=0.32,
                gaze_horizontal_limit=0.32,
                gaze_vertical_up_limit=0.28,
                gaze_vertical_down_limit=0.32,
                earbud_confidence_threshold=0.28,
            )
        else:
            policy = cls(
                level=StrictnessLevel.MAXIMUM,
                mode=mode,
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
                    EventType.SUSPICIOUS_HEAD_POSE: 1.0,
                    EventType.GAZE_OFF_SCREEN: 2.5,
                    EventType.FACE_OCCLUDED: 2.0,
                    EventType.CAMERA_OBSTRUCTED: 1.0,
                    EventType.HAND_NEAR_EAR: 1.0,
                    EventType.HAND_NEAR_FACE: 4.0,
                    EventType.HANDS_NOT_VISIBLE: 10.0,
                    EventType.POSSIBLE_PRESENTATION_ATTACK: 10.0,
                },
                speech_articulation_amplitude=0.14,
                speech_min_crossings=3,
                head_repeat_episode_count=3,
                head_min_reversals=3,
                yaw_limit_degrees=24.0,
                pitch_limit_degrees=22.0,
                pitch_up_limit_degrees=20.0,
                pitch_down_limit_degrees=22.0,
                gaze_offset_limit=0.28,
                gaze_horizontal_limit=0.28,
                gaze_vertical_up_limit=0.24,
                gaze_vertical_down_limit=0.28,
                hands_absent_grace_seconds=6.0,
                smoothing_window_frames=3,
                phone_confidence_threshold=0.35,
                headphone_confidence_threshold=0.22,
                earbud_confidence_threshold=0.25,
            )

        # Apply physical-paper exam profile adjustments if requested
        if mode is ExamMode.PHYSICAL_PAPER:
            policy.pitch_down_limit_degrees = 60.0
            policy.gaze_vertical_down_limit = 0.70
            policy.yaw_limit_degrees = max(policy.yaw_limit_degrees, 38.0)
            if EventType.LOOKING_AWAY in policy.event_min_duration:
                policy.event_min_duration[EventType.LOOKING_AWAY] = max(
                    policy.event_min_duration[EventType.LOOKING_AWAY], 3.5
                )

        return policy

    @classmethod
    def for_mode(
        cls,
        mode: ExamMode,
        level: StrictnessLevel = StrictnessLevel.STANDARD,
    ) -> "ExamPolicy":
        """Convenience constructor building a policy for a specific exam mode."""
        return cls.for_level(level=level, mode=mode)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the policy into the evidence manifest."""
        return {
            "level": self.level.value,
            "mode": self.mode.value,
            "enabled_events": sorted(e.value for e in self.enabled_events),
            "min_event_duration_seconds": self.min_event_duration_seconds,
            "absence_tolerance_seconds": self.absence_tolerance_seconds,
            "event_min_duration": {
                k.value: v
                for k, v in sorted(self.event_min_duration.items(), key=lambda kv: kv[0].value)
            },
            "behavioural_thresholds": {
                "speech_min_sampling_fps": self.speech_min_sampling_fps,
                "speech_articulation_amplitude": self.speech_articulation_amplitude,
                "speech_min_crossings": self.speech_min_crossings,
                "speech_closed_ratio": self.speech_closed_ratio,
                "speech_window_seconds": self.speech_window_seconds,
                "yaw_limit_degrees": self.yaw_limit_degrees,
                "pitch_limit_degrees": self.pitch_limit_degrees,
                "pitch_up_limit_degrees": self.pitch_up_limit_degrees,
                "pitch_down_limit_degrees": self.pitch_down_limit_degrees,
                "gaze_offset_limit": self.gaze_offset_limit,
                "gaze_horizontal_limit": self.gaze_horizontal_limit,
                "gaze_vertical_up_limit": self.gaze_vertical_up_limit,
                "gaze_vertical_down_limit": self.gaze_vertical_down_limit,
                "blink_threshold": self.blink_threshold,
                "liveness_grace_seconds": self.liveness_grace_seconds,
                "hands_absent_grace_seconds": self.hands_absent_grace_seconds,
                "smoothing_window_frames": self.smoothing_window_frames,
                "head_pattern_window_seconds": self.head_pattern_window_seconds,
                "head_episode_min_seconds": self.head_episode_min_seconds,
                "head_repeat_episode_count": self.head_repeat_episode_count,
                "head_reversal_velocity_deg_per_s": self.head_reversal_velocity_deg_per_s,
                "head_min_reversals": self.head_min_reversals,
                "gaze_corroboration_margin": self.gaze_corroboration_margin,
            },
            "confidence_floors": {
                "phone": self.phone_confidence_threshold,
                "headphones": self.headphone_confidence_threshold,
                "earbuds": self.earbud_confidence_threshold,
            },
            "wearable_detection": {
                "ear_region_padding_ratio": self.ear_region_padding_ratio,
                "enable_ear_region_zoom": self.enable_ear_region_zoom,
                "ear_roi_target_px": self.ear_roi_target_px,
                "wearable_confirmation_sweeps": self.wearable_confirmation_sweeps,
                "wearable_confirmation_window_sweeps": self.wearable_confirmation_window_sweeps,
                "hand_at_ear_corroboration": self.hand_at_ear_corroboration,
            },
        }


# Severity assigned to each new observation type. Severity orders a proctor's
# review queue; it is never summed into a score.
BEHAVIOURAL_SEVERITY: dict[EventType, EventSeverity] = {
    # Speech-like mouth activity is inferred from lip movement with no audio, so it
    # is triaged below observations the camera can actually confirm.
    EventType.CANDIDATE_SPEAKING: EventSeverity.MEDIUM,
    EventType.SUSPICIOUS_HEAD_POSE: EventSeverity.MEDIUM,
    EventType.POSSIBLE_PRESENTATION_ATTACK: EventSeverity.MEDIUM,  # weak signal
    EventType.HEADPHONES_DETECTED: EventSeverity.HIGH,
    EventType.EARBUDS_SUSPECTED: EventSeverity.MEDIUM,  # weak signal, never HIGH
    EventType.SMARTWATCH_DETECTED: EventSeverity.MEDIUM,
    EventType.FACE_OCCLUDED: EventSeverity.MEDIUM,
    EventType.CAMERA_OBSTRUCTED: EventSeverity.CRITICAL,
    EventType.HAND_NEAR_EAR: EventSeverity.MEDIUM,
    EventType.HAND_NEAR_FACE: EventSeverity.LOW,
    EventType.HANDS_NOT_VISIBLE: EventSeverity.LOW,
    EventType.LOOKING_AWAY: EventSeverity.MEDIUM,
    EventType.GAZE_OFF_SCREEN: EventSeverity.LOW,
}
