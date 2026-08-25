"""Translates raw detector output into reportable proctoring observations.

This is the policy layer between measurement and the temporal aggregator.  The
detectors in this package say what they saw; :class:`BehaviourObserver` decides
which of those observations the current examination actually reports, debounces
them so a single noisy frame cannot manufacture an incident, and assembles the
context used to annotate review snapshots.

It lives here rather than in the engine because none of it is orchestration — it
is entirely a function of the exam policy, and separating it keeps the engine to
the job of moving frames through stages.
"""

import logging
from collections import deque
from typing import Any

from proctoring.analysis.hands import HandAnalysisResult
from proctoring.analysis.policy import ExamPolicy
from proctoring.core.events import EventType
from proctoring.evidence.annotator import AnnotationContext
from proctoring.observation import FrameObservation

LOGGER = logging.getLogger(__name__)


def _to_xyxy(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Convert a YuNet ``(x, y, w, h)`` box to ``(x1, y1, x2, y2)``."""
    x, y, w, h = bbox
    return int(x), int(y), int(x + w), int(y + h)


class BehaviourObserver:
    """Maps per-frame measurements onto the events this examination reports."""

    def __init__(self, policy: ExamPolicy) -> None:
        self._policy_ref = policy
        # Short histories debouncing each behavioural boolean.
        self._history: dict[EventType, deque[bool]] = {}
        self._hands_absent_since: float | None = None

    @property
    def policy(self) -> ExamPolicy:
        return self._policy_ref

    def reset(self) -> None:
        """Clear all per-session state."""
        self._history.clear()
        self._hands_absent_since = None

    def map_to_events(self, obs: FrameObservation) -> dict[EventType, dict[str, Any]]:
        """Map this frame's behavioural measurements onto reportable event types.

        The exam policy is applied here and only here: an observation that the
        current strictness level does not report is simply never handed to the
        temporal aggregator, so it produces no incident and no evidence.

        Behavioural observations require a face in frame. Without one, ``NO_FACE``
        already describes the situation, and claiming the candidate's hands are
        missing when the candidate themselves is missing would be noise.
        """
        active: dict[EventType, dict[str, Any]] = {}
        policy = self._policy_ref
        dynamics = obs.facial_dynamics
        hands = obs.hand_analysis
        face_present = bool(dynamics and dynamics.face_found)

        if face_present:
            if dynamics.is_speaking and policy.allows(EventType.CANDIDATE_SPEAKING):
                active[EventType.CANDIDATE_SPEAKING] = {
                    "confidence": float(dynamics.speech_activity or 0.5),
                    "bbox": dynamics.mouth_region,
                    "description": "Mouth articulation consistent with speech",
                }

            if dynamics.is_looking_away and policy.allows(EventType.LOOKING_AWAY):
                pose = dynamics.head_pose
                active[EventType.LOOKING_AWAY] = {
                    "confidence": 1.0,
                    "bbox": dynamics.face_bbox,
                    "description": (
                        f"Head turned away from the screen "
                        f"(yaw {pose.yaw:.0f}°, pitch {pose.pitch:.0f}°)"
                        if pose
                        else "Head turned away from the screen"
                    ),
                }

            if dynamics.liveness_state == "NO_BLINK_DETECTED" and policy.allows(
                EventType.POSSIBLE_PRESENTATION_ATTACK
            ):
                active[EventType.POSSIBLE_PRESENTATION_ATTACK] = {
                    "confidence": 0.5,
                    "bbox": dynamics.face_bbox,
                    "description": (
                        f"No blink observed in {dynamics.observed_seconds:.0f}s of continuous "
                        f"face presence — consistent with a photograph or replayed video, "
                        f"but also with a candidate who blinks rarely"
                    ),
                }

            if dynamics.is_gaze_off_screen and policy.allows(EventType.GAZE_OFF_SCREEN):
                active[EventType.GAZE_OFF_SCREEN] = {
                    "confidence": float(dynamics.gaze_offset or 0.5),
                    "bbox": dynamics.face_bbox,
                    "description": "Gaze directed away from the screen area",
                }

        if face_present and hands is not None:
            if hands.hand_near_ear and policy.allows(EventType.HAND_NEAR_EAR):
                active[EventType.HAND_NEAR_EAR] = {
                    "confidence": 1.0,
                    "bbox": hands.hands[0].bbox if hands.hands else None,
                    "description": "Hand raised to the ear region",
                }
            elif hands.hand_near_face and policy.allows(EventType.HAND_NEAR_FACE):
                # Only when not already reported as the more specific ear case, so
                # one gesture does not produce two overlapping incidents.
                active[EventType.HAND_NEAR_FACE] = {
                    "confidence": 1.0,
                    "bbox": hands.hands[0].bbox if hands.hands else None,
                    "description": "Hand within the face region",
                }

            self._track_hands_absent(obs, hands, active)

        for detection in obs.wearables.detections if obs.wearables else []:
            if not policy.allows(detection.event_type):
                continue
            active[detection.event_type] = {
                "confidence": detection.confidence,
                "bbox": detection.bbox,
                "object_class": detection.target,
                "description": (
                    f"{detection.target.replace('_', ' ').capitalize()} detected "
                    f"(confidence {detection.confidence:.2f}, "
                    f"reliability {detection.reliability})"
                ),
            }

        return self._debounce(active, policy)

    def _debounce(
        self,
        active: dict[EventType, dict[str, Any]],
        policy: ExamPolicy,
    ) -> dict[EventType, dict[str, Any]]:
        """Require a majority of recent frames to agree before reporting a behaviour.

        Per-frame verdicts from the landmark models flicker: a hand tracker drops a
        frame, a blink briefly changes the mouth blendshapes, the head crosses the
        yaw threshold and comes back. Feeding that directly to the temporal
        aggregator fragments one real behaviour into several short incidents and
        manufactures brief ones that never happened.

        A simple majority over the last ``smoothing_window_frames`` frames removes
        that without hiding anything real: a behaviour that genuinely persists wins
        the vote within a frame or two, and the aggregator's duration thresholds
        still decide what qualifies.

        Wearable detections bypass this — they already run on a decimated cadence
        with their own carry-forward, so a second smoothing layer would only add lag.
        """
        window = max(1, int(policy.smoothing_window_frames))
        if window == 1:
            return active

        bypass = {
            EventType.HEADPHONES_DETECTED,
            EventType.EARBUDS_SUSPECTED,
            EventType.SMARTWATCH_DETECTED,
        }
        smoothable = policy.enabled_events - bypass

        smoothed: dict[EventType, dict[str, Any]] = {
            k: v for k, v in active.items() if k not in smoothable
        }

        for event_type in smoothable:
            history = self._history.setdefault(event_type, deque(maxlen=window))
            history.append(event_type in active)
            if sum(history) * 2 > len(history):
                # Carry this frame's detail when present, else the last known detail.
                smoothed[event_type] = active.get(
                    event_type, {"confidence": 1.0, "description": event_type.value}
                )

        return smoothed

    def _track_hands_absent(
        self,
        obs: FrameObservation,
        hands: HandAnalysisResult,
        active: dict[EventType, dict[str, Any]],
    ) -> None:
        """Report hands out of view only after they have been gone a while.

        Hands leave frame constantly and innocently — reaching for water, resting in
        the lap, a camera angled at the face. Only a sustained absence is worth a
        proctor's time, so the grace period from the policy must elapse first.
        """
        if hands.hands_visible:
            self._hands_absent_since = None
            return

        if self._hands_absent_since is None:
            self._hands_absent_since = obs.timestamp_seconds
            return

        absent_for = obs.timestamp_seconds - self._hands_absent_since
        if absent_for >= self._policy_ref.hands_absent_grace_seconds and self._policy_ref.allows(
            EventType.HANDS_NOT_VISIBLE
        ):
            active[EventType.HANDS_NOT_VISIBLE] = {
                "confidence": 1.0,
                "description": (
                    f"No hand visible for {absent_for:.0f}s while the candidate is present"
                ),
            }

    def build_annotation_context(self, obs: FrameObservation) -> AnnotationContext:
        """Collect everything drawable for this frame into a review context."""
        dynamics = obs.facial_dynamics
        hands = obs.hand_analysis

        measurements: dict[str, Any] = {}
        if dynamics and dynamics.face_found:
            if dynamics.head_pose:
                measurements["yaw"] = dynamics.head_pose.yaw
                measurements["pitch"] = dynamics.head_pose.pitch
            measurements["mouth open"] = dynamics.mouth_open_ratio
            measurements["speech act."] = dynamics.speech_activity
            measurements["speaking"] = dynamics.is_speaking
            measurements["gaze off"] = dynamics.gaze_offset
        if hands is not None:
            measurements["hands"] = hands.hands_detected
        if obs.similarity is not None:
            measurements["identity sim"] = obs.similarity

        return AnnotationContext(
            face_boxes=[_to_xyxy(b) for b in obs.face_boxes],
            face_labels=[obs.face_status.lower().replace("_", " ")] * len(obs.face_boxes),
            hand_landmarks=[h.landmarks for h in hands.hands] if hands else [],
            hand_boxes=[h.bbox for h in hands.hands] if hands else [],
            object_boxes=[
                (o["class_name"], o["confidence"], tuple(o["bbox"])) for o in obs.prohibited_objects
            ],
            wearable_boxes=[
                (d.target, d.confidence, d.bbox)
                for d in (obs.wearables.detections if obs.wearables else [])
            ],
            ear_regions=dynamics.ear_regions if dynamics else [],
            mouth_region=dynamics.mouth_region if dynamics else None,
            measurements=measurements,
        )
