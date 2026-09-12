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

from proctoring.analysis.gaze import GazeDirection, GazeObservation
from proctoring.analysis.hands import HandAnalysisResult, HandState
from proctoring.analysis.head_movement import HeadMovementPattern, HeadMovementTracker
from proctoring.analysis.paper import PaperAnalysisResult, PaperState
from proctoring.analysis.policy import ExamMode, ExamPolicy
from proctoring.core.events import EventType
from proctoring.evidence.annotator import AnnotationContext
from proctoring.observation import FrameObservation
from proctoring.preprocessing.camera_health import CameraAnomaly

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

        self._head_movement = HeadMovementTracker(
            window_seconds=policy.head_pattern_window_seconds,
            episode_min_seconds=policy.head_episode_min_seconds,
            repeat_episode_count=policy.head_repeat_episode_count,
            reversal_velocity_deg_per_s=policy.head_reversal_velocity_deg_per_s,
            min_reversals=policy.head_min_reversals,
        )
        self.last_head_pattern: HeadMovementPattern | None = None
        """The most recent head-movement description, exposed for the live HUD and
        for the timeline record. Read-only as far as callers are concerned."""

        # Sweep-level voting for worn devices. Keyed by event type; see
        # :meth:`_confirm_wearables` for why frames cannot be counted directly.
        self._wearable_votes: dict[EventType, deque[bool]] = {}
        self._last_wearable_result: object | None = None

    @property
    def policy(self) -> ExamPolicy:
        return self._policy_ref

    def reset(self) -> None:
        """Clear all per-session state."""
        self._history.clear()
        self._hands_absent_since = None
        self._head_movement.reset()
        self.last_head_pattern = None
        self._wearable_votes.clear()
        self._last_wearable_result = None

    def map_to_events(self, obs: FrameObservation) -> dict[EventType, dict[str, Any]]:
        """Map this frame's behavioural measurements onto reportable event types.

        The exam policy is applied here and only here: an observation that the
        current strictness level does not report is simply never handed to the
        temporal aggregator, so it produces no incident and no evidence.
        """
        active: dict[EventType, dict[str, Any]] = {}
        policy = self._policy_ref
        dynamics = obs.facial_dynamics
        hands = obs.hand_analysis
        face_present = bool(dynamics and dynamics.face_found)

        # 1. Camera stream health and obstruction checks (runs even if no face present)
        if obs.camera_health:
            if obs.camera_health.is_frozen and policy.allows(EventType.CAMERA_FRAME_FROZEN):
                active[EventType.CAMERA_FRAME_FROZEN] = {
                    "confidence": 1.0,
                    "description": f"Camera stream frozen for {obs.camera_health.freeze_duration_seconds:.1f}s",
                }
            elif obs.camera_health.anomaly == CameraAnomaly.STREAM_DISCONNECTED and policy.allows(
                EventType.CAMERA_DISCONNECTED
            ):
                active[EventType.CAMERA_DISCONNECTED] = {
                    "confidence": 1.0,
                    "description": "Camera feed disconnected or delivering empty buffers",
                }

        if (
            obs.occlusion
            and obs.occlusion.camera_obstructed
            and policy.allows(EventType.CAMERA_OBSTRUCTED)
        ):
            active[EventType.CAMERA_OBSTRUCTED] = {
                "confidence": float(obs.occlusion.confidence),
                "description": "Camera lens appears obstructed or covered",
            }

        # 2. Face-present behavioural observations
        if face_present and dynamics is not None:
            if dynamics.is_speaking and policy.allows(EventType.CANDIDATE_SPEAKING):
                active[EventType.CANDIDATE_SPEAKING] = {
                    "confidence": float(dynamics.speech_activity or 0.5),
                    "bbox": dynamics.mouth_region,
                    # Deliberately hedged wording. The camera measured a mouth
                    # opening and closing repeatedly; it did not hear anything, and
                    # the observation text a proctor reads must not imply it did.
                    "description": (
                        "Possible talking: sustained speech-like mouth activity "
                        f"(articulation {float(dynamics.speech_activity or 0.0):.2f}). "
                        "No audio is recorded — confirm from the snapshot"
                    ),
                }

            # Gaze, evaluated once against the mode-aware policy thresholds.
            gaze_obs: GazeObservation | None = obs.gaze or dynamics.gaze
            if gaze_obs is not None:
                is_gaze_off = policy.is_gaze_deviated(
                    horizontal=gaze_obs.horizontal,
                    vertical=gaze_obs.vertical,
                )
                gaze_dir_desc = (
                    f"{gaze_obs.direction.value.lower()} zone"
                    if gaze_obs.direction not in (GazeDirection.CENTER, GazeDirection.UNKNOWN)
                    else "screen area"
                )
            else:
                is_gaze_off = bool(dynamics.is_gaze_off_screen)
                gaze_dir_desc = "screen area"

            # Head pose, measured against the baseline the analyzer actually applied.
            # Passing zero here discarded per-candidate calibration entirely, so a
            # camera mounted off-centre either flooded the record with look-away
            # events or masked genuine turns, depending on which way it was offset.
            pose = dynamics.head_pose
            if pose is not None:
                is_looking_away = policy.is_pose_deviated(
                    yaw=pose.yaw,
                    pitch=pose.pitch,
                    baseline_yaw=dynamics.baseline_yaw,
                    baseline_pitch=dynamics.baseline_pitch,
                )
            else:
                is_looking_away = bool(dynamics.is_looking_away)

            pattern = self._update_head_movement(obs, dynamics, is_looking_away)

            if is_looking_away and policy.allows(EventType.LOOKING_AWAY):
                corroborated = self._gaze_agrees_with_head(pose, gaze_obs, policy)
                active[EventType.LOOKING_AWAY] = {
                    # Two independent signals pointing the same way is a materially
                    # stronger observation than head orientation alone, and the
                    # confidence carried into the evidence package says so instead
                    # of reporting a flat 1.0 for every case.
                    "confidence": 0.9 if corroborated else 0.6,
                    "bbox": dynamics.face_bbox,
                    "description": self._look_away_description(
                        pose, pattern, gaze_dir_desc, corroborated
                    ),
                }

            if (
                pattern is not None
                and pattern.is_pattern
                and policy.allows(EventType.SUSPICIOUS_HEAD_POSE)
            ):
                active[EventType.SUSPICIOUS_HEAD_POSE] = {
                    "confidence": 0.7,
                    "bbox": dynamics.face_bbox,
                    "description": self._head_pattern_description(pattern),
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

            if is_gaze_off and policy.allows(EventType.GAZE_OFF_SCREEN):
                active[EventType.GAZE_OFF_SCREEN] = {
                    "confidence": float(
                        gaze_obs.confidence if gaze_obs else (dynamics.gaze_offset or 0.5)
                    ),
                    "bbox": dynamics.face_bbox,
                    "description": f"Gaze directed toward {gaze_dir_desc} outside expected viewing zone",
                }

            # Face occlusion observation
            if (
                obs.occlusion
                and obs.occlusion.is_occluded
                and policy.allows(EventType.FACE_OCCLUDED)
            ):
                state_str = obs.occlusion.state.value.lower().replace("_", " ")
                active[EventType.FACE_OCCLUDED] = {
                    "confidence": float(obs.occlusion.confidence),
                    "bbox": dynamics.face_bbox,
                    "description": f"Facial region occluded ({state_str})",
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

        # 3. Hand kinematics & dynamics observations (writing vs resting)
        if hands is not None and hands.hands:
            for h in hands.hands:
                k_state = getattr(h.kinematics, "state", None) if hasattr(h, "kinematics") else None
                if k_state == HandState.HAND_WRITING and policy.allows(EventType.HAND_WRITING):
                    active[EventType.HAND_WRITING] = {
                        "confidence": 0.85,
                        "bbox": h.bbox,
                        "description": "Handwriting motion observed on paper workspace",
                    }
                elif k_state == HandState.HAND_RESTING and policy.allows(EventType.HAND_RESTING):
                    active[EventType.HAND_RESTING] = {
                        "confidence": 0.90,
                        "bbox": h.bbox,
                        "description": "Hand resting stationary in workspace",
                    }
                elif k_state == HandState.HAND_LIFTED_FROM_PAPER and policy.allows(EventType.HAND_LIFTED_FROM_PAPER):
                    active[EventType.HAND_LIFTED_FROM_PAPER] = {
                        "confidence": 0.80,
                        "bbox": h.bbox,
                        "description": "Hand lifted abruptly from paper surface",
                    }
                elif k_state == HandState.HAND_LEAVING_WRITING_AREA and policy.allows(EventType.HAND_LEAVING_WRITING_AREA):
                    active[EventType.HAND_LEAVING_WRITING_AREA] = {
                        "confidence": 0.80,
                        "bbox": h.bbox,
                        "description": "Hand leaving designated writing workspace",
                    }

        # 4. Physical paper detection and manipulation observations
        paper = obs.paper_analysis
        if paper is not None:
            if paper.paper_present and policy.allows(EventType.PAPER_PRESENT):
                sheet_box = paper.sheets[0].bbox if paper.sheets else None
                active[EventType.PAPER_PRESENT] = {
                    "confidence": float(paper.sheets[0].confidence) if paper.sheets else 0.8,
                    "bbox": sheet_box,
                    "description": f"Physical paper present on desk workspace ({paper.paper_count} sheet(s) detected)",
                }
            elif not paper.paper_present and policy.mode == ExamMode.PHYSICAL_PAPER and policy.allows(EventType.PAPER_ABSENT):
                active[EventType.PAPER_ABSENT] = {
                    "confidence": 0.85,
                    "description": "Required physical paper absent from designated workspace",
                }

            if paper.paper_count > 1 and policy.allows(EventType.MULTIPLE_PAPERS_DETECTED):
                sheet_box = paper.sheets[0].bbox if paper.sheets else None
                active[EventType.MULTIPLE_PAPERS_DETECTED] = {
                    "confidence": float(paper.sheets[0].confidence) if paper.sheets else 0.8,
                    "bbox": sheet_box,
                    "description": f"Multiple paper sheets detected in workspace ({paper.paper_count} sheets)",
                }

            if paper.manipulation_detected and policy.allows(EventType.PAPER_MANIPULATED):
                sheet_box = paper.sheets[0].bbox if paper.sheets else None
                active[EventType.PAPER_MANIPULATED] = {
                    "confidence": 0.85,
                    "bbox": sheet_box,
                    "description": f"Physical paper manipulation or displacement observed ({paper.displacement_px:.1f}px shift)",
                }

        self._confirm_wearables(obs, hands, active, policy)

        return self._debounce(active, policy)

    # ------------------------------------------------------------------
    # Head movement
    # ------------------------------------------------------------------

    def _update_head_movement(
        self,
        obs: FrameObservation,
        dynamics: Any,
        is_deviating: bool,
    ) -> HeadMovementPattern | None:
        """Feed this frame's pose to the movement tracker and keep the description.

        The tracker is told *whether* the pose deviates rather than being given the
        angular limits, so ``PHYSICAL_PAPER`` mode's allowance for looking down at
        paper is honoured without the thresholds existing in two places.
        """
        pose = dynamics.head_pose
        if pose is None:
            return None

        pattern = self._head_movement.update(
            timestamp_seconds=obs.timestamp_seconds,
            yaw=pose.yaw - dynamics.baseline_yaw,
            pitch=pose.pitch - dynamics.baseline_pitch,
            deviating=is_deviating,
        )
        self.last_head_pattern = pattern
        return pattern

    @staticmethod
    def _gaze_agrees_with_head(
        pose: Any,
        gaze: GazeObservation | None,
        policy: ExamPolicy,
    ) -> bool:
        """Is the gaze displaced meaningfully in the same direction as the head turn?

        Head orientation alone is the weakest of the attention signals — an
        off-centre camera produces a constant reading with no misconduct involved.
        Eyes displaced the same way at the same time is an independent measurement
        agreeing, and that is worth recording next to the observation.
        """
        if pose is None or gaze is None:
            return False

        margin = policy.gaze_corroboration_margin
        if abs(pose.yaw) >= abs(pose.pitch):
            # Positive yaw is a turn to the candidate's left; positive horizontal
            # gaze is the same side, so the signs must match to corroborate.
            if abs(gaze.horizontal) < policy.gaze_horizontal_limit * margin:
                return False
            return (pose.yaw > 0) == (gaze.horizontal > 0)

        limit = policy.gaze_vertical_up_limit if pose.pitch > 0 else policy.gaze_vertical_down_limit
        if abs(gaze.vertical) < limit * margin:
            return False
        return (pose.pitch > 0) == (gaze.vertical > 0)

    @staticmethod
    def _look_away_description(
        pose: Any,
        pattern: HeadMovementPattern | None,
        gaze_dir_desc: str,
        corroborated: bool,
    ) -> str:
        """Factual wording for a look-away observation, including what agreed with it."""
        if pose is None:
            return "Head turned away from the primary zone"

        text = f"Head turned away (yaw {pose.yaw:.0f}°, pitch {pose.pitch:.0f}°)"
        if pattern is not None and pattern.sustained_seconds > 0:
            text += f", held {pattern.sustained_seconds:.1f}s"
        if corroborated:
            text += f", with gaze also directed towards the {gaze_dir_desc}"
        return text

    @staticmethod
    def _head_pattern_description(pattern: HeadMovementPattern) -> str:
        """Factual wording for a repeated or rapid head-movement pattern."""
        parts: list[str] = []
        if pattern.repeated_look_away:
            parts.append(
                f"{pattern.episode_count} separate look-away episodes in the recent window"
            )
        if pattern.rapid_repeated_movement:
            parts.append(
                f"{pattern.rapid_reversals} rapid direction changes "
                f"(peak {pattern.peak_velocity_deg_per_s:.0f}°/s)"
            )
        return "Repeated head movement: " + "; ".join(parts)

    # ------------------------------------------------------------------
    # Worn devices
    # ------------------------------------------------------------------

    def _confirm_wearables(
        self,
        obs: FrameObservation,
        hands: HandAnalysisResult | None,
        active: dict[EventType, dict[str, Any]],
        policy: ExamPolicy,
    ) -> None:
        """Report a worn device only once several detection sweeps agree.

        The detector runs on a decimated cadence and its result is carried forward
        between sweeps, so a single marginal detection used to persist long enough,
        unopposed, to qualify a whole incident on its own. Votes are therefore
        counted per *sweep*, not per frame.

        Carried-forward frames re-present the identical result object, which is what
        distinguishes them: a fresh sweep produces a new one. Counting frames instead
        would let the carry-forward window stuff the ballot with copies of one
        detection.
        """
        result = obs.wearables
        if result is None:
            return

        is_new_sweep = result is not self._last_wearable_result
        self._last_wearable_result = result

        detected = {d.event_type: d for d in result.detections}
        hand_at_ear = bool(hands is not None and hands.hand_near_ear)
        window = max(1, int(policy.wearable_confirmation_window_sweeps))

        for event_type in (
            EventType.HEADPHONES_DETECTED,
            EventType.EARBUDS_SUSPECTED,
            EventType.SMARTWATCH_DETECTED,
        ):
            if not policy.allows(event_type):
                continue

            votes = self._wearable_votes.setdefault(event_type, deque(maxlen=window))
            if is_new_sweep:
                votes.append(event_type in detected)

            required = max(1, int(policy.wearable_confirmation_sweeps))
            if (
                policy.hand_at_ear_corroboration
                and hand_at_ear
                and event_type is EventType.EARBUDS_SUSPECTED
            ):
                # An independent signal from a different model. It lowers the visual
                # evidence required by one sweep; it never substitutes for it, so a
                # hand at an ear with nothing detected still reports nothing here.
                required = max(1, required - 1)

            if sum(votes) < required:
                continue

            detection = detected.get(event_type)
            if detection is None:
                # Confirmed across the window but absent from this exact sweep — the
                # incident stays open on the aggregator's absence tolerance rather
                # than being re-asserted from a frame that did not see it.
                continue

            active[event_type] = {
                "confidence": detection.confidence,
                "bbox": detection.bbox,
                "object_class": detection.target,
                "description": self._wearable_description(detection, sum(votes), len(votes)),
            }

    @staticmethod
    def _wearable_description(detection: Any, agreeing: int, sweeps: int) -> str:
        """Factual wording for a worn-device observation, including what supported it."""
        text = (
            f"{detection.target.replace('_', ' ').capitalize()} detected "
            f"(confidence {detection.confidence:.2f}, reliability {detection.reliability}, "
            f"{agreeing}/{sweeps} detection sweeps agreed"
        )
        if detection.source == "ear_zoom":
            text += ", found in the magnified ear region"
        if detection.hand_corroborated:
            text += ", with a hand at the ear in the same frame"
        return text + ")"

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
            if obs.gaze:
                measurements["gaze dir"] = obs.gaze.direction.value
        if self.last_head_pattern is not None and self.last_head_pattern.is_pattern:
            pattern = self.last_head_pattern
            measurements["look-aways"] = pattern.episode_count
            measurements["reversals"] = pattern.rapid_reversals
        if obs.occlusion and obs.occlusion.is_occluded:
            measurements["occlusion"] = obs.occlusion.state.value
        if hands is not None:
            measurements["hands"] = hands.hands_detected
        if obs.similarity is not None:
            measurements["identity sim"] = obs.similarity
        if obs.paper_analysis is not None:
            measurements["paper"] = obs.paper_analysis.paper_count
            if obs.paper_analysis.manipulation_detected:
                measurements["paper shift"] = f"{obs.paper_analysis.displacement_px:.1f}px"

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
