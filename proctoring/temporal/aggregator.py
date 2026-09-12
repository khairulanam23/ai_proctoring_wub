import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from proctoring.analysis.phone_disambiguation import (
    PhoneClassification,
    PhoneHandDisambiguator,
)
from proctoring.core.events import (
    DetectorInfo,
    EventRecord,
    EventSeverity,
    EventStatus,
    EventType,
    ObservationDetail,
    format_seconds_to_timestamp,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class ActiveIncident:
    """Internal state tracking an ongoing continuous observation incident."""

    event_type: EventType
    severity: EventSeverity
    detector: DetectorInfo
    start_timestamp: float
    last_seen_timestamp: float
    object_class: str | None = None
    face_count: int | None = None
    similarity_score: float | None = None
    timestamps: list[float] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    bounding_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    representative_bbox: tuple[int, int, int, int] | None = None
    best_confidence: float = 0.0
    best_quality_score: float = 0.0
    best_frame_index: int = 0
    best_timestamp: float = 0.0
    detail_description: str | None = None
    correlations: list[dict[str, Any]] = field(default_factory=list)
    status: EventStatus = EventStatus.OPEN
    associated_subject_id: int | None = None
    interaction_type: str | None = None

    def add_observation(
        self,
        timestamp: float,
        frame_index: int,
        confidence: float,
        bbox: tuple[int, int, int, int] | None = None,
        similarity_score: float | None = None,
        blur_variance: float | None = None,
        correlation: dict[str, Any] | None = None,
        required_duration: float = 0.0,
    ) -> None:
        """Update incident with a new frame observation and maintain lifecycle state."""
        self.last_seen_timestamp = timestamp
        self.timestamps.append(timestamp)
        self.frame_indices.append(frame_index)
        self.confidences.append(confidence)
        if bbox is not None:
            self.bounding_boxes.append(bbox)
        if similarity_score is not None:
            self.similarity_score = similarity_score
        if correlation is not None:
            self.correlations.append(correlation)

        # Composite quality score: confidence (60%) + sharpness (40%)
        sharpness_norm = min(1.0, (blur_variance or 50.0) / 150.0)
        quality = confidence * 0.6 + sharpness_norm * 0.4

        if quality >= self.best_quality_score or self.best_quality_score == 0.0:
            self.best_quality_score = quality
            self.best_confidence = max(confidence, self.best_confidence)
            self.best_frame_index = frame_index
            self.best_timestamp = timestamp
            if bbox is not None:
                self.representative_bbox = bbox

        # Lifecycle state progression: OPEN -> ACTIVE -> QUALIFIED
        duration = max(0.0, self.last_seen_timestamp - self.start_timestamp)
        if required_duration > 0.0 and duration >= required_duration:
            self.status = EventStatus.QUALIFIED
        elif len(self.frame_indices) > 1:
            self.status = EventStatus.ACTIVE


class UnifiedTemporalAggregator:
    """Continuous temporal incident aggregator for face, object, and browser proctoring observations.

    Transforms instantaneous per-frame detections into continuous, structured EventRecords.
    Does NOT compute, accumulate, or maintain any cumulative suspicion or risk scores.
    """

    DEFAULT_SEVERITY_MAP: dict[EventType, EventSeverity] = {
        EventType.NO_FACE: EventSeverity.HIGH,
        EventType.MULTIPLE_FACES: EventSeverity.HIGH,
        EventType.UNKNOWN_FACE: EventSeverity.HIGH,
        EventType.FACE_MISMATCH: EventSeverity.HIGH,
        EventType.FACE_OCCLUDED: EventSeverity.MEDIUM,
        EventType.CAMERA_OBSTRUCTED: EventSeverity.CRITICAL,
        EventType.PERSON_ENTERED_FRAME: EventSeverity.INFO,
        EventType.PERSON_LEFT_FRAME: EventSeverity.INFO,
        EventType.PHONE_DETECTED: EventSeverity.HIGH,
        EventType.PHONE_CANDIDATE_UNCERTAIN: EventSeverity.LOW,
        EventType.PROHIBITED_OBJECT: EventSeverity.MEDIUM,
        EventType.PAPER_PRESENT: EventSeverity.INFO,
        EventType.PAPER_ABSENT: EventSeverity.LOW,
        EventType.PAPER_MANIPULATED: EventSeverity.MEDIUM,
        EventType.MULTIPLE_PAPERS_DETECTED: EventSeverity.MEDIUM,
        EventType.HAND_WRITING: EventSeverity.INFO,
        EventType.HAND_RESTING: EventSeverity.INFO,
        EventType.HAND_LIFTED_FROM_PAPER: EventSeverity.LOW,
        EventType.HAND_LEAVING_WRITING_AREA: EventSeverity.LOW,
        EventType.BROWSER_TAB_SWITCH: EventSeverity.CRITICAL,
        EventType.BROWSER_FULLSCREEN_EXIT: EventSeverity.HIGH,
        EventType.BROWSER_WINDOW_BLUR: EventSeverity.MEDIUM,
        EventType.LOOKING_AWAY: EventSeverity.MEDIUM,
        EventType.SUSPICIOUS_HEAD_POSE: EventSeverity.MEDIUM,
        EventType.GAZE_OFF_SCREEN: EventSeverity.LOW,
        EventType.HAND_NEAR_FACE: EventSeverity.LOW,
        EventType.HAND_NEAR_EAR: EventSeverity.MEDIUM,
        EventType.HANDS_NOT_VISIBLE: EventSeverity.LOW,
        EventType.CANDIDATE_SPEAKING: EventSeverity.HIGH,
        EventType.POSSIBLE_PRESENTATION_ATTACK: EventSeverity.MEDIUM,
        EventType.HEADPHONES_DETECTED: EventSeverity.HIGH,
        EventType.EARBUDS_SUSPECTED: EventSeverity.MEDIUM,
        EventType.SMARTWATCH_DETECTED: EventSeverity.MEDIUM,
        EventType.SESSION_PAUSED: EventSeverity.INFO,
        EventType.SESSION_RESUMED: EventSeverity.INFO,
        EventType.SYSTEM_ERROR: EventSeverity.HIGH,
        EventType.DETECTOR_ERROR: EventSeverity.HIGH,
        EventType.CAMERA_FRAME_FROZEN: EventSeverity.HIGH,
        EventType.CAMERA_DISCONNECTED: EventSeverity.HIGH,
        EventType.OTHER_SUSPICIOUS_ACTIVITY: EventSeverity.LOW,
    }

    def __init__(
        self,
        session_id: str = "default_session",
        absence_tolerance_seconds: float = 1.0,
        min_event_duration_seconds: float = 1.0,
        severity_overrides: dict[EventType, EventSeverity] | None = None,
        min_duration_overrides: dict[EventType, float] | None = None,
    ) -> None:
        self.session_id = str(session_id)
        self.absence_tolerance_seconds = max(0.0, float(absence_tolerance_seconds))
        self.min_event_duration_seconds = max(0.0, float(min_event_duration_seconds))
        self.severity_map = dict(self.DEFAULT_SEVERITY_MAP)
        if severity_overrides:
            self.severity_map.update(severity_overrides)

        # Per-event qualification durations. A phone glimpsed for a moment is worth
        # a proctor's attention; a hand near the face for the same moment is not.
        self.min_duration_overrides: dict[EventType, float] = dict(min_duration_overrides or {})

        # Active tracking states
        self.active_incidents: dict[str, ActiveIncident] = {}
        self.closed_events: list[EventRecord] = []
        self._event_counter = 0
        self.phone_disambiguator = PhoneHandDisambiguator()

    def _generate_event_id(self, event_type: EventType) -> str:
        self._event_counter += 1
        return f"evt_{self.session_id}_{self._event_counter:04d}_{event_type.value.lower()}"

    def update_face_observation(
        self,
        face_status: str,  # "NO_FACE", "SINGLE_FACE", "MULTIPLE_FACES", "UNKNOWN_FACE", "ENROLLED_PLUS_ADDITIONAL"
        timestamp: float,
        frame_index: int,
        face_count: int,
        detector: DetectorInfo,
        similarity_score: float | None = None,
        bboxes: list[tuple[int, int, int, int]] | None = None,
        confidences: list[float] | None = None,
        enrolled_present: bool | None = None,
        tracked_subjects: list[Any] | None = None,
    ) -> None:
        """Process face presence and identity observations for a frame.

        ``enrolled_present`` carries the multi-face identity result. When several
        people are on screen and the enrolled candidate is verifiably *not* among
        them, both ``MULTIPLE_FACES`` and ``UNKNOWN_FACE`` are opened: "the
        candidate plus someone else" and "two strangers, candidate gone" are
        materially different observations and a proctor must be able to tell them
        apart.
        """
        # A stage that did not run makes no claim about the candidate: return
        # before any incident is opened or closed, so an unmeasured frame never
        # becomes evidence of absence.
        if face_status == "NOT_MEASURED":
            return

        # Determine which face conditions this frame exhibits. More than one can be
        # true at once, which is why this is a set rather than a single value.
        active_types: set[EventType] = set()
        if face_status in ("NO_FACE", "NO_PERSON"):
            active_types.add(EventType.NO_FACE)
        elif face_status in (
            "MULTIPLE_FACES",
            "MULTIPLE_UNKNOWN_PERSONS",
            "ENROLLED_PLUS_ADDITIONAL_PERSON",
            "ENROLLED_PLUS_ADDITIONAL",
        ):
            active_types.add(EventType.MULTIPLE_FACES)
            if enrolled_present is False:
                active_types.add(EventType.UNKNOWN_FACE)
        elif face_status in (
            "UNKNOWN_FACE",
            "UNKNOWN_PERSON_ONLY",
            "FACE_MISMATCH",
            "IDENTITY_MISMATCH",
        ):
            active_types.add(EventType.UNKNOWN_FACE)

        primary_confidence = confidences[0] if (confidences and len(confidences) > 0) else 1.0
        primary_bbox = bboxes[0] if (bboxes and len(bboxes) > 0) else None

        if tracked_subjects:
            enrolled = [s for s in tracked_subjects if getattr(s, "is_enrolled", False)]
            if enrolled:
                primary_confidence = getattr(enrolled[0], "confidence", primary_confidence)
                primary_bbox = getattr(enrolled[0], "bbox", primary_bbox)
                if getattr(enrolled[0], "similarity_score", None) is not None:
                    similarity_score = enrolled[0].similarity_score
            elif tracked_subjects:
                best_s = max(tracked_subjects, key=lambda s: getattr(s, "confidence", 0.0))
                primary_confidence = getattr(best_s, "confidence", primary_confidence)
                primary_bbox = getattr(best_s, "bbox", primary_bbox)

        # Check all face incident keys
        face_event_types = {EventType.NO_FACE, EventType.MULTIPLE_FACES, EventType.UNKNOWN_FACE}
        for f_type in face_event_types:
            key = f"face_{f_type.value}"
            if key in self.active_incidents:
                incident = self.active_incidents[key]
                if f_type in active_types:
                    # Still active: add observation
                    incident.add_observation(
                        timestamp=timestamp,
                        frame_index=frame_index,
                        confidence=primary_confidence,
                        bbox=primary_bbox,
                        similarity_score=similarity_score,
                    )
                else:
                    # Condition changed: check if gap exceeded absence tolerance
                    if (timestamp - incident.last_seen_timestamp) > self.absence_tolerance_seconds:
                        self._close_incident(key)

        # Start an incident for any newly-observed condition.
        for event_type in sorted(active_types, key=lambda e: e.value):
            key = f"face_{event_type.value}"
            if key not in self.active_incidents:
                severity = self.severity_map.get(event_type, EventSeverity.HIGH)
                incident = ActiveIncident(
                    event_type=event_type,
                    severity=severity,
                    detector=detector,
                    start_timestamp=timestamp,
                    last_seen_timestamp=timestamp,
                    face_count=face_count,
                    similarity_score=similarity_score,
                    representative_bbox=primary_bbox,
                    best_confidence=primary_confidence,
                    best_frame_index=frame_index,
                    best_timestamp=timestamp,
                )
                incident.add_observation(
                    timestamp=timestamp,
                    frame_index=frame_index,
                    confidence=primary_confidence,
                    bbox=primary_bbox,
                    similarity_score=similarity_score,
                )
                self.active_incidents[key] = incident

    @staticmethod
    def _box_iou(
        box_a: tuple[int, int, int, int] | None,
        box_b: tuple[int, int, int, int] | None,
    ) -> float:
        """Compute Intersection over Union between two (x1, y1, x2, y2) boxes."""
        if box_a is None or box_b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        intersection = float(iw * ih)
        area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
        area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def update_object_observations(
        self,
        detected_objects: list[
            dict[str, Any]
        ],  # List of {"class_name": str, "confidence": float, "bbox": (x1,y1,x2,y2)}
        timestamp: float,
        frame_index: int,
        detector: DetectorInfo,
        blur_variance: float | None = None,
        hand_analysis: Any = None,
        tracked_objects: list[Any] | None = None,
    ) -> None:
        """Process detected objects for a frame using IoU spatial persistence."""
        seen_keys: set[str] = set()

        # Find all current active object incident keys
        active_obj_keys = [k for k in self.active_incidents if k.startswith("object_")]

        for obj in detected_objects:
            c_name = str(obj.get("class_name", "")).strip().lower()
            conf = float(obj.get("confidence", 0.0))
            raw_bbox = obj.get("bbox")
            bbox = tuple(int(v) for v in raw_bbox) if raw_bbox is not None else None

            if c_name == "person":
                continue  # Person counts are handled via person presence

            if c_name in ("cell phone", "phone", "mobile phone"):
                dis = self.phone_disambiguator.disambiguate(
                    phone_obj=obj, hand_analysis=hand_analysis, frame_index=frame_index
                )
                if dis.classification == PhoneClassification.HAND_FALSE_POSITIVE:
                    LOGGER.debug("Dismissed phone candidate as hand false positive: %s", dis.reason)
                    continue
                elif dis.classification in (
                    PhoneClassification.UNCERTAIN_CANDIDATE,
                    PhoneClassification.HAND_OBJECT_AMBIGUITY,
                ):
                    event_type = EventType.PHONE_CANDIDATE_UNCERTAIN
                    c_name = "phone_candidate_uncertain"
                    conf = dis.confidence
                else:
                    event_type = EventType.PHONE_DETECTED
                    conf = dis.confidence
            else:
                event_type = EventType.PROHIBITED_OBJECT

            required_dur = self.min_duration_for(event_type)

            # Match against existing active incidents of matching class via IoU
            matched_key: str | None = None
            best_iou = 0.0

            for k in active_obj_keys:
                inc = self.active_incidents[k]
                if inc.object_class == c_name and k not in seen_keys:
                    iou = self._box_iou(bbox, inc.representative_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        matched_key = k

            # If no strict IoU match but only one incident of this class exists, associate
            if matched_key is None:
                class_keys = [
                    k
                    for k in active_obj_keys
                    if self.active_incidents[k].object_class == c_name and k not in seen_keys
                ]
                if len(class_keys) == 1:
                    matched_key = class_keys[0]

            if matched_key is not None:
                seen_keys.add(matched_key)
                self.active_incidents[matched_key].add_observation(
                    timestamp=timestamp,
                    frame_index=frame_index,
                    confidence=conf,
                    bbox=bbox,
                    blur_variance=blur_variance,
                    required_duration=required_dur,
                )
            else:
                assoc_subj_id = None
                interaction = "unassociated"
                if tracked_objects and bbox is not None:
                    for to in tracked_objects:
                        to_bbox = getattr(to, "bbox", None)
                        if getattr(to, "class_name", "").lower() == c_name and self._box_iou(bbox, to_bbox) >= 0.20:
                            assoc_subj_id = getattr(to, "associated_subject_id", None)
                            interaction = getattr(to, "interaction_type", "unassociated")
                            break

                key = f"object_{c_name}"
                if assoc_subj_id is not None and assoc_subj_id > 1:
                    key = f"object_{c_name}_subj_{assoc_subj_id}"
                elif key in self.active_incidents:
                    # Disambiguate multiple instances of same class
                    key = f"object_{c_name}_{frame_index}"

                seen_keys.add(key)
                severity = self.severity_map.get(event_type, EventSeverity.MEDIUM)
                incident = ActiveIncident(
                    event_type=event_type,
                    severity=severity,
                    detector=detector,
                    start_timestamp=timestamp,
                    last_seen_timestamp=timestamp,
                    object_class=c_name,
                    representative_bbox=bbox,
                    best_confidence=conf,
                    best_frame_index=frame_index,
                    best_timestamp=timestamp,
                    associated_subject_id=assoc_subj_id,
                    interaction_type=interaction,
                )
                incident.add_observation(
                    timestamp=timestamp,
                    frame_index=frame_index,
                    confidence=conf,
                    bbox=bbox,
                    blur_variance=blur_variance,
                    required_duration=required_dur,
                )
                self.active_incidents[key] = incident

        # Check for inactive object incidents
        for k in active_obj_keys:
            if k not in seen_keys:
                incident = self.active_incidents[k]
                if (timestamp - incident.last_seen_timestamp) > self.absence_tolerance_seconds:
                    self._close_incident(k)

    def min_duration_for(self, event_type: EventType) -> float:
        """Seconds this event type must persist before it counts as qualified."""
        return self.min_duration_overrides.get(event_type, self.min_event_duration_seconds)

    def update_behaviour_observations(
        self,
        active_behaviours: dict[EventType, dict[str, Any]],
        timestamp: float,
        frame_index: int,
        detector: DetectorInfo,
    ) -> None:
        """Track continuous behavioural conditions (speaking, hands, gaze, wearables).

        ``active_behaviours`` maps each event type observed in this frame to a detail
        dict carrying at least ``confidence`` and optionally ``bbox`` and
        ``description``.  Anything absent from the mapping is treated as no longer
        observed, and its incident closes once the absence tolerance elapses — the
        same continuity rule the face and object channels use, so a one-frame
        dropout in a hand tracker does not split one incident into three.
        """
        seen_keys: set[str] = set()

        for event_type, detail in active_behaviours.items():
            key = f"behaviour_{event_type.value}"
            seen_keys.add(key)

            confidence = float(detail.get("confidence", 1.0))
            raw_bbox = detail.get("bbox")
            bbox = tuple(int(v) for v in raw_bbox) if raw_bbox is not None else None
            required_dur = self.min_duration_for(event_type)

            if key in self.active_incidents:
                self.active_incidents[key].add_observation(
                    timestamp=timestamp,
                    frame_index=frame_index,
                    confidence=confidence,
                    bbox=bbox,
                    required_duration=required_dur,
                )
                continue

            incident = ActiveIncident(
                event_type=event_type,
                severity=self.severity_map.get(event_type, EventSeverity.MEDIUM),
                detector=detector,
                start_timestamp=timestamp,
                last_seen_timestamp=timestamp,
                object_class=detail.get("object_class"),
                representative_bbox=bbox,
                best_confidence=confidence,
                best_frame_index=frame_index,
                best_timestamp=timestamp,
            )
            incident.add_observation(
                timestamp=timestamp,
                frame_index=frame_index,
                confidence=confidence,
                bbox=bbox,
                required_duration=required_dur,
            )
            # Carry the analyzer's own wording through to the closed event so the
            # observation text explains what was measured, not just which flag fired.
            if detail.get("description"):
                incident.detail_description = str(detail["description"])
            self.active_incidents[key] = incident

        for key in [k for k in self.active_incidents if k.startswith("behaviour_")]:
            if key in seen_keys:
                continue
            incident = self.active_incidents[key]
            if (timestamp - incident.last_seen_timestamp) > self.absence_tolerance_seconds:
                self._close_incident(key)

    def record_instant_event(
        self,
        event_type: EventType,
        timestamp: float,
        frame_index: int,
        description: str,
        detector: DetectorInfo,
        severity: EventSeverity | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EventRecord:
        """Record an instantaneous discrete event (e.g. BROWSER_TAB_SWITCH, FULLSCREEN_EXIT, SYSTEM_ERROR)."""
        eff_severity = severity or self.severity_map.get(event_type, EventSeverity.HIGH)
        event_id = self._generate_event_id(event_type)
        formatted_t = format_seconds_to_timestamp(timestamp)

        obs = ObservationDetail(
            description=description,
            frame_indices=[frame_index],
            timestamps=[timestamp],
            raw_confidences=[1.0],
        )

        event = EventRecord(
            event_id=event_id,
            session_id=self.session_id,
            timestamp=timestamp,
            end_timestamp=timestamp,
            duration=0.0,
            formatted_start=formatted_t,
            formatted_end=formatted_t,
            event_type=event_type,
            severity=eff_severity,
            confidence=1.0,
            average_confidence=1.0,
            detector=detector,
            observation=obs,
            metadata=metadata or {},
            status=EventStatus.QUALIFIED,  # Instantaneous events are immediately qualified
        )
        self.closed_events.append(event)
        return event

    def _close_incident(self, key: str) -> EventRecord | None:
        """Finalize an active incident and convert it to an EventRecord."""
        if key not in self.active_incidents:
            return None

        inc = self.active_incidents.pop(key)
        duration = max(0.0, inc.last_seen_timestamp - inc.start_timestamp)
        required_duration = self.min_duration_for(inc.event_type)
        is_qualified = duration >= required_duration

        event_id = self._generate_event_id(inc.event_type)
        avg_conf = float(np.mean(inc.confidences)) if inc.confidences else inc.best_confidence

        # Build factual, descriptive observation text (NO judgment or risk accusations)
        if inc.event_type == EventType.NO_FACE:
            desc = f"Face not detected for {duration:.2f}s across {len(inc.frame_indices)} frame(s)"
        elif inc.event_type == EventType.MULTIPLE_FACES:
            desc = f"Multiple faces ({inc.face_count or 2}) detected for {duration:.2f}s across {len(inc.frame_indices)} frame(s)"
        elif inc.event_type == EventType.UNKNOWN_FACE:
            sim_str = (
                f" (similarity: {inc.similarity_score:.2f})"
                if inc.similarity_score is not None
                else ""
            )
            desc = f"Unknown face detected for {duration:.2f}s{sim_str} across {len(inc.frame_indices)} frame(s)"
        elif inc.detail_description:
            desc = f"{inc.detail_description} — sustained {duration:.2f}s across {len(inc.frame_indices)} frame(s)"
        elif inc.object_class:
            desc = f"Object '{inc.object_class}' detected for {duration:.2f}s with max confidence {inc.best_confidence:.2f}"
        else:
            desc = f"Event '{inc.event_type.value}' observed for {duration:.2f}s across {len(inc.frame_indices)} frame(s)"

        obs = ObservationDetail(
            description=desc,
            object_class=inc.object_class,
            face_count=inc.face_count,
            similarity_score=inc.similarity_score,
            bounding_boxes=inc.bounding_boxes,
            frame_indices=inc.frame_indices,
            timestamps=inc.timestamps,
            raw_confidences=inc.confidences,
        )

        status = EventStatus.QUALIFIED if is_qualified else EventStatus.RECORDED

        event = EventRecord(
            event_id=event_id,
            session_id=self.session_id,
            timestamp=inc.start_timestamp,
            end_timestamp=inc.last_seen_timestamp,
            duration=duration,
            formatted_start=format_seconds_to_timestamp(inc.start_timestamp),
            formatted_end=format_seconds_to_timestamp(inc.last_seen_timestamp),
            event_type=inc.event_type,
            severity=inc.severity,
            confidence=inc.best_confidence,
            average_confidence=avg_conf,
            detector=inc.detector,
            observation=obs,
            metadata={
                "is_duration_qualified": is_qualified,
                "required_duration_seconds": round(required_duration, 3),
                "best_frame_index": inc.best_frame_index,
                "best_timestamp": inc.best_timestamp,
                "best_quality_score": round(inc.best_quality_score, 3),
                "representative_bbox": list(inc.representative_bbox)
                if inc.representative_bbox is not None
                else None,
                "total_observations": len(inc.frame_indices),
                "correlations": inc.correlations if inc.correlations else None,
            },
            status=status,
        )

        self.closed_events.append(event)
        return event

    def flush(self) -> list[EventRecord]:
        """Close all remaining active incidents at end of stream/video."""
        remaining_keys = list(self.active_incidents.keys())
        for k in remaining_keys:
            self._close_incident(k)
        return self.get_all_events()

    def get_all_events(self) -> list[EventRecord]:
        """Return all recorded and qualified events sorted chronologically."""
        return sorted(self.closed_events, key=lambda e: e.timestamp)

    def get_qualified_events(self) -> list[EventRecord]:
        """Return only events meeting duration or instantaneous qualification."""
        return [e for e in self.get_all_events() if e.status != EventStatus.RECORDED]
