"""Temporal event detection, state management, and Gantt timeline visualization."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from proctoring.detection.object_detector import DetectedObject
from proctoring.capture.video import format_timestamp


@dataclass
class ObjectPresenceEvent:
    """Consolidated temporal event representing sustained presence of an object class over time."""
    event_id: str
    object_class: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    formatted_start: str
    formatted_end: str
    detection_count: int
    max_confidence: float
    average_confidence: float
    is_duration_qualified: bool
    reason: str = ""
    observation_timestamps: List[float] = field(default_factory=list)
    representative_bbox: Optional[Tuple[int, int, int, int]] = None

    def __post_init__(self) -> None:
        if not self.reason:
            self.reason = f"Detected object class: {self.object_class.lower()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert temporal event to a JSON-serializable dictionary."""
        effective_reason = self.reason if self.reason else f"Detected object class: {self.object_class.lower()}"
        return {
            "event_id": self.event_id,
            "object_class": self.object_class,
            "reason": effective_reason,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "formatted_start": self.formatted_start,
            "formatted_end": self.formatted_end,
            "detection_count": self.detection_count,
            "max_confidence": round(self.max_confidence, 4),
            "average_confidence": round(self.average_confidence, 4),
            "is_duration_qualified": self.is_duration_qualified,
            "observation_timestamps": [round(t, 3) for t in self.observation_timestamps],
            "representative_bbox": list(self.representative_bbox) if self.representative_bbox else None,
        }


@dataclass
class PersonCountChangeEvent:
    """Discrete state transition event when the number of visible persons in scene changes."""
    event_id: str
    timestamp_seconds: float
    formatted_timestamp: str
    previous_count: int
    new_count: int
    frame_index: int
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.reason:
            self.reason = f"Person count changed from {self.previous_count} to {self.new_count}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert person count transition to a JSON-serializable dictionary."""
        effective_reason = self.reason if self.reason else f"Person count changed from {self.previous_count} to {self.new_count}"
        return {
            "event_id": self.event_id,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "formatted_timestamp": self.formatted_timestamp,
            "previous_count": self.previous_count,
            "new_count": self.new_count,
            "frame_index": self.frame_index,
            "reason": effective_reason,
        }


@dataclass
class TemporalEventReport:
    """Aggregated report containing consolidated temporal object events and person transitions."""
    events: List[ObjectPresenceEvent]
    person_count_changes: List[PersonCountChangeEvent]
    total_events: int
    qualified_events_count: int
    absence_tolerance_seconds: float
    min_event_duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert temporal report to a JSON-serializable dictionary."""
        return {
            "total_events": self.total_events,
            "qualified_events_count": self.qualified_events_count,
            "absence_tolerance_seconds": round(self.absence_tolerance_seconds, 2),
            "min_event_duration_seconds": round(self.min_event_duration_seconds, 2),
            "temporal_events": [e.to_dict() for e in self.events],
            "person_count_changes": [c.to_dict() for c in self.person_count_changes],
        }


class _ActiveTracker:
    """Internal state tracker for an ongoing object presence event."""

    def __init__(self, object_class: str, start_time: float, first_conf: float, first_bbox: Optional[Tuple[int, int, int, int]]) -> None:
        self.object_class = object_class
        self.start_time = start_time
        self.last_seen_time = start_time
        self.confidences: List[float] = [first_conf]
        self.timestamps: List[float] = [start_time]
        self.bboxes: List[Tuple[int, int, int, int]] = [first_bbox] if first_bbox else []

    def update(self, current_time: float, conf: float, bbox: Optional[Tuple[int, int, int, int]]) -> None:
        self.last_seen_time = current_time
        self.confidences.append(conf)
        self.timestamps.append(current_time)
        if bbox:
            self.bboxes.append(bbox)

    def close(self, event_id: str, min_duration: float) -> ObjectPresenceEvent:
        duration = max(0.0, self.last_seen_time - self.start_time)
        max_conf = max(self.confidences) if self.confidences else 0.0
        avg_conf = float(np.mean(self.confidences)) if self.confidences else 0.0
        is_qualified = duration >= min_duration

        # Representative bounding box: box associated with max confidence
        rep_bbox = None
        if self.bboxes and self.confidences:
            best_idx = int(np.argmax(self.confidences))
            if best_idx < len(self.bboxes):
                rep_bbox = self.bboxes[best_idx]
            else:
                rep_bbox = self.bboxes[0]

        reason_str = f"Detected object class: {self.object_class.lower()}"

        return ObjectPresenceEvent(
            event_id=event_id,
            object_class=self.object_class,
            start_seconds=self.start_time,
            end_seconds=self.last_seen_time,
            duration_seconds=duration,
            formatted_start=format_timestamp(self.start_time),
            formatted_end=format_timestamp(self.last_seen_time),
            detection_count=len(self.confidences),
            max_confidence=max_conf,
            average_confidence=avg_conf,
            is_duration_qualified=is_qualified,
            reason=reason_str,
            observation_timestamps=self.timestamps.copy(),
            representative_bbox=rep_bbox,
        )


class TemporalEventEngine:
    """State machine transforming frame-level observations into continuous temporal events."""

    def __init__(
        self,
        absence_tolerance_seconds: float = 1.0,
        min_event_duration_seconds: float = 1.0,
    ) -> None:
        self.absence_tolerance_seconds = max(0.0, float(absence_tolerance_seconds))
        self.min_event_duration_seconds = max(0.0, float(min_event_duration_seconds))

        self.active_trackers: Dict[str, _ActiveTracker] = {}
        self.completed_events: List[ObjectPresenceEvent] = []
        self.person_count_changes: List[PersonCountChangeEvent] = []
        self.last_person_count: Optional[int] = None
        self._event_counter = 1
        self._person_change_counter = 1

    def ingest_frame(
        self,
        timestamp_seconds: float,
        frame_index: int,
        person_count: int,
        detected_objects: List[DetectedObject],
    ) -> None:
        """Ingest a single sampled frame's detection telemetry into the temporal state machine."""
        t = float(timestamp_seconds)

        # 1. Track Person Count Transitions
        if self.last_person_count is None:
            self.last_person_count = person_count
        elif person_count != self.last_person_count:
            change_reason = f"Person count changed from {self.last_person_count} to {person_count}"
            change_event = PersonCountChangeEvent(
                event_id=f"person_change_{self._person_change_counter:03d}",
                timestamp_seconds=t,
                formatted_timestamp=format_timestamp(t),
                previous_count=self.last_person_count,
                new_count=person_count,
                frame_index=frame_index,
                reason=change_reason,
            )
            self.person_count_changes.append(change_event)
            self._person_change_counter += 1
            self.last_person_count = person_count

        # 2. Group detected objects by class name
        frame_classes: Dict[str, List[DetectedObject]] = {}
        for obj in detected_objects:
            cname = obj.class_name.lower().strip()
            if cname not in frame_classes:
                frame_classes[cname] = []
            frame_classes[cname].append(obj)

        # 3. Process active trackers for objects present in this frame
        for cname, objs in frame_classes.items():
            best_obj = max(objs, key=lambda o: o.confidence)

            if cname in self.active_trackers:
                tracker = self.active_trackers[cname]
                gap = t - tracker.last_seen_time

                if gap <= self.absence_tolerance_seconds:
                    # Bridge gap within tolerance
                    tracker.update(t, best_obj.confidence, best_obj.bbox)
                else:
                    # Absence tolerance exceeded -> close previous event and start fresh
                    closed_event = tracker.close(
                        event_id=f"obj_event_{self._event_counter:03d}",
                        min_duration=self.min_event_duration_seconds,
                    )
                    self.completed_events.append(closed_event)
                    self._event_counter += 1
                    self.active_trackers[cname] = _ActiveTracker(cname, t, best_obj.confidence, best_obj.bbox)
            else:
                # Start new active tracker
                self.active_trackers[cname] = _ActiveTracker(cname, t, best_obj.confidence, best_obj.bbox)

        # 4. Check for absent classes exceeding tolerance
        absent_classes = [cname for cname in self.active_trackers if cname not in frame_classes]
        for cname in absent_classes:
            tracker = self.active_trackers[cname]
            gap = t - tracker.last_seen_time
            if gap > self.absence_tolerance_seconds:
                closed_event = tracker.close(
                    event_id=f"obj_event_{self._event_counter:03d}",
                    min_duration=self.min_event_duration_seconds,
                )
                self.completed_events.append(closed_event)
                self._event_counter += 1
                del self.active_trackers[cname]

    def finalize(self, final_timestamp: Optional[float] = None) -> TemporalEventReport:
        """Close any open active trackers at end of video stream and generate final report."""
        for cname, tracker in list(self.active_trackers.items()):
            closed_event = tracker.close(
                event_id=f"obj_event_{self._event_counter:03d}",
                min_duration=self.min_event_duration_seconds,
            )
            self.completed_events.append(closed_event)
            self._event_counter += 1

        self.active_trackers.clear()

        # Sort chronologically by start timestamp
        self.completed_events.sort(key=lambda e: (e.start_seconds, e.object_class))
        self.person_count_changes.sort(key=lambda c: c.timestamp_seconds)

        qualified_count = sum(1 for e in self.completed_events if e.is_duration_qualified)

        return TemporalEventReport(
            events=self.completed_events,
            person_count_changes=self.person_count_changes,
            total_events=len(self.completed_events),
            qualified_events_count=qualified_count,
            absence_tolerance_seconds=self.absence_tolerance_seconds,
            min_event_duration_seconds=self.min_event_duration_seconds,
        )


def render_temporal_gantt_chart(
    report: TemporalEventReport,
    total_duration_seconds: float,
    output_path: Path,
) -> Path:
    """Render a publication-quality Gantt timeline visualization of temporal events and person counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dur = max(1.0, float(total_duration_seconds))
    distinct_classes = sorted(list({e.object_class for e in report.events}))
    if not distinct_classes:
        distinct_classes = ["(No relevant objects)"]

    num_classes = len(distinct_classes)
    fig, (ax_events, ax_person) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, max(4.5, 2.0 + num_classes * 0.6)),
        gridspec_kw={"height_ratios": [max(2, num_classes), 1.5]},
        sharex=True,
    )

    # Color palette
    color_map = {
        "person": "#2ecc71",
        "cell phone": "#e67e22",
        "laptop": "#3498db",
        "book": "#9b59b6",
        "tablet": "#1abc9c",
        "keyboard": "#f1c40f",
        "mouse": "#e74c3c",
        "backpack": "#34495e",
        "bottle": "#16a085",
    }

    # 1. Plot Object Presence Intervals (Gantt style)
    ax_events.set_title("Temporal Object Presence Evidence (Chronological Timeline)", fontsize=13, fontweight="bold", pad=12)
    y_positions = {cname: idx for idx, cname in enumerate(distinct_classes)}

    for event in report.events:
        y_pos = y_positions.get(event.object_class, 0)
        c = color_map.get(event.object_class.lower(), "#34495e")
        start_x = event.start_seconds
        width_x = max(0.12, event.duration_seconds)

        rect = patches.FancyBboxPatch(
            (start_x, y_pos - 0.25),
            width_x,
            0.50,
            boxstyle="round,pad=0.03",
            facecolor=c,
            edgecolor="#2c3e50",
            linewidth=1.2,
            alpha=0.90,
        )
        ax_events.add_patch(rect)

        # Label duration & event ID if wide enough
        if width_x >= 0.8:
            dur_label = f"{event.event_id}: {event.duration_seconds:.1f}s ({int(event.max_confidence * 100)}%)"
        else:
            dur_label = f"{event.event_id} ({event.duration_seconds:.1f}s)"

        label_x = start_x + width_x / 2.0
        ax_events.text(
            label_x,
            y_pos,
            dur_label,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="white" if event.object_class.lower() not in ["keyboard", "bottle"] else "black",
        )

    ax_events.set_yticks(range(num_classes))
    ax_events.set_yticklabels([c.title() for c in distinct_classes], fontsize=10, fontweight="bold")
    ax_events.set_ylim(-0.6, num_classes - 0.4)
    ax_events.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax_events.set_ylabel("Tracked Classes", fontsize=11, fontweight="bold")

    # 2. Plot Person Count Transition Step Chart
    ax_person.set_title("Person Count State Transitions (Factual Occupancy)", fontsize=11, fontweight="bold", pad=8)

    # Build piecewise constant person count steps
    time_pts = [0.0]
    count_pts = []

    initial_count = 1
    if report.person_count_changes:
        initial_count = report.person_count_changes[0].previous_count

    count_pts.append(initial_count)

    for chg in report.person_count_changes:
        time_pts.extend([chg.timestamp_seconds, chg.timestamp_seconds])
        count_pts.extend([chg.previous_count, chg.new_count])

    time_pts.append(dur)
    count_pts.append(count_pts[-1] if count_pts else initial_count)

    ax_person.step(time_pts, count_pts, where="post", color="#27ae60", linewidth=2.5, label="Person Count")
    ax_person.fill_between(time_pts, count_pts, step="post", color="#2ecc71", alpha=0.25)

    # Mark transition points with timestamp & transition count
    for chg in report.person_count_changes:
        ax_person.plot(chg.timestamp_seconds, chg.new_count, "o", color="#e74c3c", markersize=6)
        ax_person.annotate(
            f"{chg.formatted_timestamp}\n{chg.previous_count} → {chg.new_count} persons",
            (chg.timestamp_seconds, chg.new_count),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=8.0,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fff9e6", ec="#d4ac0d", lw=1),
        )

    ax_person.set_xlim(0, dur)
    ax_person.set_ylim(-0.2, max(3.5, max(count_pts) + 1.0 if count_pts else 3.0))
    ax_person.set_yticks([0, 1, 2, 3])
    ax_person.set_yticklabels(["0 Persons", "1 Person", "2 Persons", "3+ Persons"], fontsize=9)
    ax_person.set_xlabel("Video Timeline (Seconds)", fontsize=11, fontweight="bold")
    ax_person.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=180)
    plt.close(fig)

    return output_path
