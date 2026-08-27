"""Evidence retention buffer and attachment coordinator for examination sessions.

Manages in-memory best-frame retention, ROI crop requests, review snapshot annotation,
and disk validation across open and closed proctoring events.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import numpy as np

from proctoring.analysis.observer import BehaviourObserver
from proctoring.config import SessionConfig
from proctoring.core.events import EventRecord
from proctoring.evidence.annotator import AnnotationContext, EvidenceAnnotator
from proctoring.evidence.manager import EvidenceManager
from proctoring.observation import FrameObservation

LOGGER = logging.getLogger(__name__)


class EvidenceCoordinator:
    """Coordinates in-memory frame retention, evidence generation, and review snapshots."""

    def __init__(
        self,
        config: SessionConfig,
        evidence_manager: EvidenceManager,
        annotator: EvidenceAnnotator,
        observer: BehaviourObserver,
    ) -> None:
        self.config = config
        self.evidence_manager = evidence_manager
        self.annotator = annotator
        self.observer = observer

        # In-memory best-frame retention buffer
        self._evidence_frames: OrderedDict[int, np.ndarray] = OrderedDict()
        self._frame_contexts: dict[int, AnnotationContext] = {}
        self._last_accepted_frame: np.ndarray | None = None
        self._last_accepted_index: int = 0

    def reset(self) -> None:
        """Clear all buffered frames and reset accepted state."""
        self._evidence_frames.clear()
        self._frame_contexts.clear()
        self._last_accepted_frame = None
        self._last_accepted_index = 0

    def record_accepted_frame(self, frame: np.ndarray, frame_index: int) -> None:
        """Cache the most recent valid frame for fallback pairing with instant events."""
        self._last_accepted_frame = frame
        self._last_accepted_index = frame_index

    def retain_evidence_frames(
        self,
        frame: np.ndarray,
        frame_index: int,
        obs: FrameObservation | None,
        active_incidents: dict[Any, Any],
        closed_events: list[EventRecord],
    ) -> None:
        """Keep a copy of any frame that is currently the best example of an incident.

        Frames no longer referenced by open incidents or closed events are evicted.
        """
        needed = self.referenced_frame_indices(active_incidents, closed_events)
        if frame_index in needed and frame_index not in self._evidence_frames:
            self._evidence_frames[frame_index] = frame.copy()
            if obs is not None and self.config.capture_review_snapshots:
                self._frame_contexts[frame_index] = self.observer.build_annotation_context(obs)

        # Release stale frames
        for stale in [i for i in self._evidence_frames if i not in needed]:
            del self._evidence_frames[stale]
            self._frame_contexts.pop(stale, None)

        # Hard ceiling on retained frames
        while len(self._evidence_frames) > self.config.max_retained_evidence_frames:
            evicted, _ = self._evidence_frames.popitem(last=False)
            self._frame_contexts.pop(evicted, None)

    def referenced_frame_indices(
        self,
        active_incidents: dict[Any, Any],
        closed_events: list[EventRecord],
    ) -> set[int]:
        """Frame indices still needed to illustrate an open incident or closed event."""
        indices = {inc.best_frame_index for inc in active_incidents.values()}
        for event in closed_events:
            best = event.metadata.get("best_frame_index")
            if best is not None:
                indices.add(int(best))
        return indices

    def attach_evidence_to_events(self, events: list[EventRecord]) -> None:
        """Attach to each event the exact frame that demonstrated its condition."""
        if not self.config.capture_evidence:
            return

        for event in events:
            best_index = event.metadata.get("best_frame_index")
            best_timestamp = event.metadata.get("best_timestamp", event.timestamp)
            raw_bbox = event.metadata.get("representative_bbox")
            bbox = tuple(raw_bbox) if raw_bbox else None
            label = event.observation.object_class or event.event_type.value.lower()

            frame = None
            if best_index is not None:
                frame = self._evidence_frames.get(int(best_index))

            if frame is None:
                if self._last_accepted_frame is not None and event.duration == 0.0:
                    frame = self._last_accepted_frame
                    best_index = self._last_accepted_index
                    event.metadata["evidence_frame_is_nearest_available"] = True
                else:
                    event.metadata["evidence_unavailable"] = (
                        "No retained frame for this event's best_frame_index"
                    )
                    continue

            resolved_index = int(best_index) if best_index is not None else 0
            resolved_ts = float(best_timestamp) if best_timestamp is not None else 0.0

            self.evidence_manager.attach_evidence_to_event(
                event=event,
                frame=frame,
                frame_index=resolved_index,
                timestamp_seconds=resolved_ts,
                bbox=bbox,
                label=label,
            )

            if self.config.capture_review_snapshots:
                self.attach_review_snapshot(
                    event=event,
                    frame=frame,
                    frame_index=resolved_index,
                    timestamp_seconds=resolved_ts,
                )

    def attach_review_snapshot(
        self,
        event: EventRecord,
        frame: np.ndarray,
        frame_index: int,
        timestamp_seconds: float,
    ) -> None:
        """Render and attach an annotated review copy of an event's evidence frame."""
        try:
            context = self._frame_contexts.get(frame_index)
            annotated = self.annotator.annotate_event(frame, event, context)
            reference = self.evidence_manager.save_review_snapshot(
                annotated_frame=annotated,
                event_id=event.event_id,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
            )
            if reference is not None:
                reference.validation_error = None
                event.evidence.append(reference)
                event.metadata["has_review_snapshot"] = True
        except Exception as exc:
            LOGGER.warning("Review snapshot failed for %s: %s", event.event_id, exc)
            event.metadata["review_snapshot_error"] = str(exc)

    def validate_all(self, events: list[EventRecord], prune_invalid: bool = True) -> dict[str, Any]:
        """Validate all saved evidence files on disk."""
        return self.evidence_manager.validate_all_event_evidence(
            events, prune_invalid=prune_invalid
        )

    def clear(self) -> None:
        """Release memory buffers after finalization."""
        self._evidence_frames.clear()
        self._frame_contexts.clear()
