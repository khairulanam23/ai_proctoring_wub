"""Renders review snapshots: the evidence frame with the detection drawn on it.

Every event stores the original unmodified frame for integrity, plus an annotated
copy for review.  The distinction is deliberate and important:

* ``evidence/frames/`` holds the **unaltered** capture.  It is what the SHA-256
  manifest attests to, and what an appeal or dispute must be judged against.
* ``evidence/review/`` holds a **derived, marked-up** copy showing what the system
  reacted to — the face box, the hand landmarks, the device box, the measured
  angles and the reason the event fired.

A proctor reviewing a hundred sessions needs the second; a candidate disputing a
finding is entitled to the first.  Annotated snapshots are labelled as derived in
the manifest so nobody mistakes one for source evidence.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from proctoring.core.events import EventRecord, EventSeverity

LOGGER = logging.getLogger(__name__)

# Severity-keyed BGR colours, matching the live HUD so the two read alike.
SEVERITY_COLOURS: dict[EventSeverity, tuple[int, int, int]] = {
    EventSeverity.INFO: (170, 170, 170),
    EventSeverity.LOW: (200, 180, 90),
    EventSeverity.MEDIUM: (60, 190, 250),
    EventSeverity.HIGH: (80, 90, 240),
    EventSeverity.CRITICAL: (90, 60, 200),
}

_PANEL_BG = (28, 28, 30)
_TEXT = (245, 245, 245)
_MUTED = (170, 170, 170)

# MediaPipe hand skeleton, for drawing a hand as a hand rather than 21 dots.
_HAND_CONNECTIONS: Sequence[tuple[int, int]] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


@dataclass
class AnnotationContext:
    """What was observed in the frame being annotated.

    Every field is optional: the annotator draws whatever it is given and silently
    omits the rest, so a caller with only a face box gets a valid snapshot.
    """

    face_boxes: Sequence[tuple[int, int, int, int]] = ()
    face_labels: Sequence[str] = ()
    hand_landmarks: Sequence[np.ndarray] = ()
    hand_boxes: Sequence[tuple[int, int, int, int]] = ()
    object_boxes: Sequence[tuple[str, float, tuple[int, int, int, int]]] = ()
    wearable_boxes: Sequence[tuple[str, float, tuple[int, int, int, int]]] = ()
    ear_regions: Sequence[tuple[int, int, int, int]] = ()
    mouth_region: tuple[int, int, int, int] | None = None
    measurements: dict[str, Any] | None = None
    """Free-form readings shown in the side panel, e.g. yaw / speech activity."""


class EvidenceAnnotator:
    """Draws review overlays onto a copy of an evidence frame."""

    def __init__(self, font_scale: float = 0.45, line_thickness: int = 2) -> None:
        self.font_scale = font_scale
        self.line_thickness = line_thickness

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_event(
        self,
        frame: np.ndarray,
        event: EventRecord,
        context: AnnotationContext | None = None,
    ) -> np.ndarray:
        """Return an annotated copy of ``frame`` explaining why ``event`` fired."""
        canvas = frame.copy()
        colour = SEVERITY_COLOURS.get(event.severity, _MUTED)

        if context is not None:
            self._draw_detections(canvas, context)

        # Highlight the specific region the event is about, when one is recorded.
        representative = event.metadata.get("representative_bbox")
        if representative:
            x1, y1, x2, y2 = (int(v) for v in representative)
            self._draw_emphasis_box(canvas, (x1, y1, x2, y2), colour)

        self._draw_header(canvas, event, colour)
        self._draw_footer(canvas, event)
        if context is not None and context.measurements:
            self._draw_measurements(canvas, context.measurements)
        return canvas

    def annotate_frame(
        self,
        frame: np.ndarray,
        context: AnnotationContext,
        caption: str | None = None,
    ) -> np.ndarray:
        """Annotate a frame with detections but no specific event (live HUD use)."""
        canvas = frame.copy()
        self._draw_detections(canvas, context)
        if caption:
            self._draw_banner(canvas, caption, _MUTED)
        if context.measurements:
            self._draw_measurements(canvas, context.measurements)
        return canvas

    # ------------------------------------------------------------------
    # Detection overlays
    # ------------------------------------------------------------------

    def _draw_detections(self, canvas: np.ndarray, context: AnnotationContext) -> None:
        # Ear and mouth regions first, so device and hand boxes draw on top.
        for region in context.ear_regions:
            self._dashed_box(canvas, region, (140, 140, 140))
        if context.mouth_region:
            self._dashed_box(canvas, context.mouth_region, (140, 140, 140))

        for index, box in enumerate(context.face_boxes):
            label = context.face_labels[index] if index < len(context.face_labels) else "face"
            self._box_with_label(canvas, box, label, (90, 210, 120))

        for points in context.hand_landmarks:
            self._draw_hand_skeleton(canvas, points)
        for box in context.hand_boxes:
            self._box_with_label(canvas, box, "hand", (250, 200, 90), thickness=1)

        for name, confidence, box in context.object_boxes:
            self._box_with_label(canvas, box, f"{name} {confidence:.2f}", (80, 90, 240))
        for name, confidence, box in context.wearable_boxes:
            self._box_with_label(canvas, box, f"{name} {confidence:.2f}", (200, 90, 220))

    def _draw_hand_skeleton(self, canvas: np.ndarray, points: np.ndarray) -> None:
        """Draw the hand as a connected skeleton so posture is readable at a glance."""
        if points is None or len(points) < 21:
            return
        for start, end in _HAND_CONNECTIONS:
            cv2.line(
                canvas,
                (int(points[start][0]), int(points[start][1])),
                (int(points[end][0]), int(points[end][1])),
                (250, 200, 90),
                1,
                cv2.LINE_AA,
            )
        for point in points:
            cv2.circle(canvas, (int(point[0]), int(point[1])), 2, (255, 230, 150), -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Chrome
    # ------------------------------------------------------------------

    def _draw_header(
        self, canvas: np.ndarray, event: EventRecord, colour: tuple[int, int, int]
    ) -> None:
        """Title bar: what fired, how severe, and when."""
        width = canvas.shape[1]
        self._panel(canvas, 0, 0, width, 52)
        cv2.rectangle(canvas, (0, 0), (6, 52), colour, -1)

        cv2.putText(
            canvas,
            event.event_type.value.replace("_", " "),
            (16, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            cv2.LINE_AA,
        )
        detail = (
            f"{event.severity.value} · {event.formatted_start}"
            f" → {event.formatted_end} · {event.duration:.1f}s"
        )
        cv2.putText(
            canvas, detail, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _MUTED, 1, cv2.LINE_AA
        )

    def _draw_footer(self, canvas: np.ndarray, event: EventRecord) -> None:
        """Factual observation text, wrapped to the frame width."""
        height, width = canvas.shape[:2]
        lines = self._wrap(event.observation.description, width - 32)
        panel_height = 16 * len(lines) + 26
        self._panel(canvas, 0, height - panel_height, width, panel_height)

        y = height - panel_height + 18
        for line in lines:
            cv2.putText(
                canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _TEXT, 1, cv2.LINE_AA
            )
            y += 16
        cv2.putText(
            canvas,
            "Derived review image — source frame is stored unmodified",
            (16, height - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            _MUTED,
            1,
            cv2.LINE_AA,
        )

    def _draw_measurements(self, canvas: np.ndarray, measurements: dict[str, Any]) -> None:
        """Right-hand readout of the numbers behind the decision."""
        rows = [(k, self._format_value(v)) for k, v in measurements.items() if v is not None]
        if not rows:
            return
        height, width = canvas.shape[:2]
        panel_width, panel_height = 178, 18 * len(rows) + 14
        x0, y0 = width - panel_width - 8, 60
        self._panel(canvas, x0, y0, panel_width, min(panel_height, height - y0 - 8))

        y = y0 + 18
        for name, value in rows:
            if y > height - 12:
                break
            cv2.putText(
                canvas,
                name[:14],
                (x0 + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                _MUTED,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                value[:11],
                (x0 + 108, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                _TEXT,
                1,
                cv2.LINE_AA,
            )
            y += 18

    def _draw_banner(self, canvas: np.ndarray, text: str, colour: tuple[int, int, int]) -> None:
        self._panel(canvas, 0, 0, canvas.shape[1], 34)
        cv2.putText(canvas, text, (14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    @staticmethod
    def _panel(canvas: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.74) -> None:
        x2, y2 = min(x + w, canvas.shape[1]), min(y + h, canvas.shape[0])
        x, y = max(0, x), max(0, y)
        if x2 <= x or y2 <= y:
            return
        region = canvas[y:y2, x:x2]
        cv2.addWeighted(
            np.full(region.shape, _PANEL_BG, np.uint8), alpha, region, 1 - alpha, 0, region
        )

    def _box_with_label(
        self,
        canvas: np.ndarray,
        box: tuple[int, int, int, int],
        label: str,
        colour: tuple[int, int, int],
        thickness: int | None = None,
    ) -> None:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(
            canvas, (x1, y1), (x2, y2), colour, thickness or self.line_thickness, cv2.LINE_AA
        )
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
        ly = max(th + 4, y1 - 4)
        cv2.rectangle(canvas, (x1, ly - th - 4), (x1 + tw + 8, ly + 3), colour, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 4, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    def _draw_emphasis_box(
        self,
        canvas: np.ndarray,
        box: tuple[int, int, int, int],
        colour: tuple[int, int, int],
    ) -> None:
        """Corner brackets marking the exact region the event is about."""
        x1, y1, x2, y2 = box
        length = max(12, min(x2 - x1, y2 - y1) // 4)
        for (cx, cy), (dx, dy) in (
            ((x1, y1), (1, 1)),
            ((x2, y1), (-1, 1)),
            ((x1, y2), (1, -1)),
            ((x2, y2), (-1, -1)),
        ):
            cv2.line(canvas, (cx, cy), (cx + dx * length, cy), colour, 3, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (cx, cy + dy * length), colour, 3, cv2.LINE_AA)

    @staticmethod
    def _dashed_box(
        canvas: np.ndarray,
        box: tuple[int, int, int, int],
        colour: tuple[int, int, int],
        dash: int = 6,
    ) -> None:
        x1, y1, x2, y2 = (int(v) for v in box)
        for x in range(x1, x2, dash * 2):
            cv2.line(canvas, (x, y1), (min(x + dash, x2), y1), colour, 1)
            cv2.line(canvas, (x, y2), (min(x + dash, x2), y2), colour, 1)
        for y in range(y1, y2, dash * 2):
            cv2.line(canvas, (x1, y), (x1, min(y + dash, y2)), colour, 1)
            cv2.line(canvas, (x2, y), (x2, min(y + dash, y2)), colour, 1)

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    @staticmethod
    def _wrap(text: str, pixel_width: int, char_px: int = 7) -> list[str]:
        """Greedy wrap using an approximate character width."""
        max_chars = max(20, pixel_width // char_px)
        words, lines, current = text.split(), [], ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines[:3]
