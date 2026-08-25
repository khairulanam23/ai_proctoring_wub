"""Tests for the review-snapshot annotator.

The annotator writes images into the evidence package, so its failure modes matter:
a crash here must not cost an event its source evidence, and a partially-supplied
context must still yield a usable snapshot.
"""

import numpy as np
import pytest

from proctoring.core.events import (
    DetectorInfo,
    EventRecord,
    EventSeverity,
    EventType,
    ObservationDetail,
)
from proctoring.evidence.annotator import (
    SEVERITY_COLOURS,
    AnnotationContext,
    EvidenceAnnotator,
)


def _frame(size=(480, 640)) -> np.ndarray:
    return np.full((size[0], size[1], 3), 120, dtype=np.uint8)


def _event(event_type=EventType.NO_FACE, severity=EventSeverity.HIGH, **metadata) -> EventRecord:
    return EventRecord(
        event_id="evt_test_0001",
        session_id="s",
        timestamp=1.0,
        end_timestamp=3.5,
        duration=2.5,
        formatted_start="00:00:01.000",
        formatted_end="00:00:03.500",
        event_type=event_type,
        severity=severity,
        confidence=0.9,
        average_confidence=0.85,
        detector=DetectorInfo(name="test"),
        observation=ObservationDetail(description="Face not detected for 2.50s across 10 frame(s)"),
        metadata=metadata,
    )


@pytest.fixture
def annotator() -> EvidenceAnnotator:
    return EvidenceAnnotator()


def test_annotating_returns_a_new_image_of_the_same_shape(annotator):
    """The source frame must never be mutated — it is the package's source evidence."""
    frame = _frame()
    original = frame.copy()

    annotated = annotator.annotate_event(frame, _event())

    assert annotated.shape == frame.shape
    assert np.array_equal(frame, original), "annotator mutated the source frame"
    assert not np.array_equal(annotated, frame), "annotation drew nothing"


def test_annotation_works_with_no_context(annotator):
    """An event with no detection context still yields a readable snapshot."""
    annotated = annotator.annotate_event(_frame(), _event(), context=None)
    assert annotated.shape == (480, 640, 3)


def test_annotation_draws_every_supplied_detection(annotator):
    context = AnnotationContext(
        face_boxes=[(100, 100, 200, 200)],
        face_labels=["enrolled"],
        hand_landmarks=[np.full((21, 2), 50, dtype=np.float32)],
        hand_boxes=[(40, 40, 90, 90)],
        object_boxes=[("cell phone", 0.71, (250, 250, 320, 340))],
        wearable_boxes=[("headphones", 0.55, (90, 80, 220, 140))],
        ear_regions=[(95, 130, 125, 165)],
        mouth_region=(140, 170, 180, 195),
        measurements={"yaw": 12.5, "speaking": True, "hands": 1},
    )
    annotated = annotator.annotate_event(
        _frame(), _event(representative_bbox=[100, 100, 200, 200]), context
    )
    assert annotated.shape == (480, 640, 3)
    # Something was drawn across a wide area of the frame.
    assert (annotated != 120).any()


def test_every_severity_has_a_colour():
    """An unmapped severity would silently render in the fallback grey."""
    for severity in EventSeverity:
        assert severity in SEVERITY_COLOURS


def test_severity_changes_the_annotation(annotator):
    """Severity must be visually distinguishable, since it drives review triage."""
    low = annotator.annotate_event(_frame(), _event(severity=EventSeverity.LOW))
    critical = annotator.annotate_event(_frame(), _event(severity=EventSeverity.CRITICAL))
    assert not np.array_equal(low, critical)


def test_hand_skeleton_tolerates_incomplete_landmarks(annotator):
    """A truncated landmark array must be skipped, not crash the snapshot."""
    context = AnnotationContext(hand_landmarks=[np.zeros((5, 2), np.float32)])
    assert annotator.annotate_frame(_frame(), context).shape == (480, 640, 3)


def test_boxes_outside_the_frame_do_not_crash(annotator):
    """Detector boxes can exceed frame bounds after padding; drawing must survive it."""
    context = AnnotationContext(
        face_boxes=[(-50, -50, 900, 900)],
        object_boxes=[("book", 0.4, (600, 460, 800, 700))],
    )
    assert annotator.annotate_frame(_frame(), context).shape == (480, 640, 3)


def test_long_descriptions_are_wrapped_not_truncated_mid_render(annotator):
    event = _event()
    event.observation.description = "word " * 200
    assert annotator.annotate_event(_frame(), event).shape == (480, 640, 3)


def test_measurements_panel_handles_mixed_types(annotator):
    context = AnnotationContext(
        measurements={
            "yaw": 12.3456,
            "speaking": True,
            "hands": 2,
            "state": "LIVE",
            "missing": None,
        }
    )
    assert annotator.annotate_frame(_frame(), context).shape == (480, 640, 3)


def test_derived_notice_is_always_rendered(annotator):
    """Every review image must declare itself derived, so it is never mistaken for source."""
    plain = _frame()
    annotated = annotator.annotate_event(plain, _event())
    # The footer band is drawn over the bottom of the frame.
    assert not np.array_equal(annotated[-20:], plain[-20:])


def test_small_frames_are_handled(annotator):
    """Panels must clamp rather than index past the edge of a small frame."""
    assert annotator.annotate_event(_frame((120, 160)), _event()).shape == (120, 160, 3)
