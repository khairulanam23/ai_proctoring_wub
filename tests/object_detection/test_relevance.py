"""Unit tests for ObjectRelevanceFilter, ProctoringDetectionReport, and class-specific thresholding."""

import json
import numpy as np
import pytest

from src.object_detection.detector import DetectedObject, ObjectDetectionResult
from src.object_detection.relevance import (
    DEFAULT_CLASS_THRESHOLDS,
    DEFAULT_PROCTORING_RELEVANT_CLASSES,
    ObjectRelevanceFilter,
    ProctoringDetectionReport,
)


@pytest.fixture
def relevance_filter() -> ObjectRelevanceFilter:
    return ObjectRelevanceFilter()


def make_raw_result(objects: list[DetectedObject]) -> ObjectDetectionResult:
    return ObjectDetectionResult(
        objects=objects,
        count=len(objects),
        image_shape=(480, 640, 3),
        image_width=640,
        image_height=480,
        inference_time_ms=10.0,
        model_name="yolo11n.pt",
        device="cpu",
    )


def test_relevance_filter_initialization(relevance_filter: ObjectRelevanceFilter) -> None:
    """Test default relevance classes and threshold initialization."""
    assert "person" in relevance_filter.relevant_classes
    assert "cell phone" in relevance_filter.relevant_classes
    assert "laptop" in relevance_filter.relevant_classes
    assert "book" in relevance_filter.relevant_classes
    assert "tie" not in relevance_filter.relevant_classes
    assert "chair" not in relevance_filter.relevant_classes
    assert relevance_filter.get_threshold("person") == 0.25
    assert relevance_filter.get_threshold("cell phone") == 0.40


def test_relevance_filtering_classes(relevance_filter: ObjectRelevanceFilter) -> None:
    """Test separation of relevant proctoring objects from background objects."""
    raw = make_raw_result([
        DetectedObject(class_id=0, class_name="person", confidence=0.92, bbox=(10, 10, 100, 200)),
        DetectedObject(class_id=67, class_name="cell phone", confidence=0.85, bbox=(20, 20, 50, 80)),
        DetectedObject(class_id=32, class_name="tie", confidence=0.91, bbox=(40, 60, 55, 120)),
        DetectedObject(class_id=56, class_name="chair", confidence=0.74, bbox=(100, 100, 300, 400)),
    ])

    report = relevance_filter.filter(raw)

    assert report.total_detections == 4
    assert report.relevant_count == 2
    assert report.person_count == 1

    relevant_names = [o.class_name for o in report.relevant_objects]
    assert "person" in relevant_names
    assert "cell phone" in relevant_names
    assert "tie" not in relevant_names
    assert "chair" not in relevant_names

    ignored_names = [o.class_name for o in report.ignored_objects]
    assert "tie" in ignored_names
    assert "chair" in ignored_names


def test_class_specific_confidence_threshold_override() -> None:
    """Test that class-specific thresholds override global default threshold."""
    custom_filter = ObjectRelevanceFilter(
        default_threshold=0.20,
        class_thresholds={"cell phone": 0.50, "laptop": 0.30},
    )

    raw = make_raw_result([
        # cell phone at 0.45: above default (0.20) but below class threshold (0.50) -> filtered
        DetectedObject(class_id=67, class_name="cell phone", confidence=0.45, bbox=(10, 10, 50, 50)),
        # laptop at 0.35: above class threshold (0.30) -> retained
        DetectedObject(class_id=63, class_name="laptop", confidence=0.35, bbox=(60, 60, 150, 150)),
    ])

    report = custom_filter.filter(raw)
    assert report.relevant_count == 1
    assert report.relevant_objects[0].class_name == "laptop"
    assert len(report.ignored_objects) == 1
    assert report.ignored_objects[0].class_name == "cell phone"


def test_person_counting(relevance_filter: ObjectRelevanceFilter) -> None:
    """Test person count calculations for 0, 1, and multiple persons."""
    # 0 persons
    report0 = relevance_filter.filter(make_raw_result([
        DetectedObject(class_id=67, class_name="cell phone", confidence=0.88, bbox=(10, 10, 50, 50)),
    ]))
    assert report0.person_count == 0

    # 1 person
    report1 = relevance_filter.filter(make_raw_result([
        DetectedObject(class_id=0, class_name="person", confidence=0.91, bbox=(10, 10, 100, 200)),
    ]))
    assert report1.person_count == 1

    # 2 persons
    report2 = relevance_filter.filter(make_raw_result([
        DetectedObject(class_id=0, class_name="person", confidence=0.91, bbox=(10, 10, 100, 200)),
        DetectedObject(class_id=0, class_name="person", confidence=0.86, bbox=(200, 10, 300, 200)),
    ]))
    assert report2.person_count == 2


def test_multiple_simultaneous_relevant_objects(relevance_filter: ObjectRelevanceFilter) -> None:
    """Test preservation of multiple simultaneous relevant objects."""
    raw = make_raw_result([
        DetectedObject(class_id=0, class_name="person", confidence=0.95, bbox=(10, 10, 100, 200)),
        DetectedObject(class_id=67, class_name="cell phone", confidence=0.88, bbox=(30, 40, 60, 90)),
        DetectedObject(class_id=63, class_name="laptop", confidence=0.82, bbox=(120, 150, 300, 280)),
        DetectedObject(class_id=73, class_name="book", confidence=0.75, bbox=(320, 150, 400, 220)),
    ])

    report = relevance_filter.filter(raw)
    assert report.relevant_count == 4
    assert report.person_count == 1
    assert len(report.ignored_objects) == 0


def test_proctoring_report_serialization(relevance_filter: ObjectRelevanceFilter) -> None:
    """Test JSON serialization of ProctoringDetectionReport."""
    raw = make_raw_result([
        DetectedObject(class_id=0, class_name="person", confidence=0.93, bbox=(10, 10, 100, 200)),
        DetectedObject(class_id=32, class_name="tie", confidence=0.90, bbox=(30, 50, 45, 90)),
    ])

    report = relevance_filter.filter(raw, timestamp=1700000000.0)
    data = report.to_dict()

    assert data["timestamp"] == 1700000000.0
    assert data["total_detections"] == 2
    assert data["relevant_count"] == 1
    assert data["person_count"] == 1
    assert len(data["relevant_objects"]) == 1
    assert len(data["ignored_objects"]) == 1
    assert "raw_result" in data

    # Verify JSON compatibility
    json_str = json.dumps(data)
    assert "person" in json_str
    assert "tie" in json_str


def test_relevance_visualizer(relevance_filter: ObjectRelevanceFilter) -> None:
    """Test visual rendering of relevant objects and show_all debug mode."""
    raw = make_raw_result([
        DetectedObject(class_id=0, class_name="person", confidence=0.95, bbox=(10, 10, 100, 200)),
        DetectedObject(class_id=32, class_name="tie", confidence=0.90, bbox=(30, 50, 45, 90)),
    ])
    report = relevance_filter.filter(raw)

    dummy_image = np.zeros((300, 400, 3), dtype=np.uint8)

    # Default mode
    vis_default = relevance_filter.visualize_report(dummy_image, report, show_all=False)
    assert vis_default is not None
    assert vis_default.shape[1] == 400
    assert vis_default.shape[0] == 300 + 32  # includes top status bar

    # Debug mode (show_all=True)
    vis_debug = relevance_filter.visualize_report(dummy_image, report, show_all=True)
    assert vis_debug is not None
    assert vis_debug.shape[1] == 400
    assert vis_debug.shape[0] == 300 + 32
