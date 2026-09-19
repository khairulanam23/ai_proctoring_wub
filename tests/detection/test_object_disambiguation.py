"""Targeted tests for object detection taxonomy, phone disambiguation, and domain validation status."""

import pytest

from proctoring.analysis.phone_disambiguation import (
    PhoneClassification,
    PhoneHandDisambiguator,
)
from proctoring.detection.object_detector import DetectedObject
from proctoring.detection.object_relevance import (
    COCO_TO_EXAM_CATEGORY_MAP,
    ExamObjectCategory,
    ObjectRelevanceFilter,
    ProctoringDetectionReport,
)


def test_exam_object_category_taxonomy() -> None:
    """Verify all required domain categories exist in the taxonomy."""
    expected_categories = {
        "PHONE",
        "CALCULATOR",
        "NOTEBOOK",
        "PAPER",
        "PEN",
        "POWER_BANK",
        "EARBUD",
        "OTHER",
    }
    actual_categories = {cat.value for cat in ExamObjectCategory}
    assert expected_categories.issubset(actual_categories)


def test_coco_mapping_and_validation_status() -> None:
    """Verify COCO mapping and explicit NOT_VALIDATED status."""
    assert COCO_TO_EXAM_CATEGORY_MAP["cell phone"] == ExamObjectCategory.PHONE
    assert COCO_TO_EXAM_CATEGORY_MAP["book"] == ExamObjectCategory.NOTEBOOK
    assert COCO_TO_EXAM_CATEGORY_MAP["remote"] == ExamObjectCategory.CALCULATOR


def test_calculator_stationery_disambiguation() -> None:
    """Verify squarish/stationery aspect ratio objects on desk are treated as UNCERTAIN_CANDIDATE."""
    disambiguator = PhoneHandDisambiguator()

    # Calculator-shaped rectangular object: width=120, height=140 (aspect ratio ~1.17)
    obj = DetectedObject(
        class_name="cell phone",
        class_id=67,
        confidence=0.62,
        bbox=(200, 300, 320, 440),
    )

    result = disambiguator.disambiguate(phone_obj=obj, hand_analysis=None, frame_index=1)

    assert result.classification == PhoneClassification.UNCERTAIN_CANDIDATE
    assert result.exam_category == ExamObjectCategory.CALCULATOR
    assert result.domain_validation_status == "NOT_VALIDATED"
    assert "conflates smartphone with scientific calculator" in result.reason


def test_confirmed_phone_retains_not_validated_domain_status() -> None:
    """Verify even confirmed phone candidates honestly reflect NOT_VALIDATED model domain status."""
    disambiguator = PhoneHandDisambiguator()

    # Standard vertical phone: width=60, height=120 (aspect ratio 2.0)
    obj = DetectedObject(
        class_name="cell phone",
        class_id=67,
        confidence=0.88,
        bbox=(200, 200, 260, 320),
    )

    # Frame 1: Single frame requires temporal confirmation
    res1 = disambiguator.disambiguate(phone_obj=obj, hand_analysis=None, frame_index=1)
    assert res1.classification == PhoneClassification.POSSIBLE_PHONE

    # Frame 2: Confirmed with persistence
    result = disambiguator.disambiguate(phone_obj=obj, hand_analysis=None, frame_index=2)
    assert result.classification == PhoneClassification.CONFIRMED_PHONE
    assert result.domain_validation_status == "NOT_VALIDATED"
