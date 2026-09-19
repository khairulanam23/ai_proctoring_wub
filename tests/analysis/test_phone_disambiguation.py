"""Tests for contextual phone-vs-hand disambiguation."""

import numpy as np
import pytest

from proctoring.analysis.hands import HandAnalysisResult, HandObservation
from proctoring.analysis.phone_disambiguation import (
    PhoneClassification,
    PhoneHandDisambiguator,
)
from proctoring.detection.object_detector import DetectedObject


def test_high_confidence_phone_requires_temporal_or_hand_evidence():
    """Verify high raw confidence alone does not confirm without contextual or temporal evidence."""
    disambiguator = PhoneHandDisambiguator()
    phone = DetectedObject(
        class_id=67,
        class_name="cell phone",
        confidence=0.92,
        bbox=(200, 200, 280, 360),  # width 80, height 160 -> aspect ratio 2.0
    )
    # Frame 1: Raw confidence alone without hand grip yields POSSIBLE_PHONE, not CONFIRMED_PHONE
    res1 = disambiguator.disambiguate(phone, hand_analysis=None, frame_index=1)
    assert res1.classification == PhoneClassification.POSSIBLE_PHONE
    assert res1.aspect_ratio == pytest.approx(2.0, 0.05)

    # Frame 2: Temporal persistence across frames confirms
    res2 = disambiguator.disambiguate(phone, hand_analysis=None, frame_index=2)
    assert res2.classification == PhoneClassification.CONFIRMED_PHONE
    assert res2.aspect_ratio == pytest.approx(2.0, 0.05)


def test_irregular_aspect_ratio_uncertain():
    disambiguator = PhoneHandDisambiguator()
    # Hyper-elongated bounding box (aspect ratio 3.6) with moderate confidence 0.55
    phone = DetectedObject(
        class_id=67,
        class_name="cell phone",
        confidence=0.55,
        bbox=(200, 200, 250, 380),  # width 50, height 180 -> aspect ratio 3.6
    )
    res = disambiguator.disambiguate(phone, hand_analysis=None, frame_index=1)
    assert res.classification == PhoneClassification.UNCERTAIN_CANDIDATE
    assert "Irregular aspect ratio" in res.reason


def test_empty_hand_false_positive_dismissed():
    disambiguator = PhoneHandDisambiguator()

    # Hand bounding box covering (200, 200, 320, 380)
    # Simulate empty open hand with splayed fingertips far from palm
    landmarks = np.zeros((21, 2), dtype=np.float32)
    # Wrist
    landmarks[0] = [260, 370]
    # MCPs
    landmarks[5] = [240, 300]
    landmarks[17] = [280, 300]
    # Splayed extended fingertips
    landmarks[4] = [210, 260]  # thumb
    landmarks[8] = [230, 210]  # index
    landmarks[12] = [260, 205] # middle
    landmarks[16] = [290, 215] # ring
    landmarks[20] = [315, 230] # pinky

    hand = HandObservation(
        handedness="Right",
        confidence=0.95,
        bbox=(200, 200, 320, 380),
        centroid=(260, 290),
        landmarks=landmarks,
        fingertip_points=[(230, 210), (260, 205)],
    )
    hand_res = HandAnalysisResult(hands=[hand], hands_detected=1, hands_visible=True)

    # Phone candidate detected squarely inside the empty hand with marginal confidence
    phone = DetectedObject(
        class_id=67,
        class_name="cell phone",
        confidence=0.52,
        bbox=(210, 210, 310, 370),
    )

    res = disambiguator.disambiguate(phone, hand_analysis=hand_res, frame_index=1)
    assert res.classification == PhoneClassification.HAND_FALSE_POSITIVE
    assert res.hand_iou > 0.65


def test_gripping_hand_confirmed():
    disambiguator = PhoneHandDisambiguator()

    # Hand curled around phone
    landmarks = np.zeros((21, 2), dtype=np.float32)
    landmarks[0] = [260, 360]
    landmarks[5] = [240, 310]
    landmarks[17] = [280, 310]
    # Curled fingertips touching phone body
    landmarks[4] = [225, 300]
    landmarks[8] = [235, 270]
    landmarks[12] = [255, 265]
    landmarks[16] = [275, 270]
    landmarks[20] = [285, 290]

    hand = HandObservation(
        handedness="Right",
        confidence=0.95,
        bbox=(210, 250, 300, 370),
        centroid=(255, 310),
        landmarks=landmarks,
        fingertip_points=[(235, 270)],
    )
    hand_res = HandAnalysisResult(hands=[hand], hands_detected=1, hands_visible=True)

    phone = DetectedObject(
        class_id=67,
        class_name="cell phone",
        confidence=0.68,
        bbox=(220, 220, 290, 350),  # Aspect ratio ~ 1.85
    )

    res = disambiguator.disambiguate(phone, hand_analysis=hand_res, frame_index=1)
    assert res.classification == PhoneClassification.CONFIRMED_PHONE
    assert res.hand_gripping is True
