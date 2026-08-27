"""Tests for face occlusion classification and camera obstruction analysis."""

from proctoring.analysis.hands import HandAnalysisResult, HandObservation
from proctoring.analysis.occlusion import (
    FaceOcclusionClassifier,
    FaceOcclusionState,
)


def test_occlusion_no_face_normal_scene() -> None:
    classifier = FaceOcclusionClassifier()
    # Normal lighting, good variance, but no candidate in frame
    res = classifier.classify(
        face_found=False,
        face_bbox=None,
        mean_luminance=120.0,
        blur_variance=150.0,
    )
    assert res.state == FaceOcclusionState.FACE_MISSING
    assert not res.camera_obstructed
    assert not res.is_occluded
    assert res.confidence >= 0.90


def test_occlusion_camera_obstructed_dark_and_featureless() -> None:
    classifier = FaceOcclusionClassifier(min_camera_dark_luma=15.0, min_camera_blur_var=20.0)
    # Lens covered: near pitch black, no edges
    res = classifier.classify(
        face_found=False,
        face_bbox=None,
        mean_luminance=3.2,
        blur_variance=4.5,
    )
    assert res.state == FaceOcclusionState.CAMERA_OBSTRUCTED
    assert res.camera_obstructed is True
    assert res.is_occluded is True
    assert res.confidence >= 0.85


def test_occlusion_face_unoccluded() -> None:
    classifier = FaceOcclusionClassifier()
    face_box = (100, 100, 300, 300)
    # Hand far away at corner (x=500, y=500)
    hand = HandObservation(
        handedness="Right",
        confidence=0.9,
        bbox=(500, 500, 580, 580),
        centroid=(540, 540),
    )
    hands_res = HandAnalysisResult(hands_detected=1, hands=[hand], hand_near_face=False)

    res = classifier.classify(
        face_found=True,
        face_bbox=face_box,
        hands=hands_res,
    )
    assert res.state == FaceOcclusionState.NONE
    assert not res.is_occluded
    assert res.eyes_visible is True
    assert res.mouth_visible is True
    assert res.hand_overlap_ratio == 0.0


def test_occlusion_mouth_nose_obscured() -> None:
    classifier = FaceOcclusionClassifier()
    face_box = (100, 100, 300, 300)  # w=200, h=200, mouth zone roughly (130, 190, 270, 300)
    # Hand covering lower face
    hand = HandObservation(
        handedness="Right",
        confidence=0.9,
        bbox=(120, 200, 280, 300),
        centroid=(200, 250),
    )
    hands_res = HandAnalysisResult(
        hands_detected=1,
        hands=[hand],
        hand_near_face=True,
        hand_near_mouth=True,
    )

    res = classifier.classify(
        face_found=True,
        face_bbox=face_box,
        hands=hands_res,
    )
    assert res.state in (
        FaceOcclusionState.MOUTH_NOSE_OBSCURED,
        FaceOcclusionState.FACE_PARTIALLY_OBSCURED,
        FaceOcclusionState.FACE_SIGNIFICANTLY_OBSCURED,
    )
    assert res.is_occluded is True
    assert res.mouth_visible is False


def test_occlusion_eyes_obscured() -> None:
    classifier = FaceOcclusionClassifier()
    face_box = (100, 100, 300, 300)  # eyes zone roughly (100, 130, 300, 200)
    # Hand covering upper face / eyes
    hand = HandObservation(
        handedness="Left",
        confidence=0.9,
        bbox=(100, 130, 300, 190),
        centroid=(200, 160),
    )
    hands_res = HandAnalysisResult(
        hands_detected=1,
        hands=[hand],
        hand_near_face=True,
    )

    res = classifier.classify(
        face_found=True,
        face_bbox=face_box,
        hands=hands_res,
    )
    assert res.is_occluded is True
    assert res.eyes_visible is False


def test_occlusion_significant_overlap() -> None:
    classifier = FaceOcclusionClassifier(min_hand_overlap_significant=0.40)
    face_box = (100, 100, 300, 300)  # area = 40000
    # Large hand covering almost entire face (100, 100, 300, 300)
    hand = HandObservation(
        handedness="Left",
        confidence=0.95,
        bbox=(100, 100, 300, 280),
        centroid=(200, 190),
    )
    hands_res = HandAnalysisResult(
        hands_detected=1,
        hands=[hand],
        hand_near_face=True,
    )

    res = classifier.classify(
        face_found=True,
        face_bbox=face_box,
        hands=hands_res,
    )
    assert res.state == FaceOcclusionState.FACE_SIGNIFICANTLY_OBSCURED
    assert res.is_occluded is True
    assert res.hand_overlap_ratio > 0.50
