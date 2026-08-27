"""Tests for hand distance ratio refinements and proximity metrics."""

from proctoring.analysis.hands import HandAnalysisResult, HandAnalyzer, HandObservation


def test_hand_distance_ratios_calculated() -> None:
    analyzer = HandAnalyzer()
    res = HandAnalysisResult()

    # Face at (100, 100, 300, 300) -> width=200, centre=(200, 200)
    face_bbox = (100, 100, 300, 300)
    mouth_region = (150, 240, 250, 280)  # centre=(200, 260)
    ear_regions = [(80, 150, 100, 210)]  # centre=(90, 180)

    # Hand at (180, 240, 220, 280) -> centroid=(200, 260) right on the mouth
    hand = HandObservation(
        handedness="Right",
        confidence=0.9,
        bbox=(180, 240, 220, 280),
        centroid=(200, 260),
        fingertip_points=[(200, 255)],
    )
    res.hands = [hand]

    analyzer._relate_to_face(
        result=res,
        face_bbox=face_bbox,
        ear_regions=ear_regions,
        mouth_region=mouth_region,
    )

    assert res.hand_near_face is True
    assert res.hand_near_mouth is True
    assert res.distance_to_mouth_ratio is not None
    assert res.distance_to_mouth_ratio < 0.10  # Very close to mouth
    assert res.nearest_hand_distance_ratio is not None
    assert res.distance_to_ear_ratio is not None
    assert res.distance_to_ear_ratio > 0.40  # Further from ear than mouth
