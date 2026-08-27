"""Targeted unit tests for physical paper exam mode and hand writing behavior."""

from proctoring.analysis.hands import HandAnalysisResult, HandAnalyzer, HandObservation
from proctoring.analysis.policy import ExamMode, ExamPolicy, StrictnessLevel


def test_physical_paper_policy_tolerances() -> None:
    """Test that PHYSICAL_PAPER mode allows downward pitch and downward gaze."""
    screen_policy = ExamPolicy.for_level(
        level=StrictnessLevel.STANDARD, mode=ExamMode.DIGITAL_SCREEN
    )
    paper_policy = ExamPolicy.for_level(
        level=StrictnessLevel.STANDARD, mode=ExamMode.PHYSICAL_PAPER
    )

    # Downward pitch of -45 degrees (normal writing posture)
    assert screen_policy.is_pose_deviated(yaw=0.0, pitch=-45.0) is True
    assert paper_policy.is_pose_deviated(yaw=0.0, pitch=-45.0) is False

    # Extreme downward pitch (-70 degrees) is still deviated in paper mode
    assert paper_policy.is_pose_deviated(yaw=0.0, pitch=-70.0) is True

    # Downward gaze of -0.55 (looking down at sheet)
    assert screen_policy.is_gaze_deviated(horizontal=0.0, vertical=-0.55) is True
    assert paper_policy.is_gaze_deviated(horizontal=0.0, vertical=-0.55) is False

    # Lateral gaze (+0.50) is still deviated in paper mode
    assert paper_policy.is_gaze_deviated(horizontal=0.50, vertical=0.0) is True


def test_hand_writing_area_detection() -> None:
    """Test hand analysis result writing posture flags."""
    analyzer = HandAnalyzer(writing_area_top_ratio=0.45)

    # Hand at desk height (centroid y=400 on 480p frame)
    hand = HandObservation(
        handedness="Right",
        confidence=0.9,
        bbox=(200, 360, 300, 440),
        centroid=(250, 400),
        fingertip_points=[(250, 360)],
    )

    result = HandAnalysisResult(
        hands_detected=1,
        hands=[hand],
        hands_visible=True,
        hand_near_face=False,
    )

    # Synthetic frame (480, 640)
    # Face bbox in upper quadrant (220, 50, 280, 130) -> width 60, centre y=90
    face_bbox = (220, 50, 280, 130)

    # Test relating to face
    analyzer._relate_to_face(result, face_bbox, ear_regions=[], mouth_region=None)

    # Check writing posture
    assert result.hand_near_face is False
    assert hand.centroid[1] >= int(480 * 0.45)
    assert (
        result.nearest_hand_distance_ratio is not None and result.nearest_hand_distance_ratio > 1.0
    )
