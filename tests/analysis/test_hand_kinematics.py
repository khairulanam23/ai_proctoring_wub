"""Unit tests for hand movement kinematics and handwriting state classification."""

import numpy as np
import pytest

from proctoring.analysis.hands import (
    HandAnalysisResult,
    HandAnalyzer,
    HandKinematics,
    HandObservation,
    HandState,
)


def test_hand_resting_in_writing_area():
    analyzer = HandAnalyzer(writing_area_top_ratio=0.40)
    # Hand stationary at (300, 350) on 480x640 frame
    hand1 = HandObservation(
        handedness="Right",
        confidence=0.9,
        bbox=(250, 300, 350, 400),
        centroid=(300, 350),
        fingertip_points=[(300, 310)],
    )
    res1 = HandAnalysisResult(hands=[hand1], hands_detected=1, hands_visible=True)
    analyzer._update_kinematics(hand1, res1, height=480, width=640, timestamp_seconds=0.0)

    # Frame 2: identical position at t=0.25s
    hand2 = HandObservation(
        handedness="Right",
        confidence=0.9,
        bbox=(250, 300, 350, 400),
        centroid=(300, 350),
        fingertip_points=[(300, 310)],
    )
    res2 = HandAnalysisResult(hands=[hand2], hands_detected=1, hands_visible=True)
    analyzer._update_kinematics(hand2, res2, height=480, width=640, timestamp_seconds=0.25)

    assert hand2.kinematics.speed == pytest.approx(0.0, abs=1.0)
    assert hand2.kinematics.state == HandState.HAND_RESTING


def test_hand_moving_across_paper():
    analyzer = HandAnalyzer(writing_area_top_ratio=0.40)

    # Frame 1: x=200, y=350 at t=0.0
    hand1 = HandObservation(
        handedness="Right",
        confidence=0.9,
        bbox=(150, 300, 250, 400),
        centroid=(200, 350),
        fingertip_points=[(200, 310)],
    )
    res1 = HandAnalysisResult(hands=[hand1], hands_detected=1, hands_visible=True)
    analyzer._update_kinematics(hand1, res1, height=480, width=640, timestamp_seconds=0.0)

    # Frame 2: moved laterally to x=220, y=350 at t=0.25 (dx=20, vx=80 px/s)
    hand2 = HandObservation(
        handedness="Right",
        confidence=0.9,
        bbox=(170, 300, 270, 400),
        centroid=(220, 350),
        fingertip_points=[(220, 310)],
    )
    res2 = HandAnalysisResult(hands=[hand2], hands_detected=1, hands_visible=True)
    analyzer._update_kinematics(hand2, res2, height=480, width=640, timestamp_seconds=0.25)

    assert hand2.kinematics.speed == pytest.approx(80.0, abs=5.0)
    assert hand2.kinematics.state == HandState.HAND_MOVING_ACROSS_PAPER


def test_hand_writing_micro_oscillation():
    analyzer = HandAnalyzer(writing_area_top_ratio=0.40)

    # Simulate 5 frames of writing: wrist drifts slowly while fingertip oscillates up/down
    tip_y_offsets = [0, 4, -3, 5, -4]
    last_hand = None

    for i, dy in enumerate(tip_y_offsets):
        hand = HandObservation(
            handedness="Right",
            confidence=0.9,
            bbox=(250 + i * 2, 320, 330 + i * 2, 400),
            centroid=(290 + i * 2, 360),
            fingertip_points=[(290, 330 + dy)],
        )
        res = HandAnalysisResult(hands=[hand], hands_detected=1, hands_visible=True)
        analyzer._update_kinematics(
            hand, res, height=480, width=640, timestamp_seconds=i * 0.25
        )
        last_hand = hand

    assert last_hand is not None
    assert last_hand.kinematics.writing_micro_oscillation is True
    assert last_hand.kinematics.state == HandState.HAND_WRITING


def test_hand_near_ear_state():
    analyzer = HandAnalyzer(writing_area_top_ratio=0.40)
    hand = HandObservation(
        handedness="Right",
        confidence=0.9,
        bbox=(100, 100, 160, 180),
        centroid=(130, 140),
        fingertip_points=[(130, 120)],
    )
    res = HandAnalysisResult(hands=[hand], hands_detected=1, hands_visible=True, hand_near_ear=True)
    analyzer._update_kinematics(hand, res, height=480, width=640, timestamp_seconds=0.0)

    assert hand.kinematics.state == HandState.HAND_NEAR_EAR
