"""Targeted unit tests for advanced gaze tracking, EMA smoothing, and directional hysteresis."""

import numpy as np
import pytest

from proctoring.analysis.gaze import GazeDirection, GazeTracker


def test_gaze_tracker_ema_smoothing() -> None:
    """Test that exponential moving average smoothing dampens single-frame micro-saccades."""
    tracker = GazeTracker(
        horizontal_threshold=0.32,
        vertical_up_threshold=0.28,
        vertical_down_threshold=0.32,
        deadband=0.12,
        smoothing_alpha=0.5,
    )

    # Synthetic 478 face landmarks (all zeros except eyes/irises)
    landmarks = np.zeros((478, 2), dtype=np.float32)

    # Set up left eye aperture: inner=(100, 100), outer=(50, 100), top=(75, 90), bottom=(75, 110)
    landmarks[133] = [100.0, 100.0]  # inner
    landmarks[33] = [50.0, 100.0]  # outer
    landmarks[159] = [75.0, 90.0]  # top
    landmarks[145] = [75.0, 110.0]  # bottom
    for idx in (468, 469, 470, 471, 472):
        landmarks[idx] = [75.0, 100.0]  # center

    # Set up right eye aperture: inner=(150, 100), outer=(200, 100), top=(175, 90), bottom=(175, 110)
    landmarks[362] = [150.0, 100.0]  # inner
    landmarks[263] = [200.0, 100.0]  # outer
    landmarks[386] = [175.0, 90.0]  # top
    landmarks[374] = [175.0, 110.0]  # bottom
    for idx in (473, 474, 475, 476, 477):
        landmarks[idx] = [175.0, 100.0]  # center

    # Frame 1: center
    obs1 = tracker.measure(landmarks)
    assert obs1.direction == GazeDirection.CENTER
    assert abs(obs1.horizontal) < 0.05

    # Frame 2: small fixational jitter shift
    for idx in (468, 469, 470, 471, 472):
        landmarks[idx] = [79.0, 100.0]
    for idx in (473, 474, 475, 476, 477):
        landmarks[idx] = [179.0, 100.0]

    obs2 = tracker.measure(landmarks)
    # Raw horizontal is ~0.16, but smoothed horizontal should be dampened by alpha=0.5 to ~0.08
    assert obs2.raw_horizontal > 0.12
    assert obs2.horizontal < obs2.raw_horizontal
    assert obs2.horizontal == pytest.approx(obs2.raw_horizontal * 0.5, abs=0.03)

    # Frame 3: large saccadic jump (> 0.25 displacement) immediately updates
    for idx in (468, 469, 470, 471, 472):
        landmarks[idx] = [90.0, 100.0]
    for idx in (473, 474, 475, 476, 477):
        landmarks[idx] = [190.0, 100.0]

    obs3 = tracker.measure(landmarks)
    assert obs3.horizontal == pytest.approx(obs3.raw_horizontal, abs=0.01)


def test_gaze_tracker_directional_hysteresis() -> None:
    """Test that directional classification does not flicker at deadband boundary."""
    tracker = GazeTracker(deadband=0.15, hysteresis_margin=0.03, smoothing_alpha=1.0)

    # Initial state: CENTER. Boundary to become LEFT is deadband + margin = 0.18
    assert tracker._classify_direction(0.16, 0.0) == GazeDirection.CENTER
    tracker._current_direction = GazeDirection.CENTER

    # Exceeding 0.18 triggers LEFT
    assert tracker._classify_direction(0.20, 0.0) == GazeDirection.LEFT
    tracker._current_direction = GazeDirection.LEFT

    # Staying above deadband - margin = 0.12 keeps LEFT (hysteresis hold)
    assert tracker._classify_direction(0.14, 0.0) == GazeDirection.LEFT

    # Dropping below 0.12 returns to CENTER
    tracker._current_direction = tracker._classify_direction(0.10, 0.0)
    assert tracker._current_direction == GazeDirection.CENTER


def test_gaze_calibration_stability() -> None:
    """Test candidate neutral baseline gaze calibration."""
    tracker = GazeTracker()

    # Stable gaze samples
    samples = [(0.05, -0.02), (0.06, -0.01), (0.04, -0.03), (0.05, -0.02)]
    res = tracker.calibrate(samples)
    assert res["calibrated"] is True
    assert tracker.calibration.is_calibrated is True
    assert tracker.calibration.baseline_horizontal == pytest.approx(0.05, abs=0.01)

    # Reset calibration
    tracker.reset_calibration()
    assert tracker.calibration.is_calibrated is False
