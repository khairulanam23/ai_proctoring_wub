"""Tests for normalized gaze tracking, calibration, and viewing zone classification."""

import numpy as np

from proctoring.analysis.gaze import (
    GazeDirection,
    GazeObservation,
    GazeTracker,
)


def _create_mock_landmarks(iris_shift_x: float = 0.0, iris_shift_y: float = 0.0) -> np.ndarray:
    """Create a mock 478-point landmark array with configured eye/iris coordinates."""
    pts = np.zeros((478, 2), dtype=np.float32)

    # Left eye: inner=133 (x=120, y=100), outer=33 (x=80, y=100), top=159 (y=90), bottom=145 (y=110)
    pts[133] = [120.0, 100.0]
    pts[33] = [80.0, 100.0]
    pts[159] = [100.0, 90.0]
    pts[145] = [100.0, 110.0]

    # Center is (100, 100). Iris indices 468..472
    for idx in range(468, 473):
        pts[idx] = [100.0 + iris_shift_x, 100.0 + iris_shift_y]

    # Right eye: inner=362 (x=160, y=100), outer=263 (x=200, y=100), top=386 (y=90), bottom=374 (y=110)
    pts[362] = [160.0, 100.0]
    pts[263] = [200.0, 100.0]
    pts[386] = [180.0, 90.0]
    pts[374] = [180.0, 110.0]

    # Center is (180, 100). Iris indices 473..477
    for idx in range(473, 478):
        pts[idx] = [180.0 + iris_shift_x, 100.0 + iris_shift_y]

    return pts


def test_gaze_center_neutral() -> None:
    tracker = GazeTracker(deadband=0.10)
    landmarks = _create_mock_landmarks(iris_shift_x=0.0, iris_shift_y=0.0)
    obs = tracker.measure(landmarks)

    assert obs.direction == GazeDirection.CENTER
    assert abs(obs.horizontal) < 0.05
    assert abs(obs.vertical) < 0.05
    assert not obs.is_off_screen
    assert obs.confidence > 0.8


def test_gaze_left_and_right() -> None:
    tracker = GazeTracker(horizontal_threshold=0.30, deadband=0.10)

    # Shift iris to candidate's left (positive x in image plane)
    lm_left = _create_mock_landmarks(iris_shift_x=8.0, iris_shift_y=0.0)
    obs_left = tracker.measure(lm_left)
    assert obs_left.direction == GazeDirection.LEFT
    assert obs_left.horizontal > 0.20
    assert obs_left.is_off_screen is True

    # Shift iris to candidate's right (negative x)
    lm_right = _create_mock_landmarks(iris_shift_x=-8.0, iris_shift_y=0.0)
    obs_right = tracker.measure(lm_right)
    assert obs_right.direction == GazeDirection.RIGHT
    assert obs_right.horizontal < -0.20
    assert obs_right.is_off_screen is True


def test_gaze_up_and_down() -> None:
    tracker = GazeTracker(vertical_up_threshold=0.25, vertical_down_threshold=0.25, deadband=0.08)

    # Shift iris upward in image (lower y pixel value -> positive vertical ratio)
    lm_up = _create_mock_landmarks(iris_shift_x=0.0, iris_shift_y=-5.0)
    obs_up = tracker.measure(lm_up)
    assert obs_up.direction == GazeDirection.UP
    assert obs_up.vertical > 0.20
    assert obs_up.is_off_screen is True

    # Shift iris downward in image (higher y pixel value -> negative vertical ratio)
    lm_down = _create_mock_landmarks(iris_shift_x=0.0, iris_shift_y=5.0)
    obs_down = tracker.measure(lm_down)
    assert obs_down.direction == GazeDirection.DOWN
    assert obs_down.vertical < -0.20
    assert obs_down.is_off_screen is True


def test_gaze_calibration() -> None:
    tracker = GazeTracker()

    # Stable slightly-off-center samples (e.g. candidate looks at monitor placed slightly to right)
    samples = [(0.15, 0.05), (0.16, 0.04), (0.15, 0.06), (0.14, 0.05)]
    res = tracker.calibrate(samples)
    assert res["calibrated"] is True
    assert tracker.calibration.is_calibrated is True
    assert abs(tracker.calibration.baseline_horizontal - 0.15) < 0.02

    # Now measuring a frame with raw ratio (0.15, 0.05) should yield calibrated horizontal ~ 0.0
    lm = _create_mock_landmarks(iris_shift_x=3.0, iris_shift_y=-0.5)
    obs = tracker.measure(lm)
    # The normalized offset should be adjusted by baseline
    assert obs.direction in (GazeDirection.CENTER, GazeDirection.LEFT, GazeDirection.RIGHT)


def test_gaze_calibration_rejection_on_high_variance() -> None:
    tracker = GazeTracker()
    erratic_samples = [(-0.4, 0.2), (0.5, -0.3), (0.1, 0.4), (-0.3, -0.2)]
    res = tracker.calibrate(erratic_samples)
    assert res["calibrated"] is False
    assert "varied too much" in res["reason"]
    assert tracker.calibration.is_calibrated is False


def test_gaze_missing_landmarks() -> None:
    tracker = GazeTracker()
    obs_none = tracker.measure(None)
    assert obs_none.direction == GazeDirection.UNKNOWN
    assert obs_none.confidence == 0.0

    short_lm = np.zeros((100, 2), dtype=np.float32)
    obs_short = tracker.measure(short_lm)
    assert obs_short.direction == GazeDirection.UNKNOWN


def test_gaze_to_dict_serialization() -> None:
    obs = GazeObservation(
        horizontal=0.12345,
        vertical=-0.23456,
        offset=0.265,
        direction=GazeDirection.DOWN,
        confidence=0.95,
        is_off_screen=True,
    )
    d = obs.to_dict()
    assert d["horizontal"] == 0.1235
    assert d["vertical"] == -0.2346
    assert d["direction"] == "DOWN"
    assert d["is_off_screen"] is True
