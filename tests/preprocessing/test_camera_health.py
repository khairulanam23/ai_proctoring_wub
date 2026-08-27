"""Unit tests for camera health monitoring and stream anomaly detection."""

import numpy as np

from proctoring.preprocessing.camera_health import (
    CameraAnomaly,
    CameraHealthMonitor,
)
from proctoring.preprocessing.frame_quality import FrameQualityGate


def test_camera_health_normal_stream():
    """Verify normal varying frames report healthy status."""
    monitor = CameraHealthMonitor(frozen_frame_duration_seconds=3.0)

    # 5 different frames with noise/movement
    for i in range(5):
        frame = np.random.randint(60, 200, (480, 640, 3), dtype=np.uint8)
        status = monitor.assess(frame, timestamp_seconds=float(i) * 0.5)
        assert status.is_healthy is True
        assert status.anomaly == CameraAnomaly.NONE
        assert status.is_frozen is False


def test_camera_health_frozen_frames():
    """Verify repeated identical frames trigger CameraAnomaly.FRAME_FROZEN after duration."""
    monitor = CameraHealthMonitor(frozen_frame_duration_seconds=2.0)
    static_frame = np.full((480, 640, 3), 120, dtype=np.uint8)

    # Initial frame
    st0 = monitor.assess(static_frame, timestamp_seconds=0.0)
    assert st0.is_healthy is True
    assert st0.is_frozen is False

    # 1.0s elapsed: still within grace period
    st1 = monitor.assess(static_frame, timestamp_seconds=1.0)
    assert st1.is_healthy is True
    assert st1.is_frozen is False

    # 2.5s elapsed: exceeded 2.0s threshold
    st2 = monitor.assess(static_frame, timestamp_seconds=2.5)
    assert st2.is_healthy is False
    assert st2.is_frozen is True
    assert st2.anomaly == CameraAnomaly.FRAME_FROZEN
    assert st2.freeze_duration_seconds >= 2.0


def test_camera_health_black_screen():
    """Verify uniform pitch black frame triggers BLACK_SCREEN anomaly."""
    monitor = CameraHealthMonitor(min_dark_luma=5.0)
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    status = monitor.assess(black_frame, timestamp_seconds=0.0)
    assert status.is_healthy is False
    assert status.anomaly == CameraAnomaly.BLACK_SCREEN


def test_camera_health_extreme_glare():
    """Verify overexposed whiteout frame triggers EXTREME_GLARE anomaly."""
    monitor = CameraHealthMonitor(max_glare_luma=250.0)
    white_frame = np.full((480, 640, 3), 255, dtype=np.uint8)

    status = monitor.assess(white_frame, timestamp_seconds=0.0)
    assert status.is_healthy is False
    assert status.anomaly == CameraAnomaly.EXTREME_GLARE


def test_camera_health_delivery_gap():
    """Verify sudden jump in timestamps triggers DELIVERY_GAP anomaly."""
    monitor = CameraHealthMonitor(max_delivery_gap_seconds=3.0)
    frame = np.random.randint(60, 200, (480, 640, 3), dtype=np.uint8)

    monitor.assess(frame, timestamp_seconds=0.0)
    monitor.assess(frame, timestamp_seconds=0.5)

    # Jump 5 seconds
    status = monitor.assess(frame, timestamp_seconds=5.5)
    assert status.is_healthy is False
    assert status.anomaly == CameraAnomaly.DELIVERY_GAP
    assert status.gap_duration_seconds == 5.0


def test_camera_health_stream_disconnected():
    """Verify None or empty frame triggers STREAM_DISCONNECTED anomaly."""
    monitor = CameraHealthMonitor()
    status = monitor.assess(None, timestamp_seconds=0.0)
    assert status.is_healthy is False
    assert status.anomaly == CameraAnomaly.STREAM_DISCONNECTED


def test_frame_quality_gate_with_camera_health():
    """Verify FrameQualityGate carries CameraHealthStatus into FrameGateResult."""
    gate = FrameQualityGate()
    frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

    res = gate.process(frame, timestamp_seconds=0.0)
    assert res.accepted is True
    assert res.camera_health is not None
    assert res.camera_health.is_healthy is True
