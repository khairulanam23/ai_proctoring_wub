"""Unit tests for camera discovery, health check validation, and webcam stream controller."""

import numpy as np

from proctoring.capture.camera import (
    CameraDiscoveryResult,
    CaptureMode,
    WebcamHealthCheckResult,
    WebcamStream,
    discover_local_camera,
    validate_webcam_health,
)


def test_camera_discovery_result_serialization():
    res = CameraDiscoveryResult(
        device_index=0,
        device_path="/dev/video0",
        backend_name="V4L2",
        backend_code=200,
        frame_width=640,
        frame_height=480,
        fps=30.0,
        is_valid=True,
        diagnostics=["Device probed successfully"],
        error_message="",
    )
    d = res.to_dict()
    assert d["device_index"] == 0
    assert d["backend_name"] == "V4L2"
    assert d["is_valid"] is True


def test_webcam_health_check_result_serialization():
    res = WebcamHealthCheckResult(
        success=True,
        capture_mode=CaptureMode.PHYSICAL_CAMERA.value,
        device_path="/dev/video0",
        camera_index=0,
        backend_name="V4L2",
        frame_width=640,
        frame_height=480,
        frame_read_success=True,
        ai_face_detected=True,
        ai_face_count=1,
        ai_face_confidence=0.92,
        release_success=True,
        diagnostics=["All steps passed"],
        error_reason="",
    )
    d = res.to_dict()
    assert d["success"] is True
    assert d["capture_mode"] == "PHYSICAL_CAMERA"
    assert d["ai_face_count"] == 1


def test_discover_local_camera_execution():
    res = discover_local_camera(max_index=2)
    assert isinstance(res, CameraDiscoveryResult)
    assert len(res.diagnostics) > 0
    # In Linux environment with active video0, should succeed
    if res.is_valid:
        assert res.device_index is not None
        assert res.frame_width > 0
        assert res.frame_height > 0


def test_validate_webcam_health_execution():
    health = validate_webcam_health()
    assert isinstance(health, WebcamHealthCheckResult)
    assert health.release_success is True
    if health.success:
        assert health.capture_mode == CaptureMode.PHYSICAL_CAMERA.value
        assert health.frame_read_success is True


def test_webcam_stream_lifecycle():
    stream = WebcamStream(width=640, height=480)
    success, msg = stream.start()
    if success:
        assert stream.is_active is True
        assert stream.capture_mode == CaptureMode.PHYSICAL_CAMERA
        frame = stream.capture_frame()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape[:2] == (480, 640)
        ok, f2 = stream.read_frame()
        assert ok is True
        assert f2 is not None
    stream.stop()
    assert stream.is_active is False
