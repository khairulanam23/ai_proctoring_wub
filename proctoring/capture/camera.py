"""Robust camera discovery, multi-environment webcam streaming, and hardware health checks.

Supports both local native OpenCV/V4L2 video capture and Google Colab HTML5 WebRTC bridges.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class CaptureMode(str, Enum):
    """Execution capture source classification."""

    PHYSICAL_CAMERA = "PHYSICAL_CAMERA"
    BROWSER_WEBCAM = "BROWSER_WEBCAM"
    SYNTHETIC_TEST = "SYNTHETIC_TEST"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class CameraDiscoveryResult:
    """Comprehensive diagnostic result from probing local camera hardware."""

    device_index: int | None = None
    device_path: str | None = None
    backend_name: str = "UNKNOWN"
    backend_code: int = 0
    frame_width: int = 0
    frame_height: int = 0
    fps: float = 0.0
    is_valid: bool = False
    diagnostics: list[str] = field(default_factory=list)
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WebcamHealthCheckResult:
    """Structured result of end-to-end webcam hardware and AI processing validation."""

    success: bool
    capture_mode: str
    device_path: str | None = None
    camera_index: int | None = None
    backend_name: str = "UNKNOWN"
    frame_width: int | None = None
    frame_height: int | None = None
    frame_read_success: bool = False
    ai_face_detected: bool | None = None
    ai_face_count: int = 0
    ai_face_confidence: float | None = None
    release_success: bool = False
    diagnostics: list[str] = field(default_factory=list)
    error_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_local_camera(
    preferred_index: int | None = None,
    max_index: int = 5,
    preferred_backend: int | None = None,
    target_width: int = 640,
    target_height: int = 480,
) -> CameraDiscoveryResult:
    """Enumerate and validate candidate camera indices and backends.

    Checks PROCTORING_CAMERA_INDEX env var or explicit user override, testing
    Linux V4L2 and native OpenCV capture backends to verify frame readability.
    """
    diagnostics: list[str] = []

    # 1. Check environment variable override
    env_idx = os.environ.get("PROCTORING_CAMERA_INDEX")
    if preferred_index is None and env_idx is not None and env_idx.strip().isdigit():
        preferred_index = int(env_idx.strip())
        diagnostics.append(
            f"Using PROCTORING_CAMERA_INDEX environment override: index {preferred_index}"
        )

    # Determine candidate indices
    if preferred_index is not None:
        candidate_indices = [preferred_index]
    else:
        candidate_indices = list(range(max_index))

    # Candidate backends (V4L2 prioritized on Linux)
    if preferred_backend is not None:
        candidate_backends = [("UserPreferred", preferred_backend)]
    elif sys.platform.startswith("linux"):
        candidate_backends = [
            ("V4L2", cv2.CAP_V4L2),
            ("ANY", cv2.CAP_ANY),
        ]
    else:
        candidate_backends = [
            ("ANY", cv2.CAP_ANY),
        ]

    for b_name, b_code in candidate_backends:
        for idx in candidate_indices:
            dev_path = f"/dev/video{idx}" if sys.platform.startswith("linux") else f"Camera_{idx}"
            diagnostics.append(
                f"Probing {dev_path} (index {idx}) with backend {b_name} ({b_code})..."
            )

            cap = None
            try:
                cap = cv2.VideoCapture(idx, b_code)
                if not cap.isOpened():
                    diagnostics.append(f"  -> Index {idx} ({b_name}) failed to open.")
                    cap.release()
                    continue

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

                # Warm-up read (allow exposure/gain stabilization)
                for _ in range(3):
                    ret_w, _ = cap.read()
                    if ret_w:
                        pass

                ret, frame = cap.read()
                actual_backend = cap.getBackendName()
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = float(cap.get(cv2.CAP_PROP_FPS))

                if ret and frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                    h, w = frame.shape[:2]
                    diagnostics.append(
                        f"  -> ✓ VALID CAMERA FOUND: index {idx} ({actual_backend}), resolution {w}x{h} ({actual_w}x{actual_h} configured), {actual_fps:.1f} FPS"
                    )
                    cap.release()
                    return CameraDiscoveryResult(
                        device_index=idx,
                        device_path=dev_path,
                        backend_name=actual_backend,
                        backend_code=b_code,
                        frame_width=w,
                        frame_height=h,
                        fps=actual_fps if actual_fps > 0 else 30.0,
                        is_valid=True,
                        diagnostics=diagnostics,
                        error_message="",
                    )
                diagnostics.append(
                    f"  -> Index {idx} ({b_name}) opened but frame read failed or returned empty."
                )
                cap.release()

            except Exception as e:
                diagnostics.append(f"  -> Exception probing index {idx} with {b_name}: {e}")
                if cap is not None:
                    with contextlib.suppress(Exception):
                        cap.release()

    # No camera discovered
    err_msg = (
        f"No functional physical camera discovered across indices {candidate_indices}. "
        "Verify camera is plugged in and user has permission to access /dev/video*."
    )
    diagnostics.append(f"✗ {err_msg}")
    return CameraDiscoveryResult(
        device_index=None,
        device_path=None,
        backend_name="NONE",
        backend_code=0,
        frame_width=0,
        frame_height=0,
        fps=0.0,
        is_valid=False,
        diagnostics=diagnostics,
        error_message=err_msg,
    )


def validate_webcam_health(
    camera_index: int | None = None,
    face_detector: Any | None = None,
    target_width: int = 640,
    target_height: int = 480,
) -> WebcamHealthCheckResult:
    """Execute complete 7-step physical webcam health validation with AI face detection check.

    1. Camera device exists
    2. Camera can be opened
    3. Camera can produce a frame
    4. Frame is non-empty
    5. Frame has valid dimensions
    6. Frame can be processed by the AI pipeline
    7. Camera can be released cleanly
    """
    diagnostics: list[str] = []
    discovery = discover_local_camera(
        preferred_index=camera_index,
        target_width=target_width,
        target_height=target_height,
    )
    diagnostics.extend(discovery.diagnostics)

    if not discovery.is_valid or discovery.device_index is None:
        return WebcamHealthCheckResult(
            success=False,
            capture_mode=CaptureMode.UNAVAILABLE.value,
            device_path=None,
            camera_index=None,
            backend_name="NONE",
            frame_width=None,
            frame_height=None,
            frame_read_success=False,
            ai_face_detected=None,
            ai_face_count=0,
            ai_face_confidence=None,
            release_success=True,
            diagnostics=diagnostics,
            error_reason=discovery.error_message,
        )

    # Open the verified device for health check sequence
    cap = None
    release_ok = False
    frame_read_ok = False
    ai_detected = False
    ai_count = 0
    ai_conf = None
    captured_frame = None

    try:
        cap = cv2.VideoCapture(discovery.device_index, discovery.backend_code)
        if not cap.isOpened():
            diagnostics.append(
                f"Step 2 FAIL: Failed to reopen verified device index {discovery.device_index}"
            )
            return WebcamHealthCheckResult(
                success=False,
                capture_mode=CaptureMode.PHYSICAL_CAMERA.value,
                device_path=discovery.device_path,
                camera_index=discovery.device_index,
                backend_name=discovery.backend_name,
                frame_read_success=False,
                release_success=True,
                diagnostics=diagnostics,
                error_reason=f"Could not open device index {discovery.device_index}",
            )

        diagnostics.append(
            f"Step 2 PASS: Camera device index {discovery.device_index} opened successfully."
        )

        # Warm up sensor
        for _ in range(3):
            cap.read()

        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            frame_read_ok = True
            captured_frame = frame
            h, w = frame.shape[:2]
            diagnostics.append(
                f"Step 3-5 PASS: Read valid BGR frame with dimensions {w}x{h}, {frame.dtype}."
            )
        else:
            diagnostics.append("Step 3-5 FAIL: Frame read returned False or empty array.")

        # Step 6: AI Face Detection Check
        if frame_read_ok and captured_frame is not None:
            if face_detector is None:
                try:
                    from proctoring.detection.face_detector import FaceDetector

                    model_path = Path("models/face_detection_yunet_2023mar.onnx")
                    if model_path.exists():
                        face_detector = FaceDetector(model_path=model_path)
                except Exception as e:
                    diagnostics.append(f"AI Detector loading note: {e}")

            if face_detector is not None:
                try:
                    det_res = face_detector.detect(captured_frame)
                    ai_count = det_res.count
                    ai_detected = ai_count > 0
                    if ai_detected:
                        ai_conf = float(det_res.faces[0].confidence)
                        diagnostics.append(
                            f"Step 6 PASS: AI Face detector processed live frame: {ai_count} face(s) detected (confidence: {ai_conf:.4f})."
                        )
                    else:
                        diagnostics.append(
                            "Step 6 PASS: AI Face detector processed live frame successfully (0 faces detected in scene)."
                        )
                except Exception as e:
                    diagnostics.append(f"Step 6 FAIL: AI face detector error on live frame: {e}")
            else:
                diagnostics.append("Step 6 SKIP: No face detector instance provided for AI check.")

    except Exception as e:
        diagnostics.append(f"Exception during webcam health check: {e}")
    finally:
        if cap is not None:
            try:
                cap.release()
                release_ok = True
                diagnostics.append("Step 7 PASS: Camera released cleanly.")
            except Exception as e:
                diagnostics.append(f"Step 7 FAIL: Error releasing camera: {e}")
        else:
            release_ok = True

    overall_success = frame_read_ok and release_ok

    return WebcamHealthCheckResult(
        success=overall_success,
        capture_mode=CaptureMode.PHYSICAL_CAMERA.value
        if overall_success
        else CaptureMode.UNAVAILABLE.value,
        device_path=discovery.device_path,
        camera_index=discovery.device_index,
        backend_name=discovery.backend_name,
        frame_width=captured_frame.shape[1] if captured_frame is not None else None,
        frame_height=captured_frame.shape[0] if captured_frame is not None else None,
        frame_read_success=frame_read_ok,
        ai_face_detected=ai_detected,
        ai_face_count=ai_count,
        ai_face_confidence=ai_conf,
        release_success=release_ok,
        diagnostics=diagnostics,
        error_reason="" if overall_success else "Frame capture or device release failed.",
    )


class WebcamStream:
    """Universal multi-environment webcam stream controller.

    Supports native OpenCV / V4L2 camera capture on local Linux PCs with automatic
    device discovery, and Google Colab HTML5 WebRTC browser capture.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        camera_index: int | None = None,
        force_colab: bool | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.preferred_index = camera_index

        # Environment detection
        in_colab = "google.colab" in sys.modules
        has_colab_output = False
        try:
            from google.colab import output  # noqa: F401  (availability probe)

            has_colab_output = True
        except ImportError:
            pass

        if force_colab is not None:
            self.is_colab = force_colab
        else:
            self.is_colab = in_colab and has_colab_output

        self.cap: cv2.VideoCapture | None = None
        self.is_active: bool = False
        self.capture_mode: CaptureMode = CaptureMode.UNAVAILABLE
        self.active_index: int | None = None
        self.active_backend: str = "NONE"
        self.device_path: str | None = None

    def start(self) -> tuple[bool, str]:
        """Initialize, discover, and start the camera stream."""
        if self.is_colab:
            return self._start_colab_webcam()
        return self._start_local_webcam()

    def _start_local_webcam(self) -> tuple[bool, str]:
        """Discover and open native local camera via OpenCV / V4L2."""
        discovery = discover_local_camera(
            preferred_index=self.preferred_index,
            target_width=self.width,
            target_height=self.height,
        )

        if not discovery.is_valid or discovery.device_index is None:
            self.is_active = False
            self.capture_mode = CaptureMode.UNAVAILABLE
            return False, discovery.error_message

        try:
            self.cap = cv2.VideoCapture(discovery.device_index, discovery.backend_code)
            if not self.cap.isOpened():
                self.is_active = False
                self.capture_mode = CaptureMode.UNAVAILABLE
                return False, f"Could not open discovered camera index {discovery.device_index}"

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Warm-up read
            for _ in range(3):
                self.cap.read()

            ret, test_frame = self.cap.read()
            if not ret or test_frame is None or test_frame.size == 0:
                self.cap.release()
                self.cap = None
                self.is_active = False
                self.capture_mode = CaptureMode.UNAVAILABLE
                return (
                    False,
                    f"Camera index {discovery.device_index} opened but failed initial frame read.",
                )

            self.is_active = True
            self.capture_mode = CaptureMode.PHYSICAL_CAMERA
            self.active_index = discovery.device_index
            self.active_backend = discovery.backend_name
            self.device_path = discovery.device_path

            return (
                True,
                f"Physical camera initialized successfully on {discovery.device_path} (Backend: {discovery.backend_name}, Resolution: {test_frame.shape[1]}x{test_frame.shape[0]}).",
            )

        except Exception as e:
            if self.cap is not None:
                with contextlib.suppress(Exception):
                    self.cap.release()
                self.cap = None
            self.is_active = False
            self.capture_mode = CaptureMode.UNAVAILABLE
            return False, f"Local camera initialization exception: {e}"

    def _start_colab_webcam(self) -> tuple[bool, str]:
        """Initialize Google Colab browser webcam via HTML5 WebRTC JavaScript."""
        try:
            from google.colab import output as colab_output
        except ImportError:
            return False, "Google Colab output module not available."

        js_init = f"""
        (async function() {{
            if (window._proctoringStream) {{
                window._proctoringStream.getTracks().forEach(t => t.stop());
                window._proctoringStream = null;
            }}
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
                return {{ success: false, error: "Browser does not support navigator.mediaDevices.getUserMedia" }};
            }}
            try {{
                const stream = await navigator.mediaDevices.getUserMedia({{
                    video: {{
                        width: {{ ideal: {self.width} }},
                        height: {{ ideal: {self.height} }},
                        facingMode: "user"
                    }}
                }});
                window._proctoringStream = stream;
                let video = document.getElementById("proctoring_webcam_video");
                if (!video) {{
                    video = document.createElement("video");
                    video.id = "proctoring_webcam_video";
                    video.setAttribute("autoplay", "");
                    video.setAttribute("playsinline", "");
                    video.setAttribute("muted", "");
                    video.style.display = "none";
                    document.body.appendChild(video);
                }}
                video.srcObject = stream;
                await video.play();
                return {{ success: true, width: video.videoWidth || {self.width}, height: video.videoHeight || {self.height} }};
            }} catch (err) {{
                let errMsg = err.message;
                if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {{
                    errMsg = "Camera permission denied by user. Please allow camera access in browser address bar.";
                }} else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {{
                    errMsg = "No webcam device detected on your system. Please connect a camera.";
                }} else if (err.name === "NotReadableError" || err.name === "TrackStartError") {{
                    errMsg = "Webcam is already in use by another application or hardware error occurred.";
                }}
                return {{ success: false, error: errMsg }};
            }}
        }})();
        """
        try:
            result = colab_output.eval_js(js_init)
            if result and result.get("success"):
                self.is_active = True
                self.capture_mode = CaptureMode.BROWSER_WEBCAM
                return True, "Colab browser webcam initialized successfully via WebRTC."
            err_msg = (
                result.get("error", "Unknown camera error")
                if result
                else "No response from browser camera bridge."
            )
            self.is_active = False
            self.capture_mode = CaptureMode.UNAVAILABLE
            return False, err_msg
        except Exception as e:
            self.is_active = False
            self.capture_mode = CaptureMode.UNAVAILABLE
            return False, f"Failed to execute Colab camera script: {e}"

    def capture_frame(self) -> np.ndarray | None:
        """Capture and return a single BGR frame from the active stream."""
        if not self.is_active:
            return None

        if self.is_colab:
            try:
                from google.colab import output as colab_output

                js_capture = """
                (function() {
                    const video = document.getElementById("proctoring_webcam_video");
                    if (!video || !window._proctoringStream || video.readyState < 2) {
                        return null;
                    }
                    const canvas = document.createElement("canvas");
                    canvas.width = video.videoWidth || 640;
                    canvas.height = video.videoHeight || 480;
                    const ctx = canvas.getContext("2d");
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    return canvas.toDataURL("image/jpeg", 0.90);
                })();
                """
                b64_data = colab_output.eval_js(js_capture)
                if not b64_data:
                    return None
                header, encoded = b64_data.split(",", 1)
                import base64

                img_bytes = base64.b64decode(encoded)
                nparr = np.frombuffer(img_bytes, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                return None
        else:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    return frame
            return None

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Compatibility helper returning (success, frame)."""
        frame = self.capture_frame()
        return (frame is not None), frame

    def stop(self) -> None:
        """Release camera hardware and browser video tracks cleanly."""
        self.is_active = False
        self.capture_mode = CaptureMode.UNAVAILABLE

        if self.is_colab:
            try:
                from google.colab import output as colab_output

                js_stop = """
                (function() {
                    if (window._proctoringStream) {
                        window._proctoringStream.getTracks().forEach(t => t.stop());
                        window._proctoringStream = null;
                    }
                    const video = document.getElementById("proctoring_webcam_video");
                    if (video) {
                        video.srcObject = null;
                        video.remove();
                    }
                    return true;
                })();
                """
                colab_output.eval_js(js_stop)
            except Exception:
                pass

        if self.cap is not None:
            with contextlib.suppress(Exception):
                self.cap.release()
            self.cap = None

    def __enter__(self) -> WebcamStream:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
