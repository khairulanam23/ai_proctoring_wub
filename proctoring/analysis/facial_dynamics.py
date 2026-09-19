"""Facial dynamics: speech articulation, head pose and gaze direction.

Built on the MediaPipe Face Landmarker, which returns 478 dense landmarks, 52
blendshape activations and a 4x4 facial transformation matrix per face.  That
gives three signals the sparse 5-point YuNet detector cannot provide:

* **Speech-like mouth activity** — from the temporal *shape* of the articulation
  signal across a window, not its instantaneous value.  A candidate resting with
  their mouth open, yawning, or simply having a wide neutral mouth all produce a
  high ``jawOpen`` score while saying nothing.  Only repeated opening and closing
  produces a wide swing that crosses its own mid-level many times and keeps
  returning to near-closed, so this analyzer buffers a window and measures all
  three.  There is no audio: this never establishes that speech occurred.
* **Head pose** — yaw, pitch and roll decomposed from the transformation matrix.
* **Gaze** — iris centre offset within the eye aperture, from the refined iris
  landmarks (indices 468-477).

None of these observations is proof of misconduct.  Speaking may be a candidate
reading aloud, a permitted accommodation, or someone else in the room; looking
away may be thinking. The pipeline records them for a proctor and stops there.
"""

import logging
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from proctoring.analysis.gaze import GazeDirection, GazeObservation, GazeTracker

# Landmark index groups in the MediaPipe canonical face mesh.
_UPPER_LIP_INNER = 13
_LOWER_LIP_INNER = 14
_MOUTH_LEFT_CORNER = 61
_MOUTH_RIGHT_CORNER = 291

_LEFT_IRIS = (468, 469, 470, 471, 472)
_RIGHT_IRIS = (473, 474, 475, 476, 477)
_LEFT_EYE_INNER, _LEFT_EYE_OUTER = 133, 33
_RIGHT_EYE_INNER, _RIGHT_EYE_OUTER = 362, 263

# Blendshapes forming the articulation index.
#
# The previous implementation averaged six blendshapes flat, including ``mouthClose``
# — which rises as the jaw closes and therefore *cancels* the very signal being
# measured — and two one-sided lip shapes that barely move in ordinary speech. The
# mean of that set swings a few hundredths during clear speech, far below any usable
# threshold, which is why speaking was almost never reported.
#
# The index is now dominated by jaw aperture, the one blendshape that tracks
# articulation directly, with lip shaping as a secondary term that keeps
# low-jaw-movement speech (mumbling, close-lipped speech) visible.
_JAW_BLENDSHAPE = "jawOpen"
_LIP_SHAPE_BLENDSHAPES = (
    "mouthFunnel",
    "mouthPucker",
    "mouthStretchLeft",
    "mouthStretchRight",
)
_LIP_SHAPE_WEIGHT = 0.35
"""Weight of the lip-shaping term relative to jaw aperture in the articulation index."""

# Landmarks tracing the visible perimeter of each ear, used to localise earpiece
# detections. A single anchor point is kept as a fallback for sparse meshes.
_LEFT_EAR_ANCHOR = 234
_RIGHT_EAR_ANCHOR = 454
_LEFT_EAR_PERIMETER = (234, 227, 137, 177, 132, 93, 58, 172)
_RIGHT_EAR_PERIMETER = (454, 447, 366, 401, 361, 323, 288, 397)
_DEFAULT_EAR_PADDING_RATIO = 0.55

# Eyelid-closure blendshapes, used for the liveness signal.
_BLINK_BLENDSHAPES = ("eyeBlinkLeft", "eyeBlinkRight")

LOGGER = logging.getLogger(__name__)


@dataclass
class HeadPose:
    """Head orientation in degrees, relative to facing the camera squarely."""

    yaw: float = 0.0  # negative = turned to the candidate's right
    pitch: float = 0.0  # negative = looking down
    roll: float = 0.0  # head tilt

    def to_dict(self) -> dict[str, Any]:
        return {
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "roll": round(self.roll, 2),
        }


@dataclass
class FacialDynamicsResult:
    """Per-frame facial dynamics measurement.

    ``None`` on any field means the measurement could not be taken this frame
    (no face, or the landmarker was unavailable) — never a neutral stand-in.
    """

    face_found: bool = False
    landmarks: np.ndarray | None = None  # (N, 2) pixel coordinates
    face_bbox: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2)

    # Speech
    mouth_open_ratio: float | None = None
    speech_activity: float | None = None  # 0-1 articulation-movement score
    is_speaking: bool | None = None
    """``None`` means *not measured*: too little history yet, or a sampling rate too
    low for the measurement to be possible. Never a stand-in for "not speaking"."""

    speech_measurable: bool = True
    """False when the configured sampling rate is below the articulation floor. The
    frame carries no claim about speech at all in that case."""

    # Head pose & gaze
    head_pose: HeadPose | None = None
    is_looking_away: bool | None = None
    gaze: GazeObservation | None = None
    gaze_offset: float | None = None  # 0 = centred, 1 = at the eye corner
    is_gaze_off_screen: bool | None = None

    # Liveness
    eye_closure: float | None = None
    blink_count: int = 0
    observed_seconds: float = 0.0
    liveness_state: str = "UNKNOWN"  # "LIVE" | "NO_BLINK_DETECTED" | "UNKNOWN"
    calibrated: bool = False
    """Whether pose and gaze were judged against a per-candidate baseline."""

    baseline_yaw: float = 0.0
    baseline_pitch: float = 0.0
    """The neutral pose this frame was measured against. Published on the result so
    the reporting layer applies the same baseline the analyzer did — it previously
    re-evaluated head pose against a hard-coded zero, discarding calibration and
    with it the correction for an off-centre camera."""

    # Regions other analyzers reuse
    ear_regions: list[tuple[int, int, int, int]] = field(default_factory=list)
    mouth_region: tuple[int, int, int, int] | None = None

    inference_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_found": self.face_found,
            "mouth_open_ratio": round(self.mouth_open_ratio, 4)
            if self.mouth_open_ratio is not None
            else None,
            "speech_activity": round(self.speech_activity, 4)
            if self.speech_activity is not None
            else None,
            "is_speaking": self.is_speaking,
            "speech_measurable": self.speech_measurable,
            "head_pose": self.head_pose.to_dict() if self.head_pose else None,
            "is_looking_away": self.is_looking_away,
            "gaze": self.gaze.to_dict() if self.gaze else None,
            "gaze_offset": round(self.gaze_offset, 4) if self.gaze_offset is not None else None,
            "is_gaze_off_screen": self.is_gaze_off_screen,
            "eye_closure": round(self.eye_closure, 4) if self.eye_closure is not None else None,
            "blink_count": self.blink_count,
            "liveness_state": self.liveness_state,
            "calibrated": self.calibrated,
            "baseline_yaw": round(self.baseline_yaw, 2),
            "baseline_pitch": round(self.baseline_pitch, 2),
            "inference_ms": round(self.inference_ms, 2),
        }


class FacialDynamicsAnalyzer:
    """Measures speech articulation, head pose and gaze from dense face landmarks.

    The analyzer is stateful: speech detection needs a short history of mouth
    activation, so one instance must be used for one session and fed frames in
    order.
    """

    DEFAULT_MODEL = "models/face_landmarker.task"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        speech_window_frames: int = 12,
        speech_articulation_amplitude: float = 0.18,
        speech_min_crossings: int = 3,
        speech_closed_ratio: float = 0.45,
        sampling_fps: float = 4.0,
        speech_min_sampling_fps: float = 6.0,
        yaw_limit_degrees: float = 30.0,
        pitch_limit_degrees: float = 25.0,
        gaze_offset_limit: float = 0.32,
        blink_threshold: float = 0.45,
        liveness_grace_seconds: float = 45.0,
        ear_region_padding_ratio: float = _DEFAULT_EAR_PADDING_RATIO,
        max_faces: int = 2,
        device: str | None = None,
    ) -> None:
        """
        Args:
            speech_window_frames: History length for the articulation measure. At the
                default 4 fps sampling rate twelve frames is a three-second window.
                The caller is expected to derive this from the configured sampling
                rate and ``ExamPolicy.speech_window_seconds`` so the window keeps the
                same duration whatever the frame rate.
            speech_articulation_amplitude: Robust peak-to-trough swing (p90 - p10) of
                the articulation index required across the window.
            speech_min_crossings: Times the index must cross its own mid-level within
                the window. Speech crosses repeatedly; a yawn crosses twice.
            speech_closed_ratio: The index must fall below this fraction of the
                window amplitude at least once, so a held-open mouth cannot qualify.
            sampling_fps: The rate frames actually arrive at, used only to decide
                whether articulation is measurable at all.
            speech_min_sampling_fps: Floor below which the mouth signal is aliased
                beyond recovery and no speech verdict is issued.
            yaw_limit_degrees / pitch_limit_degrees: Head rotation beyond which the
                candidate is recorded as looking away.
            gaze_offset_limit: Normalised iris displacement beyond which gaze is
                recorded as off-screen.
            blink_threshold: Eyelid-closure activation above which the eye counts as
                shut. Blinks are counted on the falling edge.
            ear_region_padding_ratio: How far each ear box is grown beyond the ear
                landmarks. A slightly oversized region only weakens the geometric
                filter; an undersized one discards the detection outright.
            liveness_grace_seconds: How long a face may be continuously observed with
                zero blinks before liveness is reported as suspect. A person blinks
                every few seconds; a printed photograph or a paused video never does.
                Set generously — some people blink rarely, and a false accusation of
                spoofing is serious.
        """
        self.model_path = Path(model_path)
        self._requested_device = str(device or "cpu").lower()
        self._is_gpu_active = False
        self.device = "cpu"

        self.speech_window_frames = max(4, int(speech_window_frames))
        self.speech_articulation_amplitude = float(speech_articulation_amplitude)
        self.speech_min_crossings = int(speech_min_crossings)
        self.speech_closed_ratio = float(speech_closed_ratio)
        self.sampling_fps = float(sampling_fps)
        self.speech_min_sampling_fps = float(speech_min_sampling_fps)
        self.yaw_limit_degrees = float(yaw_limit_degrees)
        self.pitch_limit_degrees = float(pitch_limit_degrees)
        self.gaze_offset_limit = float(gaze_offset_limit)
        self.blink_threshold = float(blink_threshold)
        self.liveness_grace_seconds = float(liveness_grace_seconds)
        self.ear_region_padding_ratio = float(ear_region_padding_ratio)
        self.max_faces = int(max_faces)

        # Per-candidate baseline, set by calibrate(). Thresholds are measured
        # relative to this rather than to an assumed head-on camera.
        self.baseline_yaw: float = 0.0
        self.baseline_pitch: float = 0.0
        self.baseline_gaze: float = 0.0
        self.is_calibrated: bool = False

        self.gaze_tracker = GazeTracker(horizontal_threshold=self.gaze_offset_limit)

        # Liveness bookkeeping, spanning the whole session.
        self._blink_count = 0
        self._eye_was_closed = False
        self._face_first_seen: float | None = None
        self._face_last_seen: float | None = None

        self._landmarker = None
        self._mp = None
        self._mouth_history: deque = deque(maxlen=self.speech_window_frames)
        self.is_available = self._load()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load(self) -> bool:
        """Load the landmarker, degrading to unavailable rather than raising.

        A missing model or missing mediapipe disables the stage; the pipeline runs
        without facial dynamics rather than failing the whole session.
        """
        if not self.model_path.exists():
            return False
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            self._mp = mp

            # Try GPU delegate if requested and supported by the runtime environment
            if self._requested_device in ("cuda", "cuda:0", "gpu"):
                try:
                    self._landmarker = vision.FaceLandmarker.create_from_options(
                        vision.FaceLandmarkerOptions(
                            base_options=BaseOptions(
                                model_asset_path=str(self.model_path),
                                delegate=BaseOptions.Delegate.GPU,
                            ),
                            output_face_blendshapes=True,
                            output_facial_transformation_matrixes=True,
                            num_faces=self.max_faces,
                        )
                    )
                    self._is_gpu_active = True
                    self.device = "cuda"
                    return True
                except Exception as gpu_exc:
                    LOGGER.info(
                        "MediaPipe FaceLandmarker GPU delegate unavailable (%s); falling back to CPU XNNPACK.",
                        gpu_exc,
                    )

            # Standard CPU path (Google TFLite XNNPACK)
            self._landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(self.model_path),
                        delegate=BaseOptions.Delegate.CPU,
                    ),
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                    num_faces=self.max_faces,
                )
            )
            self._is_gpu_active = False
            self.device = "cpu"
            return True
        except Exception as exc:
            LOGGER.info("Face landmarker unavailable: %s", exc)
            return False

    @property
    def is_gpu_accelerated(self) -> bool:
        """Whether the analyzer is currently executing on a GPU device."""
        return self._is_gpu_active

    def close(self) -> None:
        """Release the underlying landmarker."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception as exc:
                LOGGER.debug("Face landmarker close failed: %s", exc)
            self._landmarker = None

    def calibrate(self, frames: Sequence[np.ndarray]) -> dict[str, Any]:
        """Learn this candidate's neutral head pose and gaze from sample frames.

        The single largest source of false ``LOOKING_AWAY`` events is a camera that
        is not where the thresholds assume it is: a laptop on a stand, a webcam
        clipped to a second monitor, or a candidate sitting off to one side all
        produce a large constant yaw with no misconduct whatsoever.

        Ten seconds of "look at your screen normally" fixes that far better than any
        threshold tuning, because it measures the offset instead of guessing it.
        Angles are subsequently reported relative to this baseline.

        The baseline is deliberately rejected if the samples disagree — a candidate
        who moved around during calibration has not given a usable neutral pose, and
        silently averaging it would bake their movement into every later reading.
        """
        yaws, pitches, gazes, gaze_pairs = [], [], [], []
        for frame in frames:
            result = self.analyze(frame)
            if not result.face_found or result.head_pose is None:
                continue
            yaws.append(result.head_pose.yaw)
            pitches.append(result.head_pose.pitch)
            if result.gaze_offset is not None:
                gazes.append(result.gaze_offset)
            if result.gaze is not None and result.gaze.direction != GazeDirection.UNKNOWN:
                gaze_pairs.append((result.gaze.horizontal, result.gaze.vertical))

        if len(yaws) < 3:
            return {
                "calibrated": False,
                "reason": f"only {len(yaws)} usable sample(s); need at least 3",
            }

        yaw_spread = float(np.std(yaws))
        pitch_spread = float(np.std(pitches))
        if yaw_spread > 12.0 or pitch_spread > 12.0:
            return {
                "calibrated": False,
                "reason": (
                    f"pose varied too much during calibration "
                    f"(yaw sd {yaw_spread:.1f}deg, pitch sd {pitch_spread:.1f}deg); "
                    f"ask the candidate to hold still and look at the screen"
                ),
            }

        self.baseline_yaw = float(np.median(yaws))
        self.baseline_pitch = float(np.median(pitches))
        self.baseline_gaze = float(np.median(gazes)) if gazes else 0.0

        if gaze_pairs:
            self.gaze_tracker.calibrate(gaze_pairs)

        self.is_calibrated = True
        self.reset(clear_calibration=False)

        return {
            "calibrated": True,
            "samples": len(yaws),
            "baseline_yaw": round(self.baseline_yaw, 2),
            "baseline_pitch": round(self.baseline_pitch, 2),
            "baseline_gaze": round(self.baseline_gaze, 3),
            "baseline_gaze_h": round(self.gaze_tracker.calibration.baseline_horizontal, 4),
            "baseline_gaze_v": round(self.gaze_tracker.calibration.baseline_vertical, 4),
            "yaw_stability_deg": round(yaw_spread, 2),
            "pitch_stability_deg": round(pitch_spread, 2),
        }

    def reset(self, clear_calibration: bool = True) -> None:
        """Clear per-session state (speech history, liveness counters, and calibration)."""
        self._mouth_history.clear()
        self._blink_count = 0
        self._eye_was_closed = False
        self._face_first_seen = None
        self._face_last_seen = None
        if clear_calibration:
            self.baseline_yaw = 0.0
            self.baseline_pitch = 0.0
            self.baseline_gaze = 0.0
            self.is_calibrated = False
            if hasattr(self, "gaze_tracker") and self.gaze_tracker is not None:
                self.gaze_tracker.reset()

    # ------------------------------------------------------------------
    # Per-frame analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        frame: np.ndarray,
        timestamp_seconds: float | None = None,
        rgb_frame: np.ndarray | None = None,
    ) -> FacialDynamicsResult:
        """Measure facial dynamics for one frame.

        ``timestamp_seconds`` is used only for liveness accounting; omit it and
        the blink measure still counts blinks but cannot report how long the face
        has been watched without one.
        ``rgb_frame`` allows passing an already-converted RGB image to eliminate
        redundant cvtColor operations across multiple MediaPipe stages.
        """
        import time

        result = FacialDynamicsResult()
        if not self.is_available or frame is None or frame.size == 0:
            return result

        t0 = time.perf_counter()
        try:
            image_data = rgb_frame if rgb_frame is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=image_data,
            )
            detection = self._landmarker.detect(mp_image)
        except Exception as exc:
            LOGGER.warning("Face landmark inference failed: %s", exc)
            return result
        result.inference_ms = (time.perf_counter() - t0) * 1000.0

        if not detection.face_landmarks:
            # No face this frame: forget history so a gap does not stitch two
            # unrelated speech bursts into one, and restart the liveness window so a
            # candidate who steps away is not judged on the time they were absent.
            self._mouth_history.clear()
            self._face_first_seen = None
            return result

        height, width = frame.shape[:2]
        landmarks = np.array(
            [[lm.x * width, lm.y * height] for lm in detection.face_landmarks[0]],
            dtype=np.float32,
        )

        result.face_found = True
        result.calibrated = self.is_calibrated
        result.baseline_yaw = self.baseline_yaw
        result.baseline_pitch = self.baseline_pitch
        result.landmarks = landmarks
        result.face_bbox = self._bbox_from_landmarks(landmarks, width, height)
        result.ear_regions = self._ear_regions(
            landmarks, width, height, padding_ratio=self.ear_region_padding_ratio
        )
        result.mouth_region = self._mouth_region(landmarks, width, height)

        blendshapes = (
            {c.category_name: c.score for c in detection.face_blendshapes[0]}
            if detection.face_blendshapes
            else {}
        )
        self._measure_speech(result, landmarks, blendshapes)
        self._measure_liveness(result, blendshapes, timestamp_seconds)

        if detection.facial_transformation_matrixes:
            result.head_pose = self._decompose_pose(
                np.array(detection.facial_transformation_matrixes[0])
            )
            # Judge deviation from this candidate's neutral pose, not from an
            # assumed head-on camera.
            result.is_looking_away = (
                abs(result.head_pose.yaw - self.baseline_yaw) > self.yaw_limit_degrees
                or abs(result.head_pose.pitch - self.baseline_pitch) > self.pitch_limit_degrees
            )

        self._measure_gaze(result, landmarks)
        return result

    # ------------------------------------------------------------------
    # Speech
    # ------------------------------------------------------------------

    def _measure_speech(
        self,
        result: FacialDynamicsResult,
        landmarks: np.ndarray,
        blendshapes: dict[str, float],
    ) -> None:
        """Score speech-like articulation across a window of recent frames.

        **There is no audio.** Nothing here establishes that the candidate spoke; it
        establishes that the mouth moved the way a mouth moves during speech. The
        event this feeds is a prompt to watch the snapshot, not a finding.

        Three conditions must hold together, because each on its own has a common
        innocent cause:

        ``amplitude``
            Robust peak-to-trough swing (p90 - p10) of the articulation index over
            the window. Rules out a still face and ordinary expression drift.
        ``crossings``
            How many times the index crosses its own mid-level. This is the measure
            that separates speech from a yawn: a yawn crosses twice however deep it
            is, while continuous articulation crosses repeatedly. Crossings are
            counted rather than frame-to-frame sign changes because at 4 fps the
            3-6 Hz syllable rate is aliased and per-frame deltas are close to noise —
            the previous implementation counted exactly those deltas.
        ``returns to closed``
            The mouth must come back down near its window minimum. A yawn, a smile
            and a resting open mouth all hold a level instead.
        """
        # Geometric mouth opening, normalised by mouth width so it is invariant to
        # how close the candidate sits to the camera. Reported for context, and used
        # as the articulation index when blendshapes are unavailable.
        if landmarks.shape[0] > max(_LOWER_LIP_INNER, _MOUTH_RIGHT_CORNER):
            vertical = float(
                np.linalg.norm(landmarks[_UPPER_LIP_INNER] - landmarks[_LOWER_LIP_INNER])
            )
            horizontal = float(
                np.linalg.norm(landmarks[_MOUTH_LEFT_CORNER] - landmarks[_MOUTH_RIGHT_CORNER])
            )
            result.mouth_open_ratio = vertical / horizontal if horizontal > 1e-6 else 0.0

        if self.sampling_fps < self.speech_min_sampling_fps:
            # Below the articulation floor the signal is aliased past recovery. Say
            # so, rather than returning a verdict the data cannot support.
            result.speech_measurable = False
            result.is_speaking = None
            return

        activation = self._articulation_index(blendshapes, result.mouth_open_ratio)
        if activation is None:
            return

        self._mouth_history.append(activation)
        if len(self._mouth_history) < max(4, self.speech_window_frames // 2):
            # Too little history to judge; withhold the verdict rather than guess.
            result.speech_activity = 0.0
            result.is_speaking = None
            return

        series = np.asarray(self._mouth_history, dtype=np.float32)
        low, high = (float(v) for v in np.percentile(series, [10, 90]))
        amplitude = high - low
        result.speech_activity = float(
            min(1.0, amplitude / max(1e-6, self.speech_articulation_amplitude))
        )

        midpoint = (float(series.min()) + float(series.max())) / 2.0
        crossings = self._count_crossings(series, midpoint)

        # "Returned to closed" is measured against the window's own range, so it holds
        # for a candidate whose neutral mouth rests slightly open.
        closed_level = low + self.speech_closed_ratio * amplitude
        returns_to_closed = bool(series.min() <= closed_level)

        result.is_speaking = bool(
            amplitude >= self.speech_articulation_amplitude
            and crossings >= self.speech_min_crossings
            and returns_to_closed
        )

    @staticmethod
    def _articulation_index(
        blendshapes: dict[str, float],
        mouth_open_ratio: float | None,
    ) -> float | None:
        """Combine jaw aperture and lip shaping into one 0-1 articulation signal.

        Jaw aperture carries most of the information. Lip shaping is added at a
        reduced weight so close-lipped or mumbled speech, which moves the jaw very
        little, still produces a measurable series. ``mouthClose`` is deliberately
        excluded: it rises as the jaw closes and averaging it in cancels the signal.
        """
        if blendshapes:
            jaw = float(blendshapes.get(_JAW_BLENDSHAPE, 0.0))
            shaping = max(
                (float(blendshapes.get(name, 0.0)) for name in _LIP_SHAPE_BLENDSHAPES),
                default=0.0,
            )
            return float(min(1.0, jaw + _LIP_SHAPE_WEIGHT * shaping))
        if mouth_open_ratio is not None:
            return float(mouth_open_ratio)
        return None

    @staticmethod
    def _count_crossings(series: np.ndarray, level: float) -> int:
        """Count transitions of ``series`` from below ``level`` to above it, and back.

        Samples sitting exactly on the level are attributed to the side the signal
        was last on, so a flat series never accumulates crossings.
        """
        above = series > level
        return int(np.count_nonzero(above[1:] != above[:-1]))

    # ------------------------------------------------------------------
    # Liveness
    # ------------------------------------------------------------------

    def _measure_liveness(
        self,
        result: FacialDynamicsResult,
        blendshapes: dict[str, float],
        timestamp_seconds: float | None,
    ) -> None:
        """Count blinks and flag a face that has never blinked.

        Identity verification compares a still image against a still template, so a
        printed photograph or a phone screen held to the camera passes it. Blinking
        is the cheapest signal that separates a person from a picture, and the
        blendshapes needed for it are already computed by the landmarker.

        This is a **weak** signal deliberately reported as a prompt, not a finding:
        low blink rates occur naturally, and a long grace period applies before
        anything is said at all.
        """
        if not blendshapes:
            return

        closure = float(np.mean([blendshapes.get(name, 0.0) for name in _BLINK_BLENDSHAPES]))
        result.eye_closure = closure

        # Count on the falling edge: a blink is a closure followed by an opening.
        is_closed = closure >= self.blink_threshold
        if self._eye_was_closed and not is_closed:
            self._blink_count += 1
        self._eye_was_closed = is_closed
        result.blink_count = self._blink_count

        if timestamp_seconds is None:
            result.liveness_state = "LIVE" if self._blink_count > 0 else "UNKNOWN"
            return

        if self._face_first_seen is None:
            self._face_first_seen = timestamp_seconds
        self._face_last_seen = timestamp_seconds
        observed = max(0.0, timestamp_seconds - self._face_first_seen)
        result.observed_seconds = observed

        if self._blink_count > 0:
            result.liveness_state = "LIVE"
        elif observed >= self.liveness_grace_seconds:
            result.liveness_state = "NO_BLINK_DETECTED"
        else:
            result.liveness_state = "UNKNOWN"

    # ------------------------------------------------------------------
    # Head pose & gaze
    # ------------------------------------------------------------------

    @staticmethod
    def _decompose_pose(matrix: np.ndarray) -> HeadPose:
        """Decompose the 4x4 facial transformation matrix into head Euler angles.

        MediaPipe returns a head-to-camera transform whose axis order does not match
        the textbook ZYX decomposition: applying that naively yields a face turned
        sideways reported as pitched downward, which produces a constant stream of
        spurious "looking away" events for a candidate sitting normally.

        The extraction below is validated against known frontal and turned-head
        imagery in ``tests/analysis/test_facial_dynamics.py``: frontal faces must
        return near-zero on all three axes, and a head turned to the side must move
        yaw, not pitch.

        The rotation block is renormalised first because the matrix carries scale,
        and scale left in place biases the extracted angles.
        """
        rotation = np.array(matrix[:3, :3], dtype=np.float64)
        scale = float(np.linalg.norm(rotation[:, 0]))
        if scale > 1e-9:
            rotation = rotation / scale

        # Clamp guards against asin domain errors from accumulated float error.
        pitch = math.asin(float(np.clip(-rotation[1, 2], -1.0, 1.0)))
        if abs(rotation[2, 2]) > 1e-6 or abs(rotation[0, 2]) > 1e-6:
            yaw = math.atan2(rotation[0, 2], rotation[2, 2])
        else:  # gimbal lock: yaw is indeterminate, fold it into roll
            yaw = 0.0
        roll = math.atan2(rotation[1, 0], rotation[1, 1])

        return HeadPose(
            yaw=math.degrees(yaw),
            pitch=math.degrees(pitch),
            roll=math.degrees(roll),
        )

    def _measure_gaze(self, result: FacialDynamicsResult, landmarks: np.ndarray) -> None:
        """Estimate normalized gaze and directional classification from iris position.

        Uses the refined iris landmarks (indices 468-477); if unavailable, gaze is
        left unmeasured.
        """
        if landmarks.shape[0] <= max(_RIGHT_IRIS):
            return

        gaze_obs = self.gaze_tracker.measure(
            landmarks=landmarks,
            gaze_offset_limit=self.gaze_offset_limit,
        )
        if gaze_obs.direction != GazeDirection.UNKNOWN:
            result.gaze = gaze_obs
            result.gaze_offset = gaze_obs.offset
            result.is_gaze_off_screen = gaze_obs.is_off_screen

    # ------------------------------------------------------------------
    # Regions
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_from_landmarks(
        landmarks: np.ndarray, width: int, height: int
    ) -> tuple[int, int, int, int]:
        x1, y1 = landmarks.min(axis=0)
        x2, y2 = landmarks.max(axis=0)
        return (max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2)))

    @staticmethod
    def _ear_regions(
        landmarks: np.ndarray,
        width: int,
        height: int,
        padding_ratio: float = _DEFAULT_EAR_PADDING_RATIO,
    ) -> list[tuple[int, int, int, int]]:
        """Approximate boxes around each ear, used to localise earpiece detections.

        Previously each box was a small square centred on a single silhouette
        landmark. That landmark sits on the *attachment* line of the ear, so a bud in
        the ear canal, a hook over the top of the ear, and the earpiece of a headset
        all fell outside the box — and the wearable detector discards anything that
        does, which silently threw away genuine detections.

        Both the whole visible ear perimeter and a generous pad are used now. A
        region that is somewhat too large costs only a weaker geometric filter; one
        that is too small costs the detection entirely.
        """
        regions: list[tuple[int, int, int, int]] = []
        face_width = float(np.ptp(landmarks[:, 0]))

        for indices, anchor in (
            (_LEFT_EAR_PERIMETER, _LEFT_EAR_ANCHOR),
            (_RIGHT_EAR_PERIMETER, _RIGHT_EAR_ANCHOR),
        ):
            available = [i for i in indices if i < landmarks.shape[0]]
            if len(available) >= 3:
                points = landmarks[available]
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
            elif landmarks.shape[0] > anchor:
                # Sparse mesh: fall back to a square around the single anchor point.
                cx, cy = landmarks[anchor]
                radius = max(12.0, face_width * 0.16)
                x1, y1, x2, y2 = cx - radius, cy - radius, cx + radius, cy + radius
            else:
                continue

            pad_x = max(8.0, (x2 - x1) * padding_ratio)
            pad_y = max(8.0, (y2 - y1) * padding_ratio)
            regions.append(
                (
                    max(0, int(x1 - pad_x)),
                    max(0, int(y1 - pad_y)),
                    min(width, int(x2 + pad_x)),
                    min(height, int(y2 + pad_y)),
                )
            )
        return regions

    @staticmethod
    def _mouth_region(
        landmarks: np.ndarray, width: int, height: int
    ) -> tuple[int, int, int, int] | None:
        if landmarks.shape[0] <= _MOUTH_RIGHT_CORNER:
            return None
        points = landmarks[
            [_MOUTH_LEFT_CORNER, _MOUTH_RIGHT_CORNER, _UPPER_LIP_INNER, _LOWER_LIP_INNER]
        ]
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        pad = max(6.0, (x2 - x1) * 0.25)
        return (
            max(0, int(x1 - pad)),
            max(0, int(y1 - pad)),
            min(width, int(x2 + pad)),
            min(height, int(y2 + pad)),
        )
