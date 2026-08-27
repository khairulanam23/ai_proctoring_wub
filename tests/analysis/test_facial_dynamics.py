"""Tests for speech articulation, head pose and gaze measurement."""

import glob
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from proctoring.analysis.facial_dynamics import FacialDynamicsAnalyzer

MODEL = Path("models/face_landmarker.task")
requires_model = pytest.mark.skipif(
    not MODEL.exists(), reason="face_landmarker.task not downloaded"
)


@pytest.fixture(scope="module")
def analyzer():
    # Above the articulation sampling floor, so speech measurements are actually
    # taken. At the 4 fps default the analyzer correctly declines to judge speech
    # at all, which would make these assertions vacuous.
    a = FacialDynamicsAnalyzer(sampling_fps=8.0)
    yield a
    a.close()


def _portrait(name: str) -> np.ndarray:
    return cv2.resize(cv2.imread(name), (640, 480))


# ---------------------------------------------------------------------------
# Head pose
# ---------------------------------------------------------------------------


def test_pose_decomposition_is_identity_free():
    """An identity rotation must decompose to zero on all three axes."""
    pose = FacialDynamicsAnalyzer._decompose_pose(np.eye(4))
    assert abs(pose.yaw) < 1e-6
    assert abs(pose.pitch) < 1e-6
    assert abs(pose.roll) < 1e-6


def test_pose_decomposition_separates_yaw_from_pitch():
    """A pure yaw rotation must move yaw only.

    Regression test: the original decomposition used the textbook ZYX order, which
    does not match MediaPipe's axis convention. It reported a head turned sideways
    as pitched steeply downward, so a candidate glancing left produced a continuous
    stream of false ``LOOKING_AWAY`` events.
    """
    angle = math.radians(30.0)
    matrix = np.eye(4)
    # Rotation about the vertical axis in MediaPipe's frame.
    matrix[:3, :3] = np.array(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    pose = FacialDynamicsAnalyzer._decompose_pose(matrix)
    assert pose.yaw == pytest.approx(30.0, abs=0.5)
    assert abs(pose.pitch) < 1.0
    assert abs(pose.roll) < 1.0


def test_pose_decomposition_ignores_matrix_scale():
    """Scale carried in the transform must not bias the extracted angles."""
    matrix = np.eye(4)
    matrix[:3, :3] *= 4.7
    pose = FacialDynamicsAnalyzer._decompose_pose(matrix)
    assert abs(pose.yaw) < 1e-6 and abs(pose.pitch) < 1e-6 and abs(pose.roll) < 1e-6


@requires_model
def test_frontal_portraits_are_not_reported_as_looking_away(analyzer):
    """Real frontal photographs must not trip the looking-away threshold.

    The dataset portraits are all roughly camera-facing. If any registers as looking
    away, the pose extraction or the threshold is wrong, and every candidate sitting
    normally would be flagged.
    """
    samples = sorted(glob.glob("data/samples/*/*_0001.jpg"))[:6]
    if not samples:
        pytest.skip("sample portraits not available")

    for path in samples:
        result = analyzer.analyze(_portrait(path))
        if not result.face_found or result.head_pose is None:
            continue
        assert result.is_looking_away is False, (
            f"{Path(path).parent.name} reported looking away at "
            f"yaw={result.head_pose.yaw:.1f} pitch={result.head_pose.pitch:.1f}"
        )


# ---------------------------------------------------------------------------
# Speech
# ---------------------------------------------------------------------------


@requires_model
def test_static_face_is_never_reported_as_speaking(analyzer):
    """The same frame repeated has zero mouth movement and must not read as speech."""
    samples = sorted(glob.glob("data/samples/*/*_0001.jpg"))
    if not samples:
        pytest.skip("sample portraits not available")

    analyzer.reset()
    frame = _portrait(samples[0])
    for _ in range(12):
        result = analyzer.analyze(frame)

    assert result.speech_activity == pytest.approx(0.0, abs=1e-3)
    assert result.is_speaking is False


def _speech_analyzer(window: int = 8):
    """An analyzer with only the articulation state the discriminator needs.

    Built without loading the landmark model so the speech logic can be exercised
    on synthetic activation series.
    """
    from collections import deque

    analyzer = FacialDynamicsAnalyzer.__new__(FacialDynamicsAnalyzer)
    analyzer.speech_window_frames = window
    analyzer.speech_articulation_amplitude = 0.18
    analyzer.speech_min_crossings = 3
    analyzer.speech_closed_ratio = 0.45
    # Above the articulation floor, so the discriminator actually runs. Below it the
    # analyzer declines to answer at all — covered in test_detection_accuracy.py.
    analyzer.sampling_fps = 8.0
    analyzer.speech_min_sampling_fps = 6.0
    analyzer._mouth_history = deque(maxlen=window)
    return analyzer


def test_speech_requires_repeated_articulation_not_just_movement():
    """A single sustained mouth opening (a yawn) must not register as speech.

    Drives the articulation logic directly with a controlled activation series, so
    the discriminator is tested independently of the landmark model.
    """
    analyzer = _speech_analyzer()
    from proctoring.analysis.facial_dynamics import FacialDynamicsResult

    # A monotonic ramp: large total movement, but the mouth only ever opens.
    ramp = FacialDynamicsResult()
    for value in [0.0, 0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.84]:
        analyzer._measure_speech(ramp, np.zeros((1, 2), np.float32), {"jawOpen": value})
    assert ramp.is_speaking is False, "a monotonic jaw drop must not count as speech"

    # An oscillating series: repeated open/close, which is what articulation looks like.
    analyzer._mouth_history.clear()
    oscillating = FacialDynamicsResult()
    for value in [0.05, 0.40, 0.06, 0.45, 0.08, 0.42, 0.05, 0.44]:
        analyzer._measure_speech(oscillating, np.zeros((1, 2), np.float32), {"jawOpen": value})
    assert oscillating.is_speaking is True


def test_speech_verdict_withheld_until_enough_history():
    """With too few frames the analyzer reports no verdict rather than guessing."""
    from proctoring.analysis.facial_dynamics import FacialDynamicsResult

    analyzer = _speech_analyzer()

    result = FacialDynamicsResult()
    analyzer._measure_speech(result, np.zeros((1, 2), np.float32), {"jawOpen": 0.3})
    assert result.is_speaking is None


# ---------------------------------------------------------------------------
# Availability and regions
# ---------------------------------------------------------------------------


def test_missing_model_degrades_to_unavailable():
    """A missing model disables the stage instead of raising."""
    analyzer = FacialDynamicsAnalyzer(model_path="models/does_not_exist.task")
    assert analyzer.is_available is False
    result = analyzer.analyze(np.full((480, 640, 3), 120, np.uint8))
    assert result.face_found is False
    assert result.is_speaking is None


@requires_model
def test_regions_are_derived_for_downstream_analyzers(analyzer):
    """Ear and mouth regions must be produced for hand and earpiece cross-checks."""
    samples = sorted(glob.glob("data/samples/*/*_0001.jpg"))
    if not samples:
        pytest.skip("sample portraits not available")

    result = analyzer.analyze(_portrait(samples[0]))
    assert result.face_found is True
    assert len(result.ear_regions) == 2
    assert result.mouth_region is not None
    x1, y1, x2, y2 = result.mouth_region
    assert x2 > x1 and y2 > y1


# ---------------------------------------------------------------------------
# Liveness (presentation-attack signal)
# ---------------------------------------------------------------------------


def _liveness_analyzer(grace: float = 10.0) -> FacialDynamicsAnalyzer:
    """A analyzer with no model loaded, for driving the liveness logic directly."""
    from collections import deque

    analyzer = FacialDynamicsAnalyzer.__new__(FacialDynamicsAnalyzer)
    analyzer.blink_threshold = 0.45
    analyzer.liveness_grace_seconds = grace
    analyzer._blink_count = 0
    analyzer._eye_was_closed = False
    analyzer._face_first_seen = None
    analyzer._face_last_seen = None
    analyzer._mouth_history = deque(maxlen=8)
    return analyzer


def _feed_blinks(analyzer, pattern, step=0.25):
    from proctoring.analysis.facial_dynamics import FacialDynamicsResult

    result = FacialDynamicsResult()
    for index, closure in enumerate(pattern):
        result = FacialDynamicsResult()
        analyzer._measure_liveness(
            result, {"eyeBlinkLeft": closure, "eyeBlinkRight": closure}, index * step
        )
    return result


def test_a_face_that_never_blinks_is_flagged_after_the_grace_period():
    """A photograph or paused video held to the camera never blinks.

    Identity verification compares a still image against a still template, so a
    printed photo passes it outright. Blinking is the cheapest signal separating a
    person from a picture.
    """
    analyzer = _liveness_analyzer(grace=10.0)
    result = _feed_blinks(analyzer, [0.02] * 60)

    assert result.blink_count == 0
    assert result.liveness_state == "NO_BLINK_DETECTED"
    assert result.observed_seconds >= 10.0


def test_a_blinking_face_is_reported_live():
    analyzer = _liveness_analyzer(grace=10.0)
    pattern = [0.8 if i % 8 in (0, 1) else 0.02 for i in range(60)]
    result = _feed_blinks(analyzer, pattern)

    assert result.blink_count > 3
    assert result.liveness_state == "LIVE"


def test_liveness_is_withheld_before_the_grace_period_elapses():
    """A short observation is not evidence of anything; the verdict is withheld."""
    analyzer = _liveness_analyzer(grace=45.0)
    result = _feed_blinks(analyzer, [0.02] * 8)
    assert result.liveness_state == "UNKNOWN"


def test_blinks_are_counted_once_per_closure():
    """A blink is a closure followed by an opening — a long closure is still one blink."""
    analyzer = _liveness_analyzer()
    result = _feed_blinks(analyzer, [0.02, 0.9, 0.9, 0.9, 0.9, 0.02, 0.02])
    assert result.blink_count == 1


def test_liveness_window_restarts_when_the_face_leaves(analyzer):
    """A candidate who steps away must not be judged on the time they were absent."""
    import glob

    samples = sorted(glob.glob("data/samples/*/*_0001.jpg"))
    if not samples:
        pytest.skip("sample portraits not available")

    analyzer.reset()
    analyzer.analyze(_portrait(samples[0]), timestamp_seconds=0.0)
    assert analyzer._face_first_seen is not None

    analyzer.analyze(np.full((480, 640, 3), 10, np.uint8), timestamp_seconds=5.0)  # no face
    assert analyzer._face_first_seen is None


def test_reset_clears_liveness_counters():
    analyzer = _liveness_analyzer()
    _feed_blinks(analyzer, [0.9, 0.02, 0.9, 0.02])
    assert analyzer._blink_count > 0
    analyzer.reset()
    assert analyzer._blink_count == 0
    assert analyzer._face_first_seen is None


# ---------------------------------------------------------------------------
# Per-candidate calibration
# ---------------------------------------------------------------------------


@requires_model
def test_calibration_learns_a_baseline_and_shifts_the_thresholds(analyzer):
    """Calibration must make thresholds relative to this candidate's neutral pose.

    An off-centre camera produces a large constant yaw with no misconduct at all,
    and it is the single largest source of false LOOKING_AWAY events.
    """
    import glob

    samples = sorted(glob.glob("data/samples/*/*_0001.jpg"))
    if not samples:
        pytest.skip("sample portraits not available")

    frame = _portrait(samples[0])
    analyzer.reset()
    analyzer.baseline_yaw = analyzer.baseline_pitch = analyzer.baseline_gaze = 0.0
    analyzer.is_calibrated = False

    report = analyzer.calibrate([frame] * 5)
    assert report["calibrated"] is True
    assert report["samples"] == 5
    assert analyzer.is_calibrated is True

    # A perfectly still calibration should produce a near-zero spread.
    assert report["yaw_stability_deg"] < 1.0

    result = analyzer.analyze(frame)
    assert result.calibrated is True
    assert result.is_looking_away is False


@requires_model
def test_calibration_is_rejected_when_the_candidate_moved(analyzer):
    """Averaging a moving candidate would bake their movement into every reading."""
    import glob

    samples = sorted(glob.glob("data/samples/*/*_0001.jpg"))
    if len(samples) < 4:
        pytest.skip("need several distinct portraits")

    analyzer.reset()
    # Different people at different angles stand in for a candidate who moved.
    report = analyzer.calibrate([_portrait(p) for p in samples[:6]])
    if report["calibrated"]:
        pytest.skip("sample portraits happened to be too similar to trigger the guard")
    assert "varied too much" in report["reason"]


def test_calibration_needs_enough_usable_samples():
    """Too few usable frames must fail loudly rather than produce a weak baseline."""
    analyzer = FacialDynamicsAnalyzer(model_path="models/does_not_exist.task")
    report = analyzer.calibrate([np.full((480, 640, 3), 120, np.uint8)] * 5)
    assert report["calibrated"] is False
    assert "usable sample" in report["reason"]
