"""Real-world detection accuracy for earpieces, speech-like activity and head movement.

These are the three behaviours webcam testing showed the pipeline was missing. Each
group below pins the reason it was missed, so a future change that reintroduces the
cause fails here rather than in an exam.

The signals are synthesised rather than captured, because the point of each test is
the discriminator, not the landmark model: driving the analyzers with controlled
series is the only way to state what separates talking from yawning, or a glance
from a pattern, in terms a reader can check.
"""

from collections import deque

import numpy as np
import pytest

from proctoring.analysis.facial_dynamics import (
    FacialDynamicsAnalyzer,
    FacialDynamicsResult,
    HeadPose,
)
from proctoring.analysis.gaze import GazeDirection, GazeObservation
from proctoring.analysis.hands import HandAnalysisResult
from proctoring.analysis.head_movement import HeadMovementTracker
from proctoring.analysis.observer import BehaviourObserver
from proctoring.analysis.policy import ExamMode, ExamPolicy, StrictnessLevel
from proctoring.analysis.wearables import (
    WearableAnalysisResult,
    WearableDetection,
    WearableDetector,
)
from proctoring.config import SessionConfig
from proctoring.core.events import DetectorInfo, EventStatus, EventType
from proctoring.observation import FrameObservation
from proctoring.temporal.aggregator import UnifiedTemporalAggregator

STRICT = ExamPolicy.for_level(StrictnessLevel.STRICT)
DETECTOR = DetectorInfo(name="test", version="1.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _observation(**overrides) -> FrameObservation:
    base = {"frame_index": 0, "timestamp_seconds": 0.0, "iso_timestamp": "1970-01-01T00:00:00Z"}
    base.update(overrides)
    return FrameObservation(**base)


def _dynamics(**overrides) -> FacialDynamicsResult:
    result = FacialDynamicsResult(face_found=True, face_bbox=(100, 100, 200, 220))
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


def _observer(level=StrictnessLevel.STRICT, mode=ExamMode.DIGITAL_SCREEN, window=1):
    policy = ExamPolicy.for_level(level, mode)
    policy.smoothing_window_frames = window
    return BehaviourObserver(policy)


def _speech_analyzer(policy=STRICT, fps=6.0):
    """An analyzer carrying only the state the articulation discriminator needs."""
    window = max(4, round(policy.speech_window_seconds * fps))
    analyzer = FacialDynamicsAnalyzer.__new__(FacialDynamicsAnalyzer)
    analyzer.speech_window_frames = window
    analyzer.speech_articulation_amplitude = policy.speech_articulation_amplitude
    analyzer.speech_min_crossings = policy.speech_min_crossings
    analyzer.speech_closed_ratio = policy.speech_closed_ratio
    analyzer.sampling_fps = fps
    analyzer.speech_min_sampling_fps = policy.speech_min_sampling_fps
    analyzer._mouth_history = deque(maxlen=window)
    return analyzer


def _feed_speech(analyzer, series):
    """Push an activation series through the analyzer, returning the final verdict."""
    result = FacialDynamicsResult()
    landmarks = np.zeros((1, 2), np.float32)
    for value in series:
        result = FacialDynamicsResult()
        analyzer._measure_speech(result, landmarks, {"jawOpen": float(np.clip(value, 0.0, 1.0))})
    return result


def _articulation(fps, seconds, hz=3.7, level=0.30, swing=0.28, seed=0):
    """A talking mouth: sustained opening and closing at the syllable rate."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fps * seconds)) / fps
    return level + swing * np.sin(2 * np.pi * hz * t) + rng.normal(0, 0.03, t.size)


def _tracker(policy=STRICT):
    return HeadMovementTracker(
        window_seconds=policy.head_pattern_window_seconds,
        episode_min_seconds=policy.head_episode_min_seconds,
        repeat_episode_count=policy.head_repeat_episode_count,
        reversal_velocity_deg_per_s=policy.head_reversal_velocity_deg_per_s,
        min_reversals=policy.head_min_reversals,
    )


def _feed_pose(segments, policy=STRICT, fps=6.0):
    """Drive the tracker with (duration_seconds, yaw, pitch) segments."""
    tracker = _tracker(policy)
    timestamp, pattern = 0.0, None
    for duration, yaw, pitch in segments:
        for _ in range(max(1, int(duration * fps))):
            pattern = tracker.update(
                timestamp, yaw, pitch, policy.is_pose_deviated(yaw=yaw, pitch=pitch)
            )
            timestamp += 1.0 / fps
    return pattern


# ---------------------------------------------------------------------------
# Ear-region localisation and wearable association
# ---------------------------------------------------------------------------


class _FakeBox:
    def __init__(self, cls, conf, xyxy):
        self.cls, self.conf = cls, conf
        self.xyxy = [np.array(xyxy, dtype=np.float32)]


class _FakePrediction:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYoloWorld:
    """Stands in for YOLO-World, reporting whatever the test scripted per image size.

    Keyed by image shape so the full-frame pass and the upscaled ear crop can be
    given different answers — which is the whole point of the second pass.
    """

    def __init__(self, names, by_shape):
        self.model = type("Inner", (), {"names": names})()
        self._by_shape = by_shape
        self.seen_shapes = []

    def predict(self, image, conf=0.0, verbose=False):
        self.seen_shapes.append(image.shape[:2])
        return [_FakePrediction(self._by_shape(image.shape[:2]))]


def _detector(names, by_shape, **kwargs):
    detector = WearableDetector.__new__(WearableDetector)
    detector.model = _FakeYoloWorld(names, by_shape)
    detector.is_available = True
    detector.confidence_threshold = kwargs.get("confidence_threshold", 0.20)
    detector.earbud_confidence_threshold = kwargs.get("earbud_confidence_threshold", 0.28)
    detector.enable_ear_region_zoom = kwargs.get("enable_ear_region_zoom", True)
    detector.ear_roi_target_px = kwargs.get("ear_roi_target_px", 320)
    return detector


def test_ear_regions_span_the_ear_not_a_single_silhouette_point():
    """The box must cover the whole ear, or genuine detections get discarded.

    A square centred on one contour landmark excluded the ear canal and anything
    hooked over the top of the ear, and the detector throws away every earpiece box
    that falls outside — which is how real earbuds went unreported.
    """
    landmarks = np.zeros((478, 2), np.float32)
    landmarks[:, 0] = np.linspace(100, 200, 478)  # face spanning 100 px
    for index, (x, y) in {
        234: (100, 150),
        227: (104, 130),
        137: (106, 165),
        177: (110, 172),
        132: (108, 176),
        93: (104, 182),
        58: (110, 188),
        172: (118, 192),
    }.items():
        landmarks[index] = (x, y)

    regions = FacialDynamicsAnalyzer._ear_regions(landmarks, 640, 480, padding_ratio=0.55)
    assert len(regions) == 2

    left = regions[0]
    height = left[3] - left[1]
    assert height >= 60, "the box must span the ear plus its padding, not a small square"
    # Every perimeter landmark must fall inside its own region.
    for index in (234, 137, 132, 93, 172):
        x, y = landmarks[index]
        assert left[0] <= x <= left[2] and left[1] <= y <= left[3]


def test_ear_regions_stay_inside_the_frame():
    """An ear at the frame edge must not produce a box with negative coordinates."""
    landmarks = np.zeros((478, 2), np.float32)
    landmarks[:, 0] = np.linspace(0, 60, 478)
    for index in (234, 227, 137, 177, 132, 93, 58, 172):
        landmarks[index] = (2, 5)
    regions = FacialDynamicsAnalyzer._ear_regions(landmarks, 640, 480)
    for x1, y1, x2, y2 in regions:
        assert 0 <= x1 < x2 <= 640
        assert 0 <= y1 < y2 <= 480


def test_earbud_found_only_in_the_magnified_ear_crop_is_reported():
    """The second pass is what makes small earpieces detectable at all.

    A bud a dozen pixels across is below the detector's resolving power on the full
    frame. The fake model reflects that: it reports nothing at frame scale and finds
    the bud once the ear is upscaled.
    """
    ear = (300, 200, 340, 250)

    def by_shape(shape):
        if shape == (480, 640):
            return []  # full frame: the bud is too small to resolve
        return [_FakeBox(0, 0.55, (100, 100, 160, 160))]  # upscaled ear crop

    detector = _detector({0: "wireless earbud in ear"}, by_shape)
    frame = np.zeros((480, 640, 3), np.uint8)
    result = detector.detect(frame, ear_regions=[ear])

    assert result.ear_regions_scanned == 1
    assert [d.target for d in result.detections] == ["earbuds"]
    detection = result.detections[0]
    assert detection.source == "ear_zoom"
    assert detection.ear_anchored is True
    # Mapped back into frame coordinates, inside the ear region it came from.
    x1, y1, x2, y2 = detection.bbox
    assert ear[0] <= x1 < x2 <= ear[2] + 1
    assert ear[1] <= y1 < y2 <= ear[3] + 1


def test_zoom_pass_can_be_disabled_and_then_finds_nothing_small():
    ear = (300, 200, 340, 250)
    detector = _detector(
        {0: "wireless earbud in ear"},
        lambda shape: [] if shape == (480, 640) else [_FakeBox(0, 0.55, (10, 10, 40, 40))],
        enable_ear_region_zoom=False,
    )
    result = detector.detect(np.zeros((480, 640, 3), np.uint8), ear_regions=[ear])
    assert result.detections == []
    assert result.ear_regions_scanned == 0


def test_earpiece_far_from_any_ear_is_still_rejected():
    """Widening the ear region must not turn the geometric filter off."""
    detector = _detector(
        {0: "bluetooth earpiece"},
        lambda shape: [_FakeBox(0, 0.9, (10, 10, 40, 40))] if shape == (480, 640) else [],
    )
    result = detector.detect(np.zeros((480, 640, 3), np.uint8), ear_regions=[(300, 200, 340, 250)])
    assert result.detections == []
    assert result.rejected_count == 1


def test_hand_at_ear_strengthens_but_never_creates_an_earpiece_detection():
    """Two weak signals agreeing is worth recording; a gesture alone is not evidence."""
    ear = (300, 200, 340, 250)

    nothing = _detector({0: "earbud"}, lambda shape: [])
    empty = nothing.detect(np.zeros((480, 640, 3), np.uint8), ear_regions=[ear], hand_at_ear=True)
    assert empty.detections == [], "a hand at the ear must not manufacture a detection"

    found = _detector(
        {0: "earbud"},
        lambda shape: [] if shape == (480, 640) else [_FakeBox(0, 0.32, (100, 100, 150, 150))],
    )
    result = found.detect(np.zeros((480, 640, 3), np.uint8), ear_regions=[ear], hand_at_ear=True)
    detection = result.detections[0]
    assert detection.hand_corroborated is True
    assert detection.reliability == "medium"
    # Never "high": two weak signals agreeing is still not proof.
    assert detection.reliability != "high"


def test_ear_anchored_detection_outranks_a_more_confident_stray_box():
    """Confidence in a box nowhere near an ear is confidence in the wrong thing."""
    result = WearableAnalysisResult(
        detections=[
            WearableDetection("earbuds", "earbud", EventType.EARBUDS_SUSPECTED, 0.80, (0, 0, 5, 5)),
            WearableDetection(
                "earbuds",
                "earbud",
                EventType.EARBUDS_SUSPECTED,
                0.40,
                (300, 200, 320, 220),
                ear_anchored=True,
            ),
        ]
    )
    best = WearableDetector._deduplicate(result).detections
    assert len(best) == 1
    assert best[0].ear_anchored is True


# ---------------------------------------------------------------------------
# Earbud / headphone persistence
# ---------------------------------------------------------------------------


def _sweep(event_type=EventType.EARBUDS_SUSPECTED, target="earbuds", confidence=0.33):
    """A fresh sweep object — identity is what marks it as a new detection pass."""
    return WearableAnalysisResult(
        ran=True,
        detections=[
            WearableDetection(
                target, "earbud", event_type, confidence, (300, 200, 320, 220), ear_anchored=True
            )
        ],
    )


def test_a_single_sweep_never_qualifies_a_worn_device():
    observer = _observer()
    events = observer.map_to_events(_observation(facial_dynamics=_dynamics(), wearables=_sweep()))
    assert EventType.EARBUDS_SUSPECTED not in events


def test_agreeing_sweeps_confirm_a_worn_device():
    observer = _observer()
    for _ in range(STRICT.wearable_confirmation_sweeps):
        events = observer.map_to_events(
            _observation(facial_dynamics=_dynamics(), wearables=_sweep())
        )
    assert EventType.EARBUDS_SUSPECTED in events


def test_hand_at_ear_lowers_the_sweeps_required_for_an_earpiece():
    """A corroborating gesture buys one sweep, and only for the earpiece case."""
    hands = HandAnalysisResult(hands_detected=1, hands_visible=True, hand_near_ear=True)
    observer = _observer()
    events = observer.map_to_events(
        _observation(facial_dynamics=_dynamics(), wearables=_sweep(), hand_analysis=hands)
    )
    assert EventType.EARBUDS_SUSPECTED in events

    # Headphones get no such discount: the gesture says nothing about over-ear cups.
    observer = _observer()
    events = observer.map_to_events(
        _observation(
            facial_dynamics=_dynamics(),
            wearables=_sweep(EventType.HEADPHONES_DETECTED, "headphones", 0.5),
            hand_analysis=hands,
        )
    )
    assert EventType.HEADPHONES_DETECTED not in events


def test_confirmation_lapses_when_sweeps_stop_agreeing():
    """A device that is no longer detected stops being asserted."""
    observer = _observer()
    for _ in range(3):
        observer.map_to_events(_observation(facial_dynamics=_dynamics(), wearables=_sweep()))
    for _ in range(3):
        events = observer.map_to_events(
            _observation(facial_dynamics=_dynamics(), wearables=WearableAnalysisResult(ran=True))
        )
    assert EventType.EARBUDS_SUSPECTED not in events


# ---------------------------------------------------------------------------
# Speech-like mouth activity
# ---------------------------------------------------------------------------


def test_sustained_talking_is_detected():
    analyzer = _speech_analyzer()
    result = _feed_speech(analyzer, _articulation(fps=6.0, seconds=8))
    assert result.is_speaking is True
    assert result.speech_activity is not None and result.speech_activity > 0.5


def test_quiet_speech_with_little_jaw_movement_is_still_detected():
    """Mumbling moves the jaw far less than clear speech and must not fall through."""
    analyzer = _speech_analyzer()
    result = _feed_speech(
        analyzer, _articulation(fps=6.0, seconds=8, level=0.15, swing=0.12, hz=4.3)
    )
    assert result.is_speaking is True


@pytest.mark.parametrize(
    "name, series",
    [
        ("still and closed", np.full(48, 0.03)),
        ("resting open mouth", np.full(48, 0.35)),
        ("silent reading", 0.06 + 0.02 * np.sin(np.linspace(0, 6, 48))),
        ("smiling", 0.10 + 0.05 * np.sin(np.linspace(0, 2, 48))),
        (
            "one brief mouth movement",
            np.r_[np.full(18, 0.03), [0.45, 0.55, 0.5, 0.1], np.full(26, 0.03)],
        ),
    ],
)
def test_innocent_mouth_movements_are_not_reported_as_talking(name, series):
    analyzer = _speech_analyzer()
    assert _feed_speech(analyzer, series).is_speaking is False, name


def test_a_yawn_is_not_talking():
    """A yawn is large but crosses its own mid-level twice; speech crosses repeatedly."""
    fps = 6.0
    t = np.arange(int(fps * 8)) / fps
    yawn = np.full(t.size, 0.05)
    inside = (t > 2) & (t < 6)
    yawn[inside] = 0.05 + 0.85 * np.sin(np.pi * (t[inside] - 2) / 4)

    analyzer = _speech_analyzer()
    verdicts = []
    landmarks = np.zeros((1, 2), np.float32)
    for value in yawn:
        result = FacialDynamicsResult()
        analyzer._measure_speech(result, landmarks, {"jawOpen": float(value)})
        verdicts.append(result.is_speaking)
    assert not any(v is True for v in verdicts), "a yawn must never register as talking"


def test_speech_is_reported_unmeasurable_below_the_sampling_floor():
    """Sampled too slowly the signal is aliased; that is *not measured*, not silence.

    At 4 fps a 3-5 Hz articulation folds into a slow wave shaped like a yawn. The
    analyzer must decline to answer rather than return a false negative that reads,
    in an evidence package, as a candidate who was observed and found silent.
    """
    analyzer = _speech_analyzer(fps=4.0)
    result = _feed_speech(analyzer, _articulation(fps=4.0, seconds=8))
    assert result.is_speaking is None
    assert result.speech_measurable is False


def test_sessions_that_report_speaking_sample_fast_enough_to_measure_it():
    strict = SessionConfig(strictness=StrictnessLevel.STRICT)
    assert strict.sampling_fps >= STRICT.speech_min_sampling_fps
    # The idle rate matters too: a quiet candidate is the one about to start talking.
    assert strict.idle_fps >= STRICT.speech_min_sampling_fps
    assert strict.speech_sampling_applied is True

    # An exam that does not report speaking pays nothing for this.
    standard = SessionConfig(strictness=StrictnessLevel.STANDARD)
    assert standard.sampling_fps == 4.0
    assert standard.speech_sampling_applied is False


def test_speaking_observation_does_not_claim_speech_occurred():
    """There is no audio. The wording a proctor reads must not imply otherwise."""
    observer = _observer()
    events = observer.map_to_events(
        _observation(facial_dynamics=_dynamics(is_speaking=True, speech_activity=0.8))
    )
    description = events[EventType.CANDIDATE_SPEAKING]["description"].lower()
    assert "possible talking" in description
    assert "no audio" in description


# ---------------------------------------------------------------------------
# Head pose persistence and movement patterns
# ---------------------------------------------------------------------------


def test_a_single_brief_glance_is_not_a_pattern():
    pattern = _feed_pose([(3.0, 0, 0), (0.3, 45, 0), (3.0, 0, 0)])
    assert pattern.episode_count == 0
    assert pattern.repeated_look_away is False
    assert pattern.is_pattern is False


def test_a_sustained_turn_is_measured_but_is_not_a_repetition():
    pattern = _feed_pose([(2.0, 0, 0), (6.0, 45, 0)])
    assert pattern.deviating is True
    assert pattern.sustained_seconds == pytest.approx(6.0, abs=0.4)
    assert pattern.dominant_direction == "LEFT"
    assert pattern.repeated_look_away is False


def test_repeated_look_away_episodes_form_a_pattern():
    segments = [(2.0, 0, 0)]
    for _ in range(STRICT.head_repeat_episode_count):
        segments += [(1.0, 45, 0), (1.5, 0, 0)]
    pattern = _feed_pose(segments)
    assert pattern.episode_count >= STRICT.head_repeat_episode_count
    assert pattern.repeated_look_away is True
    assert pattern.is_pattern is True


def test_rapid_direction_changes_are_counted_across_the_pause_between_them():
    """A head snapping left, pausing, then snapping right has reversed direction.

    Comparing raw consecutive velocities missed this: the still frames between the
    two movements read as fast-slow-slow-fast, with no adjacent pair opposed.
    """
    segments = [(1.0, 0, 0)]
    for _ in range(4):
        segments += [(0.6, 50, 0), (0.6, -50, 0)]
    pattern = _feed_pose(segments)
    assert pattern.rapid_reversals >= STRICT.head_min_reversals
    assert pattern.rapid_repeated_movement is True


def test_slow_postural_sway_is_not_rapid_movement():
    fps = 6.0
    tracker = _tracker()
    timestamp, pattern = 0.0, None
    for _ in range(120):
        yaw = 10.0 * np.sin(2 * np.pi * 0.2 * timestamp)
        pattern = tracker.update(timestamp, yaw, 0.0, STRICT.is_pose_deviated(yaw=yaw, pitch=0.0))
        timestamp += 1.0 / fps
    assert pattern.rapid_reversals == 0
    assert pattern.rapid_repeated_movement is False


def test_the_window_forgets_older_movement():
    """Episodes outside the pattern window must stop counting, or every long session
    eventually reports one."""
    segments = [(1.0, 45, 0), (1.0, 0, 0)] * 3
    segments += [(STRICT.head_pattern_window_seconds + 2.0, 0, 0)]
    pattern = _feed_pose(segments)
    assert pattern.episode_count == 0
    assert pattern.repeated_look_away is False


def test_a_backwards_clock_restarts_the_window():
    tracker = _tracker()
    for index in range(10):
        tracker.update(10.0 + index * 0.2, 45.0, 0.0, True)
    pattern = tracker.update(0.0, 0.0, 0.0, False)
    assert pattern.episode_count == 0
    assert pattern.sustained_seconds == 0.0


def test_repeated_head_movement_is_reported_as_its_own_observation():
    """SUSPICIOUS_HEAD_POSE existed in the schema but nothing ever emitted it."""
    observer = _observer()
    events = {}
    timestamp = 0.0
    for _ in range(4):
        for yaw, hold in ((45.0, 1.0), (0.0, 1.5)):
            for _ in range(int(hold * 6)):
                events = observer.map_to_events(
                    _observation(
                        timestamp_seconds=timestamp,
                        facial_dynamics=_dynamics(head_pose=HeadPose(yaw=yaw, pitch=0.0)),
                    )
                )
                timestamp += 1.0 / 6.0
    assert EventType.SUSPICIOUS_HEAD_POSE in events
    assert "Repeated head movement" in events[EventType.SUSPICIOUS_HEAD_POSE]["description"]


# ---------------------------------------------------------------------------
# Calibration, gaze correlation and exam mode
# ---------------------------------------------------------------------------


def test_head_pose_is_judged_against_the_calibrated_baseline():
    """An off-centre camera must not read as a candidate permanently turned away.

    The reporting layer previously re-evaluated pose against a hard-coded zero,
    discarding the baseline the analyzer had measured — so calibration corrected
    nothing a proctor ever saw.
    """
    observer = _observer()
    offset = STRICT.yaw_limit_degrees + 5.0

    uncalibrated = observer.map_to_events(
        _observation(facial_dynamics=_dynamics(head_pose=HeadPose(yaw=offset)))
    )
    assert EventType.LOOKING_AWAY in uncalibrated

    observer.reset()
    calibrated = observer.map_to_events(
        _observation(
            facial_dynamics=_dynamics(
                head_pose=HeadPose(yaw=offset), calibrated=True, baseline_yaw=offset
            )
        )
    )
    assert EventType.LOOKING_AWAY not in calibrated


def test_gaze_agreeing_with_the_head_turn_raises_the_recorded_confidence():
    observer = _observer()
    pose = HeadPose(yaw=STRICT.yaw_limit_degrees + 10.0)

    alone = observer.map_to_events(_observation(facial_dynamics=_dynamics(head_pose=pose)))
    observer.reset()
    agreeing = observer.map_to_events(
        _observation(
            facial_dynamics=_dynamics(head_pose=pose),
            gaze=GazeObservation(
                horizontal=0.5, vertical=0.0, direction=GazeDirection.LEFT, confidence=0.9
            ),
        )
    )
    assert (
        agreeing[EventType.LOOKING_AWAY]["confidence"] > alone[EventType.LOOKING_AWAY]["confidence"]
    )
    assert "gaze also directed" in agreeing[EventType.LOOKING_AWAY]["description"]


def test_gaze_pointing_the_other_way_does_not_corroborate():
    """Eyes turned opposite the head is not two signals agreeing."""
    observer = _observer()
    pose = HeadPose(yaw=STRICT.yaw_limit_degrees + 10.0)
    events = observer.map_to_events(
        _observation(
            facial_dynamics=_dynamics(head_pose=pose),
            gaze=GazeObservation(
                horizontal=-0.5, vertical=0.0, direction=GazeDirection.RIGHT, confidence=0.9
            ),
        )
    )
    assert "gaze also directed" not in events[EventType.LOOKING_AWAY]["description"]


def test_looking_down_at_paper_is_not_a_pattern_in_a_paper_exam():
    paper = ExamPolicy.for_level(StrictnessLevel.STRICT, ExamMode.PHYSICAL_PAPER)
    pattern = _feed_pose([(2.0, 0, 0)] + [(1.0, 0, -40), (1.5, 0, 0)] * 4, policy=paper)
    assert pattern.episode_count == 0
    assert pattern.repeated_look_away is False

    digital = _feed_pose([(2.0, 0, 0)] + [(1.0, 0, -40), (1.5, 0, 0)] * 4, policy=STRICT)
    assert digital.repeated_look_away is True, "the same movement is reportable on screen"


def test_paper_mode_still_reports_sustained_lateral_turns():
    paper = ExamPolicy.for_level(StrictnessLevel.STRICT, ExamMode.PHYSICAL_PAPER)
    pattern = _feed_pose([(2.0, 0, 0), (6.0, 60, 0)], policy=paper)
    assert pattern.deviating is True
    assert pattern.sustained_seconds > 3.0


# ---------------------------------------------------------------------------
# Temporal qualification and evidence
# ---------------------------------------------------------------------------


def _qualify(event_type, detail, seconds, policy=STRICT, fps=6.0):
    """Run one behaviour through the shared aggregator for a given duration."""
    aggregator = UnifiedTemporalAggregator(
        session_id="test",
        absence_tolerance_seconds=policy.absence_tolerance_seconds,
        min_event_duration_seconds=policy.min_event_duration_seconds,
        min_duration_overrides=policy.event_min_duration,
    )
    for index in range(int(seconds * fps) + 1):
        aggregator.update_behaviour_observations(
            {event_type: detail}, timestamp=index / fps, frame_index=index, detector=DETECTOR
        )
    return aggregator, aggregator.active_incidents[f"behaviour_{event_type.value}"]


def test_a_short_head_pattern_is_recorded_but_not_qualified():
    _, incident = _qualify(
        EventType.SUSPICIOUS_HEAD_POSE, {"confidence": 0.7, "bbox": (10, 10, 50, 50)}, seconds=0.4
    )
    assert incident.status is not EventStatus.QUALIFIED


def test_a_persistent_head_pattern_qualifies_through_the_shared_aggregator():
    required = STRICT.duration_for(EventType.SUSPICIOUS_HEAD_POSE)
    _, incident = _qualify(
        EventType.SUSPICIOUS_HEAD_POSE,
        {"confidence": 0.7, "bbox": (10, 10, 50, 50)},
        seconds=required + 1.0,
    )
    assert incident.status is EventStatus.QUALIFIED


def test_speech_qualifies_only_after_the_configured_duration():
    required = STRICT.duration_for(EventType.CANDIDATE_SPEAKING)
    _, short = _qualify(
        EventType.CANDIDATE_SPEAKING, {"confidence": 0.8, "bbox": (1, 2, 3, 4)}, required - 0.6
    )
    assert short.status is not EventStatus.QUALIFIED
    _, long = _qualify(
        EventType.CANDIDATE_SPEAKING, {"confidence": 0.8, "bbox": (1, 2, 3, 4)}, required + 1.0
    )
    assert long.status is EventStatus.QUALIFIED


@pytest.mark.parametrize(
    "event_type, dynamics_kwargs, expected_region",
    [
        (
            EventType.CANDIDATE_SPEAKING,
            {"is_speaking": True, "speech_activity": 0.8},
            "mouth_region",
        ),
        (
            EventType.LOOKING_AWAY,
            {"head_pose": HeadPose(yaw=STRICT.yaw_limit_degrees + 10.0)},
            "face_bbox",
        ),
    ],
)
def test_each_behaviour_carries_the_region_its_evidence_should_show(
    event_type, dynamics_kwargs, expected_region
):
    """The evidence stage crops what the detail dict points at.

    Speaking must point at the mouth and a head turn at the face, or the snapshot a
    proctor opens shows the wrong part of the frame.
    """
    observer = _observer()
    dynamics = _dynamics(mouth_region=(140, 180, 170, 200), **dynamics_kwargs)
    events = observer.map_to_events(_observation(facial_dynamics=dynamics))
    assert events[event_type]["bbox"] == getattr(dynamics, expected_region)


def test_a_confirmed_earpiece_points_evidence_at_the_detected_device():
    observer = _observer()
    for _ in range(STRICT.wearable_confirmation_sweeps):
        events = observer.map_to_events(
            _observation(facial_dynamics=_dynamics(), wearables=_sweep())
        )
    assert events[EventType.EARBUDS_SUSPECTED]["bbox"] == (300, 200, 320, 220)


def test_the_representative_bbox_survives_into_the_incident():
    _, incident = _qualify(
        EventType.EARBUDS_SUSPECTED,
        {"confidence": 0.4, "bbox": (300, 200, 320, 220)},
        seconds=STRICT.duration_for(EventType.EARBUDS_SUSPECTED) + 1.0,
    )
    assert incident.representative_bbox == (300, 200, 320, 220)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_no_observation_carries_a_score_of_any_kind():
    """These are descriptive observations. Nothing here ranks or grades anybody."""
    observer = _observer()
    events = observer.map_to_events(
        _observation(
            facial_dynamics=_dynamics(
                is_speaking=True,
                speech_activity=0.9,
                head_pose=HeadPose(yaw=60.0),
            )
        )
    )
    forbidden = ("risk", "suspicion", "probability", "cheat", "score")
    for detail in events.values():
        text = str(detail.get("description", "")).lower()
        assert not any(word in text for word in forbidden), text
        assert set(detail) <= {"confidence", "bbox", "description", "object_class"}


def test_documented_strictness_table_matches_the_policy():
    """The accuracy guide states these counts and limits; drift makes it a lie."""
    documented = {
        StrictnessLevel.STANDARD: (12, 40.0),
        StrictnessLevel.STRICT: (20, 28.0),
        StrictnessLevel.MAXIMUM: (23, 24.0),
    }
    for level, (count, yaw) in documented.items():
        policy = ExamPolicy.for_level(level)
        assert len(policy.enabled_events) == count, level
        assert policy.yaw_limit_degrees == yaw, level


def test_strictness_levels_stay_nested():
    """Each level must report everything the level below it does."""
    standard = ExamPolicy.for_level(StrictnessLevel.STANDARD).enabled_events
    strict = ExamPolicy.for_level(StrictnessLevel.STRICT).enabled_events
    maximum = ExamPolicy.for_level(StrictnessLevel.MAXIMUM).enabled_events
    assert standard < strict < maximum
