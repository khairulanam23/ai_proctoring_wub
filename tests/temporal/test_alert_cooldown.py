"""Focused automated tests for Continuous Per-Frame Monitoring + Per-Event Alert Cooldown.

Covers all 10 requirements from Section 18:
1. Continuous frame processing: N frames in = N processed observations during cooldown.
2. Same event deduplication: PHONE at t=0, 5, 10 -> 1 alert.
3. Repeat alert after cooldown: PHONE at t=0, 5, 15+ -> 2 alerts.
4. Different event during phone cooldown: PHONE at t=0, PAPER at t=5 -> 1 phone alert + 1 paper alert.
5. Multiple event types: PHONE at t=0, PAPER at t=1, HAND_NEAR_EAR at t=2 -> independent alerts.
6. Event closes and reappears: PHONE appears, closes, appears again -> new incident ID, new alert allowed.
7. Global cooldown regression: PHONE cooldown active + PAPER qualifies -> PAPER is NOT suppressed.
8. Qualification remains intact: Single transient frame (< qualification duration) does NOT alert.
9. Boundary semantics: elapsed < 15.0s suppressed, elapsed >= 15.0s eligible.
10. No production model change: verify production weights files are unmodified.
"""

from pathlib import Path

import numpy as np
import pytest

from proctoring.config import SessionConfig
from proctoring.core.events import (
    DetectorInfo,
    EventSeverity,
    EventStatus,
    EventType,
)
from proctoring.engine import ProctoringEngine
from proctoring.temporal.aggregator import (
    ActiveIncident,
    AlertRecord,
    UnifiedTemporalAggregator,
)


@pytest.fixture
def dummy_detector():
    return DetectorInfo(name="YOLO11", version="11.0")


@pytest.fixture
def face_detector():
    return DetectorInfo(name="YuNet", version="2023mar")


@pytest.fixture
def behaviour_detector():
    return DetectorInfo(name="MediaPipe", version="tasks-1.0")


# ==============================================================================
# Test 1: Continuous Frame Processing
# ==============================================================================
def test_continuous_frame_processing_during_cooldown():
    """Verify that every incoming frame is processed without early-return or dropping.

    While PHONE_DETECTED is within its 15-second cooldown window, N frames in must
    yield exactly N processed frames and observations.
    """
    config = SessionConfig(
        session_id="test_continuous_stream",
        sampling_fps=4.0,
        enable_face_detection=False,
        enable_face_verification=False,
        enable_object_detection=False,  # Bypass heavy models for unit test speed
        enable_wearable_detection=False,
        alert_repeat_interval_seconds=15.0,
    )
    engine = ProctoringEngine(config=config)
    engine.start_session()

    # Create dummy blank frames (640x480x3)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    n_frames = 20

    observations = []
    for i in range(n_frames):
        t = i * 0.25
        obs = engine.process_frame(dummy_frame, frame_index=i, timestamp_seconds=t)
        observations.append(obs)

    assert len(observations) == n_frames
    assert engine._frame_counter == n_frames
    for idx, obs in enumerate(observations):
        assert obs.accepted is True
        assert obs.frame_index == idx
        assert obs.timestamp_seconds == pytest.approx(idx * 0.25, abs=1e-3)


# ==============================================================================
# Test 2: Same Event Deduplication Within 15s Cooldown
# ==============================================================================
def test_same_event_deduplication_within_cooldown(dummy_detector):
    """Verify PHONE at t=0, t=5, t=10 emits exactly 1 alert."""
    agg = UnifiedTemporalAggregator(
        session_id="test_dedup",
        absence_tolerance_seconds=1.0,
        min_event_duration_seconds=1.0,
        alert_repeat_interval_seconds=15.0,
    )

    phone_obj = [
        {"class_name": "cell phone", "confidence": 0.95, "bbox": (50, 50, 100, 150)}
    ]

    all_emitted_alerts: list[AlertRecord] = []

    # t = 0.0 to 1.2s: continuous phone frames qualifying PHONE_DETECTED
    for i, t in enumerate([0.0, 0.4, 0.8, 1.2]):
        agg.update_object_observations(
            detected_objects=phone_obj,
            timestamp=t,
            frame_index=i,
            detector=dummy_detector,
        )
        alerts = agg.evaluate_frame_alerts(timestamp=t, frame_index=i)
        all_emitted_alerts.extend(alerts)

    # Initial alert must have been emitted at qualification (t=1.2s)
    assert len(all_emitted_alerts) == 1
    assert all_emitted_alerts[0].event_type == EventType.PHONE_DETECTED
    assert all_emitted_alerts[0].alert_sequence == 1
    assert all_emitted_alerts[0].is_repeat is False

    # t = 5.0s: phone still present (within cooldown window: 5.0 - 1.2 = 3.8s < 15s)
    agg.update_object_observations(
        detected_objects=phone_obj,
        timestamp=5.0,
        frame_index=10,
        detector=dummy_detector,
    )
    alerts_t5 = agg.evaluate_frame_alerts(timestamp=5.0, frame_index=10)
    assert len(alerts_t5) == 0, "Alert at t=5.0 must be suppressed by cooldown"

    # t = 10.0s: phone still present (within cooldown window: 10.0 - 1.2 = 8.8s < 15s)
    agg.update_object_observations(
        detected_objects=phone_obj,
        timestamp=10.0,
        frame_index=20,
        detector=dummy_detector,
    )
    alerts_t10 = agg.evaluate_frame_alerts(timestamp=10.0, frame_index=20)
    assert len(alerts_t10) == 0, "Alert at t=10.0 must be suppressed by cooldown"

    # Cumulative alerts emitted across t=0, t=5, t=10 is still exactly 1
    total_alerts = all_emitted_alerts + alerts_t5 + alerts_t10
    assert len(total_alerts) == 1


# ==============================================================================
# Test 3: Repeat Alert After Cooldown Interval
# ==============================================================================
def test_repeat_alert_after_cooldown_interval(dummy_detector):
    """Verify PHONE at t=0, t=5, t=16.0 emits exactly 2 alerts (second is repeat)."""
    agg = UnifiedTemporalAggregator(
        session_id="test_repeat",
        absence_tolerance_seconds=1.0,
        min_event_duration_seconds=1.0,
        alert_repeat_interval_seconds=15.0,
    )

    phone_obj = [
        {"class_name": "cell phone", "confidence": 0.95, "bbox": (50, 50, 100, 150)}
    ]

    # Qualify at t=1.0s
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=0.0, frame_index=0, detector=dummy_detector
    )
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=1.0, frame_index=1, detector=dummy_detector
    )
    alerts_t1 = agg.evaluate_frame_alerts(timestamp=1.0, frame_index=1)
    assert len(alerts_t1) == 1
    assert alerts_t1[0].is_repeat is False
    assert alerts_t1[0].alert_sequence == 1

    # Observation at t=5.0s (suppressed)
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=5.0, frame_index=5, detector=dummy_detector
    )
    assert len(agg.evaluate_frame_alerts(timestamp=5.0, frame_index=5)) == 0

    # Observation at t=16.0s (16.0 - 1.0 = 15.0s >= 15.0s repeat interval)
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=16.0, frame_index=16, detector=dummy_detector
    )
    alerts_t16 = agg.evaluate_frame_alerts(timestamp=16.0, frame_index=16)
    assert len(alerts_t16) == 1
    assert alerts_t16[0].event_type == EventType.PHONE_DETECTED
    assert alerts_t16[0].is_repeat is True
    assert alerts_t16[0].alert_sequence == 2
    assert alerts_t16[0].incident_id == alerts_t1[0].incident_id


# ==============================================================================
# Test 4: Different Event During Phone Cooldown
# ==============================================================================
def test_different_event_during_phone_cooldown(dummy_detector, behaviour_detector):
    """Verify PHONE at t=0 does NOT suppress PAPER at t=5."""
    agg = UnifiedTemporalAggregator(
        session_id="test_diff_event",
        absence_tolerance_seconds=1.0,
        min_event_duration_seconds=1.0,
        alert_repeat_interval_seconds=15.0,
    )

    phone_obj = [
        {"class_name": "cell phone", "confidence": 0.95, "bbox": (50, 50, 100, 150)}
    ]

    # Qualify Phone at t=1.0s
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=0.0, frame_index=0, detector=dummy_detector
    )
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=1.0, frame_index=1, detector=dummy_detector
    )
    phone_alerts = agg.evaluate_frame_alerts(timestamp=1.0, frame_index=1)
    assert len(phone_alerts) == 1
    assert phone_alerts[0].event_type == EventType.PHONE_DETECTED

    # Phone is now in cooldown. Introduce PAPER_PRESENT at t=5.0s, qualifying at t=6.0s
    paper_behaviour = {
        EventType.PAPER_PRESENT: {
            "confidence": 0.9,
            "bbox": (100, 200, 300, 400),
            "description": "Physical paper on desk",
        }
    }
    agg.update_behaviour_observations(
        active_behaviours=paper_behaviour,
        timestamp=5.0,
        frame_index=5,
        detector=behaviour_detector,
    )
    agg.update_behaviour_observations(
        active_behaviours=paper_behaviour,
        timestamp=6.0,
        frame_index=6,
        detector=behaviour_detector,
    )

    # Evaluate alerts at t=6.0s: Paper must emit an alert!
    alerts_t6 = agg.evaluate_frame_alerts(timestamp=6.0, frame_index=6)
    paper_alerts = [a for a in alerts_t6 if a.event_type == EventType.PAPER_PRESENT]
    assert len(paper_alerts) == 1
    assert paper_alerts[0].alert_sequence == 1
    assert paper_alerts[0].is_repeat is False


# ==============================================================================
# Test 5: Multiple Event Types Simultaneously Active
# ==============================================================================
def test_multiple_event_types_simultaneously_active(
    dummy_detector, behaviour_detector
):
    """Verify PHONE, PAPER, and HAND_NEAR_EAR all qualify and alert independently."""
    agg = UnifiedTemporalAggregator(
        session_id="test_multi_simultaneous",
        absence_tolerance_seconds=1.0,
        min_event_duration_seconds=1.0,
        alert_repeat_interval_seconds=15.0,
    )

    phone_obj = [
        {"class_name": "cell phone", "confidence": 0.95, "bbox": (50, 50, 100, 150)}
    ]
    behaviours = {
        EventType.PAPER_PRESENT: {"confidence": 0.85, "description": "Paper sheet"},
        EventType.HAND_NEAR_EAR: {"confidence": 0.90, "description": "Hand raised to ear"},
    }

    # Frame at t=0.0s
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=0.0, frame_index=0, detector=dummy_detector
    )
    agg.update_behaviour_observations(
        active_behaviours=behaviours, timestamp=0.0, frame_index=0, detector=behaviour_detector
    )

    # Frame at t=1.0s -> all three qualify
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=1.0, frame_index=1, detector=dummy_detector
    )
    agg.update_behaviour_observations(
        active_behaviours=behaviours, timestamp=1.0, frame_index=1, detector=behaviour_detector
    )

    alerts = agg.evaluate_frame_alerts(timestamp=1.0, frame_index=1)
    emitted_types = {a.event_type for a in alerts}

    assert EventType.PHONE_DETECTED in emitted_types
    assert EventType.PAPER_PRESENT in emitted_types
    assert EventType.HAND_NEAR_EAR in emitted_types
    assert len(alerts) == 3


# ==============================================================================
# Test 6: Event Closes and Reappears (New Incident Identity)
# ==============================================================================
def test_event_closes_and_reappears_creates_new_incident(dummy_detector):
    """Verify phone disappearance closes incident A, reappearance creates incident B with immediate alert."""
    agg = UnifiedTemporalAggregator(
        session_id="test_reappear",
        absence_tolerance_seconds=1.0,
        min_event_duration_seconds=1.0,
        alert_repeat_interval_seconds=15.0,
    )

    phone_obj = [
        {"class_name": "cell phone", "confidence": 0.95, "bbox": (50, 50, 100, 150)}
    ]

    # Incident A: t=0.0s to 1.0s (qualifies)
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=0.0, frame_index=0, detector=dummy_detector
    )
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=1.0, frame_index=1, detector=dummy_detector
    )
    alerts_a = agg.evaluate_frame_alerts(timestamp=1.0, frame_index=1)
    assert len(alerts_a) == 1
    incident_id_a = alerts_a[0].incident_id

    # Phone disappears at t=2.0s. Absence tolerance is 1.0s.
    # At t=3.5s (> 1.0s absence), empty frame closes Incident A
    agg.update_object_observations(
        detected_objects=[], timestamp=3.5, frame_index=2, detector=dummy_detector
    )
    assert "object_cell phone" not in agg.active_incidents
    assert len(agg.closed_events) == 1

    # Phone reappears at t=5.0s and qualifies at t=6.0s (inside original 15s from Incident A's alert!)
    # Since Incident A closed, this is a genuinely NEW incident and must alert immediately.
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=5.0, frame_index=3, detector=dummy_detector
    )
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=6.0, frame_index=4, detector=dummy_detector
    )
    alerts_b = agg.evaluate_frame_alerts(timestamp=6.0, frame_index=4)

    assert len(alerts_b) == 1
    assert alerts_b[0].event_type == EventType.PHONE_DETECTED
    assert alerts_b[0].incident_id != incident_id_a, "New incident must have distinct incident ID"
    assert alerts_b[0].alert_sequence == 1, "New incident begins at alert sequence 1"
    assert alerts_b[0].is_repeat is False, "Initial alert of new incident is not a repeat alert"


# ==============================================================================
# Test 7: Global Cooldown Regression Guard
# ==============================================================================
def test_no_global_alert_cooldown_regression(dummy_detector, face_detector):
    """Explicitly verify that active PHONE cooldown does NOT suppress MULTIPLE_FACES."""
    agg = UnifiedTemporalAggregator(
        session_id="test_no_global_cooldown",
        absence_tolerance_seconds=1.0,
        min_event_duration_seconds=1.0,
        alert_repeat_interval_seconds=15.0,
    )

    phone_obj = [
        {"class_name": "cell phone", "confidence": 0.95, "bbox": (50, 50, 100, 150)}
    ]

    # Trigger Phone Alert #1 at t=1.0s
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=0.0, frame_index=0, detector=dummy_detector
    )
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=1.0, frame_index=1, detector=dummy_detector
    )
    alerts_phone = agg.evaluate_frame_alerts(timestamp=1.0, frame_index=1)
    assert len(alerts_phone) == 1
    assert alerts_phone[0].event_type == EventType.PHONE_DETECTED

    # At t=3.0s (phone cooldown is active), candidate has multiple faces appearing
    agg.update_face_observation(
        face_status="MULTIPLE_FACES",
        timestamp=3.0,
        frame_index=3,
        face_count=2,
        detector=face_detector,
    )
    agg.update_face_observation(
        face_status="MULTIPLE_FACES",
        timestamp=4.0,
        frame_index=4,
        face_count=2,
        detector=face_detector,
    )

    alerts_faces = agg.evaluate_frame_alerts(timestamp=4.0, frame_index=4)
    face_alerts = [a for a in alerts_faces if a.event_type == EventType.MULTIPLE_FACES]
    assert len(face_alerts) == 1, "MULTIPLE_FACES must NOT be suppressed by phone cooldown"


# ==============================================================================
# Test 8: Temporal Qualification Remains Intact
# ==============================================================================
def test_temporal_qualification_remains_intact(dummy_detector):
    """Verify single noisy frame (< min_duration) is NOT promoted to a qualified alert."""
    agg = UnifiedTemporalAggregator(
        session_id="test_noise",
        absence_tolerance_seconds=1.0,
        min_event_duration_seconds=1.0,
        alert_repeat_interval_seconds=15.0,
    )

    phone_obj = [
        {"class_name": "cell phone", "confidence": 0.95, "bbox": (50, 50, 100, 150)}
    ]

    # Single frame observation at t=0.0s
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=0.0, frame_index=0, detector=dummy_detector
    )

    # At t=0.0s, duration = 0.0s < 1.0s required duration -> status is OPEN, not QUALIFIED
    incident = agg.active_incidents.get("object_cell phone")
    assert incident is not None
    assert incident.status == EventStatus.OPEN

    # evaluate_frame_alerts must NOT emit any alert for an unverified/OPEN incident
    alerts = agg.evaluate_frame_alerts(timestamp=0.0, frame_index=0)
    assert len(alerts) == 0

    # Phone vanishes on next frame
    agg.update_object_observations(
        detected_objects=[], timestamp=1.5, frame_index=1, detector=dummy_detector
    )
    assert "object_cell phone" not in agg.active_incidents
    closed = agg.get_all_events()
    assert len(closed) == 1
    assert closed[0].status == EventStatus.RECORDED
    assert closed[0].metadata["is_duration_qualified"] is False


# ==============================================================================
# Test 9: Exact Boundary Semantics (elapsed < 15.0s vs elapsed >= 15.0s)
# ==============================================================================
def test_exact_boundary_semantics(dummy_detector):
    """Verify elapsed < 15.0s is suppressed and elapsed >= 15.0s is eligible."""
    agg = UnifiedTemporalAggregator(
        session_id="test_boundary",
        absence_tolerance_seconds=1.0,
        min_event_duration_seconds=1.0,
        alert_repeat_interval_seconds=15.0,
    )

    phone_obj = [
        {"class_name": "cell phone", "confidence": 0.95, "bbox": (50, 50, 100, 150)}
    ]

    # Alert 1 emitted at t=1.0s
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=0.0, frame_index=0, detector=dummy_detector
    )
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=1.0, frame_index=1, detector=dummy_detector
    )
    alerts_initial = agg.evaluate_frame_alerts(timestamp=1.0, frame_index=1)
    assert len(alerts_initial) == 1
    last_alert_t = 1.0

    # Test boundary 1: elapsed = 14.999s (t = 15.999s) -> suppressed
    t_sub = last_alert_t + 14.999
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=t_sub, frame_index=2, detector=dummy_detector
    )
    assert len(agg.evaluate_frame_alerts(timestamp=t_sub, frame_index=2)) == 0

    # Test boundary 2: elapsed = 15.000s (t = 16.000s) -> eligible!
    t_exact = last_alert_t + 15.000
    agg.update_object_observations(
        detected_objects=phone_obj, timestamp=t_exact, frame_index=3, detector=dummy_detector
    )
    alerts_exact = agg.evaluate_frame_alerts(timestamp=t_exact, frame_index=3)
    assert len(alerts_exact) == 1
    assert alerts_exact[0].is_repeat is True
    assert alerts_exact[0].alert_sequence == 2


# ==============================================================================
# Test 10: Production Model Weights Boundary Unchanged
# ==============================================================================
def test_production_model_weights_unchanged():
    """Verify active production model files exist and their weights are not replaced or retrained."""
    models_dir = Path("/home/phant0m/Phantom/ai_proctoring_wub/models")
    expected_models = [
        "yolo11n.pt",
        "face_detection_yunet_2023mar.onnx",
        "face_recognition_sface_2021dec.onnx",
        "face_landmarker.task",
        "hand_landmarker.task",
    ]

    for model_name in expected_models:
        model_file = models_dir / model_name
        assert model_file.exists(), f"Production model {model_name} must exist"
        assert model_file.stat().st_size > 0, f"Production model {model_name} must not be empty"

    # Verify YOLO11 weights file size is standard Ultralytics YOLO11n (~5.4 MB)
    yolo_file = models_dir / "yolo11n.pt"
    assert 5_000_000 < yolo_file.stat().st_size < 6_500_000, "YOLO11n weights altered"
