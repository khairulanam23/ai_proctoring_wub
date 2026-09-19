"""Focused tests for Phase 1 AI Proctoring Architecture Correction.

Verifies:
1. Temporal qualification: OPEN is not exposed, QUALIFIED is exposed, CLOSED is not exposed.
2. Phone disambiguation bypass: high raw confidence alone does not confirm without contextual evidence.
3. Identity semantics: NO_FACE, UNKNOWN_FACE, and FACE_MISMATCH remain distinguishable.
4. Evidence review metadata: qualified incident retains structured metadata for human review/relabeling.
"""

import numpy as np
import pytest

from proctoring.analysis.hands import HandAnalysisResult, HandObservation
from proctoring.analysis.phone_disambiguation import (
    PhoneClassification,
    PhoneHandDisambiguator,
)
from proctoring.core.events import (
    DetectorInfo,
    EventRecord,
    EventSeverity,
    EventStatus,
    EventType,
)
from proctoring.detection.object_detector import DetectedObject
from proctoring.observation import FaceStatus
from proctoring.temporal.aggregator import ActiveIncident, UnifiedTemporalAggregator


# ---------------------------------------------------------------------------
# Test 1: Temporal Qualification
# ---------------------------------------------------------------------------


def test_temporal_qualification_lifecycle():
    """Verify that only QUALIFIED incidents are exposed as active events; OPEN and CLOSED are not."""
    aggregator = UnifiedTemporalAggregator(
        session_id="test_temporal_qual",
        absence_tolerance_seconds=1.0,
        min_event_duration_seconds=1.0,
    )
    detector = DetectorInfo(name="test_detector", version="1.0")

    # Helper replicating ProctoringEngine active_event_types filtering
    def get_exposed_active_event_types(agg: UnifiedTemporalAggregator) -> list[str]:
        return sorted(
            {
                inc.event_type.value
                for inc in agg.active_incidents.values()
                if inc.status == EventStatus.QUALIFIED
            }
        )

    # 1. Single frame observation at t=0.0: Incident created with status OPEN
    # Face missing
    aggregator.update_face_observation(
        face_status=FaceStatus.NO_FACE,
        timestamp=0.0,
        frame_index=0,
        face_count=0,
        detector=detector,
    )

    inc = aggregator.active_incidents.get(f"face_{EventType.NO_FACE.value}")
    assert inc is not None
    assert inc.status == EventStatus.OPEN
    # OPEN incident MUST NOT be exposed as an active qualified event
    assert get_exposed_active_event_types(aggregator) == []

    # 2. Frame at t=0.4s: Incident ongoing with status ACTIVE (duration 0.4s < 1.0s)
    aggregator.update_face_observation(
        face_status=FaceStatus.NO_FACE,
        timestamp=0.4,
        frame_index=1,
        face_count=0,
        detector=detector,
    )
    assert inc.status == EventStatus.ACTIVE
    assert get_exposed_active_event_types(aggregator) == []

    # 3. Frame at t=1.2s: Duration is 1.2s >= 1.0s -> Incident becomes QUALIFIED
    aggregator.update_face_observation(
        face_status=FaceStatus.NO_FACE,
        timestamp=1.2,
        frame_index=2,
        face_count=0,
        detector=detector,
    )
    assert inc.status == EventStatus.QUALIFIED
    # QUALIFIED incident IS exposed as an active qualified event
    assert get_exposed_active_event_types(aggregator) == [EventType.NO_FACE.value]

    # 4. Condition resolves: face returns at t=2.5s (gap 2.5 - 1.2 = 1.3s > absence_tolerance 1.0s)
    aggregator.update_face_observation(
        face_status=FaceStatus.ENROLLED,
        timestamp=2.5,
        frame_index=3,
        face_count=1,
        detector=detector,
    )
    # The incident has been closed and removed from active_incidents
    assert f"face_{EventType.NO_FACE.value}" not in aggregator.active_incidents
    # CLOSED incident MUST NOT be exposed
    assert get_exposed_active_event_types(aggregator) == []

    # Finalized event is recorded in closed_events with QUALIFIED status
    assert len(aggregator.closed_events) == 1
    assert aggregator.closed_events[0].status == EventStatus.QUALIFIED
    assert aggregator.closed_events[0].event_type == EventType.NO_FACE


# ---------------------------------------------------------------------------
# Test 2: Phone Hand Disambiguator Bypass Correction
# ---------------------------------------------------------------------------


def test_high_raw_phone_confidence_alone_does_not_confirm():
    """Verify that high raw detector confidence alone cannot bypass contextual qualification."""
    disambiguator = PhoneHandDisambiguator(high_confidence_bypass=0.85)

    # Detection with very high confidence (0.95), vertical phone aspect ratio (2.0)
    phone = DetectedObject(
        class_id=67,
        class_name="cell phone",
        confidence=0.95,
        bbox=(200, 200, 260, 320),
    )

    # Frame 1 without hand analysis: raw confidence alone must NOT produce CONFIRMED_PHONE
    res1 = disambiguator.disambiguate(phone, hand_analysis=None, frame_index=1)
    assert res1.classification != PhoneClassification.CONFIRMED_PHONE
    assert res1.classification == PhoneClassification.POSSIBLE_PHONE
    assert "awaiting temporal confirmation" in res1.reason

    # Persistence across frames is required for confirmation without hand grip
    res2 = disambiguator.disambiguate(phone, hand_analysis=None, frame_index=2)
    assert res2.classification == PhoneClassification.CONFIRMED_PHONE
    assert "multi-frame persistence" in res2.reason

    # Verify that an empty open hand dismisses even a high-confidence phone detection
    landmarks = np.zeros((21, 2), dtype=np.float32)
    landmarks[0] = [230, 310]  # wrist
    landmarks[5] = [215, 260]  # index mcp
    landmarks[17] = [245, 260]  # pinky mcp
    # Splayed open fingertips
    landmarks[4] = [190, 230]
    landmarks[8] = [210, 190]
    landmarks[12] = [230, 185]
    landmarks[16] = [250, 195]
    landmarks[20] = [265, 210]

    hand = HandObservation(
        handedness="Right",
        confidence=0.95,
        bbox=(190, 180, 270, 320),
        centroid=(230, 250),
        landmarks=landmarks,
    )
    hand_res = HandAnalysisResult(hands=[hand], hands_detected=1, hands_visible=True)

    high_conf_candidate = DetectedObject(
        class_id=67,
        class_name="cell phone",
        confidence=0.90,
        bbox=(195, 185, 265, 315),
    )
    fresh_disambiguator = PhoneHandDisambiguator()
    res_hand = fresh_disambiguator.disambiguate(
        high_conf_candidate, hand_analysis=hand_res, frame_index=10
    )
    # Contextual empty hand analysis authoritatively dismisses false positive
    assert res_hand.classification == PhoneClassification.HAND_FALSE_POSITIVE


# ---------------------------------------------------------------------------
# Test 3: Clean Identity Semantics Separation
# ---------------------------------------------------------------------------


def test_identity_semantics_distinguishability():
    """Verify that NO_FACE, UNKNOWN_FACE, and FACE_MISMATCH remain cleanly distinguishable."""
    aggregator = UnifiedTemporalAggregator(session_id="test_identity_semantics")
    detector = DetectorInfo(name="face_test", version="1.0")

    # Case A: No face available -> NO_FACE
    aggregator.update_face_observation(
        face_status=FaceStatus.NO_FACE,
        timestamp=1.0,
        frame_index=1,
        face_count=0,
        detector=detector,
    )
    key_no_face = f"face_{EventType.NO_FACE.value}"
    assert key_no_face in aggregator.active_incidents
    assert aggregator.active_incidents[key_no_face].event_type == EventType.NO_FACE

    # Case B: Face detected but unverified / unknown -> UNKNOWN_FACE
    aggregator.update_face_observation(
        face_status=FaceStatus.UNKNOWN_FACE,
        timestamp=1.0,
        frame_index=1,
        face_count=1,
        detector=detector,
        similarity_score=None,
    )
    key_unknown = f"face_{EventType.UNKNOWN_FACE.value}"
    assert key_unknown in aggregator.active_incidents
    assert aggregator.active_incidents[key_unknown].event_type == EventType.UNKNOWN_FACE

    # Case C: Face recognized but does NOT match expected exam identity -> FACE_MISMATCH
    aggregator.update_face_observation(
        face_status=FaceStatus.IDENTITY_MISMATCH,
        timestamp=1.0,
        frame_index=1,
        face_count=1,
        detector=detector,
        similarity_score=0.15,
    )
    key_mismatch = f"face_{EventType.FACE_MISMATCH.value}"
    assert key_mismatch in aggregator.active_incidents
    assert aggregator.active_incidents[key_mismatch].event_type == EventType.FACE_MISMATCH

    # Verify that closing the incidents produces distinct EventRecord descriptions
    events = aggregator.flush()
    event_map = {e.event_type: e for e in events}

    assert EventType.NO_FACE in event_map
    assert "Face not detected" in event_map[EventType.NO_FACE].observation.description

    assert EventType.UNKNOWN_FACE in event_map
    assert "Unknown face detected" in event_map[EventType.UNKNOWN_FACE].observation.description

    assert EventType.FACE_MISMATCH in event_map
    assert "Face mismatch detected" in event_map[EventType.FACE_MISMATCH].observation.description
    assert "similarity: 0.15" in event_map[EventType.FACE_MISMATCH].observation.description


# ---------------------------------------------------------------------------
# Test 4: Evidence Review Metadata Preservation
# ---------------------------------------------------------------------------


def test_qualified_incident_retains_review_metadata():
    """Verify that a qualified incident retains structured metadata required for human review/relabeling."""
    aggregator = UnifiedTemporalAggregator(
        session_id="test_evidence_metadata",
        min_event_duration_seconds=0.5,
    )
    detector = DetectorInfo(name="yolo11n", version="8.0")

    # Sustained object detection
    aggregator.update_object_observations(
        detected_objects=[
            {"class_name": "cell phone", "confidence": 0.82, "bbox": (100, 120, 180, 260)}
        ],
        timestamp=0.0,
        frame_index=1,
        detector=detector,
    )
    aggregator.update_object_observations(
        detected_objects=[
            {"class_name": "cell phone", "confidence": 0.85, "bbox": (102, 122, 182, 262)}
        ],
        timestamp=0.6,
        frame_index=2,
        detector=detector,
    )

    events = aggregator.flush()
    assert len(events) >= 1
    event = events[0]

    assert event.status == EventStatus.QUALIFIED
    assert "review_metadata" in event.metadata

    review = event.metadata["review_metadata"]
    assert review["review_status"] == "pending"
    assert review["original_event_type"] in (EventType.PHONE_DETECTED.value, EventType.PHONE_CANDIDATE_UNCERTAIN.value)
    assert review["detector_name"] == "yolo11n"
    assert review["detector_version"] == "8.0"
    assert review["best_confidence"] >= 0.80

    # Ensure spatial and temporal observation details remain fully recoverable
    assert event.metadata["best_frame_index"] in (1, 2)
    assert event.metadata["best_timestamp"] in (0.0, 0.6)
    assert event.metadata["representative_bbox"] is not None
    assert event.observation.bounding_boxes is not None
    assert len(event.observation.frame_indices) == 2


# ---------------------------------------------------------------------------
# Test 5: Future Enrollment Boundary (Clean Template Injection)
# ---------------------------------------------------------------------------


def test_future_enrollment_boundary_template_injection(tmp_path):
    """Verify AI service accepts direct enrolled face representation without hardcoded identity."""
    import numpy as np
    from proctoring.integration.schemas import StartSessionRequest
    from proctoring.integration.service import ProctoringService

    service = ProctoringService(output_dir=str(tmp_path))

    # Mock 128-dimensional L2-normalized embedding representation
    mock_vector = list(np.random.randn(128).astype(np.float32))

    req = StartSessionRequest(
        exam_id="exam_boundary_test",
        candidate_id="generic_student_id",
        candidate_name="Generic Student",
        reference_templates=[mock_vector],
    )

    handle = service.start_session(req)
    assert handle.session_id is not None

    engine = service._engines.get(handle.session_id)
    assert engine is not None
    # Template was successfully received, parsed as float32 ndarray, and enabled
    assert len(engine.config.reference_templates) == 1
    assert engine.config.reference_templates[0].shape == (128,)
    assert engine.config.enable_face_verification is True

    service.finalize_session(handle.session_id)
