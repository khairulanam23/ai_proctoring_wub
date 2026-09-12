"""Targeted tests for Phase 3: Phone, Hand, Paper, and Writing behavioral analysis."""

import numpy as np
import pytest

from proctoring.analysis.hands import (
    HandAnalysisResult,
    HandAnalyzer,
    HandKinematics,
    HandObservation,
    HandState,
)
from proctoring.analysis.observer import BehaviourObserver
from proctoring.analysis.paper import PaperAnalysisResult, PaperDetector, PaperSheet, PaperState
from proctoring.analysis.phone_disambiguation import (
    PhoneClassification,
    PhoneDisambiguationResult,
    PhoneHandDisambiguator,
)
from proctoring.analysis.policy import ExamMode, ExamPolicy, StrictnessLevel
from proctoring.core.events import EventType
from proctoring.observation import FrameObservation


# ----------------------------------------------------------------------
# 1. Phone Disambiguation & Ambiguity Classification
# ----------------------------------------------------------------------

def test_phone_confirmed_with_temporal_persistence():
    """Verify that a rectangular phone slab confirmed over 2 frames is CONFIRMED_PHONE."""
    disambiguator = PhoneHandDisambiguator()

    phone_cand = {
        "class_name": "cell phone",
        "confidence": 0.78,
        "bbox": (100, 100, 160, 210),  # Width=60, Height=110, AR=1.83 (classic phone)
    }

    # Frame 1: Single frame -> POSSIBLE_PHONE
    res1 = disambiguator.disambiguate(phone_cand, frame_index=1)
    assert res1.classification == PhoneClassification.POSSIBLE_PHONE
    assert res1.temporal_confirmations == 1

    # Frame 2: Second frame -> CONFIRMED_PHONE
    res2 = disambiguator.disambiguate(phone_cand, frame_index=2)
    assert res2.classification == PhoneClassification.CONFIRMED_PHONE
    assert res2.temporal_confirmations == 2


def test_hand_false_positive_vs_object_ambiguity():
    """Verify distinction between empty hand false positive and hand-object ambiguity."""
    disambiguator = PhoneHandDisambiguator()

    # Synthetic open hand
    landmarks = np.zeros((21, 2), dtype=np.float32)
    # Wrist
    landmarks[0] = [150, 200]
    # MCPs
    landmarks[5] = [135, 165]
    landmarks[17] = [165, 165]
    # Splayed open fingertips
    landmarks[4] = [110, 150]
    landmarks[8] = [130, 110]
    landmarks[12] = [150, 100]
    landmarks[16] = [170, 110]
    landmarks[20] = [190, 130]

    hand = HandObservation(
        handedness="Right",
        confidence=0.95,
        bbox=(110, 100, 190, 200),
        centroid=(150, 150),
        landmarks=landmarks,
    )
    hand_res = HandAnalysisResult(hands_detected=1, hands=[hand])

    # Case A: Low confidence box completely matching empty hand -> HAND_FALSE_POSITIVE
    phone_low_conf = {
        "class_name": "cell phone",
        "confidence": 0.52,
        "bbox": (115, 105, 185, 195),
    }
    res_fp = disambiguator.disambiguate(phone_low_conf, hand_analysis=hand_res, frame_index=1)
    assert res_fp.classification == PhoneClassification.HAND_FALSE_POSITIVE

    # Case B: Moderate confidence box with partial overlap -> HAND_OBJECT_AMBIGUITY
    disambiguator.reset()
    phone_mod_conf = {
        "class_name": "cell phone",
        "confidence": 0.68,
        "bbox": (130, 130, 200, 210),
    }
    res_amb = disambiguator.disambiguate(phone_mod_conf, hand_analysis=hand_res, frame_index=1)
    assert res_amb.classification == PhoneClassification.HAND_OBJECT_AMBIGUITY


# ----------------------------------------------------------------------
# 2. Hand Kinematics, Writing, and Desk Workspace Motion
# ----------------------------------------------------------------------

def test_hand_writing_motion_mapped_to_events():
    """Verify that hand in writing state over paper maps to HAND_WRITING event."""
    policy = ExamPolicy.for_mode(ExamMode.PHYSICAL_PAPER, StrictnessLevel.STRICT)
    policy.smoothing_window_frames = 1
    observer = BehaviourObserver(policy=policy)

    obs = FrameObservation(frame_index=1, timestamp_seconds=0.25, iso_timestamp="2026-09-12T10:00:00Z")

    # Hand in writing state
    hand = HandObservation(
        handedness="Right",
        confidence=0.92,
        bbox=(200, 300, 260, 360),
        centroid=(230, 330),
        kinematics=HandKinematics(
            velocity=(1.2, 0.8),
            speed=1.4,
            writing_micro_oscillation=True,
            state=HandState.HAND_WRITING,
        ),
    )
    obs.hand_analysis = HandAnalysisResult(
        hands_detected=1,
        hands=[hand],
        hand_in_writing_area=True,
        writing_posture_detected=True,
        primary_hand_state=HandState.HAND_WRITING,
    )

    events = observer.map_to_events(obs)
    assert EventType.HAND_WRITING in events
    assert "Handwriting motion observed" in events[EventType.HAND_WRITING]["description"]


def test_hand_resting_vs_leaving_writing_area():
    """Verify mapping of stationary resting hand vs hand leaving writing area."""
    policy = ExamPolicy.for_mode(ExamMode.PHYSICAL_PAPER, StrictnessLevel.STRICT)
    policy.smoothing_window_frames = 1
    observer = BehaviourObserver(policy=policy)

    # Stationary hand
    obs_resting = FrameObservation(frame_index=1, timestamp_seconds=0.25, iso_timestamp="2026-09-12T10:00:00Z")
    hand_resting = HandObservation(
        handedness="Left",
        confidence=0.90,
        bbox=(150, 320, 210, 380),
        centroid=(180, 350),
        kinematics=HandKinematics(speed=2.0, state=HandState.HAND_RESTING),
    )
    obs_resting.hand_analysis = HandAnalysisResult(hands_detected=1, hands=[hand_resting])
    events_resting = observer.map_to_events(obs_resting)
    assert EventType.HAND_RESTING in events_resting

    # Hand leaving writing area
    obs_leaving = FrameObservation(frame_index=2, timestamp_seconds=0.50, iso_timestamp="2026-09-12T10:00:01Z")
    hand_leaving = HandObservation(
        handedness="Left",
        confidence=0.90,
        bbox=(150, 180, 210, 240),
        centroid=(180, 210),
        kinematics=HandKinematics(velocity=(0.0, -45.0), speed=45.0, state=HandState.HAND_LEAVING_WRITING_AREA),
    )
    obs_leaving.hand_analysis = HandAnalysisResult(hands_detected=1, hands=[hand_leaving])
    events_leaving = observer.map_to_events(obs_leaving)
    assert EventType.HAND_LEAVING_WRITING_AREA in events_leaving


# ----------------------------------------------------------------------
# 3. Paper Behavior & Manipulation
# ----------------------------------------------------------------------

def test_paper_manipulation_events():
    """Verify paper manipulation detection and event emission."""
    policy = ExamPolicy.for_mode(ExamMode.PHYSICAL_PAPER, StrictnessLevel.STRICT)
    policy.smoothing_window_frames = 1
    observer = BehaviourObserver(policy=policy)

    obs = FrameObservation(frame_index=1, timestamp_seconds=0.25, iso_timestamp="2026-09-12T10:00:00Z")
    sheet = PaperSheet(
        quadrilateral=[(100, 200), (250, 200), (250, 400), (100, 400)],
        bbox=(100, 200, 250, 400),
        centroid=(175, 300),
        area=30000.0,
        aspect_ratio=1.33,
        confidence=0.88,
    )
    obs.paper_analysis = PaperAnalysisResult(
        paper_present=True,
        paper_count=1,
        state=PaperState.LARGE_MOVEMENT,
        sheets=[sheet],
        displacement_px=185.0,
        manipulation_detected=True,
    )

    events = observer.map_to_events(obs)
    assert EventType.PAPER_PRESENT in events
    assert EventType.PAPER_MANIPULATED in events
    assert "185.0px shift" in events[EventType.PAPER_MANIPULATED]["description"]
