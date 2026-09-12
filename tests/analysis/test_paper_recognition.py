"""Unit tests for physical paper and answer-sheet recognition."""

import cv2
import numpy as np
import pytest

from proctoring.analysis.paper import PaperAnalysisResult, PaperDetector, PaperState


def test_paper_absent_on_blank_frame():
    detector = PaperDetector()
    blank = np.full((480, 640, 3), 40, dtype=np.uint8)  # uniform dark desk
    result = detector.detect(blank)
    assert result.paper_present is False
    assert result.paper_count == 0
    assert result.state == PaperState.ABSENT


def test_single_a4_paper_detection():
    detector = PaperDetector(desk_top_ratio=0.30)
    # Create 480x640 frame with desk at bottom
    frame = np.full((480, 640, 3), 50, dtype=np.uint8)

    # Draw white A4 sheet: width 150, height 212 -> aspect ratio 1.413
    # Placed on the desk area (y: 200 to 412, x: 240 to 390)
    cv2.rectangle(frame, (240, 200), (390, 412), (240, 240, 240), -1)

    result = detector.detect(frame, frame_index=1)
    assert result.paper_present is True
    assert result.paper_count == 1
    assert result.state == PaperState.PRESENT
    sheet = result.sheets[0]
    assert sheet.aspect_ratio == pytest.approx(1.413, abs=0.10)
    assert 235 <= sheet.centroid[0] <= 395


def test_paper_large_movement_detection():
    detector = PaperDetector(desk_top_ratio=0.30)
    # Frame 1: Paper on left side (x: 100 to 250)
    frame1 = np.full((480, 640, 3), 50, dtype=np.uint8)
    cv2.rectangle(frame1, (100, 200), (250, 412), (240, 240, 240), -1)
    res1 = detector.detect(frame1, frame_index=1)
    assert res1.state == PaperState.PRESENT

    # Frame 2: Paper moved to right side (x: 400 to 550) -> displacement 300px > 0.25 * 640
    frame2 = np.full((480, 640, 3), 50, dtype=np.uint8)
    cv2.rectangle(frame2, (400, 200), (550, 412), (240, 240, 240), -1)
    res2 = detector.detect(frame2, frame_index=2)
    assert res2.state == PaperState.LARGE_MOVEMENT
    assert res2.manipulation_detected is True
    assert res2.displacement_px > 160.0


def test_paper_observation_connection_to_observer():
    """Verify that PaperAnalysisResult reaches BehaviourObserver and produces reportable events."""
    from proctoring.analysis.observer import BehaviourObserver
    from proctoring.analysis.policy import ExamMode, ExamPolicy, StrictnessLevel
    from proctoring.core.events import EventType
    from proctoring.observation import FrameObservation

    policy = ExamPolicy.for_mode(ExamMode.PHYSICAL_PAPER, StrictnessLevel.STRICT)
    # Ensure window is 1 for immediate per-frame assertion
    policy.smoothing_window_frames = 1
    observer = BehaviourObserver(policy=policy)

    obs = FrameObservation(
        frame_index=1,
        timestamp_seconds=0.25,
        iso_timestamp="2026-09-12T10:00:00Z",
    )
    obs.paper_analysis = PaperAnalysisResult(
        paper_present=True,
        paper_count=1,
        state=PaperState.PRESENT,
        manipulation_detected=False,
    )

    events = observer.map_to_events(obs)
    assert EventType.PAPER_PRESENT in events
    assert "Physical paper present" in events[EventType.PAPER_PRESENT]["description"]

    # Test manipulation
    obs_manip = FrameObservation(
        frame_index=2,
        timestamp_seconds=0.50,
        iso_timestamp="2026-09-12T10:00:01Z",
    )
    obs_manip.paper_analysis = PaperAnalysisResult(
        paper_present=True,
        paper_count=1,
        state=PaperState.LARGE_MOVEMENT,
        displacement_px=220.0,
        manipulation_detected=True,
    )
    events_manip = observer.map_to_events(obs_manip)
    assert EventType.PAPER_MANIPULATED in events_manip

