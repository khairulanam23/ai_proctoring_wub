"""Targeted test suite for Phase 6: Audio Processing and Multimodal Behavioral Analysis."""

from pathlib import Path

import numpy as np
import pytest

from proctoring import ProctoringEngine, SessionConfig
from proctoring.analysis.facial_dynamics import FacialDynamicsResult
from proctoring.audio import (
    AudioAnalyzer,
    AudioChunk,
    AudioObservation,
    AudioStatus,
    MultimodalCorrelator,
    MultimodalObservation,
    MultimodalState,
)
from proctoring.core.events import EventCategory, EventType, category_for


# ----------------------------------------------------------------------
# 1. Audio Lifecycle & Degradation Diagnostics
# ----------------------------------------------------------------------

def test_audio_graceful_absence_when_none_or_empty():
    """Verify that absent audio stream or empty chunk results in DISABLED state without failure."""
    analyzer = AudioAnalyzer()

    # None chunk
    obs_none = analyzer.analyze(chunk=None, timestamp_seconds=1.0)
    assert obs_none.status == AudioStatus.DISABLED
    assert obs_none.speech_detected is False
    assert "No audio" in obs_none.technical_notes

    # Empty chunk
    obs_empty = analyzer.analyze(chunk=np.array([], dtype=np.float32), timestamp_seconds=2.0)
    assert obs_empty.status == AudioStatus.DISABLED


def test_audio_clipping_triggers_degraded_state():
    """Verify that clipped audio triggers explicit DEGRADED technical state."""
    analyzer = AudioAnalyzer(clipping_threshold_ratio=0.05)

    # Generate 1 second of heavily clipped audio at 16kHz
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Severe clipping: sine wave amplified by 10x and clipped to [-1.0, 1.0]
    clipped_wave = np.clip(10.0 * np.sin(2 * np.pi * 440.0 * t), -1.0, 1.0)

    chunk = AudioChunk(data=clipped_wave, sample_rate=sr, timestamp_seconds=0.5)
    obs = analyzer.analyze(chunk)

    assert obs.status == AudioStatus.DEGRADED
    assert obs.clipping_ratio > 0.50
    assert "clipping detected" in obs.technical_notes.lower()
    # Confirm AUDIO_DEGRADED is categorized as a technical diagnostic, never candidate misconduct
    assert category_for(EventType.AUDIO_DEGRADED) == EventCategory.TECHNICAL_DIAGNOSTIC


def test_voice_activity_detection_on_synthetic_speech():
    """Verify that harmonic speech tone in vocal range is detected as speech while silence is not."""
    analyzer = AudioAnalyzer(min_speech_db_above_noise=10.0)
    sr = 16000

    # 1. Silence / quiet ambient noise (-60 dB)
    quiet = np.random.randn(sr).astype(np.float32) * 0.001
    obs_quiet = analyzer.analyze(AudioChunk(data=quiet, sample_rate=sr, timestamp_seconds=0.0))
    assert obs_quiet.speech_detected is False

    # 2. Vocal fundamental with harmonics (300Hz fundamental + 600Hz + 900Hz harmonic)
    t = np.linspace(0, 1.0, sr, endpoint=False)
    vocal = (
        0.35 * np.sin(2 * np.pi * 300.0 * t)
        + 0.20 * np.sin(2 * np.pi * 600.0 * t)
        + 0.10 * np.sin(2 * np.pi * 900.0 * t)
    ).astype(np.float32)
    obs_speech = analyzer.analyze(AudioChunk(data=vocal, sample_rate=sr, timestamp_seconds=1.0))
    assert obs_speech.speech_detected is True
    assert obs_speech.speech_confidence > 0.5
    assert 280.0 <= obs_speech.dominant_frequency_hz <= 320.0


def test_multiple_speaker_discrimination():
    """Verify that bimodal pitch variations trigger multiple speaker detection."""
    analyzer = AudioAnalyzer()
    sr = 16000
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)

    # Feed 12 alternating voice bursts (Male ~150Hz vs Female ~350Hz)
    obs = None
    for i in range(12):
        freq = 150.0 if (i % 2 == 0) else 350.0
        wave = (0.35 * np.sin(2 * np.pi * freq * t) + 0.15 * np.sin(2 * np.pi * (2 * freq) * t)).astype(np.float32)
        obs = analyzer.analyze(AudioChunk(data=wave, sample_rate=sr, timestamp_seconds=i * 0.25))

    assert obs is not None
    assert obs.is_multiple_speakers is True
    assert obs.speaker_count_estimate >= 2
    assert "multiple speakers" in obs.technical_notes.lower()


# ----------------------------------------------------------------------
# 2. Multimodal Synchronization & Behavioral Alignment
# ----------------------------------------------------------------------

def test_multimodal_congruent_speech():
    """Verify simultaneous visual lip movement and acoustic speech -> CONGRUENT_SPEECH."""
    correlator = MultimodalCorrelator(temporal_tolerance_seconds=0.5)

    dynamics = FacialDynamicsResult(face_found=True, is_speaking=True)
    audio = AudioObservation(
        timestamp_seconds=1.0,
        status=AudioStatus.ACTIVE,
        speech_detected=True,
        speech_confidence=0.88,
    )

    mm_obs = correlator.correlate(
        timestamp_seconds=1.0,
        facial_dynamics=dynamics,
        audio_observation=audio,
        face_count=1,
    )
    assert mm_obs.state == MultimodalState.CONGRUENT_SPEECH
    assert mm_obs.visual_speech_detected is True
    assert mm_obs.acoustic_speech_detected is True

    events = correlator.map_to_events(mm_obs)
    assert EventType.MULTIMODAL_SPEECH_CONGRUENT in events


def test_multimodal_acoustic_speech_without_lip_movement():
    """Verify acoustic speech while lips are stationary -> ACOUSTIC_ONLY."""
    correlator = MultimodalCorrelator(temporal_tolerance_seconds=0.5)

    dynamics = FacialDynamicsResult(face_found=True, is_speaking=False)  # Candidate mouth closed
    audio = AudioObservation(
        timestamp_seconds=2.0,
        status=AudioStatus.ACTIVE,
        speech_detected=True,
        speech_confidence=0.82,
    )

    mm_obs = correlator.correlate(
        timestamp_seconds=2.0,
        facial_dynamics=dynamics,
        audio_observation=audio,
        face_count=1,
    )
    assert mm_obs.state == MultimodalState.ACOUSTIC_ONLY
    assert "potential secondary room participant" in mm_obs.human_review_notes

    events = correlator.map_to_events(mm_obs)
    assert EventType.ACOUSTIC_SPEECH_WITHOUT_LIP_MOVEMENT in events


def test_multimodal_silent_lip_movement():
    """Verify lip articulation without acoustic speech -> VISUAL_ONLY."""
    correlator = MultimodalCorrelator(temporal_tolerance_seconds=0.5)

    dynamics = FacialDynamicsResult(face_found=True, is_speaking=True)  # Candidate mouthing/whispering
    audio = AudioObservation(
        timestamp_seconds=3.0,
        status=AudioStatus.ACTIVE,
        speech_detected=False,
    )

    mm_obs = correlator.correlate(
        timestamp_seconds=3.0,
        facial_dynamics=dynamics,
        audio_observation=audio,
        face_count=1,
    )
    assert mm_obs.state == MultimodalState.VISUAL_ONLY
    assert "whispering or silent mouthing" in mm_obs.human_review_notes


# ----------------------------------------------------------------------
# 3. Session Isolation & Engine Lifecycle
# ----------------------------------------------------------------------

def test_audio_session_isolation_and_reset(tmp_path: Path):
    """Verify that audio and multimodal states do not leak across session reset."""
    analyzer = AudioAnalyzer()
    correlator = MultimodalCorrelator()

    # Session 1: Build history of speech
    sr = 16000
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    wave = (0.4 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)

    for i in range(5):
        analyzer.analyze(AudioChunk(data=wave, sample_rate=sr, timestamp_seconds=i * 0.5))
        correlator.correlate(
            timestamp_seconds=i * 0.5,
            facial_dynamics=FacialDynamicsResult(is_speaking=True),
            audio_observation=AudioObservation(timestamp_seconds=i * 0.5, status=AudioStatus.ACTIVE, speech_detected=True),
        )

    assert len(analyzer._recent_pitch_history) > 0
    assert len(correlator._visual_history) > 0

    # Reset session
    analyzer.reset()
    correlator.reset()

    assert len(analyzer._recent_pitch_history) == 0
    assert len(correlator._visual_history) == 0
    assert analyzer._noise_floor_db == -60.0

    # Session 2: Fresh empty observation
    obs2 = analyzer.analyze(chunk=None, timestamp_seconds=0.0)
    assert obs2.status == AudioStatus.DISABLED


def test_engine_process_frame_with_audio_chunk(tmp_path: Path):
    """Verify that ProctoringEngine processes audio_chunk and populates FrameObservation."""
    config = SessionConfig(session_id="test_audio_sess_01", output_dir=str(tmp_path))
    engine = ProctoringEngine(config=config)

    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    sr = 16000
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
    vocal = (0.35 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
    chunk = AudioChunk(data=vocal, sample_rate=sr, timestamp_seconds=0.25)

    obs = engine.process_frame(frame=frame, frame_index=1, timestamp_seconds=0.25, audio_chunk=chunk)

    assert obs.audio_observation is not None
    assert obs.audio_observation.status == AudioStatus.ACTIVE
    assert obs.multimodal_observation is not None
    assert "audio_observation" in obs.to_dict()
    assert "multimodal_observation" in obs.to_dict()

    # Graceful frame with no audio
    obs_no_audio = engine.process_frame(frame=frame, frame_index=2, timestamp_seconds=0.50, audio_chunk=None)
    assert obs_no_audio.audio_observation is not None
    assert obs_no_audio.audio_observation.status == AudioStatus.DISABLED

    engine.finalize_session()
