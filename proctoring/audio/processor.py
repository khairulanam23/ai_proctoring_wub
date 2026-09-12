"""Modular audio processing and Voice Activity Detection (VAD) pipeline.

Provides structured audio observation extraction, Voice Activity Detection,
multi-speaker voice estimation, clipping/degradation diagnostics, and clean
session isolation.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


class AudioStatus(str, Enum):
    """Explicit technical operating state of the audio intake pipeline."""

    ACTIVE = "ACTIVE"  # Audio stream delivering valid PCM audio chunks
    DEGRADED = "DEGRADED"  # Audio stream clipped, heavily distorted, or underrun
    FAILED = "FAILED"  # I/O error, device crash, or corrupt buffer
    DISABLED = "DISABLED"  # Graceful absence of audio hardware or audio disabled


@dataclass
class AudioChunk:
    """A discrete temporal chunk of uncompressed PCM audio data."""

    data: np.ndarray  # 1D float32 normalized [-1.0, 1.0] or int16
    sample_rate: int = 16000
    channels: int = 1
    timestamp_seconds: float = 0.0
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0.0 and self.sample_rate > 0 and len(self.data) > 0:
            self.duration_seconds = len(self.data) / float(self.sample_rate)


@dataclass
class AudioObservation:
    """Structured evidence record representing audio activity within a frame or temporal chunk."""

    timestamp_seconds: float
    status: AudioStatus
    rms_energy_db: float = -96.0
    speech_detected: bool = False
    speech_confidence: float = 0.0
    speaker_count_estimate: int = 0
    is_multiple_speakers: bool = False
    dominant_frequency_hz: float = 0.0
    clipping_ratio: float = 0.0
    noise_floor_db: float = -60.0
    technical_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "status": self.status.value,
            "rms_energy_db": round(self.rms_energy_db, 2),
            "speech_detected": self.speech_detected,
            "speech_confidence": round(self.speech_confidence, 4),
            "speaker_count_estimate": self.speaker_count_estimate,
            "is_multiple_speakers": self.is_multiple_speakers,
            "dominant_frequency_hz": round(self.dominant_frequency_hz, 1),
            "clipping_ratio": round(self.clipping_ratio, 4),
            "noise_floor_db": round(self.noise_floor_db, 2),
            "technical_notes": self.technical_notes,
        }


class AudioAnalyzer:
    """Modular audio observation engine operating on temporal PCM audio streams.

    Guarantees:
    - Never makes an automated cheating judgment.
    - Explicitly isolates technical failure states (DEGRADED, FAILED, DISABLED).
    - Session isolated via reset().
    """

    def __init__(
        self,
        min_speech_db_above_noise: float = 12.0,
        speech_freq_min_hz: float = 85.0,
        speech_freq_max_hz: float = 3400.0,
        clipping_threshold_ratio: float = 0.05,
    ) -> None:
        self.min_speech_db_above_noise = min_speech_db_above_noise
        self.speech_freq_min_hz = speech_freq_min_hz
        self.speech_freq_max_hz = speech_freq_max_hz
        self.clipping_threshold_ratio = clipping_threshold_ratio

        # Adaptive background noise tracking
        self._noise_floor_db: float = -60.0
        self._recent_pitch_history: deque[float] = deque(maxlen=20)
        self._status: AudioStatus = AudioStatus.ACTIVE

    def reset(self) -> None:
        """Reset internal history and noise estimation for strict session isolation."""
        self._noise_floor_db = -60.0
        self._recent_pitch_history.clear()
        self._status = AudioStatus.ACTIVE
        LOGGER.debug("AudioAnalyzer reset: session isolation preserved.")

    def analyze(
        self,
        chunk: AudioChunk | np.ndarray | None,
        timestamp_seconds: float = 0.0,
        sample_rate: int = 16000,
    ) -> AudioObservation:
        """Extract acoustic measurements, voice activity, and multi-speaker estimation."""
        if chunk is None:
            return AudioObservation(
                timestamp_seconds=timestamp_seconds,
                status=AudioStatus.DISABLED,
                technical_notes="No audio hardware or stream attached; graceful audio absence.",
            )

        # Normalize input to float32 1D array [-1.0, 1.0]
        if isinstance(chunk, AudioChunk):
            raw_data = chunk.data
            sr = chunk.sample_rate
            t = chunk.timestamp_seconds
        else:
            raw_data = chunk
            sr = sample_rate
            t = timestamp_seconds

        if raw_data is None or len(raw_data) == 0:
            return AudioObservation(
                timestamp_seconds=t,
                status=AudioStatus.DISABLED,
                technical_notes="Empty audio buffer received.",
            )

        # Convert int16 to float32 if needed
        samples = np.asarray(raw_data, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)  # Mono mixdown

        if np.issubdtype(raw_data.dtype, np.integer):
            samples = samples / 32768.0

        samples = np.clip(samples, -1.0, 1.0)
        n_samples = len(samples)

        if n_samples < 32:
            return AudioObservation(
                timestamp_seconds=t,
                status=AudioStatus.DEGRADED,
                technical_notes="Audio buffer too small for spectral analysis.",
            )

        # 1. Clipping detection
        clipping_samples = np.sum(np.abs(samples) >= 0.995)
        clipping_ratio = float(clipping_samples) / float(n_samples)
        status = AudioStatus.ACTIVE
        notes = ""

        if clipping_ratio > self.clipping_threshold_ratio:
            status = AudioStatus.DEGRADED
            notes = f"Audio clipping detected ({clipping_ratio * 100:.1f}% samples at max amplitude)."

        # 2. RMS Energy in dBFS
        rms = float(np.sqrt(np.mean(samples**2)))
        rms_db = 20.0 * math.log10(max(1e-6, rms))

        # Adaptive noise floor tracking (slow moving average of quiet frames)
        if rms_db < self._noise_floor_db + 6.0:
            self._noise_floor_db = 0.95 * self._noise_floor_db + 0.05 * rms_db
        else:
            self._noise_floor_db = min(-30.0, self._noise_floor_db + 0.005)

        # 3. Spectral Analysis via FFT
        fft_vals = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)

        total_energy = float(np.sum(fft_vals**2))
        speech_band_mask = (freqs >= self.speech_freq_min_hz) & (freqs <= self.speech_freq_max_hz)
        speech_energy = float(np.sum(fft_vals[speech_band_mask] ** 2))

        speech_energy_ratio = speech_energy / max(1e-9, total_energy)

        # Dominant frequency
        peak_idx = int(np.argmax(fft_vals))
        dom_freq = float(freqs[peak_idx])

        # 4. Voice Activity Detection (VAD)
        # Conditions: energy significantly above background noise, high speech band concentration, and human vocal fundamental (80 - 1000 Hz peak)
        db_above_noise = rms_db - self._noise_floor_db
        is_speech = False
        confidence = 0.0

        if db_above_noise >= self.min_speech_db_above_noise and speech_energy_ratio >= 0.45:
            # Check for harmonic vocal peak
            if 80.0 <= dom_freq <= 1200.0:
                is_speech = True
                confidence = min(1.0, 0.4 + 0.3 * min(1.0, db_above_noise / 25.0) + 0.3 * speech_energy_ratio)

        # 5. Multi-Speaker Detection
        # Pitch tracking: if speech is present, record dominant vocal pitch and check for multi-modal pitch distribution
        is_multi_speaker = False
        speaker_count = 1 if is_speech else 0

        if is_speech and dom_freq > 80.0:
            self._recent_pitch_history.append(dom_freq)
            if len(self._recent_pitch_history) >= 8:
                pitches = np.array(self._recent_pitch_history)
                # Check for bimodal pitch variance (e.g. Male ~120Hz vs Female ~220Hz or alternating speakers)
                pitch_std = float(np.std(pitches))
                pitch_range = float(np.ptp(pitches))
                if pitch_std > 35.0 and pitch_range > 80.0:
                    is_multi_speaker = True
                    speaker_count = 2
                    notes = notes + (" " if notes else "") + "Bimodal acoustic pitch variance suggests multiple speakers."

        return AudioObservation(
            timestamp_seconds=t,
            status=status,
            rms_energy_db=rms_db,
            speech_detected=is_speech,
            speech_confidence=confidence,
            speaker_count_estimate=speaker_count,
            is_multiple_speakers=is_multi_speaker,
            dominant_frequency_hz=dom_freq,
            clipping_ratio=clipping_ratio,
            noise_floor_db=self._noise_floor_db,
            technical_notes=notes,
        )
