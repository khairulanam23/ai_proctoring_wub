"""Multimodal correlation of acoustic voice activity and visual facial dynamics.

Provides synchronized cross-modal observation synthesis:
- Correlates acoustic speech with visual mouth articulation.
- Distinguishes congruent candidate speech from acoustic speech without lip movement
  (e.g., secondary speaker in room or room audio) and silent lip movement (whispering/mouthing).
- Correlates visual multi-person detection with acoustic multi-speaker estimation.
- Guarantees strict session isolation and evidence-only output for human review.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from proctoring.analysis.facial_dynamics import FacialDynamicsResult
from proctoring.audio.processor import AudioObservation, AudioStatus
from proctoring.core.events import EventType

LOGGER = logging.getLogger(__name__)


class MultimodalState(str, Enum):
    """Categorical alignment between acoustic and visual speech observations."""

    CONGRUENT_SPEECH = "CONGRUENT_SPEECH"  # Visual mouth moving AND acoustic speech detected
    ACOUSTIC_ONLY = "ACOUSTIC_ONLY"  # Acoustic speech detected, but mouth stationary (possible secondary speaker)
    VISUAL_ONLY = "VISUAL_ONLY"  # Lip articulation detected, but no acoustic speech detected (mouthing/whispering)
    SILENT_AND_STILL = "SILENT_AND_STILL"  # Neither acoustic speech nor lip articulation
    AUDIO_DISABLED = "AUDIO_DISABLED"  # Audio hardware/stream is unavailable or disabled


@dataclass
class MultimodalObservation:
    """Synchronized observation combining independent visual and acoustic sensor streams."""

    timestamp_seconds: float
    state: MultimodalState
    visual_speech_detected: bool
    acoustic_speech_detected: bool
    multiple_speakers_detected: bool
    multiple_faces_detected: bool
    correlation_confidence: float
    human_review_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "state": self.state.value,
            "visual_speech_detected": self.visual_speech_detected,
            "acoustic_speech_detected": self.acoustic_speech_detected,
            "multiple_speakers_detected": self.multiple_speakers_detected,
            "multiple_faces_detected": self.multiple_faces_detected,
            "correlation_confidence": round(self.correlation_confidence, 4),
            "human_review_notes": self.human_review_notes,
        }


class MultimodalCorrelator:
    """Synthesizes independent visual dynamics and acoustic observations over a temporal window."""

    def __init__(self, temporal_tolerance_seconds: float = 0.5) -> None:
        self.temporal_tolerance_seconds = temporal_tolerance_seconds
        self._visual_history: deque[tuple[float, bool]] = deque(maxlen=30)
        self._acoustic_history: deque[tuple[float, bool]] = deque(maxlen=30)

    def reset(self) -> None:
        """Clear temporal buffers for session isolation."""
        self._visual_history.clear()
        self._acoustic_history.clear()
        LOGGER.debug("MultimodalCorrelator reset: session isolation preserved.")

    def correlate(
        self,
        timestamp_seconds: float,
        facial_dynamics: FacialDynamicsResult | None,
        audio_observation: AudioObservation | None,
        face_count: int = 1,
    ) -> MultimodalObservation:
        """Cross-reference visual dynamics and acoustic observations."""
        # 1. Visual speaking determination
        visual_speech = bool(
            facial_dynamics and (
                getattr(facial_dynamics, "is_speaking", False)
                or getattr(facial_dynamics, "speaking", False)
            )
        )
        self._visual_history.append((timestamp_seconds, visual_speech))

        # 2. Acoustic speaking determination
        if audio_observation is None or audio_observation.status == AudioStatus.DISABLED:
            return MultimodalObservation(
                timestamp_seconds=timestamp_seconds,
                state=MultimodalState.AUDIO_DISABLED,
                visual_speech_detected=visual_speech,
                acoustic_speech_detected=False,
                multiple_speakers_detected=False,
                multiple_faces_detected=face_count >= 2,
                correlation_confidence=1.0 if visual_speech else 0.0,
                human_review_notes="Audio stream is disabled; observation relies entirely on computer vision.",
            )

        acoustic_speech = bool(audio_observation.speech_detected)
        self._acoustic_history.append((timestamp_seconds, acoustic_speech))

        # Check temporal window for near-synchronous activity (accounts for packet jitter)
        recent_visual = any(
            v for t, v in self._visual_history
            if abs(t - timestamp_seconds) <= self.temporal_tolerance_seconds
        )
        recent_acoustic = any(
            a for t, a in self._acoustic_history
            if abs(t - timestamp_seconds) <= self.temporal_tolerance_seconds
        )

        multi_speakers = bool(audio_observation.is_multiple_speakers)
        multi_faces = face_count >= 2

        # 3. Categorical cross-modal mapping
        if recent_visual and recent_acoustic:
            state = MultimodalState.CONGRUENT_SPEECH
            conf = (0.7 + 0.3 * audio_observation.speech_confidence)
            notes = "Candidate lip articulation and acoustic vocalization observed simultaneously."
        elif recent_acoustic and not recent_visual:
            state = MultimodalState.ACOUSTIC_ONLY
            conf = audio_observation.speech_confidence
            notes = (
                "Acoustic speech detected while candidate mouth remained stationary; "
                "flagged for human review (potential secondary room participant or external audio)."
            )
        elif recent_visual and not recent_acoustic:
            state = MultimodalState.VISUAL_ONLY
            conf = 0.65
            notes = "Candidate lip articulation observed without corresponding acoustic speech (whispering or silent mouthing)."
        else:
            state = MultimodalState.SILENT_AND_STILL
            conf = 1.0
            notes = "No visual speech articulation or acoustic speech detected."

        if multi_speakers:
            notes += " Acoustic pitch variation indicates multiple distinct speaker timbres."

        if multi_faces and multi_speakers:
            notes += " Corroborated: multiple persons visually localized and multiple acoustic speakers detected."

        return MultimodalObservation(
            timestamp_seconds=timestamp_seconds,
            state=state,
            visual_speech_detected=visual_speech,
            acoustic_speech_detected=acoustic_speech,
            multiple_speakers_detected=multi_speakers,
            multiple_faces_detected=multi_faces,
            correlation_confidence=conf,
            human_review_notes=notes,
        )

    def map_to_events(self, obs: MultimodalObservation) -> dict[EventType, dict[str, Any]]:
        """Map multimodal observation to discrete evidence records for human review."""
        events: dict[EventType, dict[str, Any]] = {}

        if obs.state == MultimodalState.CONGRUENT_SPEECH:
            events[EventType.MULTIMODAL_SPEECH_CONGRUENT] = {
                "confidence": obs.correlation_confidence,
                "description": obs.human_review_notes,
                "state": obs.state.value,
            }
        elif obs.state == MultimodalState.ACOUSTIC_ONLY:
            events[EventType.ACOUSTIC_SPEECH_WITHOUT_LIP_MOVEMENT] = {
                "confidence": obs.correlation_confidence,
                "description": obs.human_review_notes,
                "state": obs.state.value,
            }

        if obs.multiple_speakers_detected:
            events[EventType.MULTIPLE_SPEAKERS_DETECTED] = {
                "confidence": 0.85,
                "description": "Multiple acoustic vocal pitch ranges detected across audio window.",
            }

        return events
