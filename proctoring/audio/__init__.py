"""Modular audio processing and multimodal correlation subsystem."""

from proctoring.audio.multimodal import (
    MultimodalCorrelator,
    MultimodalObservation,
    MultimodalState,
)
from proctoring.audio.processor import (
    AudioAnalyzer,
    AudioChunk,
    AudioObservation,
    AudioStatus,
)

__all__ = [
    "AudioStatus",
    "AudioChunk",
    "AudioObservation",
    "AudioAnalyzer",
    "MultimodalState",
    "MultimodalObservation",
    "MultimodalCorrelator",
]
