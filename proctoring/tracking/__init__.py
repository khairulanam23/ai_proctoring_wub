"""Persistent tracking and multi-subject temporal association for proctoring.

Provides frame-to-frame persistent identity tracking for faces, objects, and hands,
eliminating assumptions that detections[0] is always the candidate or that objects
belong to the first person in frame.
"""

from proctoring.tracking.tracker import (
    MultiSubjectTracker,
    TrackedHand,
    TrackedObject,
    TrackedSubject,
    TrackState,
)

__all__ = [
    "MultiSubjectTracker",
    "TrackState",
    "TrackedSubject",
    "TrackedObject",
    "TrackedHand",
]
