"""Behavioural analysis layer — workflow stage 6 beyond simple presence.

Adds the observations a strict examination setting needs on top of face presence
and identity: what the candidate's hands are doing, whether they are speaking,
where they are looking, and whether they are wearing an audio device.

Each analyzer degrades to unavailable rather than raising when its model is
missing, so a deployment without the MediaPipe or YOLO-World weights still runs
the core workflow.
"""

from proctoring.analysis.facial_dynamics import (
    FacialDynamicsAnalyzer,
    FacialDynamicsResult,
    HeadPose,
)
from proctoring.analysis.hands import (
    HandAnalysisResult,
    HandAnalyzer,
    HandObservation,
)
from proctoring.analysis.observer import BehaviourObserver
from proctoring.analysis.policy import (
    BEHAVIOURAL_SEVERITY,
    ExamPolicy,
    StrictnessLevel,
)
from proctoring.analysis.wearables import (
    WEARABLE_PROMPTS,
    WearableAnalysisResult,
    WearableDetection,
    WearableDetector,
)

__all__ = [
    "FacialDynamicsAnalyzer",
    "FacialDynamicsResult",
    "HeadPose",
    "HandAnalyzer",
    "HandAnalysisResult",
    "HandObservation",
    "ExamPolicy",
    "StrictnessLevel",
    "BEHAVIOURAL_SEVERITY",
    "BehaviourObserver",
    "WearableDetector",
    "WearableAnalysisResult",
    "WearableDetection",
    "WEARABLE_PROMPTS",
]
