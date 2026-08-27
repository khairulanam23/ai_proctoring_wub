"""Per-frame observation types shared by the engine and the analysis layer.

These live apart from the engine so the behavioural analysis layer can produce and
consume them without importing the orchestrator, which would be circular.
"""

from dataclasses import dataclass, field
from typing import Any

from proctoring.analysis.facial_dynamics import FacialDynamicsResult
from proctoring.analysis.gaze import GazeObservation
from proctoring.analysis.hands import HandAnalysisResult
from proctoring.analysis.occlusion import FaceOcclusionResult
from proctoring.analysis.wearables import WearableAnalysisResult
from proctoring.preprocessing.camera_health import CameraHealthStatus
from proctoring.telemetry.performance import FrameTimingRecord


class FaceStatus:
    """Scene states the workflow distinguishes after detection and verification.

    These are plain observations about what the camera saw, deliberately phrased
    without any accusation.  ``UNKNOWN_FACE`` means "this face did not match the
    enrolled template", not "an impostor is taking the exam" — the distinction
    matters because a proctor, not the system, decides what the observation means.
    """

    NOT_MEASURED = "NOT_MEASURED"  # Face detection did not run this frame
    NO_FACE = "NO_FACE"  # Face detection ran and found nobody
    ENROLLED = "ENROLLED"  # Single face, matched the enrolled candidate
    UNKNOWN_FACE = "UNKNOWN_FACE"  # Single face, did not match the enrolled template
    IDENTITY_UNCERTAIN = "IDENTITY_UNCERTAIN"  # Single face, similarity marginal / low quality
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"  # Single face, high quality but distinct from template
    UNVERIFIED = "UNVERIFIED"  # Single face, but no enrolment to compare against
    MULTIPLE_FACES = "MULTIPLE_FACES"  # More than one face in frame


@dataclass
class FrameObservation:
    """Everything stage 3-7 actually measured for one frame.

    Returned by :meth:`ProctoringEngine.process_frame` so callers — the live HUD,
    the timeline, embedding applications — read real detector output instead of
    inferring it.  A field left at ``None`` means the stage did not run, which is
    distinct from a stage running and measuring nothing.
    """

    frame_index: int
    timestamp_seconds: float
    iso_timestamp: str

    # Stage 3
    accepted: bool = True
    rejection_reason: str | None = None
    was_enhanced: bool = False
    mean_luminance: float | None = None
    blur_variance: float | None = None
    camera_health: CameraHealthStatus | None = None

    detector_failures: list[str] = field(default_factory=list)
    """Detectors that failed on this frame. Their observations are unavailable, which
    is a fact about the equipment and never about the candidate."""

    # Stages 4-5
    face_count: int | None = None
    face_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    face_confidences: list[float] = field(default_factory=list)
    identity_verified: bool | None = None
    similarity: float | None = None
    face_similarities: list[float | None] = field(default_factory=list)
    """Per-face similarity, aligned with ``face_boxes``. ``None`` where unmeasurable."""
    enrolled_face_present: bool | None = None
    """Whether the enrolled candidate is among the faces on screen."""
    unknown_face_count: int | None = None

    # Stage 6
    face_status: str = FaceStatus.NOT_MEASURED
    prohibited_objects: list[dict[str, Any]] = field(default_factory=list)

    # Stage 6b — behavioural analysis
    facial_dynamics: FacialDynamicsResult | None = None
    gaze: GazeObservation | None = None
    occlusion: FaceOcclusionResult | None = None
    hand_analysis: HandAnalysisResult | None = None
    wearables: WearableAnalysisResult | None = None

    # Stage 7
    active_event_types: list[str] = field(default_factory=list)

    # Telemetry
    timing: FrameTimingRecord | None = None

    @property
    def is_anomalous(self) -> bool:
        """True when this frame shows a condition a proctor would want to see."""
        return bool(self.active_event_types)

    @property
    def prohibited_object_names(self) -> list[str]:
        return [o["class_name"] for o in self.prohibited_objects]

    @property
    def is_speaking(self) -> bool | None:
        return self.facial_dynamics.is_speaking if self.facial_dynamics else None

    @property
    def hands_detected(self) -> int | None:
        return self.hand_analysis.hands_detected if self.hand_analysis else None

    @property
    def wearable_names(self) -> list[str]:
        return self.wearables.target_names if self.wearables else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "iso_timestamp": self.iso_timestamp,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "was_enhanced": self.was_enhanced,
            "camera_health": self.camera_health.to_dict() if self.camera_health else None,
            "detector_failures": self.detector_failures,
            "face_count": self.face_count,
            "face_status": self.face_status,
            "identity_verified": self.identity_verified,
            "similarity": round(self.similarity, 4) if self.similarity is not None else None,
            "enrolled_face_present": self.enrolled_face_present,
            "unknown_face_count": self.unknown_face_count,
            "prohibited_objects": self.prohibited_object_names,
            "facial_dynamics": self.facial_dynamics.to_dict() if self.facial_dynamics else None,
            "gaze": self.gaze.to_dict() if self.gaze else None,
            "occlusion": self.occlusion.to_dict() if self.occlusion else None,
            "hand_analysis": self.hand_analysis.to_dict() if self.hand_analysis else None,
            "wearables": self.wearables.to_dict() if self.wearables else None,
            "active_event_types": self.active_event_types,
        }
