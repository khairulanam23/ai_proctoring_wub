"""Single configuration object governing an examination proctoring session.

This module replaces four overlapping configuration classes that grew up alongside
four separate engines (``ProctoringSessionConfig``, ``RealTimeSessionConfig``,
``OptimizedProctoringConfig`` and ``ValidationConfig``).  Every knob in the
examination workflow now lives in one place, grouped by the workflow stage it
governs, so a session is reproducible from a single serialisable record.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from proctoring.analysis.policy import ExamPolicy, StrictnessLevel
from proctoring.core.paths import resolve_within, sanitise_identifier

# Sentinel distinguishing "caller left this alone" from "caller chose a value that
# happens to equal the default". Comparing against dataclass defaults cannot tell
# the two apart, which silently discarded explicit settings.
_UNSET = object()


@dataclass
class SessionConfig:
    """Every tunable parameter for one examination session, grouped by workflow stage.

    The defaults describe a conservative, evidence-first configuration: modest
    sampling rate, generous temporal qualification thresholds (so momentary noise
    never becomes an event), and the calibrated SFace operating threshold derived
    from the identity benchmark in ``docs/face_ai_baseline.md``.
    """

    # ------------------------------------------------------------------
    # Session identity
    # ------------------------------------------------------------------
    session_id: str = "exam_session"
    student_name: str = "Candidate"

    # ------------------------------------------------------------------
    # Stage 2 — Camera / frame input
    # ------------------------------------------------------------------
    sampling_fps: float = 4.0
    """Nominal rate at which frames are pulled from the source for inference."""

    enable_adaptive_sampling: bool = True
    """Drop to ``idle_fps`` while the scene is quiet, rise to ``active_fps`` once
    anything anomalous is on screen.  Cuts CPU cost during the long uneventful
    stretches that dominate a real exam without blunting event timing resolution."""

    idle_fps: float = 2.0
    active_fps: float = 4.0

    # ------------------------------------------------------------------
    # Stage 3 — Frame validation & preprocessing
    # ------------------------------------------------------------------
    enable_preprocessing: bool = True
    """Run the quality gate (resolution / brightness / blur) and apply CLAHE when
    the frame is under- or over-exposed."""

    min_frame_width: int = 160
    min_frame_height: int = 120

    # ------------------------------------------------------------------
    # Stage 4 — Face detection (YuNet)
    # ------------------------------------------------------------------
    enable_face_detection: bool = True
    face_score_threshold: float = 0.60

    min_face_size_px: int = 40
    """Reject face boxes smaller than this on either axis.  Suppresses posters,
    photographs on the wall and faces on a distant monitor being counted as a
    second person in the room."""

    # ------------------------------------------------------------------
    # Stage 5 — Face identity verification (SFace)
    # ------------------------------------------------------------------
    enable_face_verification: bool = True

    face_match_threshold: float = 0.3630
    """Calibrated cosine operating point for SFace.  At or above this score the
    detected face is treated as the enrolled candidate."""

    reference_template: np.ndarray | None = None
    """Mean embedding of the enrolled candidate, produced at enrolment."""

    reference_templates: list[np.ndarray] = field(default_factory=list)
    """Additional per-image enrolment embeddings.  A frame matches if it clears the
    threshold against *any* template, which keeps pose and lighting variation from
    reading as an identity mismatch."""

    # ------------------------------------------------------------------
    # Stage 6 — Scene / behavioural observation (object detection)
    # ------------------------------------------------------------------
    enable_object_detection: bool = True
    object_confidence_threshold: float = 0.25

    phone_confidence_threshold: Any = _UNSET
    """Deliberately higher than the generic object threshold: a wallet or dark
    notebook lying flat on a desk is the single most common YOLO phone false
    positive, and a spurious phone event is expensive for the candidate."""

    book_confidence_threshold: float = 0.35

    # ------------------------------------------------------------------
    # Stage 6b — Behavioural analysis (hands, speech, gaze, wearables)
    # ------------------------------------------------------------------
    strictness: StrictnessLevel = StrictnessLevel.STANDARD
    """Examination profile. Selects which behavioural observations are reportable
    and how long each must persist. See src/proctoring/analysis/policy.py."""

    enable_facial_dynamics: bool = True
    """Speech articulation, head pose and gaze, via the MediaPipe face landmarker."""

    enable_hand_analysis: bool = True
    """Hand presence and position relative to the face."""

    enable_wearable_detection: bool = False
    """Open-vocabulary headphone / earbud / smart-watch detection.

    Off by default: it costs roughly 250 ms per invocation on CPU — an order of
    magnitude more than every other stage combined — and earbud detection in
    particular is unreliable at webcam resolution. Enable it deliberately, and read
    docs/accuracy_and_performance.md before acting on its output."""

    wearable_detection_interval_frames: int = 8
    """Run the wearable detector once every N sampled frames. A device worn during
    an exam stays on for minutes, so sampling loses nothing while keeping the
    per-frame budget intact."""

    facial_dynamics_model: str = "models/face_landmarker.task"
    hand_model: str = "models/hand_landmarker.task"
    wearable_model: str = "yolov8s-world.pt"

    # ------------------------------------------------------------------
    # Stage 7 — Temporal qualification
    # ------------------------------------------------------------------
    absence_tolerance_seconds: Any = _UNSET
    """How long a condition may vanish before its incident is closed.  Bridges the
    one- or two-frame dropouts YuNet produces when the candidate turns their head.
    Left unset, the exam policy supplies it."""

    min_event_duration_seconds: Any = _UNSET
    """An incident shorter than this is still recorded, but marked ``RECORDED``
    rather than ``QUALIFIED`` — the transient-noise filter the workflow calls for.
    Left unset, the exam policy supplies it."""

    # ------------------------------------------------------------------
    # Stages 8-9 — Evidence creation & validation
    # ------------------------------------------------------------------
    capture_evidence: bool = True
    crop_padding_ratio: float = 0.10
    jpeg_quality: int = 95

    max_retained_evidence_frames: int = 200
    """Upper bound on decoded frames held in memory awaiting evidence capture.
    Each retained 640x480 frame costs roughly 0.9 MB, so this caps the evidence
    buffer near 180 MB even in a session that produces hundreds of incidents."""

    capture_review_snapshots: bool = True
    """Also store an annotated copy of each evidence frame under
    ``evidence/review/``, showing what the system reacted to. Source frames are
    always kept unmodified alongside them."""

    prune_invalid_evidence: bool = True
    """Detach evidence references that fail on-disk validation, so the delivered
    package never cites a file a proctor cannot open."""

    # ------------------------------------------------------------------
    # Stages 10-11 — Timeline, telemetry & packaging
    # ------------------------------------------------------------------
    _policy: ExamPolicy | None = None
    """Resolved exam policy. Left unset, it is derived from ``strictness``."""

    output_dir: str | Path = "data/results/sessions"
    create_zip: bool = False
    record_timeline: bool = True

    def __post_init__(self) -> None:
        # Guard against configurations that would silently disable qualification
        # or produce divide-by-zero timestamps downstream.
        self.sampling_fps = max(0.1, float(self.sampling_fps))
        self.idle_fps = max(0.1, float(self.idle_fps))
        self.active_fps = max(self.idle_fps, float(self.active_fps))
        self.output_dir = Path(self.output_dir)
        self.strictness = StrictnessLevel(self.strictness)

        # Identifiers reach the filesystem, so they are sanitised before anything
        # builds a path from them.
        self.session_id = sanitise_identifier(self.session_id, fallback="session")

        # Resolve the strictness preset unless the caller supplied a policy of their
        # own, then fill in only the thresholds the caller genuinely left unset.
        if self._policy is None:
            self._policy = ExamPolicy.for_level(self.strictness)

        policy = self._policy
        if self.min_event_duration_seconds is _UNSET:
            self.min_event_duration_seconds = policy.min_event_duration_seconds
        if self.absence_tolerance_seconds is _UNSET:
            self.absence_tolerance_seconds = policy.absence_tolerance_seconds
        if self.phone_confidence_threshold is _UNSET:
            self.phone_confidence_threshold = policy.phone_confidence_threshold

        self.absence_tolerance_seconds = max(0.0, float(self.absence_tolerance_seconds))
        self.min_event_duration_seconds = max(0.0, float(self.min_event_duration_seconds))
        self.phone_confidence_threshold = float(self.phone_confidence_threshold)

    @property
    def package_dir(self) -> Path:
        """Directory this session writes into, guaranteed inside ``output_dir``."""
        return self.resolve_output_path(self.session_id)

    def resolve_output_path(self, *segments: str) -> Path:
        """Join path segments under ``output_dir``, refusing to escape it.

        Belt and braces alongside :func:`sanitise_identifier`: even if a caller
        constructs an identifier by some other route, a path that resolves outside
        the configured directory is rejected rather than written.
        """
        return resolve_within(self.output_dir, *segments)

    @property
    def policy(self) -> ExamPolicy:
        """The resolved examination policy governing this session."""
        if self._policy is None:
            self._policy = ExamPolicy.for_level(self.strictness)
        return self._policy

    @property
    def has_reference_identity(self) -> bool:
        """True when at least one enrolment template is available for verification."""
        return self.reference_template is not None or bool(self.reference_templates)

    @property
    def all_reference_templates(self) -> list[np.ndarray]:
        """Every enrolment template, primary first."""
        templates: list[np.ndarray] = []
        if self.reference_template is not None:
            templates.append(self.reference_template)
        templates.extend(self.reference_templates)
        return templates

    def to_dict(self) -> dict[str, Any]:
        """Serialise the configuration for the evidence package manifest.

        Reference embeddings are reduced to a count: the manifest is handed to a
        human proctor and must not carry recoverable biometric material.
        """
        return {
            "session_id": self.session_id,
            "student_name": self.student_name,
            "sampling": {
                "sampling_fps": self.sampling_fps,
                "enable_adaptive_sampling": self.enable_adaptive_sampling,
                "idle_fps": self.idle_fps,
                "active_fps": self.active_fps,
            },
            "preprocessing": {
                "enabled": self.enable_preprocessing,
                "min_frame_width": self.min_frame_width,
                "min_frame_height": self.min_frame_height,
            },
            "face_detection": {
                "enabled": self.enable_face_detection,
                "score_threshold": self.face_score_threshold,
                "min_face_size_px": self.min_face_size_px,
            },
            "face_verification": {
                "enabled": self.enable_face_verification,
                "match_threshold": self.face_match_threshold,
                "enrolled_template_count": len(self.all_reference_templates),
            },
            "object_detection": {
                "enabled": self.enable_object_detection,
                "confidence_threshold": self.object_confidence_threshold,
                "phone_confidence_threshold": self.phone_confidence_threshold,
                "book_confidence_threshold": self.book_confidence_threshold,
            },
            "behavioural_analysis": {
                "strictness": self.strictness.value,
                "facial_dynamics": self.enable_facial_dynamics,
                "hand_analysis": self.enable_hand_analysis,
                "wearable_detection": self.enable_wearable_detection,
                "wearable_detection_interval_frames": self.wearable_detection_interval_frames,
            },
            "exam_policy": self.policy.to_dict(),
            "temporal_qualification": {
                "absence_tolerance_seconds": self.absence_tolerance_seconds,
                "min_event_duration_seconds": self.min_event_duration_seconds,
            },
            "evidence": {
                "capture_evidence": self.capture_evidence,
                "crop_padding_ratio": self.crop_padding_ratio,
                "jpeg_quality": self.jpeg_quality,
                "capture_review_snapshots": self.capture_review_snapshots,
                "prune_invalid_evidence": self.prune_invalid_evidence,
            },
            "output": {
                "output_dir": str(self.output_dir),
                "create_zip": self.create_zip,
                "record_timeline": self.record_timeline,
            },
            "has_reference_identity": self.has_reference_identity,
        }
