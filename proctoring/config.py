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

from proctoring.analysis.policy import ExamMode, ExamPolicy, StrictnessLevel
from proctoring.core.events import EventType
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
    exam_mode: ExamMode = ExamMode.DIGITAL_SCREEN

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

    speech_sampling_applied: bool = field(default=False, init=False)
    """Set when the sampling rate was raised to make speech-like activity
    measurable. Recorded in the manifest so the rate a session actually ran at is
    never a mystery to whoever reads it later."""

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
    device: str = "auto"
    """Target execution device preference ('auto', 'cuda', 'cpu').
    In 'auto' mode:
    - YOLO11 uses CUDA if available, falling back to CPU safely.
    - YuNet & SFace run on CPU (OpenCV DNN MLAS SGEMM engine).
    - MediaPipe landmarker pipelines run on CPU (Google TFLite XNNPACK).
    """

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
    per-frame budget intact.

    Scaled automatically if the sampling rate is raised for speech measurement, so
    the sweep keeps the same wall-clock cadence rather than becoming more frequent
    as a side effect of an unrelated setting."""

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

    # ------------------------------------------------------------------
    # Hardware acceleration & runtime device
    # ------------------------------------------------------------------
    device: str = "cuda"
    """Target device for model inference: 'cuda', 'cuda:0', or 'cpu'."""

    cuda_device_index: int = 0
    """CUDA GPU ordinal if device is 'cuda'."""

    prefer_gpu_backends: bool = True
    """Whether to prefer GPU backends (ORT CUDA for YuNet/SFace, PyTorch CUDA for YOLO)."""

    def __post_init__(self) -> None:
        # Guard against configurations that would silently disable qualification
        # or produce divide-by-zero timestamps downstream.
        self.sampling_fps = max(0.1, float(self.sampling_fps))
        self.idle_fps = max(0.1, float(self.idle_fps))
        self.active_fps = max(self.idle_fps, float(self.active_fps))
        self.output_dir = Path(self.output_dir)
        self.strictness = StrictnessLevel(self.strictness)
        self.device = str(self.device).lower().strip()

        # Identifiers reach the filesystem, so they are sanitised before anything
        # builds a path from them.
        self.session_id = sanitise_identifier(self.session_id, fallback="session")

        # Resolve the strictness preset unless the caller supplied a policy of their
        # own, then fill in only the thresholds the caller genuinely left unset.
        if self._policy is None:
            self._policy = ExamPolicy.for_level(self.strictness, mode=self.exam_mode)

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

        self._apply_speech_sampling_floor(policy)

    def _apply_speech_sampling_floor(self, policy: ExamPolicy) -> None:
        """Raise the sampling rate when this exam reports speech-like mouth activity.

        Articulation happens at roughly 3-5 Hz. Sampled at the 4 fps default it
        aliases into a slow wave shaped exactly like a yawn, and no threshold placed
        downstream can undo that — the measurement is lost at the point of sampling.

        An exam that has asked to observe speaking therefore has its rate raised to
        the policy's floor, including the adaptive idle rate: a candidate who has
        been sitting quietly is exactly the one about to start talking, and dropping
        to 2 fps while idle would make the very moment of interest unmeasurable.

        Sessions that do not report speaking — ``STANDARD``, or any profile with
        facial dynamics disabled — keep their configured rate and pay nothing for
        this. The adjustment is recorded in the manifest rather than applied
        silently.
        """
        if not self.enable_facial_dynamics:
            return
        if not policy.allows(EventType.CANDIDATE_SPEAKING):
            return

        floor = max(0.1, float(policy.speech_min_sampling_fps))
        if self.sampling_fps >= floor and self.idle_fps >= floor:
            return

        previous_fps = self.sampling_fps
        self.sampling_fps = max(self.sampling_fps, floor)
        self.idle_fps = max(self.idle_fps, floor)
        self.active_fps = max(self.active_fps, self.sampling_fps)
        self.speech_sampling_applied = True

        # The wearable sweep cadence is configured in frames but is really a
        # wall-clock decision — a worn device stays on for minutes, and sweeping it
        # more often buys nothing. Scaling the interval keeps that cadence fixed, so
        # raising the rate for speech does not quietly raise the cost of the most
        # expensive detector in the pipeline along with it.
        if self.sampling_fps > previous_fps > 0:
            scale = self.sampling_fps / previous_fps
            self.wearable_detection_interval_frames = max(
                1, round(self.wearable_detection_interval_frames * scale)
            )

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
            self._policy = ExamPolicy.for_level(self.strictness, mode=self.exam_mode)
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
            "exam_mode": self.exam_mode.value,
            "sampling": {
                "sampling_fps": self.sampling_fps,
                "enable_adaptive_sampling": self.enable_adaptive_sampling,
                "idle_fps": self.idle_fps,
                "active_fps": self.active_fps,
                "raised_for_speech_measurement": self.speech_sampling_applied,
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionConfig":
        """Reconstruct SessionConfig from dictionary representation."""
        sampling = data.get("sampling", {})
        preproc = data.get("preprocessing", {})
        face_det = data.get("face_detection", {})
        face_ver = data.get("face_verification", {})
        obj_det = data.get("object_detection", {})
        behav = data.get("behavioural_analysis", {})
        temp = data.get("temporal_qualification", {})
        evid = data.get("evidence", {})
        out = data.get("output", {})

        exam_mode_str = data.get("exam_mode", "DIGITAL_SCREEN")
        try:
            exam_mode = ExamMode(exam_mode_str)
        except Exception:
            exam_mode = ExamMode.DIGITAL_SCREEN

        strictness_str = behav.get("strictness", "STANDARD")
        try:
            strictness = StrictnessLevel(strictness_str)
        except Exception:
            strictness = StrictnessLevel.STANDARD

        policy = ExamPolicy.for_level(strictness, exam_mode)

        return cls(
            session_id=data.get("session_id", "exam_session"),
            student_name=data.get("student_name", "Candidate"),
            exam_mode=exam_mode,
            sampling_fps=float(sampling.get("sampling_fps", data.get("sampling_fps", 4.0))),
            enable_adaptive_sampling=bool(sampling.get("enable_adaptive_sampling", data.get("enable_adaptive_sampling", True))),
            idle_fps=float(sampling.get("idle_fps", data.get("idle_fps", 2.0))),
            active_fps=float(sampling.get("active_fps", data.get("active_fps", 8.0))),
            enable_preprocessing=bool(preproc.get("enabled", data.get("enable_preprocessing", True))),
            min_frame_width=int(preproc.get("min_frame_width", data.get("min_frame_width", 320))),
            min_frame_height=int(preproc.get("min_frame_height", data.get("min_frame_height", 240))),
            enable_face_detection=bool(face_det.get("enabled", data.get("enable_face_detection", True))),
            face_score_threshold=float(face_det.get("score_threshold", data.get("face_score_threshold", 0.6))),
            min_face_size_px=int(face_det.get("min_face_size_px", data.get("min_face_size_px", 40))),
            enable_face_verification=bool(face_ver.get("enabled", data.get("enable_face_verification", True))),
            face_match_threshold=float(face_ver.get("match_threshold", data.get("face_match_threshold", 0.3630))),
            enable_object_detection=bool(obj_det.get("enabled", data.get("enable_object_detection", True))),
            object_confidence_threshold=float(obj_det.get("confidence_threshold", data.get("object_confidence_threshold", 0.4))),
            phone_confidence_threshold=float(obj_det.get("phone_confidence_threshold", data.get("phone_confidence_threshold", 0.4))),
            book_confidence_threshold=float(obj_det.get("book_confidence_threshold", data.get("book_confidence_threshold", 0.35))),
            enable_facial_dynamics=bool(behav.get("facial_dynamics", data.get("enable_facial_dynamics", True))),
            enable_hand_analysis=bool(behav.get("hand_analysis", data.get("enable_hand_analysis", True))),
            enable_wearable_detection=bool(behav.get("wearable_detection", data.get("enable_wearable_detection", False))),
            wearable_detection_interval_frames=int(behav.get("wearable_detection_interval_frames", data.get("wearable_detection_interval_frames", 12))),
            strictness=strictness,
            _policy=policy,
            absence_tolerance_seconds=float(temp.get("absence_tolerance_seconds", data.get("absence_tolerance_seconds", 0.5))),
            min_event_duration_seconds=float(temp.get("min_event_duration_seconds", data.get("min_event_duration_seconds", 1.0))),
            capture_evidence=bool(evid.get("capture_evidence", data.get("capture_evidence", True))),
            crop_padding_ratio=float(evid.get("crop_padding_ratio", data.get("crop_padding_ratio", 0.1))),
            jpeg_quality=int(evid.get("jpeg_quality", data.get("jpeg_quality", 95))),
            capture_review_snapshots=bool(evid.get("capture_review_snapshots", data.get("capture_review_snapshots", True))),
            prune_invalid_evidence=bool(evid.get("prune_invalid_evidence", data.get("prune_invalid_evidence", True))),
            output_dir=Path(out.get("output_dir", data.get("output_dir", "data/evidence_packages"))),
            create_zip=bool(out.get("create_zip", data.get("create_zip", True))),
            record_timeline=bool(out.get("record_timeline", data.get("record_timeline", True))),
        )
