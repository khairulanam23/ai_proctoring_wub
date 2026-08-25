"""The single proctoring engine: one implementation of the examination workflow.

This module replaces four engines that had grown up in parallel and had begun to
disagree with each other — ``UnifiedProctoringEngine``,
``AdvancedValidatedProctoringEngine``, ``OptimizedProctoringEngine`` and
``RealTimeProctoringSession``.  Each implemented a partial version of the same
pipeline, so a fix applied to one silently left the other three wrong.

:class:`ProctoringEngine` executes the workflow end to end::

    examination starts
      -> camera / frame input            (live webcam or recorded video)
      -> frame validation & preprocessing (resolution, brightness, quality, CLAHE)
      -> face detection                   (YuNet)
      -> face identity verification       (SFace, against enrolled templates)
      -> scene / behavioural observation  (presence, identity, prohibited objects)
      -> temporal qualification           (transient noise filtered out)
      -> event / evidence creation        (timestamped, with the frame that shows it)
      -> evidence validation              (re-read, re-hashed, invalid data removed)
      -> timeline & telemetry             (per-frame record + latency profile)
      -> tamper-evident evidence package  (manifest + SHA-256 over every artefact)
      -> proctor / invigilator review     (a human makes the decision)

Three entry points share that one implementation:

``process_frame``
    Feed frames yourself — used by the live CLI and by embedding callers.
``run_video``
    Run the whole workflow over a recorded file.
``finalize_session``
    Close open incidents, attach and validate evidence, seal the package.

The engine observes and records.  It never scores, ranks or judges a candidate:
the final decision belongs to the human proctor, and every output is shaped to
support that review rather than to pre-empt it.
"""

import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from proctoring.analysis.facial_dynamics import FacialDynamicsAnalyzer
from proctoring.analysis.hands import HandAnalyzer
from proctoring.analysis.observer import BehaviourObserver
from proctoring.analysis.policy import ExamPolicy
from proctoring.analysis.wearables import WearableAnalysisResult, WearableDetector
from proctoring.capture.video import VideoFrameSampler
from proctoring.config import SessionConfig
from proctoring.core.errors import ErrorCategory, PipelineErrorHandler
from proctoring.core.events import (
    DetectorInfo,
    EventRecord,
    EventType,
    coerce_event_type,
)
from proctoring.detection.face_detector import DetectionResult, FaceDetection, FaceDetector
from proctoring.detection.face_verifier import FaceVerifier
from proctoring.detection.object_detector import ObjectDetectionResult, ObjectDetector
from proctoring.detection.object_relevance import ObjectRelevanceFilter
from proctoring.evidence.annotator import AnnotationContext, EvidenceAnnotator
from proctoring.evidence.manager import EvidenceManager
from proctoring.evidence.package import SessionEvidencePackage
from proctoring.observation import FaceStatus, FrameObservation
from proctoring.preprocessing.frame_quality import (
    AdaptiveImagePreprocessor,
    FrameQualityGate,
)
from proctoring.telemetry.performance import (
    FrameTimingRecord,
    PerformanceReport,
    PipelineTelemetryTracker,
)
from proctoring.telemetry.timeline import SessionTimeline, TimelineEntry
from proctoring.temporal.aggregator import UnifiedTemporalAggregator

LOGGER = logging.getLogger(__name__)


@dataclass
class SessionSummary:
    """Outcome of a finalised session, including where the sealed package landed."""

    session_id: str
    student_name: str
    package_dir: Path
    zip_path: Path | None
    started_at_iso: str
    ended_at_iso: str
    total_frames: int
    processed_frames: int
    skipped_frames: int
    total_events: int
    qualified_events: int
    evidence_files_count: int
    telemetry: PerformanceReport
    timeline_summary: dict[str, Any]
    evidence_validation: dict[str, Any]
    integrity_verified: bool
    integrity_errors: list[str]
    events: list[EventRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "student_name": self.student_name,
            "package_dir": str(self.package_dir),
            "zip_path": str(self.zip_path) if self.zip_path else None,
            "started_at_iso": self.started_at_iso,
            "ended_at_iso": self.ended_at_iso,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "skipped_frames": self.skipped_frames,
            "total_events": self.total_events,
            "qualified_events": self.qualified_events,
            "evidence_files_count": self.evidence_files_count,
            "timeline_summary": self.timeline_summary,
            "evidence_validation": {
                k: v for k, v in self.evidence_validation.items() if k != "reports"
            },
            "integrity_verified": self.integrity_verified,
            "integrity_errors": self.integrity_errors,
        }


class ProctoringEngine:
    """Executes the examination proctoring workflow over a stream of frames.

    Detectors are injected rather than constructed here, which keeps the engine
    testable without model files and lets a caller disable a stage simply by
    passing ``None``.  A stage with no detector is skipped cleanly: its timeline
    fields stay ``None`` and it contributes no events, rather than the engine
    inventing a neutral result.
    """

    def __init__(
        self,
        config: SessionConfig | None = None,
        face_detector: FaceDetector | None = None,
        face_verifier: FaceVerifier | None = None,
        object_detector: ObjectDetector | None = None,
        relevance_filter: ObjectRelevanceFilter | None = None,
        facial_dynamics: FacialDynamicsAnalyzer | None = None,
        hand_analyzer: HandAnalyzer | None = None,
        wearable_detector: WearableDetector | None = None,
    ) -> None:
        self.config = config or SessionConfig()
        self.policy: ExamPolicy = self.config.policy
        self.session_id = self.config.session_id
        self.student_name = self.config.student_name

        # --- Detectors (stages 4-6) ---
        self.face_detector = face_detector
        self.face_verifier = face_verifier
        self.object_detector = object_detector
        self.relevance_filter = relevance_filter or ObjectRelevanceFilter(
            class_thresholds={
                "cell phone": self.config.phone_confidence_threshold,
                "book": self.config.book_confidence_threshold,
            },
            default_threshold=self.config.object_confidence_threshold,
        )

        # --- Stage 3 ---
        self.quality_gate = FrameQualityGate(
            preprocessor=AdaptiveImagePreprocessor(),
            min_width=self.config.min_frame_width,
            min_height=self.config.min_frame_height,
            enable_enhancement=self.config.enable_preprocessing,
        )

        # --- Stage 6b: behavioural analyzers ---
        # Thresholds come from the exam policy so a stricter session tightens the
        # detectors themselves, not merely which events are reported.
        self.facial_dynamics = facial_dynamics
        if self.facial_dynamics is None and self.config.enable_facial_dynamics:
            self.facial_dynamics = FacialDynamicsAnalyzer(
                model_path=self.config.facial_dynamics_model,
                speech_movement_threshold=self.policy.speech_movement_threshold,
                speech_min_oscillations=self.policy.speech_min_oscillations,
                yaw_limit_degrees=self.policy.yaw_limit_degrees,
                pitch_limit_degrees=self.policy.pitch_limit_degrees,
                gaze_offset_limit=self.policy.gaze_offset_limit,
                blink_threshold=self.policy.blink_threshold,
                liveness_grace_seconds=self.policy.liveness_grace_seconds,
            )

        self.hand_analyzer = hand_analyzer
        if self.hand_analyzer is None and self.config.enable_hand_analysis:
            self.hand_analyzer = HandAnalyzer(model_path=self.config.hand_model)

        self.wearable_detector = wearable_detector
        if self.wearable_detector is None and self.config.enable_wearable_detection:
            self.wearable_detector = WearableDetector(
                model_name=self.config.wearable_model,
                confidence_threshold=self.policy.headphone_confidence_threshold,
                earbud_confidence_threshold=self.policy.earbud_confidence_threshold,
            )

        # --- Stage 7 ---
        self.temporal_aggregator = self._build_temporal_aggregator()

        # --- Stages 8-11 ---
        self.evidence_manager = self._build_evidence_manager()
        self.observer = BehaviourObserver(self.policy)
        self.annotator = EvidenceAnnotator()
        self.telemetry = PipelineTelemetryTracker(session_id=self.session_id)
        self.timeline = SessionTimeline(session_id=self.session_id)
        self.error_handler = PipelineErrorHandler(session_id=self.session_id)

        # --- Provenance recorded in the manifest ---
        self.face_detector_info = DetectorInfo(
            name="OpenCV YuNet",
            version="2023mar",
            model_file="face_detection_yunet_2023mar.onnx",
            confidence_threshold=self.config.face_score_threshold,
        )
        self.face_verifier_info = DetectorInfo(
            name="OpenCV SFace",
            version="2021dec",
            model_file="face_recognition_sface_2021dec.onnx",
            confidence_threshold=self.config.face_match_threshold,
        )
        self.object_detector_info = DetectorInfo(
            name="Ultralytics YOLO11",
            version="11.0",
            model_file=getattr(object_detector, "model_name", "yolo11n.pt"),
            device=getattr(object_detector, "device", "cpu"),
            confidence_threshold=self.config.object_confidence_threshold,
        )
        self.behaviour_detector_info = DetectorInfo(
            name="MediaPipe Face+Hand Landmarker",
            version="tasks-1.0",
            model_file="face_landmarker.task + hand_landmarker.task",
            confidence_threshold=self.policy.speech_movement_threshold,
        )
        self.wearable_detector_info = DetectorInfo(
            name="YOLO-World (open vocabulary)",
            version="v8s-world",
            model_file=self.config.wearable_model,
            confidence_threshold=self.policy.earbud_confidence_threshold,
        )
        self.browser_bridge_info = DetectorInfo(
            name="Browser Proctoring Bridge",
            version="1.0",
            device="client",
        )

        # --- Session state ---
        self.is_active: bool = False
        self.started_at_iso: str | None = None
        self._start_perf: float | None = None
        self._frame_counter: int = 0

        # Frames held in memory so each event can be illustrated by the frame that
        # actually shows it.  See _retain_evidence_frames for the eviction policy.
        self._evidence_frames: OrderedDict[int, np.ndarray] = OrderedDict()
        self._last_accepted_frame: np.ndarray | None = None
        self._last_accepted_index: int = 0

        # Annotation context for each retained evidence frame, so a review snapshot
        # can be drawn at finalisation from what was seen when the frame was captured.
        self._frame_contexts: dict[int, AnnotationContext] = {}

        # Hands-absent bookkeeping: hands leaving view is only worth recording once
        # it has persisted, so the moment it started is tracked here.

        self._last_wearable_result: WearableAnalysisResult | None = None
        self._last_wearable_timestamp: float | None = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        """Begin a session, discarding every trace of any previous one.

        The accumulating collaborators — temporal aggregator, timeline, telemetry
        and error handler — are rebuilt rather than merely cleared.  An engine is
        commonly pooled and reused across quiz attempts, and carrying even one
        event forward would place one candidate's observations in another
        candidate's sealed evidence package.

        Telemetry is rebuilt here rather than in ``__init__`` for the same reason:
        its wall clock defines the reported session duration and effective FPS, and
        starting it at construction time makes both wrong for any engine that sat
        idle in a pool before the attempt began.
        """
        self.is_active = True
        self.started_at_iso = datetime.now(timezone.utc).isoformat()
        self._start_perf = time.perf_counter()
        self._frame_counter = 0
        self._evidence_frames.clear()
        self._frame_contexts.clear()
        self._last_accepted_frame = None
        self._last_accepted_index = 0
        self._last_wearable_result = None
        self._last_wearable_timestamp = None
        self.observer.reset()

        # Rebuild everything that accumulates across a session.
        self.temporal_aggregator = self._build_temporal_aggregator()
        self.timeline = SessionTimeline(session_id=self.session_id)
        self.telemetry = PipelineTelemetryTracker(session_id=self.session_id)
        self.error_handler = PipelineErrorHandler(session_id=self.session_id)
        self.evidence_manager = self._build_evidence_manager()

        if self.facial_dynamics is not None:
            self.facial_dynamics.reset()
        LOGGER.info("Session %s started (strictness=%s)", self.session_id, self.policy.level.value)

    def _build_temporal_aggregator(self) -> UnifiedTemporalAggregator:
        """Construct a temporal aggregator configured from the exam policy."""
        return UnifiedTemporalAggregator(
            session_id=self.session_id,
            absence_tolerance_seconds=self.config.absence_tolerance_seconds,
            min_event_duration_seconds=self.config.min_event_duration_seconds,
            min_duration_overrides=dict(self.policy.event_min_duration),
        )

    def _build_evidence_manager(self) -> EvidenceManager:
        """Construct the evidence manager for this session's output directory."""
        return EvidenceManager(
            base_dir=self.config.output_dir,
            session_id=self.session_id,
            crop_padding_ratio=self.config.crop_padding_ratio,
            jpeg_quality=self.config.jpeg_quality,
        )

    @property
    def target_fps(self) -> float:
        """Sampling rate to use for the next frame.

        With adaptive sampling on, the engine idles at ``idle_fps`` while nothing
        is happening and steps up to ``active_fps`` as soon as an incident is open,
        so the frames that end up as evidence are captured at full resolution in
        time while quiet stretches cost far less CPU.
        """
        if not self.config.enable_adaptive_sampling:
            return self.config.sampling_fps
        return (
            self.config.active_fps
            if self.temporal_aggregator.active_incidents
            else self.config.idle_fps
        )

    # ------------------------------------------------------------------
    # Stage 2-7: per-frame processing
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray | None,
        frame_index: int | None = None,
        timestamp_seconds: float | None = None,
    ) -> FrameObservation:
        """Run one frame through stages 3-7 and return what was observed.

        Never raises on detector failure.  A model that throws is recorded as a
        system diagnostic and the remaining stages still run, because a crashed
        object detector must not cost the session its face-presence record.  System
        errors and detection results stay strictly separate: a diagnostic never
        becomes an event against the candidate.
        """
        t_start = time.perf_counter()

        if frame_index is None:
            frame_index = self._frame_counter
        if timestamp_seconds is None:
            timestamp_seconds = frame_index / self.config.sampling_fps
        self._frame_counter = max(self._frame_counter, frame_index) + 1

        iso_ts = datetime.now(timezone.utc).isoformat()
        timing = FrameTimingRecord(frame_index=frame_index, timestamp_seconds=timestamp_seconds)
        per_model_times: dict[str, float] = {}

        obs = FrameObservation(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            iso_timestamp=iso_ts,
            timing=timing,
        )

        # --- Stage 3: frame validation & preprocessing -------------------
        t0 = time.perf_counter()
        gate = self.quality_gate.process(frame)
        timing.preprocessing_ms = (time.perf_counter() - t0) * 1000.0

        obs.was_enhanced = gate.was_enhanced
        obs.mean_luminance = gate.mean_luminance
        obs.blur_variance = gate.blur_variance

        if not gate.accepted:
            # A capture fault, not a proctoring observation.  Recorded as a skipped
            # frame and a diagnostic; the candidate is not implicated.
            obs.accepted = False
            obs.rejection_reason = gate.rejection_reason
            self.error_handler.handle_exception(
                error=ValueError(gate.rejection_reason or "Frame rejected"),
                category=ErrorCategory.CORRUPT_FRAME,
                timestamp_seconds=timestamp_seconds,
                frame_index=frame_index,
            )
            self.telemetry.record_skipped_frame()
            timing.total_frame_ms = (time.perf_counter() - t_start) * 1000.0
            self._append_timeline(obs)
            return obs

        working_frame = gate.frame
        self._last_accepted_frame = working_frame
        self._last_accepted_index = frame_index

        # --- Stages 4-5: face detection and identity verification --------
        faces: list[FaceDetection] = []
        if self.config.enable_face_detection and self.face_detector is not None:
            faces = self._detect_faces(
                working_frame, timestamp_seconds, frame_index, timing, per_model_times
            )
            obs.face_count = len(faces)
            obs.face_boxes = [f.bbox for f in faces]
            obs.face_confidences = [f.confidence for f in faces]

            self._resolve_identity(
                working_frame, faces, obs, timestamp_seconds, frame_index, timing, per_model_times
            )

        # --- Stage 6: scene observation (prohibited objects) -------------
        if self.config.enable_object_detection and self.object_detector is not None:
            obs.prohibited_objects = self._detect_objects(
                working_frame, timestamp_seconds, frame_index, timing, per_model_times
            )

        # --- Stage 6b: behavioural analysis ------------------------------
        self._analyze_behaviour(working_frame, obs, timing, per_model_times)

        # The face landmarker also localises faces. When no dedicated detector is
        # configured, use it for presence rather than leaving the frame unmeasured —
        # otherwise a deployment without YuNet reports nothing about presence while
        # simultaneously measuring that candidate's head pose.
        if obs.face_status is FaceStatus.NOT_MEASURED and obs.facial_dynamics is not None:
            dynamics = obs.facial_dynamics
            if dynamics.face_found:
                obs.face_count = 1
                obs.face_boxes = [self._to_xywh(dynamics.face_bbox)]
                obs.face_confidences = [1.0]
                obs.face_status = FaceStatus.UNVERIFIED
            else:
                obs.face_count = 0
                obs.face_status = FaceStatus.NO_FACE

        # --- Stage 7: temporal qualification -----------------------------
        t0 = time.perf_counter()
        self.temporal_aggregator.update_face_observation(
            face_status=obs.face_status,
            timestamp=timestamp_seconds,
            frame_index=frame_index,
            face_count=obs.face_count or 0,
            detector=self.face_detector_info,
            similarity_score=obs.similarity,
            bboxes=[self._to_xyxy(b) for b in obs.face_boxes],
            confidences=obs.face_confidences,
            enrolled_present=obs.enrolled_face_present,
        )
        self.temporal_aggregator.update_object_observations(
            detected_objects=obs.prohibited_objects,
            timestamp=timestamp_seconds,
            frame_index=frame_index,
            detector=self.object_detector_info,
        )
        self.temporal_aggregator.update_behaviour_observations(
            active_behaviours=self.observer.map_to_events(obs),
            timestamp=timestamp_seconds,
            frame_index=frame_index,
            detector=self.behaviour_detector_info,
        )
        timing.temporal_postprocess_ms = (time.perf_counter() - t0) * 1000.0

        obs.active_event_types = sorted(
            {inc.event_type.value for inc in self.temporal_aggregator.active_incidents.values()}
        )

        # --- Stage 8 (part 1): retain the frames that will become evidence
        if self.config.capture_evidence:
            self._retain_evidence_frames(working_frame, frame_index, obs)

        timing.total_frame_ms = (time.perf_counter() - t_start) * 1000.0
        self.telemetry.record_frame(timing, per_model_times=per_model_times)
        self._append_timeline(obs)
        return obs

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _detect_faces(
        self,
        frame: np.ndarray,
        timestamp_seconds: float,
        frame_index: int,
        timing: FrameTimingRecord,
        per_model_times: dict[str, float],
    ) -> list[FaceDetection]:
        """Stage 4 — YuNet detection, with background faces filtered out."""
        try:
            t0 = time.perf_counter()
            result: DetectionResult = self.face_detector.detect(
                frame, score_threshold=self.config.face_score_threshold
            )
            dt = (time.perf_counter() - t0) * 1000.0
            timing.face_detector_ms = dt
            per_model_times["face_detector"] = dt

            # Drop boxes below the minimum size: a face on a poster or on a monitor
            # in the background is not a second person in the room, and counting it
            # as one would raise a MULTIPLE_FACES event against the candidate.
            return [
                f
                for f in result.faces
                if f.bbox[2] >= self.config.min_face_size_px
                and f.bbox[3] >= self.config.min_face_size_px
            ]
        except Exception as exc:
            LOGGER.warning("Face detection failed at frame %s: %s", frame_index, exc)
            self.error_handler.handle_exception(
                error=exc,
                category=ErrorCategory.MODEL_INFERENCE_FAILURE,
                timestamp_seconds=timestamp_seconds,
                frame_index=frame_index,
                details={"pipeline_stage": "face_detector"},
            )
            return []

    def _resolve_identity(
        self,
        frame: np.ndarray,
        faces: list[FaceDetection],
        obs: FrameObservation,
        timestamp_seconds: float,
        frame_index: int,
        timing: FrameTimingRecord,
        per_model_times: dict[str, float],
    ) -> None:
        """Stage 5 — verify **every** detected face, not only a lone one.

        Verifying only when exactly one face is present leaves the case that matters
        most unanswered: with two people on camera the pipeline previously reported
        ``MULTIPLE_FACES`` and no identity signal at all, so "the candidate plus
        someone else" and "two strangers, candidate gone" were indistinguishable.

        Each face is scored against the enrolment templates. The frame's headline
        status reflects the strongest evidence available, and the per-face detail is
        preserved on the observation so a proctor can see which face matched.
        """
        obs.face_count = len(faces)
        obs.face_boxes = [f.bbox for f in faces]
        obs.face_confidences = [f.confidence for f in faces]

        if not faces:
            obs.face_status = FaceStatus.NO_FACE
            return

        templates = self.config.all_reference_templates
        can_verify = bool(
            self.config.enable_face_verification and self.face_verifier is not None and templates
        )

        if not can_verify:
            # No basis on which to call any face known or unknown.
            obs.face_status = FaceStatus.MULTIPLE_FACES if len(faces) > 1 else FaceStatus.UNVERIFIED
            return

        scores: list[float | None] = []
        for face in faces:
            scores.append(
                self._score_face(
                    frame, face, templates, timestamp_seconds, frame_index, timing, per_model_times
                )
            )

        measured = [s for s in scores if s is not None]
        obs.face_similarities = scores
        obs.similarity = max(measured) if measured else None

        if not measured:
            # Every embedding failed — a system fault, not an identity finding.
            obs.face_status = FaceStatus.MULTIPLE_FACES if len(faces) > 1 else FaceStatus.UNVERIFIED
            return

        matched = [s >= self.config.face_match_threshold for s in measured]
        obs.enrolled_face_present = any(matched)
        obs.unknown_face_count = sum(1 for m in matched if not m)
        obs.identity_verified = obs.enrolled_face_present

        if len(faces) > 1:
            # Multiple people is the headline; whether the candidate is among them
            # is carried alongside it rather than replacing it.
            obs.face_status = FaceStatus.MULTIPLE_FACES
        elif obs.enrolled_face_present:
            obs.face_status = FaceStatus.ENROLLED
        else:
            obs.face_status = FaceStatus.UNKNOWN_FACE

    def _score_face(
        self,
        frame: np.ndarray,
        face: FaceDetection,
        templates: list[np.ndarray],
        timestamp_seconds: float,
        frame_index: int,
        timing: FrameTimingRecord,
        per_model_times: dict[str, float],
    ) -> float | None:
        """Best cosine similarity of one face against all enrolment templates.

        Returns ``None`` when the embedding could not be computed, so a model fault
        is never mistaken for a mismatch.
        """
        try:
            t0 = time.perf_counter()
            embedding = self._extract_embedding(frame, face)
            if embedding is None:
                return None
            similarity = max(
                float(self.face_verifier.compute_similarity(embedding, t)) for t in templates
            )
            elapsed = (time.perf_counter() - t0) * 1000.0
            timing.face_embedder_ms += elapsed
            per_model_times["face_embedder"] = timing.face_embedder_ms
            return similarity
        except Exception as exc:
            LOGGER.warning("Face embedding failed at frame %s: %s", frame_index, exc)
            self.error_handler.handle_exception(
                error=exc,
                category=ErrorCategory.MODEL_INFERENCE_FAILURE,
                timestamp_seconds=timestamp_seconds,
                frame_index=frame_index,
                details={"pipeline_stage": "face_verifier"},
            )
            return None

    def _extract_embedding(self, frame: np.ndarray, face: FaceDetection) -> np.ndarray | None:
        """Obtain an SFace embedding, tolerating either verifier API shape."""
        if hasattr(self.face_verifier, "extract_embedding"):
            return self.face_verifier.extract_embedding(frame, raw_detection=face.raw_detection)
        if hasattr(self.face_verifier, "extract_feature"):
            return self.face_verifier.extract_feature(frame, face=face)
        return None

    def _detect_objects(
        self,
        frame: np.ndarray,
        timestamp_seconds: float,
        frame_index: int,
        timing: FrameTimingRecord,
        per_model_times: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Stage 6 — YOLO detection reduced to proctoring-relevant objects."""
        try:
            t0 = time.perf_counter()
            result: ObjectDetectionResult = self.object_detector.detect(frame)
            report = self.relevance_filter.filter(result)
            dt = (time.perf_counter() - t0) * 1000.0
            timing.object_detector_ms = dt
            per_model_times["object_detector"] = dt

            return [
                {
                    "class_name": obj.class_name,
                    "confidence": obj.confidence,
                    "bbox": obj.bbox,
                }
                for obj in report.relevant_objects
            ]
        except Exception as exc:
            LOGGER.warning("Object detection failed at frame %s: %s", frame_index, exc)
            self.error_handler.handle_exception(
                error=exc,
                category=ErrorCategory.MODEL_INFERENCE_FAILURE,
                timestamp_seconds=timestamp_seconds,
                frame_index=frame_index,
                details={"pipeline_stage": "object_detector"},
            )
            return []

    def _analyze_behaviour(
        self,
        frame: np.ndarray,
        obs: FrameObservation,
        timing: FrameTimingRecord,
        per_model_times: dict[str, float],
    ) -> None:
        """Stage 6b — measure hands, speech articulation, gaze and worn devices.

        Runs after face detection so the face geometry is available: hand proximity
        and earpiece plausibility are both defined relative to the face, and neither
        can be judged without it.

        Each analyzer is individually guarded. One failing model degrades its own
        signal to unmeasured and leaves the others intact, because losing gaze
        estimation is not a reason to lose the face-presence record.
        """
        behaviour_start = time.perf_counter()

        # --- Facial dynamics: speaking, head pose, gaze ---
        if self.facial_dynamics is not None and self.facial_dynamics.is_available:
            try:
                obs.facial_dynamics = self.facial_dynamics.analyze(
                    frame, timestamp_seconds=obs.timestamp_seconds
                )
                per_model_times["facial_dynamics"] = obs.facial_dynamics.inference_ms
            except Exception as exc:
                self._record_analyzer_failure(exc, "facial_dynamics", obs)

        # --- Hands, related to the face region when one was found ---
        if self.hand_analyzer is not None and self.hand_analyzer.is_available:
            try:
                dynamics = obs.facial_dynamics
                obs.hand_analysis = self.hand_analyzer.analyze(
                    frame,
                    face_bbox=dynamics.face_bbox if dynamics else None,
                    ear_regions=dynamics.ear_regions if dynamics else None,
                    mouth_region=dynamics.mouth_region if dynamics else None,
                )
                per_model_times["hand_analysis"] = obs.hand_analysis.inference_ms
            except Exception as exc:
                self._record_analyzer_failure(exc, "hand_analysis", obs)

        # --- Worn devices, on a decimated cadence ---
        if self._should_run_wearable_detection(obs.frame_index):
            try:
                dynamics = obs.facial_dynamics
                result = self.wearable_detector.detect(
                    frame,
                    ear_regions=dynamics.ear_regions if dynamics else None,
                    face_bbox=dynamics.face_bbox if dynamics else None,
                )
                obs.wearables = result
                self._last_wearable_result = result
                self._last_wearable_timestamp = obs.timestamp_seconds
                per_model_times["wearable_detector"] = result.inference_ms
            except Exception as exc:
                self._record_analyzer_failure(exc, "wearable_detector", obs)
        elif self._last_wearable_result is not None:
            # Carry the most recent reading forward on skipped frames so a device
            # incident stays continuous instead of flickering with the cadence — but
            # only for as long as a fresh sweep was actually due. Without an expiry a
            # detector that starts failing would pin the last detection open for the
            # rest of the session.
            # Explicit None check: a sweep at timestamp 0.0 is falsy, and `or` would
            # silently treat the very first reading as fresh forever.
            last = self._last_wearable_timestamp
            age = (obs.timestamp_seconds - last) if last is not None else 0.0
            if age <= self._wearable_carry_forward_seconds:
                obs.wearables = self._last_wearable_result
            else:
                LOGGER.debug("Discarding stale wearable reading (%.1fs old)", age)
                self._last_wearable_result = None
                self._last_wearable_timestamp = None

        timing.behaviour_analysis_ms = (time.perf_counter() - behaviour_start) * 1000.0

    @property
    def _wearable_carry_forward_seconds(self) -> float:
        """How long a wearable reading stays valid: two sweep intervals plus slack."""
        interval = max(1, int(self.config.wearable_detection_interval_frames))
        return (interval * 2.0) / max(0.1, self.config.sampling_fps)

    def _should_run_wearable_detection(self, frame_index: int) -> bool:
        """Is this frame due for the (expensive) open-vocabulary device sweep?"""
        if self.wearable_detector is None or not self.wearable_detector.is_available:
            return False
        interval = max(1, int(self.config.wearable_detection_interval_frames))
        return frame_index % interval == 0

    def _record_analyzer_failure(self, error: Exception, stage: str, obs: FrameObservation) -> None:
        """Log an analyzer fault as a system diagnostic, never as a candidate event."""
        LOGGER.warning("Analyzer %s failed at frame %s: %s", stage, obs.frame_index, error)
        self.error_handler.handle_exception(
            error=error,
            category=ErrorCategory.MODEL_INFERENCE_FAILURE,
            timestamp_seconds=obs.timestamp_seconds,
            frame_index=obs.frame_index,
            details={"pipeline_stage": stage},
        )

    @staticmethod
    def _to_xywh(bbox: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
        """Convert an ``(x1, y1, x2, y2)`` box to the ``(x, y, w, h)`` detector form."""
        if bbox is None:
            return (0, 0, 0, 0)
        x1, y1, x2, y2 = bbox
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    @staticmethod
    def _to_xyxy(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Convert a YuNet ``(x, y, w, h)`` box to the ``(x1, y1, x2, y2)`` form
        used for evidence cropping."""
        x, y, w, h = bbox
        return int(x), int(y), int(x + w), int(y + h)

    def _append_timeline(self, obs: FrameObservation) -> None:
        """Record the frame's real observation on the session timeline (stage 10)."""
        if not self.config.record_timeline:
            return
        self.timeline.append(
            TimelineEntry(
                frame_index=obs.frame_index,
                timestamp_seconds=obs.timestamp_seconds,
                iso_timestamp=obs.iso_timestamp,
                frame_accepted=obs.accepted,
                rejection_reason=obs.rejection_reason,
                was_enhanced=obs.was_enhanced,
                mean_luminance=obs.mean_luminance,
                blur_variance=obs.blur_variance,
                face_count=obs.face_count,
                face_boxes=[list(b) for b in obs.face_boxes],
                identity_verified=obs.identity_verified,
                cosine_similarity=obs.similarity,
                face_status=obs.face_status if obs.accepted else None,
                prohibited_objects=obs.prohibited_object_names,
                active_event_types=obs.active_event_types,
                is_anomalous_state=obs.is_anomalous,
                processing_latency_ms=obs.timing.total_frame_ms if obs.timing else 0.0,
                hands_detected=obs.hands_detected,
                hand_near_face=hands.hand_near_face if (hands := obs.hand_analysis) else None,
                hand_near_ear=hands.hand_near_ear if hands else None,
                is_speaking=dyn.is_speaking if (dyn := obs.facial_dynamics) else None,
                speech_activity=dyn.speech_activity if dyn else None,
                head_yaw=dyn.head_pose.yaw if (dyn and dyn.head_pose) else None,
                head_pitch=dyn.head_pose.pitch if (dyn and dyn.head_pose) else None,
                is_looking_away=dyn.is_looking_away if dyn else None,
                gaze_offset=dyn.gaze_offset if dyn else None,
                detected_wearables=obs.wearable_names,
            )
        )

    # ------------------------------------------------------------------
    # Stage 8: evidence frame retention
    # ------------------------------------------------------------------

    def _retain_evidence_frames(
        self,
        frame: np.ndarray,
        frame_index: int,
        obs: FrameObservation | None = None,
    ) -> None:
        """Keep a copy of any frame that is currently the best example of an incident.

        This is what makes the evidence in the package actually correspond to the
        event it documents.  The temporal aggregator tracks, per incident, which
        frame showed the condition most clearly (``best_frame_index``); that frame
        must survive in memory until the incident closes and evidence is written at
        finalisation, because by then the video source has long moved past it.

        Frames no longer referenced by any open incident or any closed event are
        dropped immediately, so memory tracks the number of incidents rather than
        the length of the session.
        """
        needed = self._referenced_frame_indices()
        if frame_index in needed and frame_index not in self._evidence_frames:
            self._evidence_frames[frame_index] = frame.copy()
            if obs is not None and self.config.capture_review_snapshots:
                # Snapshot what was observed now; at finalisation the detectors have
                # long moved on and the context could not be reconstructed.
                self._frame_contexts[frame_index] = self.observer.build_annotation_context(obs)

        # Release frames that stopped being anyone's best example.
        for stale in [i for i in self._evidence_frames if i not in needed]:
            del self._evidence_frames[stale]
            self._frame_contexts.pop(stale, None)

        # Hard ceiling as a safety net against a pathological session; oldest first.
        while len(self._evidence_frames) > self.config.max_retained_evidence_frames:
            evicted, _ = self._evidence_frames.popitem(last=False)
            self._frame_contexts.pop(evicted, None)

    def _referenced_frame_indices(self) -> set:
        """Frame indices still needed to illustrate an open incident or closed event."""
        indices = {
            inc.best_frame_index for inc in self.temporal_aggregator.active_incidents.values()
        }
        for event in self.temporal_aggregator.closed_events:
            best = event.metadata.get("best_frame_index")
            if best is not None:
                indices.add(int(best))
        return indices

    # ------------------------------------------------------------------
    # Browser / application events (stage 6, client-supplied)
    # ------------------------------------------------------------------

    def record_browser_event(
        self,
        event_type: EventType | str,
        timestamp_seconds: float,
        description: str,
        frame_index: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> EventRecord:
        """Record a client-reported browser or application event.

        Accepts a plain string as well as an :class:`EventType`, since this is
        called from outside the pipeline (a browser bridge, a CLI keypress) where
        the caller has no reason to hold the enum.
        """
        return self.temporal_aggregator.record_instant_event(
            event_type=coerce_event_type(event_type),
            timestamp=timestamp_seconds,
            frame_index=frame_index,
            description=description,
            detector=self.browser_bridge_info,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Offline entry point
    # ------------------------------------------------------------------

    def run_video(
        self,
        video_path: str | Path,
        progress_callback: Callable[[FrameObservation], None] | None = None,
    ) -> SessionSummary:
        """Run the complete workflow over a recorded video and seal the package."""
        if not self.is_active:
            self.start_session()

        sampler = VideoFrameSampler(video_path, target_sampling_fps=self.config.sampling_fps)
        for sample in sampler.sample_frames():
            observation = self.process_frame(
                sample.frame,
                frame_index=sample.frame_index,
                timestamp_seconds=sample.timestamp_seconds,
            )
            if progress_callback is not None:
                progress_callback(observation)

        return self.finalize_session()

    # ------------------------------------------------------------------
    # Stages 8-11: finalisation
    # ------------------------------------------------------------------

    def finalize_session(self) -> SessionSummary:
        """Close open incidents, attach and validate evidence, and seal the package.

        Runs the tail of the workflow in the only order that is sound:
        close incidents, attach each one's own evidence frame, validate what
        landed on disk and drop what did not, then compute the manifest over the
        final set of files.  Reordering any of these would leave the manifest
        describing a package that no longer exists.
        """
        self.is_active = False
        ended_at_iso = datetime.now(timezone.utc).isoformat()

        # --- Stage 7 (close-out) ---
        events = self.temporal_aggregator.flush()

        # --- Stage 8: attach each event's own evidence ---
        if self.config.capture_evidence:
            self._attach_evidence(events)

        # --- Stage 9: validate evidence, removing anything unreadable ---
        validation_report = self.evidence_manager.validate_all_event_evidence(
            events, prune_invalid=self.config.prune_invalid_evidence
        )

        # --- Stage 10: telemetry & timeline ---
        telemetry_report = self.telemetry.generate_report()
        timeline_summary = self.timeline.summarise()

        # --- Stage 11: tamper-evident package ---
        packager = SessionEvidencePackage(
            base_dir=self.config.output_dir,
            session_id=self.session_id,
            student_name=self.student_name,
        )
        package_dir = packager.build_package(
            events=events,
            telemetry_report=telemetry_report,
            diagnostics=self.error_handler.diagnostics,
            processing_config=self.config.to_dict(),
            models_info={
                "face_detector": self.face_detector_info.to_dict(),
                "face_verifier": self.face_verifier_info.to_dict(),
                "object_detector": self.object_detector_info.to_dict(),
            },
            create_zip=self.config.create_zip,
            timeline_entries=self.timeline.to_list() if self.config.record_timeline else None,
            timeline_summary=timeline_summary,
        )
        integrity_ok, integrity_errors = packager.verify_package_integrity()

        # Evidence frames have been written to disk; release the in-memory copies
        # and shut the landmark models down deterministically rather than leaving
        # them to interpreter teardown, which emits noise on exit.
        self._evidence_frames.clear()
        self._frame_contexts.clear()
        self.close_analyzers()

        return SessionSummary(
            session_id=self.session_id,
            student_name=self.student_name,
            package_dir=package_dir,
            zip_path=packager.zip_path,
            started_at_iso=self.started_at_iso or ended_at_iso,
            ended_at_iso=ended_at_iso,
            total_frames=telemetry_report.total_frames,
            processed_frames=telemetry_report.processed_frames,
            skipped_frames=telemetry_report.skipped_frames,
            total_events=len(events),
            qualified_events=sum(
                1 for e in events if e.metadata.get("is_duration_qualified", True)
            ),
            evidence_files_count=validation_report.get("valid_references", 0),
            telemetry=telemetry_report,
            timeline_summary=timeline_summary,
            evidence_validation=validation_report,
            integrity_verified=integrity_ok,
            integrity_errors=integrity_errors,
            events=events,
        )

    def _attach_evidence(self, events: list[EventRecord]) -> None:
        """Attach to each event the frame that actually shows it.

        Each event carries ``best_frame_index`` — the frame in which its condition
        was observed most clearly — and that exact frame is retrieved from the
        retention buffer.  Falling back to a single shared frame for every event
        (as the pipeline previously did) makes the package internally inconsistent:
        an event stamped 00:01 would be illustrated by a picture taken at 00:47.

        If the frame is genuinely unavailable the event is annotated to say so
        instead of being illustrated with an unrelated one.  An event with no
        picture is honest; an event with the wrong picture is misleading evidence
        in front of a proctor.
        """
        for event in events:
            best_index = event.metadata.get("best_frame_index")
            best_timestamp = event.metadata.get("best_timestamp", event.timestamp)
            raw_bbox = event.metadata.get("representative_bbox")
            bbox = tuple(raw_bbox) if raw_bbox else None
            label = event.observation.object_class or event.event_type.value.lower()

            frame = None
            if best_index is not None:
                frame = self._evidence_frames.get(int(best_index))

            if frame is None:
                # Instantaneous events (browser bridge) never had a frame of their
                # own; for those the closest accepted frame is the honest choice and
                # the substitution is recorded in metadata.
                if self._last_accepted_frame is not None and event.duration == 0.0:
                    frame = self._last_accepted_frame
                    best_index = self._last_accepted_index
                    event.metadata["evidence_frame_is_nearest_available"] = True
                else:
                    event.metadata["evidence_unavailable"] = (
                        "No retained frame for this event's best_frame_index"
                    )
                    continue

            self.evidence_manager.attach_evidence_to_event(
                event=event,
                frame=frame,
                frame_index=int(best_index),
                timestamp_seconds=float(best_timestamp),
                bbox=bbox,
                label=label,
            )

            # Derived review image: same frame, marked up to show what fired. The
            # unmodified source frame stays attached alongside it.
            if self.config.capture_review_snapshots:
                self._attach_review_snapshot(event, frame, int(best_index), float(best_timestamp))

    def _attach_review_snapshot(
        self,
        event: EventRecord,
        frame: np.ndarray,
        frame_index: int,
        timestamp_seconds: float,
    ) -> None:
        """Render and attach an annotated copy of an event's evidence frame.

        A failure here is deliberately silent beyond a metadata note: the review
        image is a convenience for the proctor, and losing it must never cost the
        event its source evidence.
        """
        try:
            context = self._frame_contexts.get(frame_index)
            annotated = self.annotator.annotate_event(frame, event, context)
            reference = self.evidence_manager.save_review_snapshot(
                annotated_frame=annotated,
                event_id=event.event_id,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
            )
            if reference is not None:
                reference.validation_error = None
                event.evidence.append(reference)
                event.metadata["has_review_snapshot"] = True
        except Exception as exc:
            LOGGER.warning("Review snapshot failed for %s: %s", event.event_id, exc)
            event.metadata["review_snapshot_error"] = str(exc)

    def close_analyzers(self) -> None:
        """Release the behavioural analyzer models held by this engine."""
        for analyzer in (self.facial_dynamics, self.hand_analyzer):
            if analyzer is not None:
                try:
                    analyzer.close()
                except Exception as exc:
                    LOGGER.debug("Analyzer close failed: %s", exc)
