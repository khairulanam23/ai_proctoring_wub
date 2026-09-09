"""The single proctoring engine: orchestrates the examination workflow end to end.

Coordinates input preprocessing, detection/analysis stages, temporal qualification,
evidence capture, performance telemetry, and tamper-evident packaging.

The engine observes and records. It never computes cumulative risk scores or makes
automated cheating judgments: the final decision belongs to the human invigilator.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from proctoring.analysis.facial_dynamics import FacialDynamicsAnalyzer
from proctoring.analysis.hands import HandAnalyzer
from proctoring.analysis.observer import BehaviourObserver
from proctoring.analysis.policy import ExamPolicy
from proctoring.analysis.wearables import WearableDetector
from proctoring.capture.video import VideoFrameSampler
from proctoring.config import SessionConfig
from proctoring.core.errors import ErrorCategory, PipelineErrorHandler
from proctoring.core.events import (
    DetectorInfo,
    EventRecord,
    EventType,
    coerce_event_type,
)
from proctoring.core.persistence import SessionCheckpoint, SessionJournalManager
from proctoring.detection.face_detector import FaceDetection, FaceDetector
from proctoring.detection.face_verifier import FaceVerifier
from proctoring.detection.object_detector import ObjectDetector
from proctoring.detection.object_relevance import ObjectRelevanceFilter
from proctoring.engine_evidence import EvidenceCoordinator
from proctoring.engine_stages import StageCoordinator
from proctoring.evidence.annotator import EvidenceAnnotator
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


class EngineState(str, Enum):
    """Processing lifecycle of the engine for one examination session.

    Distinct from :class:`proctoring.integration.schemas.SessionState`, which
    tracks the *attempt record* a host LMS sees. The two model different things —
    engine processing versus attempt bookkeeping — and previously shared a name,
    so ``SessionState`` meant different things depending on the import path.
    """

    CREATED = "CREATED"  # Session configured but frame processing not yet started
    CALIBRATING = "CALIBRATING"  # Performing neutral gaze/pose calibration
    RUNNING = "RUNNING"  # Active frame processing
    PAUSED = "PAUSED"  # Temporarily suspended by proctor or system
    FINALIZING = "FINALIZING"  # Actively flushing evidence and writing tamper-evident manifest
    FINALIZED = "FINALIZED"  # Completed and tamper-evident sealed
    CANCELLED = "CANCELLED"  # Aborted prematurely without sealing
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"  # Interrupted uncleanly; checkpoint available for recovery


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

    Detectors are injected rather than constructed internally, keeping the engine
    testable without model files and enabling graceful degradation when a stage
    is disabled or unavailable.
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

        # --- Stage 3: Frame Preprocessing Gate ---
        self.quality_gate = FrameQualityGate(
            preprocessor=AdaptiveImagePreprocessor(),
            min_width=self.config.min_frame_width,
            min_height=self.config.min_frame_height,
            enable_enhancement=self.config.enable_preprocessing,
        )

        # --- Core Logging & Telemetry ---
        self.error_handler = PipelineErrorHandler(session_id=self.session_id)
        self.telemetry = PipelineTelemetryTracker(session_id=self.session_id)
        self.timeline = SessionTimeline(session_id=self.session_id)
        self.observer = BehaviourObserver(self.policy)
        self.annotator = EvidenceAnnotator()

        # --- Detection & Analysis Coordinator (Stages 4-6b) ---
        self._stages = StageCoordinator(
            config=self.config,
            policy=self.policy,
            error_handler=self.error_handler,
            face_detector=face_detector,
            face_verifier=face_verifier,
            object_detector=object_detector,
            relevance_filter=relevance_filter,
            facial_dynamics=facial_dynamics,
            hand_analyzer=hand_analyzer,
            wearable_detector=wearable_detector,
        )

        # --- Temporal Aggregation (Stage 7) ---
        self.temporal_aggregator = self._build_temporal_aggregator()

        # --- Evidence Management (Stages 8-9) ---
        self.evidence_manager = self._build_evidence_manager()
        self._evidence = EvidenceCoordinator(
            config=self.config,
            evidence_manager=self.evidence_manager,
            annotator=self.annotator,
            observer=self.observer,
        )

        # --- Provenance Metadata ---
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
            model_file=getattr(self._stages.object_detector, "model_name", "yolo11n.pt"),
            device=getattr(self._stages.object_detector, "device", "cpu"),
            confidence_threshold=self.config.object_confidence_threshold,
        )
        self.behaviour_detector_info = DetectorInfo(
            name="MediaPipe Face+Hand Landmarker",
            version="tasks-1.0",
            model_file="face_landmarker.task + hand_landmarker.task",
            confidence_threshold=self.policy.speech_articulation_amplitude,
        )
        self.wearable_detector_info = DetectorInfo(
            name="YOLO-World (open vocabulary)",
            version="v8s-world",
            model_file=self.config.wearable_model,
            confidence_threshold=self.policy.earbud_confidence_threshold,
        )
        self.session_control_info = DetectorInfo(
            name="Session Control",
            version="1.0",
            device="operator",
        )
        self.browser_bridge_info = DetectorInfo(
            name="Browser Proctoring Bridge",
            version="1.0",
            device="client",
        )

        # --- Session State ---
        self.state: EngineState = EngineState.CREATED
        self._reported_detector_failures: set[str] = set()
        self.is_active: bool = False
        self.started_at_iso: str | None = None
        self._start_perf: float | None = None
        self._frame_counter: int = 0

        # --- Durable Persistence & Recovery Journal ---
        self.journal = SessionJournalManager(self.evidence_manager.package_dir)
        self._sequence_number: int = 0
        self._persisted_events_count: int = 0
        self.exam_id: str = getattr(self.config, "exam_id", "default_exam")
        self.candidate_id: str = getattr(self.config, "candidate_id", "default_candidate")

    # ------------------------------------------------------------------
    # Detector Property Accessors
    # ------------------------------------------------------------------

    @property
    def face_detector(self) -> FaceDetector | None:
        return self._stages.face_detector

    @face_detector.setter
    def face_detector(self, val: FaceDetector | None) -> None:
        self._stages.face_detector = val

    @property
    def face_verifier(self) -> FaceVerifier | None:
        return self._stages.face_verifier

    @face_verifier.setter
    def face_verifier(self, val: FaceVerifier | None) -> None:
        self._stages.face_verifier = val

    @property
    def object_detector(self) -> ObjectDetector | None:
        return self._stages.object_detector

    @object_detector.setter
    def object_detector(self, val: ObjectDetector | None) -> None:
        self._stages.object_detector = val

    @property
    def relevance_filter(self) -> ObjectRelevanceFilter | None:
        return self._stages.relevance_filter

    @relevance_filter.setter
    def relevance_filter(self, val: ObjectRelevanceFilter | None) -> None:
        self._stages.relevance_filter = val

    @property
    def facial_dynamics(self) -> FacialDynamicsAnalyzer | None:
        return self._stages.facial_dynamics

    @facial_dynamics.setter
    def facial_dynamics(self, val: FacialDynamicsAnalyzer | None) -> None:
        self._stages.facial_dynamics = val

    @property
    def hand_analyzer(self) -> HandAnalyzer | None:
        return self._stages.hand_analyzer

    @hand_analyzer.setter
    def hand_analyzer(self, val: HandAnalyzer | None) -> None:
        self._stages.hand_analyzer = val

    @property
    def wearable_detector(self) -> WearableDetector | None:
        return self._stages.wearable_detector

    @wearable_detector.setter
    def wearable_detector(self, val: WearableDetector | None) -> None:
        self._stages.wearable_detector = val

    # ------------------------------------------------------------------
    # Session Lifecycle
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        """Begin a session, resetting all accumulators."""
        self.state = EngineState.RUNNING
        self.is_active = True
        self.started_at_iso = datetime.now(timezone.utc).isoformat()
        self._start_perf = time.perf_counter()
        self._frame_counter = 0
        self._reported_detector_failures = set()

        self._stages.reset()
        self._evidence.reset()
        self.observer.reset()

        self.temporal_aggregator = self._build_temporal_aggregator()
        self.timeline = SessionTimeline(session_id=self.session_id)
        self.telemetry = PipelineTelemetryTracker(session_id=self.session_id)
        self.error_handler = PipelineErrorHandler(session_id=self.session_id)
        self._stages.error_handler = self.error_handler
        self.evidence_manager = self._build_evidence_manager()
        self._evidence.evidence_manager = self.evidence_manager

        self.journal = SessionJournalManager(self.evidence_manager.package_dir)
        self._sequence_number = 0
        self._persisted_events_count = 0
        self._write_checkpoint(EngineState.RUNNING)

        LOGGER.info("Session %s started (strictness=%s)", self.session_id, self.policy.level.value)

    def pause(self, reason: str = "Proctor pause") -> None:
        """Temporarily pause session frame processing."""
        if self.state in (EngineState.FINALIZED, EngineState.CANCELLED):
            LOGGER.warning("Cannot pause session %s in state %s", self.session_id, self.state.value)
            return
        self.state = EngineState.PAUSED
        self._record_session_control(EventType.SESSION_PAUSED, reason)
        self._persist_new_events()
        self._write_checkpoint(EngineState.PAUSED)
        LOGGER.info("Session %s paused: %s", self.session_id, reason)

    def resume(self, reason: str = "Proctor resume") -> None:
        """Resume session frame processing after a pause."""
        if self.state not in (EngineState.PAUSED, EngineState.RECOVERY_REQUIRED):
            LOGGER.warning(
                "Session %s is not paused or recovering (current state: %s)",
                self.session_id,
                self.state.value,
            )
            return
        self.state = EngineState.RUNNING
        self.is_active = True
        self._record_session_control(EventType.SESSION_RESUMED, reason)
        self._persist_new_events()
        self._write_checkpoint(EngineState.RUNNING)
        LOGGER.info("Session %s resumed: %s", self.session_id, reason)

    def _record_detector_failure(
        self, stage: str, timestamp_seconds: float, frame_index: int
    ) -> None:
        """Record a detector fault as a technical diagnostic on the event record.

        Emitted at most once per detector per session: a model that fails usually
        fails on every frame, and one incident describing the outage is more useful
        to a reviewer than hundreds of identical entries.
        """
        if stage in self._reported_detector_failures:
            return
        self._reported_detector_failures.add(stage)
        try:
            self.temporal_aggregator.record_instant_event(
                event_type=EventType.DETECTOR_ERROR,
                timestamp=timestamp_seconds,
                frame_index=frame_index,
                description=(
                    f"Detector '{stage}' failed during this session. Observations that "
                    f"depend on it are unavailable — absence of them is not evidence "
                    f"that nothing occurred."
                ),
                detector=self.session_control_info,
            )
            self._persist_new_events()
        except Exception as exc:  # pragma: no cover - diagnostics must never abort a session
            LOGGER.warning("Could not record detector failure for %s: %s", stage, exc)

    def _record_session_control(self, event_type: EventType, reason: str) -> None:
        """Put a pause or resume onto the event record.

        Without this a paused period is an unexplained hole in the timeline, and a
        reviewer cannot tell it apart from a camera failure or a candidate who
        walked away. These are technical diagnostics, never misconduct.
        """
        if not self.is_active and self.state != EngineState.PAUSED:
            return
        elapsed = (time.perf_counter() - self._start_perf) if self._start_perf else 0.0
        try:
            self.temporal_aggregator.record_instant_event(
                event_type=event_type,
                timestamp=elapsed,
                frame_index=self._frame_counter,
                description=f"{event_type.value.replace('_', ' ').capitalize()}: {reason}",
                detector=self.session_control_info,
            )
        except Exception as exc:  # pragma: no cover - bookkeeping must never abort a session
            LOGGER.warning("Could not record %s: %s", event_type.value, exc)

    def cancel(self, reason: str = "Session cancelled") -> None:
        """Abort session without sealing final evidence package."""
        self.state = EngineState.CANCELLED
        self.is_active = False
        self._write_checkpoint(EngineState.CANCELLED)
        LOGGER.info("Session %s cancelled: %s", self.session_id, reason)

    def _write_checkpoint(
        self,
        state: EngineState,
        last_frame_index: int = 0,
        last_timestamp_seconds: float = 0.0,
    ) -> None:
        """Atomically persist session recovery checkpoint to disk."""
        try:
            processed = (
                self.telemetry.processed_frames
                if hasattr(self.telemetry, "processed_frames")
                else self._frame_counter
            )
            checkpoint = SessionCheckpoint(
                schema_version="1.0",
                session_id=self.session_id,
                student_name=self.student_name,
                state=state.value,
                started_at_iso=self.started_at_iso or datetime.now(timezone.utc).isoformat(),
                last_checkpoint_utc=datetime.now(timezone.utc).isoformat(),
                last_frame_index=last_frame_index or self._frame_counter,
                last_timestamp_seconds=last_timestamp_seconds,
                processed_frames=processed,
                total_events=len(self.temporal_aggregator.closed_events),
                sequence_number=self._sequence_number,
                exam_id=self.exam_id,
                candidate_id=self.candidate_id,
                models_info={
                    "face_detector": self.face_detector_info.to_dict(),
                    "face_verifier": self.face_verifier_info.to_dict(),
                    "object_detector": self.object_detector_info.to_dict(),
                },
                processing_config=self.config.to_dict(),
            )
            self.journal.write_checkpoint(checkpoint)
        except Exception as exc:
            LOGGER.warning("Failed to write checkpoint for session %s: %s", self.session_id, exc)

    def _persist_new_events(self) -> list[EventRecord]:
        """Progressively persist any newly closed events to disk and append to journal."""
        total_closed = len(self.temporal_aggregator.closed_events)
        if total_closed <= self._persisted_events_count:
            return []
        new_events = self.temporal_aggregator.closed_events[
            self._persisted_events_count : total_closed
        ]
        for ev in new_events:
            self._sequence_number += 1
            ev.sequence_number = self._sequence_number
            if self.config.capture_evidence:
                self._evidence.attach_evidence_to_events([ev])
            self.journal.append_event(ev)
        self._persisted_events_count = total_closed
        return new_events

    def calibrate_gaze(
        self,
        calibration_frames: list[np.ndarray],
    ) -> dict[str, Any]:
        """Collect neutral gaze baseline samples from calibration frames."""
        prev_state = self.state
        self.state = EngineState.CALIBRATING
        samples: list[tuple[float, float]] = []

        gaze_tracker = None
        if self._stages.facial_dynamics and hasattr(self._stages.facial_dynamics, "gaze_tracker"):
            gaze_tracker = self._stages.facial_dynamics.gaze_tracker

        if gaze_tracker is None:
            self.state = prev_state
            return {"calibrated": False, "reason": "Gaze tracker not available"}

        for frame in calibration_frames:
            if frame is None or frame.size == 0:
                continue
            fd = self._stages.face_detector
            if fd is None:
                continue
            detection = fd.detect(frame)
            if detection.count == 0:
                # Only sample frames where a face was actually found; a baseline
                # averaged over empty frames would be meaningless.
                continue
            if self._stages.facial_dynamics is not None:
                # analyze() takes a timestamp, not a bounding box. Passing the box
                # here made calibration raise on its first frame, so per-candidate
                # gaze calibration could never run.
                res = self._stages.facial_dynamics.analyze(frame)
                if res.gaze and res.gaze.confidence > 0.5:
                    samples.append(
                        (
                            res.gaze.raw_horizontal or res.gaze.horizontal,
                            res.gaze.raw_vertical or res.gaze.vertical,
                        )
                    )

        result = gaze_tracker.calibrate(samples)
        self.state = prev_state if prev_state != EngineState.CALIBRATING else EngineState.RUNNING
        return result

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
        """Adaptive sampling rate for the next frame."""
        if not self.config.enable_adaptive_sampling:
            return self.config.sampling_fps
        return (
            self.config.active_fps
            if self.temporal_aggregator.active_incidents
            else self.config.idle_fps
        )

    # ------------------------------------------------------------------
    # Per-Frame Processing (Stages 2-8)
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray | None,
        frame_index: int | None = None,
        timestamp_seconds: float | None = None,
    ) -> FrameObservation:
        """Run one frame through stages 3-7 and return the structured observation."""
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

        if self.state == EngineState.PAUSED:
            obs.accepted = False
            obs.rejection_reason = "SESSION_PAUSED"
            timing.total_frame_ms = (time.perf_counter() - t_start) * 1000.0
            return obs

        if not self.is_active or self.state == EngineState.CREATED:
            self.start_session()

        # --- Stage 3: Frame Validation & Preprocessing ---
        t0 = time.perf_counter()
        gate = self.quality_gate.process(frame, timestamp_seconds=timestamp_seconds)
        timing.preprocessing_ms = (time.perf_counter() - t0) * 1000.0

        obs.was_enhanced = gate.was_enhanced
        obs.mean_luminance = gate.mean_luminance
        obs.blur_variance = gate.blur_variance
        obs.camera_health = gate.camera_health

        if not gate.accepted or gate.frame is None:
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
        self._evidence.record_accepted_frame(working_frame, frame_index)

        # --- Stages 4-5: Face Detection & Verification ---
        faces: list[FaceDetection] = []
        if self.config.enable_face_detection and self.face_detector is not None:
            faces = self._stages.detect_faces(
                working_frame, timestamp_seconds, frame_index, timing, per_model_times
            )
            if faces is None:
                # The detector failed. Nothing is known about who is in frame, so
                # nothing is claimed: the frame stays unmeasured and the fault is
                # recorded against the equipment, not the candidate.
                obs.face_status = FaceStatus.NOT_MEASURED
                obs.detector_failures.append("face_detector")
                self._record_detector_failure("face_detector", timestamp_seconds, frame_index)
            else:
                obs.face_count = len(faces)
                obs.face_boxes = [f.bbox for f in faces]
                obs.face_confidences = [f.confidence for f in faces]

                self._stages.resolve_identity(
                    working_frame,
                    faces,
                    obs,
                    timestamp_seconds,
                    frame_index,
                    timing,
                    per_model_times,
                )

        # --- Stage 6: Prohibited Object Detection ---
        if self.config.enable_object_detection and self.object_detector is not None:
            obs.prohibited_objects = self._stages.detect_objects(
                working_frame, timestamp_seconds, frame_index, timing, per_model_times
            )

        # --- Stage 6b: Behavioural Analysis ---
        self._stages.analyze_behaviour(working_frame, obs, timing, per_model_times)

        # Fallback presence from face landmarker if YuNet not configured
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

        # --- Stage 7: Temporal Incident Qualification ---
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
            hand_analysis=obs.hand_analysis,
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

        # --- Stage 8 (Part 1): Retain Evidence Frame ---
        if self.config.capture_evidence:
            self._evidence.retain_evidence_frames(
                frame=working_frame,
                frame_index=frame_index,
                obs=obs,
                active_incidents=self.temporal_aggregator.active_incidents,
                closed_events=self.temporal_aggregator.closed_events,
            )

        # Progressive Durable Persistence: flush any closed events to disk immediately
        spooled_events = self._persist_new_events()
        if spooled_events or (self._frame_counter % 25 == 0):
            self._write_checkpoint(
                self.state,
                last_frame_index=frame_index,
                last_timestamp_seconds=timestamp_seconds,
            )

        timing.total_frame_ms = (time.perf_counter() - t_start) * 1000.0
        self.telemetry.record_frame(timing, per_model_times=per_model_times)
        self._append_timeline(obs)
        return obs

    # ------------------------------------------------------------------
    # Browser & Application Events
    # ------------------------------------------------------------------

    def record_browser_event(
        self,
        event_type: EventType | str,
        timestamp_seconds: float,
        description: str,
        frame_index: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> EventRecord:
        """Record a client-reported browser or application event."""
        event = self.temporal_aggregator.record_instant_event(
            event_type=coerce_event_type(event_type),
            timestamp=timestamp_seconds,
            frame_index=frame_index,
            description=description,
            detector=self.browser_bridge_info,
            metadata=metadata,
        )
        self._persist_new_events()
        return event

    # ------------------------------------------------------------------
    # Offline Video Execution
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
    # Finalisation & Packaging (Stages 8-11)
    # ------------------------------------------------------------------

    def finalize_session(self) -> SessionSummary:
        """Close open incidents, attach and validate evidence, and seal the package."""
        self.state = EngineState.FINALIZING
        self._write_checkpoint(EngineState.FINALIZING)
        self.is_active = False
        ended_at_iso = datetime.now(timezone.utc).isoformat()

        # 1. Close incidents
        events = self.temporal_aggregator.flush()

        # 2. Persist any newly flushed events to journal and attach evidence
        self._persist_new_events()

        # 3. Validate evidence on disk
        validation_report = self._evidence.validate_all(
            events, prune_invalid=self.config.prune_invalid_evidence
        )

        # 4. Generate Telemetry & Timeline summaries
        telemetry_report = self.telemetry.generate_report()
        timeline_summary = self.timeline.summarise()

        # 5. Build tamper-evident evidence package
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

        # Cleanup memory buffers & analyzers
        self._evidence.clear()
        self.close_analyzers()

        self.state = EngineState.FINALIZED
        self._write_checkpoint(EngineState.FINALIZED)

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

    def close_analyzers(self) -> None:
        """Release the behavioural analyzer models held by this engine."""
        self._stages.close()

    def _append_timeline(self, obs: FrameObservation) -> None:
        """Record the frame's real observation on the session timeline (stage 10)."""
        if not self.config.record_timeline:
            return
        entry = TimelineEntry(
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
            gaze_direction=obs.gaze.direction.value
            if obs.gaze
            else (dyn.gaze.direction.value if (dyn and dyn.gaze) else None),
            occlusion_state=obs.occlusion.state.value if obs.occlusion else None,
            detected_wearables=obs.wearable_names,
        )
        self.timeline.append(entry)
        self.journal.append_timeline_entry(entry.to_dict())

    @staticmethod
    def _to_xywh(bbox: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
        return StageCoordinator.to_xywh(bbox)

    @staticmethod
    def _to_xyxy(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return StageCoordinator.to_xyxy(bbox)

    def _attach_evidence(self, events: list[EventRecord]) -> None:
        """Compatibility helper delegating to EvidenceCoordinator."""
        self._evidence.attach_evidence_to_events(events)

    def _retain_evidence_frames(
        self,
        frame: np.ndarray,
        frame_index: int,
        obs: FrameObservation | None = None,
    ) -> None:
        """Compatibility helper delegating to EvidenceCoordinator."""
        self._evidence.retain_evidence_frames(
            frame=frame,
            frame_index=frame_index,
            obs=obs,
            active_incidents=self.temporal_aggregator.active_incidents,
            closed_events=self.temporal_aggregator.closed_events,
        )

    def _referenced_frame_indices(self) -> set[int]:
        """Compatibility helper delegating to EvidenceCoordinator."""
        return self._evidence.referenced_frame_indices(
            active_incidents=self.temporal_aggregator.active_incidents,
            closed_events=self.temporal_aggregator.closed_events,
        )

    def _attach_review_snapshot(
        self,
        event: EventRecord,
        frame: np.ndarray,
        frame_index: int,
        timestamp_seconds: float,
    ) -> None:
        """Compatibility helper delegating to EvidenceCoordinator."""
        self._evidence.attach_review_snapshot(
            event=event,
            frame=frame,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
        )

    # ------------------------------------------------------------------
    # Mid-Session Crash Recovery
    # ------------------------------------------------------------------

    @classmethod
    def recover_session(cls, session_path: str | Path) -> "ProctoringEngine":
        """Recover an interrupted or crashed session from its durable checkpoint and journals."""
        path = Path(session_path)
        session_dir = path if path.is_dir() else path.parent
        journal = SessionJournalManager(session_dir)
        checkpoint = journal.load_checkpoint()
        if checkpoint is None:
            raise FileNotFoundError(f"No session checkpoint found in {session_dir}")

        cfg_dict = checkpoint.processing_config
        config = SessionConfig.from_dict(cfg_dict) if cfg_dict else SessionConfig()
        config.session_id = checkpoint.session_id
        config.student_name = checkpoint.student_name

        engine = cls(config=config)
        engine.state = EngineState.RECOVERY_REQUIRED
        engine.started_at_iso = checkpoint.started_at_iso
        engine._sequence_number = checkpoint.sequence_number
        engine._frame_counter = checkpoint.last_frame_index
        engine.journal = journal

        # Restore closed events from append-only journal
        persisted_events = journal.read_events()
        engine.temporal_aggregator.closed_events = persisted_events
        engine._persisted_events_count = len(persisted_events)

        # Restore timeline entries
        for t_entry in journal.read_timeline():
            engine.timeline.entries.append(TimelineEntry.from_dict(t_entry))

        LOGGER.info(
            "Session %s recovered successfully: state=%s, %d events, %d frames",
            checkpoint.session_id,
            engine.state.value,
            len(persisted_events),
            engine._frame_counter,
        )
        return engine

    def finalize_recovered_session(self) -> SessionSummary:
        """Seal an evidence package directly from recovered checkpoint and journal state."""
        return self.finalize_session()
