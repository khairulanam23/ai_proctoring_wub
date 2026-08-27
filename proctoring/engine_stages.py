"""Pipeline stage execution coordinator for face, object, and behavioural detectors.

Isolates individual detector error boundaries, multi-face verification matching,
decimated wearable sweeps, and MediaPipe landmarker execution from the main
ProctoringEngine orchestrator.
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any

import numpy as np

from proctoring.analysis.facial_dynamics import FacialDynamicsAnalyzer
from proctoring.analysis.hands import HandAnalyzer
from proctoring.analysis.occlusion import FaceOcclusionClassifier
from proctoring.analysis.policy import ExamPolicy
from proctoring.analysis.wearables import WearableAnalysisResult, WearableDetector
from proctoring.config import SessionConfig
from proctoring.core.errors import ErrorCategory, PipelineErrorHandler
from proctoring.detection.face_detector import DetectionResult, FaceDetection, FaceDetector
from proctoring.detection.face_verifier import FaceVerifier
from proctoring.detection.object_detector import ObjectDetectionResult, ObjectDetector
from proctoring.detection.object_relevance import ObjectRelevanceFilter
from proctoring.observation import FaceStatus, FrameObservation
from proctoring.telemetry.performance import FrameTimingRecord

LOGGER = logging.getLogger(__name__)


class StageCoordinator:
    """Coordinates detection and analysis across stages 4, 5, 6, and 6b."""

    def __init__(
        self,
        config: SessionConfig,
        policy: ExamPolicy | None = None,
        error_handler: PipelineErrorHandler | None = None,
        face_detector: FaceDetector | None = None,
        face_verifier: FaceVerifier | None = None,
        object_detector: ObjectDetector | None = None,
        relevance_filter: ObjectRelevanceFilter | None = None,
        facial_dynamics: FacialDynamicsAnalyzer | None = None,
        hand_analyzer: HandAnalyzer | None = None,
        wearable_detector: WearableDetector | None = None,
    ) -> None:
        self.config = config
        self.policy = (
            policy
            if policy is not None
            else ExamPolicy.for_level(config.strictness, config.exam_mode)
        )
        self.error_handler = (
            error_handler if error_handler is not None else PipelineErrorHandler(config.session_id)
        )

        self.face_detector = face_detector
        self.face_verifier = face_verifier
        self.object_detector = object_detector
        self.relevance_filter: ObjectRelevanceFilter | None = (
            relevance_filter
            or ObjectRelevanceFilter(
                class_thresholds={
                    "cell phone": self.config.phone_confidence_threshold,
                    "book": self.config.book_confidence_threshold,
                },
                default_threshold=self.config.object_confidence_threshold,
            )
        )

        self.facial_dynamics = facial_dynamics
        if self.facial_dynamics is None and self.config.enable_facial_dynamics:
            self.facial_dynamics = FacialDynamicsAnalyzer(
                model_path=self.config.facial_dynamics_model,
                # The articulation window is configured in seconds and converted here,
                # so it keeps the same duration whatever the sampling rate is set to.
                speech_window_frames=self.speech_window_frames,
                speech_articulation_amplitude=self.policy.speech_articulation_amplitude,
                speech_min_crossings=self.policy.speech_min_crossings,
                speech_closed_ratio=self.policy.speech_closed_ratio,
                sampling_fps=self.config.sampling_fps,
                speech_min_sampling_fps=self.policy.speech_min_sampling_fps,
                yaw_limit_degrees=self.policy.yaw_limit_degrees,
                pitch_limit_degrees=self.policy.pitch_limit_degrees,
                gaze_offset_limit=self.policy.gaze_offset_limit,
                blink_threshold=self.policy.blink_threshold,
                liveness_grace_seconds=self.policy.liveness_grace_seconds,
                ear_region_padding_ratio=self.policy.ear_region_padding_ratio,
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
                enable_ear_region_zoom=self.policy.enable_ear_region_zoom,
                ear_roi_target_px=self.policy.ear_roi_target_px,
            )

        self.occlusion_classifier = FaceOcclusionClassifier()

        self._last_wearable_result: WearableAnalysisResult | None = None
        self._last_wearable_timestamp: float | None = None

    @property
    def _wearable_accepts_hand_at_ear(self) -> bool:
        """Does the injected wearable detector take the hand-corroboration argument?

        The engine accepts a caller-supplied detector, and a deployment holding one
        written against the previous signature must keep working rather than have
        every sweep fail as a detector fault. Checked rather than assumed, in the
        same spirit as the two accepted face-verifier API shapes.
        """
        detect = getattr(self.wearable_detector, "detect", None)
        if detect is None:
            return False
        try:
            return "hand_at_ear" in inspect.signature(detect).parameters
        except (TypeError, ValueError):
            return False

    @property
    def speech_window_frames(self) -> int:
        """Articulation window length in frames, derived from the configured rate.

        Expressed in seconds in the policy because that is what the measurement
        actually depends on: at a lower sampling rate a fixed frame count would
        silently become a longer window with fewer samples in it.
        """
        frames = round(self.policy.speech_window_seconds * self.config.sampling_fps)
        return max(4, int(frames))

    def reset(self) -> None:
        """Reset internal per-session tracking state."""
        self._last_wearable_result = None
        self._last_wearable_timestamp = None
        if self.facial_dynamics is not None:
            self.facial_dynamics.reset()

    # ------------------------------------------------------------------
    # Stage 4: Face Detection
    # ------------------------------------------------------------------

    def detect_faces(
        self,
        frame: np.ndarray,
        timestamp_seconds: float,
        frame_index: int,
        timing: FrameTimingRecord,
        per_model_times: dict[str, float],
    ) -> list[FaceDetection] | None:
        """Stage 4 — YuNet detection, with background faces filtered out.

        Returns the detected faces, or ``None`` when the detector **failed**.

        The distinction is load-bearing. Returning an empty list on failure makes a
        broken detector indistinguishable from a camera pointed at an empty chair,
        so a crashed model silently produced ``NO_FACE`` against a candidate who was
        sitting right there. A detector fault is a fact about the equipment and must
        never become a fact about the candidate.
        """
        if not self.config.enable_face_detection or self.face_detector is None:
            return []

        try:
            t0 = time.perf_counter()
            result: DetectionResult = self.face_detector.detect(
                frame, score_threshold=self.config.face_score_threshold
            )
            dt = (time.perf_counter() - t0) * 1000.0
            timing.face_detector_ms = dt
            per_model_times["face_detector"] = dt

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
            return None

    # ------------------------------------------------------------------
    # Stage 5: Face Verification
    # ------------------------------------------------------------------

    def resolve_identity(
        self,
        frame: np.ndarray,
        faces: list[FaceDetection],
        obs: FrameObservation,
        timestamp_seconds: float,
        frame_index: int,
        timing: FrameTimingRecord,
        per_model_times: dict[str, float],
    ) -> None:
        """Stage 5 — verify every detected face against all enrolment templates."""
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
            obs.face_status = FaceStatus.MULTIPLE_FACES if len(faces) > 1 else FaceStatus.UNVERIFIED
            return

        scores: list[float | None] = []
        for face in faces:
            scores.append(
                self.score_face(
                    frame, face, templates, timestamp_seconds, frame_index, timing, per_model_times
                )
            )

        measured = [s for s in scores if s is not None]
        obs.face_similarities = scores
        obs.similarity = max(measured) if measured else None

        if not measured:
            obs.face_status = FaceStatus.MULTIPLE_FACES if len(faces) > 1 else FaceStatus.UNVERIFIED
            return

        matched = [s >= self.config.face_match_threshold for s in measured]
        obs.enrolled_face_present = any(matched)
        obs.unknown_face_count = sum(1 for m in matched if not m)
        obs.identity_verified = obs.enrolled_face_present

        if len(faces) > 1:
            obs.face_status = FaceStatus.MULTIPLE_FACES
        elif obs.enrolled_face_present:
            obs.face_status = FaceStatus.ENROLLED
        elif obs.similarity is not None and obs.similarity >= (
            self.config.face_match_threshold - 0.08
        ):
            obs.face_status = FaceStatus.IDENTITY_UNCERTAIN
        elif obs.similarity is not None and obs.similarity < (
            self.config.face_match_threshold - 0.15
        ):
            obs.face_status = FaceStatus.IDENTITY_MISMATCH
        else:
            obs.face_status = FaceStatus.UNKNOWN_FACE

    def score_face(
        self,
        frame: np.ndarray,
        face: FaceDetection,
        templates: list[np.ndarray],
        timestamp_seconds: float,
        frame_index: int,
        timing: FrameTimingRecord,
        per_model_times: dict[str, float],
    ) -> float | None:
        """Best cosine similarity of one face against all enrolment templates."""
        try:
            t0 = time.perf_counter()
            if self.face_verifier is None:
                return None
            embedding = self.extract_embedding(frame, face)
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

    def extract_embedding(self, frame: np.ndarray, face: FaceDetection) -> np.ndarray | None:
        """Obtain an SFace embedding, tolerating either verifier API shape."""
        if self.face_verifier is None:
            return None
        if hasattr(self.face_verifier, "extract_embedding"):
            return self.face_verifier.extract_embedding(frame, raw_detection=face.raw_detection)
        if hasattr(self.face_verifier, "extract_feature"):
            return self.face_verifier.extract_feature(frame, face=face)
        return None

    # ------------------------------------------------------------------
    # Stage 6: Object Detection
    # ------------------------------------------------------------------

    def detect_objects(
        self,
        frame: np.ndarray,
        timestamp_seconds: float,
        frame_index: int,
        timing: FrameTimingRecord,
        per_model_times: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Stage 6 — YOLO detection reduced to proctoring-relevant objects."""
        if not self.config.enable_object_detection or self.object_detector is None:
            return []

        try:
            t0 = time.perf_counter()
            result: ObjectDetectionResult = self.object_detector.detect(frame)
            relevant_objs = (
                self.relevance_filter.filter(result).relevant_objects
                if self.relevance_filter is not None
                else result.objects
            )
            dt = (time.perf_counter() - t0) * 1000.0
            timing.object_detector_ms = dt
            per_model_times["object_detector"] = dt

            return [
                {
                    "class_name": obj.class_name,
                    "confidence": obj.confidence,
                    "bbox": obj.bbox,
                }
                for obj in relevant_objs
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

    # ------------------------------------------------------------------
    # Stage 6b: Behavioural Analysis
    # ------------------------------------------------------------------

    def analyze_behaviour(
        self,
        frame: np.ndarray,
        obs: FrameObservation,
        timing: FrameTimingRecord,
        per_model_times: dict[str, float],
    ) -> None:
        """Stage 6b — measure hands, speech articulation, gaze, and worn devices."""
        behaviour_start = time.perf_counter()

        # 1. Facial dynamics: speaking, head pose, gaze
        if self.facial_dynamics is not None and self.facial_dynamics.is_available:
            try:
                obs.facial_dynamics = self.facial_dynamics.analyze(
                    frame, timestamp_seconds=obs.timestamp_seconds
                )
                if obs.facial_dynamics.gaze is not None:
                    obs.gaze = obs.facial_dynamics.gaze
                per_model_times["facial_dynamics"] = obs.facial_dynamics.inference_ms
            except Exception as exc:
                self.record_analyzer_failure(exc, "facial_dynamics", obs)

        # 2. Hands, related to the face region when found
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
                self.record_analyzer_failure(exc, "hand_analysis", obs)

        # 3. Face Occlusion & Camera Obstruction classification
        try:
            face_box = None
            if obs.face_boxes:
                b = obs.face_boxes[0]
                face_box = self.to_xyxy(b)
            elif obs.facial_dynamics and obs.facial_dynamics.face_bbox:
                face_box = obs.facial_dynamics.face_bbox

            obs.occlusion = self.occlusion_classifier.classify(
                face_found=bool(obs.face_count and obs.face_count > 0)
                or bool(obs.facial_dynamics and obs.facial_dynamics.face_found),
                face_bbox=face_box,
                landmarks=obs.facial_dynamics.landmarks if obs.facial_dynamics else None,
                hands=obs.hand_analysis,
                mean_luminance=obs.mean_luminance,
                blur_variance=obs.blur_variance,
            )
        except Exception as exc:
            LOGGER.debug("Occlusion classification failed at frame %s: %s", obs.frame_index, exc)

        # 4. Worn devices on a decimated cadence
        if (
            self.should_run_wearable_detection(obs.frame_index)
            and self.wearable_detector is not None
        ):
            try:
                dynamics = obs.facial_dynamics
                hands = obs.hand_analysis
                kwargs: dict[str, Any] = {
                    "ear_regions": dynamics.ear_regions if dynamics else None,
                    "face_bbox": dynamics.face_bbox if dynamics else None,
                }
                if self._wearable_accepts_hand_at_ear:
                    # The hand analyzer already ran for this frame; its independent
                    # view of a hand at the ear strengthens a marginal earpiece call
                    # without ever creating one on its own.
                    kwargs["hand_at_ear"] = bool(hands is not None and hands.hand_near_ear)
                result = self.wearable_detector.detect(frame, **kwargs)
                obs.wearables = result
                self._last_wearable_result = result
                self._last_wearable_timestamp = obs.timestamp_seconds
                per_model_times["wearable_detector"] = result.inference_ms
            except Exception as exc:
                self.record_analyzer_failure(exc, "wearable_detector", obs)
        elif self._last_wearable_result is not None:
            last = self._last_wearable_timestamp
            age = (obs.timestamp_seconds - last) if last is not None else 0.0
            if age <= self.wearable_carry_forward_seconds:
                obs.wearables = self._last_wearable_result
            else:
                LOGGER.debug("Discarding stale wearable reading (%.1fs old)", age)
                self._last_wearable_result = None
                self._last_wearable_timestamp = None

        timing.behaviour_analysis_ms = (time.perf_counter() - behaviour_start) * 1000.0

    @property
    def wearable_carry_forward_seconds(self) -> float:
        """How long a wearable reading stays valid: two sweep intervals plus slack."""
        interval = max(1, int(self.config.wearable_detection_interval_frames))
        return (interval * 2.0) / max(0.1, self.config.sampling_fps)

    def should_run_wearable_detection(self, frame_index: int) -> bool:
        """Check if this frame index is due for wearable detection sweep."""
        if self.wearable_detector is None or not self.wearable_detector.is_available:
            return False
        interval = max(1, int(self.config.wearable_detection_interval_frames))
        return frame_index % interval == 0

    def record_analyzer_failure(self, error: Exception, stage: str, obs: FrameObservation) -> None:
        """Log an analyzer fault as a system diagnostic, never as a candidate event."""
        LOGGER.warning("Analyzer %s failed at frame %s: %s", stage, obs.frame_index, error)
        self.error_handler.handle_exception(
            error=error,
            category=ErrorCategory.MODEL_INFERENCE_FAILURE,
            timestamp_seconds=obs.timestamp_seconds,
            frame_index=obs.frame_index,
            details={"pipeline_stage": stage},
        )

    def close(self) -> None:
        """Release any model resources held by analyzers."""
        for analyzer in (self.facial_dynamics, self.hand_analyzer):
            if analyzer is not None:
                try:
                    analyzer.close()
                except Exception as exc:
                    LOGGER.debug("Analyzer close failed: %s", exc)

    @staticmethod
    def to_xywh(bbox: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
        """Convert (x1, y1, x2, y2) bounding box to (x, y, w, h)."""
        if bbox is None:
            return (0, 0, 0, 0)
        x1, y1, x2, y2 = bbox
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    @staticmethod
    def to_xyxy(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Convert (x, y, w, h) bounding box to (x1, y1, x2, y2)."""
        x, y, w, h = bbox
        return int(x), int(y), int(x + w), int(y + h)
