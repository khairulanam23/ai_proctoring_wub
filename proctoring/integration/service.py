"""The integration boundary an LMS calls: session lifecycle over plain JSON.

:class:`ProctoringService` is the only class a host application needs.  It owns
engine instances keyed by session, accepts frames in the forms a browser actually
sends them (base64 data URL, raw JPEG bytes, or a decoded array), and returns the
dataclasses in :mod:`proctoring.integration.schemas`.

Mapped onto a Moodle ``quizaccess`` plugin:

===============================  ====================================
Moodle hook                      Service call
===============================  ====================================
attempt started                  :meth:`start_session`
enrolment capture (pre-attempt)  :meth:`enrol_candidate`
periodic frame POST from client  :meth:`ingest_frame`
client-side JS proctoring event  :meth:`record_client_event`
attempt submitted                :meth:`finalize_session`
teacher opens the review screen   :meth:`get_review`
===============================  ====================================

The service is deliberately transport-agnostic: no web framework, no HTTP.  Wiring
it to Moodle's external API, a FastAPI app, or a worker queue is the host's choice,
and keeping that out of here is what lets the pipeline be tested without a server.

Concurrency: one lock guards the session registry, and each session's engine is
used from one request at a time.  Frames for a given attempt must therefore be
ingested serially — which is what a browser posting on an interval does anyway.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from proctoring.analysis.policy import StrictnessLevel
from proctoring.config import SessionConfig
from proctoring.core.events import EventRecord, coerce_event_type
from proctoring.engine import ProctoringEngine, SessionSummary
from proctoring.integration.schemas import (
    CONTRACT_VERSION,
    DEFAULT_REVIEWER_GUIDANCE,
    RELIABILITY_NOTES,
    FrameAck,
    ObservationSummary,
    ReviewPayload,
    SessionHandle,
    SessionResult,
    SessionState,
    StartSessionRequest,
)
from proctoring.integration.session_store import SessionRecord, SessionStore

LOGGER = logging.getLogger(__name__)

FrameInput = str | bytes | bytearray | np.ndarray

# A 640x480 JPEG is tens of kilobytes; anything approaching this ceiling is either
# a misconfigured client or an attempt to exhaust memory.
MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_FRAME_PIXELS = 40_000_000


class ProctoringServiceError(RuntimeError):
    """Raised for caller mistakes: unknown session, wrong state, undecodable frame."""


class ProctoringService:
    """Manages proctoring sessions on behalf of a host LMS."""

    def __init__(
        self,
        output_dir: str | Path = "data/results/lms_sessions",
        store: SessionStore | None = None,
        detector_factory: Any | None = None,
        default_strictness: StrictnessLevel = StrictnessLevel.STANDARD,
        session_idle_timeout_seconds: float = 1800.0,
    ) -> None:
        """
        Args:
            detector_factory: Optional callable returning the detector bundle to
                inject into each engine.  Supplying one lets a deployment load the
                models once at start-up and share them, instead of paying the load
                cost per attempt.  Left unset, each session builds its own.
            session_idle_timeout_seconds: How long a session may receive no frames
                before :meth:`reap_idle_sessions` finalises it. Without this an
                abandoned attempt — a closed browser, a crashed tab — pins its
                engine, its models and up to a few hundred megabytes of retained
                evidence frames in memory for the lifetime of the process.
        """
        self.output_dir = Path(output_dir)
        self.store = store or SessionStore()
        self.detector_factory = detector_factory
        self.default_strictness = StrictnessLevel(default_strictness)

        self.session_idle_timeout_seconds = float(session_idle_timeout_seconds)

        self._engines: dict[str, ProctoringEngine] = {}
        self._detector_cache: dict[str, Any] | None = None
        self._lock = threading.RLock()
        # One lock per session. Concurrent requests for the same attempt would
        # otherwise interleave inside a single engine and corrupt its frame counter
        # and speech history.
        self._session_locks: dict[str, threading.Lock] = {}
        self._last_activity: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_session(self, request: StartSessionRequest | dict[str, Any]) -> SessionHandle:
        """Open a session for one quiz attempt and report what it can observe."""
        if isinstance(request, dict):
            request = StartSessionRequest.from_dict(request)

        # A uuid suffix rather than a timestamp: two starts for the same attempt
        # within one second previously produced identical ids, and the second
        # silently evicted the first session's engine from the registry.
        session_id = f"attempt_{request.attempt_id}_{uuid.uuid4().hex[:12]}"
        config = SessionConfig(
            session_id=session_id,
            student_name=request.candidate_name,
            strictness=StrictnessLevel(request.strictness or self.default_strictness),
            sampling_fps=request.sampling_fps,
            enable_wearable_detection=request.enable_wearable_detection,
            output_dir=self.output_dir,
            create_zip=True,
        )

        detectors = self.detector_factory() if self.detector_factory else self._default_detectors()
        engine = ProctoringEngine(config=config, **detectors)

        # Restore a stored enrolment so identity verification has a template.
        if request.enrolment_id:
            templates = self.store.load_enrolment(request.enrolment_id)
            if templates:
                config.reference_templates = templates
                config.enable_face_verification = True

        engine.start_session()

        with self._lock:
            self._engines[session_id] = engine
            self._session_locks[session_id] = threading.Lock()
            self._last_activity[session_id] = time.monotonic()
            record = SessionRecord(
                session_id=session_id,
                attempt_id=request.attempt_id,
                user_id=request.user_id,
                course_id=request.course_id,
                quiz_id=request.quiz_id,
                candidate_name=request.candidate_name,
                state=SessionState.ACTIVE,
                strictness=config.strictness.value,
                started_at_utc=datetime.now(timezone.utc).isoformat(),
                metadata=dict(request.metadata),
            )
            self.store.put(record)

        LOGGER.info(
            "Session %s started for attempt %s (strictness=%s)",
            session_id,
            request.attempt_id,
            config.strictness.value,
        )
        active, unavailable = self._detector_availability(engine)
        return SessionHandle(
            session_id=session_id,
            attempt_id=request.attempt_id,
            user_id=request.user_id,
            state=SessionState.ACTIVE,
            strictness=config.strictness.value,
            active_detectors=active,
            unavailable_detectors=unavailable,
            identity_verification_enabled=config.enable_face_verification
            and config.has_reference_identity,
        )

    def ingest_frame(
        self,
        session_id: str,
        frame: FrameInput,
        timestamp_seconds: float | None = None,
    ) -> FrameAck:
        """Process one captured frame and return the live state for the client.

        Accepts a base64 data URL (what a browser canvas produces), raw encoded
        bytes, or an already-decoded BGR array.
        """
        engine = self._require_engine(session_id)
        decoded = self._decode_frame(frame)

        with self._session_lock(session_id):
            self._last_activity[session_id] = time.monotonic()
            observation = engine.process_frame(decoded, timestamp_seconds=timestamp_seconds)
        dynamics = observation.facial_dynamics

        return FrameAck(
            session_id=session_id,
            frame_index=observation.frame_index,
            timestamp_seconds=observation.timestamp_seconds,
            accepted=observation.accepted,
            rejection_reason=observation.rejection_reason,
            face_count=observation.face_count,
            face_status=observation.face_status if observation.accepted else None,
            identity_verified=observation.identity_verified,
            hands_detected=observation.hands_detected,
            is_speaking=dynamics.is_speaking if dynamics else None,
            is_looking_away=dynamics.is_looking_away if dynamics else None,
            detected_objects=observation.prohibited_object_names,
            detected_wearables=observation.wearable_names,
            active_observations=observation.active_event_types,
            engine_state=engine.state.value,
            processing_latency_ms=observation.timing.total_frame_ms if observation.timing else 0.0,
            next_frame_due_in_seconds=1.0 / max(0.1, engine.target_fps),
        )

    def record_client_event(
        self,
        session_id: str,
        event_type: str,
        timestamp_seconds: float,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a browser-side proctoring event (tab switch, blur, fullscreen exit).

        The client sends a plain string; unknown names are recorded as unclassified
        rather than dropped, so a newer client against an older server still gets
        its signal onto the timeline.
        """
        engine = self._require_engine(session_id)
        with self._session_lock(session_id):
            self._last_activity[session_id] = time.monotonic()
            event = engine.record_browser_event(
                event_type=coerce_event_type(event_type),
                timestamp_seconds=timestamp_seconds,
                description=description or f"Client reported {event_type}",
                metadata=metadata,
            )
        return {
            "session_id": session_id,
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "recorded": True,
        }

    def pause_session(self, session_id: str, reason: str = "Proctor pause") -> dict[str, Any]:
        """Suspend processing for an attempt without ending it.

        Paused frames are rejected before any inference runs, so a paused session
        costs nothing. The pause is recorded on the event timeline as a technical
        diagnostic, so the resulting gap in observations is explained rather than
        left for a reviewer to interpret.
        """
        engine = self._require_engine(session_id)
        with self._session_lock(session_id):
            self._last_activity[session_id] = time.monotonic()
            engine.pause(reason)

        record = self.store.get(session_id)
        if record is not None:
            record.state = SessionState.PAUSED
            self.store.put(record)
        LOGGER.info("Session %s paused: %s", session_id, reason)
        return {
            "session_id": session_id,
            "state": SessionState.PAUSED.value,
            "engine_state": engine.state.value,
            "reason": reason,
        }

    def resume_session(self, session_id: str, reason: str = "Proctor resume") -> dict[str, Any]:
        """Resume processing after a pause."""
        engine = self._require_engine(session_id)
        with self._session_lock(session_id):
            self._last_activity[session_id] = time.monotonic()
            engine.resume(reason)

        record = self.store.get(session_id)
        if record is not None:
            record.state = SessionState.ACTIVE
            self.store.put(record)
        LOGGER.info("Session %s resumed: %s", session_id, reason)
        return {
            "session_id": session_id,
            "state": SessionState.ACTIVE.value,
            "engine_state": engine.state.value,
            "reason": reason,
        }

    def finalize_session(self, session_id: str) -> SessionResult:
        """Close the attempt, seal the evidence package and release the engine."""
        engine = self._require_engine(session_id)
        record = self._require_record(session_id)

        record.state = SessionState.FINALIZING
        self.store.put(record)

        try:
            summary = engine.finalize_session()
        except Exception as exc:
            record.state = SessionState.FAILED
            record.error = str(exc)
            self.store.put(record)
            self._release(session_id)
            LOGGER.exception("Finalisation failed for session %s", session_id)
            raise ProctoringServiceError(f"Failed to finalise session {session_id}: {exc}") from exc

        result = self._build_result(record, summary, engine)
        record.state = SessionState.COMPLETED
        record.ended_at_utc = result.ended_at_utc
        record.package_dir = result.package_dir
        record.result = result.to_dict()
        record.observations = [self._summarise(e).to_dict() for e in summary.events]
        record.timeline_summary = summary.timeline_summary
        record.telemetry_summary = summary.telemetry.to_dict()
        record.policy = engine.policy.to_dict()
        self.store.put(record)

        self._release(session_id)
        LOGGER.info(
            "Session %s finalised: %d observation(s), integrity=%s",
            session_id,
            result.total_observations,
            result.integrity_verified,
        )
        return result

    def abort_session(self, session_id: str, reason: str = "") -> SessionResult:
        """Abandon an attempt, still sealing whatever evidence was gathered.

        A crashed browser or a withdrawn attempt should not destroy the record of
        what happened before it, so this finalises rather than discarding, and marks
        the outcome so nobody mistakes a partial session for a complete one.
        """
        result = self.finalize_session(session_id)
        result.state = SessionState.FAILED
        result.coverage_warnings.append(
            f"Session ended before normal completion{f': {reason}' if reason else ''}. "
            "The record covers only the period captured."
        )
        record = self._require_record(session_id)
        record.state = SessionState.FAILED
        record.error = reason
        record.result = result.to_dict()
        self.store.put(record)
        return result

    # ------------------------------------------------------------------
    # Enrolment
    # ------------------------------------------------------------------

    def enrol_candidate(
        self,
        enrolment_id: str,
        frames: list[FrameInput],
        face_detector: Any | None = None,
        face_verifier: Any | None = None,
    ) -> dict[str, Any]:
        """Build and store reference templates for a candidate.

        Several frames are expected: an enrolment spanning a little pose and
        lighting variation is what keeps ordinary movement during the exam from
        reading as an identity mismatch.

        Embeddings are stored server-side under ``enrolment_id`` and never returned
        to the host — the LMS holds the identifier, not the biometric.
        """
        # Reuse the cached bundle: constructing a FaceVerifier reads a 37 MB ONNX
        # file, far too costly to repeat for every enrolment request.
        bundle = self.detector_factory() if self.detector_factory else self._default_detectors()
        detector = face_detector or bundle.get("face_detector")
        verifier = face_verifier or bundle.get("face_verifier")

        if detector is None or verifier is None:
            from proctoring.detection.face_detector import FaceDetector
            from proctoring.detection.face_verifier import FaceVerifier

            detector = detector or FaceDetector()
            verifier = verifier or FaceVerifier(detector=detector)

        templates: list[np.ndarray] = []
        reference_images: list[np.ndarray] = []
        rejected: list[str] = []

        for index, raw in enumerate(frames):
            try:
                image = self._decode_frame(raw)
                if image is None:
                    rejected.append(f"frame {index}: could not be decoded or exceeded size limits")
                    continue
                detection = detector.detect(image)
                if detection.count != 1:
                    rejected.append(
                        f"frame {index}: expected exactly one face, found {detection.count}"
                    )
                    continue
                templates.append(verifier.extract_feature(image, face=detection.faces[0]))
                reference_images.append(image)
            except Exception as exc:
                rejected.append(f"frame {index}: {exc}")

        if templates:
            # Store the photographs alongside the embeddings: a reference set that
            # can be re-derived and inspected is auditable, one that is only a
            # vector is not.
            self.store.save_enrolment(enrolment_id, templates, images=reference_images)

        return {
            "enrolment_id": enrolment_id,
            "templates_stored": len(templates),
            "frames_rejected": len(rejected),
            "rejection_reasons": rejected,
            "usable": len(templates) >= 1,
            "quality_note": (
                "Enrolment accepted."
                if len(templates) >= 3
                else "Fewer than three usable samples; identity verification will be "
                "less tolerant of pose and lighting change."
            ),
        }

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------

    def get_review(self, session_id: str) -> ReviewPayload:
        """Assemble the invigilator review payload for a completed session."""
        record = self._require_record(session_id)

        # Rebuild from the stored rows, ignoring derived keys. ``to_dict`` emits
        # convenience fields (``is_technical``) that are computed rather than stored,
        # so a naive round-trip through the constructor would fail on them.
        accepted = set(ObservationSummary.__dataclass_fields__)
        observations = [
            ObservationSummary(**{k: v for k, v in row.items() if k in accepted})
            for row in record.observations
        ]

        return ReviewPayload(
            session_id=record.session_id,
            attempt_id=record.attempt_id,
            user_id=record.user_id,
            candidate_name=record.candidate_name,
            result=record.result,
            observations=observations,
            timeline_summary=record.timeline_summary,
            telemetry_summary=record.telemetry_summary,
            policy=record.policy,
            reviewer_guidance=list(DEFAULT_REVIEWER_GUIDANCE),
        )

    def list_sessions(
        self, attempt_id: str | None = None, user_id: str | None = None
    ) -> list[SessionRecord]:
        """List known sessions, optionally filtered by attempt or user."""
        return self.store.list(attempt_id=attempt_id, user_id=user_id)

    def get_state(self, session_id: str) -> SessionState:
        """Current attempt state for a session."""
        return self._require_record(session_id).state

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _default_detectors(self) -> dict[str, Any]:
        """Load the standard detector bundle when the host supplied no factory.

        Models are loaded once and cached on the service, not per attempt: a
        FaceVerifier construction reads a 37 MB ONNX file, which is far too costly
        to repeat for every quiz attempt.

        A model that will not load is omitted rather than fatal — the session then
        reports that capability under ``unavailable_detectors`` so the host can tell
        a proctor what was not watched.
        """
        cached = self._detector_cache
        if cached is not None:
            return cached

        from proctoring.detection.face_detector import FaceDetector
        from proctoring.detection.face_verifier import FaceVerifier

        bundle: dict[str, Any] = {}
        try:
            detector = FaceDetector()
            bundle["face_detector"] = detector
            with contextlib.suppress(Exception):
                bundle["face_verifier"] = FaceVerifier(detector=detector)
        except Exception:
            pass

        self._detector_cache = bundle
        return bundle

    def _session_lock(self, session_id: str) -> threading.Lock:
        """The per-session lock, created on demand for robustness."""
        with self._lock:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[session_id] = lock
            return lock

    def _release(self, session_id: str) -> None:
        """Drop every in-memory resource held for a finished session."""
        with self._lock:
            engine = self._engines.pop(session_id, None)
            self._session_locks.pop(session_id, None)
            self._last_activity.pop(session_id, None)
        if engine is not None:
            engine.close_analyzers()

    def reap_idle_sessions(self, timeout_seconds: float | None = None) -> list[str]:
        """Finalise sessions that have gone silent, freeing their engines.

        A browser that closes mid-attempt never calls ``finalize_session``, so
        without this the engine, its loaded models and its retained evidence frames
        stay resident for the lifetime of the process — a guaranteed leak in any
        real deployment.

        Reaping *finalises* rather than discards: whatever was captured before the
        candidate disappeared is still sealed into a package and marked as an
        abnormal ending, because a partial record is evidence and deleting it is
        not the service's decision to make.

        Call periodically from a scheduler, a cron job, or a background thread.
        """
        cutoff = (
            timeout_seconds if timeout_seconds is not None else self.session_idle_timeout_seconds
        )
        now = time.monotonic()

        with self._lock:
            stale = [
                session_id
                for session_id, last in self._last_activity.items()
                if (now - last) > cutoff
            ]

        reaped: list[str] = []
        for session_id in stale:
            try:
                LOGGER.warning(
                    "Reaping idle session %s after %.0fs of inactivity", session_id, cutoff
                )
                self.abort_session(session_id, reason=f"no activity for {cutoff:.0f}s")
                reaped.append(session_id)
            except Exception as exc:
                LOGGER.error("Failed to reap session %s: %s", session_id, exc)
                self._release(session_id)
        return reaped

    @property
    def active_session_count(self) -> int:
        """Number of sessions currently holding an engine in memory."""
        with self._lock:
            return len(self._engines)

    def _require_record(self, session_id: str) -> SessionRecord:
        """Fetch a session record or fail with a clear error.

        ``store.get`` legitimately returns ``None`` for an unknown or evicted
        session. Reaching through that without checking turned a recoverable
        "unknown session" into an ``AttributeError`` raised from the middle of
        finalisation, which is both harder to diagnose and harder to handle.
        """
        record = self.store.get(session_id)
        if record is None:
            raise ProctoringServiceError(f"Unknown session: {session_id}")
        return record

    def _require_engine(self, session_id: str) -> ProctoringEngine:
        with self._lock:
            engine = self._engines.get(session_id)
        if engine is None:
            state = self.store.get(session_id)
            if state is None:
                raise ProctoringServiceError(f"Unknown session: {session_id}")
            raise ProctoringServiceError(
                f"Session {session_id} is {state.state.value}; it no longer accepts frames."
            )
        return engine

    @staticmethod
    def _decode_frame(frame: FrameInput) -> np.ndarray | None:
        """Turn whatever the client sent into a BGR array.

        Returns ``None`` for an undecodable payload rather than raising: a corrupt
        frame is a capture fault the engine already handles by recording a
        diagnostic and skipping, and one bad POST must not abort an exam.
        """
        import cv2

        if frame is None:
            return None
        if isinstance(frame, np.ndarray):
            # Arrays skip decoding, but they must not skip the size ceiling. An
            # in-process caller — or a host that decodes before handing frames over —
            # could otherwise submit an arbitrarily large raster and drive every
            # detector across it.
            return frame if ProctoringService._within_pixel_budget(frame) else None

        raw: bytes | None = None
        if isinstance(frame, (bytes, bytearray)):
            raw = bytes(frame)
        elif isinstance(frame, str):
            payload = frame.split(",", 1)[1] if frame.startswith("data:") else frame
            # Reject before decoding: base64 expands to about 3/4 its length, so an
            # oversized payload can be refused without ever allocating for it.
            if len(payload) > MAX_FRAME_BYTES * 4 // 3 + 4:
                LOGGER.warning("Rejecting oversized frame payload (%d chars)", len(payload))
                return None
            try:
                raw = base64.b64decode(payload, validate=False)
            except (binascii.Error, ValueError):
                return None

        if not raw:
            return None
        if len(raw) > MAX_FRAME_BYTES:
            LOGGER.warning("Rejecting oversized frame (%d bytes)", len(raw))
            return None

        decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or not decoded.size:
            return None
        # A small compressed file can decode to an enormous raster; check after
        # decoding as well as before.
        return decoded if ProctoringService._within_pixel_budget(decoded) else None

    @staticmethod
    def _within_pixel_budget(frame: np.ndarray) -> bool:
        """Reject rasters large enough to exhaust memory when run through detectors.

        Applied to decoded and pre-decoded frames alike: an array handed straight to
        the service would otherwise skip the ceiling entirely, and a single 9000x9000
        submission is a quarter of a gigabyte before any detector touches it.
        """
        if frame.ndim < 2 or not frame.size:
            return True  # malformed shapes are caught by the quality gate, with a reason
        pixels = int(frame.shape[0]) * int(frame.shape[1])
        if pixels > MAX_FRAME_PIXELS:
            LOGGER.warning("Rejecting frame with %dx%d pixels", frame.shape[1], frame.shape[0])
            return False
        return True

    @staticmethod
    def _detector_availability(engine: ProctoringEngine) -> tuple[list[str], list[str]]:
        """Report which observation capabilities this session actually has."""
        checks = [
            ("face_detection", engine.face_detector is not None),
            (
                "identity_verification",
                engine.face_verifier is not None and engine.config.has_reference_identity,
            ),
            ("object_detection", engine.object_detector is not None),
            (
                "facial_dynamics",
                engine.facial_dynamics is not None and engine.facial_dynamics.is_available,
            ),
            (
                "hand_analysis",
                engine.hand_analyzer is not None and engine.hand_analyzer.is_available,
            ),
            (
                "wearable_detection",
                engine.wearable_detector is not None and engine.wearable_detector.is_available,
            ),
        ]
        return ([n for n, ok in checks if ok], [n for n, ok in checks if not ok])

    def _build_result(
        self,
        record: SessionRecord,
        summary: SessionSummary,
        engine: ProctoringEngine,
    ) -> SessionResult:
        counts: dict[str, int] = {}
        for event in summary.events:
            counts[event.event_type.value] = counts.get(event.event_type.value, 0) + 1
        candidate_count = sum(1 for e in summary.events if not e.is_technical)
        technical_count = sum(1 for e in summary.events if e.is_technical)

        _, unavailable = self._detector_availability(engine)
        warnings: list[str] = []
        if unavailable:
            warnings.append(
                "Not observed this session: "
                + ", ".join(unavailable)
                + ". Absence of these observations does not mean absence of the behaviour."
            )
        if summary.skipped_frames:
            warnings.append(
                f"{summary.skipped_frames} frame(s) were unusable and skipped "
                "(camera fault or quality gate)."
            )
        if summary.telemetry.total_frames == 0:
            warnings.append("No frames were received; this session observed nothing.")
        if technical_count:
            warnings.append(
                f"{technical_count} technical diagnostic(s) were recorded (camera or detector "
                "faults). These are equipment problems, not candidate behaviour, and they may "
                "have reduced what the session was able to observe."
            )

        manifest_sha = None
        try:
            import json

            manifest = json.loads((Path(summary.package_dir) / "manifest.json").read_text())
            manifest_sha = manifest.get("manifest_sha256")
        except Exception:
            pass

        return SessionResult(
            session_id=record.session_id,
            attempt_id=record.attempt_id,
            user_id=record.user_id,
            state=SessionState.COMPLETED,
            contract_version=CONTRACT_VERSION,
            started_at_utc=summary.started_at_iso,
            ended_at_utc=summary.ended_at_iso,
            duration_seconds=summary.telemetry.session_duration_seconds,
            frames_sampled=summary.total_frames,
            frames_processed=summary.processed_frames,
            frames_rejected=summary.skipped_frames,
            total_observations=summary.total_events,
            qualified_observations=summary.qualified_events,
            candidate_observations=candidate_count,
            technical_diagnostics=technical_count,
            observations_by_type=counts,
            package_dir=str(summary.package_dir),
            package_archive=str(summary.zip_path) if summary.zip_path else None,
            manifest_sha256=manifest_sha,
            integrity_verified=summary.integrity_verified,
            integrity_errors=summary.integrity_errors,
            coverage_warnings=warnings,
        )

    @staticmethod
    def _summarise(event: EventRecord) -> ObservationSummary:
        """Reduce an internal event record to the review contract."""
        return ObservationSummary(
            event_id=event.event_id,
            event_type=event.event_type.value,
            severity=event.severity.value,
            status=event.status.value,
            category=event.category.value,
            start_timestamp=event.timestamp,
            end_timestamp=event.end_timestamp,
            formatted_start=event.formatted_start,
            formatted_end=event.formatted_end,
            duration_seconds=event.duration,
            description=event.observation.description,
            qualified=bool(event.metadata.get("is_duration_qualified", True)),
            confidence=event.confidence,
            reliability_note=RELIABILITY_NOTES.get(event.event_type.value),
            evidence=[
                {
                    "evidence_id": reference.evidence_id,
                    "media_type": reference.media_type,
                    "file_path": reference.file_path,
                    "sha256": reference.sha256,
                    "is_derived": reference.media_type == "review",
                }
                for reference in event.evidence
            ],
        )
