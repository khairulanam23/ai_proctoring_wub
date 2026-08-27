"""Regression tests for defects found during Phase 6 final validation.

Three of these guard features that were silently dead: they raised inside a
``try/except`` or mismatched a signature, so the pipeline kept running and simply
never produced the observation. Nothing failed loudly, which is precisely why they
survived earlier phases.
"""

import glob
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from proctoring import ProctoringEngine, SessionConfig
from proctoring.analysis.policy import ExamPolicy, StrictnessLevel
from proctoring.core.events import EventCategory, EventType, category_for, is_technical
from proctoring.detection.face_detector import FaceDetector
from proctoring.observation import FaceStatus

YUNET = Path("models/face_detection_yunet_2023mar.onnx")
requires_models = pytest.mark.skipif(not YUNET.exists(), reason="YuNet model not downloaded")


def _faces() -> list[np.ndarray]:
    paths = sorted(glob.glob("data/samples/*/*_0001.jpg"))
    return [cv2.resize(cv2.imread(p), (640, 480)) for p in paths[:8]]


def _blank(value: int = 120) -> np.ndarray:
    return np.full((480, 640, 3), value, dtype=np.uint8)


def _config(tmpdir, **overrides) -> SessionConfig:
    base = {"session_id": "p6", "output_dir": tmpdir, "enable_object_detection": False}
    base.update(overrides)
    return SessionConfig(**base)


# ---------------------------------------------------------------------------
# Technical / candidate separation
# ---------------------------------------------------------------------------


def test_equipment_faults_are_categorised_apart_from_candidate_behaviour():
    """The invariant is enforced structurally, not only in a comment.

    Camera and detector faults previously sat in the same undifferentiated event
    list as prohibited objects, so any total mixed "things the equipment did" with
    "things the candidate did".
    """
    for event_type in (
        EventType.CAMERA_FRAME_FROZEN,
        EventType.CAMERA_DISCONNECTED,
        EventType.CAMERA_OBSTRUCTED,
        EventType.DETECTOR_ERROR,
        EventType.SYSTEM_ERROR,
        EventType.SESSION_PAUSED,
        EventType.SESSION_RESUMED,
    ):
        assert category_for(event_type) is EventCategory.TECHNICAL_DIAGNOSTIC

    for event_type in (
        EventType.NO_FACE,
        EventType.MULTIPLE_FACES,
        EventType.PHONE_DETECTED,
        EventType.CANDIDATE_SPEAKING,
    ):
        assert category_for(event_type) is EventCategory.CANDIDATE_OBSERVATION


def test_unlisted_event_types_default_to_candidate_observation():
    """A newly added event defaults to the category that receives scrutiny."""
    assert category_for(EventType.OTHER_SUSPICIOUS_ACTIVITY) is EventCategory.CANDIDATE_OBSERVATION


def test_strictness_never_suppresses_a_technical_diagnostic():
    """A lenient exam profile must not hide that the camera broke.

    Strictness governs how closely the candidate is scrutinised. Letting it gate
    equipment faults would suppress exactly the information that explains why
    observations are missing.
    """
    for level in StrictnessLevel:
        policy = ExamPolicy.for_level(level)
        for event_type in EventType:
            if is_technical(event_type):
                assert policy.allows(event_type), f"{event_type.value} blocked at {level.value}"


def test_event_record_exposes_its_category_in_serialised_form():
    from proctoring.core.events import DetectorInfo, EventRecord, EventSeverity, ObservationDetail

    record = EventRecord(
        event_id="e",
        session_id="s",
        timestamp=0.0,
        end_timestamp=1.0,
        duration=1.0,
        formatted_start="a",
        formatted_end="b",
        event_type=EventType.CAMERA_FRAME_FROZEN,
        severity=EventSeverity.MEDIUM,
        confidence=1.0,
        average_confidence=1.0,
        detector=DetectorInfo(name="t"),
        observation=ObservationDetail(description="d"),
    )
    assert record.is_technical is True
    assert record.to_dict()["category"] == "TECHNICAL_DIAGNOSTIC"


# ---------------------------------------------------------------------------
# Detector failure isolation
# ---------------------------------------------------------------------------


class _BrokenFaceDetector:
    def detect(self, *args, **kwargs):
        raise RuntimeError("simulated YuNet failure")


@requires_models
def test_a_failed_face_detector_never_produces_a_no_face_event():
    """A crashed detector must not accuse a candidate who is sitting right there.

    The stage returned an empty list on failure, which is indistinguishable from a
    camera pointed at an empty chair, so a broken model produced NO_FACE against a
    present candidate.
    """
    frames = _faces()
    if not frames:
        pytest.skip("sample imagery unavailable")

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ProctoringEngine(
            config=_config(
                tmpdir,
                enable_facial_dynamics=False,
                enable_hand_analysis=False,
                min_event_duration_seconds=0.5,
            ),
            face_detector=_BrokenFaceDetector(),
        )
        engine.start_session()
        for index in range(10):
            observation = engine.process_frame(frames[index % len(frames)], index, index * 0.25)
        summary = engine.finalize_session()

        assert observation.face_status == FaceStatus.NOT_MEASURED
        assert "face_detector" in observation.detector_failures
        assert [e for e in summary.events if not e.is_technical] == []
        assert any(e.event_type is EventType.DETECTOR_ERROR for e in summary.events)


@requires_models
def test_detector_failure_is_reported_once_not_once_per_frame():
    """One outage is one incident; hundreds of identical entries help nobody."""
    frames = _faces()
    if not frames:
        pytest.skip("sample imagery unavailable")

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ProctoringEngine(
            config=_config(tmpdir, enable_facial_dynamics=False, enable_hand_analysis=False),
            face_detector=_BrokenFaceDetector(),
        )
        engine.start_session()
        for index in range(30):
            engine.process_frame(frames[0], index, index * 0.25)
        summary = engine.finalize_session()

        errors = [e for e in summary.events if e.event_type is EventType.DETECTOR_ERROR]
        assert len(errors) == 1


@requires_models
def test_a_working_detector_still_reports_genuine_absence():
    """The isolation fix must not blind the pipeline to a real empty chair."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ProctoringEngine(
            config=_config(
                tmpdir,
                enable_facial_dynamics=False,
                enable_hand_analysis=False,
                min_event_duration_seconds=0.5,
            ),
            face_detector=FaceDetector(model_path=YUNET),
        )
        engine.start_session()
        for index in range(10):
            engine.process_frame(_blank(), index, index * 0.25)
        summary = engine.finalize_session()

        assert any(e.event_type is EventType.NO_FACE for e in summary.events)


# ---------------------------------------------------------------------------
# Features that were silently dead
# ---------------------------------------------------------------------------


@requires_models
def test_occlusion_classification_actually_runs():
    """Occlusion was never classified: the call used ``landmarks=``, the signature
    declared ``_landmarks``, and the resulting TypeError was swallowed."""
    frames = _faces()
    if not frames:
        pytest.skip("sample imagery unavailable")

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ProctoringEngine(
            config=_config(tmpdir), face_detector=FaceDetector(model_path=YUNET)
        )
        engine.start_session()
        for index in range(5):
            observation = engine.process_frame(frames[index % len(frames)], index, index * 0.25)
        engine.finalize_session()

        assert observation.occlusion is not None, "occlusion classification did not run"
        assert observation.occlusion.state is not None


@requires_models
def test_gaze_calibration_completes_on_usable_frames():
    """Calibration raised on its first frame: it passed a bbox where the analyzer
    expects a timestamp, so per-candidate gaze baselines never existed."""
    frames = _faces()
    if len(frames) < 3:
        pytest.skip("need several sample portraits")

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ProctoringEngine(
            config=_config(tmpdir), face_detector=FaceDetector(model_path=YUNET)
        )
        report = engine.calibrate_gaze(frames)

        assert report["calibrated"] is True
        assert report["samples"] >= 3
        assert "baseline_horizontal" in report


@requires_models
def test_gaze_calibration_fails_cleanly_without_faces():
    """Unusable calibration input must report why, not raise."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ProctoringEngine(
            config=_config(tmpdir), face_detector=FaceDetector(model_path=YUNET)
        )
        report = engine.calibrate_gaze([_blank()] * 5)

        assert report["calibrated"] is False
        assert "reason" in report


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def test_pause_skips_inference_entirely():
    """A paused session must cost nothing, not merely discard its results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ProctoringEngine(config=_config(tmpdir, enable_face_detection=False))
        engine.start_session()
        engine.pause("proctor break")

        observation = engine.process_frame(_blank(), 1, 0.25)

        assert observation.accepted is False
        assert observation.rejection_reason == "SESSION_PAUSED"
        assert observation.timing.face_detector_ms == 0.0
        engine.finalize_session()


def test_pause_and_resume_are_recorded_so_gaps_are_explained():
    """An unexplained hole in the timeline is indistinguishable from a failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ProctoringEngine(config=_config(tmpdir, enable_face_detection=False))
        engine.start_session()
        engine.process_frame(_blank(), 0, 0.0)
        engine.pause("proctor break")
        engine.resume("back")
        engine.process_frame(_blank(), 1, 0.25)
        summary = engine.finalize_session()

        types = {e.event_type for e in summary.events}
        assert EventType.SESSION_PAUSED in types
        assert EventType.SESSION_RESUMED in types
        # Session control is equipment/operator bookkeeping, never misconduct.
        for event in summary.events:
            if event.event_type in (EventType.SESSION_PAUSED, EventType.SESSION_RESUMED):
                assert event.is_technical is True


def test_the_two_lifecycle_enums_are_distinct_types():
    """Engine processing state and LMS attempt state are different concepts.

    Both were named ``SessionState``, so the meaning depended on the import path.
    """
    from proctoring.engine import EngineState
    from proctoring.integration.schemas import SessionState

    assert EngineState is not SessionState
    assert EngineState.__name__ != SessionState.__name__


def test_package_manifest_separates_the_two_event_categories():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ProctoringEngine(config=_config(tmpdir, enable_face_detection=False))
        engine.start_session()
        engine.process_frame(_blank(), 0, 0.0)
        engine.pause("break")
        engine.resume("back")
        summary = engine.finalize_session()

        manifest = json.loads((Path(summary.package_dir) / "manifest.json").read_text())
        event_summary = manifest["event_summary"]
        assert "candidate_observations" in event_summary
        assert "technical_diagnostics" in event_summary
        assert event_summary["technical_diagnostics"] >= 2


# ---------------------------------------------------------------------------
# Input hardening
# ---------------------------------------------------------------------------


def test_oversized_raster_is_rejected_even_when_passed_as_an_array():
    """The pixel ceiling applied only to decoded payloads.

    An in-process caller could hand over a 9000x9000 array — a quarter of a
    gigabyte — and every detector would run across it.
    """
    from proctoring.integration import ProctoringService, SessionStore, StartSessionRequest

    with tempfile.TemporaryDirectory() as tmpdir:
        service = ProctoringService(
            output_dir=Path(tmpdir) / "p",
            store=SessionStore(Path(tmpdir) / "s"),
            detector_factory=lambda: {},
        )
        handle = service.start_session(StartSessionRequest(attempt_id="1", user_id="1"))

        huge = service.ingest_frame(handle.session_id, np.zeros((9000, 9000, 3), np.uint8))
        assert huge.accepted is False

        assert service.ingest_frame(handle.session_id, _blank()).accepted is True
        service.finalize_session(handle.session_id)


def test_unknown_session_raises_a_clear_error_not_an_attribute_error():
    """``store.get`` returns None for an evicted session; reaching through it turned
    a recoverable condition into an AttributeError from deep inside finalisation."""
    from proctoring.integration import ProctoringService, ProctoringServiceError, SessionStore

    with tempfile.TemporaryDirectory() as tmpdir:
        service = ProctoringService(
            output_dir=Path(tmpdir) / "p",
            store=SessionStore(Path(tmpdir) / "s"),
            detector_factory=lambda: {},
        )
        with pytest.raises(ProctoringServiceError, match="Unknown session"):
            service.get_state("nope")
        with pytest.raises(ProctoringServiceError, match="Unknown session"):
            service.get_review("nope")
