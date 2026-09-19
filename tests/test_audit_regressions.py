"""Regression tests for defects found in the codebase audit.

Each test names the defect it guards and describes the harm it caused, because a
regression test whose purpose is not obvious gets "fixed" by deleting it the next
time it fails. Several of these guard against silent misbehaviour rather than
crashes — the pipeline kept running and produced wrong evidence.
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from proctoring import ProctoringEngine, SessionConfig
from proctoring.analysis import StrictnessLevel
from proctoring.core.events import EventType
from proctoring.core.paths import resolve_within, sanitise_identifier
from proctoring.evidence.manager import EvidenceManager
from proctoring.integration import (
    ProctoringService,
    SessionStore,
    StartSessionRequest,
)
from proctoring.observation import FaceStatus


def _frame(value: int = 120, size=(480, 640)) -> np.ndarray:
    return np.full((size[0], size[1], 3), value, dtype=np.uint8)


def _bare_config(tmpdir, **overrides) -> SessionConfig:
    """A config with every model-backed stage off, for fast structural tests."""
    defaults = {
        "session_id": "t",
        "output_dir": tmpdir,
        "enable_face_detection": False,
        "enable_facial_dynamics": False,
        "enable_hand_analysis": False,
        "enable_object_detection": False,
    }
    defaults.update(overrides)
    return SessionConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. Cross-candidate contamination
# ---------------------------------------------------------------------------


def test_restarting_a_session_discards_the_previous_candidates_data():
    """A reused engine must not carry one candidate's evidence into another's package.

    Engines are pooled per process, so ``start_session`` is the only boundary
    between two candidates. It previously reset frame counters but not the temporal
    aggregator, timeline, telemetry or diagnostics, so candidate A's events were
    sealed into candidate B's tamper-evident package.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ProctoringEngine(config=_bare_config(tmpdir, session_id="reuse"))

        engine.start_session()
        for i in range(3):
            engine.process_frame(_frame(), frame_index=i, timestamp_seconds=i * 0.25)
        engine.record_browser_event(EventType.BROWSER_TAB_SWITCH, 1.0, "candidate A")
        first = engine.finalize_session()
        assert first.total_events == 1

        engine.start_session()
        engine.process_frame(_frame(), frame_index=0, timestamp_seconds=0.0)
        second = engine.finalize_session()

        assert second.total_frames == 1, "frame counts leaked across sessions"
        assert second.total_events == 0, "events leaked across sessions"
        assert second.timeline_summary["total_entries"] == 1, "timeline leaked across sessions"
        assert not any(e.event_type is EventType.BROWSER_TAB_SWITCH for e in second.events), (
            "candidate A's browser event appeared in candidate B's package"
        )


def test_restarting_a_session_clears_diagnostics():
    """Diagnostics are per-session too — a prior camera fault is not this one's."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ProctoringEngine(config=_bare_config(tmpdir, session_id="diag"))

        engine.start_session()
        engine.process_frame(None, frame_index=0, timestamp_seconds=0.0)  # corrupt
        assert engine.finalize_session().skipped_frames == 1

        engine.start_session()
        engine.process_frame(_frame(), frame_index=0, timestamp_seconds=0.0)
        assert engine.finalize_session().skipped_frames == 0


# ---------------------------------------------------------------------------
# 2. Path traversal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile,expected_safe",
    [
        ("../../../etc/passwd", "etc_passwd"),
        ("..", "session"),
        ("/absolute/path", "absolute_path"),
        ("", "session"),
        ("...", "session"),
        ("normal-id_123", "normal-id_123"),
    ],
)
def test_identifiers_are_reduced_to_safe_path_segments(hostile, expected_safe):
    """Host-supplied identifiers must never carry traversal into a path."""
    assert sanitise_identifier(hostile) == expected_safe


def test_evidence_cannot_be_written_outside_the_output_directory():
    """A traversal sequence in a session id must not escape the evidence root.

    ``attempt_id`` arrives over the wire from the LMS and becomes part of the
    session id, which becomes a directory name. Unsanitised, it wrote candidate
    images anywhere the process could write.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        base = root / "var" / "data" / "packages"
        base.mkdir(parents=True)
        sibling = root / "var" / "secrets"
        sibling.mkdir(parents=True)

        manager = EvidenceManager(base_dir=base, session_id="../../../secrets/stolen")
        manager.capture_frame(_frame(200), frame_index=0, timestamp_seconds=0.0)

        assert not (sibling / "stolen").exists(), "evidence escaped the output directory"
        assert base.resolve() in manager.frames_dir.resolve().parents


def test_resolve_within_rejects_escaping_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert resolve_within(tmpdir, "ok", "fine").name == "fine"
        with pytest.raises(ValueError, match="Refusing to operate outside"):
            resolve_within(tmpdir, "..", "..", "escaped")


def test_enrolment_templates_stay_inside_the_store():
    """Biometric templates must not be writable outside the configured store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        store = SessionStore(root / "store")
        store.save_enrolment("../../../escaped_biometrics", [np.ones(128, np.float32)])
        assert not (root / "escaped_biometrics.npz").exists()


# ---------------------------------------------------------------------------
# 5. Telemetry clock
# ---------------------------------------------------------------------------


def test_session_duration_excludes_time_before_the_session_started():
    """Telemetry must time the session, not the engine's lifetime.

    The tracker's wall clock started at construction, so an engine sitting in a
    pool inflated ``session_duration_seconds`` and skewed ``effective_fps`` in the
    sealed telemetry.json.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ProctoringEngine(config=_bare_config(tmpdir, session_id="clock"))
        time.sleep(0.6)  # idle in a pool
        engine.start_session()
        for i in range(3):
            engine.process_frame(_frame(), frame_index=i, timestamp_seconds=i * 0.25)
        summary = engine.finalize_session()

        assert summary.telemetry.session_duration_seconds < 0.5, (
            "idle time before start_session() was counted as session duration"
        )


# ---------------------------------------------------------------------------
# 7. Configuration sentinel
# ---------------------------------------------------------------------------


def test_explicit_config_value_survives_even_when_it_equals_a_default():
    """An explicitly-passed threshold must always win over the policy preset.

    Defaults were detected by comparing against ``__dataclass_fields__``, so a
    caller who deliberately chose the value that happened to be the default had it
    silently replaced by the strictness preset.
    """
    config = SessionConfig(
        session_id="explicit",
        strictness=StrictnessLevel.STANDARD,
        min_event_duration_seconds=1.0,  # also the old dataclass default
    )
    assert config.min_event_duration_seconds == 1.0


def test_unset_thresholds_come_from_the_policy():
    standard = SessionConfig(session_id="a", strictness=StrictnessLevel.STANDARD)
    maximum = SessionConfig(session_id="b", strictness=StrictnessLevel.MAXIMUM)
    assert standard.min_event_duration_seconds == 2.0
    assert maximum.min_event_duration_seconds == 1.0
    assert maximum.absence_tolerance_seconds < standard.absence_tolerance_seconds


def test_session_config_device_deterministic():
    """Verify SessionConfig.device has no duplicate declaration and resolves deterministically."""
    import ast
    from pathlib import Path
    import proctoring.config

    tree = ast.parse(Path(proctoring.config.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SessionConfig":
            device_assigns = [
                n for n in node.body
                if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == "device"
            ]
            assert len(device_assigns) == 1, (
                f"Expected exactly 1 device declaration in SessionConfig, found {len(device_assigns)}"
            )

    default_cfg = SessionConfig(session_id="det_default")
    assert default_cfg.device == "cuda"

    cpu_cfg = SessionConfig(session_id="det_cpu", device="cpu")
    assert cpu_cfg.device == "cpu"

    auto_cfg = SessionConfig(session_id="det_auto", device="auto")
    assert auto_cfg.device == "auto"

    cuda_norm_cfg = SessionConfig(session_id="det_cuda", device=" CUDA:0 ")
    assert cuda_norm_cfg.device == "cuda:0"


# ---------------------------------------------------------------------------
# 3. Identity across all faces
# ---------------------------------------------------------------------------


class _Face:
    def __init__(self, bbox):
        self.bbox = bbox
        self.confidence = 0.9
        self.raw_detection = None


class _TwoFaceDetector:
    def detect(self, frame, score_threshold=None):
        class Result:
            faces = [_Face((100, 100, 90, 90)), _Face((300, 100, 90, 90))]
            count = 2

        return Result()


class _AlternatingVerifier:
    """Emits a matching embedding for the first face and a non-matching one for the second."""

    def __init__(self):
        self.calls = 0

    def extract_feature(self, frame, face=None, **kwargs):
        self.calls += 1
        return np.array([1.0, 0.0]) if self.calls % 2 else np.array([0.0, 1.0])

    def compute_similarity(self, a, b):
        return float(np.dot(a, b))


def _run_two_face_session(tmpdir, template):
    config = SessionConfig(
        session_id="multi",
        output_dir=tmpdir,
        strictness=StrictnessLevel.STANDARD,
        enable_object_detection=False,
        enable_facial_dynamics=False,
        enable_hand_analysis=False,
        reference_templates=[template],
        min_event_duration_seconds=0.5,
    )
    engine = ProctoringEngine(
        config=config, face_detector=_TwoFaceDetector(), face_verifier=_AlternatingVerifier()
    )
    engine.start_session()
    observation = None
    for i in range(8):
        observation = engine.process_frame(_frame(), frame_index=i, timestamp_seconds=i * 0.25)
    return observation, engine.finalize_session()


def test_enrolled_candidate_among_several_faces_is_recognised():
    """With the candidate present, only MULTIPLE_FACES is reported."""
    with tempfile.TemporaryDirectory() as tmpdir:
        observation, summary = _run_two_face_session(tmpdir, np.array([1.0, 0.0]))

        assert observation.face_status == FaceStatus.MULTIPLE_FACES
        assert observation.enrolled_face_present is True
        assert observation.unknown_face_count == 1
        assert not any(e.event_type is EventType.UNKNOWN_FACE for e in summary.events)


def test_candidate_absent_among_several_faces_raises_unknown_face():
    """With the candidate gone, MULTIPLE_FACES **and** UNKNOWN_FACE are reported.

    Verifying only when exactly one face was present left the impostor case
    unanswered: "the candidate plus a helper" and "two strangers, candidate gone"
    produced identical output.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        observation, summary = _run_two_face_session(tmpdir, np.array([0.0, 0.0]))

        assert observation.enrolled_face_present is False
        assert observation.unknown_face_count == 2
        types = {e.event_type for e in summary.events}
        assert EventType.MULTIPLE_FACES in types
        assert EventType.UNKNOWN_FACE in types


def test_embedding_failure_is_not_reported_as_an_identity_mismatch():
    """A broken verifier yields UNVERIFIED, never UNKNOWN_FACE."""

    class BrokenVerifier:
        def extract_feature(self, frame, face=None, **kwargs):
            raise RuntimeError("simulated embedding failure")

        def compute_similarity(self, a, b):
            return 0.0

    class OneFaceDetector:
        def detect(self, frame, score_threshold=None):
            class Result:
                faces = [_Face((100, 100, 90, 90))]
                count = 1

            return Result()

    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(
            session_id="broken",
            output_dir=tmpdir,
            enable_object_detection=False,
            enable_facial_dynamics=False,
            enable_hand_analysis=False,
            reference_templates=[np.array([1.0, 0.0])],
        )
        engine = ProctoringEngine(
            config=config, face_detector=OneFaceDetector(), face_verifier=BrokenVerifier()
        )
        engine.start_session()
        observation = engine.process_frame(_frame(), frame_index=0, timestamp_seconds=0.0)
        summary = engine.finalize_session()

        assert observation.face_status == FaceStatus.UNVERIFIED
        assert not any(e.event_type is EventType.UNKNOWN_FACE for e in summary.events)


# ---------------------------------------------------------------------------
# Service: 6, 8, 9, 10
# ---------------------------------------------------------------------------


@pytest.fixture
def service(tmp_path):
    return ProctoringService(
        output_dir=tmp_path / "packages",
        store=SessionStore(tmp_path / "store"),
        detector_factory=lambda: {},
        session_idle_timeout_seconds=0.4,
    )


def test_concurrent_starts_for_one_attempt_get_distinct_sessions(service):
    """Two starts in the same second must not collide.

    Session ids were timestamped to the second, so a retry silently evicted the
    first session's engine from the registry and orphaned it.
    """
    first = service.start_session(StartSessionRequest(attempt_id="77", user_id="1"))
    second = service.start_session(StartSessionRequest(attempt_id="77", user_id="1"))

    assert first.session_id != second.session_id
    assert service.active_session_count == 2


def test_oversized_frame_payloads_are_rejected(service):
    """A huge upload must be refused rather than decoded into memory."""
    handle = service.start_session(StartSessionRequest(attempt_id="1", user_id="1"))
    ack = service.ingest_frame(
        handle.session_id, "data:image/jpeg;base64," + "A" * (12 * 1024 * 1024)
    )

    assert ack.accepted is False
    # The session must survive the rejection.
    assert service.ingest_frame(handle.session_id, _frame()).accepted is True


def test_idle_sessions_are_reaped_and_still_sealed(service):
    """An abandoned attempt must free its engine without discarding its evidence.

    A closed browser never calls finalize, so without reaping the engine, its
    models and its retained frames stayed resident for the process lifetime.
    """
    handle = service.start_session(StartSessionRequest(attempt_id="42", user_id="9"))
    service.ingest_frame(handle.session_id, _frame())
    assert service.active_session_count == 1

    time.sleep(0.5)
    reaped = service.reap_idle_sessions()

    assert handle.session_id in reaped
    assert service.active_session_count == 0
    assert Path(service.store.get(handle.session_id).package_dir).exists()


def test_active_sessions_are_not_reaped(service):
    """Reaping must only take sessions that have actually gone quiet."""
    handle = service.start_session(StartSessionRequest(attempt_id="live", user_id="1"))
    service.ingest_frame(handle.session_id, _frame())
    assert service.reap_idle_sessions(timeout_seconds=60.0) == []
    assert service.active_session_count == 1


def test_each_session_gets_its_own_lock(service):
    """Frame ingestion is serialised per session, not globally."""
    first = service.start_session(StartSessionRequest(attempt_id="a", user_id="1"))
    second = service.start_session(StartSessionRequest(attempt_id="b", user_id="2"))

    lock_a = service._session_lock(first.session_id)
    lock_b = service._session_lock(second.session_id)
    assert lock_a is not lock_b
    assert service._session_lock(first.session_id) is lock_a


def test_finalising_releases_all_session_resources(service):
    handle = service.start_session(StartSessionRequest(attempt_id="rel", user_id="1"))
    service.ingest_frame(handle.session_id, _frame())
    service.finalize_session(handle.session_id)

    assert service.active_session_count == 0
    assert handle.session_id not in service._session_locks
    assert handle.session_id not in service._last_activity


# ---------------------------------------------------------------------------
# 17. Wearable carry-forward expiry
# ---------------------------------------------------------------------------


class _OneShotWearableDetector:
    """Reports headphones on its first sweep, then nothing (as if it started failing)."""

    is_available = True

    def __init__(self):
        self.calls = 0

    def detect(self, frame, ear_regions=None, face_bbox=None):
        from proctoring.analysis.wearables import WearableAnalysisResult, WearableDetection

        self.calls += 1
        if self.calls > 1:
            return WearableAnalysisResult(ran=True, detections=[])
        return WearableAnalysisResult(
            ran=True,
            detections=[
                WearableDetection(
                    "headphones",
                    "headphones",
                    EventType.HEADPHONES_DETECTED,
                    0.7,
                    (10, 10, 60, 60),
                    ear_anchored=True,
                    reliability="high",
                ),
            ],
        )


def test_a_stale_wearable_reading_is_not_carried_forward_indefinitely():
    """A detector that stops reporting must not pin its last detection open.

    Readings are carried across skipped frames so a device incident stays
    continuous with the decimated sweep cadence. Without an expiry, a detector that
    began failing would hold the last detection open for the rest of the session.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(
            session_id="wear",
            output_dir=tmpdir,
            enable_face_detection=False,
            enable_facial_dynamics=False,
            enable_hand_analysis=False,
            enable_object_detection=False,
            enable_wearable_detection=True,
            wearable_detection_interval_frames=4,
            sampling_fps=4.0,
        )
        engine = ProctoringEngine(config=config, wearable_detector=_OneShotWearableDetector())
        engine.start_session()

        first = engine.process_frame(_frame(), frame_index=0, timestamp_seconds=0.0)
        assert "headphones" in first.wearable_names

        # Frames 1-3 are between sweeps: the reading is still fresh and carried.
        carried = engine.process_frame(_frame(), frame_index=1, timestamp_seconds=0.25)
        assert "headphones" in carried.wearable_names

        # Far beyond two sweep intervals, the stale reading must be dropped.
        stale = engine.process_frame(_frame(), frame_index=9, timestamp_seconds=30.0)
        assert stale.wearable_names == [], "a stale wearable reading was carried forward"
        engine.finalize_session()
