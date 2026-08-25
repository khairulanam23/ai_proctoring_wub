"""End-to-end tests for the merged :class:`ProctoringEngine`.

Covers the full workflow — validation, detection, temporal qualification, evidence
creation and validation, and tamper-evident packaging — plus regression coverage
for defects the four predecessor engines carried.
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from proctoring.config import SessionConfig
from proctoring.core.events import EventType
from proctoring.detection.face_detector import FaceDetector
from proctoring.detection.face_verifier import FaceVerifier
from proctoring.engine import FaceStatus, ProctoringEngine

YUNET = Path("models/face_detection_yunet_2023mar.onnx")
SFACE = Path("models/face_recognition_sface_2021dec.onnx")


# ---------------------------------------------------------------------------
# Scripted detector: lets the temporal and evidence stages be driven precisely
# without depending on model weights or on real imagery.
# ---------------------------------------------------------------------------


class _Face:
    def __init__(self, bbox, confidence=0.9):
        self.bbox = bbox
        self.confidence = confidence
        self.raw_detection = None


class _Result:
    def __init__(self, faces):
        self.faces = list(faces)
        self.count = len(self.faces)


class ScriptedFaceDetector:
    """Returns a caller-supplied face list per call, in order."""

    def __init__(self, script):
        self._script = list(script)
        self._call = 0

    def detect(self, frame, score_threshold=None):
        faces = self._script[min(self._call, len(self._script) - 1)]
        self._call += 1
        return _Result(faces)


def _marked_frame(index: int, size=(480, 640)) -> np.ndarray:
    """Frame whose white block position encodes its index, surviving JPEG round-trip."""
    frame = np.full((size[0], size[1], 3), 120, dtype=np.uint8)
    x = 10 + index * 10
    frame[200:260, x : x + 30] = 255
    return frame


def _recover_index(image: np.ndarray) -> int:
    columns = np.where(image[200:260].mean(axis=(0, 2)) > 200)[0]
    return (int(columns.min()) - 10) // 10 if len(columns) else -1


# ---------------------------------------------------------------------------
# Workflow end to end
# ---------------------------------------------------------------------------


def test_engine_end_to_end_produces_verified_package():
    """A full session yields events, evidence and a package that passes integrity checks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(
            session_id="engine_test_sess",
            student_name="Bob Jones",
            sampling_fps=4.0,
            absence_tolerance_seconds=0.5,
            min_event_duration_seconds=0.5,
            output_dir=tmpdir,
            enable_object_detection=False,
            enable_face_verification=False,
        )
        engine = ProctoringEngine(
            config=config,
            face_detector=FaceDetector(model_path=YUNET) if YUNET.exists() else None,
            face_verifier=FaceVerifier(recognizer_model_path=SFACE) if SFACE.exists() else None,
        )
        engine.start_session()

        blank = np.full((360, 480, 3), 200, dtype=np.uint8)
        for i in range(4):
            engine.process_frame(blank, frame_index=i, timestamp_seconds=i * 0.25)

        engine.record_browser_event(
            event_type=EventType.BROWSER_TAB_SWITCH,
            timestamp_seconds=1.0,
            description="Browser tab switch detected",
            frame_index=4,
        )

        summary = engine.finalize_session()

        assert summary.session_id == "engine_test_sess"
        assert summary.total_frames == 4
        assert summary.total_events >= 2  # NO_FACE incident + browser event
        assert summary.integrity_verified is True

        package = Path(summary.package_dir)
        for artefact in ("manifest.json", "events.json", "telemetry.json", "timeline.json"):
            assert (package / artefact).exists(), f"missing {artefact}"


def test_every_event_is_illustrated_by_its_own_frame():
    """Each event's evidence is the frame that showed it, not the session's last frame.

    Regression test: the predecessor engine attached a single shared frame to every
    event, so an event at 00:01 was documented with a picture taken at 00:47.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Frames 0-9 show one face, 10-19 show none, 20-29 show two.
        script = (
            [[_Face((100, 100, 80, 80))]] * 10
            + [[]] * 10
            + [[_Face((50, 50, 80, 80)), _Face((300, 60, 80, 80))]] * 10
        )

        config = SessionConfig(
            session_id="evidence_provenance",
            student_name="Candidate",
            output_dir=tmpdir,
            sampling_fps=4.0,
            min_event_duration_seconds=1.0,
            absence_tolerance_seconds=0.5,
            enable_object_detection=False,
            enable_face_verification=False,
        )
        engine = ProctoringEngine(config=config, face_detector=ScriptedFaceDetector(script))
        engine.start_session()
        for i in range(30):
            engine.process_frame(_marked_frame(i), frame_index=i, timestamp_seconds=i / 4.0)
        summary = engine.finalize_session()

        checked = 0
        for event in summary.events:
            best = event.metadata.get("best_frame_index")
            if best is None:
                continue
            for reference in event.evidence:
                if reference.media_type != "frame":
                    continue
                image = cv2.imread(str(Path(summary.package_dir) / reference.file_path))
                assert _recover_index(image) == best, (
                    f"{event.event_type.value} evidence shows frame "
                    f"{_recover_index(image)}, expected {best}"
                )
                checked += 1

        assert checked >= 2, "expected evidence frames on at least two events"


def test_timeline_is_inside_the_tamper_evident_boundary():
    """timeline.json is checksummed by the manifest, not written after sealing.

    Regression test: the timeline used to be written after the checksum sweep, so
    it sat outside the manifest and could be altered undetectably.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(
            session_id="timeline_integrity",
            student_name="Candidate",
            output_dir=tmpdir,
            enable_face_detection=False,
            enable_object_detection=False,
        )
        engine = ProctoringEngine(config=config)
        engine.start_session()
        for i in range(3):
            engine.process_frame(_marked_frame(i), frame_index=i, timestamp_seconds=i * 0.25)
        summary = engine.finalize_session()

        package = Path(summary.package_dir)
        manifest = json.loads((package / "manifest.json").read_text())
        assert "timeline.json" in manifest["integrity_checksums"]
        assert manifest["timeline_summary"]["total_entries"] == 3
        assert summary.integrity_verified is True

        # Tampering with the timeline must now be detectable.
        (package / "timeline.json").write_text("[]")
        from proctoring.evidence.package import SessionEvidencePackage

        verifier = SessionEvidencePackage(base_dir=tmpdir, session_id="timeline_integrity")
        ok, errors = verifier.verify_package_integrity()
        assert ok is False
        assert any("timeline.json" in e for e in errors)


def test_zip_archive_path_is_reported_correctly():
    """The reported archive path points at a file that exists.

    Regression test: the archive was written as ``{package_id}.zip`` but reported
    as ``{package_dir}.zip``, so the advertised path never existed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(
            session_id="zip_path",
            student_name="Candidate",
            output_dir=tmpdir,
            create_zip=True,
            enable_face_detection=False,
            enable_object_detection=False,
        )
        engine = ProctoringEngine(config=config)
        engine.start_session()
        engine.process_frame(_marked_frame(0), frame_index=0, timestamp_seconds=0.0)
        summary = engine.finalize_session()

        assert summary.zip_path is not None
        assert summary.zip_path.exists()


def test_browser_event_accepts_a_plain_string():
    """A string event type from the browser bridge is coerced, not crashed on.

    Regression test: ``record_browser_violation`` called ``.value`` on whatever it
    was given, raising ``AttributeError`` for the plain strings a client sends.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ProctoringEngine(
            config=SessionConfig(
                session_id="coercion",
                output_dir=tmpdir,
                enable_face_detection=False,
                enable_object_detection=False,
            )
        )
        engine.start_session()

        event = engine.record_browser_event("BROWSER_TAB_SWITCH", 1.0, "tab switch")
        assert event.event_type is EventType.BROWSER_TAB_SWITCH

        unknown = engine.record_browser_event("not_a_real_event", 2.0, "unclassified")
        assert unknown.event_type is EventType.OTHER_SUSPICIOUS_ACTIVITY

        engine.finalize_session()


def test_invalid_evidence_is_pruned_from_events():
    """Evidence deleted after capture is removed from the event, and the removal is logged."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script = [[]] * 8  # sustained NO_FACE so an event is produced
        config = SessionConfig(
            session_id="prune_evidence",
            student_name="Candidate",
            output_dir=tmpdir,
            min_event_duration_seconds=0.5,
            enable_object_detection=False,
            enable_face_verification=False,
        )
        engine = ProctoringEngine(config=config, face_detector=ScriptedFaceDetector(script))
        engine.start_session()
        for i in range(8):
            engine.process_frame(_marked_frame(i), frame_index=i, timestamp_seconds=i / 4.0)

        # Flush incidents and attach evidence, then destroy a file behind the
        # pipeline's back before validation runs.
        events = engine.temporal_aggregator.flush()
        engine._attach_evidence(events)
        target = next(e for e in events if e.evidence)
        (Path(engine.evidence_manager.package_dir) / target.evidence[0].file_path).unlink()

        report = engine.evidence_manager.validate_all_event_evidence(events, prune_invalid=True)
        assert report["pruned_references"] >= 1
        assert "discarded_evidence" in target.metadata
        assert all(r.is_validated for r in target.evidence)


def test_small_background_faces_do_not_become_a_second_person():
    """Sub-threshold face boxes are filtered, so a poster is not a MULTIPLE_FACES event."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # One real face plus a tiny background face well under min_face_size_px.
        script = [[_Face((100, 100, 90, 90)), _Face((400, 40, 12, 12))]] * 8
        config = SessionConfig(
            session_id="background_faces",
            student_name="Candidate",
            output_dir=tmpdir,
            min_face_size_px=40,
            enable_object_detection=False,
            enable_face_verification=False,
        )
        engine = ProctoringEngine(config=config, face_detector=ScriptedFaceDetector(script))
        engine.start_session()
        for i in range(8):
            observation = engine.process_frame(
                _marked_frame(i), frame_index=i, timestamp_seconds=i / 4.0
            )
            assert observation.face_count == 1
            assert observation.face_status != FaceStatus.MULTIPLE_FACES

        summary = engine.finalize_session()
        assert not any(e.event_type is EventType.MULTIPLE_FACES for e in summary.events)


def test_detector_failure_is_a_diagnostic_not_an_event():
    """A crashing detector degrades the session; it never implicates the candidate."""

    class BrokenDetector:
        def detect(self, frame, score_threshold=None):
            raise RuntimeError("simulated inference failure")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(
            session_id="detector_failure",
            student_name="Candidate",
            output_dir=tmpdir,
            enable_object_detection=False,
            enable_face_verification=False,
            min_event_duration_seconds=10.0,  # keep any incident below qualification
        )
        engine = ProctoringEngine(config=config, face_detector=BrokenDetector())
        engine.start_session()
        for i in range(3):
            engine.process_frame(_marked_frame(i), frame_index=i, timestamp_seconds=i * 0.25)
        summary = engine.finalize_session()

        diagnostics = json.loads((Path(summary.package_dir) / "diagnostics.json").read_text())
        assert any(d["category"] == "MODEL_INFERENCE_FAILURE" for d in diagnostics)
        assert summary.integrity_verified is True


@pytest.mark.skipif(not YUNET.exists(), reason="YuNet model not downloaded")
def test_engine_runs_with_the_real_face_detector():
    """The engine drives the real YuNet detector without error on synthetic imagery."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(
            session_id="real_detector",
            student_name="Candidate",
            output_dir=tmpdir,
            enable_object_detection=False,
            enable_face_verification=False,
        )
        engine = ProctoringEngine(config=config, face_detector=FaceDetector(model_path=YUNET))
        engine.start_session()
        for i in range(4):
            observation = engine.process_frame(
                _marked_frame(i), frame_index=i, timestamp_seconds=i * 0.25
            )
            assert observation.accepted is True
            assert observation.face_count is not None
        assert engine.finalize_session().integrity_verified is True
