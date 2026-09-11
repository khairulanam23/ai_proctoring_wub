"""Focused tests for candidate identity resolution and face verification (Tests 1-6)."""

import glob
from pathlib import Path
import cv2
import numpy as np
import pytest

from proctoring.config import SessionConfig
from proctoring.detection.face_detector import FaceDetector
from proctoring.detection.face_verifier import FaceVerifier
from proctoring.engine import ProctoringEngine
from proctoring.integration.schemas import StartSessionRequest
from proctoring.integration.service import ProctoringService
from proctoring.storage import ProctoringStorage


@pytest.fixture(scope="module")
def shared_service():
    """Service with standard model detectors."""
    return ProctoringService()


@pytest.fixture(scope="module")
def genuine_candidate_enrollment():
    storage = ProctoringStorage(Path("data"))
    rec = storage.load_enrollment("Khairul-Anam")
    assert rec.is_usable, "Khairul-Anam enrollment reference data must be usable"
    return rec


def test_1_correct_candidate_is_recognized(shared_service, genuine_candidate_enrollment):
    """Test 1: Given real enrolled reference identity and matching live face, system returns ENROLLED/verified."""
    req = StartSessionRequest(
        session_id="test_sess_ident_1",
        attempt_id="test_sess_ident_1",
        candidate_name="Khairul Anam",
        candidate_id="36",
        enrolment_id="Khairul Anam",
    )
    handle = shared_service.start_session(req)
    engine = shared_service._engines.get(handle.session_id)
    assert engine is not None
    assert len(engine.config.reference_templates) == 5
    assert engine.config.enable_face_verification is True

    # Ingest candidate's real reference image
    img = cv2.imread(str(genuine_candidate_enrollment.image_paths[0]))
    ack = shared_service.ingest_frame(handle.session_id, img)

    assert ack.accepted is True
    assert ack.face_count == 1
    assert ack.face_status == "ENROLLED"
    assert ack.identity_verified is True


def test_2_wrong_candidate_remains_mismatched(shared_service):
    """Test 2: A different face must not be incorrectly assigned to the current candidate."""
    req = StartSessionRequest(
        session_id="test_sess_ident_2",
        attempt_id="test_sess_ident_2",
        candidate_name="Khairul Anam",
        candidate_id="36",
        enrolment_id="Khairul Anam",
    )
    handle = shared_service.start_session(req)

    # Ingest a completely different real face from data/samples
    sample_files = glob.glob("data/samples/Colin_Powell/*.jpg") + glob.glob("data/samples/George_W_Bush/*.jpg")
    assert len(sample_files) > 0, "Test sample faces must be present in data/samples"
    diff_img = cv2.imread(sample_files[0])

    ack = shared_service.ingest_frame(handle.session_id, diff_img)
    assert ack.accepted is True
    assert ack.face_count == 1
    assert ack.face_status == "IDENTITY_MISMATCH"
    assert ack.identity_verified is False


def test_3_no_face_not_interpreted_as_candidate(shared_service):
    """Test 3: No face must not be interpreted as a valid candidate identity."""
    req = StartSessionRequest(
        session_id="test_sess_ident_3",
        attempt_id="test_sess_ident_3",
        candidate_name="Khairul Anam",
        candidate_id="36",
        enrolment_id="Khairul Anam",
    )
    handle = shared_service.start_session(req)

    # Ingest black frame (no face)
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ack = shared_service.ingest_frame(handle.session_id, black_frame)

    assert ack.accepted is True
    assert ack.face_count == 0
    assert ack.face_status == "NO_FACE"
    assert ack.identity_verified is not True


def test_4_multiple_faces_not_silently_accepted(shared_service, genuine_candidate_enrollment):
    """Test 4: Multiple faces must follow project policy and not silently identify the wrong person."""
    req = StartSessionRequest(
        session_id="test_sess_ident_4",
        attempt_id="test_sess_ident_4",
        candidate_name="Khairul Anam",
        candidate_id="36",
        enrolment_id="Khairul Anam",
    )
    handle = shared_service.start_session(req)

    # Composite an image with two faces side-by-side
    img1 = cv2.imread(str(genuine_candidate_enrollment.image_paths[0]))
    sample_files = glob.glob("data/samples/Colin_Powell/*.jpg")
    img2 = cv2.imread(sample_files[0])

    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))
    dual_frame = np.hstack([img1_resized, img2_resized])

    ack = shared_service.ingest_frame(handle.session_id, dual_frame)
    assert ack.accepted is True
    assert ack.face_count >= 2
    assert ack.face_status == "MULTIPLE_FACES"


def test_5_candidate_session_scoping(shared_service, genuine_candidate_enrollment):
    """Test 5: A session bound to an unenrolled student must not accept candidate face from another session."""
    req_other = StartSessionRequest(
        session_id="test_sess_ident_5",
        attempt_id="test_sess_ident_5",
        candidate_name="Unenrolled Student",
        candidate_id="9999",
        enrolment_id="unenrolled_9999",
    )
    handle_other = shared_service.start_session(req_other)
    engine_other = shared_service._engines.get(handle_other.session_id)
    assert engine_other is not None
    assert len(engine_other.config.reference_templates) == 0
    assert engine_other.config.enable_face_verification is False

    # Ingest Khairul Anam's face into this other candidate's session
    img = cv2.imread(str(genuine_candidate_enrollment.image_paths[0]))
    ack = shared_service.ingest_frame(handle_other.session_id, img)

    # Must NOT be marked ENROLLED
    assert ack.face_status != "ENROLLED"
    assert ack.identity_verified is not True


def test_6_missing_enrollment_data_fails_diagnostically(shared_service):
    """Test 6: System fails honestly and diagnostically when enrollment data is missing; does NOT fabricate."""
    req = StartSessionRequest(
        session_id="test_sess_ident_6",
        attempt_id="test_sess_ident_6",
        candidate_name="Nonexistent Candidate",
        candidate_id="8888",
        enrolment_id="nonexistent_8888",
    )
    handle = shared_service.start_session(req)
    engine = shared_service._engines.get(handle.session_id)
    assert engine is not None
    assert len(engine.config.reference_templates) == 0
    assert engine.config.enable_face_verification is False

    # Status must be UNVERIFIED, never ENROLLED
    sample_files = glob.glob("data/samples/Colin_Powell/*.jpg")
    img = cv2.imread(sample_files[0])
    ack = shared_service.ingest_frame(handle.session_id, img)

    assert ack.face_status == "UNVERIFIED"
    assert ack.identity_verified is not True
