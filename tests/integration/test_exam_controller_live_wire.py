"""Live HTTP Wire Integration Tests between ExamController and AI Proctoring Engine.

Verifies the exact wire API boundary expected by ExamController's
@exam-controller/module-ai-integration (WubProctoringProvider) on port 7001:
1. Health and Readiness: GET /api/v1/health and GET /health
2. Model catalog & thresholds: GET /api/v1/models
3. Session registration: POST /api/v1/session/start
4. Base64 frame streaming: POST /api/v1/session/{id}/frame
5. Evidence retrieval with SHA-256 header: GET /api/v1/session/{id}/evidence/{evidence_id}
6. Session finalization & tamper-evident package sealing: POST /api/v1/session/{id}/finalize
"""

from __future__ import annotations

import base64
import hashlib
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from proctoring.integration.api import create_app
from proctoring.integration.service import ProctoringService


@pytest.fixture
def wire_client():
    with tempfile.TemporaryDirectory() as tmpdir:
        service = ProctoringService(output_dir=tmpdir)
        app = create_app(service=service)
        client = TestClient(app)
        yield client, tmpdir


def encode_test_frame_base64(width: int = 640, height: int = 480) -> str:
    """Generate a realistic test frame encoded in JPEG Base64."""
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    cv2.circle(img, (width // 2, height // 2), 60, (200, 200, 200), -1)
    _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode("utf-8")


def test_exam_controller_health_probe(wire_client):
    """Verify ExamController connect() probe against /api/v1/health and /health."""
    client, _ = wire_client

    # Primary endpoint
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "engine_version" in data
    assert isinstance(data.get("models_loaded"), dict)
    assert "hardware" in data
    assert "cuda_available" in data["hardware"]

    # Fallback endpoint
    res_fb = client.get("/health")
    assert res_fb.status_code == 200
    assert res_fb.json()["status"] == "ok"


def test_exam_controller_models_catalog(wire_client):
    """Verify model catalog endpoint queried by ExamController diagnostics."""
    client, _ = wire_client
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert "face_detection" in data["models"]
    assert "object_detection" in data["models"]
    assert "hardware" in data


def test_exam_controller_end_to_end_wire_session(wire_client):
    """Verify complete ExamController exam lifecycle over HTTP wire protocol."""
    client, _ = wire_client

    client_session_id = "exam_midterm_attempt_8829"
    student_id = "candidate_usr_4910"
    student_name = "Jane Candidate"

    # Step 1: Start Session (WubProctoringProvider.submitFrame -> ensure remote session)
    start_payload = {
        "session_id": client_session_id,
        "attempt_id": client_session_id,
        "exam_id": "quiz_chemistry_101",
        "candidate_id": student_id,
        "user_id": student_id,
        "candidate_name": student_name,
        "enrolment_id": student_name,
        "strictness": "STANDARD",
        "sampling_fps": 4.0,
        "enable_wearable_detection": True,
        "metadata": {
            "requested_detectors": ["face", "phone", "wearable"],
        },
    }

    start_res = client.post("/api/v1/session/start", json=start_payload)
    assert start_res.status_code == 201
    start_data = start_res.json()
    engine_session_id = start_data["session_id"]
    assert engine_session_id is not None
    assert start_data["state"] in ("CREATED", "ACTIVE", "RUNNING")

    # Step 2: Stream camera frames
    frame_b64 = encode_test_frame_base64()

    for idx in range(3):
        frame_payload = {
            "frame_data": frame_b64,
            "frame_index": idx,
            "timestamp_seconds": idx * 0.25,
        }
        frame_res = client.post(
            f"/api/v1/session/{engine_session_id}/frame",
            json=frame_payload,
        )
        assert frame_res.status_code == 200
        ack = frame_res.json()
        assert ack["session_id"] == engine_session_id
        assert ack["frame_index"] == idx
        assert ack["accepted"] is True
        assert "active_observations" in ack
        assert "face_status" in ack

    # Step 3: Check Session State query
    state_res = client.get(f"/api/v1/session/{engine_session_id}/state")
    assert state_res.status_code == 200
    state_data = state_res.json()
    assert state_data["session_id"] == engine_session_id
    assert state_data["processed_frames"] == 3

    # Step 4: Finalize Session (WubProctoringProvider / ExamController submission)
    finalize_res = client.post(f"/api/v1/session/{engine_session_id}/finalize")
    assert finalize_res.status_code == 200
    final_data = finalize_res.json()
    assert final_data["session_id"] == engine_session_id
    assert final_data["state"] in ("COMPLETED", "FINALIZED")
    assert "manifest_sha256" in final_data
    assert len(final_data["manifest_sha256"]) == 64  # Valid 256-bit hex hash
    assert final_data.get("integrity_verified") is True


def test_evidence_retrieval_wire_endpoint(wire_client):
    """Verify GET /api/v1/session/{session_id}/evidence/{evidence_id} with SHA-256 header."""
    client, tmpdir = wire_client

    # Start session and create a dummy evidence file in package dir
    start_payload = {
        "session_id": "evidence_test_sess",
        "attempt_id": "evidence_test_sess",
        "candidate_id": "student_01",
        "candidate_name": "Student 01",
    }
    start_res = client.post("/api/v1/session/start", json=start_payload)
    engine_session_id = start_res.json()["session_id"]

    # Write a test JPEG into the session evidence directory
    session_evidence_dir = Path(tmpdir) / engine_session_id / "evidence" / "frames"
    session_evidence_dir.mkdir(parents=True, exist_ok=True)
    test_img_path = session_evidence_dir / "frame_000001.jpg"

    dummy_jpeg = np.full((240, 320, 3), 100, dtype=np.uint8)
    cv2.imwrite(str(test_img_path), dummy_jpeg)

    expected_sha256 = hashlib.sha256(test_img_path.read_bytes()).hexdigest()

    # Query evidence via API
    evidence_res = client.get(f"/api/v1/session/{engine_session_id}/evidence/000001")
    assert evidence_res.status_code == 200
    assert evidence_res.headers.get("content-type") == "image/jpeg"
    assert evidence_res.headers.get("x-evidence-sha256") == expected_sha256
    assert len(evidence_res.content) > 0
