"""Integration tests for FastAPI REST API and WebSocket real-time event streaming."""

import base64
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from proctoring.integration.api import create_app
from proctoring.integration.service import ProctoringService


@pytest.fixture
def test_app():
    with tempfile.TemporaryDirectory() as tmpdir:
        service = ProctoringService(output_dir=tmpdir)
        app = create_app(service=service)
        yield app, tmpdir


def test_health_and_models_endpoints(test_app):
    app, _ = test_app
    client = TestClient(app)

    # Health
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["engine_version"] == "1.0.0"
    assert "models_loaded" in data

    # Models
    resp_models = client.get("/api/v1/models")
    assert resp_models.status_code == 200
    m_data = resp_models.json()
    assert "models" in m_data
    assert "face_detection" in m_data["models"]
    assert "face_verification" in m_data["models"]


def test_session_lifecycle_and_streaming(test_app):
    app, _ = test_app
    client = TestClient(app)

    # 1. Start Session with platform-neutral identifiers
    start_payload = {
        "exam_id": "midterm_exam_2026",
        "candidate_id": "student_alice_101",
        "candidate_name": "Alice Wonderland",
        "strictness": "STANDARD",
        "sampling_fps": 4.0,
    }
    start_resp = client.post("/api/v1/session/start", json=start_payload)
    assert start_resp.status_code == 201
    handle = start_resp.json()
    session_id = handle["session_id"]
    assert session_id is not None
    assert handle["state"] == "ACTIVE"

    # 2. Query State
    state_resp = client.get(f"/api/v1/session/{session_id}/state")
    assert state_resp.status_code == 200
    assert state_resp.json()["engine_state"] == "RUNNING"

    # 3. Ingest Frame
    blank_img = np.full((360, 480, 3), 150, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", blank_img)
    b64_frame = base64.b64encode(buf).decode("ascii")

    frame_resp = client.post(
        f"/api/v1/session/{session_id}/frame",
        json={"frame_data": b64_frame, "frame_index": 1, "timestamp_seconds": 0.25},
    )
    assert frame_resp.status_code == 200
    frame_ack = frame_resp.json()
    assert frame_ack["accepted"] is True
    assert frame_ack["frame_index"] >= 0

    # 4. Record Client Event
    event_resp = client.post(
        f"/api/v1/session/{session_id}/event",
        json={
            "event_type": "BROWSER_TAB_SWITCH",
            "timestamp_seconds": 1.5,
            "description": "Candidate switched browser tab",
            "frame_index": 2,
        },
    )
    assert event_resp.status_code == 200
    assert event_resp.json()["event_type"] == "BROWSER_TAB_SWITCH"

    # 5. Pause and Resume
    pause_resp = client.post(f"/api/v1/session/{session_id}/pause", json={"reason": "Proctor check"})
    assert pause_resp.status_code == 200
    assert pause_resp.json()["state"] == "PAUSED"

    resume_resp = client.post(f"/api/v1/session/{session_id}/resume", json={"reason": "Proctor resume"})
    assert resume_resp.status_code == 200
    assert resume_resp.json()["state"] == "ACTIVE"

    # 6. Check Incidents
    incidents_resp = client.get(f"/api/v1/session/{session_id}/incidents")
    assert incidents_resp.status_code == 200
    incidents = incidents_resp.json()
    assert len(incidents) >= 1  # browser tab switch

    # 7. Finalize Session
    final_resp = client.post(f"/api/v1/session/{session_id}/finalize")
    assert final_resp.status_code == 200
    final_data = final_resp.json()
    assert final_data["state"] == "COMPLETED"
    assert final_data["integrity_verified"] is True
    assert final_data["manifest_sha256"] is not None


def test_idempotent_sync_batch_endpoint(test_app):
    app, _ = test_app
    client = TestClient(app)

    batch_payload = {
        "session_id": "sess_sync_test",
        "events": [
            {
                "event_id": "evt_sync_001",
                "sequence_number": 1,
                "event_type": "BROWSER_TAB_SWITCH",
                "timestamp": 1.0,
                "idempotency_key": "sess_sync_test:1:evt_sync_001",
            },
            {
                "event_id": "evt_sync_002",
                "sequence_number": 2,
                "event_type": "SESSION_PAUSED",
                "timestamp": 2.0,
                "idempotency_key": "sess_sync_test:2:evt_sync_002",
            },
        ],
    }

    # First submission: both synced
    resp1 = client.post("/api/v1/sync/batch", json=batch_payload)
    assert resp1.status_code == 200
    ack1 = resp1.json()
    assert ack1["synced_sequence_numbers"] == [1, 2]
    assert ack1["duplicate_sequence_numbers"] == []

    # Second submission (network retransmission): identified as duplicate, zero duplicate logical events
    resp2 = client.post("/api/v1/sync/batch", json=batch_payload)
    assert resp2.status_code == 200
    ack2 = resp2.json()
    assert ack2["synced_sequence_numbers"] == []
    assert ack2["duplicate_sequence_numbers"] == [1, 2]


def test_websocket_streaming_connection(test_app):
    app, _ = test_app
    client = TestClient(app)

    with client.websocket_connect("/api/v1/session/ws_test_sess/ws") as websocket:
        welcome = websocket.receive_json()
        assert welcome["event"] == "CONNECTED"
        assert welcome["session_id"] == "ws_test_sess"

        # Ping-pong heartbeat
        websocket.send_text("ping")
        resp = websocket.receive_text()
        assert resp == "pong"
