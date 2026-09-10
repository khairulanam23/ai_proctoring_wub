"""FastAPI REST and WebSocket service boundary for Exam Controller integration."""

from __future__ import annotations

import asyncio
import base64
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from proctoring.analysis.policy import StrictnessLevel
from proctoring.config import SessionConfig
from proctoring.core.events import EventType
from proctoring.engine import EngineState, ProctoringEngine
from proctoring.integration.schemas import (
    CONTRACT_VERSION,
    FrameAck,
    HealthResponse,
    ObservationSummary,
    ReviewPayload,
    SessionHandle,
    SessionResult,
    SessionState,
    StartSessionRequest,
    SyncBatchAck,
)
from proctoring.integration.service import ProctoringService, ProctoringServiceError

LOGGER = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Pydantic Schemas for API Validation
# ----------------------------------------------------------------------


class StartSessionPayload(BaseModel):
    session_id: str | None = None
    attempt_id: str | None = None
    exam_id: str | None = None
    quiz_id: str | None = None
    candidate_id: str | None = None
    user_id: str | None = None
    candidate_name: str = "Candidate"
    organization_id: str | None = None
    strictness: str = "STANDARD"
    sampling_fps: float = 4.0
    enable_wearable_detection: bool = False
    enrolment_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FrameIngestPayload(BaseModel):
    frame_data: str  # Base64 encoded JPEG / PNG
    frame_index: int = 0
    timestamp_seconds: float = 0.0


class ClientEventPayload(BaseModel):
    event_type: str
    timestamp_seconds: float
    description: str
    frame_index: int = 0
    metadata: dict[str, Any] | None = None


class ControlPayload(BaseModel):
    reason: str = "Proctor control action"


class CandidateEnrollPayload(BaseModel):
    candidate_id: str
    candidate_name: str
    images: list[str]  # Base64 encoded JPEG frames


class SyncBatchPayload(BaseModel):
    session_id: str
    events: list[dict[str, Any]]


# ----------------------------------------------------------------------
# WebSocket Connection Hub
# ----------------------------------------------------------------------


class WebSocketHub:
    """Manages active WebSocket connections per session for real-time streaming."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        if session_id not in self._connections:
            self._connections[session_id] = []
        self._connections[session_id].append(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        if session_id in self._connections and websocket in self._connections[session_id]:
            self._connections[session_id].remove(websocket)
            if not self._connections[session_id]:
                del self._connections[session_id]

    async def broadcast(self, session_id: str, message: dict[str, Any]) -> None:
        if session_id not in self._connections:
            return
        stale: list[WebSocket] = []
        for ws in self._connections[session_id]:
            try:
                await ws.send_json(message)
            except Exception:
                stale.append(ws)
        for s in stale:
            self.disconnect(session_id, s)


# ----------------------------------------------------------------------
# Application Factory
# ----------------------------------------------------------------------


def create_app(service: ProctoringService | None = None) -> FastAPI:
    """Construct FastAPI proctoring engine service."""
    app = FastAPI(
        title="AI Proctoring Engine API",
        version=CONTRACT_VERSION,
        description="Authoritative, privacy-first, and human-in-the-loop AI proctoring service.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.service = service or ProctoringService()
    app.state.ws_hub = WebSocketHub()
    app.state.seen_idempotency_keys = set()

    # ------------------------------------------------------------------
    # Health and Readiness
    # ------------------------------------------------------------------

    @app.get("/health", response_model=dict[str, Any])
    @app.get("/api/v1/health", response_model=dict[str, Any])
    async def get_health() -> dict[str, Any]:
        """Health and readiness check returning model and session stats."""
        models_dir = Path("models")
        cuda_available = False
        cuda_device = None
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_device = torch.cuda.get_device_name(0)
        except Exception:
            pass

        hardware = {
            "cuda_available": cuda_available,
            "cuda_device": cuda_device,
            "device_backends": {
                "object_detector_yolo11": "cuda" if cuda_available else "cpu",
                "face_detector_yunet": "cpu (OpenCV MLAS SGEMM)",
                "face_verifier_sface": "cpu (OpenCV MLAS SGEMM)",
                "mediapipe_landmarkers": "cpu (TFLite XNNPACK)",
            },
        }

        return HealthResponse(
            status="ok",
            engine_version="1.0.0",
            models_loaded={
                "face_detector_yunet": (models_dir / "face_detection_yunet_2023mar.onnx").exists(),
                "face_verifier_sface": (models_dir / "face_recognition_sface_2021dec.onnx").exists(),
                "face_landmarker": (models_dir / "face_landmarker.task").exists(),
                "hand_landmarker": (models_dir / "hand_landmarker.task").exists(),
                "object_detector_yolo11": (models_dir / "yolo11n.pt").exists(),
            },
            active_sessions_count=len(app.state.service._engines),
            hardware=hardware,
        ).to_dict()

    @app.get("/api/v1/models", response_model=dict[str, Any])
    async def get_models_info() -> dict[str, Any]:
        """Retrieve authoritative model catalog, versions, and operating thresholds."""
        cuda_available = False
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except Exception:
            pass

        return {
            "engine_version": "1.0.0",
            "contract_version": CONTRACT_VERSION,
            "hardware": {
                "cuda_available": cuda_available,
                "accelerated_stage": "object_detection (YOLO11)",
            },
            "models": {
                "face_detection": {
                    "name": "OpenCV YuNet",
                    "version": "2023mar",
                    "format": "ONNX",
                    "backend": "OpenCV DNN (CPU MLAS SGEMM)",
                    "device": "cpu",
                    "default_score_threshold": 0.60,
                },
                "face_verification": {
                    "name": "OpenCV SFace",
                    "version": "2021dec",
                    "format": "ONNX",
                    "backend": "OpenCV DNN (CPU MLAS SGEMM)",
                    "device": "cpu",
                    "default_cosine_threshold": 0.3630,
                },
                "object_detection": {
                    "name": "Ultralytics YOLO11",
                    "version": "11.0",
                    "backend": "Ultralytics PyTorch (CUDA)" if cuda_available else "Ultralytics PyTorch (CPU)",
                    "device": "cuda" if cuda_available else "cpu",
                    "default_confidence_threshold": 0.40,
                },
                "facial_and_hand_dynamics": {
                    "name": "MediaPipe Landmarkers",
                    "version": "tasks-1.0",
                    "backend": "MediaPipe Tasks (CPU XNNPACK)",
                    "device": "cpu",
                    "tasks": ["face_landmarker.task", "hand_landmarker.task"],
                },
            },
        }

    # ------------------------------------------------------------------
    # Session Lifecycle Endpoints
    # ------------------------------------------------------------------

    @app.post("/api/v1/session/start", response_model=dict[str, Any], status_code=status.HTTP_201_CREATED)
    async def start_session(payload: StartSessionPayload) -> dict[str, Any]:
        """Register and start an examination proctoring session."""
        req = StartSessionRequest(
            session_id=payload.session_id or payload.attempt_id,
            attempt_id=payload.attempt_id or payload.session_id,
            candidate_id=payload.candidate_id or payload.user_id,
            user_id=payload.user_id or payload.candidate_id,
            exam_id=payload.exam_id or payload.quiz_id,
            quiz_id=payload.quiz_id or payload.exam_id,
            candidate_name=payload.candidate_name,
            organization_id=payload.organization_id,
            strictness=payload.strictness,
            sampling_fps=payload.sampling_fps,
            enable_wearable_detection=payload.enable_wearable_detection,
            enrolment_id=payload.enrolment_id or payload.candidate_id or payload.user_id or payload.candidate_name,
            metadata=payload.metadata,
        )
        try:
            handle = app.state.service.start_session(req)
            return handle.to_dict()
        except Exception as exc:
            LOGGER.error("Failed to start session: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/v1/enrolment/{enrolment_id}", response_model=dict[str, Any], status_code=status.HTTP_201_CREATED)
    @app.post("/api/v1/candidate/enrol", response_model=dict[str, Any], status_code=status.HTTP_201_CREATED)
    async def enrol_candidate(
        payload: CandidateEnrollPayload,
        enrolment_id: str | None = None,
    ) -> dict[str, Any]:
        """Enroll candidate reference images and build identity verification templates."""
        target_id = enrolment_id or payload.candidate_id or payload.candidate_name
        try:
            result = app.state.service.enrol_candidate(
                enrolment_id=target_id,
                frames=payload.images,
            )
            return result
        except Exception as exc:
            LOGGER.error("Failed to enrol candidate: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/v1/session/{session_id}/pause", response_model=dict[str, Any])
    async def pause_session(session_id: str, payload: ControlPayload) -> dict[str, Any]:
        """Temporarily suspend frame processing for a session."""
        try:
            app.state.service.pause_session(session_id, reason=payload.reason)
            await app.state.ws_hub.broadcast(
                session_id,
                {"event": "SESSION_PAUSED", "session_id": session_id, "reason": payload.reason},
            )
            return {"session_id": session_id, "state": "PAUSED", "reason": payload.reason}
        except ProctoringServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/api/v1/session/{session_id}/resume", response_model=dict[str, Any])
    async def resume_session(session_id: str, payload: ControlPayload) -> dict[str, Any]:
        """Resume frame processing after a pause."""
        try:
            app.state.service.resume_session(session_id, reason=payload.reason)
            await app.state.ws_hub.broadcast(
                session_id,
                {"event": "SESSION_RESUMED", "session_id": session_id, "reason": payload.reason},
            )
            return {"session_id": session_id, "state": "ACTIVE", "reason": payload.reason}
        except ProctoringServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/api/v1/session/{session_id}/finalize", response_model=dict[str, Any])
    async def finalize_session(session_id: str) -> dict[str, Any]:
        """Close session, seal tamper-evident package, and return final summary."""
        try:
            res = app.state.service.finalize_session(session_id)
            await app.state.ws_hub.broadcast(
                session_id,
                {"event": "SESSION_FINALIZED", "session_id": session_id, "manifest_sha256": res.manifest_sha256},
            )
            return res.to_dict()
        except ProctoringServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/api/v1/session/{session_id}/recover", response_model=dict[str, Any])
    async def recover_session(session_id: str) -> dict[str, Any]:
        """Recover an interrupted or crashed session from durable storage."""
        try:
            session_dir = app.state.service.output_dir / session_id
            engine = ProctoringEngine.recover_session(session_dir)
            with app.state.service._lock:
                app.state.service._engines[session_id] = engine
            return {
                "session_id": session_id,
                "state": engine.state.value,
                "recovered_events": len(engine.temporal_aggregator.closed_events),
                "recovered_frames": engine._frame_counter,
            }
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"Session recovery failed: {exc}")

    # ------------------------------------------------------------------
    # Frame and Client Event Ingestion
    # ------------------------------------------------------------------

    @app.post("/api/v1/session/{session_id}/frame", response_model=dict[str, Any])
    async def ingest_frame(session_id: str, payload: FrameIngestPayload) -> dict[str, Any]:
        """Ingest a camera frame, returning immediate observation metadata."""
        try:
            ack = app.state.service.ingest_frame(
                session_id=session_id,
                frame=payload.frame_data,
                timestamp_seconds=payload.timestamp_seconds,
            )
            if ack.active_observations:
                await app.state.ws_hub.broadcast(
                    session_id,
                    {
                        "event": "OBSERVATION_ACTIVE",
                        "session_id": session_id,
                        "frame_index": ack.frame_index,
                        "observations": ack.active_observations,
                    },
                )
            return ack.to_dict()
        except ProctoringServiceError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/v1/session/{session_id}/event", response_model=dict[str, Any])
    async def record_client_event(session_id: str, payload: ClientEventPayload) -> dict[str, Any]:
        try:
            metadata = dict(payload.metadata or {})
            if payload.frame_index is not None:
                metadata["frame_index"] = payload.frame_index
            obs = app.state.service.record_client_event(
                session_id=session_id,
                event_type=payload.event_type,
                timestamp_seconds=payload.timestamp_seconds,
                description=payload.description,
                metadata=metadata,
            )
            await app.state.ws_hub.broadcast(
                session_id,
                {
                    "event": "CLIENT_INCIDENT",
                    "session_id": session_id,
                    "event_type": payload.event_type,
                    "description": payload.description,
                },
            )
            return obs if isinstance(obs, dict) else obs.to_dict()
        except ProctoringServiceError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # Query Endpoints
    # ------------------------------------------------------------------

    @app.get("/api/v1/session/{session_id}/state", response_model=dict[str, Any])
    async def get_session_state(session_id: str) -> dict[str, Any]:
        """Retrieve authoritative engine and attempt state for a session."""
        try:
            engine = app.state.service._require_engine(session_id)
            return {
                "session_id": session_id,
                "engine_state": engine.state.value,
                "is_active": engine.is_active,
                "processed_frames": engine._frame_counter,
                "total_events": len(engine.temporal_aggregator.closed_events),
                "active_incidents": list(engine.temporal_aggregator.active_incidents.keys()),
            }
        except ProctoringServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/v1/session/{session_id}/incidents", response_model=list[dict[str, Any]])
    async def get_session_incidents(session_id: str) -> list[dict[str, Any]]:
        """Retrieve closed events and qualified incidents for reviewer triage."""
        try:
            engine = app.state.service._require_engine(session_id)
            return [ev.to_dict() for ev in engine.temporal_aggregator.closed_events]
        except ProctoringServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/v1/session/{session_id}/review", response_model=dict[str, Any])
    async def get_session_review(session_id: str) -> dict[str, Any]:
        """Retrieve complete reviewer payload with timeline and guidance."""
        try:
            payload = app.state.service.get_review(session_id)
            return payload.to_dict()
        except ProctoringServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/v1/session/{session_id}/evidence/{evidence_id}")
    async def get_session_evidence(session_id: str, evidence_id: str):
        """Retrieve real evidence image artifact with SHA-256 integrity header."""
        evidence_path: Path | None = None
        calc_sha: str | None = None
        try:
            engine = app.state.service._require_engine(session_id)
            evidence_path = engine.evidence_manager.get_evidence_path(evidence_id)
            if evidence_path and evidence_path.exists():
                calc_sha = engine.evidence_manager._compute_sha256(evidence_path)
        except Exception:
            pass

        # Fallback: search session directories on disk
        if not evidence_path or not evidence_path.exists():
            for base_dir in [Path("data/evidence_packages"), Path("data/results/sessions"), Path("data/sessions")]:
                candidate_dir = base_dir / session_id
                if candidate_dir.exists():
                    for sub in ["evidence/frames", "evidence/crops", "evidence/review"]:
                        d = candidate_dir / sub
                        if d.exists():
                            for f in d.glob(f"*{evidence_id}*"):
                                if f.is_file():
                                    evidence_path = f
                                    import hashlib
                                    h = hashlib.sha256()
                                    with open(f, "rb") as fl:
                                        while ch := fl.read(65536):
                                            h.update(ch)
                                    calc_sha = h.hexdigest()
                                    break
                        if evidence_path:
                            break
                if evidence_path:
                    break

        if not evidence_path or not evidence_path.exists():
            raise HTTPException(status_code=404, detail=f"Evidence {evidence_id} not found for session {session_id}")

        return FileResponse(
            path=str(evidence_path),
            media_type="image/jpeg",
            headers={
                "X-Evidence-SHA256": calc_sha or "",
                "X-Evidence-ID": evidence_id,
            },
        )

    # ------------------------------------------------------------------
    # Idempotent Synchronization Ingestion Endpoint
    # ------------------------------------------------------------------

    @app.post("/api/v1/sync/batch", response_model=dict[str, Any])
    async def ingest_sync_batch(payload: SyncBatchPayload) -> dict[str, Any]:
        """Ingest offline event batches idempotently without duplicating logical records."""
        session_id = payload.session_id
        synced_seqs: list[int] = []
        duplicate_seqs: list[int] = []

        for item in payload.events:
            seq = int(item.get("sequence_number", 0))
            event_id = item.get("event_id", "")
            idempotency_key = item.get("idempotency_key") or f"{session_id}:{seq}:{event_id}"

            if idempotency_key in app.state.seen_idempotency_keys:
                duplicate_seqs.append(seq)
            else:
                app.state.seen_idempotency_keys.add(idempotency_key)
                synced_seqs.append(seq)

        ack = SyncBatchAck(
            session_id=session_id,
            synced_sequence_numbers=synced_seqs,
            duplicate_sequence_numbers=duplicate_seqs,
        )
        return ack.to_dict()

    # ------------------------------------------------------------------
    # Real-Time WebSocket Streaming
    # ------------------------------------------------------------------

    @app.websocket("/api/v1/session/{session_id}/ws")
    async def session_websocket(websocket: WebSocket, session_id: str) -> None:
        """Stream real-time incidents, diagnostics, and state changes to proctor clients."""
        await app.state.ws_hub.connect(session_id, websocket)
        try:
            # Send initial welcome state
            await websocket.send_json(
                {
                    "event": "CONNECTED",
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            while True:
                # Keep connection alive; accept ping/heartbeat from client
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            app.state.ws_hub.disconnect(session_id, websocket)
        except Exception:
            app.state.ws_hub.disconnect(session_id, websocket)

    return app


# Default app instance for ASGI servers (e.g. uvicorn proctoring.integration.api:app)
app = create_app()
