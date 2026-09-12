# AI Proctoring Engine

```text
========================================================================================
AI PROCTORING ENGINE — AUTHORITATIVE TECHNICAL ARCHITECTURE & OPERATIONS MANUAL
Production Status: FROZEN BASELINE (VERSION 6.0)
Target Integration: ExamController Examination Platform (REST / WebSocket / SSE Port 7001)
Runtime Hardware: NVIDIA GPU (RTX 3060 12GB Dev / 24GB Server) + x86_64 Multicore CPU
Execution Architecture: PyTorch CUDA (YOLO11n), ONNX Runtime CUDA (YuNet, SFace), CPU XNNPACK (MediaPipe)
========================================================================================
```

---

## 1. Overview

The **AI Proctoring Engine** is an in-house computer vision and audio telemetry analysis service engineered to provide deterministic, evidence-first proctoring observations for live and recorded examinations. The engine integrates with the university-owned **ExamController** platform.

The system continuously evaluates incoming video frames and optional audio chunks, runs deep-learning detectors across GPU and CPU hardware, maintains spatial and temporal tracking across subjects, and produces structured observations with supporting visual keyframes.

The engine acts as an evidentiary and forensic assistant. It captures, timestamps, and cryptographically packages objective observations for human review. It is not an automated grading system, an autonomous disciplinary agent, or a black-box cheating classifier.

---

## 2. Design Principles

The architecture enforces seven non-negotiable engineering principles:

1. **Evidence-First Architecture**: The engine produces observations, events, and evidence files—never a cheating verdict or disciplinary sanction. The human invigilator remains the sole authority for evaluating candidate intent.
2. **Zero Cumulative Suspicion Scores**: The system rejects cumulative suspicion scores, cheating probabilities, and trust indices. There is no counter that marks a student as "guilty" when a numerical threshold is crossed.
3. **Strict Separation of Technical Faults**: Hardware, network, and model anomalies (`TECHNICAL_DIAGNOSTIC`) are strictly separated from candidate behaviors (`CANDIDATE_OBSERVATION`). A frozen webcam, delivery gap, or audio clipping event is never converted into candidate suspicion.
4. **Session Isolation**: All candidate biometrics (facial embeddings), spatial tracking histories, calibration baselines, and temporal aggregators are strictly scoped to an individual session lifecycle. No cross-session state persists in memory.
5. **Deterministic Traceability**: Every event links a unique `event_id` to an exact `session_id`, `frame_index`, sub-second `timestamp_seconds`, spatial bounding box, and on-disk evidence keyframe.
6. **Durable Persistence & Crash Resilience**: Session state is persisted to an append-only journal (`timeline.jsonl`) on disk. If the AI service process terminates abruptly, the engine recovers the session without state loss or counter desynchronization.
7. **Production vs. Test Separation**: Production inference pathways contain zero mocks, synthetic fallbacks, or simulated shortcuts. Synthetic test media is quarantined exclusively within test suites and benchmark tooling.

---

## 3. System Architecture

```text
                           +-------------------------------------------------+
                           |      ExamController Web / Electron Client       |
                           +-----------------------+-------------------------+
                                                   | HTTP / REST / WS (Port 7001)
                                                   v
+----------------------------------------------------------------------------------------------------+
|                               FASTAPI INTEGRATION SERVICE BOUNDARY                                 |
|                                  (proctoring/integration/api.py)                                   |
|   - Session Lifecycle Management                      - Frame & Event Ingestion API                |
|   - Real-Time WebSocket Streaming Hub                 - Offline Sync & Idempotency Buffer          |
+--------------------------------------------------+-------------------------------------------------+
                                                   | Raw Media Buffers
                                                   v
+----------------------------------------------------------------------------------------------------+
|                                    PROCTORING PIPELINE ENGINE                                      |
|                                       (proctoring/engine.py)                                       |
|                                                                                                    |
|  +---------------------------+   +----------------------------+   +-----------------------------+  |
|  |   CameraHealthMonitor     |   |   FrameQualityAssessor     |   |   FacePreprocessor (CLAHE)  |  |
|  +---------------------------+   +----------------------------+   +-----------------------------+  |
|                                                                                                    |
|  +----------------------------------------------------------------------------------------------+  |
|  |                                MODEL REGISTRY (SHARED GPU VRAM)                              |  |
|  |   - YuNet Face Detector (ORT CUDA)                - SFace Face Recognizer (ORT CUDA)         |  |
|  |   - YOLO11n Object Detector (PyTorch CUDA)        - MediaPipe Landmarkers (CPU TFLite)       |  |
|  +----------------------------------------------------------------------------------------------+  |
|                                                                                                    |
|  +----------------------------------------------------------------------------------------------+  |
|  |                                      PIPELINE STAGES                                         |  |
|  |   - MultiSubjectTracker (IoU + Cosine Fusion)     - PhoneHandDisambiguator                   |  |
|  |   - PaperDetector & HandwritingAnalyzer           - Gaze & HeadPoseTrackers                  |  |
|  |   - WearableDetector (YOLO-World Open Vocab)      - Audio VAD & Multimodal Correlator        |  |
|  +----------------------------------------------------------------------------------------------+  |
|                                                                                                    |
|  +----------------------------------------------------------------------------------------------+  |
|  |                                TEMPORAL AGGREGATOR & OUTBOX                                  |  |
|  |   - Minimum Duration Debounce Filter              - Candidate vs Equipment Partitioning      |  |
|  |   - Append-Only Journal Writer (timeline.jsonl)   - Keyframe Annotation & SHA-256 Hashing    |  |
|  +----------------------------------------------------------------------------------------------+  |
+--------------------------------------------------+-------------------------------------------------+
                                                   | Sealed Output Package
                                                   v
+----------------------------------------------------------------------------------------------------+
|                                  FORENSIC EVIDENCE REPOSITORY                                      |
|                          (data/results/<session_id>/ / manifest.sha256)                            |
+----------------------------------------------------------------------------------------------------+
```

---

## 4. End-to-End Data Flow

The lifecycle of an examination frame proceeds through twelve distinct stages:

```text
1. Exam Session Creation    --> POST /api/v1/session/start initializes session directory and config.
2. Ingestion                --> POST /api/v1/session/{id}/frame receives base64/binary JPEG frame.
3. Health Validation        --> CameraHealthMonitor checks timestamp continuity, freeze, and resolution.
4. Quality Gate             --> FrameQualityAssessor measures blur (Laplacian variance) and luminance.
5. Primary Face Detection   --> YuNet executes on CUDA via ONNX Runtime (returns bbox + 5 landmarks).
6. Face Verification        --> SFace extracts 128-d embedding on CUDA; matches enrolled candidate template.
7. Object Detection         --> YOLO11n executes on CUDA via PyTorch (detects phones, laptops, books, people).
8. Landmark Extraction      --> MediaPipe executes on CPU XNNPACK (468 face blendshapes, 21 hand joints).
9. Behavioral Analysis      --> Spatial disambiguation (hand curl vs phone, paper quadrilateral tracking).
10. Temporal Qualification  --> Aggregator applies debounce windows (filtering transient 1-frame noise).
11. Journal & Evidence      --> Frame telemetry appends to timeline.jsonl; keyframe JPEG written to disk.
12. Delivery                --> Real-time WebSocket broadcasts observation; REST response returns ack.
```

---

## 5. AI Processing Pipeline

### 5.1 Video Pipeline & Camera Resilience
- **Input Resolution**: Native 640x480 BGR frames sampled at 2.0 to 4.0 FPS (adaptive).
- **Health Checks**:
  - *Freeze Detection*: Computes pairwise frame MSE. Emits `CAMERA_FRAME_FROZEN` if unchanged for >3.0 seconds.
  - *Delivery Gaps*: Identifies pauses >5.0 seconds between frames, resets freeze timers, and records `CAMERA_DISCONNECTED` followed by `CAMERA_RESTORED`.
  - *Malformed Frame Quarantine*: Sub-resolution frames (<160x120) or zero-byte buffers are rejected (`accepted=False`) without polluting stream aspect ratios.

### 5.2 Face Detection and Verification
- **Face Detector (YuNet)**: Runs via ONNX Runtime CUDA provider (~3.99 ms latency). Emits bounding boxes, confidence scores, and 5 facial landmarks (eyes, nose, mouth corners).
- **Face Recognizer (SFace)**: Generates 128-dimensional L2-normalized feature vectors on CUDA (~1.19 ms). Matches against enrolled templates using cosine distance (`COSINE_MATCH_THRESHOLD = 0.3630`). Emits `UNKNOWN_FACE` or `FACE_MISMATCH` when unconfirmed subjects appear.
- **Occlusion Analysis**: Analyzes eye, nose, and mouth landmark visibility. Emits `FACE_OCCLUDED` when landmarks are masked by clothing or hands.

### 5.3 Object Detection
- **Detector (YOLO11n)**: Ultralytics YOLO11 nano executing on CUDA via native PyTorch (~4.76 ms).
- **Class Filtering**: Isolates COCO class indices for cell phones (`cell phone`), laptops (`laptop`), books (`book`), and additional individuals (`person`). Emits `PROHIBITED_OBJECT` or `PHONE_DETECTED`.

### 5.4 Hand Analysis
- **Hand Landmarker (MediaPipe)**: Multi-hand tracking model on CPU XNNPACK extracting 21 3D joint coordinates per hand.
- **Kinematics**: Computes fingertip velocities, wrist spatial positions, and desk workspace boundaries. Detects `HAND_NEAR_FACE`, `HAND_NEAR_EAR`, and workspace departures.

### 5.5 Phone / Hand Disambiguation (`proctoring/analysis/phone_disambiguation.py`)
Standard object detectors produce false positives when a candidate rests an empty cupped hand near their chin or desk.
- The disambiguator cross-references YOLO phone bounding boxes against MediaPipe hand landmark geometry.
- Evaluates hand finger curl (MCP-to-TIP distance ratio) and palm orientation.
- If an empty curled hand overlaps a low-confidence phone box without an opaque rectangular surface, the detection is dismissed as an empty hand.
- High-confidence phone detections verified across multiple consecutive frames generate `PHONE_DETECTED`. Ambiguous overlaps generate `PHONE_CANDIDATE_UNCERTAIN` for reviewer verification.

### 5.6 Paper Detection & Handwriting Analysis (`proctoring/analysis/paper.py` & `hands.py`)
In written examination modes (`PHYSICAL_PAPER`):
- **Paper Detection**: Evaluates desk plane contours using adaptive thresholding and polygonal approximation. Identifies quadrilaterals matching standard paper aspect ratios (A4 / Letter). Emits `PAPER_PRESENT`, `PAPER_ABSENT`, or `MULTIPLE_PAPERS_DETECTED`.
- **Handwriting Analysis**: Evaluates pen-grip kinematics and micro-oscillations in the dominant hand within the paper bounding polygon. Differentiates `HAND_WRITING` (rhythmic micro-movements) from `HAND_RESTING` (static posture).

### 5.7 Facial Dynamics & Gaze Tracking
- **Landmarker (MediaPipe Face Mesh)**: Computes 468 3D facial mesh points and 52 canonical blendshapes on CPU.
- **Eye Gaze Tracking**: Evaluates iris center position relative to inner and outer eye canthi. Detects sustained lateral or downward screen departures (`GAZE_OFF_SCREEN`).
- **Head Pose Estimation**: Solves perspective-n-point (PnP) using canonical 3D facial landmark points (nose tip, chin, eye corners). Computes Euler angles (yaw, pitch, roll). Emits `LOOKING_AWAY` when yaw exceeds ±25° or pitch exceeds ±20°.
- **Blink & Liveness**: Monitors Eye Aspect Ratio (EAR). Sustained absence of blinks triggers `POSSIBLE_PRESENTATION_ATTACK` to flag static photograph presentation.

### 5.8 Multi-Subject Tracking (`proctoring/tracking/tracker.py`)
Eliminates naive single-face assumptions (`detections[0]`):
- **Track Lifecycle**: State machine transitions through `TENTATIVE` -> `CONFIRMED` -> `COASTING` -> `DELETED`.
- **Association Cost Matrix**: Fuses spatial intersection-over-union (IoU) with facial embedding cosine similarity.
- **Secondary Subject Classification**: Identifies candidate identity vs. secondary background individuals. Tracks multiple simultaneous people, emitting `MULTIPLE_PERSONS_OBSERVED` or `PERSON_ENTERED_FRAME`.

### 5.9 Temporal Reasoning (`proctoring/temporal/aggregator.py`)
Isolated single-frame anomalies frequently represent sensor noise or brief adjustments:
- Individual detections must persist across a parameterized temporal window (default: 1.5 to 3.0 seconds) to qualify as a formal event.
- Integrates debounce hysteresis: an event remains open until the condition clears continuously for `absence_tolerance_seconds`.
- Summarizes open and closed incident timelines with start, peak, and end timestamps.

### 5.10 Audio Pipeline (`proctoring/audio/`)
- **Format**: Ingests single-channel 16 kHz 16-bit PCM audio streams in 500 ms chunks.
- **Voice Activity Detection (VAD)**: Computes short-term energy and spectral centroid distributions against an adaptive noise floor. Emits `ACOUSTIC_SPEECH_DETECTED`.
- **Degradation Detection**: Identifies microphone clipping, extreme background noise, or buffer underruns, recording `AUDIO_DEGRADED` as a technical diagnostic.

### 5.11 Multimodal Correlation (`proctoring/audio/multimodal.py`)
Correlates acoustic speech events with facial visual blendshapes:
- Cross-references audio voice activity against visual mouth articulation blendshapes (`jawOpen`, `mouthPucker`).
- **`MULTIMODAL_SPEECH_CONGRUENT`**: Acoustic speech coincides with visible candidate mouth movements (candidate speaking).
- **`ACOUSTIC_SPEECH_WITHOUT_LIP_MOVEMENT`**: Voice activity is detected while candidate lips remain closed, flagging possible secondary individuals speaking in the room.

---

## 6. Event and Observation Architecture

Events are partitioned by category in `proctoring/core/events.py`:

```text
EventCategory.CANDIDATE_OBSERVATION (Candidate-Related Behaviors)
├── Face & Identity:  NO_FACE, MULTIPLE_FACES, UNKNOWN_FACE, FACE_MISMATCH, FACE_OCCLUDED
├── Attention & Pose: LOOKING_AWAY, SUSPICIOUS_HEAD_POSE, GAZE_OFF_SCREEN
├── Devices:          PHONE_DETECTED, PHONE_CANDIDATE_UNCERTAIN, HEADPHONES_DETECTED, EARBUDS_SUSPECTED
├── Paper & Hands:    PAPER_PRESENT, PAPER_ABSENT, HAND_WRITING, HAND_RESTING, HAND_NEAR_EAR
└── Speech:           CANDIDATE_SPEAKING, ACOUSTIC_SPEECH_WITHOUT_LIP_MOVEMENT, MULTIPLE_SPEAKERS

EventCategory.TECHNICAL_DIAGNOSTIC (Equipment & Stream Anomalies)
├── Camera:           CAMERA_FRAME_FROZEN, CAMERA_DISCONNECTED, CAMERA_OBSTRUCTED
├── Audio:            AUDIO_DEGRADED
└── System:           SYSTEM_ERROR, DETECTOR_ERROR, SESSION_PAUSED, SESSION_RESUMED
```

---

## 7. Evidence Architecture & Storage Layout

When a candidate observation qualifies, the engine creates forensic evidence assets in `data/results/<session_id>/`:

```text
data/results/<session_id>/
├── events.json                 # Formatted list of closed events with metadata & bounding boxes
├── timeline.jsonl              # Frame-by-frame telemetry journal (FPS, latency, detections)
├── telemetry.json              # Session-level aggregate performance metrics & hardware summaries
├── manifest.json               # Canonical SHA-256 digest catalog of every file in the package
├── manifest.sha256             # Detached SHA-256 integrity hash sealing manifest.json
└── evidence/
    ├── frames/                 # Raw, unmodified source frames captured at incident peak
    │   ├── ev_0001_raw.jpg
    │   └── ev_0002_raw.jpg
    └── review/                 # Annotated frames with bounding boxes, landmark overlays, and HUD
        ├── ev_0001_annotated.jpg
        └── ev_0002_annotated.jpg
```

---

## 8. Evidence Integrity & Tamper Detection

The system uses SHA-256 integrity hashing to secure forensic evidence packages:

```text
Evidence Files (frames/, json records)
          │
          ▼
   manifest.json  (Canonical JSON mapping each relative path to its SHA-256 hex digest)
          │
          ▼
  manifest.sha256 (Detached file containing the SHA-256 hex digest of manifest.json)
```

### Verification Procedure
Packages can be validated offline using standard UNIX coreutils:
```bash
cd data/results/<session_id>
sha256sum -c manifest.sha256
```
Or programmatically via the Python API:
```python
from proctoring.evidence.package import SessionEvidencePackage

pkg = SessionEvidencePackage(session_dir)
is_valid, errors = pkg.verify_package_integrity()
if not is_valid:
    print("Tamper detected:", errors)
```
*Note on Terminology*: The integrity seal is a **cryptographic hash** (SHA-256 digest). It detects accidental corruption or unauthorized post-exam modification of evidence files. It is not an asymmetric public-key digital signature.

---

## 9. Session Lifecycle & Crash Recovery

### 9.1 Engine State Machine
The lifecycle is defined by `EngineState` in `proctoring/engine.py`:
- `UNINITIALIZED`: Engine created, session configuration loaded.
- `INITIALIZED`: Models and storage structures prepared.
- `RUNNING` (canonical alias: `ACTIVE`): Ingesting frames and running inference.
- `PAUSED`: Frame ingestion temporarily suspended by proctor action.
- `RECOVERY_REQUIRED`: Stream interrupted or process restarted after an abrupt failure.
- `COMPLETED`: Session finalized, package sealed, resources released.

### 9.2 Crash Recovery Architecture
If the AI service terminates unexpectedly (server reboot, power outage, OOM, SIGKILL):
1. Mid-session state is preserved on disk in the append-only journal (`timeline.jsonl`) and periodic checkpoints (`checkpoint.json`).
2. An administrator or service restart invokes `ProctoringEngine.recover_session(session_dir)` or `POST /api/v1/session/{id}/recover`.
3. **Journal Counter Reconciliation**: If the service crashed between checkpoints, the engine reads `timeline.jsonl` to extract `max(timeline_frame_indices) + 1`, preventing frame counter rollback.
4. The engine restores active tracking baselines and transitions to `RUNNING` (`ACTIVE`).
5. Subsequent frame submissions resume seamlessly without duplicating existing events or corrupting the evidence package.

---

## 10. ExamController Integration Boundary

The engine provides an HTTP/REST, Server-Sent Events (SSE), and WebSocket service on port **7001**, implementing ExamController's `WubProctoringProvider` interface.

### REST API Reference
| Method | Endpoint | Description | Request Payload | Response Schema |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service readiness and GPU hardware stats | None | `HealthResponse` |
| `GET` | `/api/v1/models` | Resident model catalog & active thresholds | None | `ModelCatalog` |
| `POST` | `/api/v1/session/start` | Register and initialize an exam session | `StartSessionPayload` | `SessionHandle` (HTTP 201) |
| `POST` | `/api/v1/candidate/enrol` | Enrol reference images for facial matching | `CandidateEnrollPayload` | `EnrollmentResult` |
| `POST` | `/api/v1/session/{id}/frame` | Ingest base64 JPEG frame for inference | `FrameIngestPayload` | `FrameAck` |
| `POST` | `/api/v1/session/{id}/event` | Ingest client event (tab-switch, blur) | `ClientEventPayload` | `ObservationSummary` |
| `GET` | `/api/v1/session/{id}/state` | Get current frame count, state, incidents | None | `SessionStateSummary` |
| `GET` | `/api/v1/session/{id}/incidents`| List closed candidate events and timestamps | None | `List[IncidentSummary]` |
| `GET` | `/api/v1/session/{id}/review` | Complete review timeline and guidance | None | `ReviewPayload` |
| `GET` | `/api/v1/session/{id}/evidence/{ev_id}`| Retrieve JPEG image + `X-Evidence-SHA256` | None | JPEG Binary Stream |
| `POST` | `/api/v1/session/{id}/finalize` | Close session, seal package, emit SHA-256 | None | `SessionResult` |
| `POST` | `/api/v1/session/{id}/recover` | Recover crashed session from storage | None | `RecoverySummary` |
| `POST` | `/api/v1/sync/batch` | Idempotent bulk sync of offline events | `SyncBatchPayload` | `SyncBatchAck` |
| `WS` | `/api/v1/session/{id}/ws` | Live bidirectional event stream | WebSocket Ping/Pong | Real-time Incident JSON |

*Authentication Notice*: In current v6.0 deployment, authentication is handled at the network boundary or API gateway. The internal service boundary validates requests by `session_id` and strict schema constraints.

---

## 11. Model Inventory & Runtime Device Placement

All neural network assets are consolidated in `models/`:

| Model Asset | Size | Framework | Device Placement | Runtime Consumer | Execution Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `face_detection_yunet_2023mar.onnx` | 232 KB | ONNX Runtime | **`cuda:0`** (ORT CUDA) | `FaceDetector` | Face detection & 5 landmarks (~3.99 ms) |
| `face_recognition_sface_2021dec.onnx`| 38.7 MB | ONNX Runtime | **`cuda:0`** (ORT CUDA) | `FaceVerifier` | 128-d face identity embedding (~1.19 ms) |
| `yolo11n.pt` | 5.6 MB | Ultralytics | **`cuda:0`** (PyTorch CUDA)| `ObjectDetector` | Object detection (phones, laptops, people) (~4.76 ms) |
| `yolov8s-world.pt` | 27.2 MB | Ultralytics | **`cuda:0`** (PyTorch CUDA)| `WearableDetector` | Open-vocabulary wearable audio devices |
| `face_landmarker.task` | 3.7 MB | MediaPipe | **CPU** (XNNPACK) | `FacialDynamics` / `Gaze` | 468 3D facial mesh & blendshapes (~16.5 ms) |
| `hand_landmarker.task` | 7.8 MB | MediaPipe | **CPU** (XNNPACK) | `HandAnalyzer` | 21 3D hand joints & writing tracking |

### Shared Model Lifecycle (`proctoring/core/model_registry.py`)
To prevent VRAM duplication across concurrent exam sessions:
- Neural models are initialized as **resident singletons** managed by `ModelRegistry`.
- Multiple concurrent session threads invoke identical resident GPU weights, avoiding repeated CUDA context initialization.
- Base VRAM footprint remains steady at ~84 MB for single sessions and ~180 MB under 4 concurrent sessions.

---

## 12. Performance & Concurrency Benchmarks

### 12.1 Single-Stream Latency & Throughput (RTX 3060 12GB)
Benchmarked via `tools/benchmark/validate_live_streams.py` over 120 continuous frames:

| Metric | Measured Value | Operational SLA | Status |
| :--- | :--- | :--- | :--- |
| **Effective Throughput** | **36.33 FPS** | >= 15.0 FPS | **Passed (+142% over SLA)** |
| **Mean Frame Latency** | **27.18 ms** | < 100.0 ms | **Passed (Real-Time Interactive)** |
| **Median (p50) Latency**| **27.93 ms** | < 100.0 ms | Passed |
| **90th Percentile (p90)**| **35.12 ms** | < 100.0 ms | Passed |
| **95th Percentile (p95)**| **38.52 ms** | < 100.0 ms | Passed |
| **99th Percentile (p99)**| **47.68 ms** | < 100.0 ms | Passed |
| **Frame Acceptance Rate**| **100% (120/120)** | 100% | Zero Dropped Frames |

### 12.2 Multi-Session Concurrency Scaling (RTX 3060 12GB)
Benchmarked via `tools/benchmark/benchmark_concurrency.py`:

| Concurrent Sessions | Aggregate Throughput | Throughput / Session | Mean Latency | 95th Percentile | VRAM Allocated | Errors |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 Session** | 13.51 FPS | 13.51 FPS | 45.78 ms | 53.27 ms | 84.32 MB | 0 |
| **2 Sessions** | 27.12 FPS | 13.56 FPS | 50.19 ms | 56.19 ms | 116.32 MB | 0 |
| **4 Sessions** | 44.10 FPS | 11.03 FPS | 63.79 ms | 79.08 ms | 180.32 MB | 0 |

### 12.3 Long-Run Stability & Leak Verification
Benchmarked via `tools/benchmark/validate_long_run_stability.py` across 3 sequential sessions (450 frames total):
- **VRAM Growth**: **+0.00 MB** (no GPU memory leak).
- **Thread Count Growth**: **+0 threads** (no background thread retention).
- **File Descriptor Growth**: **+0 descriptors** (no unclosed files/sockets).
- **Host RSS Memory**: +38.0 MB post-warmup plateau (stable allocator reuse).

---

## 13. System Requirements & Fresh Machine Setup

### 13.1 Requirements
- **Operating System**: Linux x86_64 (Ubuntu 22.04 LTS or Debian 12 recommended).
- **NVIDIA GPU**: RTX 3060 (12GB) minimum for development; 24GB GPU (RTX 4090, A10, L40S) recommended for multi-stream production servers.
- **NVIDIA Driver**: Version 550 or higher.
- **CUDA Runtime**: CUDA 12.0+ or CUDA 13.0+ with cuDNN 9.x.
- **CPU**: 4+ physical cores (Intel Core i5/i7 or AMD Ryzen 5/7) for MediaPipe multithreading.
- **System Memory**: Minimum 8 GB RAM (service RSS footprint ~1.8–2.2 GB).
- **Storage**: High-speed NVMe SSD with at least 500 MB free space per 3-hour exam session.

### 13.2 Step-by-Step Installation
```bash
# 1. Clone the repository
git clone https://github.com/khairulanam23/ai_proctoring_wub.git
cd ai_proctoring_wub

# 2. Verify NVIDIA Driver and GPU Hardware
nvidia-smi

# 3. Create Python Virtual Environment (Python 3.10 through 3.14 supported)
python3 -m venv .venv
source .venv/bin/activate

# 4. Upgrade Build Tools
pip install --upgrade pip setuptools wheel

# 5. Install PyTorch with CUDA Support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 6. Install Package with All Detection & Behavior Extensions
pip install -e ".[all]"

# 7. Verify Model Weights Presence
ls -lh models/
# Expected: face_detection_yunet_2023mar.onnx, face_recognition_sface_2021dec.onnx,
# face_landmarker.task, hand_landmarker.task, yolo11n.pt, yolov8s-world.pt
```

### 13.3 Runtime & CUDA Verification Snippet
```bash
python -c "import torch, onnxruntime as ort; \
print('PyTorch CUDA Available:', torch.cuda.is_available()); \
print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); \
print('ONNX Runtime Providers:', ort.get_available_providers())"
```
*Expected Output*:
```text
PyTorch CUDA Available: True
Device Name: NVIDIA GeForce RTX 3060
ONNX Runtime Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## 14. Configuration & Environment Variables

Key runtime parameters can be passed via environment variables or configured in `SessionConfig` (`proctoring/config.py`):

| Variable | Purpose | Default | Required |
| :--- | :--- | :--- | :--- |
| `AI_HOST` | Host IP address for service bind | `0.0.0.0` | No |
| `AI_PORT` | Listening port for ExamController integration | `7001` | No |
| `AI_WORKERS` | Number of Uvicorn ASGI worker processes | `1` | No |
| `CUDA_VISIBLE_DEVICES` | Physical GPU index to bind for inference | `0` | No |
| `PROCTORING_CAMERA_INDEX`| Video device index when reading direct webcam | `0` | No |
| `PYTHONUNBUFFERED` | Disable stdout/stderr buffering for systemd logs | `1` | No |

---

## 15. Service Deployment & Operations

### 15.1 Foreground Development Runner
```bash
source .venv/bin/activate
uvicorn proctoring.integration.api:app --host 0.0.0.0 --port 7001 --log-level info
```

### 15.2 Production Launcher Script
Use the verified production script with model pre-flight checks and signal handling:
```bash
./scripts/start_production_service.sh
```

### 15.3 Systemd Service Setup (Daemon Mode)
Install the service unit template located at `deployment/ai-proctoring.service`:
```bash
# 1. Copy service file to systemd directory
sudo cp deployment/ai-proctoring.service /etc/systemd/system/

# 2. Reload systemd daemon
sudo systemctl daemon-reload

# 3. Enable service for automatic boot start
sudo systemctl enable ai-proctoring.service

# 4. Start the service
sudo systemctl start ai-proctoring.service

# 5. Inspect status and logs
sudo systemctl status ai-proctoring.service
journalctl -u ai-proctoring.service -f
```

---

## 16. Health Checks, Diagnostics & Logging

### Health Endpoint
Query the service health probe to inspect model placement and GPU VRAM:
```bash
curl -s http://127.0.0.1:7001/api/v1/health | jq .
```
Expected response:
```json
{
  "status": "ok",
  "engine_version": "1.0.0",
  "models_loaded": {
    "face_detector_yunet": true,
    "face_verifier_sface": true,
    "face_landmarker": true,
    "hand_landmarker": true,
    "object_detector_yolo11": true
  },
  "active_sessions_count": 0,
  "hardware": {
    "cuda_available": true,
    "vram_total_mb": 11902.81,
    "vram_allocated_mb": 0.0
  }
}
```

### Diagnostic Logging
The engine logs structured messages using Python's standard `logging` module under the `proctoring` namespace.
- Model loading failures emit `ERROR` or `CRITICAL` with tracebacks.
- Degraded camera frames emit `WARNING` with frame metrics (e.g. Laplacian variance, aspect ratio).
- Session transitions emit `INFO` (`SESSION_STARTED`, `SESSION_RESUMED`, `SESSION_FINALIZED`).

---

## 17. Testing & Verification

The test suite contains 456 automated test cases executed via pytest:

```bash
# Run entire test suite
.venv/bin/pytest -q
```
*Current Result*: **455 passed, 1 skipped, 0 failed in ~162s**.

### Targeted Test Suites
```bash
# 1. Camera Lifecycle Suite (Scenarios A through G)
.venv/bin/pytest tests/integration/test_camera_lifecycle.py -vv -s

# 2. Session Recovery & Isolation Suite
.venv/bin/pytest tests/core/test_session_recovery.py -vv -s

# 3. ExamController Wire REST & WebSocket Suite
.venv/bin/pytest tests/integration/test_exam_controller_live_wire.py -vv -s

# 4. Multi-Subject Tracking Suite
.venv/bin/pytest tests/tracking/test_multi_subject_tracking.py -vv -s
```

*Note on Skipped Test*: `tests/analysis/test_facial_dynamics.py:349` is intentionally skipped during automated runs because the synthetic portrait sample happened to be too geometrically similar to trigger the identity guard.

---

## 18. Operational Benchmarking Tools

Operational validation scripts reside in `tools/benchmark/`:

1. **Continuous Live Stream Validator**:
   ```bash
   .venv/bin/python tools/benchmark/validate_live_streams.py --frames 120
   ```
   Measures effective throughput (FPS), mean latency, latency percentiles, and verifies evidence package integrity.

2. **Concurrency Scaling Benchmark**:
   ```bash
   .venv/bin/python tools/benchmark/benchmark_concurrency.py --sessions 1,2,4 --frames 30
   ```
   Evaluates multi-session aggregate throughput, per-session latency, and VRAM scaling.

3. **Long-Run Stability & Leak Validator**:
   ```bash
   .venv/bin/python tools/benchmark/validate_long_run_stability.py --sessions 3 --frames-per-session 150
   ```
   Measures VRAM, thread, and file descriptor growth across sequential sessions.

---

## 19. Failure Handling & Operational Resilience

| Failure Scenario | Engine Behavior | Evidentiary / Diagnostic Output |
| :--- | :--- | :--- |
| **Camera Disconnected / Gaps** | Tracks delivery gap (>5.0s). Pauses frame sampling; clears freeze timer. | Emits `CAMERA_DISCONNECTED` followed by `CAMERA_RESTORED`. |
| **Camera Video Freeze** | Evaluates frame MSE. Flags identical frames persisting >3.0 seconds. | Emits `CAMERA_FRAME_FROZEN` with timestamp. |
| **Malformed Frame (<160x120)** | Quality filter rejects corrupt input; preserves stream aspect ratio. | Frame rejected (`accepted=False`); pipeline remains stable. |
| **Service Crash / Process Exit** | Session directory preserves journal (`timeline.jsonl`) & snapshots. | `POST /api/v1/session/{id}/recover` restores state and exact counter. |
| **CUDA / GPU Failure** | `ModelRegistry` falls back to CPU execution providers where supported. | Health check reports CPU fallback; emits `SYSTEM_ERROR` diagnostic. |
| **Audio Clipping / Distortion** | Audio analyzer detects energy threshold saturation. | Emits `AUDIO_DEGRADED` diagnostic without accusing candidate. |
| **Evidence Tampering** | Manifest hash verification compares on-disk files against SHA-256 table. | `verify_package_integrity()` returns `False` and lists altered files. |

---

## 20. Security & Privacy Considerations

1. **Local Boundary Isolation**: The service binds to `0.0.0.0:7001` or `127.0.0.1:7001`. It should be hosted behind a secure reverse proxy or internal Docker/VPC network.
2. **Server-Side Biometrics**: Raw facial embedding vectors are stored ephemerally in session memory and never transmitted over external network boundaries.
3. **Data Retention**: The AI engine stores session artifacts in `data/results/<session_id>/`. Archival and deletion policies are governed by ExamController platform settings.
4. **Secret Management**: The codebase contains zero hardcoded API keys, passwords, or credentials.

---

## 21. Accuracy Validation Status

### Engineering Validation (VERIFIED)
All computer vision pipelines, multi-subject tracking logic, spatial disambiguation filters, crash recovery mechanisms, and REST/WebSocket protocols have been verified via unit tests, integration test suites, and empirical synthetic benchmarks.

### Real-World Accuracy Validation (UNVERIFIED)
The current development and testing environment utilizes standard public benchmark portraits (e.g. LFW subsets) and synthetic test media. **Real-world precision, recall, false-positive rate (FPR), and false-negative rate (FNR) across diverse demographic populations and lighting environments have not yet been evaluated against an independently labelled university exam dataset.**

Because accuracy remains unverified in real-world environments:
- The system must **never** be configured to execute automated sanctions or disqualifications.
- Every flagged observation must be reviewed by a human proctor alongside the associated keyframe snapshot.

---

## 22. Known Limitations

1. **In-Ear Earbud Detection**: At standard 640x480 webcam distances, in-ear earbuds occupy fewer than 15 pixels and are frequently occluded by hair. The engine marks such events as `EARBUDS_SUSPECTED` at `MEDIUM` severity. Human visual confirmation is required.
2. **MediaPipe CPU Placement**: Face and hand landmarking runs on CPU via TensorFlow Lite XNNPACK delegates. While fast (~16.5 ms), high concurrency (>8 simultaneous sessions per host) will saturate CPU cores unless distributed across multiple worker nodes.
3. **Low-Light Operation (<10 Lux)**: CLAHE histogram equalization improves contrast, but extreme darkness degrades YuNet landmark confidence. Under severe darkness, the engine emits `CAMERA_OBSTRUCTED` or `NO_FACE` rather than fabricating coordinates.
4. **Single-Channel Audio**: The audio analyzer evaluates acoustic voice presence and energy, but cannot perform acoustic spatial direction-of-arrival (DoA) triangulation without a multi-microphone array.

---

## 23. Repository Directory Structure

```text
ai_proctoring_wub/
├── audit/                          # Authoritative engineering reports (Phases 0–8)
│   ├── README.md                   # Directory catalog and lifecycle guide
│   ├── phase_0_baseline.md         # Baseline CPU/GPU profiling
│   ├── ...
│   ├── phase_8_final_production_validation.md # Final production validation report
│   └── results/                    # Historical JSON benchmark outputs
├── colab/                          # Standalone Google Colab notebook
├── data/                           # Data directories and test samples
│   ├── ground_truth/               # Evaluation annotations
│   ├── results/                    # Benchmark JSON reports & session outputs
│   ├── samples/                    # Standardized facial portraits & synthetic media
│   └── students/                   # Candidate enrollment reference storage
├── deployment/                     # Production service unit definitions
│   └── ai-proctoring.service       # Systemd service unit template
├── docs/                           # Architectural, detection, and pipeline documentation
├── models/                         # Unified neural network model weights (~80 MB)
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_landmarker.task
│   ├── face_recognition_sface_2021dec.onnx
│   ├── hand_landmarker.task
│   ├── yolo11n.pt
│   └── yolov8s-world.pt
├── proctoring/                     # Core production Python package
│   ├── analysis/                   # Behavioral analyzers (gaze, hands, paper, phone, wearables)
│   ├── audio/                      # Audio VAD & multimodal correlation
│   ├── capture/                    # Video acquisition & camera health
│   ├── cli/                        # Command-line interface entrypoints
│   ├── core/                       # Contracts, model registry, events, persistence
│   ├── detection/                  # Deep learning detector wrappers (YuNet, SFace, YOLO)
│   ├── evidence/                   # Tamper-evident packaging & SHA-256 sealing
│   ├── integration/                # FastAPI boundary, schemas, wire service, outbox
│   ├── preprocessing/              # Camera health & frame quality assessment
│   ├── telemetry/                  # Performance, GPU diagnostics, timeline
│   └── tracking/                   # Spatial + cosine multi-subject tracking
├── scripts/                        # Production execution & utility scripts
│   ├── download_models.py          # Standalone model downloader
│   └── start_production_service.sh # Authoritative background daemon launcher
├── tests/                          # Automated pytest regression suite (456 tests)
├── tools/                          # Operational benchmarking & profiling utilities
│   ├── benchmark/                  # Latency, concurrency, and leak validators
│   ├── field_testing/              # Dataset generators & long session harnesses
│   └── hardening/                  # Recovery & stress testing harnesses
├── training/                       # Permanent regression dataset
│   └── regression/cases.json
├── project_state.md                # Authoritative project state specification (v6.0 FROZEN)
├── pyproject.toml                  # Packaging, dependency & tool configuration
└── README.md                       # Production operations manual (this document)
```

---

## 24. Maintenance and Versioning

Future maintenance must respect the production freeze:
1. **Dependency Updates**: Update dependencies in `pyproject.toml` and verify with `.venv/bin/pytest`.
2. **Model Upgrades**: New models must be evaluated against regression datasets in `training/regression/` and approved through champion/challenger comparative evaluation before updating `models/`.
3. **Regression Testing**: Any code change must pass all 456 regression tests and maintain 0 VRAM leak across long-run benchmarks.
4. **Integration Compatibility**: REST and WebSocket contracts carry `CONTRACT_VERSION = "1.0"`. Any breaking change requires a version bump coordinated with the ExamController engineering team.

---

## 25. Production Freeze Declaration

```text
========================================================================================
AI PROCTORING ENGINE STATUS: PRODUCTION-FROZEN BASELINE (VERSION 6.0)
========================================================================================
The AI Proctoring Engine has completed all engineering phases (Phases 0 through 8).
The system is PRODUCTION READY for deployment with ExamController.

Normal feature development is FROZEN. Maintenance is restricted to security
patches, bug fixes, and gated model promotions.
========================================================================================
```

---

## 26. Attribution & Authorship

Developed by **Mohammad Khairul Anam** while working in **Computing and Information Services Ltd. (CIS)** for **World University of Bangladesh**.
