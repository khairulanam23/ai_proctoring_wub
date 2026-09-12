# AI Proctoring Engine — Production Deployment & Architecture Manual

```text
===============================================================================
AI PROCTORING ENGINE — PRODUCTION FROZEN (VERSION 6.0)
Hardware Target: NVIDIA GPU (RTX 3060 12GB Dev / 24GB Target) + x86_64 CPU
Inference: PyTorch CUDA (YOLO11n), ONNX Runtime CUDA (YuNet, SFace), CPU XNNPACK (MediaPipe)
Integration: ExamController Wire Protocol (HTTP REST / SSE / WebSockets on Port 7001)
===============================================================================
```

---

## 1. Project Purpose & Ethical Invariants

The **AI Proctoring Engine** is an in-house, evidence-first computer vision and audio telemetry verification service engineered for integration with the **ExamController** examination platform.

### Core Ethical & Operational Invariants
1. **Human-in-the-Loop Verdicts**: The AI system **never generates automated cheating determinations**, sanctions, or disqualifications. It functions strictly as an evidentiary assistant that extracts, timestamps, and cryptographically packages forensic observations. The human proctor/invigilator retains sole authority over student intent.
2. **Zero Opaque Suspicion Scores**: The system rejects cumulative "trust scores", cheating probabilities, and black-box risk metrics.
3. **Equipment Fault Invariant**: Physical hardware, network, and model anomalies (`TECHNICAL_DIAGNOSTIC`) are strictly separated from student observations (`CANDIDATE_OBSERVATION`). A camera disconnection, frame drop, or microphone clipping is never transformed into candidate suspicion.
4. **Deterministic Auditing**: All session observations generate immutable, timestamped keyframe snapshots and an append-only journal sealed with an SHA-256 cryptographic manifest.

---

## 2. Architecture Overview & Component Responsibilities

The system decouples real-time stream ingestion, neural inference, temporal aggregation, and storage:

```
                  +-------------------------------------------------+
                  |      ExamController Web Client / Electron      |
                  +-----------------------+-------------------------+
                                          | HTTP Multipart Frames (Port 7001)
                                          v
+-----------------------------------------------------------------------------------+
|                        FASTAPI INTEGRATION SERVICE BOUNDARY                       |
|                          (proctoring/integration/api.py)                          |
+-----------------------------------------+-----------------------------------------+
                                          | Raw Frames & Audio Chunks
                                          v
+-----------------------------------------------------------------------------------+
|                           PROCTORING PIPELINE ENGINE                              |
|                             (proctoring/engine.py)                                |
|  +--------------------+  +----------------------+  +---------------------------+  |
|  | CameraHealthAssess |  | QualityAssessment    |  | Face Preprocessing (CLAHE)|  |
|  +--------------------+  +----------------------+  +---------------------------+  |
|  +-----------------------------------------------------------------------------+  |
|  |                          MODEL REGISTRY (SHARED VRAM)                       |  |
|  |   - YuNet ONNX (ORT CUDA)                 - SFace ONNX (ORT CUDA)           |  |
|  |   - YOLO11n (PyTorch CUDA)                - MediaPipe Landmarks (CPU)       |  |
|  +-----------------------------------------------------------------------------+  |
|  +-----------------------------------------------------------------------------+  |
|  |                              STAGE ANALYZERS                                |  |
|  |   - MultiSubjectTracker (Spatial + Cosine Fusion)                           |  |
|  |   - PhoneHandDisambiguator (Empty-hand FP dismissal)                        |  |
|  |   - PaperDetector & HandwritingAnalyzer (Kinematic micro-oscillations)       |  |
|  |   - Gaze & HeadPoseTrackers               - Modular Audio VAD & Correlation |  |
|  +-----------------------------------------------------------------------------+  |
|  +-----------------------------------------------------------------------------+  |
|  |                      TEMPORAL EVENT AGGREGATOR & OUTBOX                     |  |
|  |   - Debounce & Persistence Filter         - Append-Only Journal (JSONL)     |  |
|  |   - Offline Synchronous Outbox            - Keyframe Evidence Annotator     |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                       TAMPER-EVIDENT EVIDENCE STORAGE                             |
|               (manifest.sha256, events.json, timeline.jsonl, frames/)             |
+-----------------------------------------------------------------------------------+
```

---

## 3. End-to-End Pipeline & Runtime Architecture

### 3.1 Live Video Pipeline
1. **Ingestion**: Receives raw BGR frames (640x480 native target, 15–30 FPS).
2. **Health Assessment**: Evaluates freeze conditions, timestamps gaps (>5s), and aspect ratios. Sub-resolution corrupted frames (<160x120) are quarantined.
3. **Quality Gate**: Computes Laplacian variance blur score, luminance percentiles, and contrast. Applies CLAHE adaptive histogram equalization under low-light conditions.
4. **Primary Face Detection (YuNet)**: Runs on NVIDIA GPU via ONNX Runtime CUDA provider (~3.99 ms latency). Returns bounding boxes and 5 facial landmarks.
5. **Face Verification (SFace)**: Extracts 128-dimensional identity embeddings on CUDA (~1.19 ms). Matches against enrolled student templates using cosine distance thresholds (`COSINE_MATCH_THRESHOLD = 0.3630`).
6. **Object Detection (YOLO11n)**: Executes on CUDA via PyTorch Native (~4.76 ms). Detects cell phones, laptops, books, and secondary persons.
7. **Facial & Hand Landmarks (MediaPipe)**: Executes on CPU via optimized TensorFlow Lite XNNPACK delegates. Extracts 468 face blendshapes and 21 3D hand joints.
8. **Behavioral Disambiguation**: Cross-references object bounding boxes against hand skeleton joints to eliminate false-positive phone detections on empty cupped hands.
9. **Paper & Handwriting Analysis**: Evaluates quadrilateral desk contours and kinematic micro-oscillations for writing behavior in physical exam modes.

### 3.2 Live Audio Pipeline (`proctoring/audio/`)
- **Ingestion**: Processes raw 16 kHz 16-bit PCM audio streams in 500 ms windows.
- **Voice Activity Detection (VAD)**: Calculates short-term energy and spectral centroid thresholds.
- **Multimodal Correlation**: Cross-references audio speech timestamps against candidate visual mouth blendshapes (`jawOpen`, `mouthPucker`). Flags discrepancies when speech occurs while candidate lips remain closed (`EXTERNAL_VOICE_SUSPECTED`).

### 3.3 GPU/CPU Execution & Hardware Allocation
| Model / Component | Framework & Backend | Device Placement | Latency (Mean) | VRAM (Single) | VRAM (4 Concurrency) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YuNet Face Detector** | ONNX Runtime 1.30.0 (`CUDAExecutionProvider`) | `cuda:0` | 3.99 ms | ~24 MB | Shared |
| **SFace Face Recognizer**| ONNX Runtime 1.30.0 (`CUDAExecutionProvider`) | `cuda:0` | 1.19 ms | ~40 MB | Shared |
| **YOLO11n Object Detector**| PyTorch 2.11 / CUDA 13.0 Native | `cuda:0` | 4.76 ms | ~20 MB | Shared |
| **Face & Hand Landmarkers**| MediaPipe Tasks (TFLite CPU XNNPACK) | CPU (Multi-core)| 16.50 ms | 0 MB (Host RAM)| 0 MB |
| **Complete Pipeline** | Multimodal Unified Orchestration | Mixed GPU/CPU | **27.18 ms** | **84.32 MB** | **180.32 MB** |

---

## 4. Model Inventory & Lifecycle Management

All model weights are centralized in `models/`:
- `face_detection_yunet_2023mar.onnx` (232 KB): YuNet ONNX face detection model.
- `face_recognition_sface_2021dec.onnx` (38.7 MB): SFace ONNX feature extractor.
- `face_landmarker.task` (3.7 MB): MediaPipe 468-point face landmarker and blendshape model.
- `hand_landmarker.task` (7.8 MB): MediaPipe 21-point hand joint tracking model.
- `yolo11n.pt` (5.6 MB): Ultralytics YOLO11 nano general object detector.
- `yolov8s-world.pt` (27.2 MB): YOLO-World open-vocabulary wearable detector (headphones/earbuds).

### ModelRegistry Lifecycle (`proctoring/core/model_registry.py`)
To prevent duplicate CUDA allocations across concurrent exam sessions:
- Models are instantiated as **thread-safe resident singletons** protected by re-entrant mutexes.
- Concurrent sessions share model weights in VRAM, eliminating cold-start latency spikes.
- VRAM footprint scales conservatively from **84.32 MB** (1 session) to **180.32 MB** (4 sessions).

---

## 5. Session Lifecycle, State Isolation & Crash Recovery

### 5.1 State Machine
```text
[UNINITIALIZED] ──> [INITIALIZED] ──> [RUNNING / ACTIVE] <───> [PAUSED]
                                             │
                                             ├──> [COMPLETED] (Normal finalization)
                                             └──> [CRASH / TERMINATION]
                                                         │
                                                         v
                                              [RECOVERY_REQUIRED]
                                                         │
                                      recover_session()  v
                                              [RUNNING / ACTIVE]
```

### 5.2 Session Isolation
- Each exam session maintains an isolated output directory (`data/results/<session_id>/`).
- Candidate facial embeddings, tracking states, calibration baselines, and temporal aggregators are instantiated per session and cleared on finalization. No cross-session memory leakage.

### 5.3 Crash & Restart Recovery
If the host server abruptly reboots or the process terminates:
1. `ProctoringEngine.recover_session(session_dir)` inspects `timeline.jsonl` and checkpoint state.
2. Reconciles exact frame counters (`max(timeline_frames) + 1`), restoring active session state.
3. Automatically transitions state to `RUNNING` (`EngineState.ACTIVE`).
4. Subsequent frames are ingested seamlessly without resetting previous session history.

---

## 6. Tamper-Evident Evidence Architecture

Every finalized exam generates a tamper-evident package at `data/results/<session_id>/`:
```
data/results/<session_id>/
├── events.json           # Qualified forensic incidents with timestamps & bounding boxes
├── timeline.jsonl        # Append-only frame-by-frame telemetry record
├── telemetry.json        # FPS, latency, and device diagnostic summaries
├── evidence/             # Annotated keyframe JPEG captures
│   ├── ev_0001_phone.jpg
│   └── ev_0002_person.jpg
├── manifest.json         # SHA-256 file table of all package contents
└── manifest.sha256       # Detached SHA-256 signature sealing manifest.json
```

### Integrity Verification
The package can be verified offline via system tools or Python:
```bash
cd data/results/<session_id>
sha256sum -c manifest.sha256
```
Or programmatically:
```python
from proctoring.evidence.package import SessionEvidencePackage

pkg = SessionEvidencePackage(session_dir)
is_valid, errors = pkg.verify_package_integrity()
assert is_valid is True and len(errors) == 0
```

---

## 7. ExamController Integration & Wire Protocols

The engine serves an HTTP/REST and Server-Sent Events (SSE) API on port **7001** matching ExamController's `WubProctoringProvider` contract:

### Key REST Endpoints
| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` / `/api/v1/health` | Service uptime, GPU VRAM allocation, and model health |
| `GET` | `/api/v1/models` | Status and hardware placement of all loaded neural networks |
| `POST`| `/api/v1/session/start` | Starts an exam session with candidate identity & config |
| `POST`| `/api/v1/session/{id}/frame` | Ingests multipart JPEG frame, returns observation payload |
| `GET` | `/api/v1/session/{id}/evidence/{ev_id}`| Retrieves evidence image with `X-Evidence-SHA256` header |
| `POST`| `/api/v1/session/{id}/finalize` | Compiles evidence package, writes manifest, emits SHA-256 |

### Observation Response Schema (`POST /api/v1/session/{id}/frame`)
```json
{
  "session_id": "exam_101",
  "frame_index": 42,
  "timestamp_seconds": 10.5,
  "accepted": true,
  "face_detected": true,
  "identity_verified": true,
  "events": [
    {
      "event_type": "PHONE_INTERACTION_OBSERVED",
      "severity": "HIGH",
      "confidence": 0.89,
      "bounding_box": [120, 240, 60, 110],
      "evidence_id": "ev_0042_phone"
    }
  ]
}
```

---

## 8. Installation & Environment Setup

### 8.1 Hardware & Driver Requirements
- **OS**: Linux x86_64 (Ubuntu 22.04 LTS / Debian 12 recommended)
- **NVIDIA GPU**: RTX 3060 (12GB) or production server GPU (24GB VRAM)
- **NVIDIA Driver**: Version 550+
- **CUDA Toolkit**: CUDA 12.0+ or CUDA 13.0+
- **Python**: Python 3.10 through 3.14

### 8.2 Virtual Environment Installation
```bash
# 1. Clone repository and enter directory
cd /home/phant0m/Phantom/ai_proctoring_wub

# 2. Initialize virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and build tools
pip install --upgrade pip setuptools wheel

# 4. Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 5. Install dependencies in editable mode
pip install -e ".[all]"

# 6. Verify GPU runtime
python -c "import torch, onnxruntime as ort; print('PyTorch CUDA:', torch.cuda.is_available()); print('ORT Providers:', ort.get_available_providers())"
```

---

## 9. Operation & Deployment

### 9.1 Foreground Development Runner
```bash
source .venv/bin/activate
uvicorn proctoring.integration.api:app --host 0.0.0.0 --port 7001 --log-level info
```

### 9.2 Background Production Daemon
Use the pre-configured production launcher:
```bash
./scripts/start_production_service.sh
```

### 9.3 Systemd Service Setup
Install the production unit for automated boot startup and fault restart:
```bash
sudo cp deployment/ai-proctoring.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-proctoring.service

# Verify service status
sudo systemctl status ai-proctoring.service
```

---

## 10. Verification & Test Execution

### 10.1 Complete Regression Suite (455 Tests)
```bash
.venv/bin/pytest -q
# Expected output: 455 passed, 1 skipped in ~160s
```

### 10.2 Camera Lifecycle & Recovery Suite
```bash
.venv/bin/pytest tests/integration/test_camera_lifecycle.py -vv -s
# Validates Scenarios A through G (100% pass)
```

### 10.3 ExamController Live Wire Suite
```bash
.venv/bin/pytest tests/integration/test_exam_controller_live_wire.py -vv -s
```

### 10.4 Performance & Latency Benchmark
```bash
.venv/bin/python tools/benchmark/validate_live_streams.py --frames 120
# Expected throughput: >30 FPS, Mean Latency: <30 ms
```

### 10.5 Concurrency Scaling Benchmark
```bash
.venv/bin/python tools/benchmark/benchmark_concurrency.py --sessions 1,2,4 --frames 30
```

### 10.6 Long-Run Stability & Leak Audit
```bash
.venv/bin/python tools/benchmark/validate_long_run_stability.py --sessions 3 --frames-per-session 150
# Verifies +0 MB VRAM and +0 thread leaks
```

---

## 11. Known Limitations & Unverified Accuracy Areas

1. **In-Ear Earbud Detection**: At standard 640x480 webcam distances, in-ear earbuds span fewer than 15 pixels and are frequently obscured by hair. Such detections are marked `EARBUD_SUSPECTED` at `MEDIUM` severity, requiring human keyframe inspection.
2. **Extreme Dark Lighting (<10 Lux)**: While CLAHE histogram equalization improves contrast, extreme darkness degrades YuNet landmark confidence. Under these conditions, the engine emits `LOW_LIGHTING_CONDITION` diagnostic events rather than guessing face locations.
3. **Audio Speaker Disambiguation**: The audio VAD module identifies vocal presence and mouth blendshape correlation but does not perform biometric voiceprint speaker diarization.

---

## 12. Directory Structure

```
ai_proctoring_wub/
├── audit/                          # Forensic audit reports (Phases 0 through 8)
├── deployment/                     # Production systemd unit definitions
│   └── ai-proctoring.service
├── models/                         # Unified model weights (ONNX, TFLite, PyTorch)
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_landmarker.task
│   ├── face_recognition_sface_2021dec.onnx
│   ├── hand_landmarker.task
│   ├── yolo11n.pt
│   └── yolov8s-world.pt
├── proctoring/                     # Production source code
│   ├── analysis/                   # Behavioral analyzers (gaze, hands, paper, phone, wearables)
│   ├── audio/                      # Audio VAD & multimodal correlation
│   ├── capture/                    # Camera drivers & video sampling
│   ├── core/                       # Event definitions, contracts, persistence, model registry
│   ├── detection/                  # Neural network detection wrappers (YuNet, SFace, YOLO)
│   ├── evidence/                   # Tamper-evident packaging, annotation, SHA-256 sealing
│   ├── integration/                # FastAPI boundary, schemas, wire service, offline outbox
│   ├── preprocessing/              # Camera health & frame quality assessment
│   ├── temporal/                   # Temporal debounce & qualification aggregators
│   └── tracking/                   # Spatial + cosine multi-subject tracking
├── scripts/                        # Production execution scripts
│   └── start_production_service.sh
├── tests/                          # Automated pytest regression suite (456 tests)
├── tools/                          # Benchmarking, profiling & dataset validation tools
├── project_state.md                # Authoritative project state specification (v6.0)
└── README.md                       # Production deployment manual (this file)
```

---

## 13. Production Freeze Status

```text
===============================================================================
ENGINE STATUS: PRODUCTION FROZEN (VERSION 6.0)
===============================================================================
All Phase 8 validation criteria have been met.
The repository is frozen for normal feature development.
Deployments must execute via scripts/start_production_service.sh or systemd.
===============================================================================
```
