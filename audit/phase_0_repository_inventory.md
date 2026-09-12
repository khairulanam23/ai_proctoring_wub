# Phase 0 Audit: Repository Inventory & Live Inference Architecture

**Date**: 2026-09-12  
**Auditor**: Senior AI Systems Engineer  
**Repository**: `/home/phant0m/Phantom/ai_proctoring_wub`  
**Execution Phase**: Phase 0 — Establish the Real Baseline  

---

## 1. Actual System Entrypoints

| Interface | File Path | Entrypoint Function / Command | Status | Description |
| :--- | :--- | :--- | :--- | :--- |
| **CLI Dispatcher** | [proctoring/cli/\_\_main\_\_.py](file:///home/phant0m/Phantom/ai_proctoring_wub/proctoring/cli/__main__.py) | `python -m proctoring.cli [live\|video]` | `EXISTS & CONNECTED` | Top-level CLI router dispatching to live webcam HUD or recorded video analysis. |
| **Live Camera CLI** | [proctoring/cli/live.py](file:///home/phant0m/Phantom/ai_proctoring_wub/proctoring/cli/live.py) | `proctoring.cli.live.main()` | `EXISTS & CONNECTED` | OpenCV HighGUI window with real-time HUD showing bounding boxes, status banner, and event alerts. |
| **REST / WS API** | [proctoring/integration/api.py](file:///home/phant0m/Phantom/ai_proctoring_wub/proctoring/integration/api.py) | `create_app()` / `uvicorn proctoring.integration.api:app` | `EXISTS & CONNECTED` | FastAPI application exposing `/api/v1/session/*`, `/api/v1/health`, and WebSocket broadcasting. |
| **Service Facade** | [proctoring/integration/service.py](file:///home/phant0m/Phantom/ai_proctoring_wub/proctoring/integration/service.py) | `ProctoringService` | `EXISTS & CONNECTED` | Multi-session manager coordinating session lifecycles, frame ingestion, and outbox event dispatching. |
| **Core Engine** | [proctoring/engine.py](file:///home/phant0m/Phantom/ai_proctoring_wub/proctoring/engine.py) | `ProctoringEngine.process_frame()` | `EXISTS & CONNECTED` | Stateful per-session orchestrator moving frames through stages 3 to 8. |
| **Model Download** | [scripts/download_models.py](file:///home/phant0m/Phantom/ai_proctoring_wub/scripts/download_models.py) | `python scripts/download_models.py` | `EXISTS & CONNECTED` | Downloads YuNet ONNX, SFace ONNX, and MediaPipe landmarkers. |
| **Master Benchmark**| [tools/benchmark/runner.py](file:///home/phant0m/Phantom/ai_proctoring_wub/tools/benchmark/runner.py) | `Phase5BenchmarkRunner.run_full_benchmark()` | `PARTIAL & MOCKED` | Synthetic benchmark suite; contains mock object detection rectangles. |
| **Baseline Profiler**| [tools/benchmark/profile_pipeline.py](file:///home/phant0m/Phantom/ai_proctoring_wub/tools/benchmark/profile_pipeline.py) | `python tools/benchmark/profile_pipeline.py` | `EXISTS & CONNECTED` | Reproducible empirical stage-by-stage profiler created for Phase 0. |
| **Memory Profiler** | [tools/benchmark/test_memory_leak.py](file:///home/phant0m/Phantom/ai_proctoring_wub/tools/benchmark/test_memory_leak.py) | `python tools/benchmark/test_memory_leak.py` | `EXISTS & CONNECTED` | Reproducible memory stability and lifecycle test harness created for Phase 0. |

---

## 2. Runtime Model Registry & Hardware Mapping

All runtime execution details are empirically verified against active hardware (AMD Ryzen CPU + NVIDIA GeForce RTX 3060 12GB VRAM).

| Model | Purpose | Runtime Engine | Device | Input Shape | Output Shape | Actual Usage Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenCV YuNet** (`face_detection_yunet_2023mar.onnx`) | Primary Face Detection | OpenCV DNN (`cv2.FaceDetectorYN`) | **CPU** (OpenCV MLAS SGEMM) | `(H, W, 3)` uint8 BGR | `(N, 15)` float (bbox, 5 landmarks, score) | `ACTUALLY USED` | OpenCV wheel compiled without CUDA DNN target; CPU latency is ~7.0 ms. |
| **OpenCV SFace** (`face_recognition_sface_2021dec.onnx`) | Identity Face Verification | OpenCV DNN (`cv2.FaceRecognizerSF`) | **CPU** (OpenCV MLAS SGEMM) | `(112, 112, 3)` aligned face | `(1, 128)` float32 embedding vector | `ACTUALLY USED` | Cosine similarity against enrolled template. Latency is ~13.8 ms. |
| **Ultralytics YOLO11n** (`yolo11n.pt`) | Prohibited Object Detection | PyTorch 2.14.0+cu130 | **GPU (`cuda:0`)** | `(1, 3, 640, 640)` float32 | `(1, 84, 8400)` candidate detections | `ACTUALLY USED` | Detects cell phones, laptops, books. VRAM usage: ~52-67 MB. GPU utilization: <1%. Latency is ~5.1 ms. |
| **MediaPipe Face Landmarker** (`face_landmarker.task`) | Facial Dynamics, Gaze, Lip Motion | MediaPipe Tasks Python | **CPU** (TFLite XNNPACK) | `(H, W, 3)` uint8 BGR | 478 3D landmarks + 52 blendshapes | `ACTUALLY USED` | Used for head pose, looking away, and speech articulation proxy. Latency: ~8.4 ms. |
| **MediaPipe Hand Landmarker** (`hand_landmarker.task`) | Hand Presence, Posture, Coordinates | MediaPipe Tasks Python | **CPU** (TFLite XNNPACK) | `(H, W, 3)` uint8 BGR | 21 3D landmarks per hand (up to 2 hands) | `ACTUALLY USED` | Used for hand-near-face and hand-near-ear detection. Latency: ~13.4 ms. |
| **Paper Detector** (`proctoring/analysis/paper.py`) | Physical Paper Contour Detection | Algorithmic OpenCV / NumPy | **CPU** | `(H, W, 3)` uint8 BGR | Bounding quadrilaterals, aspect ratio | `CONNECTED BUT UNCONSUMED` | Runs every frame (0.5 ms), but `BehaviourObserver.map_to_events()` never references it. |
| **YOLO-World** (`yolov8s-world.pt`) | Wearable Earbud / Headphone Detection | Ultralytics Open-Vocabulary | **GPU (`cuda:0`)** | Full frame + cropped ear regions | Detection boxes & classes | `OPTIONAL & DISABLED` | Disabled by default (`enable_wearable_detection: False`). At 640x480 resolution, earbuds are only 10-15 px, yielding poor precision. |
| **CLIP ViT-B-32** (`weights/clip/ViT-B-32.pt`) | Visual Representation | PyTorch | N/A | N/A | N/A | `DEAD` | 338 MB unreferenced artifact on disk; never imported or loaded anywhere. |
| **Root `yolo11n.pt`** | Object Detection | N/A | N/A | N/A | N/A | `DUPLICATE` | Exact byte-for-byte duplicate of `models/yolo11n.pt`. |

---

## 3. Real Live Inference Path (Trace Analysis)

The live frame execution path proceeds through the following verified call sequence:

```text
Camera / Video Input Frame
  │
  ▼
1. Frame Ingestion & Session State Check
   proctoring.engine.ProctoringEngine.process_frame(frame, frame_index, timestamp_seconds)
   ├── Check engine state (PAUSED / CREATED / RUNNING)
   └── Initialize FrameObservation & FrameTimingRecord
  │
  ▼
2. Quality & Preprocessing Gate
   proctoring.preprocessing.frame_quality.FrameQualityGate.process(frame)
   ├── AdaptiveImagePreprocessor: luminance check & optional CLAHE enhancement
   ├── CameraHealthTracker: freeze detection (frame delta hash) & disconnect check
   └── Drops corrupt, undersized (<160x120), or severely blurred frames
  │
  ▼
3. Face Detection & Tracking (YuNet)
   proctoring.engine_stages.EngineStages.detect_faces(working_frame)
   ├── cv2.FaceDetectorYN.detect(working_frame) [CPU MLAS SGEMM]
   ├── Filter background faces (<40px min size, border margins)
   └── Returns list[FaceDetection] or None (records equipment failure if faulted)
  │
  ▼
4. Identity Verification (SFace)
   proctoring.engine_stages.EngineStages.resolve_identity(working_frame, faces)
   ├── cv2.FaceRecognizerSF.alignCrop(working_frame, face)
   ├── cv2.FaceRecognizerSF.feature(aligned_face) -> 128-dim unit vector
   └── Compare cosine similarity against enrolled templates (default threshold: 0.363)
  │
  ▼
5. Object Detection (YOLO11n)
   proctoring.engine_stages.EngineStages.detect_objects(working_frame)
   ├── ultralytics.YOLO(working_frame, device="cuda:0") [PyTorch CUDA]
   └── ObjectRelevanceFilter: filters COCO classes to exam-prohibited items (phone, laptop, book)
  │
  ▼
6. Behavioural Analysis
   proctoring.engine_stages.EngineStages.analyze_behaviour(working_frame, obs)
   ├── FacialDynamicsAnalyzer.analyze(working_frame) [MediaPipe Face Landmarker]
   │   └── Head pose angles (pitch, yaw, roll), gaze vector, lip articulation ratio
   ├── HandAnalyzer.analyze(working_frame) [MediaPipe Hand Landmarker]
   │   └── Hand bounding boxes, hand-near-face, hand-near-ear, kinematics tracking
   ├── OcclusionClassifier.classify(...) -> determines face obstruction vs low light
   ├── WearableDetector.detect(...) [YOLO-World] (if enabled on cadence)
   └── PaperDetector.detect(working_frame) [OpenCV Contours] (executed but output unmapped)
  │
  ▼
7. Temporal Aggregation & Incident Qualification
   proctoring.temporal.aggregator.UnifiedTemporalAggregator
   ├── update_face_observation(): debounces NO_FACE, MULTIPLE_FACES, UNKNOWN_FACE
   ├── update_object_observations(): runs PhoneHandDisambiguator (aspect ratio + persistence)
   └── update_behaviour_observations(): evaluates BehaviourObserver.map_to_events(obs)
       └── Stateful transitions: DETECTED -> CONFIRMING -> OBSERVED -> ENDED
  │
  ▼
8. Evidence Packaging & Progressive Journal Spooling
   ├── EngineEvidenceManager._retain_evidence_frames(): saves full frame & annotated crop
   ├── SessionJournalManager.append_event(ev): appends sealed EventRecord to events.jsonl
   └── Checkpoint persistence: writes session_checkpoint.json every 25 frames
  │
  ▼
9. Output Dispatch & API Transmission
   ├── Return FrameObservation to caller
   ├── Broadcast active observations over WebSocketHub to connected proctor clients
   └── On session completion: Engine.finalize_session() -> seals manifest.json & SHA-256
```

---

## 4. Documented Architectural Gaps & Disconnections

1. **Audio Proctoring Capability Gap (`NOT IMPLEMENTED`)**:
   - Audio capture, Voice Activity Detection (VAD), and acoustic timeline synchronization are completely absent from the codebase.
   - All speech observations currently rely exclusively on visual blendshape lip motion (`mouth_open`, `jaw_open`) via MediaPipe Face Landmarker.

2. **Paper Detection Output Disconnection (`DISCONNECTED`)**:
   - `PaperDetector.detect()` runs on every frame and populates `obs.paper_analysis`.
   - However, `BehaviourObserver.map_to_events()` never queries `obs.paper_analysis`. No paper-related events (`PAPER_PRESENT`, `PAPER_ABSENT`, `WRITING`) are ever emitted to the temporal aggregator.

3. **Hand Kinematics Disconnection (`DISCONNECTED`)**:
   - `HandAnalyzer._update_kinematics()` tracks velocity vectors and state machines (`RESTING`, `MOVING`, `REACHING`) internally.
   - However, neither `BehaviourObserver` nor `UnifiedTemporalAggregator` ever consumes `hand.kinematics`.

4. **Weak Hand-Phone Disambiguation (`HEURISTIC ONLY`)**:
   - `PhoneHandDisambiguator` relies on a static aspect ratio filter ($1.3 \le \text{AR} \le 2.6$) and confirms a phone incident after just 2 frames ($0.50\,\text{s}$ at $4\,\text{FPS}$).
   - It lacks spatial hand-to-phone contact tracking, kinematic correlation, or hard-negative suppression for closed fists.

5. **Wearable Visual Detection Resolution Limit (`UNRELIABLE AT 480P`)**:
   - Visual earbud detection via YOLO-World runs full-frame or on upscaled ear crops ($320\,\text{px}$).
   - In standard $640\times 480$ webcam streams, an earbud occupies only $10\text{--}15\,\text{px}$, which is well below the reliable object resolution limit for open-vocabulary detectors, producing high false-negative and false-positive rates.
