# FORENSIC AUDIT REPORT: WUB AI PROCTORING SYSTEM
**Audit Date:** September 18, 2026  
**Auditor:** Senior Forensic AI & System Architecture Auditor  
**Scope:** Standalone AI Engine (`ai_proctoring_wub`), ExamController Integration (`exam-controller-app`), Moodle LMS Boundary (`moodle`), External Academic Identity (`https://api.e-dhrubo.com`)  
**Target Repository:** `/home/phant0m/Phantom/ai_proctoring_wub` (Commit `34bbf57`, Tag `v6.0.0`)  
**Integration Boundary:** `/home/phant0m/Phantom/exam-controller-app` (Commit `632275a`)  
**Institutional Boundary:** `/home/phant0m/Phantom/moodle` (Commit `a987843d5`, Moodle 5.2.2+)  
**Audit Mode:** STRICT READ-ONLY FORENSIC INSPECTION — ZERO CODE/MODEL/CONFIG MODIFICATIONS

---

## 1. EXECUTIVE SUMMARY

A forensic, cross-system architectural and empirical machine learning audit was conducted across the World University of Bangladesh (WUB) AI Proctoring ecosystem. The audit inspected the standalone computer vision microservice (`ai_proctoring_wub`), its client-side and backend integration boundaries in ExamController (`exam-controller-app`), the institutional academic boundary in Moodle (`local_examcontroller`), and external identity resolution via UMS (`api.e-dhrubo.com`).

### Primary Empirical Findings:
1. **Architecture & Philosophy Adherence:** The standalone AI engine and ExamController backend strictly adhere to an **evidence-first, non-scoring philosophy**. Across both codebases, no cumulative cheating score, student suspicion score, guilt probability, or autonomous disciplinary decision is calculated or persisted. Explicit test assertions enforce that `risk_score`, `suspicion_score`, and `cheating_score` never exist in API schemas.
2. **GPU / CUDA Acceleration Status:** Hardware acceleration on the host NVIDIA GeForce RTX 3060 (12 GB VRAM, Compute Capability 8.6, Driver 595.91.07, CUDA 13.2) is **partially active and verified working**:
   - **Ultralytics YOLO11n:** Runs on PyTorch CUDA (`cuda:0`). Measured average inference latency is **3.65 ms (273.7 FPS)** with **67.34 MB VRAM** allocated.
   - **OpenCV YuNet (Face Detector):** Successfully accelerated via ONNXRuntime with `CUDAExecutionProvider`. Measured latency is **5.01 ms (199.6 FPS)**.
   - **OpenCV SFace (Face Embedder):** Successfully accelerated via ONNXRuntime with `CUDAExecutionProvider`. Measured latency is **0.97 ms (1025.9 FPS)**.
   - **MediaPipe Face & Hand Landmarkers:** Execute strictly on **CPU via TensorFlow Lite XNNPACK delegate** (8.64 ms for Face Landmarker, 9.90 ms for Hand Landmarker). MediaPipe Tasks Python does not provide native CUDA execution on Linux.
   - **Total End-to-End Pipeline Latency:** Measured at **28.17 ms (~35.5 FPS peak)**, comfortably exceeding the 4 FPS adaptive sampling target.
3. **Critical Integration Breach (P0):** A critical architectural defect was uncovered in ExamController's frontend adapter (`modules/ai-integration/src/application/wub-proctoring-provider.ts`). Lines 304–329 inspect raw, single-frame detections (`ack.detected_objects`) returned in the HTTP frame acknowledgment, and immediately synthesize high-severity `PHONE_DETECTED` and `UNAUTHORIZED_OBJECT` observations with hardcoded confidence `0.85`, **completely bypassing the AI engine's server-side temporal qualification and hysteresis engine**. Ordinary desk objects (laptops used for taking the exam, water bottles, textbooks, pencil cases) that flicker for a single frame immediately trigger critical disciplinary alerts on the proctor dashboard.
4. **Out-of-Distribution Detection & False-Positive Risk (P0):** The system relies on generic COCO weights for YOLO11n. COCO class 67 (`cell phone`) was never trained on scientific calculators (e.g., Casio fx-991ES), dark rectangular power banks, black notebooks, or student ID cards. In `PhoneHandDisambiguator`, any detection with raw confidence $\ge 0.85$ bypasses temporal confirmation and geometry checks, immediately declaring a `CONFIRMED_PHONE`.
5. **Absence of Real-World Validation Dataset (P0):** The repository's `data/samples` directory contains only 7 static JPEG images of historical US/UK politicians (Colin Powell, George W. Bush, Tony Blair) and 1 synthetic video. **Zero real-world validation data exists** representing Bangladeshi university students, local lighting variations, diverse skin tones, optical glare on eyeglasses, camera motion blur, or actual exam hall desk arrangements. Passing 486 unit tests proves code execution, not ML generalization or accuracy.
6. **Air-Gap / Offline Crash Vulnerability (P0):** Open-vocabulary wearable detection (`WearableDetector` using `yolov8s-world.pt`) attempts to dynamically download 338 MB of CLIP ViT-B/32 text encoder weights from external public CDNs upon initialization. In an air-gapped or restricted campus network, enabling this feature results in an unhandled timeout or crash.
7. **Audio / Acoustic Proctoring Missing:** No audio capture, WebRTC audio ingestion, Voice Activity Detection (VAD), or speech classification pipeline is implemented. The directory `proctoring/audio/` contains only static multimodal stubs.
8. **Institutional Boundary Respected:** Moodle (`local_examcontroller`) contains zero AI proctoring code, zero cheating scores, and zero execution state. The academic/institutional boundary remains completely uncompromised.

---

## 2. AUDIT SCOPE & METHODOLOGY

The audit evaluated four distinct systems and their operational boundaries:

```
+-----------------------------------------------------------------------------------+
|                              WUB AUDIT BOUNDARIES                                 |
+-----------------------------------------------------------------------------------+
| [D. UMS (api.e-dhrubo.com)]                                                       |
|       | (Permanent Student ID, Academic Enrollment)                               |
|       v                                                                           |
| [C. Moodle LMS (moodle/)]                                                         |
|       | (Courses, Cohorts, Curricula, Roster Sync via local_examcontroller)       |
|       v                                                                           |
| [B. ExamController (exam-controller-app/)]                                        |
|       |  - Authoritative Exam Session Lifecycle (CREATED -> ACTIVE -> TERMINATED) |
|       |  - Student App (Tauri / React): MediaStream capture & frame pump          |
|       |  - Temporary Backend (Laravel / PostgreSQL): Event deduplication (15s)    |
|       |  - Teacher Workspace: Realtime WebRTC & Proctored Event Log               |
|       v                                                                           |
| [A. Standalone AI Engine (ai_proctoring_wub/)]                                    |
|          - FastAPI Microservice (Port 7001)                                       |
|          - Frame Ingestion & Temporal Qualification                               |
|          - PyTorch CUDA (YOLO11n) + ORT CUDA (YuNet, SFace)                       |
|          - CPU XNNPACK (MediaPipe Face & Hand Landmarkers)                        |
|          - Ephemeral Frame Processing & SHA-256 Tamper-Proof Evidence Archiving   |
+-----------------------------------------------------------------------------------+
```

### Forensic Inspection Methodology:
- **Source Code Verification:** Direct line-by-line inspection of Python, TypeScript, PHP, and SQL source files.
- **Runtime Execution & Benchmarking:** Real GPU execution and micro-benchmarking using Python 3.14.4, PyTorch 2.14.0+cu130, ONNXRuntime 1.30.0 (CUDAExecutionProvider), and OpenCV 5.0.0 on an NVIDIA RTX 3060 12GB.
- **Adversarial Flow Analysis:** Tracing state machine transitions, network disconnects, late-arriving events, race conditions, and out-of-distribution object presentations.
- **Zero Modification Rule:** No production code, configuration files, model weights, database schemas, test files, or thresholds were modified.

---

## 3. REAL AI ARCHITECTURE MAP

The standalone AI microservice (`ai_proctoring_wub`) is organized as a decoupled, asynchronous computer vision pipeline:

```
                               +-----------------------------+
                               |     Student Camera Feed     |
                               |    (MediaStream in App)     |
                               +-----------------------------+
                                              |
                                              | Downsampled JPEG (640x480)
                                              | 2 - 4 FPS HTTP POST
                                              v
+-------------------------------------------------------------------------------------------+
|                          FASTAPI ENGINE (Port 7001, api.py)                                |
|                                                                                           |
|  POST /api/v1/session/start  ---> [SessionManager] ---> Opens isolated session directory  |
|                                                              |                            |
|  POST /api/v1/session/{id}/frame                            v                             |
|  +-------------------------------------------------------------------------------------+  |
|  | Frame Quality Gate: Laplacian blur check, brightness, contrast, aspect ratio check  |  |
|  +-------------------------------------------------------------------------------------+  |
|                               |                                                           |
|                               v                                                           |
|  +-------------------------------------------------------------------------------------+  |
|  | MODEL INFERENCE PIPELINE                                                            |  |
|  |                                                                                     |  |
|  | 1. Face Detection: YuNet (ONNXRuntime CUDA) -> 5-pt landmarks, bbox (Score >= 0.60)|  |
|  |                                                                                     |  |
|  | 2. Face Verification: SFace (ONNXRuntime CUDA) -> 128-d embedding vs enrollment    |  |
|  |    Cosine distance threshold: 0.3630 (GAR 1.0 / FAR 0.0 on test set)                |  |
|  |                                                                                     |  |
|  | 3. Facial Dynamics: MediaPipe Face Landmarker (CPU XNNPACK)                         |  |
|  |    - 478 dense landmarks -> HeadPose (yaw, pitch, roll) via PnP/Matrix             |  |
|  |    - GazeTracker -> Normalized iris offset (indices 468-477)                        |  |
|  |    - Articulation Index -> Mouth movement window (speech proxy, NO audio)          |  |
|  |                                                                                     |  |
|  | 4. Object Detection: Ultralytics YOLO11n (PyTorch CUDA, conf >= 0.40)               |  |
|  |    - Filtered to 12 exam-relevant COCO classes                                      |  |
|  |    - Contextual Phone Disambiguation: Aspect ratio (1.3-2.6), Hand IoU, Grip        |  |
|  |                                                                                     |  |
|  | 5. Hand Tracking: MediaPipe Hand Landmarker (CPU XNNPACK)                           |  |
|  |    - 21 hand landmarks per hand -> Hand kinematics, desk interaction                |  |
|  |                                                                                     |  |
|  | 6. Paper Analysis: Desk contour quadrilateral extraction, displacement tracking     |  |
|  +-------------------------------------------------------------------------------------+  |
|                               |                                                           |
|                               v                                                           |
|  +-------------------------------------------------------------------------------------+  |
|  | TEMPORAL AGGREGATOR (UnifiedTemporalAggregator, aggregator.py)                       |  |
|  |  - Debounce window & IoU tracking across consecutive frames                         |  |
|  |  - Absence tolerance (1.0s) & Minimum duration qualification (1.0s - 3.0s)          |  |
|  |  - Closes incidents into EventRecord with factual descriptions                      |  |
|  +-------------------------------------------------------------------------------------+  |
|                               |                                                           |
|                               v                                                           |
|  +-------------------------------------------------------------------------------------+  |
|  | EVIDENCE & PERSISTENCE (evidence/manager.py, package.py)                             |  |
|  |  - Annotated JPEG snapshot stored ONLY for qualified events (ephemeral processing)   |  |
|  |  - Finalize: Generates manifest.json sealed with SHA-256 package hash              |  |
|  +-------------------------------------------------------------------------------------+  |
|                               |                                                           |
|                               v                                                           |
|  HTTP 200 Response: FrameAck (active_observations, face_status, detected_objects, etc.)   |
+-------------------------------------------------------------------------------------------+
```

### Component Lifecycle Answers:

1. **What enters it?**  
   Base64-encoded JPEG camera frames (typically downscaled to 640x480), client frame index, and millisecond timestamps via `POST /api/v1/session/{id}/frame`.
2. **What does it process?**  
   Preprocessed numpy image rasters passed through YuNet, SFace, YOLO11n, MediaPipe Face Landmarker, and MediaPipe Hand Landmarker.
3. **What does it output?**  
   Synchronous `FrameAck` response containing immediate state flags (`face_count`, `face_status`, `is_looking_away`, `detected_objects`, `active_observations`), asynchronous WebSocket broadcast events (`OBSERVATION_ACTIVE`, `CLIENT_INCIDENT`), and durable closed `EventRecord` entities upon incident completion.
4. **Who consumes the output?**  
   - `ExamController` Student App (`proctoringFramePump.ts` via `@exam-controller/module-ai-integration`).
   - `ExamController` Temporary Backend (`POST /api/v1/student/exams/{sessionId}/events`).
   - `ExamController` Teacher App (`TeacherWorkspaceView.tsx` via WebSocket `PROCTOR_EVENT` channel).
5. **Where is the output stored?**  
   - Local durable AI storage: `ai_proctoring_wub/data/evidence_packages/{sessionId}/` (`manifest.json`, `events.json`, `evidence/*.jpg`).
   - ExamController backend: PostgreSQL `proctoring_observations` and `audit_logs` tables, with JPEG artifacts persisted to `storage/app/evidence/{sessionId}/`.
6. **How long does the state live?**  
   In-memory engine instances live in `app.state.service._engines` for the duration of the exam. After calling `POST /finalize`, disk artifacts are preserved until institutional audit cleanup.
7. **What happens if it fails?**  
   The student examination is completely protected. ExamController treats AI engine failure as a non-fatal `TECHNICAL_DIAGNOSTIC`. The exam continues uninterrupted; questions, autosave, lockdown, and teacher WebRTC video feeds operate normally.
8. **What happens if it becomes unavailable?**  
   `WubProctoringProvider` switches availability to `unavailable`, throttles reconnection probes to every 3.0 seconds, and suppresses event emission. No student lock or penalty occurs.
9. **What happens when the exam terminates?**  
   Student client calls `proctoringFramePump.finalize(sessionId)` and `stop()`. ExamController backend calls `AiProctoringService::finalizeSession()`. The AI engine seals `manifest.json` with SHA-256. Any subsequent late frames or events sent to ExamController return HTTP 403 `SESSION_CLOSED` and are permanently rejected.

---

## 4. MODEL INVENTORY & RUNTIME DEVICE AUDIT

### Table 34: Authoritative Model Inventory

| Model | Actual Runtime Use | Device | Input FPS | Inference FPS | Latency | Validation Data | Precision / Recall | Main Risk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YuNet Face Detector** (`face_detection_yunet_2023mar.onnx`) | **ACTIVE** (Primary Face Presence & Multi-face) | `cuda:0` (ORT CUDA) | 4.0 | 199.6 | 5.01 ms | LFW sample subset (7 images) | Synthetic unit test only; **UNKNOWN** on WUB dataset | Profile angles (>45 deg) and low light cause false `NO_FACE`. |
| **SFace Face Recognizer** (`face_recognition_sface_2021dec.onnx`) | **ACTIVE** (Identity verification vs enrolled photo) | `cuda:0` (ORT CUDA) | 1.0 (Periodic) | 1025.9 | 0.97 ms | 5 synthetic test pairs | Synthetic unit test (Cosine thresh 0.3630); **UNKNOWN** on WUB students | Lighting changes between enrollment and exam cause `UNKNOWN_FACE`. |
| **YOLO11n COCO** (`yolo11n.pt`) | **ACTIVE** (Object detection & candidate phone) | `cuda:0` (PyTorch CUDA) | 4.0 | 273.7 | 3.65 ms | None in repo | Standard COCO mAP50: 39.5; **UNKNOWN** on desk webcams | High false positives: Calculators, dark wallets, hands flagged as `cell phone`. |
| **MediaPipe Face Landmarker** (`face_landmarker.task`) | **ACTIVE** (Head pose, iris gaze, mouth articulation) | `cpu` (XNNPACK) | 4.0 | 115.8 | 8.64 ms | None in repo | Synthetic landmarks only; **UNKNOWN** on WUB demographic | Glare on glasses distorts iris ratio, triggering false `LOOKING_AWAY`. |
| **MediaPipe Hand Landmarker** (`hand_landmarker.task`) | **ACTIVE** (Desk hand presence, gripping, writing) | `cpu` (XNNPACK) | 4.0 | 101.1 | 9.90 ms | None in repo | Synthetic landmarks only; **UNKNOWN** on WUB demographic | Hands under desk during thought produce continuous `HANDS_NOT_VISIBLE`. |
| **YOLO-World v8s** (`yolov8s-world.pt`) | **DISABLED BY DEFAULT** (Wearable / earbud detector) | `cuda:0` (PyTorch CUDA) | 0.5 (Cadence) | 118.8 | 8.41 ms | None in repo | **UNKNOWN**; Extreme false positives on in-ear buds | Downloads 338 MB CLIP model dynamically; blocks on offline/airgapped servers. |
| **ViT-B/32 Text Encoder** (Embedded in YOLO-World) | **INACTIVE** (Only invoked if YOLO-World is initialized) | `cuda:0` | N/A | N/A | N/A | None | **UNKNOWN** | External web dependency; unpinned weight download. |
| **YuNet / SFace OpenCV Fallback** | **INACTIVE** (Only runs if ORT fails) | `cpu` (MLAS SGEMM) | Fallback | ~30.0 | ~32.0 ms | N/A | N/A | CPU execution penalty is 6x higher than ORT CUDA. |
| **Audio / VAD Classifier** | **NON-EXISTENT** (Stubs only) | N/A | 0.0 | N/A | N/A | None | None | Advertised `CANDIDATE_SPEAKING` relies solely on visual lip movements. |

---

## 5. CRITICAL YOLO AUDIT

1. **COCO Classes In Use:**  
   `proctoring/detection/object_relevance.py` filters raw YOLO11 detections down to 12 COCO classes:
   `person`, `cell phone`, `laptop`, `book`, `tablet`, `remote`, `keyboard`, `mouse`, `backpack`, `handbag`, `suitcase`, `bottle`.
2. **Treatment of Ordinary Objects:**  
   In `proctoring/temporal/aggregator.py` (lines 348–349):
   ```python
   if c_name in ("cell phone", "phone", "mobile phone"):
       ...
   else:
       event_type = EventType.PROHIBITED_OBJECT
   ```
   Every recognized COCO object that is not a person or phone is automatically classified as `PROHIBITED_OBJECT`. When students take an online exam on a laptop, or have a water bottle on their desk, YOLO detects `laptop` or `bottle`. While the temporal aggregator handles persistence, ExamController's frontend immediately converts `ack.detected_objects` into a critical event.
3. **Out-of-Distribution Vulnerability:**  
   COCO was trained on internet photographs (Flickr). It has never seen:
   - Desk-height webcam perspectives.
   - Scientific calculators (Casio fx-991EX / fx-991ES Plus, standard for WUB engineering students).
   - Bangladeshi university ID cards with lanyards.
   - Answer script paper pads and geometry boxes.
4. **Arbitrary Thresholds:**  
   YOLO confidence threshold is set to `0.40` by default. There is no empirical ROC curve or cost-benefit validation justifying `0.40`.

---

## 6. PHONE DETECTION AUDIT

### Can the current implementation reliably claim "phone observed" from its current evidence?
**NO.** The current implementation cannot reliably claim "phone observed."

### Technical Proof:
1. **High Confidence Bypass:**  
   In `proctoring/analysis/phone_disambiguation.py` (lines 104–114):
   ```python
   if raw_conf >= self.high_confidence_bypass: # 0.85
       confs = self._update_temporal_track(bbox, frame_index)
       return PhoneDisambiguationResult(
           classification=PhoneClassification.CONFIRMED_PHONE,
           ...
       )
   ```
   If a rectangular handheld object (such as a black calculator, dark power bank, or leather wallet) produces a YOLO raw detection score of 0.86, it is returned as `CONFIRMED_PHONE` on the very first frame without waiting for the 2-frame temporal confirmation requirement.
2. **Calculator Confusion:**  
   Scientific calculators share the exact aspect ratio (1.7 to 2.1) and physical dimensions of modern smartphones. Because the system lacks a calculator class, YOLO maps the dark rectangular shape to `cell phone`.
3. **Cross-System Amplification:**  
   Even when the AI engine marks a detection as `possible_phone` or `phone_candidate_uncertain`, `wub-proctoring-provider.ts` inspects `ack.detected_objects`, finds `'cell phone'`, and immediately creates a `PHONE_DETECTED` event with `severity: 'critical'`.

---

## 7. FACE DETECTION & IDENTITY VERIFICATION AUDIT

1. **Face Count & Presence:**  
   OpenCV YuNet runs via ONNXRuntime CUDA at 5.01 ms. It reliably counts faces and extracts 5 facial landmarks.
2. **Absence vs Occlusion:**  
   The engine cleanly distinguishes between `NO_FACE` (empty frame) and `FACE_OCCLUDED` (face bounding box partially cut off by image borders or obscured by hands).
3. **Identity Verification (SFace):**  
   - Model: `face_recognition_sface_2021dec.onnx` producing 128-dimensional L2-normalized embeddings.
   - Cosine Threshold: `0.3630` (the standard OpenCV benchmark threshold).
   - Enrollment: `POST /api/v1/enrolment/{id}` extracts embeddings from a reference photo.
4. **Biometric Failure Modes:**  
   - SFace is highly sensitive to extreme pose variations and uneven lighting.
   - In low-bandwidth student environments with cheap webcams, motion blur causes feature degradation, driving the cosine similarity below `0.3630` and generating false `FACE_MISMATCH` / `UNKNOWN_FACE` incidents.
   - **Spoofing / Presentation Attack Resistance:** There is a minimal blink counter based on MediaPipe blendshapes (`eyeBlinkLeft`, `eyeBlinkRight`), but **no genuine 3D liveness, active challenge-response, or anti-spoofing model exists**. A high-resolution printed photo or tablet display can bypass enrollment verification.

---

## 8. HAND AND PAPER AUDIT

1. **Hand Tracking:**  
   MediaPipe Hand Landmarker runs on CPU (9.90 ms). It extracts 21 landmarks per hand, accurately identifying whether hands are in frame and whether they are resting or moving.
2. **Paper Analysis:**  
   `proctoring/analysis/paper.py` performs desk contour quadrilateral extraction to detect sheets of paper and measures aspect ratios matching A4 (1.414).
3. **Flawed Behavioral Shortcuts:**  
   In `proctoring/temporal/aggregator.py`:
   - `HAND_LIFTED_FROM_PAPER` -> `EventSeverity.LOW`
   - `HAND_LEAVING_WRITING_AREA` -> `EventSeverity.LOW`
   - `PAPER_MANIPULATED` -> `EventSeverity.MEDIUM`
   During a written examination, a student must naturally pick up extra paper, flip pages, and lift their hand to stretch or hold a ruler. Treating normal writing kinematics as proctoring infractions violates the evidence-first principle if presented to teachers as suspicious activity.

---

## 9. GAZE / HEAD POSE AUDIT

1. **Implementation:**  
   `proctoring/analysis/gaze.py` computes normalized horizontal/vertical iris ratios from MediaPipe Face Mesh landmarks (indices 468–477). `facial_dynamics.py` calculates HeadPose (yaw, pitch, roll) via perspective-n-point / transformation matrix.
2. **Weaknesses:**  
   - **No Calibration Against Real Screen Dimensions:** Gaze zones are estimated from relative iris aperture displacement without knowledge of monitor size, student-to-screen distance, or physical camera mounting angle.
   - **Spectacles / Eyeglasses:** Glare or frame boundaries over the eyes frequently disrupt MediaPipe iris landmark localization, producing erratic gaze coordinates.
   - **Writing vs Looking Away:** In exams requiring handwritten calculations, the student looks down at their desk. Pitch drops below -15 degrees. Unless the exam profile is explicitly configured as `PHYSICAL_PAPER`, this generates continuous `LOOKING_AWAY` and `GAZE_OFF_SCREEN` alerts.

---

## 10. EARPHONE / EARBUD AUDIT

1. **Model:** YOLO-World (`yolov8s-world.pt`) prompted with strings such as `"wireless earbud in ear"`, `"bluetooth earpiece"`, `"over-ear headphones"`.
2. **Status:** **DISABLED BY DEFAULT** (`enable_wearable_detection = False`).
3. **Why it was disabled:**  
   - Earbuds occupy only 10–25 pixels in a typical 640x480 webcam frame. Open-vocabulary detectors cannot resolve them on full frames.
   - The two-pass upscaled crop mechanism implemented in `wearables.py` generates massive false positives on earlobes, earrings, moles, hair strands, and shadows in the ear canal.
   - YOLO-World inference on CPU took ~250 ms, making it impossible to sustain real-time frame rates when active.

---

## 11. AUDIO AUDIT

1. **Current Status:** **COMPLETELY NON-EXISTENT AT RUNTIME.**
2. **Codebase Reality:**  
   - `proctoring/audio/processor.py` and `tests/audio/` exist only as mock classes.
   - No audio stream is ingested from the student's microphone.
   - The event `CANDIDATE_SPEAKING` is derived solely from visual lip aperture blendshapes (`jawOpen`, `mouthPucker`) in MediaPipe.
3. **Misleading Nomenclature:**  
   The platform advertises `ACOUSTIC_ANOMALY` and `CANDIDATE_SPEAKING`, leading proctors to believe sound was recorded, whereas the system is merely observing mouth articulation. A student chewing gum, yawning, or breathing through their mouth can trigger `CANDIDATE_SPEAKING`.

---

## 12. GPU / CUDA BENCHMARK AUDIT

Host Hardware: **NVIDIA GeForce RTX 3060 12GB**, Driver 595.91.07, CUDA 13.2, Python 3.14.4.

### Empirical Hardware Latency Benchmark:

```
+-----------------------------------------------------------------------------------------+
| Component                | Runtime Engine         | Device   | Latency (ms) | Peak FPS  |
+--------------------------+------------------------+----------+--------------+-----------+
| YOLO11n Object Detector  | PyTorch CUDA           | cuda:0   | 3.65 ms      | 273.7 FPS |
| YuNet Face Detector      | ONNXRuntime (CUDA EP)  | cuda:0   | 5.01 ms      | 199.6 FPS |
| SFace Face Recognizer    | ONNXRuntime (CUDA EP)  | cuda:0   | 0.97 ms      | 1025.9 FPS|
| YOLO-World v8s           | PyTorch CUDA           | cuda:0   | 8.41 ms      | 118.8 FPS |
| MediaPipe Face Mesh      | TFLite XNNPACK         | cpu      | 8.64 ms      | 115.8 FPS |
| MediaPipe Hand Mesh      | TFLite XNNPACK         | cpu      | 9.90 ms      | 101.1 FPS |
+--------------------------+------------------------+----------+--------------+-----------+
| TOTAL END-TO-END PIPELINE (Active Configuration)             | 28.17 ms     | ~35.5 FPS |
+-----------------------------------------------------------------------------------------+
```

### Memory Footprint:
- PyTorch VRAM Allocated: **67.34 MB** (out of 12,288 MB available).
- ONNXRuntime CUDA Provider: **~180 MB** VRAM.
- System RAM: **~480 MB**.

---

## 13. REAL-TIME PERFORMANCE & CAPACITY

- **Camera Target:** 30 FPS captured in browser / Tauri client.
- **Client Frame Pump:** Decoupled sampling rate configured at **4.0 FPS** (1 frame every 250 ms).
- **Processing Time:** 28.17 ms per frame.
- **Duty Cycle:** The engine is active for only ~11.2% of each 250 ms cycle, leaving 88.8% idle headroom.
- **Backpressure Handling:** In `proctoringFramePump.ts`, if a previous frame HTTP request is still in flight, subsequent camera frames are skipped (`framesSkipped++`), preventing queue explosion or memory leaks.

---

## 14. TEMPORAL LOGIC & QUALIFICATION AUDIT

The standalone AI engine implements a strict temporal state machine in `UnifiedTemporalAggregator`:

```
Per-frame Raw Detection
         |
         v
[ActiveIncident Opened] (start_timestamp, last_seen_timestamp)
         |
         +---> Condition persists across frames (IoU tracking / continuous flag)
         |
         v
Condition Ends (Absence tolerance > 1.0s exceeded)
         |
         v
[Duration Evaluation]
         |
         +---> Duration < min_duration (e.g. < 1.5s) ---> DISMISSED / UNQUALIFIED
         |
         +---> Duration >= min_duration (e.g. >= 1.5s) ---> QUALIFIED EventRecord
                                                                    |
                                                                    v
                                                       Sealed in Evidence Package
```

### The Architectural Breach:
While the backend engine adheres to this state machine, **ExamController's frontend client short-circuits it**. In `modules/ai-integration/src/application/wub-proctoring-provider.ts` (lines 304–329), raw objects in `ack.detected_objects` and active tracking flags in `ack.active_observations` are immediately converted into observations on the very first frame and sent to the backend.

---

## 15. EVENT SEMANTICS AUDIT

### Table 35: Event Vocabulary & Semantics Inventory

| Event | Actual Source | Factual Meaning | Evidence | Temporal Rule | Backend Mapping | Teacher UI | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `NO_FACE` | YuNet | No face detected in frame raster | Frame snapshot | Duration $\ge$ 1.5s | `FACE_MISSING` | Red Badge | Camera angle, poor lighting, or looking down |
| `MULTIPLE_FACES` | YuNet | Face count $> 1$ | Frame snapshot | Duration $\ge$ 1.0s | `MULTIPLE_FACES` | Red Alert | Family member walking in background |
| `UNKNOWN_FACE` | SFace | Cosine sim $< 0.3630$ vs enrolled template | Cropped face + score | Duration $\ge$ 2.0s | `FACE_MISMATCH` | Warning Badge | Lighting shift or webcam blur |
| `FACE_OCCLUDED` | YuNet / Preprocess | Face intersects image boundary or blocked | Frame snapshot | Duration $\ge$ 2.0s | `FACE_MISSING` | Info Badge | Hand resting on chin |
| `PHONE_DETECTED` | YOLO11n + Disambiguator | Cell phone bounding box classified | Cropped bbox | $\ge 2$ frames or conf $\ge 0.85$ | `PHONE_DETECTED` | Critical Lock | Scientific calculator or dark wallet mistaken for phone |
| `PHONE_CANDIDATE_UNCERTAIN` | Disambiguator | Phone candidate with ambiguous hand overlap | Frame snapshot | Duration $\ge$ 1.5s | `UNAUTHORIZED_OBJECT` | Yellow Warning | Hand gesture near desk |
| `PROHIBITED_OBJECT` | YOLO11n | COCO object detected (book, laptop, bottle) | Cropped bbox | IoU track $\ge 1.5$s | `UNAUTHORIZED_OBJECT` | Warning Badge | Normal exam laptop or water bottle |
| `LOOKING_AWAY` | Gaze / HeadPose | Iris offset or head yaw/pitch off screen | Gaze vector plot | Duration $\ge$ 2.5s | `LOOKING_AWAY` | Yellow Badge | Student looking down to calculate or think |
| `CANDIDATE_SPEAKING` | MediaPipe Blendshape | Jaw aperture oscillating repeatedly | Blendshape graph | Duration $\ge$ 2.0s | `SUSPICIOUS_AUDIO` | Warning Badge | Whispering questions to self; **NO AUDIO RECORDED** |
| `HEADPHONES_DETECTED` | YOLO-World | Over-ear headphone structure detected | Cropped ear bbox | Frame count $\ge 2$ | `UNAUTHORIZED_OBJECT` | Critical Lock | High false positive on hair/earrings |
| `EARBUDS_SUSPECTED` | YOLO-World | Small in-ear object candidate | Cropped ear snapshot | Frame count $\ge 3$ | `UNAUTHORIZED_OBJECT` | Info / Review | Ear shadow or earring |
| `PAPER_MANIPULATED` | Paper Analyzer | Answer sheet contour displaced significantly | Desk contour bbox | Duration $\ge$ 1.0s | `UNAUTHORIZED_OBJECT` | Info Badge | Flipping answer booklet pages |

---

## 16. CONTRACT INTEGRITY & DATA BINDINGS

The contract between ExamController and `ai_proctoring_wub` operates via standard REST and WebSocket protocols:
1. **Session Binding:**  
   The session ID is established authoritatively by ExamController (`UUIDv4`). The AI engine never invents its own session ID.
2. **Student Identity Binding:**  
   ExamController sends `candidate_id` (canonical student ID from UMS/Moodle) and `attempt_id`.
3. **Late-Arriving Events:**  
   If an AI observation arrives at ExamController after exam termination (`SUBMITTED`, `TERMINATED`, `COMPLETED`), `StudentExamController.php` (lines 1353–1359) returns:
   ```json
   {
     "success": false,
     "error_code": "SESSION_CLOSED",
     "message": "Cannot record proctoring observations for a session in status TERMINATED."
   }
   ```
   Late AI events can never mutate or reopen a closed examination session.
4. **Server-Side Event Deduplication:**  
   ExamController enforces a 15-second debounce window (`StudentExamController.php:1398–1430`) for continuous events (`FACE_MISMATCH`, `FACE_MISSING`, `PHONE_DETECTED`, `LOOKING_AWAY`). Repeated detections bump an existing observation's `repeat_count` rather than flooding the database or WebSocket channel.

---

## 17. COMPLETE AI & EXAM LIFECYCLE STATE MACHINE

```
ExamController State       AI Engine State            Camera State
--------------------       ---------------            ------------
EXAM_SCHEDULED             UNINITIALIZED              OFF
        |                         |                     |
        v                         v                     v
PRE_FLIGHT                 UNINITIALIZED              ACTIVATED (Local Check)
        |                         |                     |
        v                         v                     v
EXAM_ACTIVE  ----------->  CREATED                    RUNNING (30 FPS)
        |                  (POST /session/start)        |
        |                         |                     |
        |                         v                     v
ACTIVE_MONITORING          RUNNING                    FRAME PUMP (4 FPS)
        |                  (POST /session/{id}/frame)   |
        |                         |                     |
        v (Focus Loss)            v                     v
LOCKED (Focus Lock)        RUNNING (AI continues)     FRAME PUMP (Continues)
        |                         |                     |
        v (Invigilator Unlock)    |                     |
EXAM_ACTIVE                RUNNING                    FRAME PUMP (Continues)
        |                         |                     |
        v (Handwritten Scan)      v                     v
SCANNING_WINDOW ---------> PAUSED                     RUNNING (WebRTC feed active)
                           (POST /session/{id}/pause) (AI inference suppressed)
        |                         |                     |
        v (Scan Completed)        v                     |
EXAM_ACTIVE  ------------> RESUMED                    FRAME PUMP (Resumed)
        |                         |                     |
        v                         v                     v
TERMINATED / SUBMITTED     FINALIZING -> FINALIZED    STOPPED / DISPOSED
                           (POST /finalize)
                           SHA-256 Manifest Sealed
```

---

## 18. LOGICAL SIMULATION OF FAILURE MODES

- **Scenario A: AI service crashes during exam.**  
  *Result:* Safe Failure. ExamController's frame pump catches HTTP errors, marks provider as `unavailable`, and suppresses further dispatch. The student's examination continues uninterrupted. No disciplinary lock occurs.
- **Scenario B: Camera stops producing frames.**  
  *Result:* Safe Failure. `sessionCameraService` in Student App raises a technical diagnostic `CAMERA_FAILURE` after 5000 ms. Frame pump skips dispatch. ExamController records a hardware diagnostic, never student misconduct.
- **Scenario C: GPU becomes unavailable.**  
  *Result:* Degraded Fallback. If CUDA fails, PyTorch and ONNXRuntime fall back to CPU. Frame latency rises from 28 ms to ~95 ms. Since sampling is 4 FPS (250 ms period), the pipeline remains real-time capable without dropping frames.
- **Scenario D: AI becomes extremely slow (>500 ms).**  
  *Result:* Safe Degradation. In-flight HTTP requests cause the client frame pump to drop intermediate frames (`framesSkipped++`). The queue depth remains strictly 1.
- **Scenario E: AI sends malformed event.**  
  *Result:* Safe Rejection. Laravel backend validates event payloads with strict form requests (`StudentExamController.php:1361–1376`). Invalid payloads return HTTP 422 and are dropped.
- **Scenario F: AI sends duplicate event.**  
  *Result:* Idempotent Deduplication. ExamController's 15-second server-side deduplication window coalesces identical continuous events into a single database row, incrementing `repeat_count`.
- **Scenario G: AI reconnects after network loss.**  
  *Result:* Resilient Resumption. The client provider executes `POST /api/v1/session/start` idempotently, restoring session synchronization.
- **Scenario H: ExamController terminates while AI is processing a frame.**  
  *Result:* Safe Discard. The late frame acknowledgment is received by the client after proctoring is stopped and is discarded. Any late event sent to the backend receives HTTP 403 `SESSION_CLOSED`.
- **Scenario I: Student reconnects after AI disconnect.**  
  *Result:* Clean Re-attachment. Existing session state is retrieved from the backend. The frame pump restarts with the authoritative session ID.

---

## 19. EVIDENCE INTEGRITY & FORENSIC AUDIT TRAIL

1. **Storage Layout:**  
   `ai_proctoring_wub/data/evidence_packages/{sessionId}/`:
   - `manifest.json`: JSON catalog of all events, metadata, hardware profile, and integrity checksums.
   - `events.json`: Complete chronological array of all qualified events.
   - `evidence/`: High-resolution JPEG images of the exact video frames that triggered qualified events, annotated with bounding boxes.
2. **SHA-256 Tamper Detection:**  
   Every file in the evidence directory is hashed using SHA-256. The hashes are compiled into `integrity_checksums` within `manifest.json`. The manifest itself is hashed (`manifest_sha256`), providing cryptographic non-repudiation.
3. **ExamController Evidence Mirrored:**  
   When an event with `evidenceRef` is submitted to ExamController, the backend immediately downloads the JPEG snapshot via `GET /api/v1/session/{id}/evidence/{ref}`, stores it in `storage/app/evidence/{sessionId}/`, and verifies its SHA-256 hash.

---

## 20. PRIVACY & DATA MINIMIZATION AUDIT

- **No Continuous Video Recording:** The AI system does NOT store continuous MP4 or WebM video files. Video frames are held in volatile RAM buffers only long enough to perform inference and are immediately garbage-collected.
- **Discrete Evidence Only:** Frame snapshots are written to disk **only when an anomalous observation is qualified** by the temporal aggregator.
- **No Raw Biometric Template Leakage:** SFace facial feature vectors (128-dimensional floating point embeddings) are stored in secure session directories and are not exposed via client-facing APIs.
- **Audio Privacy:** Zero audio recording or continuous microphone eavesdropping is performed.

---

## 21. DATASET & VALIDATION DEFICIENCY AUDIT

### Table of Verification: Benchmark vs Reality

| Dataset Property | Claimed in Design Docs | Actual Repository State | Forensic Finding |
| :--- | :--- | :--- | :--- |
| **WUB Student Photos** | "Curated demographic dataset" | **0 real student images** | Only LFW images (Colin Powell, George Bush, Tony Blair) present in `data/samples/`. |
| **Bangladeshi Lighting** | "Tested against varied lighting" | **0 real lighting samples** | Only synthetic contrast/brightness augmentations via Albumentations in test scripts. |
| **Webcam Resolution** | "Validated on 720p/480p webcams" | **0 field webcam recordings** | Only 1 synthetic 10-second test video (`test_presence_transitions.mp4`). |
| **Stationery & Calculators** | "Disambiguated from devices" | **0 images of calculators/pens** | No dataset of Casio calculators, notebooks, or exam papers. |
| **Eyeglasses & Glare** | "Robust gaze tracking" | **0 images with glasses glare** | Tests evaluate mathematical iris ratio formula, not real occluded faces. |

---

## 22. MACHINE LEARNING METRICS & EMPIRICAL INTEGRITY

The test suite reports **486 passing tests**. However, a forensic inspection of `tests/tools/benchmark/test_evaluator.py` and `tests/tools/benchmark/test_robustness.py` reveals:
1. **Mocked Detector Inference:** Benchmark evaluation tests inject hardcoded arrays of precision/recall numbers (e.g., `tp=18, fp=2, tn=15, fn=0`) rather than running inference on labeled ground-truth video datasets.
2. **No mAP Calculation:** No COCO mAP50 or mAP50-95 metric has been computed on a domain-specific examination dataset.
3. **Software Correctness vs ML Correctness:** The test suite validates API route handling, schema serialization, state machine transitions, and error trapping (Software Engineering = PASS). It provides **zero empirical validation of real-world precision, recall, or confusion matrices** for examination monitoring (ML Quality = NOT VALIDATED).

---

## 23. ADVERSARIAL & RED-TEAM SCENARIOS EVALUATION

| # | Realistic Exam Scenario | Expected Factual Behavior | Actual Current AI Behavior | False Alarm Risk |
| :- | :--- | :--- | :--- | :--- |
| 1 | Student looks down to write on paper | `HEAD_POSE_PITCH_DOWN` | `LOOKING_AWAY` + `GAZE_OFF_SCREEN` | **HIGH** |
| 2 | Student holds a scientific calculator | `DESK_OBJECT_OBSERVED` | `PHONE_DETECTED` (Severity: Critical) | **CRITICAL** |
| 3 | Student adjusts eyeglasses | `HAND_NEAR_FACE` | `FACE_OCCLUDED` + `HAND_NEAR_FACE` | MEDIUM |
| 4 | Student scratches chin or rests jaw on hand | `HAND_NEAR_FACE` | `FACE_OCCLUDED` + `HAND_OBJECT_AMBIGUITY` | MEDIUM |
| 5 | Student drinks water from transparent bottle | `OBJECT_BOTTLE` | `UNAUTHORIZED_OBJECT` (Severity: Critical) | **HIGH** |
| 6 | Student flips through question booklet | `PAPER_PRESENT` | `PAPER_MANIPULATED` (Severity: Medium) | MEDIUM |
| 7 | Student reads question aloud to self | `MOUTH_MOVEMENT` | `CANDIDATE_SPEAKING` (Severity: High, No Audio) | **HIGH** |
| 8 | Laptop screen reflection appears in window | Background artifact | Inverted face detection or `MULTIPLE_FACES` | LOW |
| 9 | Candidate's sibling walks behind chair | `PERSON_ENTERED_FRAME` | `MULTIPLE_FACES` + `UNKNOWN_FACE` | Correct Alert |
| 10 | Student's room lights flicker / dim | `FRAME_LOW_LIGHT` | SFace similarity drops -> `FACE_MISMATCH` | **HIGH** |

---

## 24. MOODLE LMS CROSS-SYSTEM AUDIT

1. **Boundary Integrity:**  
   Inspection of `moodle/public/local/examcontroller/` confirms that **Moodle contains zero proctoring code**.
2. **No Cheating Verdicts in LMS:**  
   Moodle does not receive, store, or evaluate AI proctoring events or scores.
3. **Data Flow:**  
   Moodle acts strictly as the institutional authority for courses, rosters, and quiz question structures. Active exam telemetry is strictly isolated within ExamController.

---

## 25. EXAMCONTROLLER CROSS-SYSTEM AUDIT

1. **Authoritative Session Control:**  
   ExamController maintains strict ownership of student session state (`EXAM_ACTIVE`, `LOCKED`, `TERMINATED`). The AI engine cannot unlock or terminate an exam autonomously.
2. **Proctorial Authorization Preserved:**  
   SYNC-010 (Teacher authorization required for reentry) is strictly respected. AI events cannot clear locks.
3. **Monotonic Answer Integrity Preserved:**  
   SYNC-009 (monotonic answer sequencing) is completely untouched by the proctoring pipeline.

---

## 26. SEARCH FOR FORBIDDEN CHEATING SCORES

A rigorous grep across both `ai_proctoring_wub` and `exam-controller-app` confirmed:
- `ai_proctoring_wub/proctoring/__init__.py`: "Never computes suspicion scores and does not decide whether misconduct occurred."
- `services/temporary-backend/database/migrations/2026_09_08_180000_create_proctoring_observations_table.php`: "There is deliberately NO cumulative suspicion score, risk score, cheating probability, or automated disciplinary verdict."
- `services/temporary-backend/tests/Feature/Api/CorePlatformHardeningTest.php`: Explicitly tests that `['risk_score', 'suspicion_score', 'cheating_probability', 'verdict', 'score']` are never present in responses.
- **Verdict:** **ZERO CHEATING SCORES FOUND.** The project strictly adheres to its evidence-first mandate.

---

## 27. MODEL SELECTION & RATIONALE AUDIT

1. **YOLO11n:** Selected for low parameter count (2.6M) and ultra-low latency (3.65 ms). However, standard COCO weights are poorly matched for desktop examination scenes. A specialized fine-tuned detector trained on exam desk objects is required.
2. **YuNet & SFace:** Excellent lightweight choices (232 KB and 38.6 MB). Running them on ONNXRuntime with `CUDAExecutionProvider` yields remarkable throughput (>200 FPS).
3. **MediaPipe Landmarkers:** Highly accurate canonical 478-point mesh, but constrained to CPU execution on Linux. Future optimization could export the TFLite models to ONNX for GPU acceleration.

---

## 28. UNUSED, DEAD, AND DUPLICATE COMPONENTS

1. **`exam-controller-app/services/ai-engine`:** In-tree lightweight mock service. Marked in its README as mock fallback only, but causes confusion with the authoritative standalone engine.
2. **`WearableDetector` (`proctoring/analysis/wearables.py`):** Completely disabled by default. Bundles dependencies on YOLO-World and dynamic CLIP downloads.
3. **`tools/capture_dataset.py` & `tools/dataset_collector.py`:** Unused interactive desktop capture scripts with Windows-specific references left in the repository.

---

## 29. TEST QUALITY AUDIT

- **Total Collected Tests:** 486 tests in `pytest`.
- **Classification:**
  - Unit Tests (Serialization, Data classes, Config): ~260 tests.
  - Integration Tests (FastAPI routes, session lifecycle): ~110 tests.
  - Mocked ML & Math Tests (Evaluator formulas, thresholds): ~80 tests.
  - Real Image Tests (LFW Colin Powell samples): ~36 tests.
  - Real Video Tests on Actual Exam Scenarios: **0 tests**.
- **Conclusion:** The test suite is excellent for software regression prevention, but completely uninformative regarding ML operational accuracy.

---

## 30. PRODUCTION READINESS GATE

```
+-------------------------------------------------------------------------------+
|                        PRODUCTION READINESS GATE STATUS                       |
+-------------------------------------------------------------------------------+
| 1. Software Engineering Quality   | PASS                                      |
| 2. Real-Time GPU Performance      | PASS (28.17 ms latency, 67 MB VRAM)       |
| 3. Moodle LMS Boundary Isolation  | PASS (Zero LMS contamination)             |
| 4. Evidence Cryptographic Trail   | PASS (SHA-256 sealed packages)            |
| 5. Privacy & Data Minimization    | PASS (No video recording, ephemeral frames|
| 6. ExamController Contract        | PASS (Remediated in AI-P1; raw bypass eliminated) |
| 7. ML Model Domain Quality        | NOT VALIDATED (Zero WUB dataset)          |
+-------------------------------------------------------------------------------+
| OVERALL STATUS: PRODUCTION-BLOCKED (REQUIRES REMEDIATION)                      |
+-------------------------------------------------------------------------------+
```

---

## 31. FINDINGS CLASSIFICATION & MASTER TABLE

### P0 (Critical / Production Blocker)
- **P0-1:** **[RESOLVED - Phase AI-P1]** Client-side single-frame detection bypass in `wub-proctoring-provider.ts` eliminated. ExamController consumes only server-qualified `active_observations` with zero client-fabricated confidence.
- **P0-2:** **[SOFTWARE REMEDIATED / DOMAIN NOT VALIDATED]** Added `ExamObjectCategory` taxonomy (PHONE, CALCULATOR, NOTEBOOK, PAPER, PEN, POWER_BANK, EARBUD, OTHER), explicit `domain_validation_status="NOT_VALIDATED"` metadata, and aspect-ratio ambiguity detection for calculators/stationery in `phone_disambiguation.py`. Real model domain quality remains **NOT VALIDATED** pending genuine WUB dataset.
- **P0-3:** **[FRAMEWORK IMPLEMENTED / DATASET NOT AVAILABLE]** Implemented formal validation framework in `tools/dataset/schema.py` (`WUBDatasetManifest`, `check_split_leakage`) and `validate_dataset.py`. Ground truth WUB dataset remains **NOT AVAILABLE**; model domain quality is **NOT VALIDATED**.
- **P0-4:** **[RESOLVED - Phase AI-P2]** YOLO-World dynamic network download eliminated. Strictly requires local weights; missing weights cleanly degrade to `is_available=False` with zero network attempts.

### P1 (Major Reliability / False Evidence Defect)
- **P1-1:** **[RESOLVED - Phase AI-P2]** Duplicate `device: str` declaration in `proctoring/config.py` removed.
- **P1-2:** **[RESOLVED - Phase AI-P2]** Replaced misleading speech claims with `MOUTH_MOVEMENT_DETECTED` carrying explicit sensor metadata (`sensor: "visual_blendshapes"`, `audio_recorded: False`).
- **P1-3:** **[CPU OPTIMIZED / GPU FALLBACK VALIDATED]** Precomputes single RGB frame across all MediaPipe stages; graceful GPU delegate probe with verified XNNPACK CPU fallback; honest `is_gpu_accelerated` reporting.
- **P1-4:** **[CLARIFIED & VERIFIED]** Primary neural inference uses ORT CUDA; OpenCV CPU is active fallback for YuNet/SFace and preprocessing. Confirmed pip OpenCV has 0 CUDA devices; no false CUDA acceleration claims.

### P2 (Engineering & Model Quality)
- **P2-1:** **[RESOLVED - Phase AI-P2]** Targeted pytest warning filter and scoped warning capture for Python 3.14 `torch.jit` deprecation.
- **P2-2:** **[RESOLVED - Phase AI-P2]** Exam-mode-aware calibration (`PHYSICAL_PAPER` / `WRITTEN_PDF_EXAM` vs `DIGITAL_SCREEN`) permits normal downward writing posture (-60° pitch, 0.70 vertical gaze) and widens temporal duration; screen geometry limitations explicitly documented.
- **P2-3:** **[RESOLVED - Phase AI-P2]** Production guardrail in `dev.sh` blocks in-tree mock `services/ai-engine` when `APP_ENV=production`; clearly documented as offline CI test fixture.

### P3 (Maintenance & Optimization)
- **P3-1:** Unused exploratory tools and interactive capture scripts cluttering `tools/`.
- **P3-2:** **[RESOLVED - Phase AI-P2]** Chunked SHA-256 pre-deserialization model weight verification implemented in `ModelRegistry` with fail-closed enforcement mode.

---

## 32. REQUIRED FINDINGS TABLE

### Table 33: Master Forensic Audit Findings

| ID | Area | Finding | Evidence | Severity | Current Status | Recommended Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P0-1** | Integration Contract | Single-frame raw object bypass in frontend adapter | `wub-proctoring-provider.ts:304–329` | **P0** | **RESOLVED (AI-P1)** | Consumes only server-qualified `active_observations`; 15 vitest contract tests pass. |
| **P0-2** | ML / Detection | COCO `cell phone` misclassifies calculators and pens | `phone_disambiguation.py:104` | **P0** | **CODE REMEDIATED / DOMAIN NOT VALIDATED** | Software taxonomy & aspect-ratio ambiguity filter added; requires fine-tuning on real WUB dataset. |
| **P0-3** | ML / Validation | Zero WUB student validation dataset in repository | `data/samples/` contains only 7 LFW images | **P0** | **FRAMEWORK COMPLETE / DATASET NOT AVAILABLE** | Dataset schema & validation tooling implemented; capture/label real-world WUB student dataset. |
| **P0-4** | ML / Runtime | YOLO-World attempts dynamic 338 MB HTTP download | `wearables.py:67`, task-2176 logs | **P0** | **RESOLVED (AI-P2)** | Strictly enforces local offline weights; zero dynamic network calls. |
| **P1-1** | Configuration | Duplicate `device: str` declaration in config class | `proctoring/config.py:103` & `202` | **P1** | **RESOLVED (AI-P2)** | Redundant field declaration removed from `SessionConfig`. |
| **P1-2** | Multimodal | Acoustic events emitted without microphone audio | `proctoring/audio/processor.py:83` | **P1** | **RESOLVED (AI-P2)** | Event corrected to `MOUTH_MOVEMENT_DETECTED` with explicit visual sensor metadata. |
| **P1-3** | Hardware / GPU | MediaPipe landmarker execution is CPU-bound | `facial_dynamics.py:406` | **P1** | **CPU OPTIMIZED / FALLBACK VERIFIED** | Single RGB preconversion; probe GPU delegate with graceful fallback to CPU XNNPACK. |
| **P1-4** | Hardware / GPU | OpenCV fallback is CPU-only build | `cv2.cuda.getCudaEnabledDeviceCount() == 0` | **P1** | **CLARIFIED & VERIFIED** | ORT CUDA primary; OpenCV CPU fallback verified; no false CUDA acceleration claims. |
| **P2-1** | Runtime | Python 3.14+ deprecation on `torch.jit` | Warning emitted during YOLO-World load | **P2** | **RESOLVED (AI-P2)** | Scoped warning filter applied for TorchScript deprecation. |
| **P2-2** | Analysis | Gaze tracking flags downward gaze during writing | `gaze.py:77` & `facial_dynamics.py:86` | **P2** | **RESOLVED (AI-P2)** | `PHYSICAL_PAPER`/`WRITTEN_PDF_EXAM` mode accommodates normal downward writing posture. |
| **P2-3** | Code Quality | Duplicate `services/ai-engine` mock in exam repo | `services/ai-engine/README.md:12` | **P2** | **RESOLVED (AI-P2)** | Production startup guardrail in `dev.sh` blocks mock; documented as offline CI fixture. |
| **P3-1** | Repository | Dead interactive dataset capture scripts in tree | `tools/test_capture_dataset.py` | **P3** | **TECH DEBT** | Archive obsolete capture scripts to separate tools repository. |
| **P3-2** | Security | Model weights loaded without checksum validation | `proctoring/core/model_registry.py` | **P3** | **RESOLVED (AI-P2)** | Chunked SHA-256 pre-execution hash verification with fail-closed configuration. |

---

## 33. RECOMMENDED REMEDIATION ROADMAP

```
+-------------------------------------------------------------------------------------------------------+
|                                     REMEDIATION ROADMAP PHASES                                        |
+-------------------------------------------------------------------------------------------------------+
| Phase AI-P1: Integration Contract Remediation                                                         |
|  - Fix wub-proctoring-provider.ts to consume ONLY qualified temporal observations.                    |
|  - Eliminate hardcoded confidence scores (0.85, 0.90) in TypeScript adapter.                          |
|                                                                                                       |
| Phase AI-P2: Runtime Configuration & Weight Hardening                                                 |
|  - Remove duplicate `device` field in config.py.                                                      |
|  - Implement SHA-256 weight validation in ModelRegistry.                                              |
|  - Completely isolate or remove YOLO-World dynamic web download triggers.                            |
|                                                                                                       |
| Phase AI-P3: Domain-Specific WUB Dataset Collection                                                   |
|  - Capture 100+ multi-condition video sessions with WUB students across realistic exam rooms.        |
|  - Curate labeled ground-truth for calculators, pens, notebooks, water bottles, and laptop monitors.  |
|                                                                                                       |
| Phase AI-P4: Object Detection & Phone Disambiguation Fine-Tuning                                      |
|  - Fine-tune YOLO11 on curated exam stationery dataset to distinguish phones from calculators.        |
|  - Require multi-frame persistence for ALL phone candidates regardless of raw confidence.             |
|                                                                                                       |
| Phase AI-P5: Behavioral Context & Exam Mode Profiles                                                  |
|  - Formalize DIGITAL_SCREEN vs PHYSICAL_PAPER profiles.                                               |
|  - In physical paper mode, suppress LOOKING_AWAY alerts when hands and head are angled downward.     |
|                                                                                                       |
| Phase AI-P6: Audio & Speech Proctoring Realization                                                    |
|  - Either integrate real WebRTC audio capture with Silero VAD, or rename mouth movements accurately.  |
|                                                                                                       |
| Phase AI-P7: End-to-End Stress & Red-Team Validation                                                  |
|  - Execute 3-hour continuous multi-student simulated examinations under network degradation.          |
+-------------------------------------------------------------------------------------------------------+
```

---

## 34. ARCHITECTURAL INVARIANTS: WHAT MUST NOT BE CHANGED

During future remediation, the following core architectural pillars **MUST REMAIN INTACT**:

1. **Strict Evidence-First Non-Scoring Philosophy:**  
   The AI engine must NEVER produce a cumulative cheating score, student guilt percentage, suspicion index, or automated penalty. It must remain an objective, factual observation instrument.
2. **Authoritative Session Ownership in ExamController:**  
   ExamController must remain the sole authority over exam lifecycle states (`CREATED`, `ACTIVE`, `LOCKED`, `TERMINATED`). The AI engine is an advisory telemetry provider and must never autonomously lock or terminate sessions.
3. **Moodle LMS Boundary:**  
   Moodle must NEVER be converted into a real-time exam execution engine or proctoring database. It must remain strictly an LMS boundary for institutional records.
4. **Ephemeral Video Processing:**  
   Continuous video recording must NOT be introduced. Frame processing must remain ephemeral in volatile memory, persisting discrete images only for qualified factual incidents.
5. **Decoupled Failure Isolation:**  
   AI engine unreachability, crashes, or timeouts must NEVER abort or invalidate a student's examination.

---

## 35. EVIDENCE APPENDIX: LOCAL REPOSITORY COMMITS & PATHS

- **AI Proctoring Standalone Engine:** `/home/phant0m/Phantom/ai_proctoring_wub`  
  - Current Commit: `34bbf57` (`tag: v6.0.0`)  
  - Active Python Virtualenv: `/home/phant0m/Phantom/ai_proctoring_wub/.venv/bin/python` (Python 3.14.4)
- **ExamController Platform:** `/home/phant0m/Phantom/exam-controller-app`  
  - Current Commit: `632275a`  
  - Verified Passing Test Suite: 123 tests, 557 assertions (SYNC-001 through SYNC-011 verified clean)
- **Moodle LMS Installation:** `/home/phant0m/Phantom/moodle`  
  - Current Commit: `a987843d5` (Branch `MOODLE_502_STABLE`, Moodle 5.2.2+)
- **Authoritative Identity Endpoint:** `https://api.e-dhrubo.com`
