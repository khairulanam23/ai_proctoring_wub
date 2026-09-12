# Phase 1 Pipeline Correctness and State Isolation Specification

**Document**: `audit/phase_1_pipeline_correctness.md`  
**Priority**: P0  
**Status**: **IMPLEMENTED & VERIFIED**  
**Auditor**: Senior AI Systems Engineer  
**Target Repository**: `/home/phant0m/Phantom/ai_proctoring_wub`

---

## 1. Detector Interface Contracts

To eliminate silent contract mismatches between detectors and downstream reasoning layers, a formal contract specification was introduced in [`proctoring/core/contracts.py`](file:///home/phant0m/Phantom/ai_proctoring_wub/proctoring/core/contracts.py).

### 1.1 Formal Conventions

* **Coordinate Convention**:
  * Internal Stage Convention: Pixel coordinates `[x1, y1, x2, y2]` where $x_1 \le x_2$ and $y_1 \le y_2$.
  * Normalized Storage Convention: $[0.0, 1.0]$ float relative to source frame width $W$ and height $H$.
  * Coordinate Transformation Utility: Explicit bidirectional functions:
    * `xywh_to_xyxy(bbox)`: $[x, y, w, h] \to [x_1, y_1, x_2, y_2]$
    * `xyxy_to_xywh(bbox)`: $[x_1, y_1, x_2, y_2] \to [x, y, w, h]$
    * `normalize_xyxy(bbox, width, height)`: Pixel $[x_1, y_1, x_2, y_2] \to [0.0, 1.0]$ range.
* **Confidence Semantics**:
  * Must be a float strictly bounded within $[0.0, 1.0]$.
  * Represents detector output score prior to policy thresholding.
  * Thresholding occurs strictly in policy layers (`ExamPolicy`), never hardcoded inside detector classes.
* **Timestamp Alignment**:
  * Every detection payload includes `timestamp_seconds: float` representing elapsed session seconds, plus matching ISO-8601 UTC string.
* **Failure Behavior**:
  * Detectors must never crash the frame pipeline on frame anomalies.
  * Internal failures must be captured in structured technical diagnostic records (`record_analyzer_failure`) and logged, but never silently ignored.

### 1.2 Dataclass Schema

```python
@dataclass(frozen=True)
class StandardDetection:
    label: str
    confidence: float
    bbox_pixel: tuple[float, float, float, float]
    bbox_normalized: tuple[float, float, float, float]
    track_id: int | None = None
    keypoints: tuple[tuple[float, float], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
```

---

## 2. Reconnected Pipeline Detectors

Prior to Phase 1, several detector outputs executed computationally expensive inferences every frame but discarded their results before reaching the observation or event layers. These outputs have now been reconnected.

### 2.1 Reconnection Mapping

| Stage / Detector | Prior State | Phase 1 Corrected Output Path | Emitted Event / Diagnostic |
| :--- | :--- | :--- | :--- |
| **`PaperDetector`** | `detect()` executed Canny edge + contour analysis, but `PaperAnalysisResult` was never inspected by `BehaviourObserver`. | Wired `obs.paper_analysis` into `BehaviourObserver.map_to_events()`. Added paper metrics to `build_annotation_context()`. | `EventType.PAPER_PRESENT`, `EventType.PAPER_MANIPULATED`, `EventType.MULTIPLE_PAPERS_DETECTED`, `EventType.PAPER_ABSENT` |
| **`HandAnalyzer` Kinematics** | `_update_kinematics()` tracked wrist displacement and velocity across frames, but results were ignored. | Wired `obs.hand_analysis.kinematics` states into `BehaviourObserver.map_to_events()`. | `EventType.HAND_WRITING`, `EventType.HAND_RESTING` |
| **`PhoneDetector`** | Raw YOLO box was disambiguated with heuristic aspect ratio, but temporal hysteresis lacked clear duration semantics. | Mapped to `EventType.PHONE_DETECTED` with strict duration floors (`event_min_duration[PHONE_DETECTED] = 0.25s–0.50s`). | `EventType.PHONE_DETECTED` |
| **`WearableDetector`** | Ear ROI zoom was run, but isolated single-frame positives produced false alarms. | Multi-sweep voting window (`wearable_confirmation_sweeps = 2`, `window = 3`) with hand-at-ear corroboration. | `EventType.EARBUDS_SUSPECTED`, `EventType.HEADPHONES_DETECTED`, `EventType.SMARTWATCH_DETECTED` |

---

## 3. Comprehensive Pipeline Exception Handling Audit

A forensic audit of all `try/except` blocks across the proctoring pipeline was performed. Silent exception swallows (`LOGGER.debug` or `pass` without telemetry) were eliminated.

| Source File | Line(s) | Original Behavior | Corrected Behavior | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `proctoring/engine_stages.py` | 338-342 | `except Exception as exc: LOGGER.debug(...)` silently ignored paper detection failures. | Recorded via `self.record_analyzer_failure(exc, "paper_detector", frame_index=obs.frame_index)`. | System errors in paper edge detection must appear in technical diagnostic stream for proctor auditability without penalizing candidate. |
| `proctoring/engine_stages.py` | 350-353 | `except Exception as exc: LOGGER.debug(...)` silently ignored face occlusion classification failures. | Recorded via `self.record_analyzer_failure(exc, "occlusion_classifier", frame_index=obs.frame_index)`. | Hardware or model inference faults in occlusion checks must be transparently tracked as technical metrics. |
| `proctoring/evidence/package.py` | 134-142 | Checksum computation looped over all files in package directory; if `manifest.sha256` existed from a prior run, it was included in `integrity_checksums`, causing self-hash mismatch during validation. | Explicitly excluded `manifest.sha256` and `signature.bin` from internal `integrity_checksums`. | Cryptographic manifest verification failed catastrophically on session re-finalization or shared output directories. |
| `proctoring/analysis/facial_dynamics.py` | 373 | `self.reset()` was called inside `calibrate()`, which erased the newly calculated baseline angles and set `is_calibrated = False`. | Updated `reset(clear_calibration: bool = True)` and passed `clear_calibration=False` in `calibrate()`. | Calibration baseline was immediately wiped out upon learning; this fix preserves calibrated thresholds while clearing pre-session frame history. |
| `proctoring/engine_stages.py` | 385-392 | `EngineStages.reset()` only reset `facial_dynamics` and `quality_gate`, leaving `hand_analyzer`, `paper_detector`, and `wearable_detector` dirty across sessions. | Added explicit `reset()` calls to `hand_analyzer`, `paper_detector`, and `wearable_detector`. | Avoids cross-session state leakage between examinees. |

---

## 4. Session State Isolation & Lifecycle Architecture

### 4.1 Lifecycle Protocol

The `ProctoringEngine` and `EngineStages` now expose four deterministic lifecycle primitives:

```python
# 1. Session Initialization
engine.create_session(
    session_id="session_candidate_101",
    student_id="cand_101",
    policy=ExamPolicy.for_level(StrictnessLevel.STRICT),
    reference_embedding=ref_emb,
)

# 2. Per-Frame Processing
frame_result = engine.process_frame(frame, timestamp_seconds=elapsed_sec)

# 3. Session Reset (Clears all state machines without reloading models)
engine.reset_session()

# 4. Session Teardown (Finalizes packages and clears references)
engine.destroy_session()
```

### 4.2 State Isolation Boundary

* **Process-Wide Heavy Singletons (Loaded ONCE at startup, read-only)**:
  * YOLO11n object detector weights (`models/yolo11n.pt`).
  * YuNet face detector model (`models/face_detection_yunet_2023mar.onnx`).
  * SFace face recognizer model (`models/face_recognition_sface_2021dec.onnx`).
  * MediaPipe Face Landmarker model (`models/face_landmarker.task`).
* **Session-Scoped State Machines (Reset between sessions)**:
  * `FacialDynamicsAnalyzer`: Mouth history buffer (`deque`), blink counter, eye closure flag, calibration baseline (`baseline_yaw`, `baseline_pitch`, `baseline_gaze`, `is_calibrated`).
  * `GazeTracker`: Eye calibration offsets and temporal smoothing filter.
  * `HandAnalyzer`: Hand kinematic velocity tracking, wrist displacement vectors, gesture state machine.
  * `PaperDetector`: Desk ROI contour history, reference corners, temporal displacement accumulator.
  * `WearableDetector`: Ear crop cache and multi-sweep confirmation vote counters.
  * `TemporalAggregator`: Active candidate incidents, debounce windows, qualified event state.
  * `EvidenceManager`: Package manifest, cryptographic hashes, frame snapshots.

### 4.3 Isolation Verification

The test suite in [`tests/core/test_session_isolation.py`](file:///home/phant0m/Phantom/ai_proctoring_wub/tests/core/test_session_isolation.py) was implemented to guarantee:
1. **Explicit Lifecycle Transition**: Verified correct state transitions from `ACTIVE` to `RESET` to `DESTROYED`.
2. **Calibration & Kinematic Isolation**: Verified that Session 1's non-zero yaw baseline and hand kinematics do not carry over into Session 2 after `reset_session()`.
3. **Interleaved Concurrent Sessions**: Verified that processing frames alternately between Session A and Session B with independent engines produces identical results to processing Session A sequentially followed by Session B.
