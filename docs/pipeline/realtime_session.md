# Phase 8 — Real-Time Proctoring Session & Evidence Pipeline Report

## Executive Summary

Phase 8 builds the **real-time examination proctoring session pipeline** connecting live webcam frame ingestion, AI inference, temporal event state aggregation, deterministic evidence capture (`frames/`, `crops/`), chronological **session timeline tracking**, and final cryptographically signed evidence packaging (`manifest.json`). Implemented in `proctoring/capture/` and `proctoring/engine.py`, this pipeline enforces strict zero cumulative risk scoring, preserves data minimization by discarding normal compliant frames from disk, and isolates camera disconnects or technical faults into `diagnostics.json`.

---

## 1. Unified Real-Time Architecture

The real-time proctoring pipeline coordinates the following execution flow:

```text
Webcam Stream (WebRTC / OpenCV / Video)
                 │
                 ▼
       Frame Ingestion & Clock (4 FPS)
                 │
                 ▼
   Adaptive Preprocessing (CLAHE)
                 │
                 ▼
   ┌──────────────────────────────────────────────┐
   │             AI Inference Stage               │
   ├──────────────────────┬───────────────────────┤
   │ Face Detection       │ YuNet ONNX (0.60)     │
   │ Identity Match       │ SFace ONNX (T=0.3630) │
   │ Prohibited Objects   │ YOLO11 (Phone/Book)   │
   └──────────────────────┴───────────────────────┘
                 │
                 ▼
    Unified Temporal Aggregator (0.5s Tolerance)
                 │
                 ▼
  ┌────────────────────────────────────────────────┐
  │              Observation Status                │
  ├──────────────────────┬─────────────────────────┤
  │ COMPLIANT            │ Discard frame in memory │
  │                      │ (No disk I/O)           │
  ├──────────────────────┼─────────────────────────┤
  │ SUSPICIOUS INCIDENT  │ Save Keyframe & Crops   │
  │                      │ Record Event Timeline   │
  └──────────────────────┴─────────────────────────┘
                 │
                 ▼
 Final Evidence Package Builder (SHA-256 Manifest)
```

---

## 2. Chronological Session Timeline Schema

Every processed frame generates a lightweight timeline observation recorded in `timeline.json` ([proctoring/engine.py](file:///home/khairul-anam/Documents/quizaccess_proctoring_2026050700/ai_proctoring/proctoring/engine.py)):

```json
[
  {
    "frame_index": 1,
    "timestamp_seconds": 0.0,
    "iso_timestamp": "2026-08-25T10:33:00.123456+00:00",
    "face_count": 1,
    "is_enrolled_face_present": true,
    "cosine_similarity": 0.7245,
    "detected_prohibited_objects": [],
    "active_events": [],
    "processing_latency_ms": 11.20,
    "is_suspicious_state": false
  },
  {
    "frame_index": 17,
    "timestamp_seconds": 4.0,
    "iso_timestamp": "2026-08-25T10:33:04.123456+00:00",
    "face_count": 0,
    "is_enrolled_face_present": false,
    "cosine_similarity": null,
    "detected_prohibited_objects": [],
    "active_events": [
      "NO_FACE"
    ],
    "processing_latency_ms": 9.80,
    "is_suspicious_state": true
  }
]
```

---

## 3. Real-Time Session Report & Evidence Validation

```json
{
  "session_id": "exam_sess_20260825_103350",
  "student_name": "Candidate_Alice",
  "start_time_iso": "2026-08-25T10:33:50.000000+00:00",
  "end_time_iso": "2026-08-25T10:33:58.200000+00:00",
  "total_session_duration_seconds": 8.20,
  "total_frames_sampled": 32,
  "total_frames_processed": 32,
  "mean_frame_latency_ms": 11.15,
  "effective_throughput_fps": 89.20,
  "total_events_detected": 1,
  "qualified_events_count": 1,
  "evidence_files_stored": 2,
  "package_directory": "data/results/phase8_realtime_sessions/exam_sess_20260825_103350",
  "zip_archive_path": "data/results/phase8_realtime_sessions/exam_sess_20260825_103350.zip",
  "integrity_verified": true
}
```

---

## 4. Key Guarantees & Implementation Invariants

1. **No Risk Scoring**: The system never aggregates numbers or outputs cheating accusations.
2. **Data Minimization**: Normal examination frames are discarded in memory; only qualified keyframes and cropped ROIs for flagged incidents are preserved on disk.
3. **Cryptographic Tamper-Proofing**: `manifest.json` records SHA-256 hashes for all evidence images.
4. **Fault Recovery**: Camera disconnects or corrupt frames produce diagnostic telemetry rather than false suspicious events.

---

### **PHASE 8 STATUS: COMPLETE**
