# Phase 4 — AI Proctoring Evidence Pipeline & Real-World Validation

## 1. System Architecture

```text
                     EXAMINATION STREAM (Webcam / Video)
                                     │
                                     ▼
                    FRAME DECODING & PREPROCESSING
                  (Quality Gating, Blur, Illumination)
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
       FACE AI SUBSYSTEM                       OBJECT AI SUBSYSTEM
     • YuNet Face Detector                   • YOLO11 Object Detector
     • SFace Identity Embedder               • Relevance Filter (Phone, Book, etc.)
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                      TEMPORAL EVENT AGGREGATION
                  (Continuous Incident Consolidation)
                                     │
                                     ▼
                      EVIDENCE CAPTURE & VALIDATION
                   • Full Key Frames (evidence/frames/)
                   • Padded Cropped ROIs (evidence/crops/)
                   • Cryptographic SHA-256 Checksums
                                     │
                                     ▼
                      PERFORMANCE & TELEMETRY TRACKER
                  (Per-Stage Latencies: P50, P95, FPS, RSS)
                                     │
                                     ▼
                         SESSION EVIDENCE PACKAGE
                  ├── manifest.json (Signed Hash)
                  ├── events.json (Factual Observations)
                  ├── telemetry.json (Detailed Profiling)
                  ├── diagnostics.json (Fault Logs)
                  └── evidence/ (Frames & Crops)
```

---

## 2. Event Schema Specification

Every suspicious observation is represented as an independent, factual `EventRecord`:

```json
{
  "event_id": "evt_session_0001_phone_detected",
  "session_id": "exam_session_101",
  "timestamp": 14.25,
  "end_timestamp": 17.50,
  "duration": 3.25,
  "formatted_start": "00:00:14.250",
  "formatted_end": "00:00:17.500",
  "event_type": "PHONE_DETECTED",
  "severity": "HIGH",
  "confidence": 0.8850,
  "average_confidence": 0.8620,
  "detector": {
    "name": "Ultralytics YOLO11",
    "version": "11.0",
    "model_file": "yolo11n.pt",
    "device": "cpu",
    "confidence_threshold": 0.25
  },
  "observation": {
    "description": "Object 'cell phone' detected for 3.25s with max confidence 0.89 across 13 frame(s)",
    "object_class": "cell phone",
    "bounding_boxes": [[320, 200, 390, 320]],
    "frame_indices": [57, 58, 59, 60],
    "timestamps": [14.25, 14.50, 14.75, 15.00],
    "raw_confidences": [0.85, 0.89, 0.87, 0.84]
  },
  "evidence": [
    {
      "evidence_id": "ev_frm_000060",
      "media_type": "frame",
      "file_path": "evidence/frames/frame_000060_00015000ms.jpg",
      "timestamp_seconds": 15.0,
      "formatted_timestamp": "00:00:15.000",
      "frame_index": 60,
      "sha256": "3a7b...12c",
      "is_validated": true
    },
    {
      "evidence_id": "ev_crp_0001_cell_phone",
      "media_type": "crop",
      "file_path": "evidence/crops/crop_evt_0001_cell_phone_000060.jpg",
      "timestamp_seconds": 15.0,
      "formatted_timestamp": "00:00:15.000",
      "frame_index": 60,
      "bbox": [320, 200, 390, 320],
      "sha256": "4e9f...88a",
      "is_validated": true
    }
  ],
  "metadata": {
    "is_duration_qualified": true,
    "best_frame_index": 60,
    "best_timestamp": 15.0
  },
  "status": "VALIDATED"
}
```

---

## 3. Strict Non-Violation Principles

1. **Zero Cumulative Risk Scoring**: The system never aggregates violation points (`score += 10`) and never attempts to compute an automated "cheating index".
2. **Proctor Decision Support**: The AI generates strictly descriptive, factual evidence for human review by authorized invigilators.
3. **Traceability**: Every event links to detector parameters, raw confidences, timestamps, and verifiable on-disk images.
