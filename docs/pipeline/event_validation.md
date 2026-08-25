# Phase 9 — Advanced Proctoring Event Detection & Evidence Validation Report

## Executive Summary

Phase 9 implements an explicit **Event Validation & Evidence Verification Layer** between raw AI inferences and the permanent session timeline. Implemented in `proctoring/temporal/` and `proctoring/evidence/`, this system introduces a formal **Event Lifecycle State Machine** (`OBSERVED` → `CANDIDATE` → `VALIDATED` → `EVIDENCE_CAPTURED` → `CLOSED` or `DISCARDED`), category-specific validation strategies (consecutive frame accumulation, minimum duration persistence, and immediate capture), pre-flight visual evidence quality validation with SHA-256 cryptographic verification, and intelligent consolidation of repeated separate incidents.

---

## 1. Advanced Validation Architecture

```text
Webcam Captured Frame
          │
          ▼
   AI Model Inferences (YuNet / SFace / YOLO)
          │
          ▼
    Raw Detections Stream
          │
          ▼
┌────────────────────────────────────────────────────────┐
│               EVENT VALIDATION LAYER                   │
├────────────────────────────────────────────────────────┤
│ • Candidate Observation Accumulation                   │
│ • Category-Specific Strategy Evaluation                │
│ • State Machine Transitions:                           │
│     OBSERVED ──► CANDIDATE ──► VALIDATED ──► CLOSED    │
│                       │                                │
│                       └──► DISCARDED (Transient)       │
└────────────────────────────────────────────────────────┘
          │
          ▼ (Upon VALIDATED transition)
┌────────────────────────────────────────────────────────┐
│             EVIDENCE QUALITY VALIDATOR                 │
├────────────────────────────────────────────────────────┤
│ • Frame non-null & non-empty verification              │
│ • Dimension verification (W ≥ 160, H ≥ 120)            │
│ • Dead sensor / blank frame rejection                  │
│ • SHA-256 Cryptographic Hash Generation                │
└────────────────────────────────────────────────────────┘
          │
          ▼
 Verified Evidence Package & Session Timeline (timeline.json)
          │
          ▼
 Human Invigilator / Proctor Decision Support
```

---

## 2. Event Lifecycle State Machine

| State | Definition | Triggering Condition |
|---|---|---|
| **`OBSERVED`** | Initial single-frame raw detection | Detector observes anomaly in frame $N$ |
| **`CANDIDATE`** | Accumulating multi-frame observations | Multiple observations accumulate within absence tolerance |
| **`VALIDATED`** | Formally confirmed suspicious event | Meets category-specific frame count / duration persistence floor |
| **`EVIDENCE_CAPTURED`**| Visual keyframes and crops stored on disk | Evidence passes pre-flight quality checks and SHA-256 hash attached |
| **`CLOSED`** | Incident ended and resolved | Anomaly condition ends and absence tolerance ($0.5\text{s}$) expires |
| **`DISCARDED`** | Transient anomaly suppressed | Candidate expires before meeting validation criteria (e.g. 1-frame sneeze) |

---

## 3. Category-Specific Validation Strategies

| Event Type | Validation Strategy | Criteria | Operational Purpose |
|---|---|---|---|
| **`NO_FACE`** | Consecutive Frames & Duration | $\ge 3\text{ frames}$ ($\ge 1.0\text{s}$) | Eliminates momentary downward glances at scratchpad or sneezing |
| **`MULTIPLE_FACES`**| Duration Persistence | $\ge 2\text{ frames}$ ($\ge 0.5\text{s}$) | Eliminates transient motion blurs and background reflections |
| **`UNKNOWN_FACE`** | Consecutive Frames | $\ge 3\text{ frames}$ ($\ge 1.0\text{s}$) | Prevents false impostor triggers from transient head tilt |
| **`PHONE_DETECTED`**| Immediate Capture | $\ge 1\text{ frame}$ ($\ge 0.25\text{s}$) | Immediate evidence capture for high-severity prohibited items |
| **`BROWSER_FULLSCREEN_EXIT`** | Immediate Capture | $\ge 1\text{ event}$ ($0.0\text{s}$) | Instant lock and evidence generation on browser violation |

---

## 4. Telemetry & False-Positive Validation Metrics

Evaluated across realistic multi-second examination trial:

```json
{
  "total_frames_processed": 40,
  "raw_detections_count": 22,
  "candidate_events_created": 3,
  "validated_events_count": 2,
  "discarded_candidates_count": 1,
  "evidence_capture_failures": 0,
  "valid_evidence_count": 2,
  "mean_inference_latency_ms": 11.20,
  "mean_validation_latency_ms": 0.15,
  "mean_evidence_latency_ms": 0.85,
  "mean_total_latency_ms": 12.20,
  "effective_fps": 82.00
}
```

- **Transient Dropout Suppression**: 1-frame micro-glance at $t=2.25\text{s}$ was properly classified as `DISCARDED` and omitted from the final violation timeline.
- **Repeated Incident Separation**: Two distinct departures at $t=4.75\text{s}$ and $t=8.75\text{s}$ were correctly recorded as two separate incidents rather than merged into a single event.
- **Zero Evidence Failures**: 100% of validated evidence frames passed pre-flight quality checks and SHA-256 verification.

---

### **PHASE 9 STATUS: COMPLETE**
