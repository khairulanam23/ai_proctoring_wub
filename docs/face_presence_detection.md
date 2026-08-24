# Face Presence & Multi-Face Detection Architecture

## Overview
This document describes the Phase 1 Face Presence and Multi-Face Detection subsystem for the AI Proctoring project. The module monitors exam video streams and images to determine whether an authorized examinee is present, absent, or if unauthorized additional individuals enter the camera field of view.

> [!IMPORTANT]
> **Observation-Based Evidence Philosophy**: This component does **not** compute cumulative risk scores or automatically declare cheating. It serves as an objective observer that records timestamped state transitions and structured evidence for human/proctor review.

---

## Detection States

The presence detector categorizes each processed frame into one of three mutually exclusive states:

| Status Enum | Detected Faces | Operational Meaning |
| :--- | :---: | :--- |
| `NO_FACE` | `0` | Examinee absent, camera blocked, or subject fully out of frame. |
| `SINGLE_FACE` | `1` | Normal nominal state (single examinee present). |
| `MULTIPLE_FACES` | `2+` | Multiple individuals detected in camera view (potential third-party assistance). |

---

## Architecture & Data Flow

```text
Input Frame (Image / Video Stream)
              │
              ▼
   FaceDetector (YuNet DNN)
              │
      (Bounding Boxes, Confidence Scores, 5 Facial Landmarks)
              │
              ▼
   FacePresenceAnalyzer
              │
      (Evaluate Face Count -> NO_FACE / SINGLE_FACE / MULTIPLE_FACES)
              │
              ▼
   VideoPresenceAnalyzer (State Transition Engine)
              │
      (Detect Status Changes: e.g. SINGLE_FACE -> MULTIPLE_FACES)
              │
              ▼
   Structured Evidence Output (JSON / Logs)
```

---

## Structured Result Types

### 1. `FacePresenceResult` (Per Frame)
```python
@dataclass
class FacePresenceResult:
    status: FacePresenceStatus       # NO_FACE, SINGLE_FACE, MULTIPLE_FACES
    face_count: int                  # Total number of detected faces
    faces: List[FaceDetection]       # Bounding boxes, landmarks, confidence
    image_shape: Tuple[int, int, int]# (H, W, C)
    inference_time_ms: float         # Detection latency
    frame_index: Optional[int]       # Sequential frame index
    timestamp_sec: Optional[float]   # Timestamp in seconds
```

### 2. `PresenceEvent` (Timeline Transition Event)
Represents a recorded state change across video frames:
```json
{
  "timestamp_formatted": "00:00:01.000",
  "timestamp_sec": 1.0,
  "frame_index": 31,
  "event_type": "MULTIPLE_FACES",
  "previous_status": "SINGLE_FACE",
  "current_status": "MULTIPLE_FACES",
  "face_count": 2,
  "source_media": "exam_session.mp4",
  "faces": [
    {
      "bbox": [302, 137, 81, 103],
      "confidence": 0.9433,
      "landmarks": [[332.6, 167.0], [368.4, 167.3], [356.1, 186.0], [335.4, 206.1], [366.0, 206.7]]
    }
  ]
}
```

---

## State Transition Logic

In video streams, analyzing 30 frames per second generates substantial redundant data. The `VideoPresenceAnalyzer` suppresses repetitive frame logging and emits high-priority events only on **state transitions**:

1. **Initial State Capture**: Emits the initial status at `00:00:00.000`.
2. **Transition Trigger**: An event is generated only when `current_status != previous_status`.
   * `SINGLE_FACE -> MULTIPLE_FACES`: Logs intrusion of second person with exact timestamp.
   * `SINGLE_FACE -> NO_FACE`: Logs departure of examinee.
   * `NO_FACE -> SINGLE_FACE`: Logs return of examinee.
   * `MULTIPLE_FACES -> SINGLE_FACE`: Logs exit of second person.
3. **Cumulative Statistics**: Maintains full frame-by-frame status distribution percentages and CPU execution latencies.

---

## CPU Performance Baseline

Benchmarked on host CPU (`x86_64`) across 50 iterations per resolution using OpenCV YuNet:

| Resolution | Format / Target | Avg Latency (ms) | Min Latency (ms) | Max Latency (ms) | Throughput (FPS) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **250 × 250** | Cropped Face ROI | **4.11 ms** | 3.03 ms | 8.83 ms | ~243 FPS |
| **480 × 360** | Standard Web Stream | **13.20 ms** | 8.68 ms | 18.88 ms | ~75 FPS |
| **640 × 480** | VGA Webcam | **22.07 ms** | 17.26 ms | 37.88 ms | ~45 FPS |
| **1280 × 720** | 720p HD | **82.31 ms** | 68.71 ms | 110.61 ms | ~12 FPS |

*Recommendation*: For real-time CPU exam proctoring, processing downscaled frames at **480×360** or **640×480** delivers high real-time throughput (>45 FPS) with minimal latency overhead.

---

## Limitations & Technical Considerations

1. **Extreme Yaw/Profile Angles**: YuNet is optimized for frontal and semi-frontal faces (±60° yaw). Extreme head rotation (>75° sideways) may momentarily register as `NO_FACE`.
2. **Heavy Facial Occlusions**: Dense masking or hands completely covering the nose and mouth can reduce detector confidence below the 0.6 threshold.
3. **Lighting Extremes**: Severe backlight or darkness degrades edge contrast; future phases may incorporate image pre-normalization.
