# Structured Evidence Persistence & Visual Explanation Packaging

## Overview

The **Evidence Persistence & Visual Explanation Packaging layer** transforms raw frame telemetry and temporal state events into a standardized, human-reviewable evidence package.

Every generated screenshot, crop, and manifest item provides **clear, precise visual evidence explaining exactly WHY an area was detected and marked**, without subjective risk or cheating scores.

> [!IMPORTANT]
> **Factual Telemetry Principle**: The evidence persistence layer records **purely factual, objective observations** (e.g. `"Cell phone detected by object detector"`, `"Additional person (#02) detected in exam perimeter"`, `"Person count transition detected (1 -> 2 persons)"`). It **does NOT** compute suspicion scores, violation probabilities, or automated disciplinary penalties. Final determination rests exclusively with human invigilators.

---

## 1. Visual Evidence Design

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                           EVIDENCE SCREENSHOT                            │
├──────────────────────────────────────────┬───────────────────────────────┤
│                                          │    PROCTORING AI EVIDENCE     │
│   ORIGINAL WEBCAM FRAME                  ├───────────────────────────────┤
│   • Exact BBoxes (x1, y1)-(x2, y2)       │ TIME: 00:07.500 | FRAME: 219  │
│   • Class Palette (Green=Candidate,      │ PERSONS IN SCENE: 2           │
│     Red=2nd Person, Orange=Phone...)     │                               │
│   • Precision Corner Brackets            │ KEY FRAME REASON:             │
│   • Tag Banners & Leader Lines           │ EVENT PEAK (Cell phone 87%)   │
│   • Top Status Bar                       │                               │
│                                          │ MARKED REGIONS (2):           │
│                                          │ • [person_01] PERSON #01      │
│                                          │   Why: Primary candidate      │
│                                          │ • [phone_01] CELL PHONE (87%) │
│                                          │   Why: Cell phone detected    │
│                                          │                               │
│                                          │ TEMPORAL CONTEXT:             │
│                                          │ Event: obj_event_003 (2.5s)   │
│                                          │ Observations: 6 samples       │
└──────────────────────────────────────────┴───────────────────────────────┘
```

### Zero-Occlusion Side Explanation Panel
To prevent metadata overlays from covering the candidate's face, hands, or detected devices, the explanation panel is rendered as a dedicated **sidebar attached to the right of the video canvas**.

---

## 2. Crop Evidence with Metadata Headers

For every marked object, an individual cropped image is extracted with a **58px metadata header banner**:

```text
┌────────────────────────────────────────────────────────┐
│ CELL PHONE | 87% | ID: phone_01                        │
│ TIME: 00:07.500 | FRAME: 000219 | BBOX: (742,1012)-(901,1284) │
│ WHY: Cell phone detected by object detector            │
├────────────────────────────────────────────────────────┤
│                                                        │
│               [ CROPPED OBJECT IMAGE ]                 │
│                 (10% context padding)                  │
│                                                        │
└────────────────────────────────────────────────────────┘
```

* **Bounds Clamping**: Bounding boxes are mathematically clamped to $[0, W]$ and $[0, H]$.
* **Context Padding**: 10% context margin around the object provides environmental context.
* **Header Banner**: Explicitly displays class, confidence, coordinates, timestamp, frame number, and factual reason.

---

## 3. Directory Structure

```text
data/results/object_detection/evidence/
│
├── manifest.json
│
├── frames/
│   ├── frame_000000_00_00_000.jpg
│   ├── frame_000030_00_01_000.jpg
│   └── frame_000045_00_01_500.jpg
│
├── crops/
│   ├── frame_000030_person_01.jpg
│   ├── frame_000030_person_02.jpg
│   └── frame_000030_phone_01.jpg
│
├── events/
│   ├── obj_event_001.json
│   └── person_change_001.json
│
└── timeline/
    ├── timeline.json
    └── timeline_gantt.png
```

---

## 4. Machine-Readable Manifest (`manifest.json`)

All asset references in `manifest.json` use **portable relative paths** with explicit `reason` and `explanation` attributes:

```json
{
  "schema_version": "1.0",
  "package_id": "evidence_pkg_20260824_101208",
  "created_at": "2026-08-24T10:12:08.043008+00:00",
  "source": {
    "type": "video",
    "filename": "exam_video.mp4",
    "duration_seconds": 30.0,
    "fps": 30.0,
    "width": 1920,
    "height": 1080,
    "total_frames": 900
  },
  "summary": {
    "total_video_frames": 900,
    "sampled_frames_count": 60,
    "evidence_frames_count": 6,
    "object_crops_count": 8,
    "temporal_events_count": 2,
    "person_count_changes_count": 1,
    "target_sampling_fps": 2.0
  },
  "events": [
    {
      "event_id": "obj_event_001",
      "modality": "object",
      "event_type": "temporal_object_presence",
      "object_class": "cell phone",
      "start_seconds": 5.0,
      "end_seconds": 7.5,
      "duration_seconds": 2.5,
      "formatted_start": "00:05.000",
      "formatted_end": "00:07.500",
      "detection_count": 6,
      "max_confidence": 0.8742,
      "average_confidence": 0.8415,
      "is_duration_qualified": true,
      "reason": "Cell phone continuously observed for 2.50s across 6 sampled frames.",
      "key_frame_ids": ["frame_000150", "frame_000219", "frame_000225"]
    }
  ],
  "frames": [
    {
      "frame_id": "frame_000219",
      "frame_index": 219,
      "timestamp_seconds": 7.5,
      "formatted_timestamp": "00:07.500",
      "person_count": 2,
      "frame_relative_path": "frames/frame_000219_00_07_500.jpg",
      "selection_reason": "KEY FRAME: PEAK CONFIDENCE (cell phone 87%)",
      "explanation": "Key evidence frame captured at 00:07.500 (Frame 00219) due to peak confidence. Persons in scene: 2.",
      "event_context": {
        "event_id": "obj_event_001",
        "duration_seconds": 2.5,
        "detection_count": 6,
        "max_confidence": 0.8742
      },
      "objects": [
        {
          "object_id": "frame_000219_person_01",
          "instance_id": "person_01",
          "class_name": "person",
          "confidence": 0.9412,
          "bbox": [120, 100, 450, 480],
          "reason": "Primary candidate person detected in exam area",
          "timestamp_seconds": 7.5,
          "frame_index": 219,
          "crop_relative_path": "crops/frame_000219_person_01.jpg"
        },
        {
          "object_id": "frame_000219_phone_02",
          "instance_id": "phone_02",
          "class_name": "cell phone",
          "confidence": 0.8742,
          "bbox": [742, 1012, 901, 1284],
          "reason": "Cell phone detected by object detector",
          "timestamp_seconds": 7.5,
          "frame_index": 219,
          "crop_relative_path": "crops/frame_000219_phone_02.jpg"
        }
      ]
    }
  ],
  "artifacts": {
    "timeline_data": "timeline/timeline.json"
  }
}
```

---

## 5. Timeline Gantt Chart Visualization

The visualizer generates `timeline/timeline_gantt.png` featuring:
1. **Object Presence Rows**: Colored Gantt interval bars with Event ID, duration, and max confidence badges (`obj_event_001: 2.5s (87%)`).
2. **Factual Person Count Step Curve**: Stepped curve with callouts showing exact transition timestamps (`00:01.000: 1 → 2 persons`).
