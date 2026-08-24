# Structured Evidence Persistence & Multi-Modal Packaging (Phase 5)

## Overview

**Phase 5** establishes a model-independent **Evidence Persistence and Packaging layer**. It converts frame-by-frame telemetry and temporal state events into a standardized, self-contained evidence package suitable for human invigilator review, API transmission, and future Moodle proctoring plugin integration.

> [!IMPORTANT]
> **Safety Principle**: The evidence persistence layer records **purely factual, objective observations** (e.g. `"cell phone present from 00:01.000 to 00:02.500"`, `"person count changed from 1 to 2"`). It **does NOT** compute suspicion scores, violation probabilities, or automated disciplinary penalties. Final determination rests exclusively with human invigilators.

---

## 1. Architecture & Pipeline

```text
┌──────────────────────────────────────────────────────────┐
│                  ObjectDetector (Phase 1)                │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│              ObjectRelevanceFilter (Phase 2)             │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                VideoFrameSampler (Phase 3)               │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│               TemporalEventEngine (Phase 4)              │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                 EvidenceBuilder (Phase 5)                │
│ • Intelligent key-frame selector (Start, Peak, End)      │
│ • Bounding box bounds clamping & 10% padded cropping     │
│ • Professional visual evidence frame renderer            │
│ • Relative path manifest generator                       │
│ • ZIP archive packager                                   │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                  EvidencePackage                         │
│ ├── manifest.json                                        │
│ ├── frames/ (annotated key full-frame snapshots)         │
│ ├── crops/ (high-resolution object crops)                │
│ ├── events/ (individual event JSON files)                │
│ ├── timeline/ (Gantt chart PNG & timeline JSON)          │
│ └── evidence_package.zip                                 │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

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
│   └── frame_000030_cell_phone_02.jpg
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

## 3. Intelligent Key-Frame Selection & Deduplication

Rather than saving every frame (which would cause disk bloat), the `EvidenceBuilder` selects critical evidence frames:
1. **EVENT START**: The first frame where a relevant object appears.
2. **EVENT PEAK**: The frame where the object achieved its highest detection confidence.
3. **EVENT END**: The last frame where the object was observed.
4. **PERSON TRANSITIONS**: The exact frame where a person count transition occurred ($1 \rightarrow 2$, $2 \rightarrow 0$, $0 \rightarrow 1$).

### Deduplication
If a frame serves multiple roles (e.g. both Start and Peak, or shared across multiple events), only **one** full-frame image is rendered and saved; events cross-reference the single `frame_id` in their `key_frame_ids` list.

---

## 4. Object Cropping with Boundary Safety

For every relevant detected object on key frames, an individual cropped image is extracted:
* **Bounds Clamping**: Strict mathematical clamping to $[0, W]$ and $[0, H]$ prevents coordinate overflow or negative indexing.
* **Context Padding**: Adds configurable padding (default: `10%`) around the bounding box so the human proctor can see surrounding context (e.g. candidate holding a phone).
* **Quality Preservation**: Saved at 95% JPEG quality.

---

## 5. Machine-Readable Manifest (`manifest.json`)

All asset references in `manifest.json` use **portable relative paths** (zero machine-specific absolute paths), allowing the package to be unpacked seamlessly on local PCs, Kaggle, cloud servers, or Moodle.

```json
{
  "schema_version": "1.0",
  "package_id": "evidence_pkg_20260824_094629",
  "created_at": "2026-08-24T09:46:29.256800+00:00",
  "source": {
    "type": "video",
    "filename": "test_presence_transitions.mp4",
    "duration_seconds": 4.0,
    "fps": 30.0,
    "width": 480,
    "height": 360,
    "total_frames": 120
  },
  "summary": {
    "total_video_frames": 120,
    "sampled_frames_count": 8,
    "evidence_frames_count": 6,
    "object_crops_count": 7,
    "temporal_events_count": 2,
    "person_count_changes_count": 3,
    "target_sampling_fps": 2.0
  },
  "events": [
    {
      "event_id": "obj_event_001",
      "modality": "object",
      "event_type": "temporal_object_presence",
      "object_class": "person",
      "start_seconds": 0.0,
      "end_seconds": 1.5,
      "duration_seconds": 1.5,
      "formatted_start": "00:00.000",
      "formatted_end": "00:01.500",
      "detection_count": 4,
      "max_confidence": 0.9446,
      "average_confidence": 0.9435,
      "is_duration_qualified": true,
      "key_frame_ids": ["frame_000000", "frame_000030", "frame_000045"]
    }
  ],
  "frames": [
    {
      "frame_id": "frame_000030",
      "frame_index": 30,
      "timestamp_seconds": 1.0,
      "formatted_timestamp": "00:01.000",
      "person_count": 2,
      "frame_relative_path": "frames/frame_000030_00_01_000.jpg",
      "selection_reason": "person_count_change (1->2)",
      "objects": [
        {
          "object_id": "frame_000030_obj_01",
          "class_name": "person",
          "confidence": 0.944,
          "bbox": [19, 102, 233, 287],
          "crop_relative_path": "crops/frame_000030_person_01.jpg"
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

## 6. Face Subsystem Integration Contract

The `FaceEvidenceAdapter` provides a clean contract interface for future integration of face presence/verification observations into the unified evidence package without modifying existing face recognition logic:

```python
from src.object_detection.evidence import FaceEvidenceAdapter

face_event = FaceEvidenceAdapter.create_face_presence_event(
    event_id="face_event_001",
    status="SINGLE_FACE",
    start_seconds=0.0,
    end_seconds=30.0,
    frame_indices=[0, 15, 30],
    face_count=1,
    key_frame_ids=["frame_000000"],
)
```

---

## 7. CLI Usage

### Generate Complete Evidence Package & ZIP Archive
```bash
python scripts/analyze_video.py path/to/video.mp4 \
    --sample-fps 2.0 \
    --save-evidence \
    --evidence-dir data/results/object_detection/evidence \
    --zip-evidence \
    --plot-timeline data/results/object_detection/evidence/timeline/timeline_gantt.png
```

---

## 8. Kaggle GPU Execution

```bash
python scripts/kaggle_object_detection.py \
    --video /kaggle/input/exam_sessions/student_01.mp4 \
    --sample-fps 2.0 \
    --save-evidence \
    --evidence-dir /kaggle/working/evidence \
    --zip-evidence
```

---

## 9. Known Phase 5 Limitations

1. **Local Filesystem Packaging**: Packages are written locally/on-disk; streaming directly to cloud S3 or Moodle REST endpoints will be handled during integration phases.
2. **Single Camera Source**: Formatted for primary candidate webcam streams; multi-angle camera bundle support will be added in future multi-camera phases.
3. **No Automatic Cheating Scoring**: Reports objective visual facts for human proctor review.
