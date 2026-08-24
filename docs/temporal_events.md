# Temporal Event Detection & State Management (Phase 4)

## Overview

**Phase 4** introduces a **temporal state engine** that transforms frame-by-frame visual detections into continuous, consolidated temporal events over video streams.

> [!IMPORTANT]
> **Safety Principle**: The temporal state engine is an **evidence aggregation mechanism**, NOT a cheating detector. It records objective, reviewable facts (e.g. `"cell phone present for 3.5s"` or `"person count changed from 1 to 2"`). It **does NOT** compute suspicion scores, violation probabilities, or automated disciplinary decisions. Final determination rests solely with human proctors/invigilators.

---

## 1. Temporal Architecture

```text
┌──────────────────────────────────────────────────────────┐
│           Raw Sampled Video Frames (Phase 3)             │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│              ObjectRelevanceFilter (Phase 2)             │
│ • Filters examination taxonomy (person, phone, book...)  │
│ • Evaluates per-class confidence cutoffs                 │
└────────────────────────────┬─────────────────────────────┘
                             │
                             │ (Raw frame detections & person counts)
                             ▼
┌──────────────────────────────────────────────────────────┐
│               TemporalEventEngine                        │
│ • State machine per object class (Absent → Present → End)│
│ • Absence tolerance gap bridging (e.g. 1.0s)             │
│ • Person count transition logger (e.g. 1 → 2)            │
│ • Duration qualification filter                          │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                Consolidated Output                       │
│ • Temporal Events (start, end, duration, confidence)     │
│ • Person Count Transitions (prev_count, new_count, time) │
│ • Complete raw timeline preserved                        │
│ • Publication-quality Gantt timeline visualization       │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Event Consolidation & Absence Tolerance

### The Problem with Naive Frame-by-Frame Logs
Sampling at 2 FPS produces repeated disjoint entries:
```text
00:01.000 → phone (0.88)
00:01.500 → phone (0.92)
00:02.000 → phone (0.85)
```

### Consolidated Temporal Event
The `TemporalEventEngine` merges consecutive and near-consecutive observations into a single continuous event:
```text
Object:       cell phone
Start:        00:01.000
End:          00:02.000
Duration:     1.000 seconds
Observations: 3 samples
Max Conf:     0.92
Avg Conf:     0.88
```

### Absence Tolerance Gap Bridging
In real-world webcam conditions, occasional occlusions or model drops can cause a single missing sample. The `absence_tolerance_seconds` parameter (default: `1.0s`) prevents premature event fragmentation:
* If an object disappears for less than the tolerance window and reappears, it is tracked as a single continuous event.
* If the gap exceeds the tolerance window, the previous event is finalized and a new event begins upon reappearance.

---

## 3. Duration Qualification

To distinguish brief transitory glimpses from sustained presence without discarding evidence:
* **Observed Event**: All events, regardless of duration, are retained in the evidence log.
* **`is_duration_qualified`**: Flagged as `True` if `duration_seconds >= min_event_duration_seconds` (default: `1.0s`).

---

## 4. Person Count Transitions

The engine records discrete step changes in scene occupancy:
```json
{
  "event_id": "person_change_001",
  "timestamp_seconds": 1.0,
  "formatted_timestamp": "00:01.000",
  "previous_count": 1,
  "new_count": 2,
  "frame_index": 30
}
```

*Note: Changes in person count are recorded objectively as factual transitions.*

---

## 5. Visual Gantt Chart

The visualizer generates a publication-quality timeline chart displaying:
1. **Tracked Object Intervals**: Colored horizontal duration bars with length and confidence annotations.
2. **Person Count State Stepped Plot**: Step curve showing occupancy transitions over time with annotated shift markers.

**Output Path**: `data/results/object_detection/video/timeline_gantt.png`

---

## 6. CLI Usage

### Analyze Video with Temporal Events & Gantt Chart
```bash
python scripts/analyze_video.py path/to/video.mp4 \
    --sample-fps 2.0 \
    --absence-tolerance 1.0 \
    --min-event-duration 1.0 \
    --plot-timeline data/results/object_detection/video/timeline_gantt.png \
    --json data/results/object_detection/video/evidence_temporal.json
```

---

## 7. JSON Evidence Schema

```json
{
  "video_path": "path/to/video.mp4",
  "target_sampling_fps": 2.0,
  "total_sampled_frames": 8,
  "temporal_events": [
    {
      "event_id": "obj_event_001",
      "object_class": "cell phone",
      "start_seconds": 1.0,
      "end_seconds": 2.5,
      "duration_seconds": 1.5,
      "formatted_start": "00:01.000",
      "formatted_end": "00:02.500",
      "detection_count": 4,
      "max_confidence": 0.9124,
      "average_confidence": 0.8845,
      "is_duration_qualified": true,
      "observation_timestamps": [1.0, 1.5, 2.0, 2.5],
      "representative_bbox": [100, 120, 150, 180]
    }
  ],
  "person_count_changes": [
    {
      "event_id": "person_change_001",
      "timestamp_seconds": 1.0,
      "formatted_timestamp": "00:01.000",
      "previous_count": 1,
      "new_count": 2,
      "frame_index": 30
    }
  ],
  "timeline": [ ... ]
}
```

---

## 8. Known Phase 4 Limitations

1. **Class-Level Tracking**: Does not track individual object instances across time (e.g. distinguishing Phone A from Phone B).
2. **Context-Free Events**: Records object presence intervals; contextual correlation with candidate gaze or active screen status is deferred to later evidence layers.
3. **No Automated Disciplinary Scores**: The engine produces human-readable evidence for invigilator review.
