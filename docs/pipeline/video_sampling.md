# Video & Webcam Frame Sampling Pipeline (Phase 3)

## Overview

**Phase 3** introduces a robust **Video / Webcam Frame Sampling Pipeline** that periodically processes frames from video streams through the existing `ObjectDetector` and `ObjectRelevanceFilter` subsystems.

> [!IMPORTANT]
> **Safety Principle**: The video analysis pipeline records **objective visual evidence** over time. It **does NOT** compute cumulative risk/suspicion scores, nor does it make automated cheating or violation decisions. A human proctor/invigilator reviews the timeline evidence.

---

## 1. Architecture & Pipeline

```text
┌──────────────────────────────────────────────────────────┐
│              Video Stream (MP4, AVI, WebM)               │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                  VideoFrameSampler                       │
│ • Probes video resolution, source FPS, and frame count   │
│ • Configurable sampling rate (e.g. 2 FPS from 30 FPS)    │
│ • Fast frame skipping via OpenCV grab()                  │
│ • Exact timestamp calculation: t = frame_idx / fps       │
└────────────────────────────┬─────────────────────────────┘
                             │
                             │ (FrameSample: BGR array, frame_idx, timestamp)
                             ▼
┌──────────────────────────────────────────────────────────┐
│                   ObjectDetector                         │
│ • YOLO11n GPU/CPU inference                              │
│ • Raw bounding boxes, class names, confidences           │
└────────────────────────────┬─────────────────────────────┘
                             │
                             │ (ObjectDetectionResult)
                             ▼
┌──────────────────────────────────────────────────────────┐
│               ObjectRelevanceFilter                      │
│ • Filters proctoring taxonomy (person, phone, laptop...) │
│ • Per-class confidence thresholds                        │
│ • Person count tracking (0, 1, 2+ persons)               │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│           VideoObjectAnalyzer / Timeline                 │
│ • Chronological TimelineEntry records                    │
│ • Object presence intervals (first/last seen, counts)    │
│ • Performance telemetry (inference & pipeline FPS)       │
│ • JSON-serializable VideoAnalysisReport                  │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Frame Sampling Strategy

Processing every single frame in a 30 FPS or 60 FPS video is computationally wasteful for proctoring. The `VideoFrameSampler` reduces computational overhead by sampling at a configurable target rate (default: `2.0 FPS`):

$$\text{step} = \max\left(1, \text{round}\left(\frac{\text{source\_fps}}{\text{target\_sampling\_fps}}\right)\right)$$

* **Fast Decoding**: Uses OpenCV `cap.grab()` for skipped frames to advance the decoder position without incurring full JPEG/H.264 image decompression overhead.
* **Exact Timestamps**: Calculates $t = \frac{\text{frame\_index}}{\text{source\_fps}}$ and formats timestamps into standardized `MM:SS.mmm` strings (e.g. `00:12.500`).

---

## 3. Timeline Evidence Structure

The pipeline generates an objective chronological timeline:

```text
[00:00.000] Frame      0 | Persons: 1 | Relevant: person (0.92)
[00:00.500] Frame     15 | Persons: 1 | Relevant: person (0.91)
[00:01.000] Frame     30 | Persons: 1 | Relevant: person (0.94), cell phone (0.87)
[00:01.500] Frame     45 | Persons: 1 | Relevant: person (0.93), cell phone (0.85)
[00:02.000] Frame     60 | Persons: 0 | Relevant: (none)
```

### Temporal Presence Intervals
The analyzer summarizes occurrences of each unique relevant object class:
* `class_name`: Name of detected object (e.g. `cell phone`).
* `first_seen_seconds`: First timestamp where object was detected.
* `last_seen_seconds`: Most recent timestamp where object was detected.
* `detection_count`: Total number of sampled frames containing the object.
* `max_confidence`: Highest confidence observed.
* `timestamps`: Full list of detection timestamps.

---

## 4. CLI Usage

### Analyze Video File
```bash
python scripts/analyze_video.py path/to/video.mp4 --sample-fps 2.0
```

### Save Annotated Visual Frames
```bash
python scripts/analyze_video.py path/to/video.mp4 \
    --sample-fps 2.0 \
    --save-frames \
    --frames-dir data/results/object_detection/video/frames/
```

### Export JSON Evidence Report
```bash
python scripts/analyze_video.py path/to/video.mp4 \
    --sample-fps 2.0 \
    --json data/results/object_detection/video/evidence.json
```

---

## 5. Kaggle GPU Execution

Run video sampling on Kaggle NVIDIA GPUs with automatic CUDA acceleration:

```bash
python scripts/kaggle_object_detection.py \
    --video /kaggle/input/test_proctoring_videos/sample.mp4 \
    --sample-fps 2.0 \
    --save-frames \
    --json /kaggle/working/evidence.json
```

---

## 6. Python API Example

```python
from pathlib import Path
from src.object_detection import ObjectDetector, ObjectRelevanceFilter, VideoObjectAnalyzer

# Initialize components
detector = ObjectDetector(model_name="yolo11n.pt")
relevance_filter = ObjectRelevanceFilter(default_threshold=0.25)
analyzer = VideoObjectAnalyzer(
    detector=detector,
    relevance_filter=relevance_filter,
    target_sampling_fps=2.0,
)

# Run video analysis
report = analyzer.analyze_video(
    video_path=Path("data/samples/synthetic/test_presence_transitions.mp4"),
    save_annotated_dir=Path("data/results/object_detection/video/frames/"),
)

print(f"Sampled {report.total_sampled_frames} frames in {report.total_processing_time_ms:.2f} ms")
for entry in report.timeline:
    print(
        f"[{entry.formatted_timestamp}] Persons: {entry.person_count}, Objects: {entry.relevant_count}"
    )
```

---

## 7. Known Phase 3 Limitations

1. **Prerecorded Stream Focus**: Designed for file-based streams; live browser WebRTC webcam streaming will be integrated in future phases.
2. **Periodic Sampling Window**: Events occurring entirely between sample points (e.g. a 0.2s phone flash between 0.5s intervals) may not be sampled; sampling FPS can be tuned up (e.g. 5–10 FPS) where higher temporal resolution is required.
3. **No Automatic Cheating Classification**: The pipeline outputs factual telemetry; human invigilators make all proctoring determinations.
