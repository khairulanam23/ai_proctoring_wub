# Proctoring Object Selection & Relevance Filtering (Phase 2)

## Overview

**Phase 2** introduces a configurable **relevance filtering layer** on top of the raw YOLO object detector. It isolates objects meaningful in an examination environment from general background clutter while preserving complete raw detection data for downstream evidence systems.

> [!IMPORTANT]
> **Safety Principle**: A relevant object is **NOT** automatically a violation or an indication of cheating. The system merely identifies and structures visual objects for human invigilator review. No cumulative risk or suspicion scores are computed.

---

## 1. Relevance Architecture

```text
┌──────────────────────────────────────────────────────────┐
│                   Raw Image Input                        │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│         ObjectDetector (YOLO11n Inference)               │
│ • Raw bounding boxes, class names, confidence scores     │
└────────────────────────────┬─────────────────────────────┘
                             │
                             │ (ObjectDetectionResult)
                             ▼
┌──────────────────────────────────────────────────────────┐
│              ObjectRelevanceFilter                       │
│ • Relevance taxonomy lookup                              │
│ • Class-specific confidence thresholding                 │
│ • Person count extraction (0, 1, 2+ persons)             │
│ • Raw data preservation                                  │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│           ProctoringDetectionReport                      │
│ • relevant_objects: List[DetectedObject]                 │
│ • ignored_objects: List[DetectedObject]                  │
│ • person_count: int                                      │
│ • JSON-serializable structured telemetry                 │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Relevant Object Taxonomy

The following MS COCO classes are configured as initial candidate objects for proctoring monitoring:

| Object Class | Category Description | Default Confidence Threshold |
| :--- | :--- | :---: |
| **`person`** | Candidate / additional people | `0.25` |
| **`cell phone`** | Mobile communication devices | `0.40` |
| **`laptop`** | Secondary computer screens / laptops | `0.35` |
| **`book`** | Physical books, notebooks, notes | `0.30` |
| **`tablet`** | Digital tablets / iPads | `0.35` |
| **`remote`** | Remote controls / electronic devices | `0.35` |
| **`keyboard`** | Peripheral input hardware | `0.30` |
| **`mouse`** | Peripheral input hardware | `0.30` |
| **`backpack`** | Bags / storage items | `0.30` |
| **`handbag`** | Bags / storage items | `0.30` |
| **`suitcase`** | Luggage / storage items | `0.35` |
| **`bottle`** | Allowed / restricted beverages | `0.30` |

### Irrelevant Background Objects
Objects outside the proctoring taxonomy (e.g. `chair`, `tie`, `cup`, `dining table`, `car`) are routed to `ignored_objects` and excluded from default proctoring visualizations.

---

## 3. Difference Between DETECTED and RELEVANT

* **`DETECTED`**: The raw YOLO model detected a visual feature corresponding to an 80-class COCO category with confidence above the base detector threshold.
* **`RELEVANT`**: The detected object belongs to the enabled proctoring taxonomy AND satisfies its class-specific confidence threshold.
* **`SUSPICIOUS`**: *Not implemented in Phase 2*. Suspiciousness requires temporal persistence across video frames and context (e.g. "phone held for 3 seconds"), which is handled in subsequent phases and human review.

---

## 4. Person Handling

The relevance layer calculates the exact count of verified persons in the scene:
* `person_count = 0`: No candidate visible.
* `person_count = 1`: Standard single-candidate setting.
* `person_count >= 2`: Multiple persons detected simultaneously in frame.

*Note: Person counts are reported as objective counts without making cheating inferences.*

---

## 5. Python API Usage

```python
import cv2
from src.object_detection import ObjectDetector, ObjectRelevanceFilter

# 1. Initialize detector and relevance filter
detector = ObjectDetector(model_name="yolo11n.pt")
relevance_filter = ObjectRelevanceFilter(
    default_threshold=0.25,
    class_thresholds={"cell phone": 0.40, "laptop": 0.35},
)

# 2. Run detection and filtering
image = cv2.imread("data/samples/Colin_Powell/Colin_Powell_0001.jpg")
raw_result = detector.detect(image)
report = relevance_filter.filter(raw_result)

print(f"Persons: {report.person_count}")
print(f"Relevant Objects ({report.relevant_count}):")
for obj in report.relevant_objects:
    print(f"  • {obj.class_name}: {obj.confidence:.2f} at bbox [{obj.x1}, {obj.y1}, {obj.x2}, {obj.y2}]")

# 3. Export structured JSON report
report_json = report.to_dict()
```

---

## 6. Visualization & CLI

### Default Visual Mode (Relevant Objects Only)
```bash
python scripts/detect_objects.py path/to/image.jpg
```
Draws prominent high-contrast bounding boxes with class labels and confidence percentages for relevant objects, plus a top metadata banner showing person and object counts.

### Debug Mode (Show All Detections)
```bash
python scripts/detect_objects.py path/to/image.jpg --show-all
```
Renders relevant objects in vivid colors and ignored background objects in subdued gray boxes labeled `[class_name XX%]`.

### JSON Report Export
```bash
python scripts/detect_objects.py path/to/image.jpg --json
```

---

## 7. Known Phase 2 Limitations

1. **Single-Frame Static Scope**: Analyzes isolated frames without temporal smoothing (temporal event tracking is deferred to Phase 4).
2. **Standard COCO Vocabulary**: Uses pretrained COCO weights; specialized objects like handwritten cheat sheets or smartwatches require dedicated dataset tuning.
3. **No Risk / Violation Decision**: Objective detection output only.
