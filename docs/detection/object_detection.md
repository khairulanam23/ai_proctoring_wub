# Object Detection Foundation (Phase 1)

## Overview

The **Object Detection Foundation** provides a modular, lightweight AI subsystem for real-time visual object detection within candidate examination environments.

It establishes a clean separation between **Local Development** (architecture, configuration, unit testing on CPU) and **Kaggle Execution** (NVIDIA GPU inference, live model downloads, video processing, and benchmarking).

---

## 1. Architecture & Workflow

```text
┌──────────────────────────────────────────────────────────┐
│                   LOCAL ENVIRONMENT                      │
│ • Code development & modular design                      │
│ • Unit tests & input validation (CPU)                    │
│ • Zero model weight downloads into Git                   │
└────────────────────────────┬─────────────────────────────┘
                             │
                             │ Git Sync (GitHub)
                             ▼
┌──────────────────────────────────────────────────────────┐
│                 KAGGLE GPU ENVIRONMENT                   │
│ • NVIDIA GPU execution (CUDA)                            │
│ • Pip install: requirements-kaggle.txt                   │
│ • Lightweight YOLO11n loading                            │
│ • Real-time inference & GPU performance benchmarking     │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Model Selection Rationale

* **Model**: **`YOLO11n` (Ultralytics YOLO Nano)**
* **Parameters**: $\approx 2.6\text{M}$ parameters (extremely lightweight)
* **Pretrained Weights**: MS COCO (80 classes including `person`, `cell phone`, `laptop`, `book`, `chair`, `keyboard`, etc.)
* **Selection Justification**:
  * Ultra-low inference latency on both NVIDIA GPUs and modern CPUs.
  * Native PyTorch / TensorRT export compatibility for eventual real-time proctoring streams.
  * Clean Python API via Ultralytics without requiring custom C++ bindings for Phase 1.

---

## 3. Installation & Dependency Separation

### A. Local Machine (Core runtime & Testing)
```bash
pip install -e .
```

### B. GPU Runtime / Full Multi-Modal Environment
```bash
pip install -e ".[kaggle]"    # or: pip install -e ".[all]"
```

Using standard extras in `pyproject.toml` installs PyTorch, torchvision, and Ultralytics without requiring ad-hoc requirements files.

---

## 4. Object Detection Core API

### Initialization
```python
from src.object_detection import ObjectDetector

# Automatic device resolution (CUDA if available, else CPU)
detector = ObjectDetector(
    model_name="yolo11n.pt",
    confidence_threshold=0.25,
    device=None,  # Auto-selects "cuda" or "cpu"
)

print(f"Active Device: {detector.device}")
```

### Inference
```python
import cv2

image = cv2.imread("path/to/test.jpg")
result = detector.detect(image, confidence_threshold=0.30)

print(f"Total Objects Detected: {result.count}")
for obj in result.objects:
    print(
        f"  • {obj.class_name}: {obj.confidence:.2f} at bbox [{obj.x1}, {obj.y1}, {obj.x2}, {obj.y2}]"
    )
```

### Visualization
```python
annotated_bgr = detector.visualize(image, result)
cv2.imwrite("data/results/object_detection/output.jpg", annotated_bgr)
```

---

## 5. Detection Result Data Structure

### `DetectedObject`
* `class_id: int`: Integer class identifier (0 for person, 67 for cell phone, etc.).
* `class_name: str`: Human-readable class name.
* `confidence: float`: Detection confidence score $\in [0.0, 1.0]$.
* `bbox: Tuple[int, int, int, int]`: Bounding box `(x1, y1, x2, y2)` in pixel coordinates.
* `x1, y1, x2, y2, width, height`: Convenient coordinate properties.
* `to_dict() -> Dict[str, Any]`: JSON-serializable dictionary.

### `ObjectDetectionResult`
* `objects: List[DetectedObject]`: List of detected objects passing the confidence cutoff.
* `count: int`: Total number of detected objects.
* `image_width, image_height, image_shape`: Dimensions of the processed image.
* `inference_time_ms: float`: Raw model inference latency in milliseconds.
* `model_name: str`: Identifier or path of the model used.
* `device: str`: Execution device (`"cuda"` or `"cpu"`).
* `to_dict() -> Dict[str, Any]`: Complete structured dictionary.

---

## 6. Running CLI Tools

### Single Image Detection
```bash
python scripts/detect_objects.py data/samples/Colin_Powell/Colin_Powell_0001.jpg --threshold 0.25
```

### Kaggle GPU Execution & Performance Benchmarking
```bash
python scripts/kaggle_object_detection.py --image path/to/image.jpg --warmup 5 --iterations 20
```

---

## 7. Known Phase 1 Limitations

1. **Static Frame Analysis Only**: Phase 1 detects objects in individual frames; temporal persistence across video frames is deferred to Phase 4.
2. **Raw COCO Taxonomy**: Detects all 80 standard COCO classes without exam-specific proctoring relevance filtering (deferred to Phase 2).
3. **No Risk / Suspicion Scoring**: Conforming to project safety rules, the detector reports raw visual detections only without computing violation scores.
4. **Model Weight Management**: Model weights (`*.pt`) are git-ignored and downloaded directly during runtime in Kaggle.
