"""Object detection module using Ultralytics YOLO with automatic GPU/CPU device resolution."""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import cv2


@dataclass
class DetectedObject:
    """Individual object detected within an image."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    def to_dict(self) -> Dict[str, Any]:
        """Convert detected object to dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class ObjectDetectionResult:
    """Structured result of an object detection inference operation."""
    objects: List[DetectedObject]
    count: int
    image_shape: Tuple[int, int, int]
    image_width: int
    image_height: int
    inference_time_ms: float
    model_name: str
    device: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection result to a JSON-serializable dictionary."""
        return {
            "count": self.count,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "model_name": self.model_name,
            "device": self.device,
            "objects": [obj.to_dict() for obj in self.objects],
        }


class ObjectDetector:
    """Modular object detector wrapping Ultralytics YOLO with CPU/GPU support."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        auto_load: bool = True,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.model_name = self.model_path.name if self.model_path else model_name
        self.confidence_threshold = float(confidence_threshold)
        self.device = self._resolve_device(device)
        self.model = None

        if auto_load:
            self.load_model()

    def _resolve_device(self, requested_device: Optional[str]) -> str:
        """Resolve execution device (explicit override or auto-detect CUDA/CPU)."""
        if requested_device is not None:
            return requested_device.lower()

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        return "cpu"

    def load_model(self) -> None:
        """Load the YOLO model from specified path or model identifier."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Ultralytics is not installed. Please install 'ultralytics' or run "
                "'pip install -r requirements-kaggle.txt' in your GPU environment."
            ) from e

        target = str(self.model_path) if self.model_path is not None else self.model_name
        self.model = YOLO(target)

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> ObjectDetectionResult:
        """Execute object detection on a BGR image array.

        Args:
            image: Input 3-channel BGR numpy image array.
            confidence_threshold: Optional override for detection confidence cutoff.

        Returns:
            ObjectDetectionResult containing detected objects, bounding boxes, and metadata.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Invalid input image: must be a non-empty numpy array.")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel BGR image, got shape: {image.shape}")

        if self.model is None:
            self.load_model()

        active_threshold = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else self.confidence_threshold
        )

        h, w = image.shape[:2]
        t0 = time.perf_counter()

        # Run inference
        results = self.model(
            image,
            conf=active_threshold,
            device=self.device,
            verbose=False,
        )
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        detected_objects: List[DetectedObject] = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            names = results[0].names

            if boxes is not None:
                xyxy_tensor = getattr(boxes, "xyxy", None)
                conf_tensor = getattr(boxes, "conf", None)
                cls_tensor = getattr(boxes, "cls", None)

                if xyxy_tensor is not None and conf_tensor is not None and cls_tensor is not None:
                    xyxy_arr = xyxy_tensor.cpu().numpy() if hasattr(xyxy_tensor, "cpu") else np.array(xyxy_tensor)
                    conf_arr = conf_tensor.cpu().numpy() if hasattr(conf_tensor, "cpu") else np.array(conf_tensor)
                    cls_arr = cls_tensor.cpu().numpy() if hasattr(cls_tensor, "cpu") else np.array(cls_tensor)

                    for xyxy, conf, cls_id in zip(xyxy_arr, conf_arr, cls_arr):
                        cid = int(cls_id)
                        cname = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                        # Clip to image dimensions
                        x1 = max(0, min(w, x1))
                        y1 = max(0, min(h, y1))
                        x2 = max(0, min(w, x2))
                        y2 = max(0, min(h, y2))

                        detected_objects.append(
                            DetectedObject(
                                class_id=cid,
                                class_name=cname,
                                confidence=float(conf),
                                bbox=(x1, y1, x2, y2),
                            )
                        )

        return ObjectDetectionResult(
            objects=detected_objects,
            count=len(detected_objects),
            image_shape=image.shape,
            image_width=w,
            image_height=h,
            inference_time_ms=inference_time_ms,
            model_name=self.model_name,
            device=self.device,
        )

    def visualize(
        self,
        image: np.ndarray,
        result: ObjectDetectionResult,
        line_thickness: int = 2,
    ) -> np.ndarray:
        """Render detection bounding boxes, class labels, and confidence percentages on the image."""
        vis_img = image.copy()

        # Color palette for visually appealing distinct class boxes
        palette = [
            (0, 255, 0),      # Green
            (255, 128, 0),    # Orange
            (0, 200, 255),    # Yellow
            (255, 0, 128),    # Pink/Magenta
            (0, 128, 255),    # Amber
            (200, 255, 0),    # Lime
            (255, 0, 0),      # Blue
            (0, 255, 255),    # Bright Yellow
        ]

        for obj in result.objects:
            color = palette[obj.class_id % len(palette)]
            x1, y1, x2, y2 = obj.bbox

            # Draw bounding box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, line_thickness)

            # Label text format: "class_name XX%"
            label = f"{obj.class_name} {int(obj.confidence * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            label_y1 = max(0, y1 - text_h - 8)
            label_y2 = y1
            label_x2 = min(vis_img.shape[1], x1 + text_w + 8)

            # Draw filled banner background for readable text
            cv2.rectangle(vis_img, (x1, label_y1), (label_x2, label_y2), color, -1)
            # Text label in contrasting dark color
            cv2.putText(
                vis_img,
                label,
                (x1 + 4, label_y2 - 4),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        return vis_img
