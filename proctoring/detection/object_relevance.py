"""Proctoring object relevance filtering, per-class thresholding, and structured detection reporting."""

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from proctoring.detection.object_detector import DetectedObject, ObjectDetectionResult

# Initial candidate classes relevant to an examination monitoring environment (COCO taxonomy)
DEFAULT_PROCTORING_RELEVANT_CLASSES: set[str] = {
    "person",
    "cell phone",
    "laptop",
    "book",
    "tablet",
    "remote",
    "keyboard",
    "mouse",
    "backpack",
    "handbag",
    "suitcase",
    "bottle",
}

# Initial default per-class confidence thresholds
DEFAULT_CLASS_THRESHOLDS: dict[str, float] = {
    "person": 0.25,
    "cell phone": 0.40,
    "laptop": 0.35,
    "book": 0.30,
    "tablet": 0.35,
    "remote": 0.35,
    "keyboard": 0.30,
    "mouse": 0.30,
    "backpack": 0.30,
    "handbag": 0.30,
    "suitcase": 0.35,
    "bottle": 0.30,
}


@dataclass
class ProctoringDetectionReport:
    """Structured proctoring detection report containing filtered relevant objects and person counts."""

    raw_result: ObjectDetectionResult
    relevant_objects: list[DetectedObject]
    ignored_objects: list[DetectedObject]
    person_count: int
    total_detections: int
    relevant_count: int
    image_width: int
    image_height: int
    inference_time_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a JSON-serializable dictionary."""
        return {
            "timestamp": round(self.timestamp, 3),
            "total_detections": self.total_detections,
            "relevant_count": self.relevant_count,
            "person_count": self.person_count,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "relevant_objects": [obj.to_dict() for obj in self.relevant_objects],
            "ignored_objects": [obj.to_dict() for obj in self.ignored_objects],
            "raw_result": self.raw_result.to_dict(),
        }


class ObjectRelevanceFilter:
    """Configurable relevance filter and structured reporter for examination environments."""

    def __init__(
        self,
        relevant_classes: set[str] | None = None,
        class_thresholds: dict[str, float] | None = None,
        default_threshold: float = 0.25,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.default_threshold = float(default_threshold)

        if relevant_classes is not None:
            self.relevant_classes = {c.lower().strip() for c in relevant_classes}
        else:
            self.relevant_classes = {c.lower() for c in DEFAULT_PROCTORING_RELEVANT_CLASSES}

        self.class_thresholds: dict[str, float] = {}
        # Populate defaults
        for k, v in DEFAULT_CLASS_THRESHOLDS.items():
            self.class_thresholds[k.lower()] = float(v)
        # Apply user overrides
        if class_thresholds is not None:
            for k, v in class_thresholds.items():
                self.class_thresholds[k.lower().strip()] = float(v)

    def get_threshold(self, class_name: str) -> float:
        """Get the effective confidence threshold for a given object class."""
        return self.class_thresholds.get(class_name.lower(), self.default_threshold)

    def is_relevant(self, class_name: str, confidence: float) -> bool:
        """Evaluate whether a detected object meets proctoring relevance and confidence criteria."""
        if not self.enabled:
            return True

        normalized_name = class_name.lower().strip()
        if normalized_name not in self.relevant_classes:
            return False

        effective_thresh = self.get_threshold(normalized_name)
        return confidence >= effective_thresh

    def filter(
        self,
        detection_result: ObjectDetectionResult,
        timestamp: float | None = None,
    ) -> ProctoringDetectionReport:
        """Filter raw detection result into relevant objects, count persons, and generate a report.

        Args:
            detection_result: Raw ObjectDetectionResult from detector.
            timestamp: Optional epoch timestamp.

        Returns:
            ProctoringDetectionReport preserving raw data and categorizing relevant/ignored objects.
        """
        relevant_objects: list[DetectedObject] = []
        ignored_objects: list[DetectedObject] = []

        for obj in detection_result.objects:
            if self.is_relevant(obj.class_name, obj.confidence):
                relevant_objects.append(obj)
            else:
                ignored_objects.append(obj)

        person_count = sum(1 for obj in relevant_objects if obj.class_name.lower() == "person")

        return ProctoringDetectionReport(
            raw_result=detection_result,
            relevant_objects=relevant_objects,
            ignored_objects=ignored_objects,
            person_count=person_count,
            total_detections=detection_result.count,
            relevant_count=len(relevant_objects),
            image_width=detection_result.image_width,
            image_height=detection_result.image_height,
            inference_time_ms=detection_result.inference_time_ms,
            timestamp=timestamp if timestamp is not None else time.time(),
        )

    def visualize_report(
        self,
        image: np.ndarray,
        report: ProctoringDetectionReport,
        show_all: bool = False,
        line_thickness: int = 2,
    ) -> np.ndarray:
        """Render detection visualization with proctoring relevance distinction.

        Args:
            image: Original BGR numpy image array.
            report: ProctoringDetectionReport.
            show_all: If True, draws ignored background objects in subdued gray for debugging.
            line_thickness: Line thickness in pixels.

        Returns:
            Annotated BGR numpy image.
        """
        vis_img = image.copy()

        # Proctoring palette for relevant objects (high-contrast, distinct colors)
        palette = {
            "person": (0, 255, 0),  # Green
            "cell phone": (0, 140, 255),  # Orange
            "laptop": (255, 128, 0),  # Amber/Blue
            "book": (255, 0, 180),  # Magenta/Pink
            "tablet": (0, 215, 255),  # Gold
            "keyboard": (200, 255, 0),  # Cyan/Lime
            "mouse": (200, 200, 0),  # Teal
            "backpack": (180, 105, 255),  # Purple
            "handbag": (180, 105, 255),
            "suitcase": (180, 105, 255),
            "bottle": (255, 255, 0),  # Yellow
            "remote": (0, 255, 255),  # Bright Yellow
        }

        # 1. Optionally draw ignored/background objects in subdued style (Debug mode)
        if show_all and report.ignored_objects:
            for obj in report.ignored_objects:
                x1, y1, x2, y2 = obj.bbox
                # Subdued gray box
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (120, 120, 120), 1, lineType=cv2.LINE_AA)
                label = f"[{obj.class_name} {int(obj.confidence * 100)}%]"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.40
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
                label_y1 = max(0, y1 - th - 4)
                cv2.rectangle(
                    vis_img,
                    (x1, label_y1),
                    (min(vis_img.shape[1], x1 + tw + 4), y1),
                    (80, 80, 80),
                    -1,
                )
                cv2.putText(
                    vis_img,
                    label,
                    (x1 + 2, y1 - 3),
                    font,
                    font_scale,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

        # 2. Draw relevant objects (Default & Debug mode)
        for obj in report.relevant_objects:
            color = palette.get(obj.class_name.lower(), (0, 255, 255))
            x1, y1, x2, y2 = obj.bbox

            # Draw prominent bounding box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, line_thickness)

            # Label text format: "class_name XX%"
            label = f"{obj.class_name} {int(obj.confidence * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thickness = 1

            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            label_y1 = max(0, y1 - th - 8)
            label_y2 = y1
            label_x2 = min(vis_img.shape[1], x1 + tw + 8)

            # Filled banner for text readability
            cv2.rectangle(vis_img, (x1, label_y1), (label_x2, label_y2), color, -1)
            # High-contrast text label
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

        # 3. Add top status bar showing Person Count & Relevant Object summary
        bar_h = 32
        bar = np.zeros((bar_h, vis_img.shape[1], 3), dtype=np.uint8)
        status_text = f"Persons: {report.person_count} | Relevant Objects: {report.relevant_count} | Total Detections: {report.total_detections}"
        cv2.putText(
            bar,
            status_text,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return np.vstack([bar, vis_img])
