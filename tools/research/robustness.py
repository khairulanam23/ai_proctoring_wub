"""Robustness evaluation framework, visual condition generators, threshold benchmarking, and comparison visualizer."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import cv2
import numpy as np

from proctoring.detection.object_detector import DetectedObject, ObjectDetectionResult, ObjectDetector
from proctoring.detection.object_relevance import ObjectRelevanceFilter, ProctoringDetectionReport


class VisualCondition:
    """Standardized visual condition identifiers."""
    ORIGINAL = "original"
    LOW_LIGHT = "low_light"
    HIGH_LIGHT = "high_light"
    LOW_CONTRAST = "low_contrast"
    GAUSSIAN_BLUR = "gaussian_blur"
    MOTION_BLUR = "motion_blur"
    PARTIAL_OCCLUSION = "partial_occlusion"
    LOW_RESOLUTION = "low_resolution"
    JPEG_COMPRESSION = "jpeg_compression"
    PERSPECTIVE_TILT = "perspective_tilt"


class ImageAugmenter:
    """Controlled visual transformation engine for proctoring stress testing."""

    @staticmethod
    def apply_condition(
        image: np.ndarray,
        condition: str,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """Apply a controlled synthetic visual transformation to a BGR image array.

        Args:
            image: Source BGR image numpy array.
            condition: VisualCondition identifier string.
            intensity: Scaling factor for transformation effect (default 1.0).

        Returns:
            Transformed BGR image numpy array.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image must be a non-empty numpy array.")

        cond = condition.lower().strip()
        h, w = image.shape[:2]

        if cond == VisualCondition.ORIGINAL:
            return image.copy()

        elif cond == VisualCondition.LOW_LIGHT:
            # Shift brightness downward (dark room / dim lighting)
            shift = int(-55 * intensity)
            return np.clip(image.astype(np.int16) + shift, 0, 255).astype(np.uint8)

        elif cond == VisualCondition.HIGH_LIGHT:
            # Shift brightness upward (window glare / direct backlight)
            shift = int(50 * intensity)
            return np.clip(image.astype(np.int16) + shift, 0, 255).astype(np.uint8)

        elif cond == VisualCondition.LOW_CONTRAST:
            # Scale dynamic range toward mean gray
            factor = max(0.1, 1.0 - (0.5 * intensity))
            mean_val = 128.0
            return np.clip((image.astype(np.float32) - mean_val) * factor + mean_val, 0, 255).astype(np.uint8)

        elif cond == VisualCondition.GAUSSIAN_BLUR:
            # Out-of-focus webcam blur
            ksize = int(11 * intensity) | 1  # ensure odd integer
            return cv2.GaussianBlur(image, (ksize, ksize), 0)

        elif cond == VisualCondition.MOTION_BLUR:
            # Candidate motion / head movement blur
            ksize = max(3, int(15 * intensity))
            kernel = np.zeros((ksize, ksize), dtype=np.float32)
            kernel[int((ksize - 1) / 2), :] = np.ones(ksize, dtype=np.float32) / ksize
            return cv2.filter2D(image, -1, kernel)

        elif cond == VisualCondition.PARTIAL_OCCLUSION:
            # Occlusion block simulating hands / objects partially covering candidate or device
            occ = image.copy()
            occ_w = int(w * 0.25 * intensity)
            occ_h = int(h * 0.25 * intensity)
            cx, cy = w // 2, h // 2
            x1 = max(0, cx - occ_w // 2)
            y1 = max(0, cy - occ_h // 2)
            x2 = min(w, x1 + occ_w)
            y2 = min(h, y1 + occ_h)
            occ[y1:y2, x1:x2] = (15, 15, 15)  # Dark occlusion patch
            return occ

        elif cond == VisualCondition.LOW_RESOLUTION:
            # Heavy downsampling simulating poor network bandwidth / 240p webcam
            scale = max(0.1, 1.0 / (4.0 * intensity))
            down = cv2.resize(image, (max(16, int(w * scale)), max(16, int(h * scale))), interpolation=cv2.INTER_AREA)
            return cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)

        elif cond == VisualCondition.JPEG_COMPRESSION:
            # Compression artifacts
            quality = max(5, int(100 - (75 * intensity)))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, enc = cv2.imencode(".jpg", image, encode_param)
            return cv2.imdecode(enc, cv2.IMREAD_COLOR)

        elif cond == VisualCondition.PERSPECTIVE_TILT:
            # Slanted camera angle
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            offset = int(w * 0.12 * intensity)
            pts2 = np.float32([[offset, 0], [w - offset, 0], [0, h], [w, h]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

        else:
            raise ValueError(f"Unknown visual condition: '{condition}'")

    @classmethod
    def generate_condition_suite(cls, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate a dictionary of all standard visual condition variants for an input image."""
        conditions = [
            VisualCondition.ORIGINAL,
            VisualCondition.LOW_LIGHT,
            VisualCondition.HIGH_LIGHT,
            VisualCondition.GAUSSIAN_BLUR,
            VisualCondition.MOTION_BLUR,
            VisualCondition.PARTIAL_OCCLUSION,
            VisualCondition.LOW_RESOLUTION,
            VisualCondition.JPEG_COMPRESSION,
        ]
        return {c: cls.apply_condition(image, c) for c in conditions}


@dataclass
class ConfusionMetrics:
    """Binary/multiclass detection confusion matrix and precision/recall metrics."""
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def evaluate_detections(
    predicted_classes: List[str],
    expected_classes: List[str],
) -> ConfusionMetrics:
    """Evaluate multiset class predictions against expected ground truth labels.

    Args:
        predicted_classes: List of detected class names.
        expected_classes: List of expected ground truth class names.

    Returns:
        ConfusionMetrics containing TP, FP, FN, precision, recall, and F1 score.
    """
    pred_counts: Dict[str, int] = {}
    for c in predicted_classes:
        name = c.lower().strip()
        pred_counts[name] = pred_counts.get(name, 0) + 1

    exp_counts: Dict[str, int] = {}
    for c in expected_classes:
        name = c.lower().strip()
        exp_counts[name] = exp_counts.get(name, 0) + 1

    tp = 0
    all_classes = set(pred_counts.keys()).union(set(exp_counts.keys()))
    for c in all_classes:
        p_c = pred_counts.get(c, 0)
        e_c = exp_counts.get(c, 0)
        tp += min(p_c, e_c)

    total_pred = len(predicted_classes)
    total_exp = len(expected_classes)
    fp = max(0, total_pred - tp)
    fn = max(0, total_exp - tp)

    precision = (tp / total_pred) if total_pred > 0 else (1.0 if total_exp == 0 else 0.0)
    recall = (tp / total_exp) if total_exp > 0 else (1.0 if total_pred == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return ConfusionMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


@dataclass
class ThresholdEvaluationResult:
    """Results of running detection at a specific confidence cutoff threshold."""
    threshold: float
    total_detections: int
    class_counts: Dict[str, int]
    average_confidence: float
    metrics: Optional[ConfusionMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": round(self.threshold, 2),
            "total_detections": self.total_detections,
            "class_counts": self.class_counts,
            "average_confidence": round(self.average_confidence, 4),
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


class ThresholdEvaluator:
    """Sweep evaluator analyzing detection performance across multiple confidence thresholds."""

    DEFAULT_THRESHOLDS: List[float] = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]

    @classmethod
    def evaluate_threshold_sweep(
        cls,
        detector: ObjectDetector,
        images_with_ground_truth: List[Tuple[np.ndarray, Optional[List[str]]]],
        thresholds: Optional[List[float]] = None,
    ) -> List[ThresholdEvaluationResult]:
        """Run detection sweep over multiple confidence thresholds.

        Args:
            detector: ObjectDetector instance.
            images_with_ground_truth: List of (image_array, optional_expected_classes_list).
            thresholds: List of float confidence thresholds to evaluate.

        Returns:
            List of ThresholdEvaluationResult items.
        """
        eval_thresholds = thresholds if thresholds is not None else cls.DEFAULT_THRESHOLDS
        results: List[ThresholdEvaluationResult] = []

        for thresh in eval_thresholds:
            all_preds: List[str] = []
            all_exps: List[str] = []
            all_confs: List[float] = []
            class_counts: Dict[str, int] = {}
            has_gt = False

            for img, exp in images_with_ground_truth:
                det_res = detector.detect(img, confidence_threshold=thresh)
                preds = [o.class_name for o in det_res.objects]
                all_preds.extend(preds)

                for o in det_res.objects:
                    all_confs.append(o.confidence)
                    class_counts[o.class_name] = class_counts.get(o.class_name, 0) + 1

                if exp is not None:
                    has_gt = True
                    all_exps.extend(exp)

            avg_conf = float(np.mean(all_confs)) if all_confs else 0.0
            metrics = evaluate_detections(all_preds, all_exps) if has_gt else None

            results.append(
                ThresholdEvaluationResult(
                    threshold=thresh,
                    total_detections=len(all_preds),
                    class_counts=class_counts,
                    average_confidence=avg_conf,
                    metrics=metrics,
                )
            )

        return results


def generate_visual_comparison_grid(
    condition_images: Dict[str, np.ndarray],
    condition_results: Dict[str, ObjectDetectionResult],
    output_path: Union[str, Path],
    cols: int = 3,
) -> Path:
    """Generate a high-resolution multi-panel visual comparison sheet.

    Args:
        condition_images: Dictionary mapping condition name to source BGR image.
        condition_results: Dictionary mapping condition name to ObjectDetectionResult.
        output_path: Destination file path (e.g. .png or .jpg).
        cols: Number of columns in grid layout.

    Returns:
        Path to written comparison grid image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rendered_panels: List[np.ndarray] = []
    target_w, target_h = 360, 270

    palette = {
        "person": (46, 204, 113),
        "cell phone": (230, 126, 34),
        "laptop": (52, 152, 219),
        "book": (155, 89, 182),
        "tablet": (26, 188, 156),
    }

    for cond_name, img in condition_images.items():
        res = condition_results.get(cond_name)
        panel = cv2.resize(img.copy(), (target_w, target_h))
        orig_h, orig_w = img.shape[:2]
        scale_x = target_w / float(orig_w)
        scale_y = target_h / float(orig_h)

        # Draw detections
        if res is not None:
            for obj in res.objects:
                c = palette.get(obj.class_name.lower(), (0, 215, 255))
                x1 = int(obj.x1 * scale_x)
                y1 = int(obj.y1 * scale_y)
                x2 = int(obj.x2 * scale_x)
                y2 = int(obj.y2 * scale_y)

                cv2.rectangle(panel, (x1, y1), (x2, y2), c, 2)
                label = f"{obj.class_name} {int(obj.confidence * 100)}%"
                font_scale = 0.42
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                by1 = max(0, y1 - th - 4)
                cv2.rectangle(panel, (x1, by1), (min(target_w, x1 + tw + 4), y1), c, -1)
                cv2.putText(panel, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw Condition Header Title Banner
        det_count = res.count if res else 0
        banner = np.zeros((28, target_w, 3), dtype=np.uint8)
        title = f"{cond_name.replace('_', ' ').upper()} ({det_count} det)"
        cv2.putText(banner, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        full_panel = np.vstack([banner, panel])
        rendered_panels.append(full_panel)

    # Pad to complete grid rows
    num_panels = len(rendered_panels)
    rows = int(np.ceil(num_panels / cols))
    blank_panel = np.zeros_like(rendered_panels[0])
    while len(rendered_panels) < rows * cols:
        rendered_panels.append(blank_panel)

    row_strips = []
    for r in range(rows):
        row_panels = rendered_panels[r * cols : (r + 1) * cols]
        row_strips.append(np.hstack(row_panels))

    grid = np.vstack(row_strips)
    cv2.imwrite(str(output_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return output_path
