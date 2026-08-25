"""Face detection module using OpenCV YuNet."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FaceDetection:
    """Structured representation of a single detected face."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: list[
        tuple[float, float]
    ]  # 5 landmarks: right eye, left eye, nose tip, right mouth, left mouth
    raw_detection: np.ndarray  # Raw 15-element array from YuNet [x, y, w, h, x1, y1, ..., score]


@dataclass
class DetectionResult:
    """Result of running face detection on an image."""

    faces: list[FaceDetection]
    count: int
    image_shape: tuple[int, int, int]  # (H, W, C)


class FaceDetector:
    """Wrapper for OpenCV YuNet Face Detection."""

    def __init__(
        self,
        model_path: str | Path = "models/face_detection_yunet_2023mar.onnx",
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        backend_id: int = cv2.dnn.DNN_BACKEND_OPENCV,
        target_id: int = cv2.dnn.DNN_TARGET_CPU,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YuNet model file not found at: {self.model_path}")

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.backend_id = backend_id
        self.target_id = target_id

        # Initial dummy input size, dynamically updated on detect()
        self.detector = cv2.FaceDetectorYN.create(
            model=str(self.model_path),
            config="",
            input_size=(320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
            backend_id=self.backend_id,
            target_id=self.target_id,
        )

    def detect(
        self,
        image: np.ndarray,
        score_threshold: float | None = None,
    ) -> DetectionResult:
        """Detect faces in a BGR image.

        Args:
            image: BGR image array (H, W, 3).
            score_threshold: Optional override for detection confidence threshold.

        Returns:
            DetectionResult containing list of detected faces and metadata.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Invalid input image: must be a non-empty numpy array.")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel BGR image, got shape: {image.shape}")

        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))

        if score_threshold is not None and score_threshold != self.score_threshold:
            self.detector.setScoreThreshold(score_threshold)

        _, raw_faces = self.detector.detect(image)

        # Restore default threshold if overridden
        if score_threshold is not None and score_threshold != self.score_threshold:
            self.detector.setScoreThreshold(self.score_threshold)

        detections: list[FaceDetection] = []
        if raw_faces is not None and len(raw_faces) > 0:
            for raw in raw_faces:
                bbox = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
                landmarks = [
                    (float(raw[4]), float(raw[5])),  # Right eye
                    (float(raw[6]), float(raw[7])),  # Left eye
                    (float(raw[8]), float(raw[9])),  # Nose tip
                    (float(raw[10]), float(raw[11])),  # Right mouth corner
                    (float(raw[12]), float(raw[13])),  # Left mouth corner
                ]
                confidence = float(raw[14])
                detections.append(
                    FaceDetection(
                        bbox=bbox,
                        confidence=confidence,
                        landmarks=landmarks,
                        raw_detection=raw,
                    )
                )

        return DetectionResult(
            faces=detections,
            count=len(detections),
            image_shape=image.shape,
        )
