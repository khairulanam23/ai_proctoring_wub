"""Face presence and multi-face analysis module."""

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from proctoring.detection.face_detector import FaceDetection, FaceDetector


class FacePresenceStatus(str, Enum):
    """Enumeration of face presence states."""

    NO_FACE = "NO_FACE"
    SINGLE_FACE = "SINGLE_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"


@dataclass
class FacePresenceResult:
    """Structured result of face presence analysis on a single frame."""

    status: FacePresenceStatus
    face_count: int
    faces: list[FaceDetection]
    image_shape: tuple[int, int, int]  # (H, W, C)
    inference_time_ms: float
    frame_index: int | None = None
    timestamp_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        return {
            "status": self.status.value,
            "face_count": self.face_count,
            "image_shape": list(self.image_shape),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 3)
            if self.timestamp_sec is not None
            else None,
            "faces": [
                {
                    "bbox": list(face.bbox),
                    "confidence": round(face.confidence, 4),
                    "landmarks": [list(lm) for lm in face.landmarks],
                }
                for face in self.faces
            ],
        }


class FacePresenceAnalyzer:
    """Analyzer for evaluating face presence status on images/frames."""

    def __init__(
        self,
        detector: FaceDetector | None = None,
        model_path: str | Path = "models/face_detection_yunet_2023mar.onnx",
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        backend_id: int = cv2.dnn.DNN_BACKEND_OPENCV,
        target_id: int = cv2.dnn.DNN_TARGET_CPU,
    ) -> None:
        if detector is not None:
            self.detector = detector
        else:
            self.detector = FaceDetector(
                model_path=model_path,
                score_threshold=score_threshold,
                nms_threshold=nms_threshold,
                backend_id=backend_id,
                target_id=target_id,
            )

    def analyze(
        self,
        image: np.ndarray,
        frame_index: int | None = None,
        timestamp_sec: float | None = None,
        score_threshold: float | None = None,
    ) -> FacePresenceResult:
        """Analyze face presence in a single frame.

        Args:
            image: BGR image array (H, W, 3).
            frame_index: Optional sequential frame number.
            timestamp_sec: Optional timestamp in seconds.
            score_threshold: Optional confidence threshold override.

        Returns:
            FacePresenceResult containing status, face details, and timing.
        """
        t0 = time.perf_counter()
        det_result = self.detector.detect(image, score_threshold=score_threshold)
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        count = det_result.count
        if count == 0:
            status = FacePresenceStatus.NO_FACE
        elif count == 1:
            status = FacePresenceStatus.SINGLE_FACE
        else:
            status = FacePresenceStatus.MULTIPLE_FACES

        return FacePresenceResult(
            status=status,
            face_count=count,
            faces=det_result.faces,
            image_shape=det_result.image_shape,
            inference_time_ms=inference_time_ms,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
        )
