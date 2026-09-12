"""Face detection module using OpenCV YuNet with GPU acceleration via ONNX Runtime."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


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


class _OpenCVYuNetBackend:
    """CPU fallback backend using OpenCV cv2.FaceDetectorYN."""

    def __init__(
        self,
        model_path: Path,
        score_threshold: float,
        nms_threshold: float,
        top_k: int,
        backend_id: int = cv2.dnn.DNN_BACKEND_OPENCV,
        target_id: int = cv2.dnn.DNN_TARGET_CPU,
    ) -> None:
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.backend_id = backend_id
        self.target_id = target_id

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

    def detect(self, image: np.ndarray, score_threshold: float | None = None) -> np.ndarray | None:
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))

        if score_threshold is not None and score_threshold != self.score_threshold:
            self.detector.setScoreThreshold(score_threshold)

        _, raw_faces = self.detector.detect(image)

        if score_threshold is not None and score_threshold != self.score_threshold:
            self.detector.setScoreThreshold(self.score_threshold)

        return raw_faces


class _ORTYuNetBackend:
    """GPU-accelerated backend using ONNX Runtime with CUDAExecutionProvider."""

    OUT_NAMES = [
        "cls_8", "cls_16", "cls_32",
        "obj_8", "obj_16", "obj_32",
        "bbox_8", "bbox_16", "bbox_32",
        "kps_8", "kps_16", "kps_32",
    ]
    STRIDES = [8, 16, 32]
    INPUT_SIZE = 640

    def __init__(
        self,
        model_path: Path,
        score_threshold: float,
        nms_threshold: float,
        top_k: int,
        prefer_cuda: bool = True,
        device_id: int = 0,
    ) -> None:
        from proctoring.core.model_registry import ModelRegistry

        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.session = ModelRegistry.get_ort_session(
            model_path=model_path,
            prefer_cuda=prefer_cuda,
            device_id=device_id,
        )

    @property
    def is_cuda(self) -> bool:
        return bool(self.session.is_cuda)

    def detect(self, image: np.ndarray, score_threshold: float | None = None) -> np.ndarray | None:
        h, w = image.shape[:2]
        target_size = self.INPUT_SIZE
        scale = min(target_size / w, target_size / h)
        nw = int(round(w * scale))
        nh = int(round(h * scale))

        resized = cv2.resize(image, (nw, nh))
        pad_x = (target_size - nw) // 2
        pad_y = (target_size - nh) // 2
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

        blob = cv2.dnn.blobFromImage(canvas)
        outputs = self.session.run(self.OUT_NAMES, {"input": blob})

        eff_thresh = score_threshold if score_threshold is not None else self.score_threshold
        raw_faces: list[np.ndarray] = []

        for i, stride in enumerate(self.STRIDES):
            cols = target_size // stride
            rows = target_size // stride
            cls = outputs[i].reshape(rows, cols)
            obj = outputs[i + 3].reshape(rows, cols)
            bbox = outputs[i + 6].reshape(rows, cols, 4)
            kps = outputs[i + 9].reshape(rows, cols, 10)

            scores = np.sqrt(np.clip(cls, 0.0, 1.0) * np.clip(obj, 0.0, 1.0))
            mask = scores >= eff_thresh
            if not np.any(mask):
                continue

            r_idxs, c_idxs = np.where(mask)
            sc = scores[mask]
            bb = bbox[mask]
            kp = kps[mask]

            cx = (c_idxs + bb[:, 0]) * stride
            cy = (r_idxs + bb[:, 1]) * stride
            bw = np.exp(bb[:, 2]) * stride
            bh = np.exp(bb[:, 3]) * stride

            x1 = ((cx - bw * 0.5) - pad_x) / scale
            y1 = ((cy - bh * 0.5) - pad_y) / scale
            bw = bw / scale
            bh = bh / scale

            stride_raw = np.zeros((len(sc), 15), dtype=np.float32)
            stride_raw[:, 0] = x1
            stride_raw[:, 1] = y1
            stride_raw[:, 2] = bw
            stride_raw[:, 3] = bh
            for n in range(5):
                stride_raw[:, 4 + 2 * n] = (((kp[:, 2 * n] + c_idxs) * stride) - pad_x) / scale
                stride_raw[:, 4 + 2 * n + 1] = (
                    ((kp[:, 2 * n + 1] + r_idxs) * stride) - pad_y
                ) / scale
            stride_raw[:, 14] = sc
            raw_faces.append(stride_raw)

        if not raw_faces:
            return None

        all_candidates = np.vstack(raw_faces)
        boxes = [
            [int(c[0]), int(c[1]), int(c[2]), int(c[3])]
            for c in all_candidates
        ]
        scores_list = all_candidates[:, 14].tolist()

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=scores_list,
            score_threshold=eff_thresh,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
        )

        if len(indices) == 0:
            return None

        keep_indices = [int(idx) for idx in indices]
        return all_candidates[keep_indices]


class FaceDetector:
    """Wrapper for YuNet Face Detection with automatic CUDA / CPU backend selection."""

    def __init__(
        self,
        model_path: str | Path = "models/face_detection_yunet_2023mar.onnx",
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        backend_id: int = cv2.dnn.DNN_BACKEND_OPENCV,
        target_id: int = cv2.dnn.DNN_TARGET_CPU,
        device: str | None = None,
        prefer_cuda: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YuNet model file not found at: {self.model_path}")

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

        # Resolve execution device
        wants_cuda = False
        if device is not None:
            wants_cuda = device.lower() in ("cuda", "cuda:0", "gpu")
        elif prefer_cuda:
            try:
                import torch

                wants_cuda = torch.cuda.is_available()
            except ImportError:
                wants_cuda = False

        self._backend: _ORTYuNetBackend | _OpenCVYuNetBackend
        self.backend_name = "opencv_cpu"
        self.device = "cpu"

        if wants_cuda:
            try:
                ort_backend = _ORTYuNetBackend(
                    model_path=self.model_path,
                    score_threshold=self.score_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=self.top_k,
                    prefer_cuda=True,
                )
                if ort_backend.is_cuda:
                    self._backend = ort_backend
                    self.backend_name = "ort_cuda"
                    self.device = "cuda:0"
                    LOGGER.info("YuNet FaceDetector initialized with ORT CUDA backend.")
                else:
                    LOGGER.info(
                        "ORT CUDA not available for YuNet; falling back to OpenCV CPU backend."
                    )
                    self._backend = _OpenCVYuNetBackend(
                        model_path=self.model_path,
                        score_threshold=self.score_threshold,
                        nms_threshold=self.nms_threshold,
                        top_k=self.top_k,
                        backend_id=backend_id,
                        target_id=target_id,
                    )
            except Exception as exc:
                LOGGER.warning(
                    "Failed to initialize ORT CUDA backend for YuNet (%s); falling back to OpenCV CPU.",
                    exc,
                )
                self._backend = _OpenCVYuNetBackend(
                    model_path=self.model_path,
                    score_threshold=self.score_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=self.top_k,
                    backend_id=backend_id,
                    target_id=target_id,
                )
        else:
            self._backend = _OpenCVYuNetBackend(
                model_path=self.model_path,
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k,
                backend_id=backend_id,
                target_id=target_id,
            )

    @property
    def detector(self) -> Any:
        """Legacy compatibility property returning underlying detector or backend."""
        return getattr(self._backend, "detector", self._backend)

    @property
    def is_gpu_accelerated(self) -> bool:
        """Whether the face detector is currently executing on a GPU device."""
        return self.device.startswith("cuda")

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

        raw_faces = self._backend.detect(image, score_threshold=score_threshold)

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
