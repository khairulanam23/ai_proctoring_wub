"""Face recognition and verification module using OpenCV SFace."""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2

from src.face.detector import FaceDetector, FaceDetection


@dataclass
class VerificationResult:
    """Structured result of a face verification comparison."""
    success: bool
    same_person: bool
    similarity: Optional[float]
    threshold: float
    distance_metric: str  # "cosine" or "l2"
    ref_face_count: int
    test_face_count: int
    message: str
    timing_ms: Dict[str, float] = field(default_factory=dict)


class FaceVerifier:
    """Wrapper for OpenCV SFace Face Recognition and Verification."""

    # Default baseline thresholds from OpenCV Zoo SFace specifications
    DEFAULT_COSINE_THRESHOLD = 0.363
    DEFAULT_L2_THRESHOLD = 1.128

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        recognizer_model_path: Union[str, Path] = "models/face_recognition_sface_2021dec.onnx",
        default_metric: str = "cosine",
        default_threshold: Optional[float] = None,
        backend_id: int = cv2.dnn.DNN_BACKEND_OPENCV,
        target_id: int = cv2.dnn.DNN_TARGET_CPU,
    ) -> None:
        self.recognizer_model_path = Path(recognizer_model_path)
        if not self.recognizer_model_path.exists():
            raise FileNotFoundError(f"SFace model file not found at: {self.recognizer_model_path}")

        self.detector = detector if detector is not None else FaceDetector()
        self.default_metric = default_metric.lower()
        if self.default_metric not in ("cosine", "l2"):
            raise ValueError(f"Unsupported metric: {default_metric}. Use 'cosine' or 'l2'.")

        if default_threshold is not None:
            self.default_threshold = default_threshold
        else:
            self.default_threshold = (
                self.DEFAULT_COSINE_THRESHOLD
                if self.default_metric == "cosine"
                else self.DEFAULT_L2_THRESHOLD
            )

        self.recognizer = cv2.FaceRecognizerSF.create(
            model=str(self.recognizer_model_path),
            config="",
            backend_id=backend_id,
            target_id=target_id,
        )

    def extract_feature(
        self,
        image: np.ndarray,
        face: FaceDetection,
    ) -> np.ndarray:
        """Extract a 128-dimensional embedding from a detected face.

        Args:
            image: BGR image array.
            face: FaceDetection object containing the raw YuNet 15-element array.

        Returns:
            128-element 1D numpy array representing the face embedding.
        """
        aligned_face = self.recognizer.alignCrop(image, face.raw_detection)
        feature = self.recognizer.feature(aligned_face)
        return feature

    def compute_similarity(
        self,
        feature_a: np.ndarray,
        feature_b: np.ndarray,
        metric: Optional[str] = None,
    ) -> float:
        """Compute the similarity or distance between two feature vectors.

        Args:
            feature_a: 128-d feature array.
            feature_b: 128-d feature array.
            metric: "cosine" or "l2".

        Returns:
            Computed score (higher is more similar for cosine, lower is more similar for l2).
        """
        chosen_metric = (metric or self.default_metric).lower()
        if chosen_metric == "cosine":
            dis_type = getattr(
                cv2,
                "FaceRecognizerSF_FR_COSINE",
                getattr(cv2, "FACE_RECOGNIZER_SF_FR_COSINE", getattr(cv2.FaceRecognizerSF, "FR_COSINE", 0)),
            )
        elif chosen_metric == "l2":
            dis_type = getattr(
                cv2,
                "FaceRecognizerSF_FR_NORM_L2",
                getattr(cv2, "FACE_RECOGNIZER_SF_FR_NORM_L2", getattr(cv2.FaceRecognizerSF, "FR_NORM_L2", 1)),
            )
        else:
            raise ValueError(f"Unsupported metric: {chosen_metric}. Use 'cosine' or 'l2'.")

        score = self.recognizer.match(feature_a, feature_b, dis_type)
        return float(score)

    def verify(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        threshold: Optional[float] = None,
        metric: Optional[str] = None,
    ) -> VerificationResult:
        """Verify whether two images belong to the same person.

        Args:
            image_a: Reference BGR image array.
            image_b: Test BGR image array.
            threshold: Optional threshold override.
            metric: "cosine" or "l2".

        Returns:
            VerificationResult with match decision, scores, face counts, and latencies.
        """
        chosen_metric = (metric or self.default_metric).lower()
        if threshold is not None:
            active_threshold = threshold
        else:
            active_threshold = (
                self.DEFAULT_COSINE_THRESHOLD
                if chosen_metric == "cosine"
                else self.DEFAULT_L2_THRESHOLD
            )

        timing: Dict[str, float] = {}

        # 1. Detect faces in Image A
        t0 = time.perf_counter()
        det_a = self.detector.detect(image_a)
        timing["detect_image_a_ms"] = (time.perf_counter() - t0) * 1000.0

        # 2. Detect faces in Image B
        t0 = time.perf_counter()
        det_b = self.detector.detect(image_b)
        timing["detect_image_b_ms"] = (time.perf_counter() - t0) * 1000.0

        if det_a.count == 0 or det_b.count == 0:
            msg = "Verification rejected: "
            if det_a.count == 0 and det_b.count == 0:
                msg += "No face detected in either reference or test image."
            elif det_a.count == 0:
                msg += "No face detected in reference image."
            else:
                msg += "No face detected in test image."

            return VerificationResult(
                success=False,
                same_person=False,
                similarity=None,
                threshold=active_threshold,
                distance_metric=chosen_metric,
                ref_face_count=det_a.count,
                test_face_count=det_b.count,
                message=msg,
                timing_ms=timing,
            )

        # Select primary face (highest confidence)
        primary_face_a = max(det_a.faces, key=lambda f: f.confidence)
        primary_face_b = max(det_b.faces, key=lambda f: f.confidence)

        # 3. Feature extraction
        t0 = time.perf_counter()
        feat_a = self.extract_feature(image_a, primary_face_a)
        timing["extract_feature_a_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        feat_b = self.extract_feature(image_b, primary_face_b)
        timing["extract_feature_b_ms"] = (time.perf_counter() - t0) * 1000.0

        # 4. Feature matching
        t0 = time.perf_counter()
        similarity = self.compute_similarity(feat_a, feat_b, metric=chosen_metric)
        timing["match_ms"] = (time.perf_counter() - t0) * 1000.0

        timing["total_inference_ms"] = sum(timing.values())

        # 5. Decision logic
        if chosen_metric == "cosine":
            same_person = similarity >= active_threshold
        else:  # L2 distance: lower distance = higher match
            same_person = similarity <= active_threshold

        decision_str = "SAME PERSON" if same_person else "DIFFERENT PERSON"
        message = (
            f"Verification successful: {decision_str} "
            f"(Score: {similarity:.4f}, Threshold: {active_threshold:.4f}, Metric: {chosen_metric})"
        )

        return VerificationResult(
            success=True,
            same_person=same_person,
            similarity=similarity,
            threshold=active_threshold,
            distance_metric=chosen_metric,
            ref_face_count=det_a.count,
            test_face_count=det_b.count,
            message=message,
            timing_ms=timing,
        )
