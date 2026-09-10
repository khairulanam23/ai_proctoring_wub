from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from proctoring.detection.face_detector import FaceDetection, FaceDetector

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from proctoring.preprocessing.face_preprocessing import PreprocessingResult


@dataclass
class VerificationResult:
    """Structured result of a single-pair face verification comparison."""

    success: bool
    same_person: bool
    similarity: float | None
    threshold: float
    distance_metric: str  # "cosine" or "l2"
    ref_face_count: int
    test_face_count: int
    message: str
    timing_ms: dict[str, float] = field(default_factory=dict)
    ref_preprocessing: PreprocessingResult | None = None
    test_preprocessing: PreprocessingResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        return {
            "success": self.success,
            "same_person": self.same_person,
            "similarity": round(self.similarity, 4) if self.similarity is not None else None,
            "threshold": round(self.threshold, 4),
            "distance_metric": self.distance_metric,
            "ref_face_count": self.ref_face_count,
            "test_face_count": self.test_face_count,
            "message": self.message,
            "timing_ms": {k: round(v, 2) for k, v in self.timing_ms.items()},
            "ref_preprocessing": self.ref_preprocessing.to_dict()
            if self.ref_preprocessing
            else None,
            "test_preprocessing": self.test_preprocessing.to_dict()
            if self.test_preprocessing
            else None,
        }


@dataclass
class MultiReferenceVerificationResult:
    """Structured result of multi-reference template verification."""

    success: bool
    same_person: bool
    similarity: float | None  # Robust combined score
    template_similarity: float | None  # Cosine similarity to mean identity template
    max_similarity: float | None  # Highest similarity across individual references
    mean_similarity: float | None  # Average similarity across individual references
    threshold: float
    distance_metric: str
    valid_reference_count: int
    total_reference_count: int
    test_face_count: int
    individual_scores: list[float] = field(default_factory=list)
    message: str = ""
    timing_ms: dict[str, float] = field(default_factory=dict)
    test_preprocessing: PreprocessingResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "same_person": self.same_person,
            "similarity": round(self.similarity, 4) if self.similarity is not None else None,
            "template_similarity": round(self.template_similarity, 4)
            if self.template_similarity is not None
            else None,
            "max_similarity": round(self.max_similarity, 4)
            if self.max_similarity is not None
            else None,
            "mean_similarity": round(self.mean_similarity, 4)
            if self.mean_similarity is not None
            else None,
            "threshold": round(self.threshold, 4),
            "distance_metric": self.distance_metric,
            "valid_reference_count": self.valid_reference_count,
            "total_reference_count": self.total_reference_count,
            "individual_scores": [round(s, 4) for s in self.individual_scores],
            "message": self.message,
            "timing_ms": {k: round(v, 2) for k, v in self.timing_ms.items()},
        }


class FaceVerifier:
    """Wrapper for OpenCV SFace Face Recognition and Verification with robust preprocessing and multi-image enrollment."""

    # Default calibrated operational threshold
    DEFAULT_COSINE_THRESHOLD = 0.3630
    DEFAULT_L2_THRESHOLD = 1.1280

    def __init__(
        self,
        detector: FaceDetector | None = None,
        preprocessor: Any | None = None,
        recognizer_model_path: str | Path = "models/face_recognition_sface_2021dec.onnx",
        default_metric: str = "cosine",
        default_threshold: float | None = None,
        backend_id: int = cv2.dnn.DNN_BACKEND_OPENCV,
        target_id: int = cv2.dnn.DNN_TARGET_CPU,
        device: str | None = None,
    ) -> None:
        self.recognizer_model_path = Path(recognizer_model_path)
        if not self.recognizer_model_path.exists():
            raise FileNotFoundError(f"SFace model file not found at: {self.recognizer_model_path}")

        self.device = "cpu"
        if device is not None and device.lower() in ("cuda", "cuda:0", "gpu"):
            has_cuda = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
            if has_cuda:
                backend_id = getattr(cv2.dnn, "DNN_BACKEND_CUDA", backend_id)
                target_id = getattr(cv2.dnn, "DNN_TARGET_CUDA", target_id)
                self.device = "cuda"
            else:
                LOGGER.debug(
                    "SFace FaceVerifier: OpenCV build lacks CUDA DNN support; running on CPU (MLAS SGEMM)."
                )
                self.device = "cpu"

        self.backend_id = backend_id
        self.target_id = target_id

        self.detector = detector if detector is not None else FaceDetector(device=device)
        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            from proctoring.preprocessing.face_preprocessing import FacePreprocessor

            self.preprocessor = FacePreprocessor(detector=self.detector)

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
            backend_id=self.backend_id,
            target_id=self.target_id,
        )

    @property
    def is_gpu_accelerated(self) -> bool:
        """Whether the face verifier is currently executing on a GPU device."""
        return self.device.startswith("cuda")

    def extract_feature(
        self,
        image: np.ndarray,
        face: FaceDetection | None = None,
        normalize_l2: bool = True,
    ) -> np.ndarray:
        """Extract a 128-dimensional embedding from a face or pre-aligned image.

        Args:
            image: BGR image array (can be original image with face detection or 112x112 aligned face).
            face: Optional FaceDetection object. If provided, alignCrop is performed.
            normalize_l2: If True, normalizes the embedding vector to unit L2 norm.

        Returns:
            128-element 1D numpy array representing the face embedding.
        """
        if face is not None:
            aligned_face = self.recognizer.alignCrop(image, face.raw_detection)
        else:
            if image.shape[:2] != (112, 112):
                aligned_face = cv2.resize(image, (112, 112))
            else:
                aligned_face = image

        feature = self.recognizer.feature(aligned_face)
        if normalize_l2:
            norm = np.linalg.norm(feature)
            if norm > 1e-12:
                feature = feature / norm
        return feature

    def compute_similarity(
        self,
        feature_a: np.ndarray,
        feature_b: np.ndarray,
        metric: str | None = None,
    ) -> float:
        """Compute the similarity or distance between two feature vectors with strict mathematical L2 handling.

        Args:
            feature_a: 128-d feature array.
            feature_b: 128-d feature array.
            metric: "cosine" or "l2".

        Returns:
            Computed score (higher is more similar for cosine, lower is more similar for l2).
        """
        chosen_metric = (metric or self.default_metric).lower()

        # Ensure 2D (1, 128) shape
        fa = feature_a.reshape(1, -1)
        fb = feature_b.reshape(1, -1)

        # L2-normalize
        norm_a = np.linalg.norm(fa)
        norm_b = np.linalg.norm(fb)

        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0 if chosen_metric == "cosine" else float("inf")

        fa_norm = fa / norm_a
        fb_norm = fb / norm_b

        if chosen_metric == "cosine":
            # Exact inner product of unit vectors
            score = float(np.dot(fa_norm, fb_norm.T)[0, 0])
            return float(np.clip(score, -1.0, 1.0))
        if chosen_metric == "l2":
            # Euclidean distance between unit vectors
            return float(np.linalg.norm(fa_norm - fb_norm))
        raise ValueError(f"Unsupported metric: {chosen_metric}. Use 'cosine' or 'l2'.")

    def verify(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        threshold: float | None = None,
        metric: str | None = None,
        use_preprocessing: bool = True,
    ) -> VerificationResult:
        """Verify whether two images belong to the same person using robust preprocessing and alignment."""
        chosen_metric = (metric or self.default_metric).lower()
        active_threshold = threshold if threshold is not None else self.default_threshold

        timing: dict[str, float] = {}

        if use_preprocessing:
            # 1. Preprocess Reference Image A
            t0 = time.perf_counter()
            prep_a = self.preprocessor.preprocess(image_a)
            timing["preprocess_image_a_ms"] = (time.perf_counter() - t0) * 1000.0

            # 2. Preprocess Test Image B
            t0 = time.perf_counter()
            prep_b = self.preprocessor.preprocess(image_b)
            timing["preprocess_image_b_ms"] = (time.perf_counter() - t0) * 1000.0

            if not prep_a.success or not prep_b.success:
                rejection_msg = "Verification rejected: "
                if not prep_a.success and not prep_b.success:
                    rejection_msg += f"Reference ({prep_a.message}) & Test ({prep_b.message})"
                elif not prep_a.success:
                    rejection_msg += f"Reference image: {prep_a.message}"
                else:
                    rejection_msg += f"Test image: {prep_b.message}"

                return VerificationResult(
                    success=False,
                    same_person=False,
                    similarity=None,
                    threshold=active_threshold,
                    distance_metric=chosen_metric,
                    ref_face_count=prep_a.face_count,
                    test_face_count=prep_b.face_count,
                    message=rejection_msg,
                    timing_ms=timing,
                    ref_preprocessing=prep_a,
                    test_preprocessing=prep_b,
                )

            # 3. Extract Features from Normalized/Aligned Crops
            t0 = time.perf_counter()
            feat_a = self.extract_feature(prep_a.normalized_face, normalize_l2=True)
            timing["extract_feature_a_ms"] = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            feat_b = self.extract_feature(prep_b.normalized_face, normalize_l2=True)
            timing["extract_feature_b_ms"] = (time.perf_counter() - t0) * 1000.0

            ref_face_count = prep_a.face_count
            test_face_count = prep_b.face_count

        else:
            # Fallback legacy mode
            t0 = time.perf_counter()
            det_a = self.detector.detect(image_a)
            timing["detect_image_a_ms"] = (time.perf_counter() - t0) * 1000.0

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

            primary_face_a = max(det_a.faces, key=lambda f: f.confidence)
            primary_face_b = max(det_b.faces, key=lambda f: f.confidence)

            t0 = time.perf_counter()
            feat_a = self.extract_feature(image_a, primary_face_a, normalize_l2=True)
            timing["extract_feature_a_ms"] = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            feat_b = self.extract_feature(image_b, primary_face_b, normalize_l2=True)
            timing["extract_feature_b_ms"] = (time.perf_counter() - t0) * 1000.0

            prep_a = None
            prep_b = None
            ref_face_count = det_a.count
            test_face_count = det_b.count

        # 4. Feature Matching
        t0 = time.perf_counter()
        similarity = self.compute_similarity(feat_a, feat_b, metric=chosen_metric)
        timing["match_ms"] = (time.perf_counter() - t0) * 1000.0

        timing["total_inference_ms"] = sum(timing.values())

        # 5. Decision logic
        if chosen_metric == "cosine":
            same_person = similarity >= active_threshold
        else:  # L2 distance
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
            ref_face_count=ref_face_count,
            test_face_count=test_face_count,
            message=message,
            timing_ms=timing,
            ref_preprocessing=prep_a,
            test_preprocessing=prep_b,
        )

    def verify_multi_reference(
        self,
        reference_images: list[np.ndarray],
        test_image: np.ndarray,
        threshold: float | None = None,
        metric: str | None = None,
    ) -> MultiReferenceVerificationResult:
        """Verify test image against a multi-image enrolled identity template.

        Args:
            reference_images: List of reference BGR images for the enrolled identity.
            test_image: Test BGR image to verify.
            threshold: Optional threshold override.
            metric: "cosine" or "l2".

        Returns:
            MultiReferenceVerificationResult containing template, max, and individual scores.
        """
        chosen_metric = (metric or self.default_metric).lower()
        active_threshold = threshold if threshold is not None else self.default_threshold

        timing: dict[str, float] = {}

        if not reference_images:
            return MultiReferenceVerificationResult(
                success=False,
                same_person=False,
                similarity=None,
                template_similarity=None,
                max_similarity=None,
                mean_similarity=None,
                threshold=active_threshold,
                distance_metric=chosen_metric,
                valid_reference_count=0,
                total_reference_count=0,
                test_face_count=0,
                message="Multi-reference verification rejected: No reference images provided.",
            )

        # 1. Preprocess & Extract Features for all Reference Images
        t0 = time.perf_counter()
        valid_ref_embeddings = []
        for img in reference_images:
            prep = self.preprocessor.preprocess(img)
            if prep.success and prep.normalized_face is not None:
                feat = self.extract_feature(prep.normalized_face, normalize_l2=True)
                valid_ref_embeddings.append(feat)
        timing["preprocess_all_refs_ms"] = (time.perf_counter() - t0) * 1000.0

        if not valid_ref_embeddings:
            return MultiReferenceVerificationResult(
                success=False,
                same_person=False,
                similarity=None,
                template_similarity=None,
                max_similarity=None,
                mean_similarity=None,
                threshold=active_threshold,
                distance_metric=chosen_metric,
                valid_reference_count=0,
                total_reference_count=len(reference_images),
                test_face_count=0,
                message="Multi-reference verification rejected: All reference images failed quality or single-face checks.",
                timing_ms=timing,
            )

        # 2. Construct Normalized Identity Template Vector
        t0 = time.perf_counter()
        template_raw = np.mean(valid_ref_embeddings, axis=0)
        template_embedding = template_raw / np.linalg.norm(template_raw)
        timing["build_template_ms"] = (time.perf_counter() - t0) * 1000.0

        # 3. Preprocess Test Image
        t0 = time.perf_counter()
        prep_test = self.preprocessor.preprocess(test_image)
        timing["preprocess_test_ms"] = (time.perf_counter() - t0) * 1000.0

        if not prep_test.success or prep_test.normalized_face is None:
            return MultiReferenceVerificationResult(
                success=False,
                same_person=False,
                similarity=None,
                template_similarity=None,
                max_similarity=None,
                mean_similarity=None,
                threshold=active_threshold,
                distance_metric=chosen_metric,
                valid_reference_count=len(valid_ref_embeddings),
                total_reference_count=len(reference_images),
                test_face_count=prep_test.face_count,
                message=f"Multi-reference verification rejected: Test image ({prep_test.message})",
                timing_ms=timing,
                test_preprocessing=prep_test,
            )

        # 4. Extract Test Feature
        t0 = time.perf_counter()
        feat_test = self.extract_feature(prep_test.normalized_face, normalize_l2=True)
        timing["extract_test_feature_ms"] = (time.perf_counter() - t0) * 1000.0

        # 5. Compute Template, Max, and Individual Similarities
        t0 = time.perf_counter()
        individual_scores = [
            self.compute_similarity(ref_feat, feat_test, metric=chosen_metric)
            for ref_feat in valid_ref_embeddings
        ]
        template_sim = self.compute_similarity(template_embedding, feat_test, metric=chosen_metric)
        max_sim = (
            float(np.max(individual_scores))
            if chosen_metric == "cosine"
            else float(np.min(individual_scores))
        )
        mean_sim = float(np.mean(individual_scores))

        # Robust score: combination of template and maximum match
        if chosen_metric == "cosine":
            robust_score = max(template_sim, max_sim)
            same_person = robust_score >= active_threshold
        else:
            robust_score = min(template_sim, max_sim)
            same_person = robust_score <= active_threshold

        timing["matching_ms"] = (time.perf_counter() - t0) * 1000.0
        timing["total_ms"] = sum(timing.values())

        decision_str = "SAME PERSON" if same_person else "DIFFERENT PERSON"
        msg = (
            f"Multi-reference verification successful: {decision_str} "
            f"(Robust Score: {robust_score:.4f}, Template: {template_sim:.4f}, Max: {max_sim:.4f}, "
            f"Enrolled: {len(valid_ref_embeddings)}/{len(reference_images)}, Threshold: {active_threshold:.4f})"
        )

        return MultiReferenceVerificationResult(
            success=True,
            same_person=same_person,
            similarity=robust_score,
            template_similarity=template_sim,
            max_similarity=max_sim,
            mean_similarity=mean_sim,
            threshold=active_threshold,
            distance_metric=chosen_metric,
            valid_reference_count=len(valid_ref_embeddings),
            total_reference_count=len(reference_images),
            test_face_count=prep_test.face_count,
            individual_scores=individual_scores,
            message=msg,
            timing_ms=timing,
            test_preprocessing=prep_test,
        )
