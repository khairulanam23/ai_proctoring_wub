"""Face preprocessing, quality assessment, pose estimation, and landmark-based alignment module."""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np

from proctoring.detection.face_detector import FaceDetection


class PreprocessingStatus(str, Enum):
    """Quality and validity status outcomes for face preprocessing."""

    GOOD = "GOOD"
    SUCCESS = "SUCCESS"  # Backward compatibility alias for GOOD
    MODERATE_POSE = "MODERATE_POSE"
    EXTREME_POSE = "EXTREME_POSE"
    LOW_RESOLUTION = "LOW_RESOLUTION"
    FACE_TOO_SMALL = "FACE_TOO_SMALL"  # Backward compatibility alias
    BLUR_WARNING = "BLUR_WARNING"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    INVALID_IMAGE = "INVALID_IMAGE"


@dataclass
class PoseMetrics:
    """Estimated geometric head pose and facial symmetry indicators from 5 landmarks."""

    yaw_ratio: float  # Horizontal nose offset ratio relative to eyes (0.50 = perfectly frontal)
    roll_angle_deg: float  # Planar tilt in degrees
    pitch_ratio: float  # Vertical nose placement ratio between eyes and mouth
    is_frontal: bool
    is_extreme_pose: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "yaw_ratio": round(self.yaw_ratio, 3),
            "roll_angle_deg": round(self.roll_angle_deg, 1),
            "pitch_ratio": round(self.pitch_ratio, 3),
            "is_frontal": self.is_frontal,
            "is_extreme_pose": self.is_extreme_pose,
        }


@dataclass
class QualityMetrics:
    """Comprehensive quality, resolution, and pose assessment of detected facial region."""

    face_width: int
    face_height: int
    min_dimension: int
    is_size_valid: bool
    blur_score: float  # Variance of the Laplacian
    is_sharp: bool
    confidence: float
    pose: PoseMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_width": self.face_width,
            "face_height": self.face_height,
            "min_dimension": self.min_dimension,
            "is_size_valid": self.is_size_valid,
            "blur_score": round(self.blur_score, 2),
            "is_sharp": self.is_sharp,
            "confidence": round(self.confidence, 4),
            "pose": self.pose.to_dict() if self.pose else None,
        }


@dataclass
class PreprocessingResult:
    """Structured result of face preprocessing, quality validation, and alignment."""

    success: bool
    status: PreprocessingStatus
    aligned_face: np.ndarray | None  # (112, 112, 3) BGR aligned
    normalized_face: np.ndarray | None  # (112, 112, 3) BGR preprocessed for SFace
    raw_face: FaceDetection | None
    face_count: int
    quality: QualityMetrics | None
    transform_matrix: np.ndarray | None  # (2, 3) Affine matrix
    timing_ms: dict[str, float] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status.value,
            "face_count": self.face_count,
            "message": self.message,
            "quality": self.quality.to_dict() if self.quality else None,
            "aligned_shape": list(self.aligned_face.shape)
            if self.aligned_face is not None
            else None,
            "timing_ms": {k: round(v, 2) for k, v in self.timing_ms.items()},
        }


class FacePreprocessor:
    """Robust face preprocessor with 5-point landmark alignment, pose estimation, and quality gating."""

    # Official canonical 112x112 5-point landmark coordinate template for OpenCV SFace
    # 0: Right Eye (subject's right / viewer's left)
    # 1: Left Eye (subject's left / viewer's right)
    # 2: Nose Tip
    # 3: Right Mouth Corner
    # 4: Left Mouth Corner
    CANONICAL_5_POINTS_112 = np.array(
        [
            [38.2946, 51.6963],  # Right Eye
            [73.5318, 51.5014],  # Left Eye
            [56.0252, 71.7366],  # Nose Tip
            [41.5493, 92.3655],  # Right Mouth
            [70.7299, 92.2041],  # Left Mouth
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        detector: Any | None = None,
        output_size: tuple[int, int] = (112, 112),
        min_face_size: int = 40,
        min_blur_score: float = 15.0,
        illumination_mode: str = "none",  # "none" (native SFace), "mild_contrast", or "clahe"
        enable_illumination_norm: bool | None = None,  # Backward compatibility
        clahe_clip_limit: float = 1.5,
        clahe_tile_grid: tuple[int, int] = (4, 4),
        max_yaw_deviation: float = 0.25,  # Deviation from 0.50 (e.g. < 0.25 or > 0.75 triggers extreme pose)
    ) -> None:
        if detector is not None:
            self.detector = detector
        else:
            from proctoring.detection.face_detector import FaceDetector

            self.detector = FaceDetector()
        self.output_size = output_size
        self.min_face_size = min_face_size
        self.min_blur_score = min_blur_score
        self.max_yaw_deviation = max_yaw_deviation

        # Handle backward compatibility for enable_illumination_norm boolean
        if enable_illumination_norm is not None:
            self.illumination_mode = "clahe" if enable_illumination_norm else "none"
        else:
            self.illumination_mode = illumination_mode.lower()

        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid,
        )

        # Precompute canonical template scaled to output_size
        if output_size == (112, 112):
            self.canonical_template = self.CANONICAL_5_POINTS_112
        else:
            scale_x = output_size[0] / 112.0
            scale_y = output_size[1] / 112.0
            self.canonical_template = self.CANONICAL_5_POINTS_112 * np.array(
                [scale_x, scale_y], dtype=np.float32
            )

    def compute_blur_score(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        """Calculate image sharpness score using the variance of the Laplacian."""
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def estimate_pose(self, landmarks: list[tuple[float, float]]) -> PoseMetrics:
        """Estimate head pose and facial symmetry indicators from 5 facial landmarks."""
        re = landmarks[0]  # Right Eye
        le = landmarks[1]  # Left Eye
        nt = landmarks[2]  # Nose Tip
        rm = landmarks[3]  # Right Mouth
        lm = landmarks[4]  # Left Mouth

        # 1. Roll angle (planar tilt between eyes)
        dx_eyes = le[0] - re[0]
        dy_eyes = le[1] - re[1]
        roll_rad = math.atan2(dy_eyes, dx_eyes)
        roll_deg = math.degrees(roll_rad)

        # 2. Yaw symmetry ratio (horizontal position of nose relative to eye span)
        eye_span = max(1.0, math.hypot(dx_eyes, dy_eyes))
        # Project nose onto eye axis
        proj_nose = (nt[0] - re[0]) * (dx_eyes / eye_span) + (nt[1] - re[1]) * (dy_eyes / eye_span)
        yaw_ratio = float(np.clip(proj_nose / eye_span, 0.0, 1.0))

        # 3. Pitch vertical ratio
        eye_mid = ((re[0] + le[0]) / 2.0, (re[1] + le[1]) / 2.0)
        mouth_mid = ((rm[0] + lm[0]) / 2.0, (rm[1] + lm[1]) / 2.0)
        dist_eye_nose = math.hypot(nt[0] - eye_mid[0], nt[1] - eye_mid[1])
        dist_nose_mouth = math.hypot(mouth_mid[0] - nt[0], mouth_mid[1] - nt[1])
        total_vert = max(1.0, dist_eye_nose + dist_nose_mouth)
        pitch_ratio = float(dist_eye_nose / total_vert)

        # Classify pose
        yaw_dev = abs(yaw_ratio - 0.50)
        is_frontal = yaw_dev <= 0.12 and abs(roll_deg) <= 15.0
        is_extreme_pose = yaw_dev >= self.max_yaw_deviation or abs(roll_deg) >= 45.0

        return PoseMetrics(
            yaw_ratio=yaw_ratio,
            roll_angle_deg=roll_deg,
            pitch_ratio=pitch_ratio,
            is_frontal=is_frontal,
            is_extreme_pose=is_extreme_pose,
        )

    def align_face(
        self,
        image: np.ndarray,
        landmarks: list[tuple[float, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align face using 5-point similarity transformation (least-squares/LMEDS).

        Args:
            image: Original BGR image.
            landmarks: 5 landmark coordinates from YuNet [right_eye, left_eye, nose, mouth_r, mouth_l].

        Returns:
            Tuple of (aligned_face_image_112x112, 2x3 affine_transformation_matrix).
        """
        src_pts = np.array(landmarks, dtype=np.float32)
        # Use cv2.LMEDS for robust least-squares similarity transformation
        transform_matrix, inliers = cv2.estimateAffinePartial2D(
            src_pts, self.canonical_template, method=cv2.LMEDS
        )

        if transform_matrix is None:
            # Fallback: estimate from 3 primary landmarks (eyes + nose)
            transform_matrix = cv2.getAffineTransform(src_pts[:3], self.canonical_template[:3])

        aligned_face = cv2.warpAffine(
            image,
            transform_matrix,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return aligned_face, transform_matrix

    def normalize_illumination(
        self, aligned_bgr: np.ndarray, mode: str | None = None
    ) -> np.ndarray:
        """Apply controlled illumination normalization based on selected mode."""
        active_mode = (mode or self.illumination_mode).lower()

        if active_mode in ("none", "raw"):
            return aligned_bgr.copy()

        if active_mode == "mild_contrast":
            # Conservative linear contrast enhancement (preserves natural gradient structure)
            return cv2.convertScaleAbs(aligned_bgr, alpha=1.05, beta=2)

        if active_mode == "clahe":
            # CLAHE on Luminance channel only (CIE LAB)
            lab = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_norm = self.clahe.apply(l_channel)
            normalized_lab = cv2.merge([l_norm, a_channel, b_channel])
            return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

        return aligned_bgr.copy()

    def preprocess(
        self,
        image: np.ndarray,
        score_threshold: float | None = None,
        require_single_face: bool = True,
    ) -> PreprocessingResult:
        """Perform full validation, detection, quality assessment, pose estimation, and 5-point alignment.

        Args:
            image: Input BGR image array.
            score_threshold: Optional confidence threshold for face detector.
            require_single_face: If True, rejects inputs with 0 or >1 face.

        Returns:
            PreprocessingResult containing aligned/normalized crops and quality metadata.
        """
        timing: dict[str, float] = {}

        # 1. Image Validation
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return PreprocessingResult(
                success=False,
                status=PreprocessingStatus.INVALID_IMAGE,
                aligned_face=None,
                normalized_face=None,
                raw_face=None,
                face_count=0,
                quality=None,
                transform_matrix=None,
                timing_ms={"total_ms": 0.0},
                message="Invalid input image: must be a non-empty numpy array.",
            )

        if len(image.shape) != 3 or image.shape[2] != 3:
            return PreprocessingResult(
                success=False,
                status=PreprocessingStatus.INVALID_IMAGE,
                aligned_face=None,
                normalized_face=None,
                raw_face=None,
                face_count=0,
                quality=None,
                transform_matrix=None,
                timing_ms={"total_ms": 0.0},
                message=f"Invalid image format: expected 3-channel BGR, got shape {image.shape}.",
            )

        # 2. Face Detection
        t0 = time.perf_counter()
        det_result = self.detector.detect(image, score_threshold=score_threshold)
        timing["detect_ms"] = (time.perf_counter() - t0) * 1000.0

        # 3. Face Count Validation
        if det_result.count == 0:
            timing["total_ms"] = timing["detect_ms"]
            return PreprocessingResult(
                success=False,
                status=PreprocessingStatus.NO_FACE,
                aligned_face=None,
                normalized_face=None,
                raw_face=None,
                face_count=0,
                quality=None,
                transform_matrix=None,
                timing_ms=timing,
                message="Preprocessing rejected: No face detected in image.",
            )

        if require_single_face and det_result.count > 1:
            timing["total_ms"] = timing["detect_ms"]
            return PreprocessingResult(
                success=False,
                status=PreprocessingStatus.MULTIPLE_FACES,
                aligned_face=None,
                normalized_face=None,
                raw_face=None,
                face_count=det_result.count,
                quality=None,
                transform_matrix=None,
                timing_ms=timing,
                message=f"Preprocessing rejected: Multiple faces detected ({det_result.count} faces). Exactly one face required.",
            )

        # Select primary face (highest confidence)
        primary_face = max(det_result.faces, key=lambda f: f.confidence)
        x, y, w, h = primary_face.bbox
        min_dim = min(w, h)

        # 4. Quality & Pose Assessment
        blur_score = self.compute_blur_score(image, primary_face.bbox)
        is_size_valid = min_dim >= self.min_face_size
        is_sharp = blur_score >= self.min_blur_score
        pose = self.estimate_pose(primary_face.landmarks)

        quality = QualityMetrics(
            face_width=w,
            face_height=h,
            min_dimension=min_dim,
            is_size_valid=is_size_valid,
            blur_score=blur_score,
            is_sharp=is_sharp,
            confidence=primary_face.confidence,
            pose=pose,
        )

        if not is_size_valid:
            timing["total_ms"] = timing["detect_ms"]
            return PreprocessingResult(
                success=False,
                status=PreprocessingStatus.LOW_RESOLUTION,
                aligned_face=None,
                normalized_face=None,
                raw_face=primary_face,
                face_count=det_result.count,
                quality=quality,
                transform_matrix=None,
                timing_ms=timing,
                message=f"Preprocessing rejected: Face dimensions ({w}x{h}) smaller than minimum allowed ({self.min_face_size}px).",
            )

        # 5. Landmark-Based Alignment
        t0 = time.perf_counter()
        aligned_face, transform_matrix = self.align_face(image, primary_face.landmarks)
        timing["align_ms"] = (time.perf_counter() - t0) * 1000.0

        # 6. Controlled Illumination Normalization
        t0 = time.perf_counter()
        normalized_face = self.normalize_illumination(aligned_face)
        timing["norm_ms"] = (time.perf_counter() - t0) * 1000.0

        timing["total_ms"] = sum(timing.values())

        # Determine definitive quality status
        if pose.is_extreme_pose:
            status = PreprocessingStatus.EXTREME_POSE
            msg = f"Preprocessing completed with extreme pose warning (yaw ratio: {pose.yaw_ratio:.2f}, roll: {pose.roll_angle_deg:.1f}°)."
        elif not pose.is_frontal:
            status = PreprocessingStatus.MODERATE_POSE
            msg = f"Preprocessing completed with moderate pose (yaw ratio: {pose.yaw_ratio:.2f})."
        elif not is_sharp:
            status = PreprocessingStatus.BLUR_WARNING
            msg = f"Preprocessing completed with blur warning (blur score: {blur_score:.1f})."
        else:
            status = PreprocessingStatus.GOOD
            msg = "Preprocessing successful (Good quality frontal face)."

        return PreprocessingResult(
            success=True,
            status=status,
            aligned_face=aligned_face,
            normalized_face=normalized_face,
            raw_face=primary_face,
            face_count=det_result.count,
            quality=quality,
            transform_matrix=transform_matrix,
            timing_ms=timing,
            message=msg,
        )
