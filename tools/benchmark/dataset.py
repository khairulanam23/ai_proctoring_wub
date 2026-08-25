"""Structured evaluation dataset, ground-truth label schema, and multi-modal test case generation."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from proctoring.core.events import EventType
from tools.research.robustness import ImageAugmenter, VisualCondition


class EvaluationCategory(str, Enum):
    """Standardized test categories covering realistic examination scenarios."""

    NORMAL_EXAM = "NORMAL_EXAM"  # Category A: Single student, acceptable lighting, normal posture
    FACE_ABSENCE = "FACE_ABSENCE"  # Category B: Face leaves camera, turns away, camera obstructed
    MULTIPLE_PERSON = "MULTIPLE_PERSON"  # Category C: Second person appears, multiple faces
    IDENTITY_VERIFICATION = (
        "IDENTITY_VERIFICATION"  # Category D: Genuine vs impostor identity checks
    )
    OBJECT_DETECTION = (
        "OBJECT_DETECTION"  # Category E: Prohibited objects (phone, book) vs normal objects
    )
    ENVIRONMENTAL_STRESS = (
        "ENVIRONMENTAL_STRESS"  # Category F: Low light, high light, blur, noise, contrast
    )
    MOVEMENT_POSE = "MOVEMENT_POSE"  # Category G: Normal movement, looking left/right/down
    ADVERSARIAL_EDGE_CASES = "ADVERSARIAL_EDGE_CASES"  # Category H: Border proximity, small face, occlusions, corrupt frames


class GroundTruthLabel(str, Enum):
    """Explicit ground-truth classifications for test samples."""

    NORMAL = "NORMAL"
    FACE_ABSENT = "FACE_ABSENT"
    MULTIPLE_PERSON = "MULTIPLE_PERSON"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    PROHIBITED_OBJECT = "PROHIBITED_OBJECT"
    LOW_IMAGE_QUALITY = "LOW_IMAGE_QUALITY"
    SUSPICIOUS_POSE = "SUSPICIOUS_POSE"
    BROWSER_VIOLATION = "BROWSER_VIOLATION"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass
class EvaluationSample:
    """Individual test sample with ground truth metadata and input media."""

    sample_id: str
    category: EvaluationCategory
    ground_truth_label: GroundTruthLabel
    expected_events: list[EventType]
    description: str
    image_path: str | None = None
    image_array: np.ndarray | None = None
    video_path: str | None = None
    enrolled_identity: str | None = None
    subject_identity: str | None = None
    is_genuine_match: bool | None = None
    ground_truth_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    ground_truth_objects: list[str] = field(default_factory=list)
    mock_detected_objects: list[dict[str, Any]] = field(default_factory=list)
    browser_events: list[tuple[float, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "category": self.category.value,
            "ground_truth_label": self.ground_truth_label.value,
            "expected_events": [e.value for e in self.expected_events],
            "description": self.description,
            "image_path": self.image_path,
            "video_path": self.video_path,
            "enrolled_identity": self.enrolled_identity,
            "subject_identity": self.subject_identity,
            "is_genuine_match": self.is_genuine_match,
            "ground_truth_bboxes": [list(b) for b in self.ground_truth_bboxes],
            "ground_truth_objects": self.ground_truth_objects,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationDataset:
    """Collection of structured evaluation samples across all test categories."""

    dataset_name: str
    version: str
    samples: list[EvaluationSample]
    created_at_utc: str = ""

    def get_by_category(self, category: EvaluationCategory) -> list[EvaluationSample]:
        return [s for s in self.samples if s.category == category]

    def get_by_label(self, label: GroundTruthLabel) -> list[EvaluationSample]:
        return [s for s in self.samples if s.ground_truth_label == label]

    def summary(self) -> dict[str, Any]:
        cat_counts = {}
        label_counts = {}
        for s in self.samples:
            cat_counts[s.category.value] = cat_counts.get(s.category.value, 0) + 1
            label_counts[s.ground_truth_label.value] = (
                label_counts.get(s.ground_truth_label.value, 0) + 1
            )

        return {
            "total_samples": len(self.samples),
            "category_distribution": cat_counts,
            "label_distribution": label_counts,
        }


class EvaluationDatasetBuilder:
    """Builds a standardized, multi-modal evaluation dataset from sample repositories."""

    @staticmethod
    def build_default_benchmark_dataset(
        samples_dir: str | Path = "data/samples",
    ) -> EvaluationDataset:
        """Construct full Phase 5 benchmark dataset covering Categories A through H."""
        base_path = Path(samples_dir)
        samples: list[EvaluationSample] = []
        sample_idx = 0

        def next_id(prefix: str) -> str:
            nonlocal sample_idx
            sample_idx += 1
            return f"SMP_{prefix}_{sample_idx:04d}"

        # 1. Discover LFW samples
        identities: dict[str, list[Path]] = {}
        if base_path.exists():
            for d in sorted(base_path.iterdir()):
                if d.is_dir() and d.name != "synthetic":
                    imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")))
                    if imgs:
                        identities[d.name] = imgs

        primary_id = (
            "Colin_Powell"
            if "Colin_Powell" in identities
            else (list(identities.keys())[0] if identities else "Subject_A")
        )
        primary_imgs = identities.get(primary_id, [])

        # ----------------------------------------------------------------------
        # CATEGORY A: NORMAL EXAM CONDITIONS
        # ----------------------------------------------------------------------
        if primary_imgs:
            for idx, img_p in enumerate(primary_imgs[:3]):
                img = cv2.imread(str(img_p))
                samples.append(
                    EvaluationSample(
                        sample_id=next_id("NORM"),
                        category=EvaluationCategory.NORMAL_EXAM,
                        ground_truth_label=GroundTruthLabel.NORMAL,
                        expected_events=[],
                        description=f"Normal enrolled student ({primary_id}) visible and centered",
                        image_path=str(img_p),
                        image_array=img,
                        enrolled_identity=primary_id,
                        subject_identity=primary_id,
                        is_genuine_match=True,
                        metadata={"lighting": "normal", "posture": "standard"},
                    )
                )

        # ----------------------------------------------------------------------
        # CATEGORY B: FACE ABSENCE
        # ----------------------------------------------------------------------
        no_face_canvas = np.full((360, 480, 3), 215, dtype=np.uint8)
        cv2.putText(
            no_face_canvas,
            "Empty Desk / Wall",
            (60, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (90, 90, 90),
            2,
        )
        samples.append(
            EvaluationSample(
                sample_id=next_id("ABSC"),
                category=EvaluationCategory.FACE_ABSENCE,
                ground_truth_label=GroundTruthLabel.FACE_ABSENT,
                expected_events=[EventType.NO_FACE],
                description="Student leaves webcam frame completely (Empty desk canvas)",
                image_array=no_face_canvas,
                enrolled_identity=primary_id,
                subject_identity=None,
                metadata={"reason": "candidate_left_frame"},
            )
        )

        # Camera obstructed (black frame)
        black_canvas = np.zeros((360, 480, 3), dtype=np.uint8)
        samples.append(
            EvaluationSample(
                sample_id=next_id("ABSC"),
                category=EvaluationCategory.FACE_ABSENCE,
                ground_truth_label=GroundTruthLabel.FACE_ABSENT,
                expected_events=[EventType.NO_FACE],
                description="Camera lens physically covered (Black frame)",
                image_array=black_canvas,
                enrolled_identity=primary_id,
                subject_identity=None,
                metadata={"reason": "lens_obstructed"},
            )
        )

        # ----------------------------------------------------------------------
        # CATEGORY C: MULTIPLE PERSON
        # ----------------------------------------------------------------------
        other_ids = [k for k in identities if k != primary_id]
        if primary_imgs and other_ids:
            sec_id = other_ids[0]
            sec_imgs = identities[sec_id]
            if sec_imgs:
                img_a = cv2.imread(str(primary_imgs[0]))
                img_b = cv2.imread(str(sec_imgs[0]))
                h = 360
                ia_r = cv2.resize(img_a, (int(img_a.shape[1] * h / img_a.shape[0]), h))
                ib_r = cv2.resize(img_b, (int(img_b.shape[1] * h / img_b.shape[0]), h))
                dual_canvas = np.hstack([ia_r, ib_r])

                samples.append(
                    EvaluationSample(
                        sample_id=next_id("MULT"),
                        category=EvaluationCategory.MULTIPLE_PERSON,
                        ground_truth_label=GroundTruthLabel.MULTIPLE_PERSON,
                        expected_events=[EventType.MULTIPLE_FACES],
                        description=f"Two persons in scene: Enrolled ({primary_id}) + Assistant ({sec_id})",
                        image_array=dual_canvas,
                        enrolled_identity=primary_id,
                        subject_identity=f"{primary_id}+{sec_id}",
                        metadata={"face_count": 2},
                    )
                )

        # ----------------------------------------------------------------------
        # CATEGORY D: IDENTITY VERIFICATION (GENUINE VS IMPOSTOR)
        # ----------------------------------------------------------------------
        if primary_imgs and other_ids:
            # Genuine pairs
            if len(primary_imgs) >= 2:
                img_gen = cv2.imread(str(primary_imgs[1]))
                samples.append(
                    EvaluationSample(
                        sample_id=next_id("ID_GEN"),
                        category=EvaluationCategory.IDENTITY_VERIFICATION,
                        ground_truth_label=GroundTruthLabel.NORMAL,
                        expected_events=[],
                        description=f"Genuine student pair ({primary_id} image 2 vs reference)",
                        image_path=str(primary_imgs[1]),
                        image_array=img_gen,
                        enrolled_identity=primary_id,
                        subject_identity=primary_id,
                        is_genuine_match=True,
                    )
                )

            # Impostor pairs across distinct identities
            for oth_k in other_ids[:4]:
                oth_img_p = identities[oth_k][0]
                img_imp = cv2.imread(str(oth_img_p))
                samples.append(
                    EvaluationSample(
                        sample_id=next_id("ID_IMP"),
                        category=EvaluationCategory.IDENTITY_VERIFICATION,
                        ground_truth_label=GroundTruthLabel.IDENTITY_MISMATCH,
                        expected_events=[EventType.UNKNOWN_FACE],
                        description=f"Impostor student ({oth_k}) attempting exam instead of {primary_id}",
                        image_path=str(oth_img_p),
                        image_array=img_imp,
                        enrolled_identity=primary_id,
                        subject_identity=oth_k,
                        is_genuine_match=False,
                    )
                )

        # ----------------------------------------------------------------------
        # CATEGORY E: OBJECT DETECTION (PROHIBITED OBJECTS)
        # ----------------------------------------------------------------------
        if primary_imgs:
            base_face = cv2.resize(cv2.imread(str(primary_imgs[0])), (480, 360))

            # Phone canvas
            phone_canvas = base_face.copy()
            cv2.rectangle(phone_canvas, (320, 200), (390, 320), (25, 25, 25), -1)
            cv2.putText(
                phone_canvas,
                "PHONE",
                (330, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            samples.append(
                EvaluationSample(
                    sample_id=next_id("OBJ_PHN"),
                    category=EvaluationCategory.OBJECT_DETECTION,
                    ground_truth_label=GroundTruthLabel.PROHIBITED_OBJECT,
                    expected_events=[EventType.PHONE_DETECTED],
                    description="Student holding cell phone in camera view",
                    image_array=phone_canvas,
                    enrolled_identity=primary_id,
                    subject_identity=primary_id,
                    ground_truth_objects=["cell phone"],
                    mock_detected_objects=[
                        {
                            "class_name": "cell phone",
                            "confidence": 0.89,
                            "bbox": (320, 200, 390, 320),
                        }
                    ],
                )
            )

            # Book canvas
            book_canvas = base_face.copy()
            cv2.rectangle(book_canvas, (50, 220), (180, 340), (180, 100, 50), -1)
            cv2.putText(
                book_canvas, "BOOK", (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

            samples.append(
                EvaluationSample(
                    sample_id=next_id("OBJ_BK"),
                    category=EvaluationCategory.OBJECT_DETECTION,
                    ground_truth_label=GroundTruthLabel.PROHIBITED_OBJECT,
                    expected_events=[EventType.PROHIBITED_OBJECT],
                    description="Textbook placed on desk within camera view",
                    image_array=book_canvas,
                    enrolled_identity=primary_id,
                    subject_identity=primary_id,
                    ground_truth_objects=["book"],
                    mock_detected_objects=[
                        {"class_name": "book", "confidence": 0.82, "bbox": (50, 220, 180, 340)}
                    ],
                )
            )

        # ----------------------------------------------------------------------
        # CATEGORY F: ENVIRONMENTAL STRESS (PERTURBATIONS)
        # ----------------------------------------------------------------------
        if primary_imgs:
            clean_face = cv2.imread(str(primary_imgs[0]))

            # Low light
            dark_face = ImageAugmenter.apply_condition(
                clean_face, VisualCondition.LOW_LIGHT, intensity=1.0
            )
            samples.append(
                EvaluationSample(
                    sample_id=next_id("ENV_DARK"),
                    category=EvaluationCategory.ENVIRONMENTAL_STRESS,
                    ground_truth_label=GroundTruthLabel.NORMAL,
                    expected_events=[],
                    description="Dim room lighting (low illumination test)",
                    image_array=dark_face,
                    enrolled_identity=primary_id,
                    subject_identity=primary_id,
                    metadata={"condition": "low_light"},
                )
            )

            # High light / glare
            bright_face = ImageAugmenter.apply_condition(
                clean_face, VisualCondition.HIGH_LIGHT, intensity=0.8
            )
            samples.append(
                EvaluationSample(
                    sample_id=next_id("ENV_BRT"),
                    category=EvaluationCategory.ENVIRONMENTAL_STRESS,
                    ground_truth_label=GroundTruthLabel.NORMAL,
                    expected_events=[],
                    description="High illumination / webcam glare",
                    image_array=bright_face,
                    enrolled_identity=primary_id,
                    subject_identity=primary_id,
                    metadata={"condition": "high_light"},
                )
            )

            # Gaussian blur / soft focus
            blur_face = ImageAugmenter.apply_condition(
                clean_face, VisualCondition.GAUSSIAN_BLUR, intensity=1.0
            )
            samples.append(
                EvaluationSample(
                    sample_id=next_id("ENV_BLUR"),
                    category=EvaluationCategory.ENVIRONMENTAL_STRESS,
                    ground_truth_label=GroundTruthLabel.NORMAL,
                    expected_events=[],
                    description="Mild camera defocus / motion blur",
                    image_array=blur_face,
                    enrolled_identity=primary_id,
                    subject_identity=primary_id,
                    metadata={"condition": "gaussian_blur"},
                )
            )

        # ----------------------------------------------------------------------
        # CATEGORY G: MOVEMENT / POSE
        # ----------------------------------------------------------------------
        if primary_imgs:
            tilt_face = ImageAugmenter.apply_condition(
                clean_face, VisualCondition.PERSPECTIVE_TILT, intensity=0.8
            )
            samples.append(
                EvaluationSample(
                    sample_id=next_id("POSE_TLT"),
                    category=EvaluationCategory.MOVEMENT_POSE,
                    ground_truth_label=GroundTruthLabel.NORMAL,
                    expected_events=[],
                    description="Student head tilt during typing/writing",
                    image_array=tilt_face,
                    enrolled_identity=primary_id,
                    subject_identity=primary_id,
                    metadata={"condition": "perspective_tilt"},
                )
            )

        # ----------------------------------------------------------------------
        # CATEGORY H: ADVERSARIAL / EDGE CASES
        # ----------------------------------------------------------------------
        # Corrupt / empty frame buffer
        samples.append(
            EvaluationSample(
                sample_id=next_id("EDGE_CORR"),
                category=EvaluationCategory.ADVERSARIAL_EDGE_CASES,
                ground_truth_label=GroundTruthLabel.SYSTEM_ERROR,
                expected_events=[],
                description="Corrupt 0-size image buffer (testing pipeline recovery)",
                image_array=np.array([]),
                enrolled_identity=primary_id,
                metadata={"condition": "empty_buffer"},
            )
        )

        return EvaluationDataset(
            dataset_name="Phase5_Comprehensive_Benchmark_Dataset",
            version="1.0.0",
            samples=samples,
        )
