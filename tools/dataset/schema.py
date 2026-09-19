"""WUB Proctoring Evaluation and Validation Dataset Schema.

Provides structured metadata specifications, annotation schemas, split integrity
protocols, and scenario definitions for future controlled data collection at World University
of Bangladesh (WUB).

CRITICAL NOTICE:
Real student dataset collection has not yet occurred.
WUB ML validation dataset: NOT AVAILABLE
Model domain quality: NOT VALIDATED
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class DatasetSplit(str, Enum):
    """Dataset partition with strict subject-level isolation to prevent data leakage."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class WUBExamScenario(str, Enum):
    """Representative examination scenarios including both compliant baseline and anomalous behaviors."""

    # 1. Normal / Negative Baseline Conditions (Crucial to prevent false positives)
    DIGITAL_NORMAL_WORKING = "DIGITAL_NORMAL_WORKING"
    WRITTEN_NORMAL_HANDWRITING = "WRITTEN_NORMAL_HANDWRITING"
    PERMITTED_CALCULATOR_USE = "PERMITTED_CALCULATOR_USE"
    PERMITTED_SCRATCHPAPER_USE = "PERMITTED_SCRATCHPAPER_USE"
    NORMAL_DESK_ADJUSTMENT = "NORMAL_DESK_ADJUSTMENT"
    DRINKING_WATER_OR_MEDICATION = "DRINKING_WATER_OR_MEDICATION"
    THINKING_POSTURE_HAND_ON_CHIN = "THINKING_POSTURE_HAND_ON_CHIN"
    EYEGLASSES_ADJUSTMENT = "EYEGLASSES_ADJUSTMENT"

    # 2. Environmental & Hardware Variations
    DIM_EVENING_ROOM = "DIM_EVENING_ROOM"
    NATURAL_DAYLIGHT_WINDOW_GLARE = "NATURAL_DAYLIGHT_WINDOW_GLARE"
    BACKLIT_WINDOW = "BACKLIT_WINDOW"
    LOW_ANGLE_LAPTOP_WEBCAM = "LOW_ANGLE_LAPTOP_WEBCAM"
    HIGH_ANGLE_MONITOR_WEBCAM = "HIGH_ANGLE_MONITOR_WEBCAM"

    # 3. Target Disambiguation Scenarios (Confusable objects)
    SCIENTIFIC_CALCULATOR_VS_PHONE = "SCIENTIFIC_CALCULATOR_VS_PHONE"
    NOTEBOOK_COVER_VS_PHONE = "NOTEBOOK_COVER_VS_PHONE"
    POWER_BANK_ON_DESK = "POWER_BANK_ON_DESK"
    EAR_OCCLUSION_BY_HAIR_VS_EARBUD = "EAR_OCCLUSION_BY_HAIR_VS_EARBUD"

    # 4. Prohibited Behaviors
    ACTIVE_PHONE_COMMUNICATION = "ACTIVE_PHONE_COMMUNICATION"
    SECONDARY_PERSON_IN_ROOM = "SECONDARY_PERSON_IN_ROOM"
    IMPOSTOR_CANDIDATE_SUBSTITUTION = "IMPOSTOR_CANDIDATE_SUBSTITUTION"
    CAMERA_LENS_OBSTRUCTED = "CAMERA_LENS_OBSTRUCTED"
    UNAUTHORIZED_HEADPHONES = "UNAUTHORIZED_HEADPHONES"


@dataclass
class BoundingBoxAnnotation:
    """Normalized or pixel bounding box with domain category annotation."""

    category: str
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    is_permitted: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "bbox": list(self.bbox),
            "is_permitted": self.is_permitted,
            "attributes": self.attributes,
        }


@dataclass
class SampleMetadata:
    """Ground-truth metadata definition for a single frame or video sample."""

    sample_id: str
    participant_pseudonym: str
    split: DatasetSplit
    scenario: WUBExamScenario
    lighting_condition: str
    camera_model: str
    resolution: tuple[int, int]
    annotations: list[BoundingBoxAnnotation] = field(default_factory=list)
    timestamp_offset_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "participant_pseudonym": self.participant_pseudonym,
            "split": self.split.value,
            "scenario": self.scenario.value,
            "lighting_condition": self.lighting_condition,
            "camera_model": self.camera_model,
            "resolution": list(self.resolution),
            "annotations": [a.to_dict() for a in self.annotations],
            "timestamp_offset_seconds": self.timestamp_offset_seconds,
        }


@dataclass
class WUBDatasetManifest:
    """Dataset manifest defining global statistics, split allocations, and leakage verification."""

    dataset_name: str = "WUB_Exam_Proctoring_Benchmark"
    version: str = "1.0.0-framework"
    validation_dataset_status: str = "NOT AVAILABLE"
    model_domain_quality: str = "NOT VALIDATED"
    samples: list[SampleMetadata] = field(default_factory=list)

    def check_split_leakage(self) -> list[str]:
        """Assert that no participant pseudonym appears across multiple splits."""
        split_participants: dict[DatasetSplit, set[str]] = {
            DatasetSplit.TRAIN: set(),
            DatasetSplit.VAL: set(),
            DatasetSplit.TEST: set(),
        }
        for s in self.samples:
            split_participants[s.split].add(s.participant_pseudonym)

        violations: list[str] = []
        train_val = split_participants[DatasetSplit.TRAIN] & split_participants[DatasetSplit.VAL]
        if train_val:
            violations.append(f"Leakage between TRAIN and VAL: {train_val}")

        train_test = split_participants[DatasetSplit.TRAIN] & split_participants[DatasetSplit.TEST]
        if train_test:
            violations.append(f"Leakage between TRAIN and TEST: {train_test}")

        val_test = split_participants[DatasetSplit.VAL] & split_participants[DatasetSplit.TEST]
        if val_test:
            violations.append(f"Leakage between VAL and TEST: {val_test}")

        return violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "version": self.version,
            "validation_dataset_status": self.validation_dataset_status,
            "model_domain_quality": self.model_domain_quality,
            "total_samples": len(self.samples),
            "samples": [s.to_dict() for s in self.samples],
        }
