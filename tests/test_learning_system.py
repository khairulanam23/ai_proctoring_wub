"""Comprehensive tests for controlled continuous learning system, dataset versioning,
model registry, and permanent regression suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from proctoring.analysis.policy import ExamMode, ExamPolicy, StrictnessLevel
from proctoring.analysis.wearables import (
    EventType,
    WearableCategory,
    WearableDetection,
    WearableDetector,
)
from proctoring.learning.annotations import (
    AnnotatedBBox,
    HumanAnnotationRecord,
    ReviewStatus,
    TargetObjectLabel,
)
from proctoring.learning.datasets import DatasetManager
from proctoring.learning.evaluation import ChampionChallengerEvaluator
from proctoring.learning.inbox import FlagReason, TrainingInboxManager
from proctoring.learning.registry import ModelMetadata, ModelMetrics, ModelRegistry, ModelStatus


def test_training_inbox_lifecycle(tmp_path: Path):
    inbox = TrainingInboxManager(base_dir=tmp_path)
    blank_img = np.full((120, 160, 3), 128, dtype=np.uint8)

    sample = inbox.capture_sample(
        image=blank_img,
        session_id="sess_test_101",
        frame_index=42,
        target_class="cell phone",
        flag_reason=FlagReason.UNCERTAIN_DETECTION,
        confidence=0.51,
        bbox=(20, 30, 80, 100),
        is_hard_negative=False,
    )

    assert sample.sample_id is not None
    assert (tmp_path / sample.image_rel_path).exists()

    pending = inbox.list_pending_samples()
    assert len(pending) == 1
    assert pending[0].sample_id == sample.sample_id
    assert pending[0].flag_reason == FlagReason.UNCERTAIN_DETECTION

    # Test deletion
    deleted = inbox.delete_sample(sample.sample_id)
    assert deleted is True
    assert len(inbox.list_pending_samples()) == 0
    assert not (tmp_path / sample.image_rel_path).exists()


def test_dataset_manager_session_grouping(tmp_path: Path):
    inbox = TrainingInboxManager(base_dir=tmp_path / "inbox")
    img = np.full((100, 100, 3), 200, dtype=np.uint8)

    # Create 6 samples across 3 distinct sessions (2 per session)
    samples = []
    annotations = {}
    for sess_idx in range(1, 4):
        sess_id = f"session_{sess_idx}"
        for f_idx in (10, 20):
            samp = inbox.capture_sample(
                image=img,
                session_id=sess_id,
                frame_index=f_idx,
                target_class="cell phone",
                flag_reason=FlagReason.UNCERTAIN_DETECTION,
                confidence=0.60,
                bbox=(10, 10, 50, 90),
            )
            samples.append(samp)
            annotations[samp.sample_id] = HumanAnnotationRecord(
                sample_id=samp.sample_id,
                annotator_id="expert_reviewer_1",
                review_status=ReviewStatus.VERIFIED,
                objects=[
                    AnnotatedBBox(
                        label=TargetObjectLabel.PHONE,
                        bbox=(10, 10, 50, 90),
                        is_hard_negative=False,
                    )
                ],
                quality_pass=True,
            )

    manager = DatasetManager(datasets_root=tmp_path / "datasets")
    manifest = manager.create_versioned_dataset(
        version="v1.0.0",
        samples=samples,
        annotations=annotations,
        split_ratio=(0.60, 0.20, 0.20),
    )

    assert manifest.total_samples == 6
    assert manifest.version == "v1.0.0"
    assert "phone" in manifest.classes

    # Ensure dataset manifest can be reloaded
    loaded = manager.load_manifest("v1.0.0")
    assert loaded is not None
    assert loaded.total_samples == 6


def test_model_registry_promotion_and_rollback(tmp_path: Path):
    registry = ModelRegistry(registry_root=tmp_path)
    dummy_model_file = tmp_path / "yolo_v1.pt"
    dummy_model_file.write_text("weights")

    # 1. Register candidate model with good metrics
    good_metrics = ModelMetrics(
        precision=0.92,
        recall=0.89,
        f1_score=0.90,
        false_positive_rate=0.03,
        false_negative_rate=0.11,
        cpu_latency_ms=28.5,
        peak_memory_mb=120.0,
        hard_negative_accuracy=0.95,
    )

    meta1 = registry.register_model(
        model_name="proctoring_yolo",
        version="v1.0.0",
        model_type="object_detector",
        artifact_path=dummy_model_file,
        metrics=good_metrics,
    )
    assert meta1.status == ModelStatus.CANDIDATE

    # Promote to active
    success, msg = registry.evaluate_and_promote(meta1.model_id)
    assert success is True
    active = registry.get_active_model("object_detector")
    assert active is not None
    assert active.model_id == meta1.model_id
    assert active.status == ModelStatus.ACTIVE

    # 2. Register slow candidate model exceeding latency budget
    slow_metrics = ModelMetrics(
        precision=0.95,
        recall=0.95,
        f1_score=0.95,
        cpu_latency_ms=150.0,  # Exceeds max 60ms budget
        hard_negative_accuracy=0.95,
    )
    meta2 = registry.register_model(
        model_name="proctoring_yolo",
        version="v2.0.0",
        model_type="object_detector",
        artifact_path=dummy_model_file,
        metrics=slow_metrics,
    )
    success2, msg2 = registry.evaluate_and_promote(meta2.model_id, max_latency_ms=60.0)
    assert success2 is False
    assert "exceeds budget" in msg2

    # Active model remains v1.0.0
    active_now = registry.get_active_model("object_detector")
    assert active_now.model_id == meta1.model_id


def test_champion_challenger_evaluation():
    evaluator = ChampionChallengerEvaluator()

    champ_metrics = ModelMetrics(
        precision=0.90,
        recall=0.88,
        false_positive_rate=0.04,
        cpu_latency_ms=30.0,
    )

    # Challenger with better precision and recall, similar latency
    challenger_good = ModelMetrics(
        precision=0.94,
        recall=0.91,
        false_positive_rate=0.02,
        cpu_latency_ms=32.0,
    )

    report = evaluator.compare(
        champion_id="yolo_v1",
        challenger_id="yolo_v2",
        champion_metrics=champ_metrics,
        challenger_metrics=challenger_good,
        dataset_name="val_v1.0.0",
        regression_passed=True,
    )
    assert report.recommendation == "PROCEED_TO_APPROVAL"
    assert report.delta_precision > 0.0

    # Challenger failing regression invariants
    report_fail = evaluator.compare(
        champion_id="yolo_v1",
        challenger_id="yolo_v3",
        champion_metrics=champ_metrics,
        challenger_metrics=challenger_good,
        dataset_name="val_v1.0.0",
        regression_passed=False,
    )
    assert report_fail.recommendation == "REJECT"
    assert "regression" in report_fail.reasons[0].lower()


def test_permanent_regression_suite_manifest():
    """Verify all permanent regression invariants against active code."""
    cases_path = Path("training/regression/cases.json")
    assert cases_path.exists()
    cases = json.loads(cases_path.read_text())
    assert len(cases) >= 5

    # Test REG_002 invariant: lobule earring
    detector = WearableDetector(auto_load=False)
    ear_regions = [(100, 100, 160, 200)]
    earring_det = WearableDetection(
        target="earbuds",
        prompt="small earbud",
        event_type=EventType.EARBUDS_SUSPECTED,
        confidence=0.35,
        bbox=(125, 185, 140, 198),
    )
    cat, _ = detector._classify_category(earring_det, ear_regions)
    assert cat == WearableCategory.OTHER_EAR_OBJECT

    # Test REG_003 invariant: physical paper downward pitch tolerance
    paper_policy = ExamPolicy.for_level(
        level=StrictnessLevel.STANDARD, mode=ExamMode.PHYSICAL_PAPER
    )
    assert paper_policy.is_pose_deviated(yaw=0.0, pitch=-45.0) is False
    assert paper_policy.is_gaze_deviated(horizontal=0.0, vertical=-0.55) is False
