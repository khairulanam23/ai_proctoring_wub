"""Targeted tests for the unified proctoring training pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from proctoring.learning import (
    AnnotatedBBox,
    FlagReason,
    HumanAnnotationRecord,
    ModelMetrics,
    ModelStatus,
    ReviewStatus,
    TargetObjectLabel,
    TrainingConfig,
    TrainingPipeline,
)


def test_training_pipeline_end_to_end_flow(tmp_path: Path):
    """Verify complete flow: capture -> annotate -> build dataset with assets -> dry-run train -> registry."""
    inbox_dir = tmp_path / "inbox"
    datasets_dir = tmp_path / "datasets"
    registry_dir = tmp_path / "registry"
    runs_dir = tmp_path / "runs"
    reg_cases_path = Path("training/regression/cases.json")

    pipeline = TrainingPipeline(
        inbox_dir=inbox_dir,
        datasets_dir=datasets_dir,
        registry_dir=registry_dir,
        runs_dir=runs_dir,
        regression_cases_path=reg_cases_path,
    )

    # 1. Capture samples across 3 sessions
    img = np.full((120, 160, 3), 200, dtype=np.uint8)
    samples = []
    annotations = {}

    for s_idx in range(1, 4):
        sess_id = f"sess_{s_idx:03d}"
        for f_idx in (10, 20):
            samp = pipeline.capture_evidence(
                image=img,
                session_id=sess_id,
                frame_index=f_idx,
                target_class="cell phone",
                flag_reason=FlagReason.UNCERTAIN_DETECTION,
                confidence=0.55,
                bbox=(20, 30, 80, 110),
            )
            samples.append(samp)
            annotations[samp.sample_id] = HumanAnnotationRecord(
                sample_id=samp.sample_id,
                annotator_id="reviewer_alpha",
                review_status=ReviewStatus.VERIFIED,
                objects=[
                    AnnotatedBBox(
                        label=TargetObjectLabel.PHONE,
                        bbox=(20, 30, 80, 110),
                        is_hard_negative=False,
                    )
                ],
                quality_pass=True,
            )

    # 2. Build versioned dataset
    manifest = pipeline.build_dataset(
        version="v1.0.0",
        samples=samples,
        annotations=annotations,
        split_ratio=(0.60, 0.20, 0.20),
    )

    assert manifest.total_samples == 6
    assert len(manifest.provenance_hash) > 0
    ds_dir = datasets_dir / "dataset_v1_0_0"
    assert (ds_dir / "data.yaml").exists()
    assert (ds_dir / "dataset_manifest.json").exists()

    # Verify physical assets and labels were generated
    train_labels = list((ds_dir / "train" / "labels").glob("*.txt"))
    assert len(train_labels) > 0
    # Read first label
    first_label_txt = train_labels[0].read_text(encoding="utf-8")
    assert first_label_txt.startswith("0 ")  # class 0 = phone

    # 3. Validate prerequisites & run dry-run training
    config = TrainingConfig(
        base_model="yolov8n.pt",
        dataset_version="v1.0.0",
        model_type="object_detector",
        epochs=10,
        batch_size=8,
    )

    valid, msg = pipeline.validate_training_prerequisites(config)
    assert valid is True

    synthetic_metrics = ModelMetrics(
        precision=0.91,
        recall=0.88,
        f1_score=0.895,
        false_positive_rate=0.03,
        false_negative_rate=0.12,
        cpu_latency_ms=25.0,
        hard_negative_accuracy=0.94,
    )
    run_summary = pipeline.execute_dry_run_training(config, synthetic_eval_metrics=synthetic_metrics)
    assert run_summary.status == "DRY_RUN_VALIDATED"
    assert run_summary.regression_passed is True

    # 4. Evaluate and register candidate in model registry
    meta, eval_report = pipeline.evaluate_and_register_candidate(
        summary=run_summary,
        version="v1.0.0",
        model_name="yolo_proctor",
    )
    assert meta.status == ModelStatus.CANDIDATE
    assert meta.model_id == "yolo_proctor_v1_0_0"

    # Promote to active
    success, p_msg = pipeline.registry.evaluate_and_promote(meta.model_id)
    assert success is True
    active = pipeline.registry.get_active_model("object_detector")
    assert active is not None
    assert active.model_id == meta.model_id
    assert active.status == ModelStatus.ACTIVE


def test_training_pipeline_missing_dataset_fails_validation(tmp_path: Path):
    """Verify that training fails gracefully when dataset does not exist."""
    pipeline = TrainingPipeline(
        inbox_dir=tmp_path / "inbox",
        datasets_dir=tmp_path / "datasets",
        registry_dir=tmp_path / "registry",
        runs_dir=tmp_path / "runs",
        regression_cases_path=Path("training/regression/cases.json"),
    )
    config = TrainingConfig(base_model="yolov8n.pt", dataset_version="v99.9.9")
    valid, msg = pipeline.validate_training_prerequisites(config)
    assert valid is False
    assert "not found" in msg

    with pytest.raises(ValueError, match="Training prerequisite check failed"):
        pipeline.execute_dry_run_training(config)
