"""Tests for Phase 3: Controlled taxonomy, review separation, dataset pipeline, and training harness."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from proctoring.learning.annotations import (
    AnnotatedBBox,
    HumanAnnotationRecord,
    NON_TRAINABLE_LABELS,
    YOLO_LABEL_MAP,
    ReviewStatus,
    TargetObjectLabel,
    get_yolo_class_names,
)
from tools.dataset.export_reviewed_dataset import build_reviewed_dataset
from tools.train_object_detector import run_training_pipeline, validate_dataset


def test_controlled_label_taxonomy_and_yolo_mapping():
    """Verify that the Phase 3 controlled taxonomy is enforced and non-trainable classes are excluded."""
    # 1. Verify positive object classes are mapped
    positive_classes = [
        TargetObjectLabel.PHONE,
        TargetObjectLabel.SCIENTIFIC_CALCULATOR,
        TargetObjectLabel.POWER_BANK,
        TargetObjectLabel.NOTEBOOK,
        TargetObjectLabel.BOOK,
        TargetObjectLabel.PENCIL_CASE,
        TargetObjectLabel.ID_CARD,
        TargetObjectLabel.EARBUDS,
        TargetObjectLabel.HEADPHONES,
        TargetObjectLabel.PEN,
        TargetObjectLabel.PENCIL,
        TargetObjectLabel.PAPER,
        TargetObjectLabel.KEYBOARD,
        TargetObjectLabel.MOUSE,
        TargetObjectLabel.OTHER,
    ]
    for p_class in positive_classes:
        assert p_class in YOLO_LABEL_MAP, f"Expected {p_class} to be in YOLO_LABEL_MAP"

    # 2. Verify non-trainable classes are NOT in YOLO_LABEL_MAP
    assert TargetObjectLabel.UNCERTAIN in NON_TRAINABLE_LABELS
    assert TargetObjectLabel.UNCERTAIN not in YOLO_LABEL_MAP
    assert TargetObjectLabel.NOT_A_RELEVANT_OBJECT in NON_TRAINABLE_LABELS
    assert TargetObjectLabel.NOT_A_RELEVANT_OBJECT not in YOLO_LABEL_MAP

    # 3. Verify to_yolo_format ignores non-trainable labels
    record = HumanAnnotationRecord(
        sample_id="test_samp_01",
        annotator_id="admin_01",
        review_status=ReviewStatus.CONFIRMED,
        objects=[
            AnnotatedBBox(label=TargetObjectLabel.PHONE, bbox=(10, 20, 100, 200)),
            AnnotatedBBox(label=TargetObjectLabel.UNCERTAIN, bbox=(50, 60, 150, 250)),
            AnnotatedBBox(label=TargetObjectLabel.NOT_A_RELEVANT_OBJECT, bbox=(0, 0, 50, 50)),
        ],
    )
    yolo_lines = record.to_yolo_format(img_width=640, img_height=480)
    # Exactly one valid bounding box should be output (the phone)
    assert len(yolo_lines) == 1
    assert yolo_lines[0].startswith(str(YOLO_LABEL_MAP[TargetObjectLabel.PHONE]))


def test_dataset_export_filters_unreviewed_and_uncertain_records():
    """Verify that only CONFIRMED and CORRECTED records are exported to the training dataset."""
    records = [
        {
            "sample_id": "samp_confirmed_01",
            "session_id": "sess_01",
            "original_event_type": "PHONE_DETECTED",
            "original_confidence": 0.89,
            "original_detector": "yolo11n",
            "review_status": "CONFIRMED",
            "reviewed_label": "phone",
            "bbox": [100, 100, 300, 300],
        },
        {
            "sample_id": "samp_corrected_02",
            "session_id": "sess_02",
            "original_event_type": "PHONE_DETECTED",
            "original_confidence": 0.85,
            "original_detector": "yolo11n",
            "review_status": "CORRECTED",
            "reviewed_label": "scientific_calculator",
            "bbox": [80, 120, 220, 280],
        },
        {
            "sample_id": "samp_pending_03",
            "session_id": "sess_03",
            "original_event_type": "PHONE_DETECTED",
            "original_confidence": 0.70,
            "review_status": "PENDING",
            "reviewed_label": "phone",
        },
        {
            "sample_id": "samp_uncertain_04",
            "session_id": "sess_04",
            "original_event_type": "PHONE_DETECTED",
            "original_confidence": 0.65,
            "review_status": "UNCERTAIN",
            "reviewed_label": "uncertain",
        },
        {
            "sample_id": "samp_rejected_05",
            "session_id": "sess_05",
            "original_event_type": "PHONE_DETECTED",
            "original_confidence": 0.55,
            "review_status": "REJECTED",
            "reviewed_label": "not_a_relevant_object",
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = build_reviewed_dataset(
            records=records,
            output_dir=tmpdir,
            dataset_version="v0.3.0",
        )

        # Only samples 1 and 2 should be included
        assert manifest["total_samples"] == 2
        exported_sample_ids = [s["sample_id"] for s in manifest["samples"]]
        assert "samp_confirmed_01" in exported_sample_ids
        assert "samp_corrected_02" in exported_sample_ids
        assert "samp_pending_03" not in exported_sample_ids
        assert "samp_uncertain_04" not in exported_sample_ids
        assert "samp_rejected_05" not in exported_sample_ids

        # Verify class distribution
        assert manifest["class_distribution"]["phone"] == 1
        assert manifest["class_distribution"]["scientific_calculator"] == 1

        # Verify data.yaml was created
        ds_dir = Path(tmpdir) / "dataset_v0_3_0"
        assert (ds_dir / "data.yaml").exists()
        assert (ds_dir / "dataset_manifest.json").exists()


def test_session_grouped_partition_prevents_leakage():
    """Verify that multiple frames from the same examination session are never split across train/test."""
    records = []
    # Session 1: 3 frames
    for i in range(3):
        records.append({
            "sample_id": f"sess1_frame_{i}",
            "session_id": "session_alpha",
            "review_status": "CONFIRMED",
            "reviewed_label": "phone",
        })
    # Session 2: 3 frames
    for i in range(3):
        records.append({
            "sample_id": f"sess2_frame_{i}",
            "session_id": "session_beta",
            "review_status": "CORRECTED",
            "reviewed_label": "power_bank",
        })

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = build_reviewed_dataset(
            records=records,
            output_dir=tmpdir,
            dataset_version="v0.3.1",
        )

        # Verify that each session's frames all belong to the SAME split
        session_splits = {}
        for s in manifest["samples"]:
            sample_prefix = s["sample_id"].split("_")[0]
            if sample_prefix not in session_splits:
                session_splits[sample_prefix] = set()
            session_splits[sample_prefix].add(s["split"])

        for sess, splits in session_splits.items():
            assert len(splits) == 1, f"Session {sess} leaked across multiple splits: {splits}"


def test_training_harness_validation_and_candidate_generation():
    """Verify that the training harness validates the dataset and produces an isolated candidate model artifact."""
    # Create a minimal verified dataset
    records = [
        {
            "sample_id": "cand_sample_01",
            "session_id": "sess_candidate_01",
            "original_event_type": "PHONE_DETECTED",
            "review_status": "CONFIRMED",
            "reviewed_label": "phone",
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = build_reviewed_dataset(
            records=records,
            output_dir=tmpdir,
            dataset_version="v0.3.2",
        )
        ds_dir = Path(tmpdir) / "dataset_v0_3_2"

        # Validate dataset
        val = validate_dataset(ds_dir)
        assert val["valid"] is True
        assert val["total_samples"] == 1
        assert "scientific_calculator" in val["insufficient_data_classes"]

        # Run smoke training harness
        runs_dir = Path(tmpdir) / "runs"
        result = run_training_pipeline(
            dataset_dir=ds_dir,
            epochs=1,
            is_smoke_run=True,
            output_base_dir=runs_dir,
        )

        assert result["status"] == "SMOKE / NOT FOR PRODUCTION"
        assert Path(result["weights_path"]).exists()
        assert (Path(result["run_dir"]) / "MODEL_MANIFEST.json").exists()
        assert (Path(result["run_dir"]) / "evaluation_report.json").exists()

        # Read model manifest and verify candidate promotion status
        manifest_data = json.loads((Path(result["run_dir"]) / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
        assert manifest_data["production_promoted"] is False
        assert "Requires explicit developer review" in manifest_data["promotion_boundary"]
