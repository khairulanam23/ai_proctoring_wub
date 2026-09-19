"""Targeted tests for WUB dataset schema, split leakage detection, and directory validation."""

import json
from pathlib import Path

import pytest

from tools.dataset.schema import (
    BoundingBoxAnnotation,
    DatasetSplit,
    SampleMetadata,
    WUBDatasetManifest,
    WUBExamScenario,
)
from tools.dataset.validate_dataset import validate_dataset_directory


def test_wub_dataset_manifest_schema() -> None:
    """Verify manifest records sample metadata and honest NOT AVAILABLE / NOT VALIDATED status."""
    sample = SampleMetadata(
        sample_id="sample_001",
        participant_pseudonym="STUDENT_HASH_A1",
        split=DatasetSplit.TRAIN,
        scenario=WUBExamScenario.PERMITTED_CALCULATOR_USE,
        lighting_condition="DAYLIGHT",
        camera_model="Integrated HD Webcam",
        resolution=(1280, 720),
        annotations=[
            BoundingBoxAnnotation(category="CALCULATOR", bbox=(100, 100, 200, 220), is_permitted=True)
        ],
    )

    manifest = WUBDatasetManifest(samples=[sample])
    assert manifest.validation_dataset_status == "NOT AVAILABLE"
    assert manifest.model_domain_quality == "NOT VALIDATED"

    d = manifest.to_dict()
    assert d["total_samples"] == 1
    assert d["samples"][0]["scenario"] == "PERMITTED_CALCULATOR_USE"


def test_split_leakage_detection() -> None:
    """Verify that participant appearing across train and test splits is caught as leakage."""
    manifest = WUBDatasetManifest(
        samples=[
            SampleMetadata(
                sample_id="s1",
                participant_pseudonym="STUDENT_LEAK_01",
                split=DatasetSplit.TRAIN,
                scenario=WUBExamScenario.DIGITAL_NORMAL_WORKING,
                lighting_condition="STANDARD",
                camera_model="CAM1",
                resolution=(640, 480),
            ),
            SampleMetadata(
                sample_id="s2",
                participant_pseudonym="STUDENT_LEAK_01",
                split=DatasetSplit.TEST,
                scenario=WUBExamScenario.ACTIVE_PHONE_COMMUNICATION,
                lighting_condition="STANDARD",
                camera_model="CAM1",
                resolution=(640, 480),
            ),
        ]
    )

    leakage = manifest.check_split_leakage()
    assert len(leakage) == 1
    assert "Leakage between TRAIN and TEST" in leakage[0]


def test_validate_dataset_directory(tmp_path: Path) -> None:
    """Verify validation tool checks directory structure and manifest content."""
    dataset_dir = tmp_path / "wub_eval_dataset"
    dataset_dir.mkdir()
    (dataset_dir / "train").mkdir()
    (dataset_dir / "val").mkdir()
    (dataset_dir / "test").mkdir()

    manifest = WUBDatasetManifest(
        samples=[
            SampleMetadata(
                sample_id="sample_test_01",
                participant_pseudonym="STUDENT_CLEAN_01",
                split=DatasetSplit.TEST,
                scenario=WUBExamScenario.WRITTEN_NORMAL_HANDWRITING,
                lighting_condition="STANDARD",
                camera_model="WEBCAM_A",
                resolution=(640, 480),
            )
        ]
    )
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    report = validate_dataset_directory(dataset_dir)
    assert report["manifest_valid"] is True
    assert report["sample_count"] == 1
    assert len(report["leakage_violations"]) == 0
    assert report["validation_dataset_status"] == "NOT AVAILABLE"
    assert report["model_domain_quality"] == "NOT VALIDATED"
