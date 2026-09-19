"""Validation tooling for WUB proctoring dataset directory structure, metadata, and partition leakage."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from tools.dataset.schema import (
    BoundingBoxAnnotation,
    DatasetSplit,
    SampleMetadata,
    WUBDatasetManifest,
    WUBExamScenario,
)

LOGGER = logging.getLogger("validate_dataset")


def validate_dataset_directory(dataset_dir: str | Path) -> dict[str, Any]:
    """Inspect and validate a dataset directory structure and manifest.

    Expected layout:
        <dataset_dir>/
        ├── manifest.json
        ├── train/
        ├── val/
        └── test/
    """
    root = Path(dataset_dir).resolve()
    report: dict[str, Any] = {
        "dataset_path": str(root),
        "exists": root.is_dir(),
        "splits_present": {},
        "manifest_valid": False,
        "sample_count": 0,
        "leakage_violations": [],
        "validation_dataset_status": "NOT AVAILABLE",
        "model_domain_quality": "NOT VALIDATED",
        "errors": [],
    }

    if not root.is_dir():
        report["errors"].append(f"Dataset root directory does not exist: {root}")
        return report

    for split in DatasetSplit:
        split_dir = root / split.value
        report["splits_present"][split.value] = split_dir.is_dir()

    manifest_file = root / "manifest.json"
    if not manifest_file.is_file():
        report["errors"].append("manifest.json not found in dataset root.")
        return report

    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        samples: list[SampleMetadata] = []
        for raw in data.get("samples", []):
            sample = SampleMetadata(
                sample_id=raw["sample_id"],
                participant_pseudonym=raw["participant_pseudonym"],
                split=DatasetSplit(raw["split"]),
                scenario=WUBExamScenario(raw["scenario"]),
                lighting_condition=raw.get("lighting_condition", "STANDARD"),
                camera_model=raw.get("camera_model", "GENERIC_WEBCAM"),
                resolution=tuple(raw.get("resolution", (640, 480))),
                annotations=[
                    BoundingBoxAnnotation(
                        category=a["category"],
                        bbox=tuple(a["bbox"]),
                        is_permitted=a.get("is_permitted", False),
                        attributes=a.get("attributes", {}),
                    )
                    for a in raw.get("annotations", [])
                ],
                timestamp_offset_seconds=raw.get("timestamp_offset_seconds", 0.0),
            )
            samples.append(sample)

        manifest = WUBDatasetManifest(
            dataset_name=data.get("dataset_name", "Unknown"),
            version=data.get("version", "1.0"),
            validation_dataset_status=data.get("validation_dataset_status", "NOT AVAILABLE"),
            model_domain_quality=data.get("model_domain_quality", "NOT VALIDATED"),
            samples=samples,
        )

        report["manifest_valid"] = True
        report["sample_count"] = len(samples)
        report["validation_dataset_status"] = manifest.validation_dataset_status
        report["model_domain_quality"] = manifest.model_domain_quality

        leakage = manifest.check_split_leakage()
        report["leakage_violations"] = leakage
        if leakage:
            report["errors"].extend(leakage)

    except Exception as exc:
        report["errors"].append(f"Manifest parsing failed: {exc}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate WUB proctoring dataset partition and schema.")
    parser.add_argument("dataset_dir", help="Path to dataset directory")
    args = parser.parse_args()

    results = validate_dataset_directory(args.dataset_dir)
    print(json.dumps(results, indent=2))
    if results["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
