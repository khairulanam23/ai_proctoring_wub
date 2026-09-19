"""Memory-safe, targeted Phase 3 acceptance test suite.

Validates all Phase 3 requirements (Section 4 A-G):
- 4.A Controlled taxonomy (trainable classes vs non-trainable review categories)
- 4.B Human review states (PENDING, CONFIRMED, CORRECTED, REJECTED, UNCERTAIN & eligibility filtering)
- 4.C Provenance tracking & audit trail
- 4.D SHA-256 integrity verification
- 4.E Session-group splitting (preventing data leakage across partitions)
- 4.F Biometric-data exclusion
- 4.G Candidate training pipeline & production model protection
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from proctoring.learning.annotations import (
    NON_TRAINABLE_LABELS,
    YOLO_LABEL_MAP,
    AnnotatedBBox,
    HumanAnnotationRecord,
    ReviewStatus,
    TargetObjectLabel,
    get_yolo_class_names,
)
from proctoring.learning.datasets import DatasetManager
from proctoring.learning.inbox import FlagReason, InboxSample, TrainingInboxManager
from proctoring.learning.pipeline import (
    TrainingConfig,
    TrainingPipeline,
)
from proctoring.learning.registry import (
    ModelMetrics,
    ModelStatus,
)


# ==============================================================================
# 4.A Controlled Taxonomy
# ==============================================================================
def test_phase3_a_controlled_taxonomy(tmp_path: Path):
    """Verify supported trainable taxonomy contains all 15 intended Phase 3 classes

    and non-trainable categories remain separate and excluded from training.
    """
    expected_trainable = {
        "phone",
        "scientific_calculator",
        "power_bank",
        "notebook",
        "book",
        "pencil_case",
        "id_card",
        "earbuds",
        "headphones",
        "pen",
        "pencil",
        "paper",
        "keyboard",
        "mouse",
        "other",
    }

    # Verify all 15 classes exist in TargetObjectLabel and YOLO_LABEL_MAP
    mapped_classes = set(get_yolo_class_names().values())
    assert mapped_classes == expected_trainable, (
        f"Mismatch in trainable YOLO classes: {mapped_classes ^ expected_trainable}"
    )
    assert len(YOLO_LABEL_MAP) == 15

    # Verify non-trainable categories remain strictly separate
    expected_non_trainable = {
        TargetObjectLabel.UNCERTAIN,
        TargetObjectLabel.NOT_A_RELEVANT_OBJECT,
        TargetObjectLabel.HARD_NEGATIVE_OBJECT,
    }
    assert expected_non_trainable == NON_TRAINABLE_LABELS

    # Verify non-trainable categories are NOT in YOLO_LABEL_MAP
    for non_trainable in expected_non_trainable:
        assert non_trainable not in YOLO_LABEL_MAP

    # Verify that converting an annotation with non-trainable categories generates NO YOLO boxes
    ann = HumanAnnotationRecord(
        sample_id="test_samp_non_trainable",
        annotator_id="reviewer_1",
        review_status=ReviewStatus.CONFIRMED,
        objects=[
            AnnotatedBBox(label=TargetObjectLabel.NOT_A_RELEVANT_OBJECT, bbox=(10, 10, 50, 50)),
            AnnotatedBBox(label=TargetObjectLabel.UNCERTAIN, bbox=(20, 20, 60, 60)),
        ],
    )
    yolo_lines = ann.to_yolo_format(img_width=640, img_height=480)
    assert len(yolo_lines) == 0, "Non-trainable categories must never produce positive training boxes"

    # Verify non-trainable category does NOT appear in manifest.classes
    inbox = TrainingInboxManager(base_dir=tmp_path / "inbox")
    s = inbox.capture_sample(
        image=np.full((50, 50, 3), 120, dtype=np.uint8),
        session_id="sess_tax",
        frame_index=1,
        target_class="other",
        flag_reason=FlagReason.UNCERTAIN_DETECTION,
        confidence=0.5,
    )
    ann_tax = HumanAnnotationRecord(
        sample_id=s.sample_id,
        annotator_id="rev_tax",
        review_status=ReviewStatus.CONFIRMED,
        objects=[
            AnnotatedBBox(label=TargetObjectLabel.NOT_A_RELEVANT_OBJECT, bbox=(5, 5, 25, 25)),
            AnnotatedBBox(label=TargetObjectLabel.HARD_NEGATIVE_OBJECT, bbox=(10, 10, 30, 30)),
        ],
    )
    dataset_mgr = DatasetManager(datasets_root=tmp_path / "datasets_tax")
    m = dataset_mgr.create_versioned_dataset("v1.0.0", [s], {s.sample_id: ann_tax})
    assert "not_a_relevant_object" not in m.classes
    assert "hard_negative_object" not in m.classes
    assert "uncertain" not in m.classes


# ==============================================================================
# 4.B Human Review States and Training Eligibility
# ==============================================================================
def test_phase3_b_human_review_states_and_eligibility(tmp_path: Path):
    """Verify supported review states (PENDING, CONFIRMED, CORRECTED, REJECTED, UNCERTAIN)

    and ensure that rejected/uncertain/pending samples cannot become positive training data.
    """
    # 1. Verify all required enum states exist
    required_states = {"pending", "confirmed", "corrected", "rejected", "uncertain"}
    actual_states = {s.value for s in ReviewStatus}
    assert required_states.issubset(actual_states)

    # 2. Build sample inbox with 5 samples covering all review states
    inbox = TrainingInboxManager(base_dir=tmp_path / "inbox")
    img = np.full((100, 100, 3), 128, dtype=np.uint8)

    samples: list[InboxSample] = []
    annotations: dict[str, HumanAnnotationRecord] = {}

    status_map = [
        ("s_pending", ReviewStatus.PENDING, True),
        ("s_confirmed", ReviewStatus.CONFIRMED, True),
        ("s_corrected", ReviewStatus.CORRECTED, True),
        ("s_rejected", ReviewStatus.REJECTED, False),
        ("s_uncertain", ReviewStatus.UNCERTAIN, False),
    ]

    for sid, status, q_pass in status_map:
        samp = inbox.capture_sample(
            image=img,
            session_id=f"session_{sid}",
            frame_index=1,
            target_class="phone",
            flag_reason=FlagReason.UNCERTAIN_DETECTION,
            confidence=0.7,
            bbox=(10, 10, 80, 80),
        )
        samples.append(samp)
        annotations[samp.sample_id] = HumanAnnotationRecord(
            sample_id=samp.sample_id,
            annotator_id="reviewer_1",
            review_status=status,
            objects=[AnnotatedBBox(label=TargetObjectLabel.PHONE, bbox=(10, 10, 80, 80))],
            quality_pass=q_pass,
        )

    dataset_mgr = DatasetManager(datasets_root=tmp_path / "datasets")
    manifest = dataset_mgr.create_versioned_dataset(
        version="v1.0.0",
        samples=samples,
        annotations=annotations,
    )

    # Only CONFIRMED and CORRECTED (with quality_pass=True) may be included
    assert manifest.total_samples == 2
    included_sessions = set(manifest.session_ids)
    assert "session_s_confirmed" in included_sessions
    assert "session_s_corrected" in included_sessions
    assert "session_s_pending" not in included_sessions
    assert "session_s_rejected" not in included_sessions
    assert "session_s_uncertain" not in included_sessions


# ==============================================================================
# 4.C Provenance Tracking
# ==============================================================================
def test_phase3_c_provenance_retention(tmp_path: Path):
    """Verify exported dataset retains full source provenance and valid provenance hash."""
    inbox = TrainingInboxManager(base_dir=tmp_path / "inbox")
    img = np.full((120, 120, 3), 150, dtype=np.uint8)

    sample = inbox.capture_sample(
        image=img,
        session_id="session_exam_999",
        frame_index=142,
        target_class="scientific_calculator",
        flag_reason=FlagReason.ANOMALOUS_GEOMETRY,
        confidence=0.62,
        bbox=(20, 20, 90, 110),
        metadata={"incident_id": "inc_exam_999_001", "room": "lab_4"},
    )

    ann = HumanAnnotationRecord(
        sample_id=sample.sample_id,
        annotator_id="auditor_lead",
        review_status=ReviewStatus.CONFIRMED,
        objects=[
            AnnotatedBBox(
                label=TargetObjectLabel.SCIENTIFIC_CALCULATOR,
                bbox=(20, 20, 90, 110),
            )
        ],
        quality_pass=True,
        metadata={"audit_checked": True},
    )

    dataset_mgr = DatasetManager(datasets_root=tmp_path / "datasets")
    manifest = dataset_mgr.create_versioned_dataset(
        version="v2.0.0",
        samples=[sample],
        annotations={sample.sample_id: ann},
        inbox_base_dir=tmp_path / "inbox",
    )

    assert manifest.total_samples == 1
    assert "session_exam_999" in manifest.session_ids
    assert len(manifest.provenance_hash) == 64  # Valid SHA-256 hexdigest length
    assert "scientific_calculator" in manifest.classes

    # Ensure metadata preserves source incident details
    assert sample.metadata["incident_id"] == "inc_exam_999_001"
    assert sample.frame_index == 142
    assert sample.session_id == "session_exam_999"

    # Ensure exported dataset contains sample-level provenance records and provenance.json
    assert len(manifest.sample_records) == 1
    rec = manifest.sample_records[0]
    assert rec["sample_id"] == sample.sample_id
    assert rec["session_id"] == "session_exam_999"
    assert rec["frame_index"] == 142
    assert rec["metadata"]["incident_id"] == "inc_exam_999_001"
    assert rec["annotator_id"] == "auditor_lead"
    assert rec["review_status"] == "confirmed"
    assert len(rec["image_sha256"]) == 64

    prov_json = tmp_path / "datasets" / "dataset_v2_0_0" / "provenance.json"
    assert prov_json.exists(), "provenance.json must be persisted in exported dataset directory"


# ==============================================================================
# 4.D SHA-256 Integrity Verification
# ==============================================================================
def test_phase3_d_sha256_integrity(tmp_path: Path):
    """Verify evidence files receive SHA-256, hash matches recalculation, and mutation is detected."""
    evidence_file = tmp_path / "evidence_sample_01.jpg"
    content = b"TEST_IMAGE_BINARY_DATA_PAYLOAD_FOR_SHA256_TESTING"
    evidence_file.write_bytes(content)

    # 1. Compute SHA-256 hash
    hasher = hashlib.sha256()
    hasher.update(content)
    expected_hash = hasher.hexdigest()

    # 2. Store in metadata
    meta = {
        "file_path": str(evidence_file),
        "sha256": expected_hash,
        "size_bytes": len(content),
    }

    # 3. Recalculate and assert match
    actual_hash = hashlib.sha256(evidence_file.read_bytes()).hexdigest()
    assert actual_hash == meta["sha256"]

    # 4. Modify file and assert mismatch
    tampered_file = tmp_path / "evidence_sample_01.jpg"
    tampered_file.write_bytes(b"TAMPERED_PAYLOAD_DIFFERENT_HASH")
    tampered_hash = hashlib.sha256(tampered_file.read_bytes()).hexdigest()
    assert tampered_hash != meta["sha256"], "Integrity check must fail when file content is modified"

    # 5. Verify DatasetManager SHA-256 integrity verification and tamper detection on exported dataset
    inbox = TrainingInboxManager(base_dir=tmp_path / "inbox_sha")
    sample = inbox.capture_sample(
        image=np.full((80, 80, 3), 100, dtype=np.uint8),
        session_id="session_sha_test",
        frame_index=1,
        target_class="phone",
        flag_reason=FlagReason.FALSE_NEGATIVE,
        confidence=0.9,
    )
    ann = HumanAnnotationRecord(
        sample_id=sample.sample_id,
        annotator_id="rev_sha",
        review_status=ReviewStatus.CONFIRMED,
        objects=[AnnotatedBBox(label=TargetObjectLabel.PHONE, bbox=(10, 10, 50, 50))],
    )
    dataset_mgr = DatasetManager(datasets_root=tmp_path / "datasets_sha")
    manifest = dataset_mgr.create_versioned_dataset(
        version="v1.0.0",
        samples=[sample],
        annotations={sample.sample_id: ann},
        inbox_base_dir=tmp_path / "inbox_sha",
    )
    # Check integrity passes initially
    valid, errors = dataset_mgr.verify_dataset_integrity("v1.0.0")
    assert valid is True
    assert len(errors) == 0

    # Modify the exported evidence image file
    exported_img = tmp_path / "datasets_sha" / "dataset_v1_0_0" / manifest.sample_records[0]["image_file"]
    exported_img.write_bytes(b"CORRUPTED_TAMPERED_BYTES")

    # Check integrity detects tampering
    valid_after_tamper, tamper_errors = dataset_mgr.verify_dataset_integrity("v1.0.0")
    assert valid_after_tamper is False
    assert len(tamper_errors) > 0
    assert "Integrity violation" in tamper_errors[0]


# ==============================================================================
# 4.E Session-Group Splitting (Data Leakage Prevention)
# ==============================================================================
def test_phase3_e_session_group_splitting(tmp_path: Path):
    """Verify that train/val/test splitting is strictly session-grouped."""
    inbox = TrainingInboxManager(base_dir=tmp_path / "inbox")
    img = np.full((100, 100, 3), 100, dtype=np.uint8)

    samples = []
    annotations = {}
    sessions = ["session_A", "session_B", "session_C", "session_D"]

    for sess in sessions:
        # Multiple frames per session
        for f_idx in (10, 20, 30):
            samp = inbox.capture_sample(
                image=img,
                session_id=sess,
                frame_index=f_idx,
                target_class="book",
                flag_reason=FlagReason.FALSE_NEGATIVE,
                confidence=0.8,
                bbox=(10, 10, 60, 60),
            )
            samples.append(samp)
            annotations[samp.sample_id] = HumanAnnotationRecord(
                sample_id=samp.sample_id,
                annotator_id="reviewer_1",
                review_status=ReviewStatus.CONFIRMED,
                objects=[AnnotatedBBox(label=TargetObjectLabel.BOOK, bbox=(10, 10, 60, 60))],
                quality_pass=True,
            )

    dataset_mgr = DatasetManager(datasets_root=tmp_path / "datasets")
    manifest = dataset_mgr.create_versioned_dataset(
        version="v3.0.0",
        samples=samples,
        annotations=annotations,
        split_ratio=(0.50, 0.25, 0.25),
        inbox_base_dir=tmp_path / "inbox",
    )

    dataset_dir = tmp_path / "datasets" / "dataset_v3_0_0"

    train_files = {p.stem for p in (dataset_dir / "train" / "labels").glob("*.txt")}
    val_files = {p.stem for p in (dataset_dir / "val" / "labels").glob("*.txt")}
    test_files = {p.stem for p in (dataset_dir / "test" / "labels").glob("*.txt")}

    sample_session_map = {s.sample_id: s.session_id for s in samples}

    train_sessions = {sample_session_map[sid] for sid in train_files}
    val_sessions = {sample_session_map[sid] for sid in val_files}
    test_sessions = {sample_session_map[sid] for sid in test_files}

    # Strict partition isolation assertion: No session can appear in more than one partition
    assert train_sessions.isdisjoint(val_sessions), "Train and Val partitions must not share sessions"
    assert train_sessions.isdisjoint(test_sessions), "Train and Test partitions must not share sessions"
    assert val_sessions.isdisjoint(test_sessions), "Val and Test partitions must not share sessions"


# ==============================================================================
# 4.F Biometric-Data Exclusion
# ==============================================================================
def test_phase3_f_biometric_data_exclusion(tmp_path: Path):
    """Verify that dataset exports, manifests, and annotation records exclude biometric identity vectors."""
    inbox = TrainingInboxManager(base_dir=tmp_path / "inbox")
    img = np.full((100, 100, 3), 128, dtype=np.uint8)

    # Intentionally inject raw biometric vectors into sample metadata
    sample = inbox.capture_sample(
        image=img,
        session_id="session_candidate_10",
        frame_index=5,
        target_class="phone",
        flag_reason=FlagReason.UNCERTAIN_DETECTION,
        confidence=0.75,
        metadata={
            "candidate_id": "c_10",
            "exam_name": "Midterm",
            "embedding": [0.123, 0.456, 0.789],
            "face_vector": [0.99, 0.88, 0.77],
            "reference_template": [0.5, 0.5, 0.5],
            "sface_vector": [0.11, 0.22],
        },
    )

    ann = HumanAnnotationRecord(
        sample_id=sample.sample_id,
        annotator_id="reviewer_1",
        review_status=ReviewStatus.CONFIRMED,
        objects=[AnnotatedBBox(label=TargetObjectLabel.PHONE, bbox=(20, 20, 80, 80))],
    )

    dataset_mgr = DatasetManager(datasets_root=tmp_path / "datasets")
    manifest = dataset_mgr.create_versioned_dataset(
        version="v4.0.0",
        samples=[sample],
        annotations={sample.sample_id: ann},
    )

    manifest_dict = manifest.to_dict()
    manifest_str = json.dumps(manifest_dict)
    prov_str = (tmp_path / "datasets" / "dataset_v4_0_0" / "provenance.json").read_text()

    # Assert no face embedding vectors or biometric template fields exist in manifest or provenance
    forbidden_biometric_keys = [
        "embedding",
        "embeddings",
        "reference_template",
        "reference_templates",
        "face_vector",
        "biometric_vector",
        "sface_vector",
    ]

    for key in forbidden_biometric_keys:
        assert key not in manifest_dict, f"Biometric key '{key}' found in manifest"
        assert f'"{key}"' not in manifest_str, f"Biometric field '{key}' leaked into manifest JSON"
        assert f'"{key}"' not in prov_str, f"Biometric field '{key}' leaked into provenance JSON"


# ==============================================================================
# 4.G Candidate Training Pipeline & Production Model Protection
# ==============================================================================
def test_phase3_g_candidate_training_pipeline_and_production_protection(tmp_path: Path):
    """Verify candidate training workflow can initialize and register candidates,

    while keeping production models frozen and protected from automated replacement.
    """
    pipeline = TrainingPipeline(
        inbox_dir=tmp_path / "inbox",
        datasets_dir=tmp_path / "datasets",
        registry_dir=tmp_path / "registry",
        runs_dir=tmp_path / "runs",
        regression_cases_path=Path("training/regression/cases.json"),
    )

    # 1. Establish an existing ACTIVE production model
    prod_weights = tmp_path / "production_yolo.pt"
    prod_weights.write_text("ACTIVE_PRODUCTION_WEIGHTS")

    prod_metrics = ModelMetrics(
        precision=0.94,
        recall=0.92,
        f1_score=0.93,
        cpu_latency_ms=22.0,
        hard_negative_accuracy=0.96,
    )
    prod_meta = pipeline.registry.register_model(
        model_name="yolo_production",
        version="v1.0.0",
        model_type="object_detector",
        artifact_path=prod_weights,
        metrics=prod_metrics,
    )
    pipeline.registry.evaluate_and_promote(prod_meta.model_id)
    active_prod = pipeline.registry.get_active_model("object_detector")
    assert active_prod is not None
    assert active_prod.model_id == "yolo_production_v1_0_0"

    # 2. Build a tiny valid dataset
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    sample = pipeline.capture_evidence(
        image=img,
        session_id="sess_candidate_1",
        frame_index=1,
        target_class="phone",
        flag_reason=FlagReason.UNCERTAIN_DETECTION,
        confidence=0.6,
        bbox=(10, 10, 50, 90),
    )
    ann = HumanAnnotationRecord(
        sample_id=sample.sample_id,
        annotator_id="rev_1",
        review_status=ReviewStatus.CONFIRMED,
        objects=[AnnotatedBBox(label=TargetObjectLabel.PHONE, bbox=(10, 10, 50, 90))],
        quality_pass=True,
    )
    pipeline.build_dataset(
        version="v1.0.1",
        samples=[sample],
        annotations={sample.sample_id: ann},
    )

    # 3. Validate prerequisites for candidate training
    config = TrainingConfig(
        base_model="yolov8n.pt",
        dataset_version="v1.0.1",
        model_type="object_detector",
    )
    valid, msg = pipeline.validate_training_prerequisites(config)
    assert valid is True

    # 4. Execute dry-run candidate training without heavy GPU compute
    candidate_metrics = ModelMetrics(
        precision=0.90,
        recall=0.88,
        f1_score=0.89,
        cpu_latency_ms=24.0,
        hard_negative_accuracy=0.92,
    )
    run_summary = pipeline.execute_dry_run_training(
        config=config, synthetic_eval_metrics=candidate_metrics
    )
    assert run_summary.status == "DRY_RUN_VALIDATED"

    # 5. Register candidate in model registry
    cand_meta, eval_report = pipeline.evaluate_and_register_candidate(
        summary=run_summary,
        version="v1.0.1",
        model_name="yolo_candidate",
    )

    # 6. Verify candidate is registered as CANDIDATE and production model is NOT replaced
    assert cand_meta.status == ModelStatus.CANDIDATE
    current_active = pipeline.registry.get_active_model("object_detector")
    assert current_active is not None
    assert current_active.model_id == "yolo_production_v1_0_0", (
        "Production active model must remain frozen and not replaced by candidate"
    )


# ==============================================================================
# Phase 3 Smoke Validation (Explicitly separated from fast unit suite)
# ==============================================================================
@pytest.mark.smoke
def test_phase3_real_chain_smoke_validation():
    """Explicit smoke validation executing the real camera/frame input -> AI inference ->

    event qualification -> evidence generation -> persistence chain.
    Requires authentic candidate camera frames and live model stack.
    """
    from scripts.validate_phase3_real_chain import main as run_real_chain

    exit_code = run_real_chain()
    assert exit_code == 0, "Phase 3 real-world chain validation failed"

