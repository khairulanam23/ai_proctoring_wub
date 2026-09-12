# Phase 0 Audit: Cleanup Candidates

In strict adherence to the **Global Engineering Rules (Rule 1 & 2)** and the **Cleanup Policy**, no source files, model weights, or packages are deleted during Phase 0. Instead, candidates are cataloged with their verified file path, reason for candidacy, references across the codebase, runtime impact, and risk assessment.

---

## 1. Duplicate Model Weights

| Candidate | Reason | References Found | Runtime References | Test References | Replacement | Risk | Recommended Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolo11n.pt` (root directory) | Exact binary duplicate of `models/yolo11n.pt` (SHA256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`). Size: 5.4 MB. | 0 explicit references (root file). `models/yolo11n.pt` is referenced in `config.py` and `api.py`. | None. `proctoring.config.SessionConfig.object_detector_model` defaults to `"models/yolo11n.pt"`. | None. | `models/yolo11n.pt` | Low. Root file is an accidental artifact from an Ultralytics download. | Move to archive or remove in Phase 1 after verifying all configs point to `models/yolo11n.pt`. |

---

## 2. Unreferenced Model Weights

| Candidate | Reason | References Found | Runtime References | Test References | Replacement | Risk | Recommended Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `weights/clip/ViT-B-32.pt` | Large CLIP model weights (338 MB). Completely unreferenced in source code. | 0 occurrences in `proctoring/`, `tests/`, `tools/`. | None. No module imports or loads `ViT-B-32.pt`. | None. | None required (unused). | Low. | Verify no external training pipeline expects this weight before archiving/removing. |

---

## 3. Virtual Environment Conflicting Packages

| Candidate | Reason | References Found | Runtime References | Test References | Replacement | Risk | Recommended Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Multiple OpenCV distributions in `.venv` (`opencv-python`, `opencv-python-headless`, `opencv-contrib-python`) | Multiple OpenCV pip packages installed concurrently cause namespace collisions, inconsistent video backend loading, and undefined C-extension symbol lookups. | `import cv2` | Runtime uses whatever `.so` was imported first (currently 5.0.0). | Tests use `cv2`. | Standardize on a single clean OpenCV build. | Medium. Cleaning virtual environment must be tested against existing tests. | Standardize in Phase 1 dependencies audit. |
| Missing production dependencies in `pyproject.toml` (`fastapi`, `uvicorn`, `pydantic`) | `proctoring/integration/api.py` and `service.py` require `fastapi`, `uvicorn`, and `pydantic`, but they are omitted from `pyproject.toml` dependencies. | `api.py`, `schemas.py`, `service.py` | API entrypoints fail to start without them if installed via `pip install .`. | `tests/integration/test_api.py` | Add explicitly to `pyproject.toml`. | Low. Required for deployment correctness. | Add to `pyproject.toml` in Phase 1. |

---

## 4. Scaffolding / Dead Analysis Pathways

| Candidate | Reason | References Found | Runtime References | Test References | Replacement | Risk | Recommended Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Disconnected `PaperDetector` output | `PaperDetector.detect()` runs on CPU every frame, but `BehaviourObserver.map_to_events()` never maps `obs.paper_analysis` to any events or incidents. | `engine_stages.py:466` | Output is computed, attached to `FrameObservation`, but discarded by event observer. | None testing end-to-end paper events. | Wire into `BehaviourObserver` in Phase 1/3. | None. Do not delete; connect consumer. | Connect consumer in Phase 1. |
| Disconnected `HandKinematics` output | `HandAnalyzer._update_kinematics()` computes velocities and trajectory features, but `kinematics` is never consumed by `observer.py`, `aggregator.py`, or `engine.py`. | `proctoring/analysis/hands.py` | Computed internally but never inspected outside `hands.py`. | None. | Wire into temporal tracking and gesture reasoning in Phase 2/3. | None. Do not delete; connect consumer. | Connect consumer in Phase 2. |
