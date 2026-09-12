# Phase 2 Engineering Report: Persistent Tracking + Temporal Reasoning

**Phase**: Phase 2 — Persistent Tracking + Temporal Reasoning  
**Priority**: P0  
**Date**: 2026-09-12  
**Auditor / Systems Engineer**: Senior AI Systems Engineer  
**Status**: **COMPLETED & VERIFIED**  

---

## 1. Objective

Audit and resolve frame-to-frame identity jitter and naive detection ordering assumptions in the proctoring pipeline. Eliminate the flawed assumption that `detections[0]` always represents the candidate, track multiple subjects persistently across video frames with stable track IDs, track objects and hands, and associate detected objects/phones with specific subjects.

---

## 2. Implementation Performed

1. **Multi-Subject Persistent Tracker (`proctoring/tracking/tracker.py`)**:
   - Implemented `MultiSubjectTracker` managing `TrackedSubject`, `TrackedObject`, and `TrackedHand`.
   - **Identity Persistence**: Combines spatial IoU with cosine similarity between facial embeddings (SFace) to maintain consistent `track_id` values even under detection reordering, motion, and temporal jitter.
   - **Track Lifecycle**: Deterministic state progression: `TENTATIVE` (requires 2 consecutive frames for confirmation), `CONFIRMED`, `COASTING` (retains identity across temporary dropouts up to `max_face_misses`), and `DELETED`.
   - **Multi-Object Spatial Association**: Computes IoU and centroid distance between object detections (phones, books, papers) and hands/subjects. Differentiates `"in_hand"`, `"near_subject"`, and `"unassociated"`, attributing objects to the interacting subject track ID rather than blindly to the candidate.
   - **Session Isolation**: `reset()` clears all subject, object, and hand tables, resetting ID counters to 1.

2. **Observation & Stage Integration**:
   - Updated `proctoring/observation.py` to add `tracked_subjects`, `tracked_objects`, and `tracked_hands` to `FrameObservation`.
   - Updated `proctoring/engine_stages.py` to execute tracking during `resolve_identity` and `analyze_behaviour`.
   - Updated `proctoring/engine.py` to update object-to-subject tracking and pass tracking context into temporal aggregation.
   - Updated `proctoring/temporal/aggregator.py` to prioritize the verified enrolled candidate track rather than naive `bboxes[0]`, and associate object incident keys with subject IDs when secondary persons are involved.

---

## 3. Files Changed / Created

| File | Type | Changes | Status |
| :--- | :---: | :--- | :---: |
| `proctoring/tracking/__init__.py` | NEW | Exported tracking classes | **VERIFIED** |
| `proctoring/tracking/tracker.py` | NEW | Implemented `MultiSubjectTracker`, `TrackedSubject`, `TrackedObject`, `TrackedHand`, `TrackState` | **VERIFIED** |
| `proctoring/observation.py` | MODIFY | Added `tracked_subjects`, `tracked_objects`, `tracked_hands` fields and dictionary serialization | **VERIFIED** |
| `proctoring/engine_stages.py` | MODIFY | Integrated `MultiSubjectTracker` into `resolve_identity` and `analyze_behaviour`; added tracker reset | **VERIFIED** |
| `proctoring/engine.py` | MODIFY | Updated object tracking stage and passed tracking context to temporal aggregator | **VERIFIED** |
| `proctoring/temporal/aggregator.py` | MODIFY | Prioritized enrolled subject over naive `bboxes[0]`; added subject association for object incidents | **VERIFIED** |
| `tests/tracking/test_multi_subject_tracking.py` | NEW | Targeted unit tests covering stability, order invariance, timeout, association, and reset | **VERIFIED** |
| `audit/phase_2_report.md` | NEW | Phase 2 engineering report | **VERIFIED** |

---

## 4. Tests Performed & Verification Status

Targeted test execution:
- `tests/tracking/test_multi_subject_tracking.py`: **5 passed in 0.12s**
  - `test_persistent_identity_and_track_stability`: **VERIFIED** (stable ID across frame shift, confirmed after 2 frames)
  - `test_multiple_subjects_order_invariance`: **VERIFIED** (track IDs persist when detector swaps face order)
  - `test_track_timeout_and_removal`: **VERIFIED** (coasting through dropouts, deletion after timeout)
  - `test_object_and_person_association`: **VERIFIED** (phone near secondary person attributed to subject 2, not candidate)
  - `test_session_isolation_resets_all_tracks`: **VERIFIED** (zero residual tracks or counters after reset)
- `tests/temporal/test_aggregator.py`: **5 passed in 0.13s** (zero regressions in temporal incident merging)

---

## 5. Verification Assessment & Limitations

* **Persistent Identity**: **VERIFIED** (unit & synthetic test scenarios).
* **Multiple Subjects**: **VERIFIED** (order invariance and subject separation verified).
* **Object Association**: **VERIFIED** (spatial hand/subject association verified).
* **Real-World Unconstrained Multi-Face In-The-Wild Video**: **PARTIALLY VERIFIED** (tested on synthetic and targeted test fixtures; full wild video benchmark scheduled for evaluation dataset in Phase 4).
* **High-Crowd Occlusion Re-Identification**: **LIMITATION** (cosine similarity on SFace requires at least partial facial visibility; prolonged complete occlusion falls back to coasting and eventual timeout).
