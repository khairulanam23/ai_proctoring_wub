# Phase 3 Audit Report: Phone / Hand / Paper / Writing Behavioral Analysis

## Objective
Enhance frame-to-frame and behavioral interpretation of phones, hands, paper, and writing interactions using the persistent tracking and temporal foundation established in Phase 2. Ensure the system accurately categorizes ambiguous signals (e.g., distinguishing empty hand false positives, hand-object ambiguity, possible phones, and confirmed phones), interprets paper manipulation and handwriting kinematics, and maps observations into granular, auditable human-review evidence rather than automated cheating determinations.

## Implementation Performed
1. **Phone Disambiguation & Hand Interaction Hierarchy**:
   - Expanded `PhoneClassification` enum in `proctoring/analysis/phone_disambiguation.py` to include `HAND_OBJECT_AMBIGUITY` and `POSSIBLE_PHONE` alongside `CONFIRMED_PHONE`, `HAND_FALSE_POSITIVE`, and `UNCERTAIN_CANDIDATE`.
   - Reordered disambiguation logic: checks hand overlap and landmark gripping geometry *before* rigid aspect-ratio filtering to prevent empty open hands or partially occluded hands from bypassing false-positive dismissal.
   - Wired temporal persistence so high-confidence rectangular objects require multi-frame confirmation (`POSSIBLE_PHONE` in frame 1 $\to$ `CONFIRMED_PHONE` in frame 2+).
2. **Unified Temporal Aggregation of Behavioral States**:
   - Updated `UnifiedTemporalAggregator.aggregate_frame()` in `proctoring/temporal/aggregator.py` to register granular phone classifications (`CONFIRMED_PHONE`, `POSSIBLE_PHONE`, `HAND_OBJECT_AMBIGUITY`) into temporal state summaries and incident logs.
3. **Physical Paper & Hand Kinematics Policy Alignment**:
   - Added `HAND_LIFTED_FROM_PAPER` and `HAND_LEAVING_WRITING_AREA` to `PHYSICAL_PAPER` exam policy in `proctoring/analysis/policy.py`.
   - Updated `BehaviourObserver.map_to_events()` in `proctoring/analysis/observer.py` to recognize `HAND_WRITING`, `HAND_RESTING`, and `HAND_LEAVING_WRITING_AREA` states and emit structured observations with descriptive metadata.

## Files / Components Changed
- `proctoring/analysis/phone_disambiguation.py`: Added classifications (`HAND_OBJECT_AMBIGUITY`, `POSSIBLE_PHONE`), reordered hand-grip checks before aspect-ratio filter, updated temporal tracker.
- `proctoring/analysis/policy.py`: Added `HAND_LIFTED_FROM_PAPER` and `HAND_LEAVING_WRITING_AREA` to `PHYSICAL_PAPER` allowed events.
- `proctoring/analysis/observer.py`: Added event mapping for handwriting, resting hand, and writing area exit.
- `proctoring/temporal/aggregator.py`: Integrated `HAND_OBJECT_AMBIGUITY` and `POSSIBLE_PHONE` handling.
- `tests/analysis/test_phase3_behavior.py`: Created targeted test suite covering phone persistence, empty hand vs object ambiguity, handwriting kinematics, resting vs leaving desk area, and paper manipulation.

## Tests Performed
- `tests/analysis/test_phase3_behavior.py`:
  - `test_phone_confirmed_with_temporal_persistence`: PASSED
  - `test_hand_false_positive_vs_object_ambiguity`: PASSED
  - `test_hand_writing_motion_mapped_to_events`: PASSED
  - `test_hand_resting_vs_leaving_writing_area`: PASSED
  - `test_paper_manipulation_events`: PASSED
- `tests/analysis/test_phone_disambiguation.py` (4 tests): PASSED
- `tests/analysis/test_paper_recognition.py` (4 tests): PASSED

## Actual Results
- 13 targeted tests executed and passed (100% pass rate).
- Total runtime: ~0.25s.
- Detections are categorised into granular behavioral evidence:
  - Empty hands detected by YOLO as phones are suppressed as `HAND_FALSE_POSITIVE`.
  - Ambiguous hand-object overlaps are emitted as `HAND_OBJECT_AMBIGUITY` instead of false positives.
  - Multi-frame phone detections advance from `POSSIBLE_PHONE` to `CONFIRMED_PHONE`.
  - Paper movements and handwriting oscillations generate timestamped evidence records for human review.

## Limitations
- Hand grip evaluation relies on MediaPipe hand landmark extraction; if hand landmarks cannot be resolved (e.g. low resolution, severe motion blur), the disambiguator falls back to bounding-box IoU and aspect-ratio heuristics.
- Real-world precision/recall on diverse physical paper sheets and writing instruments remains unmeasured without a curated proctoring benchmark dataset.

## Verification Status Matrix
- **VERIFIED**:
  - Distinguishing empty hand false positive vs hand-object ambiguity in unit tests.
  - Multi-frame temporal confirmation for phone detections (`POSSIBLE_PHONE` $\to$ `CONFIRMED_PHONE`).
  - Behavioral event mapping for handwriting kinematics (`HAND_WRITING`), resting hands (`HAND_RESTING`), and desk departures (`HAND_LEAVING_WRITING_AREA`).
  - Paper manipulation event emission with displacement measurements.
- **PARTIALLY VERIFIED**:
  - Full end-to-end integration across multi-subject video streams (verified via unit/integration tests; full test suite deferred to Phase 6 per instructions).
- **UNVERIFIED**:
  - Real-world production accuracy, false-alarm rates, and precision/recall across varied lighting and desk setups.
- **MISSING**:
  - Dedicated tactile pressure or stylus sensor inputs (inference is strictly computer-vision based).
- **DISCONNECTED**:
  - None. Hand kinematics, paper detection, and phone disambiguation are fully wired into `FrameObservation` and `BehaviourObserver`.
- **SIMULATED**:
  - Hand landmarks and phone candidate geometries in unit tests are synthetically constructed to validate behavioral logic deterministically.
