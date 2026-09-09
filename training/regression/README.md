# Permanent Regression Dataset Suite

This directory contains the permanent regression specifications for the AI Proctoring Engine.
Every candidate model must execute against this regression suite before being submitted for human promotion review.

## Invariants Tested:

1. **Phone vs Hand Disambiguation (REG_001)**:
   - Splayed empty hand positions must never trigger confirmed `PHONE_DETECTED`.
   - Bounding box aspect ratios outside standard smartphone ratios (~1.3 - 2.6) with marginal confidence must be flagged as `PHONE_CANDIDATE_UNCERTAIN`.

2. **AirPods / Earbuds vs Other Ear Objects (REG_002)**:
   - Punctate reflections, earrings, and glasses stems must be classified as `other_ear_object`, preventing false earbud accusations.

3. **Physical Paper Posture Tolerances (REG_003)**:
   - Looking downward at physical answer booklet (-45° pitch, downward gaze) in `PHYSICAL_PAPER` mode must not trigger `SUSPICIOUS_HEAD_POSE` or `GAZE_OFF_SCREEN`.

4. **Paper Quadrilateral Recognition (REG_004)**:
   - A4 / Letter format sheets on desk must maintain detected status and detect large displacements or lifting.

5. **Offline Outbox & Recovery Durability (REG_005)**:
   - Events generated during complete network or process interruption must survive in atomic journal checkpoints and synchronize idempotently.
