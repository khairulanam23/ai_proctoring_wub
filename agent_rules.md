# Agent Engineering Rules & Behavioral Contract

```yaml
standard: ponytail_engineering_philosophy
repository: ai_proctoring_wub
authority: authoritative_agent_guidance
status: active
enforcement: strict
```

These rules define the engineering standards for all AI coding agents operating on the `ai_proctoring_wub` repository. They combine the minimal-change discipline of the upstream **Ponytail** project with strict domain-specific protections for the AI proctoring system.

---

## 1. Core Engineering Philosophy (The Ponytail Ladder)

Lazy means efficient and disciplined, not negligent or careless. The best code is the code never written. Before writing or changing code, stop at the first rung that holds:

```text
1. Does this need to exist at all?     → No: skip it (YAGNI).
2. Already in this codebase?           → Reuse it; do not rewrite or duplicate.
3. Does the Standard Library do it?    → Use stdlib.
4. Native platform feature covers it?  → Use native platform features.
5. Installed dependency solves it?     → Use existing dependency. Never add a package for a few lines.
6. Can it be one clean line?           → Keep it to one line.
7. Only then:                          → Write the minimum readable code that works.
```

### Ladder Rules & Execution Principles
* **Read before climbing**: Always inspect the affected codebase and trace the complete flow end-to-end before touching any file.
* **Lazy, never negligent**: Trust-boundary validation, error handling, security, data-loss prevention, and hardware recovery are **never** on the chopping block.
* **Shortest working diff wins**: Avoid unrequested abstractions, premature factories, single-implementation interfaces, and speculative scaffolding.
* **Deletion over addition**: Prefer simplifying or removing dead code over creating new boilerplate. Boring, clear code beats clever hacks.
* **Root-cause fixes only**: When fixing bugs, grep all call sites and fix the root cause once where calls converge.

---

## 2. Strict Production AI Protection Boundaries

The repository contains a frozen, production-validated AI proctoring engine.

1. **Frozen Production Subsystems**:
   Do **NOT** modify or destabilize production:
   - Face detection / Face verification (`proctoring/detection/`)
   - Object detection (`proctoring/detection/object_detector.py`)
   - Behavioral analysis (`proctoring/analysis/`)
   - Temporal state aggregation (`proctoring/temporal/`)
   - Incident lifecycle & event generation (`proctoring/core/events.py`, `proctoring/engine.py`)
   - Evidence packaging & SHA-256 manifests (`proctoring/evidence/`)
   - Model weights and model loading (`models/`, `proctoring/core/model_registry.py`)
   - Production service orchestration & API (`proctoring/integration/`)
   - Production configuration defaults (`proctoring/config.py`)

2. **No Unrelated Refactoring**:
   Do not refactor working production code, alter imports, rename fields, or "clean up" existing code paths outside the scope of your assigned task.

3. **No Unsolicited Model Modifications**:
   Do not retrain models, alter model weights, download external checkpoints, or change confidence thresholds without explicit user instruction.

4. **Zero Production Coupling for Research Tools**:
   Research utilities, dataset capture tools, and benchmarking scripts must remain strictly isolated. Production inference and the incident engine must never import or depend on research tools.

---

## 3. Data Integrity & Trust Boundary Rules

1. **Preserve Raw Research Data**:
   Raw recordings, sensor streams, and session manifests are permanent evidentiary artifacts. Capture tools must never delete, overwrite, or mutate previously collected raw sessions.
2. **Never Silently Overwrite**:
   Before creating or recording into a session directory, verify whether it already exists. Refuse to execute or require explicit operator confirmation if a collision is detected.
3. **Instruction is NOT Ground Truth**:
   Experimental prompts given to participants (e.g. "Put in an earphone") represent `instructed_condition`, **not** verified `ground_truth`. Ground truth requires human validation and annotation. Never conflate the two in schemas or manifests.
4. **No Fabricated Data or Test Results**:
   Never generate synthetic data and pretend it is real production telemetry. Never report that a test passed or hardware was verified unless the command was actually executed and succeeded.

---

## 4. Verification & Documentation Contract

1. **Verify Changes with Concrete Evidence**:
   Every code change must be validated with automated tests. Report the exact commands executed and the output received.
2. **Explicitly Report Hardware Status**:
   If hardware (e.g. a physical webcam, GPU, or microphone) is missing from the environment, state: `Not performed.` Never simulate hardware and claim physical verification.
3. **Continuous Maintenance of `project_state.md`**:
   The `project_state.md` document is the authoritative handoff record for future engineering agents. Update it after completing any substantive implementation, detailing:
   - Architecture & boundaries
   - Files added, modified, and untouched
   - Actual test runs with exact outputs
   - Limitations and deferred work
