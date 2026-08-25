# AI Proctoring System

Computer-vision examination proctoring that produces **evidence for a human proctor**,
not verdicts. The system observes, timestamps and packages what the camera saw. It
computes no suspicion score and decides nothing about a candidate — the invigilator
reviews the package and makes the call.

---

## The workflow

Every stage below is implemented by one engine, `proctoring/engine.py`:

```
Examination starts
      ↓
Camera / frame input ─────────────── webcam or recorded video
      ↓
Frame validation & preprocessing ─── resolution, brightness, blur, CLAHE
      ↓
Face detection ───────────────────── YuNet
      ↓
Face identity verification ───────── SFace, against enrolled templates
      ↓
Scene / behavioural observation ──── no face · enrolled candidate · unknown person
                                     multiple people · prohibited objects
                                     hands · speaking · head pose · gaze
                                     headphones · earbuds · smart watch
                                     liveness (blink) · per-candidate calibration
      ↓
Temporal qualification ───────────── transient one-frame noise filtered out,
                                     duration and continuity tracked
      ↓
Event / evidence creation ────────── timestamp, type, duration, the frame that
                                     shows it, factual observation text
      ↓
Evidence validation ──────────────── re-read, re-decode, re-hash; invalid or
                                     incomplete references removed
      ↓
Timeline & telemetry ─────────────── per-frame record, latency, FPS, diagnostics
      ↓
Tamper-evident evidence package ──── manifest.json · events.json · timeline.json
                                     telemetry.json · evidence files · SHA-256
      ↓
Proctor / invigilator review ─────── AI provides evidence, a human decides
```

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate

pip install -e .                    # face presence, identity, evidence packaging
pip install -e ".[behaviour]"       # + hands, speech, head pose, gaze, liveness
pip install -e ".[all]"             # + objects and worn devices

# Fetch every model into models/ (optional ones degrade gracefully if skipped)
python scripts/download_models.py
```

Installing provides a `proctor` command; everything below also works as
`python -m proctoring.cli …`.

### Run a live webcam session

```bash
# Enrol the candidate first so identity verification has something to match
proctor live --enroll --student "A. Candidate"

# A stricter exam profile, with worn-device detection
proctor live --enroll --strictness STRICT --detect-wearables
```

A window opens showing detections and the live state of each workflow stage.

The HUD shows every stage live: face box and identity, hand skeletons, speaking,
head yaw/pitch, gaze offset, detected devices, and which observations are currently
open.

| Key | Action |
|-----|--------|
| `q` / `Esc` | Stop the session and seal the evidence package |
| `t` | Record a browser tab-switch event |
| `f` | Record a fullscreen-exit event |

During enrolment: `space` captures a reference sample, `a` captures them
automatically, `q` skips enrolment.

Without `--enroll` the session still runs, but every face is reported
`UNVERIFIED` rather than matched or unknown — the system will not call a face
unknown when it was never shown what known looks like.

### Process a recorded video

```bash
proctor video exam_recording.mp4 --student "A. Candidate" --zip
```

### Embed the engine

```python
from proctoring import ProctoringEngine, SessionConfig
from proctoring.detection import FaceDetector, FaceVerifier

engine = ProctoringEngine(
    config=SessionConfig(session_id="exam_001", student_name="A. Candidate"),
    face_detector=FaceDetector(),
    face_verifier=FaceVerifier(),
)
engine.start_session()

for frame_index, (frame, timestamp) in enumerate(your_frame_source):
    observation = engine.process_frame(frame, frame_index, timestamp)
    print(observation.face_status, observation.prohibited_object_names)

summary = engine.finalize_session()
print(summary.package_dir, summary.integrity_verified)
```

---

## Project layout

```
proctoring/                The pipeline — one installable package, one engine
├── config.py              SessionConfig: every tunable, grouped by stage
├── engine.py              ProctoringEngine: the workflow, end to end
├── observation.py         FrameObservation: what a single frame measured
├── core/                  Event schema, error handling, path safety
├── capture/               Camera discovery, webcam stream, video sampling
├── preprocessing/         Frame quality gate, CLAHE, face alignment
├── detection/             YuNet, SFace, YOLO, relevance filtering
├── analysis/              Hands, speech, pose, gaze, liveness, devices, exam policy
├── temporal/              Temporal qualification, candidate lifecycle
├── evidence/              Evidence capture, quality validation, packaging
├── telemetry/             Session timeline, latency profiling
├── integration/           LMS boundary — the stable surface a Moodle plugin calls
└── cli/                   `python -m proctoring.cli live | video`

tools/                     Offline evaluation — not part of a live session
├── harness.py             Engine wrapper adding lifecycle statistics
├── benchmark/             Accuracy and threshold benchmarking
├── field_testing/         Simulated field trials and long-session studies
├── hardening/             Stress, recovery and package-verification suites
├── audit/                 Regression suite and requirement audit tables
├── optimization/          Ablation study and before/after comparison
└── research/              Superseded single-modality prototypes, kept for reference

scripts/                   Standalone CLI utilities (model download, analysis)
tests/                     Mirrors proctoring/, plus tests/tools/
docs/                      accuracy_and_performance.md · moodle_integration.md
                           pipeline/ · detection/ · evaluation/
models/                    ONNX weights (not committed)
data/                      Samples and generated session packages (not committed)
```

---

## What a session produces

```
data/results/live_sessions/<session_id>/
├── manifest.json      Package identity, config, models, summaries, SHA-256 of every file
├── events.json        Qualified observations with their evidence references
├── timeline.json      Per-frame record of what was actually measured
├── telemetry.json     Latency percentiles, per-model timings, resource usage
├── diagnostics.json   System faults (camera drops, inference errors)
└── evidence/
    ├── frames/        UNMODIFIED source captures, one per event
    ├── crops/         Padded regions of interest
    └── review/        ANNOTATED copies showing what the system reacted to
```

`frames/` is what the manifest attests to and what an appeal must be judged against.
`review/` is the system's interpretation drawn on top — marked derived everywhere it
appears. Never present a review image as source evidence.

`manifest.json` hashes every other file in the package, and is itself hashed. Any
later modification to any artefact is detectable via
`SessionEvidencePackage.verify_package_integrity()`.

### Exam strictness

`--strictness STANDARD | STRICT | MAXIMUM` selects which behavioural observations
are reportable and how quickly they qualify. `STANDARD` reports only unambiguous
events; `STRICT` adds speaking, looking away and hands at the ear; `MAXIMUM` adds
gaze and hand-position observations that carry a high false-positive rate.

Raising strictness raises false positives. That is a deliberate trade — see the
[accuracy and performance guide](docs/accuracy_and_performance.md) before choosing.

### Event severity is triage, not scoring

`INFO` / `LOW` / `MEDIUM` / `HIGH` / `CRITICAL` order a proctor's review queue. They
are not summed, weighted or aggregated into a candidate risk score, and no part of
the system infers intent from an observation.

### Qualified vs recorded

An incident shorter than `min_event_duration_seconds` is still written to the
package, but marked `RECORDED` rather than `QUALIFIED`. Nothing is hidden from the
proctor; brief noise is simply flagged as brief.

---

## Testing and quality gates

```bash
pytest                              # full suite (259 tests)
pytest tests/test_audit_regressions.py   # defects found in audit, each guarded
pytest tests/integration            # the LMS contract a Moodle plugin depends on
pytest tests/tools                  # offline evaluation harnesses

ruff check . && ruff format --check .    # lint and formatting
mypy                                     # types (strict on core/ and integration/)
```

Tests that need model weights skip automatically when `models/` is empty. CI runs
the same gates on Python 3.10–3.12; see `.github/workflows/ci.yml`. Install the
pre-commit hooks with `pre-commit install`.

---

## Documentation

| Guide | Covers |
|---|---|
| [Accuracy and performance](docs/accuracy_and_performance.md) | What each detector detects, measured speed, how much to trust each signal, tuning, known limitations |
| [Moodle integration](docs/moodle_integration.md) | The service contract a `quizaccess` plugin calls, deployment shape, data protection |

## Scope

The Moodle plugin itself is not built. The integration boundary it will use
(`proctoring/integration/`) exists, is versioned, and is tested — see the
integration guide.

## Honest limits

This system has **no audio**: speaking is inferred from lip movement alone. In-ear
earbud detection is unreliable by nature and is reported as *suspected*, never
confirmed. Detection accuracy has not been quantified against a labelled dataset,
and demographic performance variation has not been evaluated. Read
[docs/accuracy_and_performance.md](docs/accuracy_and_performance.md) §1 and §8
before relying on any of it.
