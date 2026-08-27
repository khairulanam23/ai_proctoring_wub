# Accuracy and Performance Guide

What each detector actually detects, how fast it runs, how much you should trust
it, and how to tune it for your examinations.

> **Read this before acting on any observation.** The pipeline reports what it
> measured. It does not decide whether misconduct occurred, and several of its
> signals are much weaker than their confidence numbers make them look.

---

## 0. What the system will not do

It produces no risk score, no cheating probability, no suspiciousness percentage and
no automated decision. It records observations and evidence; a human invigilator
decides what they mean.

Equipment faults are held apart from candidate behaviour structurally, not by
convention: every event carries a `category` of `CANDIDATE_OBSERVATION` or
`TECHNICAL_DIAGNOSTIC`, and the two are counted separately everywhere they surface.
A failed detector yields `NOT_MEASURED`, never `NO_FACE`.

---

## 1. Scope of what has been measured

Being precise about this matters, because the two are often conflated:

| | Status |
|---|---|
| **Speed / throughput / memory** | **Measured.** Every number in §3 comes from `scripts/benchmark_pipeline.py` on the hardware named there. Reproduce with one command. |
| **Correctness of the pipeline logic** | **Tested.** 288 automated tests cover temporal qualification, evidence provenance, package integrity, policy gating, the LMS boundary, and every defect found in the codebase audit. |
| **Detection accuracy (precision / recall)** | **Not quantified.** No labelled proctoring dataset with ground-truth "was this candidate actually speaking / wearing an earbud" has been evaluated. |

The reliability ratings in §4 are **engineering judgement** based on the sensor
physics — target size in pixels, occlusion, class ambiguity — and on functional
testing against sample imagery. They are not measured error rates, and this guide
does not present them as such. §7 explains how to measure them on your own footage,
which is the only way to get numbers that mean anything for your deployment.

---

## 2. What each detector does

### Face presence — YuNet
Finds faces and their bounding boxes. Boxes below `min_face_size_px` (default 40 px)
are discarded, which suppresses posters, photographs and faces on a background
monitor being counted as a second person.

Produces `NO_FACE`, `MULTIPLE_FACES`.

### Identity verification — SFace
Compares a 128-dimensional embedding against the candidate's enrolment templates,
matching if cosine similarity clears `face_match_threshold` (default 0.3630) against
**any** template. Multi-template enrolment is what stops ordinary pose and lighting
change from reading as an impostor.

Produces `UNKNOWN_FACE`. Without enrolment the face is reported `UNVERIFIED` — the
system will not call a face unknown when it was never shown what known looks like.

**Every face is verified, not only a lone one.** With several people on camera the
observation carries `enrolled_face_present` and `unknown_face_count` alongside
`MULTIPLE_FACES`, and `UNKNOWN_FACE` is additionally raised when the enrolled candidate
is verifiably absent. Verifying only single faces left the impostor case unanswered:
"the candidate plus a helper" and "two strangers, candidate gone" produced identical
output.

### Speech-like mouth activity — MediaPipe Face Landmarker
An articulation index is built from `jawOpen`, with `mouthFunnel` / `mouthPucker` /
`mouthStretch*` added at reduced weight so close-lipped speech stays visible. Three
conditions must hold together across a rolling window (`speech_window_seconds`,
default 3 s):

1. **Amplitude** — the robust swing (p90 − p10) exceeds
   `speech_articulation_amplitude`. Rules out a still face and expression drift.
2. **Crossings** — the index crosses its own mid-level at least
   `speech_min_crossings` times. This is what separates speech from a yawn: a yawn
   crosses twice however deep it is, continuous articulation crosses repeatedly.
3. **Returns to closed** — the mouth comes back down near the window minimum. A
   yawn, a smile and a resting open mouth all hold a level instead.

> **Fixed defect.** The previous implementation averaged six blendshapes flat,
> including `mouthClose` — which rises as the jaw closes and therefore cancelled the
> signal being measured. The mean swung a few hundredths during clear speech, far
> below any usable threshold, so speaking was almost never reported.

> **Fixed defect — sampling.** Speech articulates at roughly 3–5 Hz. Sampled at the
> old 4 fps default it aliased into a slow wave shaped exactly like a yawn: measured
> over a 3 s window, talking and yawning both produced **one** mid-level crossing.
> No threshold can separate them, because the information is gone before any
> threshold sees it. Sessions that report speaking therefore raise their sampling
> rate — including the adaptive idle rate — to `speech_min_sampling_fps` (default 6),
> and the manifest records that they did. Below that floor the analyzer reports
> **not measured** rather than **not speaking**: a rate that makes the measurement
> impossible must never read as a candidate who was observed and found silent.

**There is no audio.** This is lip movement only, and the observation is worded as
*possible talking* for exactly that reason.

Produces `CANDIDATE_SPEAKING`.

### Head pose and gaze — MediaPipe Face Landmarker
Yaw, pitch and roll are decomposed from the 4×4 facial transformation matrix. Gaze
comes from iris centre displacement within the eye aperture, using the refined iris
landmarks (468–477).

> **Fixed defect:** the original decomposition used the textbook ZYX order, which
> does not match MediaPipe's axis convention. It reported a head turned *sideways*
> as pitched steeply *downward*, so a candidate glancing left produced a continuous
> stream of false `LOOKING_AWAY` events. `tests/analysis/test_facial_dynamics.py`
> now asserts that a pure yaw rotation moves yaw only, and that real frontal
> portraits never trip the looking-away threshold.

Produces `LOOKING_AWAY`, `GAZE_OFF_SCREEN`, `SUSPICIOUS_HEAD_POSE`.

**Movement patterns, not just angles.** A single threshold on yaw answers only "is
the head turned right now", which is not the question a proctor has: a candidate who
turned once to answer the door and one who checks the same spot every twenty seconds
read identically. `HeadMovementTracker` keeps a rolling window
(`head_pattern_window_seconds`, default 20 s) and derives how long the current turn
has been held, how many *separate* deviation episodes occurred, and how many rapid
direction reversals happened above `head_reversal_velocity_deg_per_s`. Enough
repeated episodes or fast reversals produce `SUSPICIOUS_HEAD_POSE` — an observation
about a *pattern*, distinct from any single `LOOKING_AWAY`. Whether a given pose
counts as deviating is still decided by `ExamPolicy`, so `PHYSICAL_PAPER` mode's
allowance for looking down at paper is honoured without the thresholds existing in
two places.

**Gaze corroboration.** When gaze is displaced meaningfully in the same direction as
the head turn, the recorded confidence rises and the observation text says so. Head
orientation alone is the weakest attention signal; two independent measurements
agreeing is materially stronger, and a flat confidence of 1.0 for every case told a
proctor nothing.

**Per-candidate calibration.** `FacialDynamicsAnalyzer.calibrate(frames)` learns the
candidate's neutral yaw, pitch and gaze from a short "look at your screen normally"
capture, and all three thresholds are then measured relative to that baseline. This
addresses the single largest source of false `LOOKING_AWAY` events: a camera that is
not where the thresholds assume it is — a laptop on a stand, a webcam clipped to a
second monitor, a candidate sitting off to one side. Calibration is *rejected* if the
samples disagree by more than 12° of standard deviation, because a candidate who moved
during calibration has not given a usable neutral pose and averaging it would bake
their movement into every later reading.

> **Fixed defect.** The baseline was measured and then thrown away. The reporting
> layer re-evaluated head pose against a hard-coded zero, so calibration corrected
> nothing a proctor ever saw and an off-centre camera still produced constant
> `LOOKING_AWAY`. The baseline the analyzer applied is now published on the result
> and used by the reporting layer, pinned by
> `tests/analysis/test_detection_accuracy.py`.

### Liveness — MediaPipe Face Landmarker

Identity verification compares a still image against a still template, so **a printed
photograph or a phone screen held to the camera passes it outright**. Blinking is the
cheapest signal separating a person from a picture, and the `eyeBlinkLeft` /
`eyeBlinkRight` blendshapes are already computed by the landmarker.

Blinks are counted on the falling edge of eyelid closure. A face observed continuously
for longer than `liveness_grace_seconds` (default 45 s) with zero blinks is reported as
`POSSIBLE_PRESENTATION_ATTACK` at `MEDIUM` severity — never higher. The grace period is
deliberately generous: some people blink rarely, and a false accusation of spoofing is
serious. The liveness window restarts whenever the face leaves frame, so a candidate who
steps away is not judged on the time they were absent.

Produces `POSSIBLE_PRESENTATION_ATTACK`.

### Temporal smoothing

Per-frame verdicts from the landmark models flicker: a hand tracker drops a frame, a
blink briefly changes the mouth blendshapes, the head crosses the yaw threshold and
comes back. Feeding that straight to the temporal aggregator fragments one real
behaviour into several short incidents and manufactures brief ones that never happened.

A majority vote over `smoothing_window_frames` (default 3) removes it. Measured on a
flickering signal: raw `##.##.##` becomes `########` (dropouts bridged), and an isolated
blip `..#...` becomes `......` (suppressed). Wearable detections bypass smoothing — they
already run on a decimated cadence with their own carry-forward, so a second layer would
add only lag.

### Hands — MediaPipe Hand Landmarker
Up to two hands at 21 landmarks each, then related geometrically to the face.
Proximity uses the **closest** point of the hand including fingertips, not the
centroid: a candidate reaching for an earpiece touches it with a fingertip long
before their hand's centre arrives. Distances are normalised by face width, so the
result does not depend on how close the candidate sits.

Produces `HAND_NEAR_EAR`, `HAND_NEAR_FACE`, `HANDS_NOT_VISIBLE`.

### Objects — YOLO11
Standard COCO classes filtered to proctoring-relevant ones. The phone threshold is
raised to 0.40 by default because a wallet or dark notebook lying flat on a desk is
the single most common phone false positive.

Produces `PHONE_DETECTED`, `PROHIBITED_OBJECT`.

### Headphones, earbuds, smart watches — YOLO-World
None of these classes exist in COCO, so an open-vocabulary detector takes free-text
prompts instead. Detections of ear-worn targets are **cross-checked against the ear
regions** derived from the face landmarks; a candidate box nowhere near an ear is
discarded outright, which removes the bulk of spurious earbud boxes fired by
earrings and hair clips.

**Two passes.** At typical webcam framing an in-ear bud is on the order of fifteen
pixels wide — below what the detector resolves on a full frame, which is why a
full-frame sweep alone reported headphones well and earbuds essentially never. Each
ear region is therefore cropped and upscaled to `ear_roi_target_px` (default 320) and
re-examined. Boxes found in a crop are mapped back into frame coordinates and are
ear-anchored by construction. This is the single largest recall improvement available
without changing models, and it roughly doubles the cost of a sweep (measured: 260 ms
full-frame, 465 ms with both ears) — see §3.

> **Fixed defect — ear regions.** Each region used to be a small square centred on
> one face-silhouette landmark. That landmark sits on the *attachment* line of the
> ear, so a bud in the canal, a hook over the top of the ear and a headset earpiece
> all fell outside it — and anything outside is discarded. Genuine detections were
> being thrown away by the filter meant to protect against false ones. Regions are
> now built from the whole visible ear perimeter plus `ear_region_padding_ratio`.

**Confirmation across sweeps.** A worn device is reported only once
`wearable_confirmation_sweeps` (default 2) detection sweeps agree, counted per
*sweep* rather than per frame — the carried-forward result is re-presented on every
skipped frame, so counting frames would let one marginal detection stuff the ballot.
A hand at the ear in the same frame, seen independently by the hand analyzer, lowers
the requirement by one sweep for earpieces only. It never creates a detection on its
own, and it cannot lift an earbud above "medium" reliability: two weak signals
agreeing is still not proof.

Produces `HEADPHONES_DETECTED`, `EARBUDS_SUSPECTED`, `SMARTWATCH_DETECTED`.

---

## 3. Measured performance

Hardware: Linux x86-64, Python 3.11, **CPU only**, 640×480 frames, real face imagery.
Reproduce with `python scripts/benchmark_pipeline.py --wearables`.

### Per-stage cost, measured in isolation

| Stage | Median | p95 | Notes |
|---|---:|---:|---|
| Frame quality gate (+ CLAHE) | 8.1 ms | 13.9 ms | CLAHE only runs under extreme lighting |
| Face detection (YuNet) | 16.7 ms | 19.4 ms | Every frame |
| Identity verification (SFace) | 10.1 ms | 13.5 ms | Only when exactly one face is present |
| Facial dynamics (speech/pose/gaze) | 14.4 ms | 22.5 ms | One model yields all three signals |
| Hand analysis | 17.0 ms | 31.3 ms | Every frame |
| Object detection (YOLO11n) | 91.0 ms | 107.0 ms | The expensive optional stage |
| **Wearable detection (YOLO-World), full frame** | **237.3 ms** | **264.7 ms** | ~14× the landmark models |
| **Wearable detection, + ear-region zoom pass** | **~465 ms** | — | Two extra small inferences; what makes earbuds detectable |

The shape of that table drives the whole design. Face, hands, speech, pose and gaze
together cost about **56 ms** — the entire behavioural capability is cheaper than
one YOLO pass. Wearable detection alone costs more than everything else combined,
which is why it is **off by default** and, when on, runs once every
`wearable_detection_interval_frames` (default 8) rather than every frame. A device
worn during an exam stays on for minutes; sampling it every two seconds loses
nothing. That interval is held at two seconds of wall clock: if the sampling rate is
raised for speech measurement, the frame interval is scaled to match, so the most
expensive detector in the pipeline does not become more frequent as a side effect of
an unrelated setting.

### End-to-end session throughput

Face detection + full behavioural analysis, object and wearable detection off:

| Strictness | Throughput | Median latency | p95 | Events (60 frames) |
|---|---:|---:|---:|---:|
| STANDARD | 9.2 fps | 72.8 ms | 313.2 ms | 8 |
| STRICT | 9.8 fps | 71.0 ms | 301.9 ms | 9 |
| MAXIMUM | 9.9 fps | 71.0 ms | 310.5 ms | 10 |

Strictness costs essentially nothing in CPU — it changes which observations are
*reported*, not which detectors run. It changes the event count, which is the point.

### Capacity

At the default 4 fps sampling rate the per-frame budget is 250 ms and the pipeline
uses about 73 ms of it — **29% of budget**, leaving roughly **3 concurrent sessions
per CPU core**. Enabling object detection roughly triples per-frame cost and drops
that to about one session per core.

The p95 of ~310 ms exceeds the 250 ms budget. This is expected and harmless: it is
dominated by first-invocation model warm-up and occasional scheduler stalls, and the
engine's adaptive sampling absorbs it. It is not a sustained rate.

**Memory:** the ~2.1 GB peak RSS reported by the benchmark is the *benchmark
process*, which imports torch and YOLO-World. A face-and-behaviour session without
those is far smaller. In-memory evidence frames are separately capped by
`max_retained_evidence_frames` (default 200, about 180 MB at 640×480).

---

## 4. How much to trust each signal

| Observation | Reliability | Why | Fires wrongly when |
|---|---|---|---|
| `POSSIBLE_PRESENTATION_ATTACK` | **Low** | Blink rate varies naturally between people | A candidate who blinks rarely; poor lighting hiding eyelid movement |
| `NO_FACE` | **High** | Large target, mature detector | Candidate leans out of frame briefly; severe backlighting |
| `MULTIPLE_FACES` | **High** | Same | A face on a background poster or screen above the 40 px floor |
| `PHONE_DETECTED` | **Medium-high** | Distinct COCO class | Wallet, dark notebook, TV remote flat on a desk |
| `UNKNOWN_FACE` | **Medium** | Depends entirely on enrolment quality | Single-sample enrolment; big lighting change; glasses on/off |
| `HEADPHONES_DETECTED` | **Medium** | Large, high-contrast, ear-anchored | Over-ear hats, large hair, headband |
| `CANDIDATE_SPEAKING` | **Medium** | Oscillation test rejects yawns | Chewing; muttering while thinking; **reading aloud, which may be permitted** |
| `LOOKING_AWAY` | **Medium** | Pose is well-defined once decomposed correctly | Camera mounted off-centre; second monitor; laptop on a stand |
| `HAND_NEAR_EAR` | **Medium** | Clear geometry | Scratching; resting head on hand; adjusting hair or glasses |
| `SMARTWATCH_DETECTED` | **Medium** | Distinct shape, needs wrist in frame | Bracelets, fitness bands |
| `GAZE_OFF_SCREEN` | **Low-medium** | No per-candidate calibration | Any non-standard screen size or camera position |
| `HAND_NEAR_FACE` | **Low** | Extremely common innocent posture | Constantly — this is why `STANDARD` and `STRICT` do not report it |
| `HANDS_NOT_VISIBLE` | **Low** | Depends entirely on framing | A laptop camera angled at the face may never see hands at all |
| **`EARBUDS_SUSPECTED`** | **Low** | See below | Earrings, moles, ear shadow, hair, headphone cables |

### Why earbuds are the weak link

An in-ear device occupies a few dozen pixels at typical webcam distance, is
frequently occluded by hair, and is visually similar to an earring or the natural
shadow of the ear canal. This is a limitation of the sensor and the target, not of
the model — no confidence threshold fixes it.

The pipeline handles this by refusing to overstate it:

- Severity is capped at `MEDIUM`, never `HIGH`.
- Reliability never rates `high` regardless of the model's confidence score.
- Detections must be geometrically anchored to an ear region or they are discarded.
- A higher confidence floor and a longer qualification duration apply than for any
  other target.
- The review payload attaches an inline caveat next to the finding.

**Treat `EARBUDS_SUSPECTED` as a prompt to look at the snapshot, never as a finding.**
If earpiece detection is critical to your examinations, the honest answer is a
custom-trained detector on your own imagery, not this one.

The ear-region zoom pass and the sweep-confirmation requirement both raise how often
a real earpiece is found and how much agreement is needed before it is reported. They
do not turn this into a reliable detection, and nothing here has been measured against
labelled ground truth. `EARBUDS_SUSPECTED` stays `MEDIUM` severity with an inline
reliability note for that reason.

### Verifying it on your own camera

Detection quality depends on your camera, lighting and framing far more than on any
threshold in this document. Before relying on any of it, run the guided check:

```bash
python scripts/validate_detection.py                    # every scenario
python scripts/validate_detection.py --group earphone   # one behaviour
python scripts/validate_detection.py --list
```

It walks through the behaviours and, just as importantly, their innocent lookalikes —
a yawn, a single glance, reading silently, looking down at paper — and reports what
the pipeline concluded for each. A scenario that cannot run (no YOLO-World weights,
no camera) is reported as **skipped**, never as passed.

---

## 5. Strictness levels

Set with `--strictness` on the CLI or `strictness` in `SessionConfig`.

| | STANDARD | STRICT | MAXIMUM |
|---|---|---|---|
| Reportable observations | 12 | 20 | 23 |
| Default qualification | 2.0 s | 1.5 s | 1.0 s |
| Absence tolerance | 1.5 s | 1.0 s | 0.75 s |
| Yaw limit | 40° | 28° | 24° |
| Sampling rate | 4 fps | 6 fps | 6 fps |
| Possible talking | — | ✓ | ✓ |
| Looking away | — | ✓ | ✓ |
| Repeated head movement | — | ✓ | ✓ |
| Hand at ear | — | ✓ | ✓ |
| Earbuds suspected | — | ✓ | ✓ |
| Liveness (no blink) | — | ✓ | ✓ |
| Hand near face | — | — | ✓ |
| Hands not visible | — | — | ✓ |
| Gaze off screen | — | — | ✓ |

`STRICT` and `MAXIMUM` sample at 6 fps rather than 4 because they report speech-like
mouth activity, which is not measurable below that rate — see §2. `STANDARD` does not
report it and keeps the cheaper 4 fps.

The yaw limits at `STRICT` and `MAXIMUM` are lower than they were (32°/28°). A single
frame past the limit no longer means anything on its own: deviations must persist to
qualify, and repeated short deviations are reported as a pattern rather than as
isolated events, so the angle can sit closer to where a candidate genuinely stops
looking at their screen.

The levels are strictly nested — each reports everything the level below does — and
this is enforced by test.

**Choosing:** `STANDARD` for ordinary coursework. `STRICT` for invigilated and
higher-stakes exams. `MAXIMUM` for remote certification where a proctor genuinely
reviews every session — at this level the low-reliability signals are on, and a
proctor should expect to dismiss a meaningful share of what is flagged.

Raising strictness **raises the false-positive rate**. That is the deliberate trade,
and it is why every event carries a snapshot.

Weak signals are also made to work harder: at `STRICT`, an earbud detection must
persist 2.0 s to qualify while a phone needs only 0.25 s.

---

## 6. Tuning

Every threshold lives in `proctoring/analysis/policy.py` and is recorded in the
evidence manifest, so a review months later can tell why a session flagged what it
did — the same footage under a different policy produces a different event list.

```python
from proctoring.analysis import ExamPolicy, StrictnessLevel
from proctoring.core.events import EventType
from proctoring.config import SessionConfig

policy = ExamPolicy.for_level(StrictnessLevel.STRICT)
policy.yaw_limit_degrees = 45.0  # off-centre camera
policy.event_min_duration[EventType.CANDIDATE_SPEAKING] = 4.0  # tolerate muttering
policy.enabled_events = policy.enabled_events - {EventType.EARBUDS_SUSPECTED}

config = SessionConfig(session_id="exam_001", _policy=policy)
```

Common adjustments:

| Problem | Fix |
|---|---|
| Constant `LOOKING_AWAY` from an off-centre webcam | Raise `yaw_limit_degrees` to 45–50 |
| `CANDIDATE_SPEAKING` on a candidate who mutters while reading | Raise `event_min_duration[CANDIDATE_SPEAKING]` to 4–5 s |
| Phone false positives from desk clutter | Raise `phone_confidence_threshold` to 0.5 |
| `MULTIPLE_FACES` from a background poster | Raise `min_face_size_px` to 60–80 |
| `NO_FACE` flickering while the candidate turns their head | Raise `absence_tolerance_seconds` to 2.0 |
| Too many earbud false positives | Remove `EARBUDS_SUSPECTED` from `enabled_events` |

---

## 7. Validating on your own footage

The reliability ratings above are judgement, not measurement. To get real numbers:

1. **Record 10–20 sessions** on the actual hardware, lighting and camera placement
   your candidates will use. Camera position matters more than anything else here.
2. **Script the behaviours.** Have volunteers deliberately speak, wear earbuds, look
   away, put hands out of view — and label when each actually happened.
3. **Run each recording** at each strictness level:
   ```bash
   python -m proctoring.cli video session_01.mp4 --strictness STRICT
   ```
4. **Compare `events.json` against your labels.** Count true positives, false
   positives and misses per event type.
5. **Tune, then re-run** the same footage. Because the policy is recorded in the
   manifest, two runs are directly comparable.

A false-positive rate you measured on your own candidates is worth more than any
published benchmark.

---

## 8. Known limitations

**No audio.** Speech is inferred from lip movement alone. A candidate speaking with
minimal lip movement is not detected; a candidate chewing may be.

**Camera framing dominates everything.** A laptop camera angled at the face may
never see hands, making `HANDS_NOT_VISIBLE` meaningless. An off-centre camera makes
`LOOKING_AWAY` fire constantly. Standardise placement before tightening thresholds.

**Single-camera blind spots.** A phone below the desk, a person outside the frame,
or notes taped to the monitor bezel are all invisible. Absence of observations is
not evidence of absence, and the session result says so in its coverage warnings.

**Identity verification needs good enrolment.** Fewer than three samples makes the
threshold brittle. The enrolment response says so explicitly.

**CPU-bound.** All figures here are CPU. A CUDA GPU changes the YOLO numbers
substantially and the MediaPipe numbers much less.

**Wearable detection needs a 338 MB CLIP text encoder** downloaded on first use, in
addition to the detector weights.

**Demographic performance is unmeasured.** YuNet, SFace and MediaPipe have known
accuracy variation across skin tone, age and facial features, and this project has
not evaluated that. If you deploy this across a diverse candidate population, that
is a gap you need to close yourself — a system that flags some groups more often
than others is unfair regardless of intent.

---

## 9. Reading a session package

```
<session_id>/
├── manifest.json      Config, policy, models, summaries, SHA-256 of every file
├── events.json        Qualified observations with evidence references
├── timeline.json      Per-frame record of everything measured
├── telemetry.json     Latency percentiles, per-stage timings, resource usage
├── diagnostics.json   System faults — never candidate behaviour
└── evidence/
    ├── frames/        UNMODIFIED source captures
    ├── crops/         Padded regions of interest
    └── review/        ANNOTATED copies for proctor review
```

The distinction between `frames/` and `review/` is load-bearing:

- **`frames/`** is what the manifest attests to and what an appeal must be judged
  against. Unmodified.
- **`review/`** is the system's *interpretation* drawn on top — boxes, hand
  skeletons, measured angles, and why the event fired. Marked derived in the
  manifest and in every review payload.

Never present a `review/` image as the source evidence.

### Qualified versus recorded

An incident shorter than its required duration is still written to the package but
marked `RECORDED` rather than `QUALIFIED`. Nothing is hidden from the proctor; brief
noise is simply flagged as brief.

### Severity is triage, not scoring

`INFO` / `LOW` / `MEDIUM` / `HIGH` / `CRITICAL` order a review queue. They are never
summed, weighted, or aggregated into a candidate risk score, and no part of the
system infers intent from an observation.

---

## 10. Deployment sizing

| Configuration | Per-frame | Sessions per core @ 4 fps |
|---|---:|---:|
| Face presence only | ~25 ms | ~10 |
| Face + identity | ~35 ms | ~7 |
| **Face + identity + behaviour** (recommended) | ~73 ms | ~3 |
| + object detection | ~165 ms | ~1.5 |
| + wearable detection (every 8th frame) | ~195 ms | ~1 |

For a 200-candidate concurrent exam with the recommended configuration, budget
roughly **64–70 CPU cores**, or use a GPU and re-benchmark. Load models once at
process start and share them via `ProctoringService(detector_factory=...)` — a
`FaceVerifier` reads a 37 MB ONNX file, far too costly to repeat per attempt.

Lower `sampling_fps` to 2 to halve cost; the adaptive sampler already idles at
`idle_fps` while nothing is happening and steps up only when an incident is open.
