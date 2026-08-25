# Moodle Integration Guide

How a `quizaccess_proctoring` plugin talks to this pipeline. The plugin itself is
not built yet — this describes the boundary it will use, and that boundary exists
and is tested today.

---

## The contract

A host application touches exactly one package:

```python
from proctoring.integration import ProctoringService, StartSessionRequest
```

Everything under `proctoring/` other than `integration/` is internal and free to
change. `integration/schemas.py` carries `CONTRACT_VERSION` and changes only by
bumping it.

Two rules shape every payload:

- **No scores.** Nothing ranks or grades a candidate. The host receives observations
  and evidence; the academic judgement is the invigilator's.
- **Biometrics stay server-side.** Face embeddings never cross the boundary. A
  session references its enrolment by opaque identifier only.

---

## Mapping Moodle hooks to service calls

| Moodle event | Service call |
|---|---|
| Candidate enrols (before the attempt) | `enrol_candidate(enrolment_id, frames)` |
| `quizaccess_proctoring` attempt begins | `start_session(StartSessionRequest(...))` |
| Browser posts a webcam frame | `ingest_frame(session_id, data_url)` |
| Client-side JS detects a tab switch | `record_client_event(session_id, "BROWSER_TAB_SWITCH", ts)` |
| Attempt submitted | `finalize_session(session_id)` |
| Attempt abandoned / browser closed | `abort_session(session_id, reason)` |
| Teacher opens the review screen | `get_review(session_id)` |

---

## Lifecycle

### 1. Enrolment

```python
service = ProctoringService(output_dir="/var/moodledata/proctoring")

result = service.enrol_candidate(
    enrolment_id=f"user_{user_id}",
    frames=[data_url_1, data_url_2, data_url_3, data_url_4, data_url_5],
)
# {'templates_stored': 5, 'frames_rejected': 0, 'usable': True,
#  'quality_note': 'Enrolment accepted.'}
```

Capture **at least three** samples across slightly different poses. Fewer makes
identity verification intolerant of ordinary movement, and the response says so.

Frames with zero or more than one face are rejected individually with a reason
rather than failing the batch.

### 2. Attempt start

```python
handle = service.start_session(
    StartSessionRequest(
        attempt_id=str(attempt_id),
        user_id=str(user_id),
        course_id=str(course_id),
        quiz_id=str(quiz_id),
        candidate_name=fullname,
        strictness="STRICT",
        sampling_fps=4.0,
        enrolment_id=f"user_{user_id}",
    )
)
```

`handle.active_detectors` and `handle.unavailable_detectors` say what this session
can actually observe. **Surface this to the proctor.** A deployment missing the
MediaPipe bundles silently loses speech, gaze and hand analysis, and a review screen
that does not say so implies coverage it never had.

### 3. Frame ingestion

The browser captures from `<video>` to a canvas and POSTs a data URL:

```javascript
const canvas = document.createElement('canvas');
canvas.width = 640; canvas.height = 480;
canvas.getContext('2d').drawImage(videoEl, 0, 0, 640, 480);

const response = await fetch(PROCTOR_ENDPOINT, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        session_id: sessionId,
        frame: canvas.toDataURL('image/jpeg', 0.8),
        timestamp_seconds: (Date.now() - attemptStart) / 1000,
    }),
});
const ack = await response.json();

// Follow the server's pacing rather than a fixed interval — it idles the rate
// down while nothing is happening and raises it when something is.
setTimeout(captureFrame, ack.next_frame_due_in_seconds * 1000);
```

Server side:

```python
ack = service.ingest_frame(session_id, payload["frame"], payload["timestamp_seconds"])
return ack.to_dict()
```

`ingest_frame` accepts a base64 data URL, raw JPEG/PNG bytes, or a decoded array.

**A corrupt or truncated payload is never fatal.** It comes back with
`accepted: false` and a reason, is recorded as a capture diagnostic, and the session
continues. An exam must survive a dropped packet.

### 4. Client-side events

```python
service.record_client_event(session_id, "BROWSER_TAB_SWITCH", ts, "Tab hidden")
```

Recognised: `BROWSER_TAB_SWITCH`, `BROWSER_FULLSCREEN_EXIT`, `BROWSER_WINDOW_BLUR`,
`BROWSER_COPY_PASTE`. Unrecognised names are recorded as
`OTHER_SUSPICIOUS_ACTIVITY` rather than dropped, so a newer client against an older
server still gets its signal onto the timeline.

### 5. Finalisation

```python
result = service.finalize_session(session_id)
```

Returns package location, archive path, `manifest_sha256`, integrity status,
observation counts, and `coverage_warnings` — anything that limited what the session
could observe. Store `package_dir` and `manifest_sha256` against the attempt.

Use `abort_session` when a browser closes mid-attempt: it still seals whatever was
gathered and marks the record `FAILED`, so a partial session is never mistaken for a
complete one.

### 6. Review

```python
review = service.get_review(session_id)
```

Contains the result, the observations with their evidence references, timeline and
telemetry summaries, the exact policy that produced them, and
`reviewer_guidance` — framing text that should be rendered at the top of the review
screen, not hidden behind a help link.

Observations built on weak signals carry an inline `reliability_note`:

```json
{
  "event_type": "EARBUDS_SUSPECTED",
  "severity": "MEDIUM",
  "reliability_note": "In-ear devices are only a few dozen pixels at webcam distance
                       and are easily confused with earrings, hair or ear shadow.
                       Treat as a prompt to look at the snapshot, never as a finding
                       on its own."
}
```

**Render the note next to the finding.** A caveat a proctor does not see is a caveat
that does not exist.

Each evidence entry is marked `is_derived`. Entries with `media_type: "review"` are
annotated copies; `"frame"` entries are unmodified source captures. Never present a
derived image as source evidence.

---

## Deployment shape

The service is transport-agnostic — no web framework, no HTTP. Wire it to Moodle's
external API, a sidecar FastAPI process, or a worker queue.

**Recommended:** a Python sidecar service alongside Moodle, since PHP cannot run
these models. Moodle's plugin calls it over HTTP on localhost or an internal network.

```
Browser ──frames──▶ Moodle plugin ──HTTP──▶ Python sidecar (ProctoringService)
                          │                          │
                          └──── review UI ◀──────────┘
```

Load models once at process start and share them:

```python
def build_detectors():
    detector = FaceDetector()
    return {"face_detector": detector, "face_verifier": FaceVerifier(detector=detector)}


service = ProctoringService(detector_factory=build_detectors)
```

A `FaceVerifier` reads a 37 MB ONNX file — far too costly to repeat per attempt.

### Concurrency

One lock guards the session registry, and each session's engine is used from one
request at a time. **Frames for a given attempt must be ingested serially**, which
is what a browser posting on an interval does anyway. Different attempts are
independent.

For capacity planning see §10 of the [accuracy and performance guide](accuracy_and_performance.md):
roughly three concurrent 4 fps sessions per CPU core with the recommended
configuration.

### Session storage

`SessionStore` keeps records in memory and mirrors them to JSON. For multi-worker
deployments, subclass it against Moodle's database:

```python
class MoodleSessionStore(SessionStore):
    def put(self, record): ...  # write to mdl_quizaccess_proctoring_session
    def get(self, session_id): ...
    def list(self, attempt_id=None, user_id=None): ...
```

---

## Data protection

Face embeddings are biometric data, and in most jurisdictions that means explicit
consent, a defined retention period, and a working erasure path.

- Templates are stored under an opaque `enrolment_id`, never returned by any call.
- `SessionStore.delete_enrolment(enrolment_id)` erases them. **Wire this to your
  retention policy and to withdrawal of consent** — an erasure path nobody calls is
  not an erasure path.
- Evidence packages contain images of the candidate. Set a retention period and
  delete `package_dir` when it expires.
- `manifest.json` deliberately records only the *count* of enrolment templates, so
  the manifest handed to a proctor carries no recoverable biometric material.

The system produces no risk score, so there is no automated decision to disclose —
but you must still tell candidates what is recorded and how long it is kept.

---

## Suggested plugin settings

| Setting | Maps to | Suggested default |
|---|---|---|
| Strictness | `strictness` | `STANDARD`; `STRICT` for exams |
| Capture rate | `sampling_fps` | 4.0 |
| Require enrolment | gate the attempt on `enrol_candidate` | On for exams |
| Detect wearables | `enable_wearable_detection` | Off — read the guide first |
| Retention (days) | your deletion job | 90 |

---

## Testing without a browser

`tests/integration/test_service.py` exercises the whole boundary — lifecycle,
transport forms, corrupt payloads, abort, review contract and enrolment erasure —
with no camera and no models. Use it as the reference for how the plugin should call
the service.

```bash
pytest tests/integration -v
```
