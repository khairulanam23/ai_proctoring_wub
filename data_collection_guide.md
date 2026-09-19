# AI Proctoring Engine — Controlled Pilot Dataset Collection Guide

**Document**: `data_collection_guide.md`  
**Purpose**: Authoritative operator manual for conducting controlled real-world dataset collection for future AI model development.  
**Hardware Target**: ANYKA V380 FHD Camera (`0380:2006` on `/dev/video0`)  
**Capture Utility**: `tools/capture_dataset.py`  
**Target Environment**: Linux (`.venv`, Python 3.14.4, OpenCV V4L2)  
**Safety Scope**: Research tool only. Production AI inference, weights, thresholds, and evidence pipeline are strictly **read-only**.

---

## 1. Core Principles & Safety Invariants

Before collecting any data, the operator must adhere to the following six dataset engineering rules:

1. **`instructed_condition` is NOT Ground Truth**:
   - When a scenario instructs: *"Hold your phone in your right hand"*, the recorded frames contain the participant attempting that action.
   - It does **not** guarantee that the phone was fully visible, unoccluded, or correctly oriented in every frame.
   - Ground truth labels must be established later by human annotation and verification.
2. **Raw Video is an Immutable Source of Truth**:
   - `raw_video.mp4` must **never** be edited, re-encoded, filtered, or overwritten after capture.
   - Any future preprocessing (clipping, resizing, normalization) must be written to distinct derived directories.
3. **No Cross-Frame Data Leakage**:
   - Future train/validation/test splits **must** be grouped strictly by `participant_id` and `session_id`.
   - Never randomly split adjacent video frames across dataset partitions.
4. **Authoritative Temporal Logging**:
   - The capture tool dynamically measures physical frame arrival and synchronizes the container FPS.
   - `session_manifest.json` authoritative frame indexes (`frame_start_index`, `frame_end_index`) and timestamps define the step timeline.
5. **Zero Production AI Impact**:
   - Data collection is completely isolated from the production proctoring engine (`proctoring/`).
   - Do not import production detection models or modify thresholds during collection.
6. **No Synthetic or Fabricated Fillers**:
   - Every frame and manifest entry must represent genuine physical camera capture.

---

## 2. Participant & Session Identifiers

To preserve candidate privacy and ensure clean partitioning:

- **Participant Identifiers**: Anonymized alphanumeric IDs:
  - `P001`, `P002`, `P003`, etc.
  - Do **not** record student names, registration numbers, or unnecessary Personally Identifiable Information (PII).
- **Session Identifiers**: Sequential session numbers per participant:
  - `S001`, `S002`, `S003`, etc.
- **Directory Structure**:
  ```text
  data/research_capture/
  └── <participant_id>/
      └── <session_id>/
          └── <scenario_name>/
              ├── raw_video.mp4           # Synchronized raw capture video
              ├── session_manifest.json   # Machine-readable metadata & timing
              └── checksum.sha256         # Detached SHA-256 integrity digest
  ```

---

## 3. Preflight System Verification (Stage 0)

Before seating a participant, verify the capture environment:

### Step 1: Check Camera Hardware Node
```bash
.venv/bin/python tools/capture_dataset.py --check-camera
```
*Expected Output*:
```text
PASS: Valid camera found on /dev/video0 (Backend: V4L2, 640x480 @ 30.0 FPS)
```

### Step 2: Verify Scenario Configurations
```bash
.venv/bin/python tools/capture_dataset.py --list-scenarios
```
*Expected Output*: Displays all 4 loaded scenarios (`natural_exam`, `phone`, `paper`, `earphones`).

### Step 3: Check Available Disk Storage
```bash
df -h data/research_capture/
```
*Requirement*: Ensure at least **5 GB** of available storage before initiating a session.

---

## 4. Pilot Collection Protocol & Scenario Execution

For each participant (beginning with **P001**), execute scenarios in the following order:

```text
1. natural_exam  ──>  2. phone  ──>  3. paper  ──>  4. earphones
```

### Operator Interactive Key Controls
During capture, the operator has real-time keyboard control:
- **`SPACE`**: Pause / Resume recording (pauses countdown and preserves frame timing).
- **`s`**: Skip current step immediately.
- **`r`**: Repeat current step (restarts step countdown).
- **`q`**: Safely abort session (finalizes recorded video and manifest).
- **`y` / `n`**: Respond to branching questions when prompted.

---

### Scenario 1: `natural_exam` (Baseline Behavior)

**Objective**: Capture realistic unscripted exam behavior without devices or unauthorized materials.

- **Physical Participant Instructions**:
  - Sit naturally in front of the laptop/webcam.
  - Read questions on the screen, look down as if thinking, type on the keyboard.
  - Shift posture naturally, glance occasionally off-screen, blink normally.
  - **Do NOT exaggerate** or act theatrically. Subtle, realistic variations are desired.
- **Execution Command**:
  ```bash
  .venv/bin/python tools/capture_dataset.py \
    --participant-id P001 \
    --session-id S001 \
    --scenario natural_exam
  ```
- **Conditions Recorded**:
  1. `neutral_frontal_reading`: Candidate reading text on the monitor.
  2. `downward_gaze_reading`: Downward gaze at keyboard or desk.
  3. `keyboard_typing_active`: Hands actively typing.
  4. `offscreen_glance_left`: Brief natural gaze shift to the left.
  5. `offscreen_glance_right`: Brief natural gaze shift to the right.
  6. `posture_shift_leaning`: Slouching, leaning back, or shifting in chair.
  7. `subtle_fidgeting`: Scratching chin, resting face in palm, adjusting eyeglasses.
  8. `normal_baseline_wrapup`: Returning to neutral exam focus.

---

### Scenario 2: `phone` (Smartphones & Hand-Held Devices)

**Objective**: Capture smartphone handling, varied orientations, occlusions, and hard-negative objects.

- **Required Materials**:
  - Smartphone
  - Hard negative objects (if available): Wallet, calculator, TV/AC remote, power bank.
- **Physical Participant Instructions**:
  - Follow the on-screen prompt for each step.
  - Hold the phone at lap level, desk level, and raised towards the screen.
  - Practice single-handed grip (left/right) and two-handed typing.
  - When prompted for hard negatives, answer `Y` or `N` via the terminal.
- **Execution Command**:
  ```bash
  .venv/bin/python tools/capture_dataset.py \
    --participant-id P001 \
    --session-id S001 \
    --scenario phone
  ```
- **Conditions Recorded**:
  1. `empty_hands_baseline`: Hands resting empty on desk.
  2. `phone_right_hand_lap`: Phone held in right hand below desk level.
  3. `phone_left_hand_lap`: Phone held in left hand below desk level.
  4. `phone_desk_surface`: Phone lying flat or angled on desk surface.
  5. `phone_two_handed_typing`: Both hands holding and thumb-typing on phone.
  6. `phone_raised_screen_level`: Phone held up towards camera/monitor.
  7. `phone_partial_hand_occlusion`: Phone held with fingers covering logo/edges.
  8. `empty_cupped_hand_mimic`: **Hard Negative**: Hand cupped as if holding a phone, but completely empty.
  9. `branching questions`: Interactive checks for wallet, calculator, and power bank.

---

### Scenario 3: `paper` (Physical Paper & Written Materials)

**Objective**: Capture paper sheets, answer scripts, writing kinematics, sheet turns, and desk hard negatives.

- **Required Materials**:
  - 2–3 sheets of plain/ruled paper
  - Pen or pencil
  - Hard negative objects (if available): White laptop cover, desk notebook, book cover.
- **Physical Participant Instructions**:
  - Position paper flat on desk, write normally with pen.
  - Flip pages, shuffle sheets, move paper partially out of camera frame.
  - Cover portions of the paper with arms/hands while writing.
- **Execution Command**:
  ```bash
  .venv/bin/python tools/capture_dataset.py \
    --participant-id P001 \
    --session-id S001 \
    --scenario paper
  ```
- **Conditions Recorded**:
  1. `clear_desk_baseline`: Completely empty desk surface.
  2. `paper_single_sheet_flat`: Standard single sheet placed in front of candidate.
  3. `active_handwriting_pen`: Right/left hand actively writing with pen on paper.
  4. `paper_reading_holding`: Lifting sheet slightly off desk to read.
  5. `paper_page_turn_flip`: Flipping or sliding one sheet over another.
  6. `paper_partial_arm_occlusion`: Forearm resting over paper while thinking.
  7. `paper_edge_of_frame`: Paper positioned near desk corner/webcam boundary.
  8. `branching questions`: Hard negatives (white laptop, closed notebook, book).

---

### Scenario 4: `earphones` (Earphones, Wearables & Audio Devices)

**Objective**: Capture wireless earbuds, wired earphones, headphones, and anatomical ear hard negatives.

- **Required Materials**:
  - Wireless earbud (at least one)
  - Wired earphones or over-ear headset (if available).
- **Physical Participant Instructions**:
  - Sit facing camera, turning head slightly left/right when prompted to expose ear canal.
  - Insert earbud into left ear, then right ear, then both.
  - Adjust earbud, remove it, place it on desk.
  - Demonstrate hard negatives: touch ear with finger, wear eyeglasses, tuck hair behind ear.
- **Execution Command**:
  ```bash
  .venv/bin/python tools/capture_dataset.py \
    --participant-id P001 \
    --session-id S001 \
    --scenario earphones
  ```
- **Conditions Recorded**:
  1. `bare_ears_baseline`: Both ears uncovered and visible, turning head ±30°.
  2. `single_earbud_left`: Earbud placed in left ear.
  3. `single_earbud_right`: Earbud placed in right ear.
  4. `bilateral_earbuds`: Earbuds in both ears.
  5. `earbud_adjustment_touch`: Hand touching/adjusting earbud in ear.
  6. `earbud_removal`: Removing earbud and setting it down.
  7. `hard_neg_ear_scratch`: **Hard Negative**: Scratching earlobe or pinna with bare hand.
  8. `hard_neg_hair_shadow`: **Hard Negative**: Hair partially covering ear.
  9. `branching questions`: Checks for wired earphones and over-ear headphones.

---

## 5. Post-Recording Verification Checklist (Stage 8)

Immediately after each scenario completes, run this lightweight validation checklist:

```bash
# Set variables
PARTICIPANT="P001"
SESSION="S001"
SCENARIO="natural_exam"
SESSION_DIR="data/research_capture/${PARTICIPANT}/${SESSION}/${SCENARIO}"

# 1. Verify files exist
ls -la "${SESSION_DIR}"

# 2. Check SHA-256 checksum integrity
(cd "${SESSION_DIR}" && sha256sum -c checksum.sha256)

# 3. Verify video can be independently decoded
ffmpeg -v error -i "${SESSION_DIR}/raw_video.mp4" -f null - && echo "Video Decode: PASS"

# 4. Verify temporal fields and frame counts
python -c "
import json
with open('${SESSION_DIR}/session_manifest.json') as f:
    m = json.load(f)
print('Requested FPS :', m.get('requested_fps'))
print('Reported FPS  :', m.get('reported_camera_fps'))
print('Measured FPS  :', m.get('measured_effective_fps'))
print('Encoded FPS   :', m.get('encoded_fps'))
print('Frames Written:', m.get('frames_recorded'))
print('Capture Time  :', m.get('capture_duration_seconds'), 's')
print('Playback Time :', m.get('playback_duration_seconds'), 's')
delta = abs(m.get('active_duration_seconds', 0) - m.get('playback_duration_seconds', 0))
print(f'Playback Delta: {delta:.4f}s (PASS)' if delta < 0.2 else f'Playback Delta: {delta:.4f}s (WARN)')
"
```

### Acceptance Criteria per Session:
- [ ] Directory contains `raw_video.mp4`, `session_manifest.json`, and `checksum.sha256`.
- [ ] SHA-256 checksum outputs `raw_video.mp4: OK`.
- [ ] `ffmpeg` decodes with 0 stream errors.
- [ ] `frames_recorded` matches decoded frames.
- [ ] `playback_duration_seconds` matches `active_duration_seconds` within **0.1 seconds**.

---

## 6. Visual Quality Inspection Guide (Stage 9)

After completing all 4 scenarios for a participant, extract and inspect representative frames:

### Extract Keyframes for Inspection
```bash
# Extract 5 evenly spaced frames from a session
mkdir -p /tmp/inspection_${SCENARIO}
ffmpeg -v error -i "${SESSION_DIR}/raw_video.mp4" -vf "fps=1/10" "/tmp/inspection_${SCENARIO}/frame_%03d.png"
```

### Visual Inspection Rubric:
1. **Camera Framing**:
   - Head, shoulders, face, and hands on desk are clearly within camera FOV.
   - Candidate does not sit too close (distorted fisheye) or too far (loss of facial details).
2. **Lighting & Exposure**:
   - Face is evenly illuminated without harsh backlighting (e.g., bright window behind candidate).
   - No severe clipping/saturation on white paper or candidate forehead.
3. **Sharpness & Resolvability**:
   - Eyes, eyeglasses, and mouth contours are sharply distinguishable.
   - Small objects (earbuds, phone edges, pen tip) are distinctly resolvable.
4. **Motion Blur**:
   - Normal hand and head transitions do not degrade into unresolvable smearing.

---

## 7. Scaling Protocol & Decision Matrix (Stage 10 & 11)

### Decision Gate After Participant P001:
- **Proceed to P002** IF:
  - All 4 scenarios completed with valid manifests and matching checksums.
  - Video playback duration matches capture time (1:1 real-time speed).
  - Target objects and candidate anatomy are visually resolvable.
- **Stop & Investigate** IF:
  - Video stream fails to decode or drops frames excessively (<10 FPS).
  - Web camera disconnects or `/dev/video0` disappears.
  - A scenario instruction causes frequent confusion or participant refusal.

### Introducing Controlled Variation Across Participants:
When expanding from P001 to P002, P003, etc., introduce **natural real-world variation**:
- **Varied Seating & Angles**: Camera mounted slightly left, center, or slightly higher/lower.
- **Varied Lighting**: Daylight morning lighting, ambient afternoon lighting, warm indoor lamplight.
- **Varied Backgrounds**: Plain wall, bookshelf, home study, typical dorm environment.
- **Participant Demographics & Features**: Candidates with eyeglasses, varied hairstyles, beard/mustache, varied skin tones, left-handed vs. right-handed writing/phone usage.

---

## 8. Troubleshooting Reference

| Symptom | Cause | Remedy |
| :--- | :--- | :--- |
| `Cannot open camera: index 0` | Camera disconnected or busy | Run preflight: `.venv/bin/python tools/capture_dataset.py --check-camera`. Ensure no other app (Zoom, browser) is using `/dev/video0`. |
| Video plays back in fast-forward | Unsynchronized container FPS | Verify you are using the hardened `tools/capture_dataset.py` which automatically remuxes to measured effective FPS. |
| Session already exists error | Safety anti-overwrite guard | Increment `--session-id` (e.g., from `S001` to `S002`). |
| Insufficient disk space error | Drive storage < 500 MB | Clean temporary files or change output path with `--output-dir`. |
| GUI window fails (`cv2.error`) | Headless terminal / no X11 `$DISPLAY` | Add `--no-preview` flag to run in command-line mode. |
| Participant made an error | Mistake during instructed step | Press `r` on the keyboard to repeat the step, or note it in the session manifest. |
