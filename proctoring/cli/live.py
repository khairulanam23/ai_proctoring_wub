"""Live webcam proctoring session with an on-screen heads-up display.

Runs the complete examination workflow against a local camera — frame validation,
YuNet face detection, SFace identity verification, scene observation, temporal
qualification, evidence capture, validation and tamper-evident packaging — and
shows what each stage is seeing in real time.

Typical use::

    python -m proctoring.cli live --enroll --student "A. Candidate"

The ``--enroll`` flag captures the candidate's reference face first, which is what
gives stage 5 something to verify against.  Without it the pipeline still runs, but
every face is reported ``UNVERIFIED`` rather than matched or unknown — the system
will not call a face unknown when it was never shown what known looks like.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from proctoring.capture.camera import discover_local_camera
from proctoring.config import SessionConfig
from proctoring.core.events import EventType
from proctoring.detection.face_detector import FaceDetector
from proctoring.detection.face_verifier import FaceVerifier
from proctoring.detection.object_detector import ObjectDetector
from proctoring.engine import FaceStatus, FrameObservation, ProctoringEngine

WINDOW_TITLE = "AI Proctoring — Live Session"

# BGR colours keyed to what the proctor needs to notice first.
COLOUR_OK = (90, 210, 120)  # enrolled candidate present, nothing unusual
COLOUR_WARN = (60, 190, 250)  # something worth a look
COLOUR_ALERT = (80, 90, 240)  # face absent, unknown, or a prohibited object
COLOUR_MUTED = (170, 170, 170)
COLOUR_HAND = (250, 200, 90)
COLOUR_DEVICE = (200, 90, 220)
COLOUR_PANEL = (28, 28, 30)

# MediaPipe hand skeleton links, for drawing a hand as a hand.
_HAND_LINKS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_detectors(
    models_dir: Path,
    enable_objects: bool,
) -> tuple[FaceDetector | None, FaceVerifier | None, ObjectDetector | None]:
    """Load whichever models are present, reporting clearly on the ones that are not.

    A missing model disables its stage rather than aborting the session: a user who
    has not downloaded YOLO weights should still be able to exercise face presence
    and identity verification.
    """
    face_detector: FaceDetector | None = None
    face_verifier: FaceVerifier | None = None
    object_detector: ObjectDetector | None = None

    yunet = models_dir / "face_detection_yunet_2023mar.onnx"
    sface = models_dir / "face_recognition_sface_2021dec.onnx"

    if yunet.exists():
        face_detector = FaceDetector(model_path=yunet)
        print(f"  [ok]   YuNet face detector        {yunet.name}")
    else:
        print(f"  [MISS] YuNet not found at {yunet}")
        print("         Run: python scripts/download_models.py")

    if sface.exists() and face_detector is not None:
        face_verifier = FaceVerifier(detector=face_detector, recognizer_model_path=sface)
        print(f"  [ok]   SFace identity verifier    {sface.name}")
    elif not sface.exists():
        print(f"  [MISS] SFace not found at {sface} — identity verification disabled")

    if enable_objects:
        try:
            object_detector = ObjectDetector()
            print(
                f"  [ok]   YOLO object detector       {object_detector.model_name} ({object_detector.device})"
            )
        except Exception as exc:
            print(f"  [MISS] YOLO unavailable ({exc}) — object detection disabled")

    return face_detector, face_verifier, object_detector


# ---------------------------------------------------------------------------
# Enrolment (prerequisite for workflow stage 5)
# ---------------------------------------------------------------------------


def enroll_candidate(
    capture: cv2.VideoCapture,
    face_detector: FaceDetector,
    face_verifier: FaceVerifier,
    sample_count: int = 5,
) -> list[np.ndarray]:
    """Capture reference embeddings for the candidate before the session starts.

    Several samples are taken a short interval apart rather than one, so the
    enrolment template spans a little natural pose and lighting variation.  A
    single-frame enrolment makes the threshold brittle: the candidate leaning back
    or turning slightly then reads as an identity mismatch.
    """
    print("\n" + "-" * 62)
    print("  ENROLMENT — look directly at the camera")
    print(f"  Capturing {sample_count} reference samples. Press SPACE to take each,")
    print("  or 'a' to capture all automatically. Press 'q' to skip enrolment.")
    print("-" * 62)

    templates: list[np.ndarray] = []
    auto_mode = False
    last_auto = 0.0

    while len(templates) < sample_count:
        ok, frame = capture.read()
        if not ok or frame is None:
            print("  Camera read failed during enrolment.")
            break

        display = frame.copy()
        result = face_detector.detect(frame)
        usable = len(result.faces) == 1

        for face in result.faces:
            x, y, w, h = face.bbox
            colour = COLOUR_OK if usable else COLOUR_ALERT
            cv2.rectangle(display, (x, y), (x + w, y + h), colour, 2)

        if len(result.faces) == 0:
            hint, colour = "No face detected — centre yourself in frame", COLOUR_ALERT
        elif len(result.faces) > 1:
            hint, colour = "Multiple faces — only the candidate should be visible", COLOUR_ALERT
        else:
            hint, colour = ("Auto-capturing..." if auto_mode else "Ready — press SPACE"), COLOUR_OK

        _draw_panel(display, 0, 0, display.shape[1], 64)
        cv2.putText(
            display,
            f"ENROLMENT  {len(templates)}/{sample_count}",
            (14, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
        cv2.putText(display, hint, (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        cv2.imshow(WINDOW_TITLE, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print("  Enrolment skipped.")
            break
        if key == ord("a"):
            auto_mode = True

        take = usable and (key == ord(" ") or (auto_mode and time.time() - last_auto > 0.6))
        if take:
            try:
                embedding = face_verifier.extract_feature(frame, face=result.faces[0])
                templates.append(embedding)
                last_auto = time.time()
                print(f"  Captured reference sample {len(templates)}/{sample_count}")
            except Exception as exc:
                print(f"  Sample rejected: {exc}")

    if templates:
        print(f"  Enrolment complete — {len(templates)} template(s) registered.\n")
    else:
        print("  No enrolment templates — identity verification will report UNVERIFIED.\n")
    return templates


# ---------------------------------------------------------------------------
# Heads-up display
# ---------------------------------------------------------------------------


def _draw_panel(image: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.72) -> None:
    """Blend a translucent panel so overlay text stays readable over any scene."""
    x2, y2 = min(x + w, image.shape[1]), min(y + h, image.shape[0])
    if x2 <= x or y2 <= y:
        return
    region = image[y:y2, x:x2]
    panel = np.full(region.shape, COLOUR_PANEL, dtype=np.uint8)
    cv2.addWeighted(panel, alpha, region, 1 - alpha, 0, region)


def _status_line(obs: FrameObservation) -> tuple[str, tuple[int, int, int]]:
    """Reduce the frame observation to one plainly-worded status and a colour."""
    if not obs.accepted:
        return f"FRAME REJECTED — {obs.rejection_reason}", COLOUR_ALERT
    if obs.face_status == FaceStatus.NOT_MEASURED:
        return "Face detection not running", COLOUR_MUTED
    if obs.face_status == FaceStatus.NO_FACE:
        return "No face detected", COLOUR_ALERT
    if obs.face_status == FaceStatus.MULTIPLE_FACES:
        return f"{obs.face_count} faces in frame", COLOUR_ALERT
    if obs.face_status == FaceStatus.UNKNOWN_FACE:
        return "Face does not match enrolled candidate", COLOUR_ALERT
    if obs.face_status == FaceStatus.UNVERIFIED:
        return "Face present (not enrolled — identity unverified)", COLOUR_WARN
    return "Enrolled candidate present", COLOUR_OK


def draw_hud(
    frame: np.ndarray,
    obs: FrameObservation,
    fps: float,
    elapsed: float,
    event_count: int,
    identity_enabled: bool,
) -> np.ndarray:
    """Render the live overlay: detections, per-stage state and session counters."""
    display = frame.copy()
    height, width = display.shape[:2]

    # Face boxes, coloured by what verification concluded about them.
    box_colour = {
        FaceStatus.ENROLLED: COLOUR_OK,
        FaceStatus.UNKNOWN_FACE: COLOUR_ALERT,
        FaceStatus.MULTIPLE_FACES: COLOUR_ALERT,
        FaceStatus.UNVERIFIED: COLOUR_WARN,
    }.get(obs.face_status, COLOUR_MUTED)

    for i, (x, y, w, h) in enumerate(obs.face_boxes):
        cv2.rectangle(display, (x, y), (x + w, y + h), box_colour, 2)
        label = "candidate" if obs.face_status == FaceStatus.ENROLLED else f"face {i + 1}"
        if obs.similarity is not None and len(obs.face_boxes) == 1:
            label += f"  sim {obs.similarity:.3f}"
        cv2.putText(
            display, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_colour, 1
        )

    # Hand skeletons — posture is far more readable than a bare box.
    if obs.hand_analysis is not None:
        for hand in obs.hand_analysis.hands:
            points = hand.landmarks
            if points is not None and len(points) >= 21:
                for a, b in _HAND_LINKS:
                    cv2.line(
                        display,
                        (int(points[a][0]), int(points[a][1])),
                        (int(points[b][0]), int(points[b][1])),
                        COLOUR_HAND,
                        1,
                        cv2.LINE_AA,
                    )
            x1, y1, x2, y2 = hand.bbox
            cv2.putText(
                display,
                hand.handedness.lower(),
                (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                COLOUR_HAND,
                1,
            )

    # Worn devices.
    for detection in obs.wearables.detections if obs.wearables else []:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(display, (x1, y1), (x2, y2), COLOUR_DEVICE, 2)
        cv2.putText(
            display,
            f"{detection.target} {detection.confidence:.2f} ({detection.reliability})",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOUR_DEVICE,
            2,
        )

    # Prohibited objects.
    for obj in obs.prohibited_objects:
        x1, y1, x2, y2 = obj["bbox"]
        cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), COLOUR_ALERT, 2)
        cv2.putText(
            display,
            f"{obj['class_name']} {obj['confidence']:.2f}",
            (int(x1), max(18, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOUR_ALERT,
            2,
        )

    # Top status bar.
    status, colour = _status_line(obs)
    _draw_panel(display, 0, 0, width, 56)
    cv2.putText(display, status, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2)

    mm, ss = divmod(int(elapsed), 60)
    meta = f"{mm:02d}:{ss:02d}   {fps:4.1f} fps   frame {obs.frame_index}   events {event_count}"
    cv2.putText(display, meta, (14, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOUR_MUTED, 1)

    # Left-hand stage readout, so each workflow stage is visibly doing something.
    identity_text = "disabled"
    if identity_enabled:
        if obs.identity_verified is True:
            identity_text = f"match ({obs.similarity:.3f})"
        elif obs.identity_verified is False:
            identity_text = f"no match ({obs.similarity:.3f})"
        else:
            identity_text = "unverified"

    dynamics = obs.facial_dynamics
    hands = obs.hand_analysis

    def _tri(value, yes=COLOUR_ALERT, no=COLOUR_MUTED):
        """Colour a tri-state reading: measured-true, measured-false, unmeasured."""
        return no if value is None else (yes if value else COLOUR_OK)

    rows = [
        ("preprocess", "enhanced (CLAHE)" if obs.was_enhanced else "pass-through", COLOUR_MUTED),
        ("faces", str(obs.face_count if obs.face_count is not None else "-"), COLOUR_MUTED),
        (
            "identity",
            identity_text,
            COLOUR_OK
            if obs.identity_verified
            else (COLOUR_ALERT if obs.identity_verified is False else COLOUR_MUTED),
        ),
    ]

    if dynamics is not None:
        speaking = dynamics.is_speaking
        rows.append(
            ("speaking", "-" if speaking is None else ("yes" if speaking else "no"), _tri(speaking))
        )
        if dynamics.head_pose is not None:
            rows.append(
                (
                    "head",
                    f"y{dynamics.head_pose.yaw:+.0f} p{dynamics.head_pose.pitch:+.0f}",
                    _tri(dynamics.is_looking_away),
                )
            )
        if dynamics.gaze_offset is not None:
            rows.append(("gaze", f"{dynamics.gaze_offset:.2f}", _tri(dynamics.is_gaze_off_screen)))

    if hands is not None:
        detail = str(hands.hands_detected if hands.hands_detected is not None else "-")
        if hands.hand_near_ear:
            detail += " (at ear)"
        elif hands.hand_near_face:
            detail += " (at face)"
        rows.append(("hands", detail, COLOUR_WARN if hands.hand_near_ear else COLOUR_MUTED))

    if obs.wearables is not None and obs.wearables.detections:
        rows.append(("devices", ", ".join(obs.wearable_names), COLOUR_ALERT))

    rows.extend(
        [
            (
                "objects",
                ", ".join(obs.prohibited_object_names) or "none",
                COLOUR_ALERT if obs.prohibited_objects else COLOUR_MUTED,
            ),
            (
                "active",
                ", ".join(obs.active_event_types) or "none",
                COLOUR_ALERT if obs.active_event_types else COLOUR_MUTED,
            ),
            ("latency", f"{obs.timing.total_frame_ms:.1f} ms" if obs.timing else "-", COLOUR_MUTED),
        ]
    )
    panel_h = 22 * len(rows) + 14
    _draw_panel(display, 0, height - panel_h - 26, 360, panel_h)
    y = height - panel_h - 6
    for name, value, value_colour in rows:
        cv2.putText(
            display, f"{name:<11}", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOUR_MUTED, 1
        )
        cv2.putText(display, value[:30], (112, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, value_colour, 1)
        y += 22

    cv2.putText(
        display,
        "q quit   t tab-switch   f fullscreen-exit",
        (14, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        COLOUR_MUTED,
        1,
    )
    return display


# ---------------------------------------------------------------------------
# Session loop
# ---------------------------------------------------------------------------


def run_live_session(args: argparse.Namespace) -> int:
    """Open the camera, run the workflow until the operator quits, then seal the package."""
    print("=" * 62)
    print("  AI PROCTORING — LIVE SESSION")
    print("=" * 62)

    print("\nLoading models:")
    face_detector, face_verifier, object_detector = load_detectors(
        Path(args.models_dir), enable_objects=not args.no_objects
    )
    if face_detector is None:
        print("\nCannot start: the face detector is the minimum requirement for a session.")
        return 1

    print("\nDiscovering camera...")
    discovery = discover_local_camera(
        preferred_index=args.camera,
        target_width=args.width,
        target_height=args.height,
    )
    if not discovery.is_valid:
        print(f"  No usable camera found: {discovery.error_message}")
        return 1
    print(
        f"  Camera {discovery.device_index} via {discovery.backend_name} "
        f"at {discovery.frame_width}x{discovery.frame_height}"
    )

    capture = cv2.VideoCapture(discovery.device_index, discovery.backend_code)
    if not capture.isOpened():
        print("  Camera could not be reopened for the session.")
        return 1
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, args.width, args.height)

    templates: list[np.ndarray] = []
    if args.enroll and face_verifier is not None:
        templates = enroll_candidate(capture, face_detector, face_verifier, args.enroll_samples)

    config = SessionConfig(
        session_id=args.session_id,
        student_name=args.student,
        strictness=args.strictness,
        sampling_fps=args.fps,
        enable_facial_dynamics=not args.no_behaviour,
        enable_hand_analysis=not args.no_behaviour,
        enable_wearable_detection=args.detect_wearables,
        output_dir=args.output_dir,
        reference_templates=templates,
        enable_face_verification=bool(templates) and face_verifier is not None,
        enable_object_detection=object_detector is not None,
        create_zip=args.zip,
        min_event_duration_seconds=args.min_event_duration,
        absence_tolerance_seconds=args.absence_tolerance,
    )
    engine = ProctoringEngine(
        config=config,
        face_detector=face_detector,
        face_verifier=face_verifier,
        object_detector=object_detector,
    )

    behaviour_status = []
    if engine.facial_dynamics is not None and engine.facial_dynamics.is_available:
        behaviour_status.append("speech/pose/gaze")
    if engine.hand_analyzer is not None and engine.hand_analyzer.is_available:
        behaviour_status.append("hands")
    if engine.wearable_detector is not None and engine.wearable_detector.is_available:
        behaviour_status.append("wearables")
    print(
        f"\nStrictness: {config.strictness.value}   "
        f"Behavioural analysis: {', '.join(behaviour_status) or 'none available'}"
    )
    print(f"Session '{config.session_id}' starting — press 'q' in the video window to stop.\n")
    engine.start_session()

    start = time.time()
    frame_index = 0
    displayed_fps = 0.0
    fps_window_start, fps_window_frames = start, 0
    next_sample_at = 0.0
    last_observation: FrameObservation | None = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Camera stopped returning frames — ending session.")
                break

            elapsed = time.time() - start

            # Honour the engine's requested sampling rate.  Inference runs on a
            # subset of frames; the preview still renders every frame so the video
            # stays smooth for the operator.
            if elapsed >= next_sample_at:
                last_observation = engine.process_frame(
                    frame, frame_index=frame_index, timestamp_seconds=elapsed
                )
                frame_index += 1
                next_sample_at = elapsed + (1.0 / max(0.1, engine.target_fps))

            fps_window_frames += 1
            if time.time() - fps_window_start >= 1.0:
                displayed_fps = fps_window_frames / (time.time() - fps_window_start)
                fps_window_start, fps_window_frames = time.time(), 0

            if last_observation is not None:
                canvas = draw_hud(
                    frame,
                    last_observation,
                    displayed_fps,
                    elapsed,
                    len(engine.temporal_aggregator.closed_events),
                    identity_enabled=config.enable_face_verification,
                )
                cv2.imshow(WINDOW_TITLE, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("Stop requested by operator.")
                break
            if key == ord("t"):
                engine.record_browser_event(
                    EventType.BROWSER_TAB_SWITCH,
                    elapsed,
                    "Operator recorded a browser tab switch",
                    frame_index,
                )
                print(f"  [{elapsed:6.1f}s] browser tab switch recorded")
            elif key == ord("f"):
                engine.record_browser_event(
                    EventType.BROWSER_FULLSCREEN_EXIT,
                    elapsed,
                    "Operator recorded a fullscreen exit",
                    frame_index,
                )
                print(f"  [{elapsed:6.1f}s] fullscreen exit recorded")

    except KeyboardInterrupt:
        print("\nInterrupted — finalising the session cleanly.")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print("\nFinalising: closing incidents, capturing evidence, sealing package...")
    summary = engine.finalize_session()
    _print_summary(summary)
    return 0


def _print_summary(summary) -> None:
    """Report what the session produced, in the order the workflow produced it."""
    print("\n" + "=" * 62)
    print("  SESSION COMPLETE")
    print("=" * 62)
    print(f"  Session            {summary.session_id}  ({summary.student_name})")
    print(
        f"  Frames             {summary.total_frames} sampled / "
        f"{summary.processed_frames} processed / {summary.skipped_frames} rejected"
    )
    print(
        f"  Events             {summary.total_events} recorded, {summary.qualified_events} qualified"
    )
    print(f"  Evidence files     {summary.evidence_files_count} validated")

    pruned = summary.evidence_validation.get("pruned_references", 0)
    if pruned:
        print(f"  Evidence pruned    {pruned} unreadable reference(s) removed")

    latency = summary.telemetry.latency_overall
    print(
        f"  Latency            {latency.median_p50_ms:.1f} ms median / {latency.p95_ms:.1f} ms p95"
    )
    print(f"  Package integrity  {'VERIFIED' if summary.integrity_verified else 'FAILED'}")
    for err in summary.integrity_errors[:5]:
        print(f"      ! {err}")

    if summary.events:
        print("\n  Observations for proctor review:")
        for event in summary.events:
            marker = "*" if event.metadata.get("is_duration_qualified", True) else " "
            print(
                f"   {marker} [{event.formatted_start} - {event.formatted_end}] "
                f"{event.event_type.value:<22} {event.severity.value:<8} "
                f"{len(event.evidence)} file(s)"
            )
            print(f"       {event.observation.description}")
    else:
        print("\n  No observations recorded.")

    print(f"\n  Package  {summary.package_dir}")
    if summary.zip_path:
        print(f"  Archive  {summary.zip_path}")
    print("\n  This package records observations only. A human proctor makes the")
    print("  final decision on whether any of it constitutes misconduct.")
    print("=" * 62)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m proctoring.cli live",
        description="Run a live webcam proctoring session through the full workflow.",
    )
    parser.add_argument(
        "--session-id",
        default=f"live_{time.strftime('%Y%m%d_%H%M%S')}",
        help="Identifier for this session and its evidence package.",
    )
    parser.add_argument(
        "--student", default="Candidate", help="Candidate name recorded in the manifest."
    )
    parser.add_argument(
        "--camera", type=int, default=None, help="Camera index (default: auto-discover)."
    )
    parser.add_argument("--width", type=int, default=640, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height.")
    parser.add_argument("--fps", type=float, default=4.0, help="Inference sampling rate.")
    parser.add_argument(
        "--enroll",
        action="store_true",
        help="Capture the candidate's reference face before starting (enables identity verification).",
    )
    parser.add_argument(
        "--enroll-samples",
        type=int,
        default=5,
        help="Reference samples to capture during enrolment.",
    )
    parser.add_argument("--no-objects", action="store_true", help="Skip YOLO object detection.")
    parser.add_argument(
        "--strictness",
        choices=["STANDARD", "STRICT", "MAXIMUM"],
        default="STANDARD",
        help="Exam profile: which behavioural observations are reported and how "
        "quickly they qualify. See docs/accuracy_and_performance.md.",
    )
    parser.add_argument(
        "--no-behaviour",
        action="store_true",
        help="Skip hand, speech, head-pose and gaze analysis.",
    )
    parser.add_argument(
        "--detect-wearables",
        action="store_true",
        help="Enable headphone / earbud / smart-watch detection (adds ~250 ms per "
        "sweep on CPU; earbud detection is unreliable — read the guide).",
    )
    parser.add_argument(
        "--min-event-duration",
        type=float,
        default=1.0,
        help="Seconds a condition must persist to be marked QUALIFIED.",
    )
    parser.add_argument(
        "--absence-tolerance",
        type=float,
        default=1.0,
        help="Seconds a condition may vanish before its incident is closed.",
    )
    parser.add_argument(
        "--models-dir", default="models", help="Directory holding the ONNX model files."
    )
    parser.add_argument(
        "--output-dir",
        default="data/results/live_sessions",
        help="Directory to write the evidence package into.",
    )
    parser.add_argument(
        "--zip", action="store_true", help="Also produce a .zip archive of the package."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_live_session(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
