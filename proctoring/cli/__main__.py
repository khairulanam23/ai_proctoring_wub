"""Command dispatcher for the proctoring CLI.

python -m proctoring.cli live [options]
python -m proctoring.cli video <path> [options]
"""

import sys


def _run_video(argv: list[str]) -> int:
    """Run the workflow over a recorded video file and seal an evidence package."""
    import argparse
    from pathlib import Path

    from proctoring.cli.live import _print_summary
    from proctoring.config import SessionConfig
    from proctoring.detection.face_detector import FaceDetector
    from proctoring.detection.face_verifier import FaceVerifier
    from proctoring.detection.object_detector import ObjectDetector
    from proctoring.engine import ProctoringEngine
    from proctoring.storage import ProctoringStorage

    parser = argparse.ArgumentParser(prog="python -m proctoring.cli video")
    parser.add_argument("video", help="Path to the recorded examination video.")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--student", default="Candidate")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--no-objects", action="store_true")
    parser.add_argument(
        "--strictness",
        choices=["STANDARD", "STRICT", "MAXIMUM"],
        default="STANDARD",
        help="Exam profile governing which behavioural observations are reported.",
    )
    parser.add_argument(
        "--no-behaviour",
        action="store_true",
        help="Skip hand, speech, head-pose and gaze analysis.",
    )
    parser.add_argument(
        "--detect-wearables",
        action="store_true",
        help="Enable headphone / earbud / smart-watch detection (slow).",
    )
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args(argv)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 1

    models_dir = Path(args.models_dir)
    yunet = models_dir / "face_detection_yunet_2023mar.onnx"
    if not yunet.exists():
        print(f"YuNet model missing at {yunet}. Run: python scripts/download_models.py")
        return 1

    face_detector = FaceDetector(model_path=yunet)
    sface = models_dir / "face_recognition_sface_2021dec.onnx"
    face_verifier = (
        FaceVerifier(detector=face_detector, recognizer_model_path=sface)
        if sface.exists()
        else None
    )

    object_detector = None
    if not args.no_objects:
        try:
            object_detector = ObjectDetector()
        except Exception as exc:
            print(f"Object detection disabled ({exc})")

    storage = ProctoringStorage(args.data_root)
    config = SessionConfig(
        session_id=args.session_id or storage.session_id(args.student),
        student_name=args.student,
        strictness=args.strictness,
        sampling_fps=args.fps,
        output_dir=args.output_dir or storage.sessions_root,
        enable_facial_dynamics=not args.no_behaviour,
        enable_hand_analysis=not args.no_behaviour,
        enable_wearable_detection=args.detect_wearables,
        enable_object_detection=object_detector is not None,
        # No enrolment template is available for an arbitrary recording, so
        # identity verification is left off rather than reporting every face
        # as unknown.
        reference_templates=storage.load_templates(args.student),
        enable_face_verification=bool(storage.load_templates(args.student)),
        create_zip=args.zip,
    )
    engine = ProctoringEngine(
        config=config,
        face_detector=face_detector,
        face_verifier=face_verifier,
        object_detector=object_detector,
    )

    print(f"Processing {video_path} at {args.fps} fps...")
    processed = {"n": 0}

    def on_frame(observation) -> None:
        processed["n"] += 1
        if processed["n"] % 25 == 0:
            print(f"  {observation.timestamp_seconds:7.1f}s  {observation.face_status}")

    summary = engine.run_video(video_path, progress_callback=on_frame)
    _print_summary(summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        print(
            "Commands:\n  live    Run a live webcam session\n  video   Process a recorded video file"
        )
        return 0

    command, rest = argv[0], argv[1:]
    if command == "live":
        from proctoring.cli.live import main as live_main

        return live_main(rest)
    if command == "video":
        return _run_video(rest)

    print(f"Unknown command '{command}'. Use 'live' or 'video'.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
