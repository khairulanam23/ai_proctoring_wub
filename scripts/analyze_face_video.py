#!/usr/bin/env python3
"""CLI script for analyzing prerecorded video for face presence state transitions and evidence logging."""

import argparse
from pathlib import Path
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face import FaceDetector, FacePresenceAnalyzer, VideoPresenceAnalyzer


def handle_event(event) -> None:
    """Print state transition event to console."""
    print(f"[{event.formatted_timestamp}] State: {event.status.value:<16s} (Frame: {event.frame_index:5d})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze prerecorded video for face presence state transitions and evidence logging."
    )
    parser.add_argument("video_path", type=str, help="Path to video file (e.g. mp4, avi)")
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Process every N-th frame (default: 1, full analysis)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Detection confidence threshold (default: 0.6)",
    )
    parser.add_argument(
        "--detector-model",
        type=str,
        default="models/face_detection_yunet_2023mar.onnx",
        help="Path to YuNet ONNX model",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="data/results/video_presence_evidence.json",
        help="Path to save evidence JSON report (default: data/results/video_presence_evidence.json)",
    )

    args = parser.parse_args()
    video_path = Path(args.video_path)

    if not video_path.exists():
        print(f"Error: Video file not found at '{video_path}'", file=sys.stderr)
        return 1

    try:
        detector = FaceDetector(
            model_path=args.detector_model,
            score_threshold=args.threshold,
        )
        presence_analyzer = FacePresenceAnalyzer(detector=detector)
        video_analyzer = VideoPresenceAnalyzer(presence_analyzer=presence_analyzer)
    except Exception as e:
        print(f"Initialization error: {e}", file=sys.stderr)
        return 1

    print("============================================================")
    print("VIDEO FACE PRESENCE MONITORING")
    print("============================================================")
    print(f"Target Video: {video_path.name}")
    print("Processing video stream and logging state transitions...")
    print("------------------------------------------------------------")

    summary = video_analyzer.analyze_video(
        video_path=video_path,
        frame_step=args.step,
        on_event_callback=handle_event,
    )

    print("\n------------------------------------------------------------")
    print("ANALYSIS SUMMARY")
    print("------------------------------------------------------------")
    print(f"Source Media:         {summary.source_media}")
    print(f"Total Frames:         {summary.total_frames}")
    print(f"Processed Frames:     {summary.processed_frames}")
    print(f"Stream FPS:           {summary.fps:.2f}")
    print(f"Duration:             {summary.duration_sec:.2f} s")
    print(f"Total State Events:   {len(summary.events)}")
    print("Status Distribution:")
    for status, count in summary.status_distribution.items():
        pct = (count / summary.processed_frames * 100) if summary.processed_frames > 0 else 0
        print(f"  - {status:15s}: {count:5d} frames ({pct:5.1f}%)")
    print("Latency Profile (CPU):")
    print(f"  - Average Inference: {summary.avg_inference_time_ms:6.2f} ms")
    print(f"  - Min Inference:     {summary.min_inference_time_ms:6.2f} ms")
    print(f"  - Max Inference:     {summary.max_inference_time_ms:6.2f} ms")
    print("============================================================")

    if args.output_json:
        saved_path = summary.save_evidence_json(args.output_json)
        print(f"Structured evidence report saved to: {saved_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
