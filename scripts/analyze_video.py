#!/usr/bin/env python3
"""CLI script for video frame sampling, object detection, temporal analysis, and evidence packaging."""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from proctoring.detection import (
    ObjectDetector,
    ObjectRelevanceFilter,
    VideoObjectAnalyzer,
)


def run_video_analysis(
    video_path: Path,
    sample_fps: float = 2.0,
    model_name: str = "yolo11n.pt",
    confidence_threshold: float = 0.25,
    absence_tolerance: float = 1.0,
    min_event_duration: float = 1.0,
    device: str = None,
    show_all: bool = False,
    save_frames: bool = False,
    frames_dir: Path = Path("data/results/object_detection/video/frames/"),
    plot_timeline: Path | None = None,
    save_evidence: bool = False,
    evidence_dir: Path = Path("data/results/object_detection/evidence"),
    zip_evidence: bool = False,
    json_path: Path | None = None,
) -> int:
    """Execute video sampling, object detection, temporal analysis, and evidence packaging."""
    if not video_path.exists():
        print(f"Error: Video file not found at '{video_path}'", file=sys.stderr)
        return 1

    try:
        detector = ObjectDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device,
        )
    except ImportError as e:
        print(f"\n[Environment Notice] {e}", file=sys.stderr)
        return 2

    relevance_filter = ObjectRelevanceFilter(default_threshold=confidence_threshold)
    analyzer = VideoObjectAnalyzer(
        detector=detector,
        relevance_filter=relevance_filter,
        target_sampling_fps=sample_fps,
        absence_tolerance_seconds=absence_tolerance,
        min_event_duration_seconds=min_event_duration,
        enable_temporal_events=True,
    )

    save_dir = frames_dir if save_frames else None
    ev_dir = evidence_dir if save_evidence else None

    # Run analysis & evidence packaging
    report = analyzer.analyze_video(
        video_path=video_path,
        target_sampling_fps=sample_fps,
        save_annotated_dir=save_dir,
        plot_timeline_path=plot_timeline,
        save_evidence_dir=ev_dir,
        zip_evidence=zip_evidence,
        show_all=show_all,
    )

    # Print clean formatted summary
    meta = report.metadata
    print("============================================================")
    print("PROCTORING VIDEO OBJECT DETECTION & EVIDENCE REPORT")
    print("============================================================")
    print(f"Video File:          {Path(report.video_path).name} ({meta.width}x{meta.height})")
    print(f"Source FPS:          {meta.source_fps:.2f}")
    print(f"Total Video Frames:  {meta.total_frames} (Duration: {meta.duration_seconds:.2f}s)")
    print(f"Sampling Rate:       {report.target_sampling_fps:.2f} FPS")
    print(f"Sampled Frames:      {report.total_sampled_frames} frames")
    print(f"Absence Tolerance:   {absence_tolerance:.2f} s")
    print(f"Min Event Duration:  {min_event_duration:.2f} s")
    print(f"Execution Device:    {detector.device}")
    print("------------------------------------------------------------")
    print("CONSOLIDATED TEMPORAL OBJECT EVENTS:")
    if not report.temporal_events:
        print("  (No temporal object presence events recorded)")
    else:
        for ev in report.temporal_events:
            qual_str = "[Qualified]" if ev.is_duration_qualified else "[Short-Lived]"
            print(
                f"  • {ev.object_class:15s} {ev.formatted_start} → {ev.formatted_end} (Duration: {ev.duration_seconds:4.1f}s, {ev.detection_count:2d} samples, Max Conf: {ev.max_confidence:.2f}) {qual_str}"
            )

    print("------------------------------------------------------------")
    print("PERSON COUNT TRANSITIONS:")
    if not report.person_count_changes:
        print("  (No person count transitions observed — stable presence)")
    else:
        for chg in report.person_count_changes:
            print(
                f"  [{chg.formatted_timestamp}] Person Count Changed: {chg.previous_count} → {chg.new_count} (Frame: {chg.frame_index:5d})"
            )

    print("------------------------------------------------------------")
    print("RAW TIMELINE EVIDENCE (Sample of frames):")
    for entry in report.timeline[:8]:
        rel_str = ", ".join(
            [f"{o.class_name} ({o.confidence:.2f})" for o in entry.relevant_objects]
        )
        if not rel_str:
            rel_str = "(none)"
        print(
            f"  [{entry.formatted_timestamp}] Frame {entry.frame_index:5d} | Persons: {entry.person_count} | Relevant: {rel_str}"
        )
    if len(report.timeline) > 8:
        print(f"  ... and {len(report.timeline) - 8} more sampled frames in raw timeline.")

    print("------------------------------------------------------------")
    print("PERFORMANCE BREAKDOWN:")
    print(f"  • Avg Inference Latency:   {report.average_inference_time_ms:6.2f} ms")
    print(f"  • Avg Frame Process Time:  {report.average_frame_processing_time_ms:6.2f} ms")
    print(f"  • Total Video Processing:  {report.total_processing_time_ms / 1000.0:6.2f} s")
    print(f"  • Processing Throughput:   {report.approximate_fps:6.1f} FPS")
    print("============================================================")

    if save_evidence and report.evidence_package is not None:
        pkg = report.evidence_package
        m = pkg.manifest
        print("EVIDENCE PACKAGE ARTIFACTS:")
        print(f"  • Package Directory:       {pkg.package_dir}")
        print(f"  • Manifest:                {pkg.package_dir / 'manifest.json'}")
        print(f"  • Key Evidence Frames:     {len(m.frames)}")
        print(f"  • Object Crops Generated:  {m.summary.get('object_crops_count', 0)}")
        print(f"  • Temporal Events Logged:  {len(m.events)}")
        if pkg.zip_path is not None:
            print(f"  • ZIP Package Archive:     {pkg.zip_path}")
        print("============================================================")

    if save_frames:
        print(f"Sampled frames saved to: {frames_dir}")

    if plot_timeline:
        print(f"Gantt timeline chart saved to: {plot_timeline}")

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"JSON timeline report exported to: {json_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample video frames, run proctoring temporal analysis, and build structured evidence package."
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file (e.g. mp4, avi, mkv).",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Target inference sampling rate in frames per second (default: 2.0).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="YOLO model identifier or weights path (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.25,
        help="Confidence cutoff threshold (default: 0.25).",
    )
    parser.add_argument(
        "--absence-tolerance",
        type=float,
        default=1.0,
        help="Maximum absence gap in seconds to bridge continuous events (default: 1.0).",
    )
    parser.add_argument(
        "--min-event-duration",
        type=float,
        default=1.0,
        help="Minimum event duration in seconds for qualification (default: 1.0).",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        help="Inference device: 'cuda', 'cpu', or auto-detect.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Debug mode: include filtered/ignored background objects in annotated frames.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save annotated visual frames for each sampled frame.",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="data/results/object_detection/video/frames/",
        help="Directory where annotated frames will be saved.",
    )
    parser.add_argument(
        "--plot-timeline",
        type=str,
        default=None,
        help="Path to save publication-quality Gantt timeline chart image.",
    )
    parser.add_argument(
        "--save-evidence",
        action="store_true",
        help="Generate structured multi-modal evidence package with manifest, key frames, and crops.",
    )
    parser.add_argument(
        "--evidence-dir",
        type=str,
        default="data/results/object_detection/evidence",
        help="Directory path for the evidence package.",
    )
    parser.add_argument(
        "--zip-evidence",
        action="store_true",
        help="Package the evidence directory into an accompanying .zip archive.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to export structured JSON timeline evidence report.",
    )

    args = parser.parse_args()
    return run_video_analysis(
        video_path=Path(args.video_path),
        sample_fps=args.sample_fps,
        model_name=args.model,
        confidence_threshold=args.threshold,
        absence_tolerance=args.absence_tolerance,
        min_event_duration=args.min_event_duration,
        device=args.device,
        show_all=args.show_all,
        save_frames=args.save_frames,
        frames_dir=Path(args.frames_dir),
        plot_timeline=Path(args.plot_timeline) if args.plot_timeline else None,
        save_evidence=args.save_evidence,
        evidence_dir=Path(args.evidence_dir),
        zip_evidence=args.zip_evidence,
        json_path=Path(args.json) if args.json else None,
    )


if __name__ == "__main__":
    sys.exit(main())
