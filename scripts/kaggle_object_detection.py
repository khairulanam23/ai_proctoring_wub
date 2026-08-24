#!/usr/bin/env python3
"""Kaggle GPU Object Detection, Video Sampling, Temporal Analysis, and Evidence Packaging Runner."""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.object_detection import (
    ObjectDetector,
    ObjectRelevanceFilter,
    ProctoringDetectionReport,
    VideoObjectAnalyzer,
)


def print_environment_diagnostics() -> None:
    """Inspect and report Python, PyTorch, CUDA, and Ultralytics runtime state."""
    print("=" * 60)
    print("AI PROCTORING — KAGGLE GPU ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)
    print(f"Python Version:      {sys.version.split()[0]}")

    try:
        import torch
        print(f"PyTorch Version:     {torch.__version__}")
        print(f"CUDA Available:      {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Device Count:    {torch.cuda.device_count()}")
            print(f"GPU Device Name:     {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version:        {torch.version.cuda}")
        else:
            print("GPU Device Name:     None (Running in CPU mode)")
    except ImportError:
        print("PyTorch Version:     Not installed")

    try:
        import ultralytics
        print(f"Ultralytics Version: {ultralytics.__version__}")
    except ImportError:
        print("Ultralytics:         Not installed")

    print("=" * 60)


def run_image_benchmark(
    detector: ObjectDetector,
    relevance_filter: ObjectRelevanceFilter,
    image: np.ndarray,
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
) -> None:
    """Run warm-up and timed benchmark iterations on a single image."""
    print("\n" + "=" * 60)
    print(f"BENCHMARKING IMAGE OBJECT DETECTION ({detector.device.upper()})")
    print("=" * 60)
    print(f"Model:               {detector.model_name}")
    print(f"Device:              {detector.device}")
    print(f"Input Resolution:    {image.shape[1]}x{image.shape[0]} px")
    print(f"Warm-up Iterations:  {warmup_iters}")
    print(f"Benchmark Iterations:{benchmark_iters}")
    print("-" * 60)

    # 1. Warm-up Phase
    for _ in range(warmup_iters):
        raw = detector.detect(image)
        _ = relevance_filter.filter(raw)

    # 2. Timed Benchmark Phase
    infer_latencies_ms = []
    total_latencies_ms = []

    for _ in range(benchmark_iters):
        t0 = time.perf_counter()
        raw = detector.detect(image)
        _ = relevance_filter.filter(raw)
        total_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        infer_latencies_ms.append(raw.inference_time_ms)

    avg_infer_ms = float(np.mean(infer_latencies_ms))
    avg_total_ms = float(np.mean(total_latencies_ms))
    min_infer_ms = float(np.min(infer_latencies_ms))
    max_infer_ms = float(np.max(infer_latencies_ms))
    approx_fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0

    print("IMAGE BENCHMARK PERFORMANCE METRICS:")
    print(f"  • Model Inference (Avg):  {avg_infer_ms:6.2f} ms [Min: {min_infer_ms:6.2f} ms, Max: {max_infer_ms:6.2f} ms]")
    print(f"  • Total Pipeline (Avg):   {avg_total_ms:6.2f} ms")
    print(f"  • Approximate Throughput: {approx_fps:6.1f} FPS")
    print("=" * 60)


def run_video_pipeline(
    detector: ObjectDetector,
    relevance_filter: ObjectRelevanceFilter,
    video_path: Path,
    sample_fps: float = 2.0,
    absence_tolerance: float = 1.0,
    min_event_duration: float = 1.0,
    show_all: bool = False,
    save_frames: bool = False,
    frames_dir: Path = Path("data/results/object_detection/video/frames/"),
    plot_timeline: Optional[Path] = None,
    save_evidence: bool = False,
    evidence_dir: Path = Path("data/results/object_detection/evidence"),
    zip_evidence: bool = False,
    json_path: Optional[Path] = None,
) -> None:
    """Run full video frame sampling, temporal analysis, and evidence packaging on Kaggle GPU."""
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

    print(f"\nProcessing video stream on {detector.device.upper()}: {video_path.name} at {sample_fps} FPS sampling rate...")
    report = analyzer.analyze_video(
        video_path=video_path,
        target_sampling_fps=sample_fps,
        save_annotated_dir=save_dir,
        plot_timeline_path=plot_timeline,
        save_evidence_dir=ev_dir,
        zip_evidence=zip_evidence,
        show_all=show_all,
    )

    meta = report.metadata
    print("\n" + "=" * 60)
    print("KAGGLE GPU VIDEO ANALYSIS & EVIDENCE REPORT")
    print("=" * 60)
    print(f"Video File:          {Path(report.video_path).name} ({meta.width}x{meta.height})")
    print(f"Source FPS:          {meta.source_fps:.2f}")
    print(f"Total Video Frames:  {meta.total_frames} (Duration: {meta.duration_seconds:.2f}s)")
    print(f"Sampled Frames:      {report.total_sampled_frames} frames (at {report.target_sampling_fps} FPS)")
    print(f"Execution Device:    {detector.device.upper()}")
    print("-" * 60)
    print("CONSOLIDATED TEMPORAL EVENTS:")
    for ev in report.temporal_events:
        qual_str = "[Qualified]" if ev.is_duration_qualified else "[Short-Lived]"
        print(f"  • {ev.object_class:15s} {ev.formatted_start} → {ev.formatted_end} (Duration: {ev.duration_seconds:4.1f}s, Max Conf: {ev.max_confidence:.2f}) {qual_str}")
    if not report.temporal_events:
        print("  (None)")

    print("-" * 60)
    print("PERSON COUNT TRANSITIONS:")
    for chg in report.person_count_changes:
        print(f"  [{chg.formatted_timestamp}] Person Count: {chg.previous_count} → {chg.new_count}")
    if not report.person_count_changes:
        print("  (Stable person count across video stream)")

    print("-" * 60)
    print("GPU THROUGHPUT & LATENCY:")
    print(f"  • Avg Inference Latency:   {report.average_inference_time_ms:6.2f} ms")
    print(f"  • Avg Frame Process Time:  {report.average_frame_processing_time_ms:6.2f} ms")
    print(f"  • Processing Speed:        {report.approximate_fps:6.1f} FPS")
    print("=" * 60)

    if save_evidence and report.evidence_package is not None:
        pkg = report.evidence_package
        m = pkg.manifest
        print("EVIDENCE PACKAGE SUMMARY:")
        print(f"  • Package Path:            {pkg.package_dir}")
        print(f"  • Manifest File:           {pkg.package_dir / 'manifest.json'}")
        print(f"  • Key Evidence Frames:     {len(m.frames)}")
        print(f"  • Object Crops Generated:  {m.summary.get('object_crops_count', 0)}")
        print(f"  • Temporal Events Logged:  {len(m.events)}")
        if pkg.zip_path is not None:
            print(f"  • ZIP Package Archive:     {pkg.zip_path}")
        print("=" * 60)

    if save_frames:
        print(f"Annotated frames saved to: {frames_dir}")

    if plot_timeline:
        print(f"Gantt timeline chart saved to: {plot_timeline}")

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"JSON evidence report saved to: {json_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kaggle GPU object detection, temporal analysis, and evidence packaging runner."
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Path to single test image file.",
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default=None,
        help="Path to video file for frame sampling timeline analysis.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Target inference sampling rate in FPS for video processing (default: 2.0).",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yolo11n.pt",
        help="YOLO model name or weights path (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.25,
        help="Base confidence cutoff threshold (default: 0.25).",
    )
    parser.add_argument(
        "--absence-tolerance",
        type=float,
        default=1.0,
        help="Absence gap tolerance in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--min-event-duration",
        type=float,
        default=1.0,
        help="Minimum event duration in seconds (default: 1.0).",
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
        help="Show both relevant and ignored objects in visualization.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save annotated visual frames for sampled video frames.",
    )
    parser.add_argument(
        "--plot-timeline",
        type=str,
        default=None,
        help="Path to export Gantt timeline chart image.",
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
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warm-up iterations for image benchmark (default: 5).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of benchmark iterations for image benchmark (default: 20).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/results/object_detection/kaggle_result.jpg",
        help="Output path for annotated image.",
    )

    args = parser.parse_args()

    # 1. Diagnostics
    print_environment_diagnostics()

    # 2. Initialize Detector & Relevance Filter
    try:
        detector = ObjectDetector(
            model_name=args.model,
            confidence_threshold=args.threshold,
            device=args.device,
        )
    except Exception as e:
        print(f"\n[Model Loading Error] {e}", file=sys.stderr)
        return 2

    relevance_filter = ObjectRelevanceFilter(default_threshold=args.threshold)

    # 3. Video Pipeline Execution
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video file '{args.video}' not found.", file=sys.stderr)
            return 1
        run_video_pipeline(
            detector=detector,
            relevance_filter=relevance_filter,
            video_path=video_path,
            sample_fps=args.sample_fps,
            absence_tolerance=args.absence_tolerance,
            min_event_duration=args.min_event_duration,
            show_all=args.show_all,
            save_frames=args.save_frames,
            plot_timeline=Path(args.plot_timeline) if args.plot_timeline else None,
            save_evidence=args.save_evidence,
            evidence_dir=Path(args.evidence_dir),
            zip_evidence=args.zip_evidence,
            json_path=Path(args.json) if args.json else None,
        )
        return 0

    # 4. Single Image Execution (Default Fallback)
    image_path = Path(args.image) if args.image else Path("data/samples/Colin_Powell/Colin_Powell_0001.jpg")
    if not image_path.exists():
        fallback_candidates = [
            Path("data/samples/Colin_Powell/Colin_Powell_0001.jpg"),
            Path("data/samples/synthetic/single_face.jpg"),
        ]
        for fb in fallback_candidates:
            if fb.exists():
                image_path = fb
                break

    if not image_path.exists():
        print(f"Error: Test image not found at '{args.image}'.", file=sys.stderr)
        return 1

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Failed to decode image from '{image_path}'", file=sys.stderr)
        return 1

    print(f"\nRunning detection on: {image_path.name} ({image.shape[1]}x{image.shape[0]} px)...")
    raw_result = detector.detect(image)
    report = relevance_filter.filter(raw_result)

    print("------------------------------------------------------------")
    print(f"Total Detections:    {report.total_detections}")
    print(f"Person Count:        {report.person_count}")
    print(f"Relevant Detections: {report.relevant_count}")
    for obj in report.relevant_objects:
        print(f"  • {obj.class_name:16s}: {obj.confidence:.4f} [x1={obj.x1}, y1={obj.y1}, x2={obj.x2}, y2={obj.y2}]")
    print("------------------------------------------------------------")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis = relevance_filter.visualize_report(image, report, show_all=args.show_all)
    cv2.imwrite(str(out_path), vis)
    print(f"Visualization saved to: {out_path}")

    run_image_benchmark(
        detector=detector,
        relevance_filter=relevance_filter,
        image=image,
        warmup_iters=args.warmup,
        benchmark_iters=args.iterations,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
