#!/usr/bin/env python3
"""CLI script for running object detection and proctoring relevance filtering on an image."""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from proctoring.detection import (
    ObjectDetector,
    ObjectRelevanceFilter,
)


def run_detection(
    image_path: Path,
    model_name: str = "yolo11n.pt",
    confidence_threshold: float = 0.25,
    device: str = None,
    show_all: bool = False,
    output_json: bool = False,
    output_path: Path = Path("data/results/object_detection/object_detection.jpg"),
) -> int:
    """Run object detection, apply relevance filtering, print report, and save visualization."""
    if not image_path.exists():
        print(f"Error: Image not found at '{image_path}'", file=sys.stderr)
        return 1

    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        print(f"Error: Could not decode image at '{image_path}'", file=sys.stderr)
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

    # Initialize relevance filter
    relevance_filter = ObjectRelevanceFilter(default_threshold=confidence_threshold)

    # 1. Execute raw detection
    raw_result = detector.detect(image)

    # 2. Apply proctoring relevance filtering
    report = relevance_filter.filter(raw_result)

    if output_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        # Print clean formatted console summary
        print("============================================================")
        print("PROCTORING OBJECT DETECTION & RELEVANCE REPORT")
        print("============================================================")
        print(f"Model:               {raw_result.model_name}")
        print(f"Device:              {raw_result.device}")
        print(
            f"Image:               {image_path.name} ({report.image_width}x{report.image_height})"
        )
        print(f"Inference Latency:   {report.inference_time_ms:.2f} ms")
        print(f"Person Count:        {report.person_count}")
        print(
            f"Relevant Objects:    {report.relevant_count} / {report.total_detections} total detected"
        )
        print("------------------------------------------------------------")

        print("Relevant Proctoring Objects:")
        if report.relevant_count == 0:
            print("  (None)")
        else:
            for obj in report.relevant_objects:
                print(
                    f"  • {obj.class_name:18s} {obj.confidence:.4f}  [bbox: ({obj.x1}, {obj.y1}, {obj.x2}, {obj.y2})]"
                )

        if report.ignored_objects:
            print("\nIgnored / Background Objects:")
            for obj in report.ignored_objects:
                print(
                    f"  - {obj.class_name:18s} {obj.confidence:.4f}  [Filtered: non-relevant or below threshold]"
                )

        print("============================================================")

    # 3. Generate and save visualization
    vis_image = relevance_filter.visualize_report(image, report, show_all=show_all)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_image)
    if not output_json:
        mode_str = (
            " (Debug Mode: showing all detections)" if show_all else " (Relevant objects only)"
        )
        print(f"Visualization saved to: {output_path}{mode_str}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect and filter proctoring-relevant objects in an image using YOLO."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to input image file.",
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
        help="Base confidence cutoff threshold (default: 0.25).",
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
        help="Debug mode: show both relevant and ignored background objects in visualization.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured proctoring detection report as JSON.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/results/object_detection/object_detection.jpg",
        help="Path to save annotated output visualization.",
    )

    args = parser.parse_args()
    return run_detection(
        image_path=Path(args.image_path),
        model_name=args.model,
        confidence_threshold=args.threshold,
        device=args.device,
        show_all=args.show_all,
        output_json=args.json,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    sys.exit(main())
