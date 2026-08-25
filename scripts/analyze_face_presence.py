#!/usr/bin/env python3
"""CLI script for face presence and multi-face analysis on a single image."""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from proctoring.detection import FaceDetector, FacePresenceAnalyzer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze face presence status (NO_FACE, SINGLE_FACE, MULTIPLE_FACES) in an image."
    )
    parser.add_argument("image_path", type=str, help="Path to input image file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Confidence score threshold for YuNet detector (default: 0.6)",
    )
    parser.add_argument(
        "--detector-model",
        type=str,
        default="models/face_detection_yunet_2023mar.onnx",
        help="Path to YuNet ONNX model file",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save structured detection evidence as JSON",
    )

    args = parser.parse_args()
    img_path = Path(args.image_path)

    if not img_path.exists():
        print(f"Error: Image file not found at '{img_path}'", file=sys.stderr)
        return 1

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Error: Could not decode image at '{img_path}'", file=sys.stderr)
        return 1

    try:
        detector = FaceDetector(
            model_path=args.detector_model,
            score_threshold=args.threshold,
        )
        analyzer = FacePresenceAnalyzer(detector=detector)
    except Exception as e:
        print(f"Initialization error: {e}", file=sys.stderr)
        return 1

    result = analyzer.analyze(image)

    print("========================================")
    print("FACE PRESENCE ANALYSIS")
    print("========================================")
    print(f"Source Image:     {img_path.name}")
    print(f"Image Resolution: {result.image_shape[1]}x{result.image_shape[0]}")
    print(f"Status:           {result.status.value}")
    print(f"Faces detected:   {result.face_count}")
    print(f"Inference Time:   {result.inference_time_ms:.2f} ms")
    print("----------------------------------------")

    if result.face_count == 0:
        print("No faces detected in the provided image.")
    else:
        for idx, face in enumerate(result.faces, start=1):
            x, y, w, h = face.bbox
            print(f"Face #{idx}")
            print(f"  Confidence:   {face.confidence:.4f}")
            print(f"  Bounding box: x={x}, y={y}, w={w}, h={h}")
            if face.landmarks:
                r_eye, l_eye, nose, r_mouth, l_mouth = face.landmarks
                print(
                    f"  Landmarks:    R.Eye=({r_eye[0]:.1f}, {r_eye[1]:.1f}), L.Eye=({l_eye[0]:.1f}, {l_eye[1]:.1f}), Nose=({nose[0]:.1f}, {nose[1]:.1f})"
                )

    print("========================================")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        evidence = {
            "source_image": img_path.name,
            "source_path": str(img_path.resolve()),
            **result.to_dict(),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2)
        print(f"Evidence saved to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
