#!/usr/bin/env python3
"""CLI script for single-pair face detection and verification experiment."""

import argparse
from pathlib import Path
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.face import FaceVerifier


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify whether two images belong to the same identity using YuNet + SFace."
    )
    parser.add_argument("image_a", type=str, help="Path to reference/person image (Image A)")
    parser.add_argument("image_b", type=str, help="Path to test image (Image B)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Custom verification threshold (default: 0.363 for cosine, 1.128 for l2)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["cosine", "l2"],
        default="cosine",
        help="Comparison metric (default: cosine)",
    )
    parser.add_argument(
        "--detector-model",
        type=str,
        default="models/face_detection_yunet_2023mar.onnx",
        help="Path to YuNet ONNX model",
    )
    parser.add_argument(
        "--recognizer-model",
        type=str,
        default="models/face_recognition_sface_2021dec.onnx",
        help="Path to SFace ONNX model",
    )

    args = parser.parse_args()

    path_a = Path(args.image_a)
    path_b = Path(args.image_b)

    if not path_a.exists():
        print(f"Error: Reference image not found at '{path_a}'", file=sys.stderr)
        return 1
    if not path_b.exists():
        print(f"Error: Test image not found at '{path_b}'", file=sys.stderr)
        return 1

    img_a = cv2.imread(str(path_a))
    img_b = cv2.imread(str(path_b))

    if img_a is None:
        print(f"Error: Could not decode image at '{path_a}'", file=sys.stderr)
        return 1
    if img_b is None:
        print(f"Error: Could not decode image at '{path_b}'", file=sys.stderr)
        return 1

    try:
        from src.face import FaceDetector
        detector = FaceDetector(model_path=args.detector_model)
        verifier = FaceVerifier(
            detector=detector,
            recognizer_model_path=args.recognizer_model,
            default_metric=args.metric,
            default_threshold=args.threshold,
        )
    except Exception as e:
        print(f"Initialization error: {e}", file=sys.stderr)
        return 1

    result = verifier.verify(
        image_a=img_a,
        image_b=img_b,
        threshold=args.threshold,
        metric=args.metric,
    )

    print("========================================")
    print("FACE VERIFICATION RESULT")
    print("========================================")
    print(f"Reference Image:           {path_a.name}")
    print(f"Test Image:                {path_b.name}")
    print(f"Reference faces detected:  {result.ref_face_count}")
    print(f"Test faces detected:       {result.test_face_count}")
    print("----------------------------------------")

    if not result.success:
        print(f"Status:                    REJECTED")
        print(f"Reason:                    {result.message}")
        print("========================================")
        return 2

    sim_str = f"{result.similarity:.4f}" if result.similarity is not None else "N/A"
    decision = "SAME PERSON" if result.same_person else "DIFFERENT PERSON"

    print(f"Similarity / Score:        {sim_str}")
    print(f"Threshold:                 {result.threshold:.4f}")
    print(f"Metric:                    {result.distance_metric}")
    print("----------------------------------------")
    print(f"Decision:                  {decision}")
    print("========================================")

    if result.timing_ms:
        print("\nLatency Profile (CPU):")
        for key, val in result.timing_ms.items():
            print(f"  - {key:25s}: {val:6.2f} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
