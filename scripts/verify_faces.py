#!/usr/bin/env python3
"""CLI script for single-pair and multi-reference face verification experiments."""

import argparse
from pathlib import Path
import sys
from typing import List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.face import FaceDetector, FacePreprocessor, FaceVerifier


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify candidate identity using YuNet detection, 5-point alignment, and SFace embeddings."
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Image paths. If 2 paths: [ref_image, test_image]. If multiple with --multi: all except last are references.",
    )
    parser.add_argument(
        "--ref",
        type=str,
        nargs="+",
        default=None,
        help="One or more reference image paths for multi-image enrollment template.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Test image path to verify.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3630,
        help="Verification threshold (default: 0.3630 for cosine)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["cosine", "l2"],
        default="cosine",
        help="Comparison metric (default: cosine)",
    )
    parser.add_argument(
        "--illumination",
        type=str,
        choices=["none", "mild_contrast", "clahe"],
        default="none",
        help="Illumination normalization mode (default: none)",
    )

    args = parser.parse_args()

    # Determine reference and test image paths
    ref_paths: List[Path] = []
    test_path: Path

    if args.ref and args.test:
        ref_paths = [Path(p) for p in args.ref]
        test_path = Path(args.test)
    elif len(args.images) >= 2:
        ref_paths = [Path(p) for p in args.images[:-1]]
        test_path = Path(args.images[-1])
    else:
        parser.print_help()
        print("\nError: Please provide at least one reference image and one test image.", file=sys.stderr)
        return 1

    # Validate file paths
    for rp in ref_paths:
        if not rp.exists():
            print(f"Error: Reference image not found at '{rp}'", file=sys.stderr)
            return 1
    if not test_path.exists():
        print(f"Error: Test image not found at '{test_path}'", file=sys.stderr)
        return 1

    # Read test image
    test_img = cv2.imread(str(test_path))
    if test_img is None:
        print(f"Error: Could not decode test image at '{test_path}'", file=sys.stderr)
        return 1

    detector = FaceDetector()
    preprocessor = FacePreprocessor(detector=detector, illumination_mode=args.illumination)
    verifier = FaceVerifier(detector=detector, preprocessor=preprocessor, default_threshold=args.threshold)

    # 1. Single Reference Case
    if len(ref_paths) == 1:
        ref_path = ref_paths[0]
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            print(f"Error: Could not decode reference image at '{ref_path}'", file=sys.stderr)
            return 1

        result = verifier.verify(ref_img, test_img, threshold=args.threshold, metric=args.metric)

        print("============================================================")
        print("FACE VERIFICATION REPORT (SINGLE REFERENCE)")
        print("============================================================")
        print(f"Reference Image:           {ref_path.name}")
        print(f"Test Image:                {test_path.name}")
        print(f"Reference Faces Detected:  {result.ref_face_count}")
        print(f"Test Faces Detected:       {result.test_face_count}")
        print("------------------------------------------------------------")

        if not result.success:
            print(f"Status:                    REJECTED")
            print(f"Reason:                    {result.message}")
            print("============================================================")
            return 2

        if result.ref_preprocessing and result.test_preprocessing:
            qr = result.ref_preprocessing.quality
            qt = result.test_preprocessing.quality
            pr = qr.pose if qr else None
            pt = qt.pose if qt else None
            print(f"Ref Quality:               Status={result.ref_preprocessing.status.value}, Conf={qr.confidence:.3f}, Blur={qr.blur_score:.1f}")
            if pr:
                print(f"Ref Pose:                  YawRatio={pr.yaw_ratio:.2f}, Roll={pr.roll_angle_deg:.1f}°")
            print(f"Test Quality:              Status={result.test_preprocessing.status.value}, Conf={qt.confidence:.3f}, Blur={qt.blur_score:.1f}")
            if pt:
                print(f"Test Pose:                 YawRatio={pt.yaw_ratio:.2f}, Roll={pt.roll_angle_deg:.1f}°")
            print("------------------------------------------------------------")

        decision = "SAME PERSON" if result.same_person else "DIFFERENT PERSON"
        print(f"Cosine Similarity Score:   {result.similarity:.4f}")
        print(f"Calibrated Threshold:      {result.threshold:.4f}")
        print(f"Decision:                  {decision}")
        print("============================================================")
        if result.timing_ms:
            print("\nLatency Profile (CPU):")
            for k, v in result.timing_ms.items():
                print(f"  - {k:25s}: {v:6.2f} ms")
        return 0

    # 2. Multi-Reference Enrollment Case
    else:
        ref_imgs = []
        for rp in ref_paths:
            img = cv2.imread(str(rp))
            if img is not None:
                ref_imgs.append(img)

        m_res = verifier.verify_multi_reference(
            reference_images=ref_imgs,
            test_image=test_img,
            threshold=args.threshold,
            metric=args.metric,
        )

        print("============================================================")
        print("FACE VERIFICATION REPORT (MULTI-IMAGE ENROLLMENT)")
        print("============================================================")
        print(f"Total Enrolled References: {m_res.total_reference_count}")
        print(f"Valid Quality References:  {m_res.valid_reference_count}")
        print(f"Test Image:                {test_path.name}")
        print("------------------------------------------------------------")

        if not m_res.success:
            print(f"Status:                    REJECTED")
            print(f"Reason:                    {m_res.message}")
            print("============================================================")
            return 2

        decision = "SAME PERSON" if m_res.same_person else "DIFFERENT PERSON"
        print(f"Individual Match Scores:   {[round(s, 4) for s in m_res.individual_scores]}")
        print(f"Template Similarity:       {m_res.template_similarity:.4f}")
        print(f"Maximum Match Similarity:  {m_res.max_similarity:.4f}")
        print(f"Combined Robust Score:     {m_res.similarity:.4f}")
        print(f"Calibrated Threshold:      {m_res.threshold:.4f}")
        print(f"Decision:                  {decision}")
        print("============================================================")
        if m_res.timing_ms:
            print("\nLatency Profile (CPU):")
            for k, v in m_res.timing_ms.items():
                print(f"  - {k:25s}: {v:6.2f} ms")
        return 0


if __name__ == "__main__":
    sys.exit(main())
