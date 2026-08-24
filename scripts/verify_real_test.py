#!/usr/bin/env python3
"""Phase 2: Real-Image Face Verification Experiment Script."""

from pathlib import Path
import sys
import time

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.face import FaceDetector, FaceVerifier


def run_real_verification(
    ref_path: Path = Path("data/real_test/reference.jpg"),
    test_path: Path = Path("data/real_test/test.jpg"),
) -> int:
    print("============================================================")
    print("PHASE 2 — REAL-IMAGE FACE VERIFICATION EXPERIMENT")
    print("============================================================")
    print(f"Reference Image Path: {ref_path}")
    print(f"Test Image Path:      {test_path}")
    print("------------------------------------------------------------")

    # 1. File existence checks
    if not ref_path.exists():
        print(f"Error: Reference image not found at '{ref_path}'.", file=sys.stderr)
        print("Please place 'reference.jpg' into 'data/real_test/'.", file=sys.stderr)
        return 1
    if not test_path.exists():
        print(f"Error: Test image not found at '{test_path}'.", file=sys.stderr)
        print("Please place 'test.jpg' into 'data/real_test/'.", file=sys.stderr)
        return 1

    # 2. Image decoding checks
    ref_img = cv2.imread(str(ref_path))
    test_img = cv2.imread(str(test_path))

    if ref_img is None or ref_img.size == 0:
        print(f"Error: Failed to decode reference image from '{ref_path}'. File may be corrupted or empty.", file=sys.stderr)
        return 1
    if test_img is None or test_img.size == 0:
        print(f"Error: Failed to decode test image from '{test_path}'. File may be corrupted or empty.", file=sys.stderr)
        return 1

    print("Image Properties:")
    print(f"  - Reference Image: {ref_img.shape[1]}x{ref_img.shape[0]} (Channels: {ref_img.shape[2]}, Dtype: {ref_img.dtype})")
    print(f"  - Test Image:      {test_img.shape[1]}x{test_img.shape[0]} (Channels: {test_img.shape[2]}, Dtype: {test_img.dtype})")
    print("------------------------------------------------------------")

    # 3. Model Initialization
    detector = FaceDetector()
    verifier = FaceVerifier(detector=detector)

    # 4. Face Detection
    t0 = time.perf_counter()
    det_ref = detector.detect(ref_img)
    t_det_ref_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    det_test = detector.detect(test_img)
    t_det_test_ms = (time.perf_counter() - t0) * 1000.0

    print("Face Detection Results:")
    print(f"  - Reference faces detected: {det_ref.count}")
    print(f"  - Test faces detected:      {det_test.count}")

    # Check for 0 or multiple faces
    if det_ref.count != 1 or det_test.count != 1:
        print("\n[!] Face Presence Anomaly Detected:")
        if det_ref.count == 0:
            print("  - Reference image: NO FACE detected.")
        elif det_ref.count > 1:
            print(f"  - Reference image: MULTIPLE FACES detected ({det_ref.count} faces).")

        if det_test.count == 0:
            print("  - Test image: NO FACE detected.")
        elif det_test.count > 1:
            print(f"  - Test image: MULTIPLE FACES detected ({det_test.count} faces).")

        print("\nDecision: REJECTED — Cannot perform 1:1 verification when face count != 1.")
        print("============================================================")
        return 2

    face_ref = det_ref.faces[0]
    face_test = det_test.faces[0]

    print(f"  - Reference face confidence: {face_ref.confidence:.4f} (BBox: {face_ref.bbox})")
    print(f"  - Test face confidence:      {face_test.confidence:.4f} (BBox: {face_test.bbox})")
    print("------------------------------------------------------------")

    # 5. Feature Extraction (128-d Embedding)
    t0 = time.perf_counter()
    feat_ref = verifier.extract_feature(ref_img, face_ref)
    t_feat_ref_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    feat_test = verifier.extract_feature(test_img, face_test)
    t_feat_test_ms = (time.perf_counter() - t0) * 1000.0

    print("SFace Feature Embedding Dimensions:")
    print(f"  - Reference Embedding Shape: {feat_ref.shape} (Dtype: {feat_ref.dtype}, L2 Norm: {np.linalg.norm(feat_ref):.4f})")
    print(f"  - Test Embedding Shape:      {feat_test.shape} (Dtype: {feat_test.dtype}, L2 Norm: {np.linalg.norm(feat_test):.4f})")
    print("------------------------------------------------------------")

    # 6. Similarity & Verification
    t0 = time.perf_counter()
    similarity = verifier.compute_similarity(feat_ref, feat_test, metric="cosine")
    t_match_ms = (time.perf_counter() - t0) * 1000.0

    threshold = verifier.default_threshold
    same_person = similarity >= threshold
    decision = "SAME PERSON" if same_person else "DIFFERENT PERSON"

    total_time_ms = t_det_ref_ms + t_det_test_ms + t_feat_ref_ms + t_feat_test_ms + t_match_ms

    print("Verification Comparison:")
    print(f"  - Metric:           Cosine Similarity")
    print(f"  - Cosine Similarity: {similarity:.4f}")
    print(f"  - Baseline Threshold:{threshold:.4f}")
    print(f"  - Final Decision:    {decision}")
    print("------------------------------------------------------------")
    print("Latency Profile (CPU):")
    print(f"  - Reference Detection:   {t_det_ref_ms:6.2f} ms")
    print(f"  - Test Detection:        {t_det_test_ms:6.2f} ms")
    print(f"  - Reference Embedding:   {t_feat_ref_ms:6.2f} ms")
    print(f"  - Test Embedding:        {t_feat_test_ms:6.2f} ms")
    print(f"  - Similarity Matching:   {t_match_ms:6.3f} ms")
    print(f"  - Total Processing Time: {total_time_ms:6.2f} ms")
    print("============================================================")

    print("\nSHORT EXPERIMENT REPORT")
    print("------------------------------------------------------------")
    print(f"Status:          {'PASS' if same_person else 'FAIL'}")
    print(f"Similarity:      {similarity:.4f}")
    print(f"Threshold:       {threshold:.4f}")
    print(f"Decision:        {decision}")
    print(f"Processing time: {total_time_ms:.2f} ms")
    print("Problems:        None" if same_person else "Problems: Similarity below baseline threshold")
    print("============================================================")

    return 0


if __name__ == "__main__":
    sys.exit(run_real_verification())
