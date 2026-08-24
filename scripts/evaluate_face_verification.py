#!/usr/bin/env python3
"""Batch evaluation script for face verification baseline benchmarking."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import List, Tuple, Dict, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.face import FaceDetector, FaceVerifier


@dataclass
class ImagePair:
    """Pair of images with ground truth identity label."""
    image_a_path: Path
    image_b_path: Path
    is_same_person: bool
    identity_a: str
    identity_b: str


@dataclass
class Metrics:
    """Evaluation metrics for a given threshold."""
    threshold: float
    total: int
    positive_total: int
    negative_total: int
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float


def build_image_pairs_from_directory(data_dir: Path, max_pairs_per_person: int = 5) -> List[ImagePair]:
    """Construct positive and negative pairs from directory structure.
    
    Expected format:
        data_dir/
            <identity_1>/
                img1.jpg, img2.jpg, ...
            <identity_2>/
                img1.jpg, img2.jpg, ...
    """
    identities: Dict[str, List[Path]] = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for person_folder in sorted(data_dir.iterdir()):
        if person_folder.is_dir() and not person_folder.name.startswith("."):
            imgs = [p for p in sorted(person_folder.iterdir()) if p.suffix.lower() in valid_exts]
            if len(imgs) >= 2:
                identities[person_folder.name] = imgs

    if len(identities) < 2:
        raise ValueError(f"Need at least 2 identities with >= 2 images each. Found {len(identities)}.")

    pairs: List[ImagePair] = []
    id_names = sorted(identities.keys())

    # 1. Generate Positive Pairs (Same Identity)
    for name in id_names:
        imgs = identities[name]
        count = 0
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                pairs.append(
                    ImagePair(
                        image_a_path=imgs[i],
                        image_b_path=imgs[j],
                        is_same_person=True,
                        identity_a=name,
                        identity_b=name,
                    )
                )
                count += 1
                if count >= max_pairs_per_person:
                    break
            if count >= max_pairs_per_person:
                break

    # 2. Generate Negative Pairs (Different Identities)
    neg_count = 0
    total_pos = len(pairs)
    for i in range(len(id_names)):
        for j in range(i + 1, len(id_names)):
            name_a = id_names[i]
            name_b = id_names[j]
            imgs_a = identities[name_a]
            imgs_b = identities[name_b]
            for img_a in imgs_a[:2]:
                for img_b in imgs_b[:2]:
                    pairs.append(
                        ImagePair(
                            image_a_path=img_a,
                            image_b_path=img_b,
                            is_same_person=False,
                            identity_a=name_a,
                            identity_b=name_b,
                        )
                    )
                    neg_count += 1
                    if neg_count >= total_pos:
                        break
                if neg_count >= total_pos:
                    break
            if neg_count >= total_pos:
                break
        if neg_count >= total_pos:
            break

    return pairs


def compute_metrics(
    scores: List[Optional[float]],
    labels: List[bool],
    threshold: float,
    metric: str = "cosine",
) -> Metrics:
    """Calculate confusion matrix and accuracy metrics for a specific threshold."""
    tp = tn = fp = fn = 0
    valid_total = 0
    pos_total = sum(1 for l in labels if l)
    neg_total = len(labels) - pos_total

    for score, is_same in zip(scores, labels):
        if score is None:
            # Face detection failed on at least one image
            if is_same:
                fn += 1
            else:
                tn += 1
            continue

        valid_total += 1
        if metric == "cosine":
            pred_same = score >= threshold
        else:  # L2 distance
            pred_same = score <= threshold

        if pred_same and is_same:
            tp += 1
        elif not pred_same and not is_same:
            tn += 1
        elif pred_same and not is_same:
            fp += 1
        else:
            fn += 1

    total = len(labels)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return Metrics(
        threshold=threshold,
        total=total,
        positive_total=pos_total,
        negative_total=neg_total,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_score=f1,
    )


def run_evaluation(
    data_dir: Path,
    metric: str = "cosine",
    detector_model: str = "models/face_detection_yunet_2023mar.onnx",
    recognizer_model: str = "models/face_recognition_sface_2021dec.onnx",
) -> None:
    """Run full evaluation suite over image pairs and perform threshold sweep."""
    print("============================================================")
    print("FACE VERIFICATION BENCHMARK & THRESHOLD CALIBRATION")
    print("============================================================")
    print(f"Dataset Directory:   {data_dir}")
    print(f"Distance Metric:     {metric}")

    detector = FaceDetector(model_path=detector_model)
    verifier = FaceVerifier(
        detector=detector,
        recognizer_model_path=recognizer_model,
        default_metric=metric,
    )

    pairs = build_image_pairs_from_directory(data_dir)
    pos_count = sum(1 for p in pairs if p.is_same_person)
    neg_count = len(pairs) - pos_count
    print(f"Total Evaluated Pairs: {len(pairs)} (Positive: {pos_count}, Negative: {neg_count})")
    print("------------------------------------------------------------")
    print("Extracting features and computing pair similarities...")

    scores: List[Optional[float]] = []
    labels: List[bool] = []
    detection_failures = 0

    latencies: List[float] = []

    for i, pair in enumerate(pairs, 1):
        img_a = cv2.imread(str(pair.image_a_path))
        img_b = cv2.imread(str(pair.image_b_path))

        if img_a is None or img_b is None:
            scores.append(None)
            labels.append(pair.is_same_person)
            detection_failures += 1
            continue

        res = verifier.verify(img_a, img_b, metric=metric)
        if res.success:
            scores.append(res.similarity)
            if "total_inference_ms" in res.timing_ms:
                latencies.append(res.timing_ms["total_inference_ms"])
        else:
            scores.append(None)
            detection_failures += 1

        labels.append(pair.is_same_person)

    print(f"Feature extraction completed. Detection failures: {detection_failures}/{len(pairs)}")
    if latencies:
        print(f"Average CPU Inference Latency per Pair: {np.mean(latencies):.2f} ms")

    # Threshold Sweep
    print("\n------------------------------------------------------------")
    print(f"THRESHOLD SWEEP RESULTS ({metric.upper()})")
    print("------------------------------------------------------------")
    print(f"{'Threshold':>9s} | {'Acc':>6s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s} | {'TP':>4s} | {'TN':>4s} | {'FP':>4s} | {'FN':>4s}")
    print("-" * 65)

    if metric == "cosine":
        test_thresholds = np.linspace(0.10, 0.60, 26)
    else:
        test_thresholds = np.linspace(0.80, 1.40, 25)

    best_metrics: Optional[Metrics] = None

    for thresh in test_thresholds:
        m = compute_metrics(scores, labels, thresh, metric=metric)
        print(
            f"{m.threshold:9.3f} | {m.accuracy*100:5.1f}% | {m.precision*100:5.1f}% | "
            f"{m.recall*100:5.1f}% | {m.f1_score:6.3f} | {m.tp:4d} | {m.tn:4d} | {m.fp:4d} | {m.fn:4d}"
        )
        if best_metrics is None or m.accuracy > best_metrics.accuracy or (m.accuracy == best_metrics.accuracy and m.f1_score > best_metrics.f1_score):
            best_metrics = m

    print("============================================================")
    print("OPTIMAL CANDIDATE THRESHOLD SUMMARY")
    print("============================================================")
    if best_metrics:
        print(f"Recommended Threshold:     {best_metrics.threshold:.3f}")
        print(f"Peak Accuracy:             {best_metrics.accuracy*100:.2f}%")
        print(f"Precision:                 {best_metrics.precision*100:.2f}%")
        print(f"Recall:                    {best_metrics.recall*100:.2f}%")
        print(f"F1-Score:                  {best_metrics.f1_score:.4f}")
        print(f"True Positives (TP):       {best_metrics.tp}")
        print(f"True Negatives (TN):       {best_metrics.tn}")
        print(f"False Positives (FP):      {best_metrics.fp}")
        print(f"False Negatives (FN):      {best_metrics.fn}")
    print("============================================================")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate YuNet + SFace face verification performance.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/samples",
        help="Path to dataset directory containing identity subfolders",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["cosine", "l2"],
        default="cosine",
        help="Distance/similarity metric (default: cosine)",
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
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Dataset directory not found at '{data_dir}'", file=sys.stderr)
        return 1

    run_evaluation(
        data_dir=data_dir,
        metric=args.metric,
        detector_model=args.detector_model,
        recognizer_model=args.recognizer_model,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
