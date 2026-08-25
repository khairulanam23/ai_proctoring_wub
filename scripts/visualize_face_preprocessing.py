#!/usr/bin/env python3
"""Visualization tool for face preprocessing, landmark alignment, pose estimation, and normalization."""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from proctoring.detection.face_detector import FaceDetector
from proctoring.preprocessing.face_preprocessing import FacePreprocessor


def create_preprocessing_visualization(
    image_path: Path,
    output_path: Path = Path("data/results/preprocessing_visualization.jpg"),
) -> None:
    """Run preprocessing on an image and create a structured 5-panel visual diagnostic comparison."""
    if not image_path.exists():
        print(f"Error: Image not found at '{image_path}'", file=sys.stderr)
        sys.exit(1)

    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        print(f"Error: Could not decode image from '{image_path}'", file=sys.stderr)
        sys.exit(1)

    detector = FaceDetector()
    preprocessor = FacePreprocessor(detector=detector)

    result = preprocessor.preprocess(image, require_single_face=False)

    print("============================================================")
    print("FACE PREPROCESSING & ALIGNMENT DIAGNOSTIC VISUALIZER")
    print("============================================================")
    print(f"Input Image:    {image_path.name} ({image.shape[1]}x{image.shape[0]})")
    print(f"Status:         {result.status.value}")
    print(f"Faces Detected: {result.face_count}")
    print(f"Message:        {result.message}")

    if not result.success or result.raw_face is None:
        print("Preprocessing failed. Cannot generate full visualization.", file=sys.stderr)
        sys.exit(1)

    face = result.raw_face
    q = result.quality
    pose = q.pose if q else None

    print("------------------------------------------------------------")
    print(f"Bounding Box:   x={face.bbox[0]}, y={face.bbox[1]}, w={face.bbox[2]}, h={face.bbox[3]}")
    print(f"Confidence:     {face.confidence:.4f}")
    print(f"Sharpness:      Blur Score={q.blur_score:.2f} (Sharp={q.is_sharp})")
    print(
        f"Resolution:     {q.min_dimension}px >= {preprocessor.min_face_size}px (Valid={q.is_size_valid})"
    )
    if pose:
        print(
            f"Pose Est:       Yaw Ratio={pose.yaw_ratio:.3f}, Roll={pose.roll_angle_deg:.1f}°, Pitch Ratio={pose.pitch_ratio:.3f}"
        )
        print(f"Pose State:     Frontal={pose.is_frontal}, Extreme={pose.is_extreme_pose}")
    print("Landmarks (5 Points):")
    landmark_names = ["Right Eye", "Left Eye", "Nose Tip", "Right Mouth", "Left Mouth"]
    for name, (lx, ly) in zip(landmark_names, face.landmarks, strict=False):
        print(f"  - {name:12s}: ({lx:6.1f}, {ly:6.1f})")
    print("------------------------------------------------------------")
    print("Timing Breakdown (CPU):")
    for k, v in result.timing_ms.items():
        print(f"  - {k:14s}: {v:6.2f} ms")
    print("============================================================")

    # Build 5-Panel Visualization
    target_h = 300
    target_w = 260

    def fit_into_panel(
        src_img: np.ndarray, ph: int = 300, pw: int = 260, title: str = "", subtitle: str = ""
    ) -> np.ndarray:
        panel = np.zeros((ph, pw, 3), dtype=np.uint8)
        ih, iw = src_img.shape[:2]
        scale = min((pw - 20) / iw, (ph - 60) / ih)
        rw, rh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(src_img, (rw, rh), interpolation=cv2.INTER_LINEAR)
        yo = 40 + (ph - 60 - rh) // 2
        xo = (pw - rw) // 2
        panel[yo : yo + rh, xo : xo + rw] = resized
        if title:
            cv2.putText(panel, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if subtitle:
            cv2.putText(
                panel, subtitle, (10, ph - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1
            )
        return panel

    # Panel 1: Original Image
    h, w = image.shape[:2]
    panel1 = fit_into_panel(image, target_h, target_w, "1. Original", f"{w}x{h}")

    # Panel 2: Detected Face with BBox & 5 Landmarks
    detected_vis = image.copy()
    x, y, bw, bh = face.bbox
    cv2.rectangle(detected_vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
    for (lx, ly), color in zip(face.landmarks, colors, strict=False):
        cv2.circle(detected_vis, (int(lx), int(ly)), 4, color, -1)
        cv2.circle(detected_vis, (int(lx), int(ly)), 5, (255, 255, 255), 1)
    sub2 = (
        f"Yaw:{pose.yaw_ratio:.2f} Roll:{pose.roll_angle_deg:.0f}°"
        if pose
        else f"Conf: {face.confidence:.2f}"
    )
    panel2 = fit_into_panel(detected_vis, target_h, target_w, "2. Landmarks", sub2)

    # Panel 3: Bounding Box Crop
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + bw), min(h, y + bh)
    raw_crop = image[y1:y2, x1:x2]
    panel3 = fit_into_panel(raw_crop, target_h, target_w, "3. BBox Crop", f"{bw}x{bh} Unaligned")

    # Panel 4: 5-Point Aligned Crop
    panel4 = fit_into_panel(
        result.aligned_face, target_h, target_w, "4. 5-Pt Aligned", "112x112 Canonical"
    )

    # Panel 5: Final SFace Input
    panel5 = fit_into_panel(
        result.normalized_face, target_h, target_w, "5. SFace Input", "128-d Feature Ready"
    )

    sep = np.full((target_h, 3, 3), (70, 70, 70), dtype=np.uint8)
    combined = np.hstack([panel1, sep, panel2, sep, panel3, sep, panel4, sep, panel5])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)
    print(f"\nDiagnostic visualization saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize face preprocessing, landmark alignment, and pose estimation."
    )
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        default="data/samples/Colin_Powell/Colin_Powell_0001.jpg",
        help="Path to input face image.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/results/preprocessing_visualization.jpg",
        help="Path to save output visualization image.",
    )
    args = parser.parse_args()

    target_path = Path(args.image_path)
    if not target_path.exists():
        fallback = Path("data/samples/synthetic/single_face.jpg")
        if fallback.exists():
            target_path = fallback

    create_preprocessing_visualization(target_path, Path(args.output))


if __name__ == "__main__":
    main()
