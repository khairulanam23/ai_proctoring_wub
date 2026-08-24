#!/usr/bin/env python3
"""Utility script to generate controlled multi-face test images and transition test videos."""

from pathlib import Path
import sys
import numpy as np
import cv2


def generate_synthetic_media(
    samples_dir: Path = Path("data/samples"),
    out_dir: Path = Path("data/samples/synthetic"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. No Face Image (Solid background with desk pattern)
    no_face_img = np.full((360, 480, 3), (220, 220, 220), dtype=np.uint8)
    cv2.putText(
        no_face_img,
        "No Subject / Empty Desk",
        (50, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (100, 100, 100),
        2,
    )
    cv2.imwrite(str(out_dir / "no_face.jpg"), no_face_img)
    print("Generated: data/samples/synthetic/no_face.jpg")

    # Find sample face images from LFW samples
    cp_imgs = list(samples_dir.glob("Colin_Powell/*.jpg"))
    gwb_imgs = list(samples_dir.glob("George_W_Bush/*.jpg"))
    tb_imgs = list(samples_dir.glob("Tony_Blair/*.jpg"))

    if not cp_imgs or not gwb_imgs:
        print("Warning: Sample images not found in data/samples. Skipping multi-face canvas generation.")
        return

    img_a = cv2.imread(str(cp_imgs[0]))
    img_b = cv2.imread(str(gwb_imgs[0]))
    img_c = cv2.imread(str(tb_imgs[0])) if tb_imgs else img_a

    # Resize faces to standard 220x220
    face_a = cv2.resize(img_a, (220, 220))
    face_b = cv2.resize(img_b, (220, 220))
    face_c = cv2.resize(img_c, (220, 220))

    # 2. Single Face Canvas (360x480)
    single_canvas = np.full((360, 480, 3), (240, 240, 240), dtype=np.uint8)
    single_canvas[70:290, 130:350] = face_a
    cv2.imwrite(str(out_dir / "single_face.jpg"), single_canvas)
    print("Generated: data/samples/synthetic/single_face.jpg")

    # 3. Two Faces Canvas (360x480)
    two_canvas = np.full((360, 480, 3), (240, 240, 240), dtype=np.uint8)
    two_canvas[70:290, 20:240] = face_a
    two_canvas[70:290, 240:460] = face_b
    cv2.imwrite(str(out_dir / "multi_face_2.jpg"), two_canvas)
    print("Generated: data/samples/synthetic/multi_face_2.jpg")

    # 4. Three Faces Canvas (360x720)
    three_canvas = np.full((360, 720, 3), (240, 240, 240), dtype=np.uint8)
    three_canvas[70:290, 20:240] = face_a
    three_canvas[70:290, 250:470] = face_b
    three_canvas[70:290, 480:700] = face_c
    cv2.imwrite(str(out_dir / "multi_face_3.jpg"), three_canvas)
    print("Generated: data/samples/synthetic/multi_face_3.jpg")

    # 5. Synthetic Video with Transitions (30 FPS, 4 seconds = 120 frames)
    # Sec 1 (0.0-1.0s, Frames 1-30): Single Face
    # Sec 2 (1.0-2.0s, Frames 31-60): Multiple Faces (Two Faces)
    # Sec 3 (2.0-3.0s, Frames 61-90): No Face
    # Sec 4 (3.0-4.0s, Frames 91-120): Single Face
    video_path = out_dir / "test_presence_transitions.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(str(video_path), fourcc, 30.0, (480, 360))

    if out_video.isOpened():
        for _ in range(30):
            out_video.write(single_canvas)
        for _ in range(30):
            out_video.write(two_canvas)
        for _ in range(30):
            out_video.write(no_face_img)
        for _ in range(30):
            out_video.write(single_canvas)
        out_video.release()
        print(f"Generated: {video_path} (120 frames, 4 state segments)")
    else:
        print(f"Failed to open video writer for {video_path}")


if __name__ == "__main__":
    generate_synthetic_media()
