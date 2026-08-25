#!/usr/bin/env python3
"""Download official OpenCV Zoo YuNet and SFace ONNX models with validation."""

import sys
import urllib.request
from pathlib import Path

import cv2

MODELS = {
    "face_detection_yunet_2023mar.onnx": {
        "min_size": 200_000,  # ~232 KB
        "urls": [
            "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
    },
    "face_recognition_sface_2021dec.onnx": {
        "min_size": 35_000_000,  # ~38.7 MB
        "urls": [
            "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        ],
    },
    # MediaPipe task bundles powering the behavioural stage. Optional: without them
    # the pipeline still runs presence and identity, and reports speech, gaze and
    # hand analysis as unavailable rather than failing.
    "face_landmarker.task": {
        "min_size": 3_000_000,  # ~3.8 MB — 478 landmarks, 52 blendshapes, head pose
        "optional": True,
        "purpose": "speech articulation, head pose and gaze",
        "urls": [
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        ],
    },
    "hand_landmarker.task": {
        "min_size": 6_000_000,  # ~7.8 MB — 21 landmarks per hand
        "optional": True,
        "purpose": "hand presence and position",
        "urls": [
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        ],
    },
}


def download_file(urls: list[str], dest_path: Path, min_size: int) -> bool:
    """Download a file with fallback URLs and validate that it is a real binary ONNX model."""
    if dest_path.exists() and dest_path.stat().st_size >= min_size:
        print(
            f"✓ {dest_path.name} already exists and is valid ({dest_path.stat().st_size:,} bytes)."
        )
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    for url in urls:
        print(f"Downloading {dest_path.name} from: {url} ...")
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (AI Proctoring Downloader)"}
            )
            with (
                urllib.request.urlopen(req, timeout=30) as response,
                open(temp_path, "wb") as out_file,
            ):
                chunk_size = 1024 * 1024  # 1 MB chunks
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)

            size = temp_path.stat().st_size
            if size < min_size:
                print(
                    f"  Warning: Downloaded file size too small ({size} bytes). Likely LFS pointer or HTML error. Trying next mirror..."
                )
                temp_path.unlink(missing_ok=True)
                continue

            # Verify it's a binary file
            with open(temp_path, "rb") as f:
                header = f.read(100)
                if (
                    b"version https://git-lfs" in header
                    or b"<!DOCTYPE html>" in header
                    or b"<html" in header
                ):
                    print("  Warning: Downloaded file is LFS text or HTML. Trying next mirror...")
                    temp_path.unlink(missing_ok=True)
                    continue

            temp_path.rename(dest_path)
            print(f"✓ Successfully saved {dest_path.name} ({dest_path.stat().st_size:,} bytes).")
            return True

        except Exception as e:
            print(f"  Failed from {url}: {e}")
            temp_path.unlink(missing_ok=True)

    return False


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"

    all_ok = True
    for filename, meta in MODELS.items():
        dest = models_dir / filename
        ok = download_file(meta["urls"], dest, meta["min_size"])
        if not ok and meta.get("optional"):
            print(
                f"  NOTE: {filename} is optional — {meta.get('purpose', 'extra analysis')} "
                f"will be reported as unavailable rather than failing the session."
            )
            ok = True
        if not ok:
            all_ok = False

    if not all_ok:
        print("Error: One or more models could not be downloaded.", file=sys.stderr)
        return 1

    # Verify initialization
    print("\nVerifying model initialization with OpenCV...")
    try:
        det_path = models_dir / "face_detection_yunet_2023mar.onnx"
        rec_path = models_dir / "face_recognition_sface_2021dec.onnx"

        detector = cv2.FaceDetectorYN.create(
            model=str(det_path),
            config="",
            input_size=(320, 320),
        )
        print("✓ YuNet initialized successfully:", detector is not None)

        recognizer = cv2.FaceRecognizerSF.create(
            model=str(rec_path),
            config="",
        )
        print("✓ SFace initialized successfully:", recognizer is not None)
    except Exception as e:
        print(f"Verification failed: {e}", file=sys.stderr)
        return 1

    print("\nAll models downloaded and verified successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
