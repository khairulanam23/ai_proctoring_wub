#!/usr/bin/env python3
"""Measure per-stage cost and end-to-end throughput of the proctoring pipeline.

Produces the numbers in ``docs/accuracy_and_performance.md``.  Re-run it on your
own hardware before planning capacity — the published figures are from one CPU and
will not match a different machine.

    python scripts/benchmark_pipeline.py
    python scripts/benchmark_pipeline.py --resolution 1280x720 --iterations 30
    python scripts/benchmark_pipeline.py --json out.json
"""

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proctoring.analysis import (  # noqa: E402
    FacialDynamicsAnalyzer,
    HandAnalyzer,
    StrictnessLevel,
    WearableDetector,
)
from proctoring.config import SessionConfig  # noqa: E402
from proctoring.detection.face_detector import FaceDetector  # noqa: E402
from proctoring.detection.face_verifier import FaceVerifier  # noqa: E402
from proctoring.engine import ProctoringEngine  # noqa: E402
from proctoring.preprocessing.frame_quality import FrameQualityGate  # noqa: E402


def load_sample_frames(resolution) -> list[np.ndarray]:
    """Real face imagery if available, else synthetic frames.

    Synthetic frames understate landmark-model cost, because the models exit early
    when they find no face — the report says which was used.
    """
    samples = sorted(Path("data/samples").glob("*/*_000*.jpg"))
    if samples:
        return [cv2.resize(cv2.imread(str(p)), resolution) for p in samples[:8]]
    return [np.full((resolution[1], resolution[0], 3), 120, np.uint8) for _ in range(8)]


def time_stage(
    name: str, fn: Callable[[], Any], iterations: int, warmup: int = 3
) -> dict[str, Any]:
    """Time a callable, discarding warm-up runs that include lazy initialisation."""
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000.0)

    timings.sort()
    return {
        "stage": name,
        "median_ms": round(statistics.median(timings), 2),
        "mean_ms": round(statistics.fmean(timings), 2),
        "p95_ms": round(timings[int(len(timings) * 0.95) - 1], 2),
        "min_ms": round(timings[0], 2),
        "max_ms": round(timings[-1], 2),
    }


def benchmark_stages(frames, iterations: int, include_wearables: bool) -> list[dict[str, Any]]:
    """Measure each detector in isolation."""
    results: list[dict[str, Any]] = []
    frame = frames[0]
    counter = {"i": 0}

    def next_frame():
        counter["i"] = (counter["i"] + 1) % len(frames)
        return frames[counter["i"]]

    gate = FrameQualityGate()
    results.append(
        time_stage("frame quality gate (+CLAHE)", lambda: gate.process(next_frame()), iterations)
    )

    yunet = Path("models/face_detection_yunet_2023mar.onnx")
    if yunet.exists():
        detector = FaceDetector(model_path=yunet)
        results.append(
            time_stage("face detection (YuNet)", lambda: detector.detect(next_frame()), iterations)
        )

        sface = Path("models/face_recognition_sface_2021dec.onnx")
        if sface.exists():
            verifier = FaceVerifier(detector=detector, recognizer_model_path=sface)
            detection = detector.detect(frame)
            if detection.count:
                face = detection.faces[0]
                results.append(
                    time_stage(
                        "identity verification (SFace)",
                        lambda: verifier.extract_feature(frame, face=face),
                        iterations,
                    )
                )

    dynamics = FacialDynamicsAnalyzer()
    if dynamics.is_available:
        results.append(
            time_stage(
                "facial dynamics (speech/pose/gaze)",
                lambda: dynamics.analyze(next_frame()),
                iterations,
            )
        )
        dynamics.close()

    hands = HandAnalyzer()
    if hands.is_available:
        results.append(time_stage("hand analysis", lambda: hands.analyze(next_frame()), iterations))
        hands.close()

    try:
        from ultralytics import YOLO

        yolo = YOLO("yolo11n.pt")
        results.append(
            time_stage(
                "object detection (YOLO11n)",
                lambda: yolo.predict(next_frame(), verbose=False),
                max(6, iterations // 2),
            )
        )
    except Exception as exc:
        results.append({"stage": "object detection (YOLO11n)", "unavailable": str(exc)[:80]})

    if include_wearables:
        wearables = WearableDetector()
        if wearables.is_available:
            results.append(
                time_stage(
                    "wearable detection (YOLO-World)",
                    lambda: wearables.detect(next_frame()),
                    max(5, iterations // 4),
                )
            )
        else:
            results.append(
                {"stage": "wearable detection (YOLO-World)", "unavailable": "model unavailable"}
            )

    return results


def benchmark_end_to_end(
    frames, strictness: StrictnessLevel, frame_count: int, include_wearables: bool
) -> dict[str, Any]:
    """Measure a whole session at one strictness level, as deployed."""
    import tempfile

    yunet = Path("models/face_detection_yunet_2023mar.onnx")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SessionConfig(
            session_id=f"bench_{strictness.value.lower()}",
            strictness=strictness,
            output_dir=tmpdir,
            enable_object_detection=False,
            enable_face_verification=False,
            enable_wearable_detection=include_wearables,
        )
        engine = ProctoringEngine(
            config=config,
            face_detector=FaceDetector(model_path=yunet) if yunet.exists() else None,
        )
        engine.start_session()

        start = time.perf_counter()
        for index in range(frame_count):
            engine.process_frame(
                frames[index % len(frames)], frame_index=index, timestamp_seconds=index / 4.0
            )
        elapsed = time.perf_counter() - start
        summary = engine.finalize_session()

        latency = summary.telemetry.latency_overall
        return {
            "strictness": strictness.value,
            "frames": frame_count,
            "wall_seconds": round(elapsed, 2),
            "throughput_fps": round(frame_count / elapsed, 1),
            "median_latency_ms": round(latency.median_p50_ms, 2),
            "p95_latency_ms": round(latency.p95_ms, 2),
            "max_latency_ms": round(latency.max_ms, 2),
            "peak_rss_mb": summary.telemetry.resource_usage.get("peak_rss_mb"),
            "events": summary.total_events,
            "qualified_events": summary.qualified_events,
            "stage_means_ms": summary.telemetry.stage_latencies_mean,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--resolution", default="640x480", help="Frame size, WxH.")
    parser.add_argument("--iterations", type=int, default=15, help="Timed runs per stage.")
    parser.add_argument("--session-frames", type=int, default=60, help="Frames per end-to-end run.")
    parser.add_argument(
        "--wearables",
        action="store_true",
        help="Include the open-vocabulary wearable detector (slow).",
    )
    parser.add_argument("--json", type=str, default=None, help="Write the full report here.")
    args = parser.parse_args()

    width, height = (int(v) for v in args.resolution.lower().split("x"))
    frames = load_sample_frames((width, height))
    used_real = bool(sorted(Path("data/samples").glob("*/*_000*.jpg")))

    import platform

    print("=" * 74)
    print("  AI PROCTORING — PIPELINE BENCHMARK")
    print("=" * 74)
    print(f"  Resolution   {width}x{height}")
    print(
        f"  Frames       {'real sample imagery' if used_real else 'SYNTHETIC (understates landmark cost)'}"
    )
    print(f"  Platform     {platform.platform()}")
    print(f"  Python       {platform.python_version()}")

    print("\n" + "-" * 74)
    print("  PER-STAGE COST (isolated)")
    print("-" * 74)
    print(f"  {'stage':<38}{'median':>10}{'p95':>10}{'max':>10}")
    stages = benchmark_stages(frames, args.iterations, args.wearables)
    for row in stages:
        if "unavailable" in row:
            print(f"  {row['stage']:<38}{'unavailable':>30}")
        else:
            print(
                f"  {row['stage']:<38}{row['median_ms']:>9.1f}{row['p95_ms']:>10.1f}{row['max_ms']:>10.1f}"
            )

    print("\n" + "-" * 74)
    print("  END-TO-END SESSION THROUGHPUT")
    print("-" * 74)
    print(f"  {'strictness':<14}{'fps':>8}{'median':>10}{'p95':>10}{'peak RSS':>12}{'events':>9}")
    sessions = []
    for level in StrictnessLevel:
        result = benchmark_end_to_end(frames, level, args.session_frames, args.wearables)
        sessions.append(result)
        print(
            f"  {result['strictness']:<14}{result['throughput_fps']:>8.1f}"
            f"{result['median_latency_ms']:>10.1f}{result['p95_latency_ms']:>10.1f}"
            f"{result['peak_rss_mb'] or 0:>11.0f}M{result['events']:>9}"
        )

    budget_ms = 1000.0 / 4.0
    slowest = max(s["median_latency_ms"] for s in sessions)
    print(
        f"\n  At 4 fps the per-frame budget is {budget_ms:.0f} ms; "
        f"the slowest profile uses {slowest:.0f} ms "
        f"({slowest / budget_ms * 100:.0f}% of budget)."
    )
    concurrent = int(budget_ms / slowest) if slowest > 0 else 0
    print(f"  One CPU core sustains roughly {max(1, concurrent)} concurrent 4 fps session(s).")

    report = {
        "resolution": [width, height],
        "used_real_imagery": used_real,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "stages": stages,
        "sessions": sessions,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  Full report written to {args.json}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
