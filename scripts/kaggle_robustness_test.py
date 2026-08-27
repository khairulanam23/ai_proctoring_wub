#!/usr/bin/env python3
"""Kaggle GPU Real-World Robustness Testing, Threshold Calibration, and Edge Optimization Runner."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import psutil

from proctoring.detection import (
    ObjectDetector,
    ObjectRelevanceFilter,
)
from tools.benchmark.robustness import (
    ImageAugmenter,
    ThresholdEvaluator,
    generate_visual_comparison_grid,
)


def get_memory_stats() -> dict[str, Any]:
    """Inspect GPU VRAM and system CPU RAM usage."""
    stats = {
        "cpu_ram_used_mb": round(psutil.Process().memory_info().rss / (1024 * 1024), 2),
        "gpu_allocated_mb": 0.0,
        "gpu_reserved_mb": 0.0,
    }
    try:
        import torch

        if torch.cuda.is_available():
            stats["gpu_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
            stats["gpu_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
    except Exception:
        pass
    return stats


def run_condition_stress_test(
    detector: ObjectDetector,
    relevance_filter: ObjectRelevanceFilter,
    source_image: np.ndarray,
    output_grid_path: Path,
) -> dict[str, Any]:
    """Evaluate detector across 8 controlled visual conditions and render comparison grid."""
    print("\n" + "=" * 60)
    print("TEST 1: CONTROLLED VISUAL CONDITION STRESS TEST")
    print("=" * 60)

    conditions = ImageAugmenter.generate_condition_suite(source_image)
    cond_results = {}
    cond_summary = {}

    for cond_name, img_variant in conditions.items():
        raw = detector.detect(img_variant)
        rep = relevance_filter.filter(raw)
        cond_results[cond_name] = raw

        rel_names = [f"{o.class_name} ({o.confidence:.2f})" for o in rep.relevant_objects]
        cond_summary[cond_name] = {
            "total_detections": raw.count,
            "relevant_count": rep.relevant_count,
            "person_count": rep.person_count,
            "inference_time_ms": round(raw.inference_time_ms, 2),
            "relevant_objects": rel_names,
        }
        print(
            f"  • {cond_name.replace('_', ' ').title():20s}: {rep.relevant_count} relevant [Persons: {rep.person_count}] in {raw.inference_time_ms:5.1f} ms -> {rel_names}"
        )

    output_grid_path.parent.mkdir(parents=True, exist_ok=True)
    generate_visual_comparison_grid(conditions, cond_results, output_grid_path)
    print(f"\nVisual comparison grid saved to: {output_grid_path}")

    return cond_summary


def run_threshold_sweep_test(
    detector: ObjectDetector,
    source_image: np.ndarray,
) -> list[dict[str, Any]]:
    """Evaluate multiple confidence cutoff thresholds."""
    print("\n" + "=" * 60)
    print("TEST 2: CONFIDENCE THRESHOLD CALIBRATION SWEEP")
    print("=" * 60)

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
    # Provide synthetic test image with expected label (person)
    images_gt = [(source_image, ["person"])]
    sweep_results = ThresholdEvaluator.evaluate_threshold_sweep(detector, images_gt, thresholds)

    results_data = []
    print(
        f"  {'Threshold':<10} {'Detections':<12} {'Avg Conf':<12} {'Precision':<12} {'Recall':<12} {'F1':<10}"
    )
    print("  " + "-" * 66)

    for r in sweep_results:
        m = r.metrics
        p_str = f"{m.precision:.2f}" if m else "N/A"
        r_str = f"{m.recall:.2f}" if m else "N/A"
        f1_str = f"{m.f1:.2f}" if m else "N/A"
        print(
            f"  {r.threshold:<10.2f} {r.total_detections:<12d} {r.average_confidence:<12.4f} {p_str:<12s} {r_str:<12s} {f1_str:<10s}"
        )
        results_data.append(r.to_dict())

    return results_data


def run_resolution_benchmark(
    detector: ObjectDetector,
    source_image: np.ndarray,
    iterations: int = 15,
) -> dict[str, Any]:
    """Benchmark inference latency and memory across 480p, 720p, and 1080p resolutions."""
    print("\n" + "=" * 60)
    print("TEST 3: INPUT RESOLUTION LATENCY & THROUGHPUT BENCHMARK")
    print("=" * 60)

    resolutions = {
        "480p (640x480)": (640, 480),
        "720p (1280x720)": (1280, 720),
        "1080p (1920x1080)": (1920, 1080),
    }
    res_summary = {}

    for res_name, (w, h) in resolutions.items():
        scaled = cv2.resize(source_image, (w, h))
        # Warmup
        for _ in range(3):
            _ = detector.detect(scaled)

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = detector.detect(scaled)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_ms = float(np.mean(latencies))
        p50_ms = float(np.percentile(latencies, 50))
        p95_ms = float(np.percentile(latencies, 95))
        min_ms = float(np.min(latencies))
        max_ms = float(np.max(latencies))
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

        res_summary[res_name] = {
            "avg_ms": round(avg_ms, 2),
            "p50_ms": round(p50_ms, 2),
            "p95_ms": round(p95_ms, 2),
            "min_ms": round(min_ms, 2),
            "max_ms": round(max_ms, 2),
            "throughput_fps": round(fps, 1),
            "memory": get_memory_stats(),
        }
        print(
            f"  • {res_name:18s}: Avg {avg_ms:5.1f} ms [P50: {p50_ms:5.1f} ms, P95: {p95_ms:5.1f} ms] -> {fps:5.1f} FPS"
        )

    return res_summary


def run_long_run_stability_test(
    detector: ObjectDetector,
    source_image: np.ndarray,
    cycles: int = 50,
) -> dict[str, Any]:
    """Run repetitive streaming cycles to evaluate memory growth and leak stability."""
    print("\n" + "=" * 60)
    print(f"TEST 4: LONG-RUN STREAMING STABILITY TEST ({cycles} CYCLES)")
    print("=" * 60)

    mem_start = get_memory_stats()
    t0 = time.perf_counter()

    for _i in range(cycles):
        _ = detector.detect(source_image)

    total_s = time.perf_counter() - t0
    mem_end = get_memory_stats()
    cpu_growth = mem_end["cpu_ram_used_mb"] - mem_start["cpu_ram_used_mb"]
    gpu_growth = mem_end["gpu_allocated_mb"] - mem_start["gpu_allocated_mb"]

    print(f"  • Completed {cycles} continuous inference iterations in {total_s:.2f} s")
    print(
        f"  • CPU RAM Start: {mem_start['cpu_ram_used_mb']} MB -> End: {mem_end['cpu_ram_used_mb']} MB (Delta: {cpu_growth:+.2f} MB)"
    )
    print(
        f"  • GPU VRAM Start: {mem_start['gpu_allocated_mb']} MB -> End: {mem_end['gpu_allocated_mb']} MB (Delta: {gpu_growth:+.2f} MB)"
    )

    return {
        "cycles": cycles,
        "total_seconds": round(total_s, 2),
        "mem_start": mem_start,
        "mem_end": mem_end,
        "cpu_growth_mb": round(cpu_growth, 2),
        "gpu_growth_mb": round(gpu_growth, 2),
        "leak_detected": bool(cpu_growth > 50.0 or gpu_growth > 50.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run real-world robustness evaluation, threshold sweep, and GPU performance benchmark."
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default="data/samples/Colin_Powell/Colin_Powell_0001.jpg",
        help="Path to base test image file.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yolo11n.pt",
        help="YOLO model identifier or weights path (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        help="Inference device: 'cuda', 'cpu', or auto-detect.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/results/object_detection/robustness",
        help="Directory to save robustness visual grid and JSON report.",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(args.image)
    if not img_path.exists():
        fallback_candidates = [
            Path("data/samples/Colin_Powell/Colin_Powell_0001.jpg"),
            Path("data/samples/synthetic/single_face.jpg"),
        ]
        for fb in fallback_candidates:
            if fb.exists():
                img_path = fb
                break

    if not img_path.exists():
        print(f"Error: Base test image not found at '{args.image}'", file=sys.stderr)
        return 1

    source_image = cv2.imread(str(img_path))
    if source_image is None:
        print(f"Error: Failed to decode image from '{img_path}'", file=sys.stderr)
        return 1

    # Initialize detector and relevance filter
    try:
        detector = ObjectDetector(model_name=args.model, device=args.device)
    except Exception as e:
        print(f"\n[Model Loading Notice] {e}", file=sys.stderr)
        return 2

    relevance_filter = ObjectRelevanceFilter()

    # 1. Condition Stress Test
    grid_path = out_dir / "comparison_grid.jpg"
    cond_summary = run_condition_stress_test(detector, relevance_filter, source_image, grid_path)

    # 2. Threshold Sweep Test
    threshold_results = run_threshold_sweep_test(detector, source_image)

    # 3. Resolution Benchmark
    res_summary = run_resolution_benchmark(detector, source_image)

    # 4. Long-Run Stability Test
    stability_summary = run_long_run_stability_test(detector, source_image, cycles=30)

    # 5. Export JSON Report
    report = {
        "test_image": img_path.name,
        "model_name": detector.model_name,
        "device": detector.device,
        "memory_diagnostics": get_memory_stats(),
        "visual_condition_stress_test": cond_summary,
        "threshold_calibration_sweep": threshold_results,
        "resolution_benchmark": res_summary,
        "stability_test": stability_summary,
    }

    report_json_path = out_dir / "robustness_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("ROBUSTNESS BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Visual Grid: {grid_path}")
    print(f"JSON Report: {report_json_path}")
    print("============================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
