#!/usr/bin/env python3
"""Guided webcam check for earphone, talking and head-movement detection.

This is a *validation procedure*, not a benchmark. It does not measure accuracy
against a labelled dataset and makes no claim to. It answers one question, on the
machine and camera an exam will actually use: **when a person does the thing, does
the pipeline report it, and when they do the innocent lookalike, does it stay
quiet?**

Each scenario names what you should do, how long to do it, and what the pipeline is
expected to conclude. Scenarios marked ``expect: nothing`` are the ones that matter
most — a detector that fires on everything is not a working detector.

    python scripts/validate_detection.py                     # every scenario
    python scripts/validate_detection.py --group earphone    # one group
    python scripts/validate_detection.py --list
    python scripts/validate_detection.py --json report.json

Requires a webcam and the MediaPipe model bundles. Earphone scenarios additionally
need the YOLO-World weights; they are skipped with a clear message rather than
silently passing if it is unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proctoring.analysis.policy import ExamMode, StrictnessLevel  # noqa: E402
from proctoring.config import SessionConfig  # noqa: E402
from proctoring.core.events import EventType  # noqa: E402
from proctoring.detection.face_detector import FaceDetector  # noqa: E402
from proctoring.engine import ProctoringEngine  # noqa: E402


@dataclass
class Scenario:
    """One thing to do in front of the camera, and what should come of it."""

    key: str
    group: str
    instruction: str
    seconds: float
    expected: set[EventType] = field(default_factory=set)
    """Events that must appear. Empty means the pipeline should report nothing."""

    forbidden: set[EventType] = field(default_factory=set)
    """Events that must NOT appear even though something else might."""

    mode: ExamMode = ExamMode.DIGITAL_SCREEN
    needs_wearables: bool = False


_TALK = EventType.CANDIDATE_SPEAKING
_AWAY = EventType.LOOKING_AWAY
_PATTERN = EventType.SUSPICIOUS_HEAD_POSE
_BUDS = EventType.EARBUDS_SUSPECTED
_CANS = EventType.HEADPHONES_DETECTED

SCENARIOS: list[Scenario] = [
    # -- Earphones ---------------------------------------------------------
    Scenario(
        "no_earphones",
        "earphone",
        "Sit normally with NOTHING in or over your ears. Look at the screen.",
        15,
        forbidden={_BUDS, _CANS},
        needs_wearables=True,
    ),
    Scenario(
        "earbud_left",
        "earphone",
        "Put an earbud in your LEFT ear only. Face the camera normally.",
        15,
        expected={_BUDS},
        needs_wearables=True,
    ),
    Scenario(
        "earbud_right",
        "earphone",
        "Move the earbud to your RIGHT ear only.",
        15,
        expected={_BUDS},
        needs_wearables=True,
    ),
    Scenario(
        "earbuds_both",
        "earphone",
        "Wear earbuds in BOTH ears.",
        15,
        expected={_BUDS},
        needs_wearables=True,
    ),
    Scenario(
        "wired_earphone",
        "earphone",
        "Wear WIRED earphones, with the cable visible beside your face.",
        15,
        expected={_BUDS},
        needs_wearables=True,
    ),
    Scenario(
        "headset",
        "earphone",
        "Wear over-ear headphones or a headset.",
        15,
        expected={_CANS},
        needs_wearables=True,
    ),
    # -- Talking -----------------------------------------------------------
    Scenario(
        "silent",
        "talking",
        "Sit in silence with your mouth closed. Do not move your lips.",
        15,
        forbidden={_TALK},
    ),
    Scenario(
        "brief_mouth_movement",
        "talking",
        "Open and close your mouth ONCE, then stay silent.",
        15,
        forbidden={_TALK},
    ),
    Scenario(
        "silent_reading",
        "talking",
        "Read something on screen silently, without moving your lips.",
        15,
        forbidden={_TALK},
    ),
    Scenario(
        "yawn",
        "talking",
        "Yawn once, slowly and fully, then close your mouth and stay silent.",
        15,
        forbidden={_TALK},
    ),
    Scenario(
        "sustained_talking",
        "talking",
        "Talk continuously — read a paragraph OUT LOUD at a normal pace.",
        20,
        expected={_TALK},
    ),
    # -- Head movement -----------------------------------------------------
    Scenario(
        "normal_movement",
        "head",
        "Work normally: small posture shifts, occasional glance at the keyboard.",
        20,
        forbidden={_AWAY, _PATTERN},
    ),
    Scenario(
        "brief_glance",
        "head",
        "Glance to your left ONCE for about a second, then face the screen again.",
        15,
        forbidden={_PATTERN},
    ),
    Scenario(
        "sustained_turn",
        "head",
        "Turn your head fully to the left and HOLD it there.",
        15,
        expected={_AWAY},
    ),
    Scenario(
        "looking_up",
        "head",
        "Tilt your head back and look up at the ceiling. Hold it.",
        15,
        expected={_AWAY},
    ),
    Scenario(
        "repeated_look_away",
        "head",
        "Look away to the side and back again, repeatedly, about once every two seconds.",
        25,
        expected={_PATTERN},
    ),
    Scenario(
        "paper_looking_down",
        "head",
        "PHYSICAL PAPER mode: look down at the desk and write, as in a paper exam.",
        20,
        forbidden={_AWAY, _PATTERN},
        mode=ExamMode.PHYSICAL_PAPER,
    ),
]


def _run_scenario(scenario: Scenario, camera: int, output_dir: Path) -> dict[str, Any]:
    """Run one scenario against the live camera and collect what was reported."""
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        return {"key": scenario.key, "status": "ERROR", "detail": "camera unavailable"}

    config = SessionConfig(
        session_id=f"validate_{scenario.key}",
        student_name="Validation",
        strictness=StrictnessLevel.STRICT,
        exam_mode=scenario.mode,
        enable_face_verification=False,
        enable_object_detection=False,
        enable_wearable_detection=scenario.needs_wearables,
        capture_evidence=False,
        output_dir=output_dir,
    )
    engine = ProctoringEngine(config=config, face_detector=FaceDetector())

    if scenario.needs_wearables and (
        engine.wearable_detector is None or not engine.wearable_detector.is_available
    ):
        capture.release()
        return {
            "key": scenario.key,
            "status": "SKIPPED",
            "detail": "YOLO-World weights unavailable — earphone detection cannot be tested",
        }

    engine.start_session()
    print(f"\n  {scenario.instruction}")
    print(f"  Recording for {scenario.seconds:.0f}s — press 'q' in the window to abort.")

    start = time.time()
    frame_index, next_sample = 0, 0.0
    try:
        while True:
            elapsed = time.time() - start
            if elapsed >= scenario.seconds:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if elapsed >= next_sample:
                engine.process_frame(frame, frame_index, elapsed)
                frame_index += 1
                next_sample = elapsed + 1.0 / max(0.1, engine.target_fps)

            remaining = scenario.seconds - elapsed
            cv2.putText(
                frame,
                f"{scenario.key}  {remaining:4.1f}s",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow("validation", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        summary = engine.finalize_session()
        capture.release()
        cv2.destroyAllWindows()

    observed = {e.event_type for e in summary.events}
    qualified = {e.event_type for e in summary.events if e.status.value == "QUALIFIED"}

    missing = sorted(e.value for e in scenario.expected - qualified)
    spurious = sorted(e.value for e in scenario.forbidden & qualified)
    passed = not missing and not spurious

    return {
        "key": scenario.key,
        "group": scenario.group,
        "status": "PASS" if passed else "FAIL",
        "frames": frame_index,
        "expected": sorted(e.value for e in scenario.expected),
        "forbidden": sorted(e.value for e in scenario.forbidden),
        "qualified": sorted(e.value for e in qualified),
        "all_observed": sorted(e.value for e in observed),
        "missing": missing,
        "spurious": spurious,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted({s.group for s in SCENARIOS}))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--list", action="store_true", help="Print the scenarios and exit.")
    parser.add_argument("--json", type=Path, help="Write the results to this file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation"),
        help="Where session packages are written.",
    )
    args = parser.parse_args(argv)

    scenarios = [s for s in SCENARIOS if args.group is None or s.group == args.group]

    if args.list:
        for scenario in scenarios:
            expect = ", ".join(sorted(e.value for e in scenario.expected)) or "nothing"
            print(
                f"  {scenario.group:9s} {scenario.key:22s} {scenario.seconds:4.0f}s  expect: {expect}"
            )
        return 0

    print(__doc__.split("\n\n")[1])
    print(f"\n{len(scenarios)} scenario(s). Press Enter before each one, once you are ready.")

    results = []
    for index, scenario in enumerate(scenarios, start=1):
        print(f"\n[{index}/{len(scenarios)}] {scenario.group.upper()} — {scenario.key}")
        input("  Press Enter to begin... ")
        results.append(_run_scenario(scenario, args.camera, args.output_dir))

    print("\n" + "=" * 72)
    for result in results:
        line = f"  {result['status']:8s} {result['key']}"
        if result.get("missing"):
            line += f"   NOT reported: {', '.join(result['missing'])}"
        if result.get("spurious"):
            line += f"   falsely reported: {', '.join(result['spurious'])}"
        if result.get("detail"):
            line += f"   ({result['detail']})"
        print(line)

    passed = sum(1 for r in results if r["status"] == "PASS")
    skipped = sum(1 for r in results if r["status"] == "SKIPPED")
    print("=" * 72)
    print(f"  {passed} passed, {len(results) - passed - skipped} failed, {skipped} skipped")
    if skipped:
        print("  Skipped scenarios were NOT tested. Do not read them as passing.")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"  Written to {args.json}")

    return 0 if passed + skipped == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
