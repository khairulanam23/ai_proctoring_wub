#!/usr/bin/env python3
"""Research Dataset Capture Utility for AI Proctoring Model Development.

================================================================================
AI PROCTORING DATASET CAPTURE TOOL (RESEARCH-SIDE UTILITY)
================================================================================
Purpose:
  Provides a guided, structured experiment controller to capture real-world
  video data under configurable conditions (smartphone, paper, earphones,
  natural exam behaviors) with exact timeline metadata manifests.

Production Protection Invariant:
  This is an isolated research utility. It does NOT participate in production AI
  inference, does NOT invoke production detection pipelines, and does NOT generate
  verified ground truth. Instruction prompts recorded in the manifest represent
  `instructed_condition` for downstream human verification.
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from proctoring.capture.camera import (
    discover_local_camera,
)

LOGGER = logging.getLogger("capture_dataset")
SCHEMA_VERSION = "1.0.0"
MIN_FREE_DISK_BYTES = 500 * 1024 * 1024  # 500 MB safety threshold


@dataclass
class QuestionOption:
    key: str
    label: str
    next_step: str | None = None


@dataclass
class ScenarioQuestion:
    prompt: str
    options: list[QuestionOption] = field(default_factory=list)


@dataclass
class ScenarioStep:
    step_id: str
    instruction: str
    instructed_condition: str
    preparation_seconds: int = 5
    recording_seconds: int = 10
    notes: str = ""
    question: ScenarioQuestion | None = None
    next_step: str | None = None


@dataclass
class ScenarioConfig:
    scenario_id: str
    name: str
    description: str
    version: str
    default_preparation_seconds: int
    default_recording_seconds: int
    steps: list[ScenarioStep]

    @classmethod
    def from_yaml_file(cls, path: Path | str) -> ScenarioConfig:
        filepath = Path(path)
        if not filepath.is_file():
            raise FileNotFoundError(f"Scenario configuration file not found: {filepath}")

        with open(filepath, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML content in {filepath}: expected mapping")

        scenario_id = str(data.get("scenario_id", filepath.stem)).strip()
        name = str(data.get("name", scenario_id)).strip()
        description = str(data.get("description", "")).strip()
        version = str(data.get("version", "1.0")).strip()
        default_prep = int(data.get("default_preparation_seconds", 5))
        default_rec = int(data.get("default_recording_seconds", 10))

        raw_steps = data.get("steps", [])
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError(f"Scenario {scenario_id} contains no steps.")

        steps: list[ScenarioStep] = []
        for i, s in enumerate(raw_steps):
            if not isinstance(s, dict):
                raise ValueError(f"Step {i} in {scenario_id} must be a dictionary")
            step_id = str(s.get("step_id", f"step_{i+1}")).strip()
            instruction = str(s.get("instruction", "")).strip()
            instructed_cond = str(s.get("instructed_condition", step_id)).strip()
            prep_sec = int(s.get("preparation_seconds", default_prep))
            rec_sec = int(s.get("recording_seconds", default_rec))
            notes = str(s.get("notes", "")).strip()
            next_step = s.get("next_step")

            raw_q = s.get("question")
            question = None
            if isinstance(raw_q, dict):
                q_prompt = str(raw_q.get("prompt", "")).strip()
                opts: list[QuestionOption] = []
                for opt in raw_q.get("options", []):
                    if isinstance(opt, dict):
                        opts.append(
                            QuestionOption(
                                key=str(opt.get("key", "")).strip().upper(),
                                label=str(opt.get("label", "")).strip(),
                                next_step=opt.get("next_step"),
                            )
                        )
                question = ScenarioQuestion(prompt=q_prompt, options=opts)

            steps.append(
                ScenarioStep(
                    step_id=step_id,
                    instruction=instruction,
                    instructed_condition=instructed_cond,
                    preparation_seconds=prep_sec,
                    recording_seconds=rec_sec,
                    notes=notes,
                    question=question,
                    next_step=next_step,
                )
            )

        return cls(
            scenario_id=scenario_id,
            name=name,
            description=description,
            version=version,
            default_preparation_seconds=default_prep,
            default_recording_seconds=default_rec,
            steps=steps,
        )


@dataclass
class StepRecord:
    step_id: str
    step_index: int
    instruction: str
    instructed_condition: str
    preparation_start_seconds: float
    preparation_end_seconds: float
    recording_start_seconds: float
    recording_end_seconds: float
    duration_seconds: float
    frame_start_index: int = 0
    frame_end_index: int = 0
    frames_recorded: int = 0
    operator_response: str | None = None
    status: str = "completed"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sanitize_identifier(value: str, name: str = "Identifier") -> str:
    cleaned = value.strip()
    if not re.match(r"^[A-Za-z0-9_-]+$", cleaned):
        raise ValueError(
            f"{name} '{value}' contains invalid characters. Use alphanumeric, dashes, and underscores only."
        )
    return cleaned


def calculate_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_disk_space(target_dir: Path, min_bytes: int = MIN_FREE_DISK_BYTES) -> tuple[bool, int, int]:
    target_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target_dir)
    return (usage.free >= min_bytes), usage.free, usage.total


class VideoCaptureSession:
    """Manages continuous video capture, countdown timelines, operator interactions, and manifests."""

    def __init__(
        self,
        participant_id: str,
        session_id: str,
        scenario: ScenarioConfig,
        output_base_dir: Path,
        camera_index: int | None = None,
        target_width: int = 1280,
        target_height: int = 720,
        target_fps: float = 30.0,
        show_preview: bool = True,
        dry_run: bool = False,
        dry_run_frames: int = 100,
        duration_scale: float = 1.0,
        pixel_format: str = "AUTO",
    ) -> None:
        self.participant_id = sanitize_identifier(participant_id, "Participant ID")
        self.session_id = sanitize_identifier(session_id, "Session ID")
        self.scenario = scenario
        self.output_base_dir = output_base_dir
        self.camera_index = camera_index
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self.show_preview = show_preview
        self.dry_run = dry_run
        self.dry_run_frames = dry_run_frames
        self.duration_scale = max(0.0, float(duration_scale))
        self.pixel_format = str(pixel_format).upper()

        # Destination paths
        self.session_dir = self.output_base_dir / self.participant_id / self.session_id / self.scenario.scenario_id
        self.video_path = self.session_dir / "raw_video.mp4"
        self.staging_video_path = self.session_dir / "raw_video_staging.mp4"
        self.manifest_path = self.session_dir / "session_manifest.json"
        self.checksum_path = self.session_dir / "checksum.sha256"

        # Hardware & recording state
        self.cap: cv2.VideoCapture | None = None
        self.writer: cv2.VideoWriter | None = None
        self.camera_info: dict[str, Any] = {}
        self.steps_recorded: list[StepRecord] = []
        self.capture_start_time: float = 0.0
        self.capture_end_time: float = 0.0
        self.capture_start_iso: str = ""
        self.capture_end_iso: str = ""
        self.total_frames_written: int = 0
        self.frame_read_failures: int = 0
        self.session_status: str = "initialized"
        self.interrupted: bool = False
        self.total_paused_seconds: float = 0.0
        self.initial_writer_fps: float = float(target_fps)

        # GUI availability check
        self.can_gui = bool(self.show_preview and os.environ.get("DISPLAY") and not self.dry_run)

    def get_active_session_elapsed(self) -> float:
        """Returns the active elapsed wall-clock recording seconds, excluding pause duration."""
        if self.capture_start_time <= 0.0:
            return 0.0
        total_wall = max(0.0, time.monotonic() - self.capture_start_time)
        return max(0.0, total_wall - self.total_paused_seconds)

    def prepare_session_directory(self) -> None:
        if self.session_dir.exists() and (self.video_path.exists() or self.manifest_path.exists()):
            raise FileExistsError(
                f"Session recording already exists at {self.session_dir}. "
                "Refusing to overwrite existing raw research data. Choose a new Session ID."
            )
        self.session_dir.mkdir(parents=True, exist_ok=True)

        has_space, free_bytes, _ = check_disk_space(self.session_dir)
        if not has_space:
            raise OSError(
                f"Insufficient disk space on {self.session_dir}. "
                f"Free: {free_bytes / (1024*1024):.1f} MB, required at least {MIN_FREE_DISK_BYTES / (1024*1024):.1f} MB."
            )

    def initialize_camera(self) -> bool:
        if self.dry_run:
            self.initial_writer_fps = float(self.target_fps)
            self.camera_info = {
                "device_index": None,
                "device_path": "SYNTHETIC_MOCK",
                "backend_name": "SYNTHETIC",
                "requested_resolution": [self.target_width, self.target_height],
                "actual_resolution": [self.target_width, self.target_height],
                "requested_fps": float(self.target_fps),
                "reported_camera_fps": float(self.target_fps),
                "measured_effective_fps": float(self.target_fps),
                "configured_fps": float(self.target_fps),
                "pixel_format": "SYNTHETIC",
                "codec": "mp4v",
            }
            return True

        discovery = discover_local_camera(
            preferred_index=self.camera_index,
            target_width=self.target_width,
            target_height=self.target_height,
        )

        if not discovery.is_valid or discovery.device_index is None:
            LOGGER.error(f"Camera discovery failed: {discovery.error_message}")
            return False

        cap = cv2.VideoCapture(discovery.device_index, discovery.backend_code)
        if not cap.isOpened():
            time.sleep(0.3)
            cap = cv2.VideoCapture(discovery.device_index, cv2.CAP_ANY)
        if not cap.isOpened():
            LOGGER.error(f"Failed to open discovered camera index {discovery.device_index}")
            return False

        # Set pixel format (FourCC) if requested or auto-configured for HD
        used_fourcc = "DEFAULT"
        if self.pixel_format == "MJPG" or (self.pixel_format == "AUTO" and self.target_width > 640):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            used_fourcc = "MJPG"
        elif self.pixel_format == "YUYV":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
            used_fourcc = "YUYV"

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Warm up & probe initial hardware arrival rate
        warm_frames = 0
        warm_t0 = time.monotonic()
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                warm_frames += 1
        warm_elapsed = max(0.001, time.monotonic() - warm_t0)
        calibrated_fps = warm_frames / warm_elapsed if warm_frames > 0 else self.target_fps

        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            LOGGER.error("Opened camera but initial frame read failed.")
            cap.release()
            return False

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_camera_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if reported_camera_fps <= 0.0 or np.isnan(reported_camera_fps):
            reported_camera_fps = self.target_fps

        estimated_fps = round(calibrated_fps, 2) if calibrated_fps > 1.0 else float(self.target_fps)
        self.initial_writer_fps = estimated_fps

        self.cap = cap
        self.camera_info = {
            "device_index": discovery.device_index,
            "device_path": discovery.device_path,
            "backend_name": discovery.backend_name,
            "requested_resolution": [self.target_width, self.target_height],
            "actual_resolution": [actual_w, actual_h],
            "requested_fps": float(self.target_fps),
            "reported_camera_fps": round(reported_camera_fps, 2),
            "measured_effective_fps": 0.0,
            "configured_fps": round(reported_camera_fps, 2),
            "pixel_format": used_fourcc,
            "codec": "mp4v",
        }
        return True

    def initialize_video_writer(self) -> bool:
        w = self.camera_info["actual_resolution"][0]
        h = self.camera_info["actual_resolution"][1]
        fps = self.initial_writer_fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.staging_video_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            LOGGER.error(f"Failed to open VideoWriter with mp4v for {self.staging_video_path}")
            return False

        self.writer = writer
        return True

    def acquire_frame(self, simulated_idx: int = 0) -> tuple[bool, np.ndarray | None]:
        if self.dry_run:
            w, h = self.camera_info["actual_resolution"]
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Add synthetic visual indicator
            cv2.putText(
                frame,
                f"MOCK FRAME {simulated_idx}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            return True, frame

        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            self.frame_read_failures += 1
            return False, None
        return True, frame

    def overlay_hud(
        self,
        frame: np.ndarray,
        phase: str,
        step_idx: int,
        total_steps: int,
        instruction: str,
        countdown: float,
        controls_hint: str = "[P]ause [S]kip [R]epeat [A]bort",
    ) -> np.ndarray:
        hud = frame.copy()
        h, w = hud.shape[:2]

        # Top banner
        cv2.rectangle(hud, (0, 0), (w, 90), (20, 20, 20), -1)
        # Bottom banner
        cv2.rectangle(hud, (0, h - 50), (w, h), (20, 20, 20), -1)

        banner_color = (0, 165, 255) if phase == "PREPARATION" else (0, 0, 255)
        cv2.putText(
            hud,
            f"{phase}: {countdown:.1f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            banner_color,
            2,
        )
        cv2.putText(
            hud,
            f"Step {step_idx}/{total_steps} | Participant: {self.participant_id} | Session: {self.session_id}",
            (w - 550, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        # Truncate instruction if overly long
        inst_text = instruction if len(instruction) <= 80 else instruction[:77] + "..."
        cv2.putText(
            hud,
            inst_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            hud,
            controls_hint,
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )

        return hud

    def run_countdown_phase(
        self,
        phase_name: str,
        duration_seconds: float,
        step_idx: int,
        total_steps: int,
        instruction: str,
        key_listener: Callable[[], str | None] | None = None,
    ) -> tuple[str, float]:
        """Runs a countdown phase while continuously recording camera frames to raw_video.mp4.

        Returns (action, elapsed_phase_time).
        Actions can be: 'completed', 'skipped', 'repeated', 'aborted', 'paused'.
        """
        start_mono = time.monotonic()
        phase_duration = float(duration_seconds)
        simulated_counter = 0

        if phase_duration <= 0.0:
            ok, frame = self.acquire_frame(self.total_frames_written)
            if ok and frame is not None and self.writer is not None:
                self.writer.write(frame)
                self.total_frames_written += 1
            return "completed", 0.0

        paused = False
        pause_start_mono = 0.0

        while True:
            if self.interrupted:
                return "aborted", time.monotonic() - start_mono

            if paused:
                time.sleep(0.05)
                if self.can_gui:
                    pk = cv2.waitKey(50) & 0xFF
                    if pk in (ord("p"), ord(" ")):
                        paused = False
                        p_dur = max(0.0, time.monotonic() - pause_start_mono)
                        self.total_paused_seconds += p_dur
                        start_mono += p_dur
                        print("\n[RESUMED] Resumed capture.")
                if key_listener and key_listener() == "resume":
                    paused = False
                    p_dur = max(0.0, time.monotonic() - pause_start_mono)
                    self.total_paused_seconds += p_dur
                    start_mono += p_dur
                continue

            now_mono = time.monotonic()
            elapsed = now_mono - start_mono
            remaining = max(0.0, phase_duration - elapsed)

            ok, frame = self.acquire_frame(self.total_frames_written + simulated_counter)
            simulated_counter += 1

            if ok and frame is not None:
                if self.writer is not None:
                    self.writer.write(frame)
                    self.total_frames_written += 1

                if self.can_gui:
                    hud_frame = self.overlay_hud(
                        frame,
                        phase_name,
                        step_idx,
                        total_steps,
                        instruction,
                        remaining,
                    )
                    cv2.imshow("Dataset Capture Controller", hud_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("a"), 27):  # ESC or q or a -> abort
                        return "aborted", elapsed
                    if key == ord("s"):
                        return "skipped", elapsed
                    if key == ord("r"):
                        return "repeated", elapsed
                    if key == ord("p"):
                        paused = True
                        pause_start_mono = time.monotonic()
                        print("\n[PAUSED] Capture paused. Press 'p' to resume.")
                        continue

                if self.dry_run and self.duration_scale > 0:
                    fps = self.camera_info.get("configured_fps", 30.0)
                    time.sleep(1.0 / max(1.0, fps))
            else:
                time.sleep(0.01)

            if key_listener:
                k = key_listener()
                if k == "abort":
                    return "aborted", elapsed
                if k == "skip":
                    return "skipped", elapsed
                if k == "repeat":
                    return "repeated", elapsed
                if k == "pause":
                    paused = True
                    pause_start_mono = time.monotonic()
                    continue

            sys.stdout.write(
                f"\r[{phase_name}] Step {step_idx}/{total_steps} | Time: {remaining:4.1f}s | Frames: {self.total_frames_written} "
            )
            sys.stdout.flush()

            if remaining <= 0.0:
                sys.stdout.write("\n")
                return "completed", elapsed

    def execute(
        self,
        input_provider: Callable[[str], str] | None = None,
        key_listener: Callable[[], str | None] | None = None,
    ) -> bool:
        """Executes the guided scenario workflow end-to-end."""
        self.prepare_session_directory()

        print("=" * 70)
        print("  AI PROCTORING DATASET CAPTURE CONTROLLER")
        print("=" * 70)
        print(f"Participant : {self.participant_id}")
        print(f"Session     : {self.session_id}")
        print(f"Scenario    : {self.scenario.name} ({self.scenario.scenario_id})")
        print(f"Output Dir  : {self.session_dir}")
        print("-" * 70)

        # Preflight camera
        print("Initializing camera preflight...")
        if not self.initialize_camera():
            print("ERROR: Camera preflight failed.")
            self.session_status = "camera_init_failed"
            return False

        print(f"Camera Device : {self.camera_info.get('device_path')}")
        print(f"Resolution    : {self.camera_info.get('actual_resolution')[0]}x{self.camera_info.get('actual_resolution')[1]}")
        print(f"Configured FPS: {self.camera_info.get('configured_fps'):.1f}")

        if not self.initialize_video_writer():
            print("ERROR: Could not initialize video writer.")
            self.session_status = "writer_init_failed"
            self.cleanup()
            return False

        print("Video writer initialized. Single raw video stream ready.")
        print("-" * 70)

        # Setup interrupt handler
        def sigint_handler(signum: int, frame: Any) -> None:
            print("\n[!] Ctrl+C received: cleanly stopping recording and finalizing manifest...")
            self.interrupted = True

        old_sigint = signal.signal(signal.SIGINT, sigint_handler)

        self.capture_start_time = time.monotonic()
        self.capture_start_iso = datetime.now(timezone.utc).isoformat()
        self.session_status = "in_progress"

        steps = self.scenario.steps
        step_lookup = {s.step_id: i for i, s in enumerate(steps)}
        step_idx = 0
        total_steps = len(steps)

        try:
            while step_idx < total_steps and not self.interrupted:
                current_step = steps[step_idx]
                step_num = step_idx + 1

                print(f"\n[{step_num}/{total_steps}] STEP: {current_step.step_id}")
                print(f"Instruction : {current_step.instruction}")
                print(f"Condition   : {current_step.instructed_condition}")

                # Handle branching questions if any
                operator_response: str | None = None
                if current_step.question:
                    q = current_step.question
                    print(f"\nQUESTION: {q.prompt}")
                    opts_str = " / ".join(f"[{o.key}] {o.label}" for o in q.options)
                    print(f"Options : {opts_str}")

                    # Ask input
                    chosen_opt: QuestionOption | None = None
                    if input_provider:
                        ans = input_provider(q.prompt).strip().upper()
                    else:
                        ans = input(f"Select option ({'/'.join(o.key for o in q.options)}): ").strip().upper()

                    operator_response = ans
                    for opt in q.options:
                        if opt.key == ans:
                            chosen_opt = opt
                            break

                    if chosen_opt and chosen_opt.next_step:
                        print(f"Branching based on response '{ans}' -> Next step: {chosen_opt.next_step}")
                        q_now = round(self.get_active_session_elapsed(), 3)
                        self.steps_recorded.append(
                            StepRecord(
                                step_id=current_step.step_id,
                                step_index=step_num,
                                instruction=current_step.instruction,
                                instructed_condition=current_step.instructed_condition,
                                preparation_start_seconds=q_now,
                                preparation_end_seconds=q_now,
                                recording_start_seconds=q_now,
                                recording_end_seconds=q_now,
                                duration_seconds=0.0,
                                frame_start_index=self.total_frames_written,
                                frame_end_index=self.total_frames_written,
                                frames_recorded=0,
                                operator_response=ans,
                                status="completed",
                                notes=f"Branch selection: {ans} -> {chosen_opt.next_step}",
                            )
                        )
                        if chosen_opt.next_step in step_lookup:
                            step_idx = step_lookup[chosen_opt.next_step]
                            continue
                        if chosen_opt.next_step == "phone_session_complete" or chosen_opt.next_step.endswith("_complete"):
                            break

                # 1. Preparation Phase
                prep_duration = current_step.preparation_seconds * self.duration_scale
                prep_start_sec = round(self.get_active_session_elapsed(), 3)
                act_prep, elapsed_prep = self.run_countdown_phase(
                    "PREPARATION",
                    prep_duration,
                    step_num,
                    total_steps,
                    current_step.instruction,
                    key_listener=key_listener,
                )
                prep_end_sec = round(self.get_active_session_elapsed(), 3)

                if act_prep == "aborted":
                    print("\n[!] Operator aborted during preparation phase.")
                    self.session_status = "aborted"
                    break
                if act_prep == "skipped":
                    print("\n[*] Step skipped by operator.")
                    self.steps_recorded.append(
                        StepRecord(
                            step_id=current_step.step_id,
                            step_index=step_num,
                            instruction=current_step.instruction,
                            instructed_condition=current_step.instructed_condition,
                            preparation_start_seconds=prep_start_sec,
                            preparation_end_seconds=prep_end_sec,
                            recording_start_seconds=prep_end_sec,
                            recording_end_seconds=prep_end_sec,
                            duration_seconds=0.0,
                            frame_start_index=self.total_frames_written,
                            frame_end_index=self.total_frames_written,
                            frames_recorded=0,
                            operator_response=operator_response,
                            status="skipped",
                            notes=current_step.notes,
                        )
                    )
                    step_idx += 1
                    continue
                if act_prep == "repeated":
                    print("\n[*] Repeating preparation phase.")
                    continue

                # 2. Recording Phase
                rec_start_frame = self.total_frames_written
                rec_duration = current_step.recording_seconds * self.duration_scale
                rec_start_sec = round(self.get_active_session_elapsed(), 3)
                act_rec, elapsed_rec = self.run_countdown_phase(
                    "RECORDING",
                    rec_duration,
                    step_num,
                    total_steps,
                    current_step.instruction,
                    key_listener=key_listener,
                )
                rec_end_sec = round(self.get_active_session_elapsed(), 3)
                rec_end_frame = self.total_frames_written

                status = "completed"
                if act_rec == "aborted":
                    status = "aborted"
                    self.session_status = "aborted"
                elif act_rec == "skipped":
                    status = "skipped"
                elif act_rec == "repeated":
                    status = "repeated"

                self.steps_recorded.append(
                    StepRecord(
                        step_id=current_step.step_id,
                        step_index=step_num,
                        instruction=current_step.instruction,
                        instructed_condition=current_step.instructed_condition,
                        preparation_start_seconds=prep_start_sec,
                        preparation_end_seconds=prep_end_sec,
                        recording_start_seconds=rec_start_sec,
                        recording_end_seconds=rec_end_sec,
                        duration_seconds=round(rec_end_sec - rec_start_sec, 3),
                        frame_start_index=rec_start_frame,
                        frame_end_index=rec_end_frame,
                        frames_recorded=rec_end_frame - rec_start_frame,
                        operator_response=operator_response,
                        status=status,
                        notes=current_step.notes,
                    )
                )

                if act_rec == "aborted":
                    break
                if act_rec == "repeated":
                    print("\n[*] Repeating step recording.")
                    continue

                # Advance to next step (or explicit target)
                if current_step.next_step:
                    if current_step.next_step in step_lookup:
                        step_idx = step_lookup[current_step.next_step]
                        continue
                    if current_step.next_step == "phone_session_complete" or current_step.next_step.endswith("_complete"):
                        break

                step_idx += 1

            if not self.interrupted and self.session_status == "in_progress":
                self.session_status = "completed"
            elif self.interrupted and self.session_status == "in_progress":
                self.session_status = "interrupted"

        except KeyboardInterrupt:
            print("\n[!] KeyboardInterrupt caught in session loop.")
            self.session_status = "interrupted"
        finally:
            signal.signal(signal.SIGINT, old_sigint)
            self.capture_end_time = time.monotonic()
            self.capture_end_iso = datetime.now(timezone.utc).isoformat()
            self.cleanup()
            self.finalize_manifest()

        return self.session_status in ("completed", "interrupted", "aborted")

    def cleanup(self) -> None:
        """Closes video writer, camera, and preview windows cleanly."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.can_gui:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def finalize_manifest(self) -> None:
        """Verifies the captured video, computes checksums, synchronizes container FPS, and writes manifest."""
        # Close writer before reading staging file
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        wall_duration = max(0.0, self.capture_end_time - self.capture_start_time)
        active_duration = max(0.001, wall_duration - self.total_paused_seconds)

        if self.total_frames_written > 0 and active_duration > 0:
            measured_effective_fps = round(self.total_frames_written / active_duration, 2)
        else:
            measured_effective_fps = self.initial_writer_fps

        self.camera_info["measured_effective_fps"] = measured_effective_fps
        self.camera_info["observed_fps"] = measured_effective_fps  # backward-compat

        # Synchronize container FPS to ensure truthful playback duration matching active capture time
        if self.staging_video_path.exists() and self.total_frames_written > 0:
            LOGGER.info(
                f"Synchronizing container FPS to measured effective FPS: {measured_effective_fps:.2f}"
            )
            cap_staging = cv2.VideoCapture(str(self.staging_video_path))
            if cap_staging.isOpened():
                w = int(cap_staging.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap_staging.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                final_writer = cv2.VideoWriter(str(self.video_path), fourcc, measured_effective_fps, (w, h))
                while True:
                    ret, frame = cap_staging.read()
                    if not ret:
                        break
                    final_writer.write(frame)
                cap_staging.release()
                final_writer.release()
                self.staging_video_path.unlink(missing_ok=True)
            else:
                if self.staging_video_path != self.video_path:
                    self.staging_video_path.rename(self.video_path)
        elif self.staging_video_path.exists() and self.staging_video_path != self.video_path:
            self.staging_video_path.rename(self.video_path)

        file_exists = self.video_path.exists()
        file_size = self.video_path.stat().st_size if file_exists else 0
        sha256_hash = calculate_file_sha256(self.video_path) if (file_exists and file_size > 0) else ""

        # Post-capture video readability verification
        verification_passed = False
        reopened_frames = 0
        reopened_duration = 0.0
        reopened_fps = 0.0
        if file_exists and file_size > 0:
            chk_cap = cv2.VideoCapture(str(self.video_path))
            if chk_cap.isOpened():
                reopened_frames = int(chk_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                reopened_fps = round(float(chk_cap.get(cv2.CAP_PROP_FPS)), 2)
                if reopened_fps > 0:
                    reopened_duration = round(reopened_frames / reopened_fps, 3)
                verification_passed = (reopened_frames > 0 and reopened_frames == self.total_frames_written)
                chk_cap.release()

        # Detached checksum file
        if sha256_hash:
            with open(self.checksum_path, "w", encoding="utf-8") as f:
                f.write(f"{sha256_hash}  {self.video_path.name}\n")

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "research_tool": "capture_dataset.py",
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "scenario_id": self.scenario.scenario_id,
            "scenario_name": self.scenario.name,
            "scenario_version": self.scenario.version,
            "capture_start": self.capture_start_iso,
            "capture_end": self.capture_end_iso,
            "total_duration_seconds": round(wall_duration, 3),
            "active_duration_seconds": round(active_duration, 3),
            "paused_duration_seconds": round(self.total_paused_seconds, 3),
            "session_status": self.session_status,
            "camera": self.camera_info,
            "video": {
                "relative_path": self.video_path.name,
                "file_size_bytes": file_size,
                "frames_written": self.total_frames_written,
                "encoded_fps": measured_effective_fps,
                "capture_duration_seconds": round(active_duration, 3),
                "playback_duration_seconds": reopened_duration,
                "reopened_frames_verified": reopened_frames,
                "reopened_fps_verified": reopened_fps,
                "reopened_duration_seconds": reopened_duration,
                "frame_read_failures": self.frame_read_failures,
                "verification_passed": verification_passed,
                "sha256": sha256_hash,
            },
            "steps": [s.to_dict() for s in self.steps_recorded],
        }

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("\n" + "=" * 70)
        print("  CAPTURE SESSION COMPLETE")
        print("=" * 70)
        print(f"Status                  : {self.session_status}")
        print(f"Total Frames            : {self.total_frames_written} written (Verified: {reopened_frames})")
        print(f"Measured Effective FPS  : {measured_effective_fps:.2f} (Encoded: {measured_effective_fps:.2f})")
        print(f"Reported / Nominal FPS  : {self.camera_info.get('reported_camera_fps', 0):.1f} (Requested: {self.camera_info.get('requested_fps', 0):.1f})")
        print(f"Active Capture Duration : {active_duration:.2f}s (Total Wall: {wall_duration:.2f}s, Paused: {self.total_paused_seconds:.2f}s)")
        print(f"Video Playback Duration : {reopened_duration:.2f}s")
        print(f"Raw Video               : {self.video_path} ({file_size} bytes)")
        print(f"Manifest                : {self.manifest_path}")
        print(f"Checksum SHA-256        : {sha256_hash}")
        print(f"Verification            : {'PASS' if verification_passed else 'WARNING / INCOMPLETE'}")
        print("=" * 70)


def list_available_scenarios(scenarios_dir: Path) -> list[ScenarioConfig]:
    scenarios: list[ScenarioConfig] = []
    if not scenarios_dir.exists():
        return scenarios
    for p in sorted(scenarios_dir.glob("*.yaml")):
        try:
            sc = ScenarioConfig.from_yaml_file(p)
            scenarios.append(sc)
        except Exception as e:
            LOGGER.warning(f"Failed to load scenario from {p}: {e}")
    return scenarios


def resolve_scenario(scenario_arg: str, scenarios_dir: Path) -> ScenarioConfig:
    arg_path = Path(scenario_arg)
    if arg_path.is_file():
        return ScenarioConfig.from_yaml_file(arg_path)

    # Search in scenarios_dir
    candidate_yaml = scenarios_dir / f"{scenario_arg}.yaml"
    if candidate_yaml.is_file():
        return ScenarioConfig.from_yaml_file(candidate_yaml)

    # Search by scenario_id
    for p in scenarios_dir.glob("*.yaml"):
        try:
            sc = ScenarioConfig.from_yaml_file(p)
            if sc.scenario_id == scenario_arg:
                return sc
        except Exception:
            continue

    raise FileNotFoundError(
        f"Scenario '{scenario_arg}' could not be resolved as a file or in {scenarios_dir}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Proctoring Isolated Guided Dataset Capture Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check camera hardware preflight
  python tools/capture_dataset.py --check-camera

  # List available experimental scenarios
  python tools/capture_dataset.py --list-scenarios

  # Start guided smartphone dataset capture session
  python tools/capture_dataset.py --participant P001 --session S001 --scenario phone

  # Headless capture without GUI window
  python tools/capture_dataset.py --participant P002 --session S001 --scenario earphones --no-preview

  # Dry-run test capture (synthetic frames, no physical camera required)
  python tools/capture_dataset.py --participant P999 --session S999 --scenario paper --dry-run
        """,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="phone",
        help="Scenario identifier or path to scenario YAML (default: phone)",
    )
    parser.add_argument(
        "--participant",
        type=str,
        default="P001",
        help="Pseudonymous participant identifier (e.g. P001)",
    )
    parser.add_argument(
        "--session",
        type=str,
        default="S001",
        help="Session identifier (e.g. S001)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/capture_sessions"),
        help="Base root directory for session storage (default: data/capture_sessions)",
    )
    parser.add_argument(
        "--scenarios-dir",
        type=Path,
        default=Path("configs/capture_scenarios"),
        help="Directory containing scenario YAML files (default: configs/capture_scenarios)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Specific camera hardware device index override (e.g. 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested capture width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested capture height in pixels (default: 720)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Requested capture frame rate (default: 30.0)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable interactive GUI camera preview window (safe for headless/CLI environments)",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit",
    )
    parser.add_argument(
        "--check-camera",
        action="store_true",
        help="Execute camera discovery and health check preflight, then exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate capture pipeline using synthetic frames without requiring physical camera",
    )
    parser.add_argument(
        "--dry-run-frames",
        type=int,
        default=50,
        help="Number of frames to generate in dry-run mode",
    )
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="Multiplier to scale step durations (useful for accelerated testing/simulation)",
    )
    parser.add_argument(
        "--pixel-format",
        type=str,
        default="AUTO",
        choices=["AUTO", "MJPG", "YUYV"],
        help="Requested camera pixel format/FourCC ('AUTO', 'MJPG', 'YUYV', default: AUTO)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.list_scenarios:
        scenarios = list_available_scenarios(args.scenarios_dir)
        print("\nAvailable Capture Scenarios:")
        print("=" * 60)
        for s in scenarios:
            print(f"- {s.scenario_id:<15} : {s.name} ({len(s.steps)} steps)")
            if s.description:
                print(f"  Description: {s.description.strip()}")
        print("=" * 60)
        return 0

    if args.check_camera:
        print("\nRunning Camera Hardware Preflight Check...")
        print("=" * 60)
        res = discover_local_camera(preferred_index=args.camera_index, target_width=args.width, target_height=args.height)
        for d in res.diagnostics:
            print(f"  {d}")
        print("-" * 60)
        if res.is_valid:
            print(f"PASS: Valid camera found on {res.device_path} (Backend: {res.backend_name}, {res.frame_width}x{res.frame_height} @ {res.fps:.1f} FPS)")
            return 0
        print(f"FAIL: {res.error_message}")
        return 1

    try:
        scenario = resolve_scenario(args.scenario, args.scenarios_dir)
    except Exception as e:
        print(f"Error resolving scenario: {e}")
        return 1

    session = VideoCaptureSession(
        participant_id=args.participant,
        session_id=args.session,
        scenario=scenario,
        output_base_dir=args.output_dir,
        camera_index=args.camera_index,
        target_width=args.width,
        target_height=args.height,
        target_fps=args.fps,
        show_preview=not args.no_preview,
        dry_run=args.dry_run,
        dry_run_frames=args.dry_run_frames,
        duration_scale=args.duration_scale,
        pixel_format=args.pixel_format,
    )

    success = session.execute()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
