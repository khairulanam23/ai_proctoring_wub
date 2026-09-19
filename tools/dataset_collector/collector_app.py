"""Interactive Guided Dataset Collection Application with live camera preview,

mouse/keyboard controls, dual-photo workflow, and automatic manifest generation.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.dataset_collector.camera_utils import (
    DiscoveredCamera,
    enumerate_available_cameras,
    open_camera_stream,
)
from tools.dataset_collector.config_loader import (
    ActivitiesConfig,
    ActivityDefinition,
    load_activities_config,
    resolve_resource_path,
)
from tools.dataset_collector.manifest import (
    APPLICATION_NAME,
    APPLICATION_VERSION,
    ActivityRecord,
    PhotoRecord,
    SessionManifest,
    calculate_sha256,
    check_disk_space,
    generate_session_readme,
    sanitize_identifier,
    write_manifest_and_checksums,
)

LOGGER = logging.getLogger("dataset_collector.app")


@dataclass
class ButtonRect:
    id: str
    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    color: tuple[int, int, int]
    enabled: bool = True

    def contains(self, x: int, y: int) -> bool:
        return self.enabled and (self.x1 <= x <= self.x2) and (self.y1 <= y <= self.y2)


def open_folder_in_file_manager(folder_path: Path) -> bool:
    """Safely launches native Windows File Explorer, macOS Finder, or Linux file manager."""
    try:
        resolved = str(folder_path.resolve())
        if sys.platform.startswith("win"):
            os.startfile(resolved)
            return True
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", resolved])
            return True
        subprocess.Popen(["xdg-open", resolved])
        return True
    except Exception as e:
        LOGGER.warning(f"Could not open directory {folder_path}: {e}")
        return False


class DatasetCollectorApp:
    """Manages the full lifecycle of the interactive dataset collection session."""

    def __init__(
        self,
        participant_id: str = "P001",
        session_id: str = "S001",
        config_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        preferred_camera_index: int | None = None,
        target_width: int = 1280,
        target_height: int = 720,
        dry_run: bool = False,
    ) -> None:
        self.participant_id = sanitize_identifier(participant_id, "Participant ID")
        self.session_id = sanitize_identifier(session_id, "Session ID")
        self.target_width = target_width
        self.target_height = target_height
        self.dry_run = dry_run

        # Load activity configuration
        self.config: ActivitiesConfig = load_activities_config(config_path)
        self.activities: list[ActivityDefinition] = self.config.activities

        # Determine output base directory (default: DatasetOutput next to executable or cwd)
        if output_dir is not None:
            self.output_base = Path(output_dir)
        else:
            self.output_base = resolve_resource_path("DatasetOutput")

        self.session_dir = self.output_base / self.participant_id / self.session_id
        self.images_dir = self.session_dir / "images"

        # Hardware & camera state
        self.available_cameras: list[DiscoveredCamera] = []
        self.selected_camera_idx: int = 0
        self.cap: cv2.VideoCapture | None = None
        self.camera_info: dict[str, Any] = {}
        self.preferred_camera_index = preferred_camera_index

        # Collection state
        self.current_activity_idx: int = 0
        self.activity_records: dict[str, ActivityRecord] = {}
        self.window_name = f"{APPLICATION_NAME} v{APPLICATION_VERSION}"
        self.pending_mouse_action: str | None = None
        self.session_manifest: SessionManifest | None = None
        self.is_running = False

        # Flash animation on capture
        self.flash_counter: int = 0
        self.status_message: str = "Welcome to AI Proctoring Dataset Collector"
        self.status_message_time: float = time.monotonic()

        # Visual button tracking for mouse interaction
        self.active_buttons: list[ButtonRect] = []
        self.camera_online: bool = False

    def set_status(self, msg: str) -> None:
        self.status_message = msg
        self.status_message_time = time.monotonic()

    def discover_cameras(self) -> None:
        """Finds available cameras using platform-appropriate backends."""
        if self.dry_run:
            self.available_cameras = [
                DiscoveredCamera(
                    index=0,
                    backend_name="SYNTHETIC",
                    backend_code=cv2.CAP_ANY,
                    width=self.target_width,
                    height=self.target_height,
                    fps=30.0,
                    is_valid=True,
                    label="Synthetic Mock Camera (Dry Run)",
                )
            ]
            self.selected_camera_idx = 0
            return

        cams = enumerate_available_cameras(max_index=5, target_width=self.target_width, target_height=self.target_height)
        if not cams:
            LOGGER.warning("No physical cameras discovered during enumeration; falling back to index 0.")
            cams = [
                DiscoveredCamera(
                    index=0,
                    backend_name="ANY",
                    backend_code=cv2.CAP_ANY,
                    width=self.target_width,
                    height=self.target_height,
                    fps=30.0,
                    is_valid=True,
                    label="Camera 0 (Default)",
                )
            ]
        self.available_cameras = cams

        # Select preferred index if matched
        if self.preferred_camera_index is not None:
            for i, c in enumerate(self.available_cameras):
                if c.index == self.preferred_camera_index:
                    self.selected_camera_idx = i
                    break
        else:
            self.selected_camera_idx = 0

    def open_current_camera(self) -> bool:
        """Opens video stream for currently selected camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.dry_run:
            self.camera_info = {
                "device_index": 0,
                "backend_name": "SYNTHETIC",
                "actual_resolution": [self.target_width, self.target_height],
                "fps": 30.0,
            }
            return True

        if not self.available_cameras:
            self.discover_cameras()

        selected = self.available_cameras[self.selected_camera_idx]
        cap, info = open_camera_stream(
            selected.index,
            backend_code=selected.backend_code,
            target_width=self.target_width,
            target_height=self.target_height,
        )
        if cap is None or not cap.isOpened():
            LOGGER.error(f"Failed to open camera: {selected.label}")
            return False

        self.cap = cap
        self.camera_info = info
        LOGGER.info(f"Opened camera stream: {selected.label} -> {info}")
        return True

    def switch_to_next_camera(self) -> None:
        """Cycles to the next available camera."""
        if self.activities and self.activity_records:
            curr_act = self.activities[self.current_activity_idx]
            curr_rec = self.activity_records.get(curr_act.id)
            if curr_rec and len(curr_rec.photos) == 1:
                self.set_status("Camera locked: Complete Photo 2 or press [R] to retake Photo 1 before switching.")
                LOGGER.warning("Camera switch blocked mid-activity to preserve camera consistency across photo pair.")
                return

        if len(self.available_cameras) <= 1:
            self.discover_cameras()
            if len(self.available_cameras) <= 1:
                self.set_status("No other cameras detected.")
                return

        self.selected_camera_idx = (self.selected_camera_idx + 1) % len(self.available_cameras)
        success = self.open_current_camera()
        current_cam = self.available_cameras[self.selected_camera_idx]
        if success:
            self.set_status(f"Switched to: {current_cam.label}")
        else:
            self.set_status(f"Failed to switch to: {current_cam.label}")

    def acquire_frame(self) -> np.ndarray:
        """Reads a frame from the open camera or produces a synthetic frame in dry-run mode."""
        if self.dry_run:
            self.camera_online = True
            img = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
            t_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
            cv2.putText(img, f"MOCK CAMERA FEED [{t_str}]", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(img, f"Participant: {self.participant_id} | Session: {self.session_id}", (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            cv2.rectangle(img, (200, 220), (self.target_width - 200, self.target_height - 100), (40, 40, 40), 2)
            cv2.putText(img, "ALIGN PARTICIPANT & WORKSPACE WITHIN THIS REGION", (self.target_width // 2 - 380, self.target_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
            return img

        if self.cap is None or not self.cap.isOpened():
            self.open_current_camera()

        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                self.camera_online = True
                return frame

        self.camera_online = False
        # Fallback offline frame
        fallback = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        cv2.putText(fallback, "CAMERA STREAM DISCONNECTED", (self.target_width // 2 - 250, self.target_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(fallback, "Check USB connection / Press [C] to retry", (self.target_width // 2 - 260, self.target_height // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        return fallback

    def init_session_records(self) -> None:
        """Initializes records, supporting resume from an existing session folder."""
        # Ensure disk space
        ok, free_bytes, _ = check_disk_space(self.session_dir)
        if not ok:
            raise OSError(f"Insufficient disk space in {self.session_dir}. At least 500 MB required.")

        self.images_dir.mkdir(parents=True, exist_ok=True)

        for act in self.activities:
            act_dir = self.images_dir / act.id
            existing_photos: list[PhotoRecord] = []

            # Check if photos exist on disk (resumption support)
            if act_dir.exists():
                for p_idx in (1, 2):
                    p_file = act_dir / f"photo_{p_idx:02d}.jpg"
                    if p_file.exists() and p_file.stat().st_size > 0:
                        img_read = cv2.imread(str(p_file))
                        h, w = img_read.shape[:2] if img_read is not None else (0, 0)
                        existing_photos.append(
                            PhotoRecord(
                                photo_number=p_idx,
                                filename=p_file.name,
                                relative_path=f"images/{act.id}/{p_file.name}",
                                width=w,
                                height=h,
                                file_size_bytes=p_file.stat().st_size,
                                sha256=calculate_sha256(p_file),
                                captured_at=datetime.fromtimestamp(p_file.stat().st_mtime, timezone.utc).isoformat(),
                            )
                        )

            status = "completed" if len(existing_photos) >= 2 else ("incomplete" if existing_photos else "pending")
            self.activity_records[act.id] = ActivityRecord(
                activity_id=act.id,
                activity_name=act.name,
                category=act.category,
                instructed_condition=act.instructed_condition,
                purpose=act.purpose,
                status=status,
                photos=existing_photos,
            )

        # Find first incomplete activity
        for i, act in enumerate(self.activities):
            rec = self.activity_records[act.id]
            if len(rec.photos) < 2:
                self.current_activity_idx = i
                break
        else:
            self.current_activity_idx = len(self.activities) - 1

        total_captured = sum(len(r.photos) for r in self.activity_records.values())
        if total_captured > 0:
            self.set_status(f"Resumed existing session: {total_captured} photos already saved.")

    def on_mouse_event(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handles mouse clicks on visual HUD buttons."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn in self.active_buttons:
                if btn.contains(x, y):
                    self.pending_mouse_action = btn.id
                    break

    def capture_current_photo(self, raw_frame: np.ndarray) -> bool:
        """Saves current camera frame as photo 1 or photo 2 for the active activity."""
        if not self.dry_run and not self.camera_online:
            self.set_status("ERROR: Cannot capture photo while camera is disconnected!")
            LOGGER.warning("Capture rejected: camera is disconnected or frame read failed.")
            return False

        curr_act = self.activities[self.current_activity_idx]
        record = self.activity_records[curr_act.id]

        target_photo_num = len(record.photos) + 1
        if target_photo_num > 2:
            self.set_status(f"Both photos already taken for {curr_act.id}. Press [R] to retake or [N] for next.")
            return False

        act_dir = self.images_dir / curr_act.id
        act_dir.mkdir(parents=True, exist_ok=True)

        filename = f"photo_{target_photo_num:02d}.jpg"
        filepath = act_dir / filename

        # Save high quality JPEG (quality 95)
        success = cv2.imwrite(str(filepath), raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success or not filepath.exists() or filepath.stat().st_size == 0:
            self.set_status("ERROR: Failed to save photo to disk!")
            return False

        h, w = raw_frame.shape[:2]
        sha256_hash = calculate_sha256(filepath)
        captured_time = datetime.now(timezone.utc).isoformat()

        new_photo = PhotoRecord(
            photo_number=target_photo_num,
            filename=filename,
            relative_path=f"images/{curr_act.id}/{filename}",
            width=w,
            height=h,
            file_size_bytes=filepath.stat().st_size,
            sha256=sha256_hash,
            captured_at=captured_time,
        )
        record.photos.append(new_photo)

        if len(record.photos) >= 2:
            record.status = "completed"

        # Trigger visual shutter flash
        self.flash_counter = 4
        self.set_status(f"Saved Photo {target_photo_num} for {curr_act.id} ({curr_act.name})")
        LOGGER.info(f"Captured {new_photo.relative_path} [{sha256_hash[:10]}]")
        return True

    def retake_photo(self) -> None:
        """Removes the most recent photo of the active activity so it can be re-captured."""
        curr_act = self.activities[self.current_activity_idx]
        record = self.activity_records[curr_act.id]

        if not record.photos:
            self.set_status("No photos captured yet for this activity to retake.")
            return

        removed = record.photos.pop()
        p_file = self.session_dir / removed.relative_path
        if p_file.exists():
            p_file.unlink(missing_ok=True)

        record.status = "incomplete"
        self.set_status(f"Removed Photo {removed.photo_number}. Ready to retake.")

    def render_hud(self, frame: np.ndarray, state: str = "CAPTURE") -> np.ndarray:
        """Composes a professional, high-contrast, clickable UI around the camera preview."""
        canvas_w = 1280
        canvas_h = 820
        top_h = 160
        bottom_h = 100
        view_h = canvas_h - top_h - bottom_h  # 560px

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        self.active_buttons.clear()

        # 1. Top Panel (Dark Slate / Charcoal)
        cv2.rectangle(canvas, (0, 0), (canvas_w, top_h), (25, 25, 30), -1)
        cv2.line(canvas, (0, top_h), (canvas_w, top_h), (60, 60, 70), 2)

        # Title and session status
        app_title = f"{APPLICATION_NAME} v{APPLICATION_VERSION}"
        cv2.putText(canvas, app_title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        sess_str = f"Participant: {self.participant_id} | Session: {self.session_id}"
        cv2.putText(canvas, sess_str, (canvas_w - 440, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        curr_act = self.activities[self.current_activity_idx]
        curr_rec = self.activity_records[curr_act.id]

        # Activity Header & Category
        cat_color = (180, 100, 0)
        if "PHONE" in curr_act.category:
            cat_color = (0, 140, 255)
        elif "PAPER" in curr_act.category:
            cat_color = (0, 200, 100)
        elif "EAR" in curr_act.category:
            cat_color = (220, 60, 180)

        cv2.putText(
            canvas,
            f"Activity {self.current_activity_idx + 1} of {len(self.activities)}: {curr_act.name}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
        )
        # Category tag
        cv2.putText(canvas, f"[{curr_act.category}]", (canvas_w - 300, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cat_color, 2)

        # Instructions
        act_text = f"Action : {curr_act.physical_action}"
        if len(act_text) > 105:
            act_text = act_text[:102] + "..."
        vis_text = f"Visible: {curr_act.visible_state}"
        if len(vis_text) > 105:
            vis_text = vis_text[:102] + "..."

        cv2.putText(canvas, act_text, (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(canvas, vis_text, (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(canvas, f"Condition: {curr_act.instructed_condition}", (20, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

        # 2. Camera View (Center)
        # Fit frame into (canvas_w, view_h) preserving aspect ratio
        fh, fw = frame.shape[:2]
        scale = min(canvas_w / fw, view_h / fh)
        scaled_w = int(fw * scale)
        scaled_h = int(fh * scale)
        resized_frame = cv2.resize(frame, (scaled_w, scaled_h))

        offset_x = (canvas_w - scaled_w) // 2
        offset_y = top_h + (view_h - scaled_h) // 2
        canvas[offset_y : offset_y + scaled_h, offset_x : offset_x + scaled_w] = resized_frame

        # Overlay shutter flash if active
        if self.flash_counter > 0:
            white_flash = np.full_like(canvas[offset_y : offset_y + scaled_h, offset_x : offset_x + scaled_w], 255)
            alpha = self.flash_counter * 0.2
            canvas[offset_y : offset_y + scaled_h, offset_x : offset_x + scaled_w] = cv2.addWeighted(
                white_flash, alpha, canvas[offset_y : offset_y + scaled_h, offset_x : offset_x + scaled_w], 1.0 - alpha, 0
            )
            self.flash_counter -= 1

        # Current Capture Target Badge (Centered above frame)
        n_photos = len(curr_rec.photos)
        if n_photos == 0:
            target_text = f"TARGET: Photo 1 of 2 -> {curr_act.photo_1}"
            badge_color = (0, 120, 255)
        elif n_photos == 1:
            target_text = f"TARGET: Photo 2 of 2 -> {curr_act.photo_2}"
            badge_color = (0, 180, 220)
        else:
            target_text = "✓ Both photos captured for this activity! Press [Next] to continue."
            badge_color = (0, 180, 0)

        # Draw target banner inside video view
        cv2.rectangle(canvas, (offset_x + 10, offset_y + 10), (offset_x + scaled_w - 10, offset_y + 45), (15, 15, 20), -1)
        cv2.putText(canvas, target_text, (offset_x + 25, offset_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, badge_color, 2)

        # Photo status indicators (Right side overlay)
        p1_color = (0, 180, 0) if n_photos >= 1 else (60, 60, 60)
        p1_text = "Photo 1: SAVED ✓" if n_photos >= 1 else "Photo 1: PENDING"
        cv2.rectangle(canvas, (canvas_w - 230, offset_y + 55), (canvas_w - 20, offset_y + 90), p1_color, -1)
        cv2.putText(canvas, p1_text, (canvas_w - 220, offset_y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        p2_color = (0, 180, 0) if n_photos >= 2 else (60, 60, 60)
        p2_text = "Photo 2: SAVED ✓" if n_photos >= 2 else "Photo 2: PENDING"
        cv2.rectangle(canvas, (canvas_w - 230, offset_y + 100), (canvas_w - 20, offset_y + 135), p2_color, -1)
        cv2.putText(canvas, p2_text, (canvas_w - 220, offset_y + 123), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 3. Bottom Controls Panel
        cv2.rectangle(canvas, (0, canvas_h - bottom_h), (canvas_w, canvas_h), (25, 25, 30), -1)
        cv2.line(canvas, (0, canvas_h - bottom_h), (canvas_w, canvas_h - bottom_h), (60, 60, 70), 2)

        # Status line
        cv2.putText(canvas, f"STATUS: {self.status_message}", (20, canvas_h - bottom_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

        # Buttons definition
        btn_y1 = canvas_h - bottom_h + 38
        btn_y2 = btn_y1 + 45

        # Button: [Capture]
        btn_capture_label = f"Capture Photo {min(2, n_photos + 1)}" if n_photos < 2 else "Complete ✓"
        btn_capture_color = (0, 160, 0) if n_photos < 2 else (60, 60, 60)
        self.active_buttons.append(ButtonRect("capture", f"[SPACE] {btn_capture_label}", 20, btn_y1, 230, btn_y2, btn_capture_color, enabled=(n_photos < 2)))

        # Button: [Retake]
        self.active_buttons.append(ButtonRect("retake", "[R] Retake", 240, btn_y1, 360, btn_y2, (0, 120, 200), enabled=(n_photos > 0)))

        # Button: [Next]
        next_enabled = (n_photos >= 2) or (self.current_activity_idx < len(self.activities) - 1)
        self.active_buttons.append(ButtonRect("next", "[N] Next >", 370, btn_y1, 480, btn_y2, (180, 120, 0), enabled=next_enabled))

        # Button: [Back]
        back_enabled = self.current_activity_idx > 0
        self.active_buttons.append(ButtonRect("back", "[B] < Back", 490, btn_y1, 600, btn_y2, (80, 80, 80), enabled=back_enabled))

        # Button: [Camera]
        self.active_buttons.append(ButtonRect("camera", "[C] Switch Cam", 610, btn_y1, 760, btn_y2, (160, 80, 30), enabled=True))

        # Button: [Open Folder]
        self.active_buttons.append(ButtonRect("folder", "[O] Open Folder", 770, btn_y1, 930, btn_y2, (140, 40, 140), enabled=True))

        # Button: [Finish / Save]
        finish_label = "[Q] Finish & Exit"
        self.active_buttons.append(ButtonRect("finish", finish_label, 940, btn_y1, 1100, btn_y2, (30, 30, 180), enabled=True))

        # Draw all buttons
        for btn in self.active_buttons:
            bg_color = btn.color if btn.enabled else (45, 45, 50)
            text_color = (255, 255, 255) if btn.enabled else (100, 100, 100)
            cv2.rectangle(canvas, (btn.x1, btn.y1), (btn.x2, btn.y2), bg_color, -1)
            cv2.rectangle(canvas, (btn.x1, btn.y1), (btn.x2, btn.y2), (80, 80, 90), 1)
            cv2.putText(canvas, btn.label, (btn.x1 + 10, btn.y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        return canvas

    def render_camera_setup_screen(self, frame: np.ndarray) -> np.ndarray:
        """Renders the startup camera selection and framing preview screen."""
        canvas_w = 1280
        canvas_h = 800
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        self.active_buttons.clear()

        # Top Header
        cv2.rectangle(canvas, (0, 0), (canvas_w, 90), (25, 25, 30), -1)
        cv2.putText(canvas, f"{APPLICATION_NAME} — Camera Setup & Framing Check", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        cv2.putText(canvas, "Verify: Face, upper torso, hands, and desk area are clearly visible.", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Center Video Stream
        fh, fw = frame.shape[:2]
        scale = min(canvas_w / fw, 580 / fh)
        sw, sh = int(fw * scale), int(fh * scale)
        rf = cv2.resize(frame, (sw, sh))
        ox, oy = (canvas_w - sw) // 2, 100 + (580 - sh) // 2
        canvas[oy : oy + sh, ox : ox + sw] = rf

        # Framing guide rectangle
        guide_w, guide_h = int(sw * 0.85), int(sh * 0.85)
        gx, gy = ox + (sw - guide_w) // 2, oy + (sh - guide_h) // 2
        cv2.rectangle(canvas, (gx, gy), (gx + guide_w, gy + guide_h), (0, 255, 200), 2)
        cv2.putText(canvas, "OPTIMAL FRAMING ZONE", (gx + 10, gy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)

        # Current Camera Label
        cam_label = self.available_cameras[self.selected_camera_idx].label if self.available_cameras else "Camera 0"
        cv2.rectangle(canvas, (ox + 20, oy + 20), (ox + 450, oy + 60), (20, 20, 25), -1)
        cv2.putText(canvas, f"Active: {cam_label}", (ox + 30, oy + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Bottom Buttons
        btn_y1 = canvas_h - 90
        btn_y2 = btn_y1 + 55

        # [Confirm & Begin]
        self.active_buttons.append(ButtonRect("confirm_camera", "[ENTER] Confirm Camera & Start", 20, btn_y1, 380, btn_y2, (0, 160, 0), enabled=True))
        # [Switch Camera]
        self.active_buttons.append(ButtonRect("switch_camera", "[C] Switch / Try Next Camera", 400, btn_y1, 740, btn_y2, (180, 90, 20), enabled=True))
        # [Exit]
        self.active_buttons.append(ButtonRect("exit_setup", "[Q] Exit", 760, btn_y1, 920, btn_y2, (40, 40, 180), enabled=True))

        for btn in self.active_buttons:
            cv2.rectangle(canvas, (btn.x1, btn.y1), (btn.x2, btn.y2), btn.color, -1)
            cv2.rectangle(canvas, (btn.x1, btn.y1), (btn.x2, btn.y2), (90, 90, 100), 1)
            cv2.putText(canvas, btn.label, (btn.x1 + 20, btn.y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return canvas

    def finalize_session(self, aborted: bool = False) -> Path:
        """Writes manifest, checksums, and README to the session folder."""
        total_photos = sum(len(r.photos) for r in self.activity_records.values())
        completed_activities = sum(1 for r in self.activity_records.values() if r.status == "completed")
        expected_total_photos = len(self.activities) * self.config.photos_per_activity

        if aborted:
            status = "aborted"
        elif completed_activities == len(self.activities) and total_photos == expected_total_photos:
            status = "completed"
        else:
            status = "incomplete"

        manifest = SessionManifest(
            participant_id=self.participant_id,
            session_id=self.session_id,
            camera=self.camera_info,
            activities=list(self.activity_records.values()),
            session_status=status,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        manifest_path, checksum_path = write_manifest_and_checksums(self.session_dir, manifest)
        generate_session_readme(self.session_dir, self.participant_id, self.session_id)

        print("\n" + "=" * 70)
        print("  DATASET COLLECTION SESSION FINALIZED")
        print("=" * 70)
        print(f"Status              : {status.upper()}")
        print(f"Participant ID      : {self.participant_id}")
        print(f"Session ID          : {self.session_id}")
        print(f"Activities Done     : {completed_activities} / {len(self.activities)}")
        print(f"Total Photos Saved  : {total_photos}")
        print(f"Output Directory    : {self.session_dir}")
        print(f"Manifest            : {manifest_path}")
        print(f"Checksum File       : {checksum_path}")
        print("=" * 70)

        return self.session_dir

    def run(self) -> int:
        """Main application execution loop."""
        self.is_running = True
        self.discover_cameras()
        if not self.open_current_camera():
            LOGGER.error("Initial camera opening failed.")

        self.init_session_records()

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.on_mouse_event)

        current_state = "SETUP"  # "SETUP", "CAPTURE", "COMPLETE"

        try:
            while self.is_running:
                # Check if window closed by user
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    LOGGER.info("Application window closed by user.")
                    break

                raw_frame = self.acquire_frame()
                self.pending_mouse_action = None

                if current_state == "SETUP":
                    rendered = self.render_camera_setup_screen(raw_frame)
                    cv2.imshow(self.window_name, rendered)
                    key = cv2.waitKey(20) & 0xFF

                    action = self.pending_mouse_action
                    if key in (13, 32) or action == "confirm_camera":  # ENTER or SPACE
                        current_state = "CAPTURE"
                        self.set_status(f"Ready for Activity 1: {self.activities[0].name}")
                    elif key in (ord("c"), ord("C")) or action == "switch_camera":
                        self.switch_to_next_camera()
                    elif key in (27, ord("q"), ord("Q")) or action == "exit_setup":
                        LOGGER.info("User exited from setup screen.")
                        break

                elif current_state == "CAPTURE":
                    rendered = self.render_hud(raw_frame, state="CAPTURE")
                    cv2.imshow(self.window_name, rendered)
                    key = cv2.waitKey(20) & 0xFF

                    action = self.pending_mouse_action

                    # 1. Capture photo [SPACE]
                    if key == 32 or action == "capture":
                        self.capture_current_photo(raw_frame)
                        curr_rec = self.activity_records[self.activities[self.current_activity_idx].id]
                        # If both captured, automatically advance if next is incomplete
                        if len(curr_rec.photos) >= 2 and self.current_activity_idx < len(self.activities) - 1:
                            pass

                    # 2. Retake [R]
                    elif key in (ord("r"), ord("R")) or action == "retake":
                        self.retake_photo()

                    # 3. Next Activity [N] / [ENTER]
                    elif key in (ord("n"), ord("N"), 13) or action == "next":
                        if self.current_activity_idx < len(self.activities) - 1:
                            self.current_activity_idx += 1
                            next_act = self.activities[self.current_activity_idx]
                            self.set_status(f"Activity {self.current_activity_idx + 1}: {next_act.name}")
                        else:
                            # Reached the end
                            self.set_status("All activities reviewed! Finalizing session.")
                            current_state = "COMPLETE"

                    # 4. Previous Activity [P] / [B]
                    elif key in (ord("p"), ord("P"), ord("b"), ord("B")) or action == "back":
                        if self.current_activity_idx > 0:
                            self.current_activity_idx -= 1
                            prev_act = self.activities[self.current_activity_idx]
                            self.set_status(f"Activity {self.current_activity_idx + 1}: {prev_act.name}")

                    # 5. Switch camera [C]
                    elif key in (ord("c"), ord("C")) or action == "camera":
                        self.switch_to_next_camera()

                    # 6. Open output folder [O]
                    elif key in (ord("o"), ord("O")) or action == "folder":
                        self.set_status("Opening output directory in File Explorer...")
                        open_folder_in_file_manager(self.session_dir)

                    # 7. Finish / Quit [Q] / [ESC]
                    elif key in (27, ord("q"), ord("Q")) or action == "finish":
                        LOGGER.info("User requested finish/exit.")
                        break

                elif current_state == "COMPLETE":
                    # Display completion frame
                    rendered = self.render_hud(raw_frame, state="COMPLETE")
                    cv2.imshow(self.window_name, rendered)
                    key = cv2.waitKey(20) & 0xFF
                    action = self.pending_mouse_action

                    if key in (ord("o"), ord("O")) or action == "folder":
                        open_folder_in_file_manager(self.session_dir)
                    elif key in (27, ord("q"), ord("Q")) or action == "finish":
                        break

        except KeyboardInterrupt:
            LOGGER.warning("KeyboardInterrupt caught.")
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

            # Finalize session manifest
            self.finalize_session()

        return 0


def main() -> int:
    """CLI entry point for Dataset Collector."""
    import argparse

    parser = argparse.ArgumentParser(description=f"{APPLICATION_NAME} — Still-Image Collector")
    parser.add_argument("--participant", default="P001", help="Participant ID (default: P001)")
    parser.add_argument("--session", default="S001", help="Session ID (default: S001)")
    parser.add_argument("--config", default=None, help="Path to activities.yaml configuration")
    parser.add_argument("--output-dir", default=None, help="Directory to store collected datasets")
    parser.add_argument("--camera-index", type=int, default=None, help="Camera hardware index override")
    parser.add_argument("--dry-run", action="store_true", help="Run without physical camera (synthetic test)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = DatasetCollectorApp(
        participant_id=args.participant,
        session_id=args.session,
        config_path=args.config,
        output_dir=args.output_dir,
        preferred_camera_index=args.camera_index,
        dry_run=args.dry_run,
    )
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
