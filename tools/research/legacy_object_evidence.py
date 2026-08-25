"""Structured evidence persistence, key-frame selection, object cropping, and multi-modal packaging."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import zipfile
import cv2
import numpy as np

from proctoring.detection.object_detector import DetectedObject
from proctoring.capture.video import VideoMetadata, format_timestamp
from tools.research.legacy_object_temporal import ObjectPresenceEvent, PersonCountChangeEvent, TemporalEventReport
from tools.research.legacy_object_video import TimelineEntry, VideoAnalysisReport


@dataclass
class EvidenceObject:
    """Individual object detected on a key evidence frame with exact bbox, instance ID, and factual reason."""
    object_id: str
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    instance_id: str = ""
    reason: str = ""
    timestamp_seconds: float = 0.0
    frame_index: int = 0
    crop_relative_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.reason:
            self.reason = f"Detected object class: {self.class_name.lower()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "instance_id": self.instance_id or self.object_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "reason": self.reason if self.reason else f"Detected object class: {self.class_name.lower()}",
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "frame_index": self.frame_index,
            "crop_relative_path": self.crop_relative_path,
        }


@dataclass
class EvidenceFrame:
    """Key evidence visual frame selected from temporal transitions, peaks, or events."""
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    formatted_timestamp: str
    person_count: int
    objects: List[EvidenceObject]
    frame_relative_path: str
    width: int
    height: int
    selection_reason: str
    explanation: str = ""
    video_position_pct: float = 0.0
    event_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "formatted_timestamp": self.formatted_timestamp,
            "person_count": self.person_count,
            "frame_relative_path": self.frame_relative_path,
            "width": self.width,
            "height": self.height,
            "selection_reason": self.selection_reason,
            "explanation": self.explanation,
            "video_position_pct": round(self.video_position_pct, 2),
            "event_context": self.event_context,
            "objects": [obj.to_dict() for obj in self.objects],
        }


@dataclass
class EvidenceEvent:
    """Consolidated evidence event for human proctor review."""
    event_id: str
    modality: str  # "object", "face", "browser", "audio", "system"
    event_type: str  # "temporal_object_presence", "person_count_change", "face_observation"
    object_class: Optional[str]
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    formatted_start: str
    formatted_end: str
    detection_count: int
    max_confidence: float
    average_confidence: float
    is_duration_qualified: bool
    reason: str = ""
    key_frame_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reason:
            if self.object_class:
                self.reason = f"Detected object class: {self.object_class.lower()}"
            else:
                self.reason = f"Observation for {self.event_type}"

    def to_dict(self) -> Dict[str, Any]:
        effective_reason = self.reason if self.reason else f"Detected object class: {str(self.object_class).lower()}"
        return {
            "event_id": self.event_id,
            "modality": self.modality,
            "event_type": self.event_type,
            "object_class": self.object_class,
            "reason": effective_reason,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "formatted_start": self.formatted_start,
            "formatted_end": self.formatted_end,
            "detection_count": self.detection_count,
            "max_confidence": round(self.max_confidence, 4),
            "average_confidence": round(self.average_confidence, 4),
            "is_duration_qualified": self.is_duration_qualified,
            "key_frame_ids": self.key_frame_ids,
            "metadata": self.metadata,
        }


@dataclass
class EvidenceSource:
    """Metadata describing the input examination media stream."""
    type: str  # "video", "webcam_stream"
    filename: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    total_frames: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "filename": self.filename,
            "duration_seconds": round(self.duration_seconds, 2),
            "fps": round(self.fps, 2),
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
        }


@dataclass
class EvidenceManifest:
    """Machine-readable manifest summarizing all evidence assets in the package."""
    schema_version: str
    package_id: str
    created_at: str
    source: EvidenceSource
    summary: Dict[str, Any]
    events: List[EvidenceEvent]
    frames: List[EvidenceFrame]
    artifacts: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "package_id": self.package_id,
            "created_at": self.created_at,
            "source": self.source.to_dict(),
            "summary": self.summary,
            "events": [e.to_dict() for e in self.events],
            "frames": [f.to_dict() for f in self.frames],
            "artifacts": self.artifacts,
        }


@dataclass
class EvidencePackage:
    """In-memory and filesystem representation of an evidence package."""
    package_dir: Path
    manifest: EvidenceManifest
    zip_path: Optional[Path] = None


class FaceEvidenceAdapter:
    """Contract interface allowing future face observations to integrate cleanly into EvidenceEvents."""

    @staticmethod
    def create_face_presence_event(
        event_id: str,
        status: str,
        start_seconds: float,
        end_seconds: float,
        frame_indices: List[int],
        face_count: int,
        key_frame_ids: Optional[List[str]] = None,
    ) -> EvidenceEvent:
        """Construct a standardized EvidenceEvent from face presence observations."""
        duration = max(0.0, end_seconds - start_seconds)
        return EvidenceEvent(
            event_id=event_id,
            modality="face",
            event_type="face_observation",
            object_class="face",
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            duration_seconds=duration,
            formatted_start=format_timestamp(start_seconds),
            formatted_end=format_timestamp(end_seconds),
            detection_count=len(frame_indices),
            max_confidence=1.0,
            average_confidence=1.0,
            is_duration_qualified=True,
            reason=f"Face presence state: {status} ({face_count} face(s) visible)",
            key_frame_ids=key_frame_ids or [],
            metadata={
                "presence_status": status,
                "face_count": face_count,
                "frame_indices": frame_indices,
            },
        )


class EvidenceBuilder:
    """Builder responsible for self-contained visual evidence generation, key-frame selection, and packaging."""

    PALETTE = {
        "person_01": (46, 204, 113),       # Primary candidate: Emerald Green
        "person_multi": (231, 76, 60),      # Secondary person: Crimson Red (Alert)
        "cell phone": (230, 126, 34),       # Mobile phone: Amber Orange
        "laptop": (52, 152, 219),           # Laptop: Sky Blue
        "book": (155, 89, 182),             # Book: Amethyst Purple
        "tablet": (26, 188, 156),           # Tablet: Teal
        "keyboard": (241, 196, 15),         # Keyboard: Sunflower Yellow
        "mouse": (230, 100, 100),           # Mouse: Coral
        "backpack": (52, 73, 94),           # Backpack: Midnight Navy
        "bottle": (22, 160, 133),           # Bottle: Sea Green
    }

    def __init__(
        self,
        crop_padding_ratio: float = 0.10,
        jpeg_quality: int = 95,
    ) -> None:
        self.crop_padding_ratio = float(crop_padding_ratio)
        self.jpeg_quality = int(jpeg_quality)

    @classmethod
    def get_factual_detection_reason(
        cls,
        class_name: str,
        instance_index: int = 1,
        person_count: int = 1,
        is_person_transition: bool = False,
        prev_count: int = 0,
        new_count: int = 0,
    ) -> str:
        """Generate a strictly factual, objective reason explaining why a region was marked."""
        c = class_name.lower().strip()
        if is_person_transition:
            return f"Person count changed from {prev_count} to {new_count}"
        if c == "person":
            if person_count > 1 and instance_index > 1:
                return f"Detected object class: person (#{instance_index:02d} in scene)"
            return "Detected object class: person"
        return f"Detected object class: {c}"

    @classmethod
    def crop_object(
        cls,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        class_name: str = "object",
        confidence: float = 1.0,
        instance_id: str = "obj_01",
        frame_index: int = 0,
        timestamp_seconds: float = 0.0,
        reason: str = "",
        padding_ratio: float = 0.10,
        add_header: bool = True,
    ) -> np.ndarray:
        """Safely crop object bounding box with bounds clamping, padding, and a self-contained metadata header."""
        if image is None or image.size == 0:
            raise ValueError("Cannot crop from empty or invalid image array.")

        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # Safe Coordinate Clamping
        safe_x1 = max(0, min(w - 1, int(x1)))
        safe_y1 = max(0, min(h - 1, int(y1)))
        safe_x2 = max(safe_x1 + 1, min(w, int(x2)))
        safe_y2 = max(safe_y1 + 1, min(h, int(y2)))

        # Calculate padding
        box_w = max(1, safe_x2 - safe_x1)
        box_h = max(1, safe_y2 - safe_y1)
        pad_x = int(box_w * padding_ratio)
        pad_y = int(box_h * padding_ratio)

        crop_x1 = max(0, safe_x1 - pad_x)
        crop_y1 = max(0, safe_y1 - pad_y)
        crop_x2 = min(w, safe_x2 + pad_x)
        crop_y2 = min(h, safe_y2 + pad_y)

        crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        if crop.size == 0:
            crop = image[safe_y1:safe_y2, safe_x1:safe_x2].copy()

        if not add_header:
            return crop

        # Scale banner dynamically based on crop size
        crop_h, crop_w = crop.shape[:2]
        banner_w = max(380, crop_w)
        banner_h = 62
        banner = np.full((banner_h, banner_w, 3), 22, dtype=np.uint8)

        badge_color = cls.PALETTE.get(class_name.lower(), (0, 215, 255))
        cv2.rectangle(banner, (0, 0), (6, banner_h), badge_color, -1)

        t_str = format_timestamp(timestamp_seconds)
        header_l1 = f"{class_name.upper()} | {int(confidence * 100)}% | ID: {instance_id}"
        header_l2 = f"TIME: {t_str} | FRAME: {frame_index:05d} | BBOX: ({safe_x1},{safe_y1}) -> ({safe_x2},{safe_y2})"
        header_l3 = f"WHY MARKED: {reason if reason else cls.get_factual_detection_reason(class_name)}"

        cv2.putText(banner, header_l1, (14, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(banner, header_l2, (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(banner, header_l3, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 215, 255), 1, cv2.LINE_AA)

        if crop_w < banner_w:
            padded_crop = np.full((crop_h, banner_w, 3), 18, dtype=np.uint8)
            pad_offset = (banner_w - crop_w) // 2
            padded_crop[:, pad_offset : pad_offset + crop_w] = crop
            crop = padded_crop

        return np.vstack([banner, crop])

    @classmethod
    def render_annotated_evidence_frame(
        cls,
        image: np.ndarray,
        frame_index: int,
        timestamp_seconds: float,
        person_count: int,
        objects: List[DetectedObject],
        selection_reason: str,
        video_filename: str = "exam_video.mp4",
        total_video_frames: int = 0,
        total_video_duration_seconds: float = 0.0,
        event_context: Optional[Dict[str, Any]] = None,
        person_change_context: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Render a self-contained, publication-quality visual evidence canvas with zero external dependencies.

        Supports portrait (1080x1920), landscape (1920x1080), 720p, 480p, and variable resolutions.
        """
        h, w = image.shape[:2]
        canvas = image.copy()

        # Dynamic Scale Factor based on resolution
        is_portrait = h > w
        scale_ref = max(0.42, min(1.10, max(h, w) / 1350.0))
        font_scale_tag = max(0.40, scale_ref * 0.52)

        # Video Position % calculation
        if total_video_frames > 0:
            video_pos_pct = (frame_index / total_video_frames) * 100.0
        elif total_video_duration_seconds > 0:
            video_pos_pct = (timestamp_seconds / total_video_duration_seconds) * 100.0
        else:
            video_pos_pct = 0.0

        ts_str = format_timestamp(timestamp_seconds)
        person_idx = 0
        obj_info_list = []

        # 1. Annotate Bounding Boxes, High-Contrast Outlines, Tags, and Precision Coordinates
        for i, obj in enumerate(objects, start=1):
            cname = obj.class_name.lower().strip()
            if cname == "person":
                person_idx += 1
                inst_id = f"person_{person_idx:02d}"
                label_name = f"PERSON #{person_idx:02d}"
                color = cls.PALETTE["person_01"] if person_idx == 1 else cls.PALETTE["person_multi"]
            else:
                inst_id = f"{cname.replace(' ', '_')}_{i:02d}"
                label_name = f"{obj.class_name.upper()} #{i:02d}"
                color = cls.PALETTE.get(cname, (0, 215, 255))

            # Clamped coordinates
            x1 = max(0, min(w - 1, int(obj.x1)))
            y1 = max(0, min(h - 1, int(obj.y1)))
            x2 = max(x1 + 1, min(w, int(obj.x2)))
            y2 = max(y1 + 1, min(h, int(obj.y2)))

            # Double outline (outer dark shadow + inner high-contrast color)
            border_thickness = max(2, int(round(scale_ref * 3.0)))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), border_thickness + 2, cv2.LINE_AA)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, border_thickness, cv2.LINE_AA)

            # Precision corner brackets
            c_len = max(8, min(28, (x2 - x1) // 5, (y2 - y1) // 5))
            cv2.line(canvas, (x1, y1), (x1 + c_len, y1), (255, 255, 255), border_thickness)
            cv2.line(canvas, (x1, y1), (x1 + c_len, y1), (255, 255, 255), border_thickness)
            cv2.line(canvas, (x2, y2), (x2 - c_len, y2), (255, 255, 255), border_thickness)
            cv2.line(canvas, (x2, y2), (x2 - c_len, y2), (255, 255, 255), border_thickness)

            # Object Tag Banner above bounding box
            tag_text = f"{label_name} | {int(obj.confidence * 100)}% | ({x1},{y1}) -> ({x2},{y2})"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_tag, 1)

            if y1 - th - 12 >= 0:
                tag_y1 = y1 - th - 12
                tag_y2 = y1
            else:
                tag_y1 = y1
                tag_y2 = y1 + th + 12

            tag_x2 = min(w, x1 + tw + 14)
            cv2.rectangle(canvas, (x1, tag_y1), (tag_x2, tag_y2), (15, 18, 24), -1)
            cv2.rectangle(canvas, (x1, tag_y1), (tag_x2, tag_y2), color, 1)
            cv2.putText(
                canvas,
                tag_text,
                (x1 + 6, tag_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_tag,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Record factual reason for side panel
            reason_str = cls.get_factual_detection_reason(
                cname,
                instance_index=person_idx if cname == "person" else i,
                person_count=person_count,
            )
            obj_info_list.append({
                "idx": i,
                "inst_id": inst_id,
                "label": label_name,
                "class_title": obj.class_name.title(),
                "confidence": obj.confidence,
                "bbox_str": f"({x1},{y1}) -> ({x2},{y2})",
                "color": color,
                "reason": reason_str,
            })

        # 2. Top Canvas Banner (Header)
        top_banner_h = max(38, int(round(44 * scale_ref)))
        top_banner = np.full((top_banner_h, w, 3), 16, dtype=np.uint8)
        frame_tot_str = f"{total_video_frames:06d}" if total_video_frames > 0 else "N/A"
        header_text = (
            f"AI PROCTORING EVIDENCE | VIDEO: {video_filename} | "
            f"TIME: {ts_str} | FRAME: {frame_index:06d}/{frame_tot_str} ({video_pos_pct:.1f}%) | "
            f"PERSONS: {person_count}"
        )
        cv2.putText(
            top_banner,
            header_text,
            (12, int(top_banner_h * 0.65)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.40, scale_ref * 0.48),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.line(top_banner, (0, top_banner_h - 1), (w, top_banner_h - 1), (0, 210, 255), 1)
        frame_composite = np.vstack([top_banner, canvas])

        # 3. Dedicated Side Information Panel (Zero Occlusion)
        if is_portrait:
            panel_w = max(480, min(720, int(w * 0.52)))
        else:
            panel_w = max(420, min(680, int(w * 0.36)))

        panel_h = frame_composite.shape[0]
        panel = np.full((panel_h, panel_w, 3), 24, dtype=np.uint8)

        # Panel Header
        p_hdr_h = top_banner_h
        cv2.rectangle(panel, (0, 0), (panel_w, p_hdr_h), (32, 38, 48), -1)
        cv2.putText(
            panel,
            "AI PROCTORING EVIDENCE",
            (14, int(p_hdr_h * 0.65)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.45, scale_ref * 0.55),
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.line(panel, (0, p_hdr_h - 1), (panel_w, p_hdr_h - 1), (50, 65, 82), 1)

        y_cursor = p_hdr_h + int(20 * scale_ref)
        line_spacing = max(18, int(round(26 * scale_ref)))
        f_head = max(0.40, scale_ref * 0.46)
        f_body = max(0.36, scale_ref * 0.40)
        f_sub = max(0.32, scale_ref * 0.36)

        def draw_line(txt: str, color=(220, 220, 220), scale=f_body, bold=False, extra_space=0):
            nonlocal y_cursor
            if y_cursor < panel_h - 40:
                cv2.putText(
                    panel,
                    txt,
                    (14, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    color,
                    2 if bold else 1,
                    cv2.LINE_AA,
                )
                y_cursor += line_spacing + extra_space

        def draw_divider():
            nonlocal y_cursor
            if y_cursor < panel_h - 40:
                cv2.line(panel, (14, y_cursor), (panel_w - 14, y_cursor), (48, 58, 72), 1)
                y_cursor += int(14 * scale_ref)

        # A. Video Information Block
        draw_line(f"VIDEO: {video_filename}", (255, 255, 255), f_body, bold=True)
        draw_line(f"TIME: {ts_str}", (220, 220, 220), f_body)
        draw_line(f"FRAME: {frame_index:06d} / {frame_tot_str}", (220, 220, 220), f_body)
        draw_line(f"VIDEO POSITION: {video_pos_pct:.1f}%", (0, 215, 255), f_body, bold=True)
        draw_divider()

        # B. Detection Summary Block
        draw_line("DETECTION SUMMARY", (0, 215, 255), f_head, bold=True)
        p_color = (46, 204, 113) if person_count == 1 else (231, 76, 60)
        draw_line(f"Persons: {person_count}", p_color, f_body, bold=True)
        draw_line(f"Relevant Objects: {len(obj_info_list)}", (255, 255, 255), f_body)
        draw_divider()

        # C. Marked Objects Details Block
        draw_line("DETECTED OBJECTS", (0, 215, 255), f_head, bold=True)
        if not obj_info_list:
            draw_line("  (No relevant objects in this frame)", (140, 140, 140), f_sub)
        else:
            for item in obj_info_list[:5]:
                draw_line(
                    f"OBJECT #{item['idx']:02d}: {item['class_title']}",
                    item["color"],
                    f_body,
                    bold=True,
                )
                draw_line(f"  Confidence: {int(item['confidence'] * 100)}%", (200, 200, 200), f_sub)
                draw_line(f"  BBox: {item['bbox_str']}", (180, 180, 180), f_sub)
                draw_line(f"  WHY MARKED: {item['reason']}", (255, 255, 255), f_sub)
                y_cursor += int(4 * scale_ref)
        draw_divider()

        # D. Temporal Information Block (if applicable)
        if event_context and "event_id" in event_context:
            draw_line("TEMPORAL EVENT", (0, 215, 255), f_head, bold=True)
            draw_line(f"Event ID: {event_context.get('event_id', 'N/A')}", (255, 255, 255), f_sub)
            if "first_seen" in event_context:
                draw_line(f"First Seen: {event_context['first_seen']}", (200, 200, 200), f_sub)
            if "last_seen" in event_context:
                draw_line(f"Last Seen: {event_context['last_seen']}", (200, 200, 200), f_sub)
            if "duration_seconds" in event_context:
                draw_line(f"Duration: {event_context['duration_seconds']:.3f} sec", (200, 200, 200), f_sub)
            if "detection_count" in event_context:
                draw_line(f"Observations: {event_context['detection_count']}", (200, 200, 200), f_sub)
            if "max_confidence" in event_context:
                draw_line(f"Max Confidence: {int(event_context['max_confidence'] * 100)}%", (200, 200, 200), f_sub)
            draw_line("EVIDENCE TYPE: OBJECT DETECTION", (0, 215, 255), f_sub, bold=True)
            draw_divider()

        # E. Person Count Event Block (if applicable)
        if person_change_context or (event_context and event_context.get("event_type") == "person_count_change"):
            ctx = person_change_context or event_context or {}
            prev_c = ctx.get("previous_count", 0)
            new_c = ctx.get("new_count", person_count)
            diff = new_c - prev_c
            draw_line("PERSON COUNT EVENT", (231, 76, 60), f_head, bold=True)
            draw_line(f"Previous: {prev_c}", (200, 200, 200), f_sub)
            draw_line(f"Current: {new_c}", (255, 255, 255), f_sub, bold=True)
            draw_line(f"Change: {diff:+d}", (0, 215, 255), f_sub, bold=True)
            draw_line(f"WHY MARKED: Person count changed from {prev_c} to {new_c}", (255, 255, 255), f_sub)
            draw_divider()

        # F. Key-Frame Selection Reason Block
        draw_line("KEY FRAME REASON:", (0, 215, 255), f_head, bold=True)
        draw_line(f"• {selection_reason.upper()}", (255, 255, 255), f_body, bold=True)

        # 4. Panel Footer
        cv2.line(panel, (14, panel_h - 32), (panel_w - 14, panel_h - 32), (48, 58, 72), 1)
        cv2.putText(
            panel,
            "FACTUAL DETECTOR OBSERVATIONS ONLY",
            (14, panel_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.30, scale_ref * 0.34),
            (130, 130, 130),
            1,
            cv2.LINE_AA,
        )

        # Combine video frame and side explanation panel
        composite = np.hstack([frame_composite, panel])

        # 5. Bottom Universal Footer Banner across entire composite width
        tot_w = composite.shape[1]
        bot_banner_h = max(34, int(round(38 * scale_ref)))
        bot_banner = np.full((bot_banner_h, tot_w, 3), 18, dtype=np.uint8)
        cv2.line(bot_banner, (0, 0), (tot_w, 0), (45, 55, 68), 1)

        footer_msg = f"KEY FRAME: {selection_reason.upper()}  |  All information shown above is generated from factual detector observations."
        cv2.putText(
            bot_banner,
            footer_msg,
            (14, int(bot_banner_h * 0.65)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.36, scale_ref * 0.42),
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return np.vstack([composite, bot_banner])

    def build_package(
        self,
        report: VideoAnalysisReport,
        output_dir: Union[str, Path],
        frame_cache: Optional[Dict[int, np.ndarray]] = None,
        create_zip: bool = False,
    ) -> EvidencePackage:
        """Construct the deterministic evidence package directory, key frames, crops, and manifest.

        Args:
            report: VideoAnalysisReport containing timeline and temporal events.
            output_dir: Destination directory for the evidence package.
            frame_cache: Optional map of {frame_index: bgr_frame_array} for saving visual crops/frames.
            create_zip: If True, packages the entire directory into an accompanying .zip archive.

        Returns:
            EvidencePackage containing manifest and filesystem paths.
        """
        pkg_dir = Path(output_dir)
        pkg_dir.mkdir(parents=True, exist_ok=True)

        frames_dir = pkg_dir / "frames"
        crops_dir = pkg_dir / "crops"
        events_dir = pkg_dir / "events"
        timeline_dir = pkg_dir / "timeline"

        for d in (frames_dir, crops_dir, events_dir, timeline_dir):
            d.mkdir(parents=True, exist_ok=True)

        pkg_id = f"evidence_pkg_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        created_iso = datetime.now(timezone.utc).isoformat()
        video_name = Path(report.video_path).name
        total_frames = report.metadata.total_frames
        total_dur = report.metadata.duration_seconds

        # Map timeline entries by timestamp / frame_index
        timeline_by_idx: Dict[int, TimelineEntry] = {e.frame_index: e for e in report.timeline}
        timeline_by_time: Dict[float, TimelineEntry] = {round(e.timestamp_seconds, 3): e for e in report.timeline}

        # 1. Identify Key Frame Indices (Start, Peak, End, Person Transitions)
        key_frames_info: Dict[int, Tuple[str, TimelineEntry, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = {}

        # A. From Temporal Object Events
        for ev in report.temporal_events:
            t_start_round = round(ev.start_seconds, 3)
            t_end_round = round(ev.end_seconds, 3)
            ev_ctx = {
                "event_id": ev.event_id,
                "event_type": "temporal_object_presence",
                "object_class": ev.object_class,
                "reason": ev.reason if ev.reason else f"Detected object class: {ev.object_class.lower()}",
                "first_seen": ev.formatted_start,
                "last_seen": ev.formatted_end,
                "duration_seconds": ev.duration_seconds,
                "detection_count": ev.detection_count,
                "max_confidence": ev.max_confidence,
            }

            # Start entry
            start_entry = timeline_by_time.get(t_start_round)
            if start_entry and start_entry.frame_index not in key_frames_info:
                key_frames_info[start_entry.frame_index] = ("EVENT START", start_entry, ev_ctx, None)

            # Peak entry
            best_entry = None
            best_conf = -1.0
            for t_obs in ev.observation_timestamps:
                ent = timeline_by_time.get(round(t_obs, 3))
                if ent:
                    for obj in ent.relevant_objects:
                        if obj.class_name.lower() == ev.object_class.lower() and obj.confidence > best_conf:
                            best_conf = obj.confidence
                            best_entry = ent

            if best_entry and best_entry.frame_index not in key_frames_info:
                key_frames_info[best_entry.frame_index] = ("PEAK CONFIDENCE", best_entry, ev_ctx, None)

            # End entry
            end_entry = timeline_by_time.get(t_end_round)
            if end_entry and end_entry.frame_index not in key_frames_info:
                key_frames_info[end_entry.frame_index] = ("EVENT END", end_entry, ev_ctx, None)

        # B. From Person Count Changes
        for chg in report.person_count_changes:
            ent = timeline_by_idx.get(chg.frame_index)
            if ent and ent.frame_index not in key_frames_info:
                chg_ctx = {
                    "event_id": chg.event_id,
                    "event_type": "person_count_change",
                    "reason": chg.reason if chg.reason else f"Person count changed from {chg.previous_count} to {chg.new_count}",
                    "previous_count": chg.previous_count,
                    "new_count": chg.new_count,
                    "timestamp": chg.formatted_timestamp,
                }
                key_frames_info[ent.frame_index] = ("PERSON COUNT CHANGE", ent, None, chg_ctx)

        # Fallback: if no key frames selected (clean session), record initial frame
        if not key_frames_info and report.timeline:
            first_ent = report.timeline[0]
            key_frames_info[first_ent.frame_index] = ("SESSION START", first_ent, None, None)

        # 2. Process Key Frames & Generate Visual Assets
        evidence_frames_list: List[EvidenceFrame] = []
        frame_id_map: Dict[int, str] = {}
        crops_count = 0

        for f_idx in sorted(key_frames_info.keys()):
            reason, ent, ev_ctx, chg_ctx = key_frames_info[f_idx]
            ts_str_clean = ent.formatted_timestamp.replace(":", "_").replace(".", "_")
            frame_id = f"frame_{f_idx:06d}"
            frame_id_map[f_idx] = frame_id

            frame_filename = f"{frame_id}_{ts_str_clean}.jpg"
            frame_rel_path = f"frames/{frame_filename}"

            raw_frame = frame_cache.get(f_idx) if frame_cache else None
            w = ent.relevant_objects[0].x2 if (raw_frame is None and ent.relevant_objects) else (raw_frame.shape[1] if raw_frame is not None else report.metadata.width)
            h = ent.relevant_objects[0].y2 if (raw_frame is None and ent.relevant_objects) else (raw_frame.shape[0] if raw_frame is not None else report.metadata.height)

            evidence_objects: List[EvidenceObject] = []
            person_count_in_frame = ent.person_count
            person_seq = 0

            # Process Crops for relevant objects on this key frame
            for obj_idx, obj in enumerate(ent.relevant_objects, start=1):
                cname = obj.class_name.lower().strip()
                if cname == "person":
                    person_seq += 1
                    inst_id = f"person_{person_seq:02d}"
                else:
                    inst_id = f"{cname.replace(' ', '_')}_{obj_idx:02d}"

                obj_id = f"{frame_id}_{inst_id}"
                crop_filename = f"{frame_id}_{inst_id}.jpg"
                crop_rel_path = f"crops/{crop_filename}"

                factual_reason = self.get_factual_detection_reason(
                    class_name=cname,
                    instance_index=person_seq if cname == "person" else obj_idx,
                    person_count=person_count_in_frame,
                )

                if raw_frame is not None:
                    crop_img = self.crop_object(
                        image=raw_frame,
                        bbox=obj.bbox,
                        class_name=obj.class_name,
                        confidence=obj.confidence,
                        instance_id=inst_id,
                        frame_index=f_idx,
                        timestamp_seconds=ent.timestamp_seconds,
                        reason=factual_reason,
                        padding_ratio=self.crop_padding_ratio,
                        add_header=True,
                    )
                    cv2.imwrite(str(crops_dir / crop_filename), crop_img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    crops_count += 1

                evidence_objects.append(
                    EvidenceObject(
                        object_id=obj_id,
                        instance_id=inst_id,
                        class_name=obj.class_name,
                        confidence=obj.confidence,
                        bbox=obj.bbox,
                        reason=factual_reason,
                        timestamp_seconds=ent.timestamp_seconds,
                        frame_index=f_idx,
                        crop_relative_path=crop_rel_path if raw_frame is not None else None,
                    )
                )

            # Save annotated full-frame image with side explanation panel
            if raw_frame is not None:
                annotated_img = self.render_annotated_evidence_frame(
                    image=raw_frame,
                    frame_index=f_idx,
                    timestamp_seconds=ent.timestamp_seconds,
                    person_count=ent.person_count,
                    objects=ent.relevant_objects,
                    selection_reason=reason,
                    video_filename=video_name,
                    total_video_frames=total_frames,
                    total_video_duration_seconds=total_dur,
                    event_context=ev_ctx,
                    person_change_context=chg_ctx,
                )
                cv2.imwrite(str(frames_dir / frame_filename), annotated_img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

            video_pos_pct = (f_idx / total_frames) * 100.0 if total_frames > 0 else 0.0
            explanation_text = f"Key evidence frame captured at {ent.formatted_timestamp} (Frame {f_idx:05d} / {total_frames}) due to {reason.lower()}. Persons in scene: {ent.person_count}."

            evidence_frames_list.append(
                EvidenceFrame(
                    frame_id=frame_id,
                    frame_index=f_idx,
                    timestamp_seconds=ent.timestamp_seconds,
                    formatted_timestamp=ent.formatted_timestamp,
                    person_count=ent.person_count,
                    objects=evidence_objects,
                    frame_relative_path=frame_rel_path if raw_frame is not None else None,
                    width=w,
                    height=h,
                    selection_reason=reason,
                    explanation=explanation_text,
                    video_position_pct=video_pos_pct,
                    event_context=ev_ctx or chg_ctx,
                )
            )

        # 3. Build Evidence Events with Key Frame Cross-References
        evidence_events_list: List[EvidenceEvent] = []

        for ev in report.temporal_events:
            matched_frame_ids = []
            for f_idx in sorted(key_frames_info.keys()):
                _, ent, _, _ = key_frames_info[f_idx]
                if round(ev.start_seconds, 3) <= round(ent.timestamp_seconds, 3) <= round(ev.end_seconds, 3):
                    matched_frame_ids.append(frame_id_map[f_idx])

            ev_reason = ev.reason if ev.reason else f"Detected object class: {ev.object_class.lower()}"
            ev_item = EvidenceEvent(
                event_id=ev.event_id,
                modality="object",
                event_type="temporal_object_presence",
                object_class=ev.object_class,
                start_seconds=ev.start_seconds,
                end_seconds=ev.end_seconds,
                duration_seconds=ev.duration_seconds,
                formatted_start=ev.formatted_start,
                formatted_end=ev.formatted_end,
                detection_count=ev.detection_count,
                max_confidence=ev.max_confidence,
                average_confidence=ev.average_confidence,
                is_duration_qualified=ev.is_duration_qualified,
                reason=ev_reason,
                key_frame_ids=matched_frame_ids,
                metadata={
                    "representative_bbox": list(ev.representative_bbox) if ev.representative_bbox else None,
                    "observation_timestamps": [round(t, 3) for t in ev.observation_timestamps],
                },
            )
            evidence_events_list.append(ev_item)

            with open(events_dir / f"{ev.event_id}.json", "w", encoding="utf-8") as f:
                json.dump(ev_item.to_dict(), f, indent=2)

        for chg in report.person_count_changes:
            chg_frame_id = frame_id_map.get(chg.frame_index)
            chg_reason = chg.reason if chg.reason else f"Person count changed from {chg.previous_count} to {chg.new_count}"
            chg_item = EvidenceEvent(
                event_id=chg.event_id,
                modality="object",
                event_type="person_count_change",
                object_class="person",
                start_seconds=chg.timestamp_seconds,
                end_seconds=chg.timestamp_seconds,
                duration_seconds=0.0,
                formatted_start=chg.formatted_timestamp,
                formatted_end=chg.formatted_timestamp,
                detection_count=1,
                max_confidence=1.0,
                average_confidence=1.0,
                is_duration_qualified=True,
                reason=chg_reason,
                key_frame_ids=[chg_frame_id] if chg_frame_id else [],
                metadata={
                    "previous_count": chg.previous_count,
                    "new_count": chg.new_count,
                    "frame_index": chg.frame_index,
                },
            )
            evidence_events_list.append(chg_item)

            with open(events_dir / f"{chg.event_id}.json", "w", encoding="utf-8") as f:
                json.dump(chg_item.to_dict(), f, indent=2)

        # 4. Save Timeline Artifacts
        timeline_json_path = timeline_dir / "timeline.json"
        with open(timeline_json_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in report.timeline], f, indent=2)

        artifacts_map = {
            "timeline_data": "timeline/timeline.json",
        }

        # 5. Build and Save Evidence Manifest
        manifest = EvidenceManifest(
            schema_version="1.0",
            package_id=pkg_id,
            created_at=created_iso,
            source=EvidenceSource(
                type="video",
                filename=video_name,
                duration_seconds=report.metadata.duration_seconds,
                fps=report.metadata.source_fps,
                width=report.metadata.width,
                height=report.metadata.height,
                total_frames=report.metadata.total_frames,
            ),
            summary={
                "total_video_frames": report.metadata.total_frames,
                "sampled_frames_count": report.total_sampled_frames,
                "evidence_frames_count": len(evidence_frames_list),
                "object_crops_count": crops_count,
                "temporal_events_count": len(report.temporal_events),
                "person_count_changes_count": len(report.person_count_changes),
                "target_sampling_fps": report.target_sampling_fps,
                "total_processing_time_ms": report.total_processing_time_ms,
            },
            events=evidence_events_list,
            frames=evidence_frames_list,
            artifacts=artifacts_map,
        )

        manifest_path = pkg_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        # 6. Optionally Archive to ZIP with Relative Paths
        zip_file_path = None
        if create_zip:
            zip_file_path = pkg_dir.parent / f"{pkg_dir.name}.zip"
            with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in pkg_dir.rglob("*"):
                    if file_path.is_file():
                        rel_in_zip = file_path.relative_to(pkg_dir)
                        zf.write(file_path, arcname=str(rel_in_zip))

        return EvidencePackage(
            package_dir=pkg_dir,
            manifest=manifest,
            zip_path=zip_file_path,
        )
