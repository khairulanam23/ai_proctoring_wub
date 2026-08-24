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

from src.object_detection.detector import DetectedObject
from src.object_detection.sampling import VideoMetadata, format_timestamp
from src.object_detection.temporal import ObjectPresenceEvent, PersonCountChangeEvent, TemporalEventReport
from src.object_detection.video import TimelineEntry, VideoAnalysisReport


@dataclass
class EvidenceObject:
    """Individual object detected on a key evidence frame."""
    object_id: str
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    crop_relative_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "crop_relative_path": self.crop_relative_path,
        }


@dataclass
class EvidenceFrame:
    """Key evidence visual frame selected from temporal transitions or peaks."""
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
            "objects": [obj.to_dict() for obj in self.objects],
        }


@dataclass
class EvidenceEvent:
    """Consolidated evidence event for human proctor review."""
    event_id: str
    modality: str  # "object", "face", "browser", "audio", "system"
    event_type: str  # "object_presence", "person_count_change", "face_observation"
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
    key_frame_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "modality": self.modality,
            "event_type": self.event_type,
            "object_class": self.object_class,
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
            key_frame_ids=key_frame_ids or [],
            metadata={
                "presence_status": status,
                "face_count": face_count,
                "frame_indices": frame_indices,
            },
        )


class EvidenceBuilder:
    """Builder responsible for key-frame selection, object cropping, and evidence packaging."""

    def __init__(
        self,
        crop_padding_ratio: float = 0.10,
        jpeg_quality: int = 95,
    ) -> None:
        self.crop_padding_ratio = float(crop_padding_ratio)
        self.jpeg_quality = int(jpeg_quality)

    @staticmethod
    def crop_object(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding_ratio: float = 0.10,
    ) -> np.ndarray:
        """Safely crop object bounding box with bounds clamping and optional padding.

        Args:
            image: Source BGR image numpy array.
            bbox: (x1, y1, x2, y2) in pixel coordinates.
            padding_ratio: Fractional padding around object (e.g. 0.10 = 10%).

        Returns:
            Cropped BGR numpy image array.
        """
        if image is None or image.size == 0:
            raise ValueError("Cannot crop from empty or invalid image array.")

        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # Calculate padding
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad_x = int(box_w * padding_ratio)
        pad_y = int(box_h * padding_ratio)

        # Clamped coordinates with padding
        crop_x1 = max(0, min(w - 1, x1 - pad_x))
        crop_y1 = max(0, min(h - 1, y1 - pad_y))
        crop_x2 = max(crop_x1 + 1, min(w, x2 + pad_x))
        crop_y2 = max(crop_y1 + 1, min(h, y2 + pad_y))

        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            # Fallback minimum valid slice
            crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        return crop.copy()

    @staticmethod
    def render_annotated_evidence_frame(
        image: np.ndarray,
        frame_index: int,
        timestamp_seconds: float,
        person_count: int,
        objects: List[DetectedObject],
        selection_reason: str,
    ) -> np.ndarray:
        """Render a clean, professional human-proctor readable visual evidence frame."""
        vis = image.copy()
        h, w = vis.shape[:2]

        palette = {
            "person": (46, 204, 113),      # Green
            "cell phone": (230, 126, 34),  # Orange
            "laptop": (52, 152, 219),      # Blue
            "book": (155, 89, 182),        # Purple
            "tablet": (26, 188, 156),      # Teal
            "keyboard": (241, 196, 15),    # Yellow
            "mouse": (231, 76, 60),        # Red
            "backpack": (52, 73, 94),      # Navy
            "bottle": (22, 160, 133),      # Emerald
        }

        # 1. Draw high-contrast bounding boxes with labels
        for obj in objects:
            c = palette.get(obj.class_name.lower(), (0, 215, 255))
            x1, y1, x2, y2 = obj.bbox

            # Draw rectangle
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2, cv2.LINE_AA)

            # Draw label banner
            label = f"{obj.class_name.title()} {int(obj.confidence * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
            banner_y1 = max(0, y1 - th - 8)
            cv2.rectangle(vis, (x1, banner_y1), (min(w, x1 + tw + 8), y1), c, -1)
            cv2.putText(vis, label, (x1 + 4, y1 - 4), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        # 2. Add Top Header Information Banner
        banner_h = 36
        banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
        ts_str = format_timestamp(timestamp_seconds)
        header_text = f"TIME: {ts_str} | FRAME: {frame_index:05d} | PERSONS: {person_count} | KEY FRAME: {selection_reason.upper()}"
        cv2.putText(banner, header_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        return np.vstack([banner, vis])

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

        # Map timeline entries by timestamp / frame_index
        timeline_by_idx: Dict[int, TimelineEntry] = {e.frame_index: e for e in report.timeline}
        timeline_by_time: Dict[float, TimelineEntry] = {round(e.timestamp_seconds, 3): e for e in report.timeline}

        # 1. Identify Key Frame Indices (Start, Peak, End, Person Transitions)
        key_frames_info: Dict[int, Tuple[str, TimelineEntry]] = {}  # {frame_idx: (reason, timeline_entry)}

        # A. From Temporal Object Events
        for ev in report.temporal_events:
            # Find start entry
            t_start_round = round(ev.start_seconds, 3)
            t_end_round = round(ev.end_seconds, 3)

            start_entry = timeline_by_time.get(t_start_round)
            if start_entry and start_entry.frame_index not in key_frames_info:
                key_frames_info[start_entry.frame_index] = (f"event_start ({ev.object_class})", start_entry)

            # Find peak entry (frame with max confidence for this class)
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
                key_frames_info[best_entry.frame_index] = (f"event_peak ({ev.object_class})", best_entry)

            # Find end entry
            end_entry = timeline_by_time.get(t_end_round)
            if end_entry and end_entry.frame_index not in key_frames_info:
                key_frames_info[end_entry.frame_index] = (f"event_end ({ev.object_class})", end_entry)

        # B. From Person Count Changes
        for chg in report.person_count_changes:
            ent = timeline_by_idx.get(chg.frame_index)
            if ent and ent.frame_index not in key_frames_info:
                key_frames_info[ent.frame_index] = (f"person_count_change ({chg.previous_count}->{chg.new_count})", ent)

        # Fallback: if no key frames selected (e.g. clean session), record initial frame
        if not key_frames_info and report.timeline:
            first_ent = report.timeline[0]
            key_frames_info[first_ent.frame_index] = ("session_initial_frame", first_ent)

        # 2. Process Key Frames & Generate Visual Assets
        evidence_frames_list: List[EvidenceFrame] = []
        frame_id_map: Dict[int, str] = {}  # {frame_index: frame_id}
        crops_count = 0

        for f_idx in sorted(key_frames_info.keys()):
            reason, ent = key_frames_info[f_idx]
            ts_str_clean = ent.formatted_timestamp.replace(":", "_").replace(".", "_")
            frame_id = f"frame_{f_idx:06d}"
            frame_id_map[f_idx] = frame_id

            frame_filename = f"{frame_id}_{ts_str_clean}.jpg"
            frame_rel_path = f"frames/{frame_filename}"

            raw_frame = frame_cache.get(f_idx) if frame_cache else None
            w = ent.relevant_objects[0].x2 if (raw_frame is None and ent.relevant_objects) else (raw_frame.shape[1] if raw_frame is not None else report.metadata.width)
            h = ent.relevant_objects[0].y2 if (raw_frame is None and ent.relevant_objects) else (raw_frame.shape[0] if raw_frame is not None else report.metadata.height)

            evidence_objects: List[EvidenceObject] = []

            # Save annotated frame image if raw pixels available
            if raw_frame is not None:
                annotated_img = self.render_annotated_evidence_frame(
                    image=raw_frame,
                    frame_index=f_idx,
                    timestamp_seconds=ent.timestamp_seconds,
                    person_count=ent.person_count,
                    objects=ent.relevant_objects,
                    selection_reason=reason,
                )
                cv2.imwrite(str(frames_dir / frame_filename), annotated_img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

            # Process Crops for relevant objects on this key frame
            for obj_idx, obj in enumerate(ent.relevant_objects, start=1):
                obj_id = f"{frame_id}_obj_{obj_idx:02d}"
                cname_clean = obj.class_name.lower().replace(" ", "_")
                crop_filename = f"{frame_id}_{cname_clean}_{obj_idx:02d}.jpg"
                crop_rel_path = f"crops/{crop_filename}"

                if raw_frame is not None:
                    crop_img = self.crop_object(raw_frame, obj.bbox, padding_ratio=self.crop_padding_ratio)
                    cv2.imwrite(str(crops_dir / crop_filename), crop_img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    crops_count += 1

                evidence_objects.append(
                    EvidenceObject(
                        object_id=obj_id,
                        class_name=obj.class_name,
                        confidence=obj.confidence,
                        bbox=obj.bbox,
                        crop_relative_path=crop_rel_path if raw_frame is not None else None,
                    )
                )

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
                )
            )

        # 3. Build Evidence Events with Key Frame Cross-References
        evidence_events_list: List[EvidenceEvent] = []

        for ev in report.temporal_events:
            # Map associated key frames
            matched_frame_ids = []
            for f_idx in sorted(key_frames_info.keys()):
                _, ent = key_frames_info[f_idx]
                if round(ev.start_seconds, 3) <= round(ent.timestamp_seconds, 3) <= round(ev.end_seconds, 3):
                    matched_frame_ids.append(frame_id_map[f_idx])

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
                key_frame_ids=matched_frame_ids,
                metadata={
                    "representative_bbox": list(ev.representative_bbox) if ev.representative_bbox else None,
                    "observation_timestamps": [round(t, 3) for t in ev.observation_timestamps],
                },
            )
            evidence_events_list.append(ev_item)

            # Write individual event JSON
            with open(events_dir / f"{ev.event_id}.json", "w", encoding="utf-8") as f:
                json.dump(ev_item.to_dict(), f, indent=2)

        for chg in report.person_count_changes:
            chg_frame_id = frame_id_map.get(chg.frame_index)
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
                key_frame_ids=[chg_frame_id] if chg_frame_id else [],
                metadata={
                    "previous_count": chg.previous_count,
                    "new_count": chg.new_count,
                    "frame_index": chg.frame_index,
                },
            )
            evidence_events_list.append(chg_item)

            # Write individual event JSON
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
                filename=Path(report.video_path).name,
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
