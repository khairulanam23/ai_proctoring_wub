"""Evidence capture, ROI cropping, deterministic storage, and rigorous evidence validation."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from proctoring.core.events import (
    EventRecord,
    EventStatus,
    EvidenceReference,
    format_seconds_to_timestamp,
)
from proctoring.core.paths import sanitise_identifier

LOGGER = logging.getLogger(__name__)


@dataclass
class EvidenceValidationResult:
    """Outcome of validating a saved evidence asset on disk."""

    is_valid: bool
    evidence_id: str
    file_path: str
    file_exists: bool
    can_open: bool
    image_shape: tuple[int, int, int] | None = None
    file_size_bytes: int = 0
    calculated_sha256: str | None = None
    expected_sha256: str | None = None
    checksum_match: bool = False
    bbox_valid: bool = True
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "evidence_id": self.evidence_id,
            "file_path": self.file_path,
            "file_exists": self.file_exists,
            "can_open": self.can_open,
            "image_shape": list(self.image_shape) if self.image_shape else None,
            "file_size_bytes": self.file_size_bytes,
            "calculated_sha256": self.calculated_sha256,
            "expected_sha256": self.expected_sha256,
            "checksum_match": self.checksum_match,
            "bbox_valid": self.bbox_valid,
            "error_message": self.error_message,
        }


class EvidenceManager:
    """Manages visual evidence capture, object/face cropping, and integrity validation."""

    def __init__(
        self,
        base_dir: str | Path = "data/evidence_packages",
        session_id: str = "default_session",
        crop_padding_ratio: float = 0.10,
        jpeg_quality: int = 95,
    ) -> None:
        self.base_dir = Path(base_dir)
        # Session identifiers arrive from a host LMS and are used as a path segment,
        # so they are sanitised here rather than trusted.
        self.session_id = sanitise_identifier(session_id, fallback="session")
        self.crop_padding_ratio = max(0.0, float(crop_padding_ratio))
        self.jpeg_quality = int(jpeg_quality)

        # Directory layout
        self.package_dir = self.base_dir / self.session_id
        self.frames_dir = self.package_dir / "evidence" / "frames"
        self.crops_dir = self.package_dir / "evidence" / "crops"
        self.metadata_dir = self.package_dir / "evidence" / "metadata"
        # Derived, marked-up copies for proctor review. Kept apart from frames/
        # so source evidence and annotated evidence are never confused.
        self.review_dir = self.package_dir / "evidence" / "review"

        self._ensure_directories()
        self._saved_frame_paths: dict[int, str] = {}  # frame_index -> relative path
        self._evidence_counter = 0

    def _ensure_directories(self) -> None:
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.review_dir.mkdir(parents=True, exist_ok=True)

    def _compute_sha256(self, file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def capture_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_seconds: float,
    ) -> EvidenceReference | None:
        """Save a full-resolution key frame to the evidence package."""
        if frame is None or frame.size == 0:
            return None

        # Check if this frame was already saved to avoid duplicate writes
        if frame_index in self._saved_frame_paths:
            rel_path = self._saved_frame_paths[frame_index]
            abs_path = self.package_dir / rel_path
            if abs_path.exists():
                sha = self._compute_sha256(abs_path)
                return EvidenceReference(
                    evidence_id=f"ev_frm_{frame_index:06d}",
                    media_type="frame",
                    file_path=rel_path,
                    timestamp_seconds=timestamp_seconds,
                    formatted_timestamp=format_seconds_to_timestamp(timestamp_seconds),
                    frame_index=frame_index,
                    sha256=sha,
                    file_size_bytes=abs_path.stat().st_size,
                    is_validated=True,
                )

        try:
            self._ensure_directories()
            ts_ms = int(timestamp_seconds * 1000)
            filename = f"frame_{frame_index:06d}_{ts_ms:08d}ms.jpg"
            abs_path = self.frames_dir / filename
            rel_path = f"evidence/frames/{filename}"

            success = cv2.imwrite(
                str(abs_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not success or not abs_path.exists():
                return None

            sha = self._compute_sha256(abs_path)
            size = abs_path.stat().st_size
            self._saved_frame_paths[frame_index] = rel_path
            self._evidence_counter += 1

            return EvidenceReference(
                evidence_id=f"ev_frm_{frame_index:06d}",
                media_type="frame",
                file_path=rel_path,
                timestamp_seconds=timestamp_seconds,
                formatted_timestamp=format_seconds_to_timestamp(timestamp_seconds),
                frame_index=frame_index,
                sha256=sha,
                file_size_bytes=size,
                is_validated=True,
            )
        except Exception as exc:
            LOGGER.warning("Evidence frame write failed at frame %s: %s", frame_index, exc)
            return None

    def crop_region_of_interest(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],  # (x1, y1, x2, y2)
        event_id: str,
        frame_index: int,
        timestamp_seconds: float,
        label: str = "roi",
    ) -> EvidenceReference | None:
        """Crop and save a padded bounding box region of interest (e.g. face or object)."""
        if frame is None or frame.size == 0 or bbox is None:
            return None

        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            if bw <= 0 or bh <= 0:
                return None

            # Apply safety padding
            pad_x = int(bw * self.crop_padding_ratio)
            pad_y = int(bh * self.crop_padding_ratio)

            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                return None

            self._ensure_directories()
            clean_label = label.replace(" ", "_").lower()
            filename = f"crop_{event_id}_{clean_label}_{frame_index:06d}.jpg"
            abs_path = self.crops_dir / filename
            rel_path = f"evidence/crops/{filename}"

            success = cv2.imwrite(
                str(abs_path),
                crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not success or not abs_path.exists():
                return None

            sha = self._compute_sha256(abs_path)
            size = abs_path.stat().st_size
            self._evidence_counter += 1

            return EvidenceReference(
                evidence_id=f"ev_crp_{self._evidence_counter:04d}_{clean_label}",
                media_type="crop",
                file_path=rel_path,
                timestamp_seconds=timestamp_seconds,
                formatted_timestamp=format_seconds_to_timestamp(timestamp_seconds),
                frame_index=frame_index,
                bbox=bbox,
                sha256=sha,
                file_size_bytes=size,
                is_validated=True,
            )
        except Exception as exc:
            LOGGER.warning("Evidence crop write failed for %s: %s", event_id, exc)
            return None

    def save_review_snapshot(
        self,
        annotated_frame: np.ndarray,
        event_id: str,
        frame_index: int,
        timestamp_seconds: float,
    ) -> EvidenceReference | None:
        """Store an annotated copy of an evidence frame for proctor review.

        The returned reference is tagged ``media_type="review"`` and marked derived
        in its metadata. It is checksummed like any other artefact so the review
        image itself cannot be swapped, but it must never be mistaken for the source
        capture: annotations are the system's interpretation, not the record.
        """
        if annotated_frame is None or annotated_frame.size == 0:
            return None
        try:
            self._ensure_directories()
            filename = f"review_{event_id}_{frame_index:06d}.jpg"
            abs_path = self.review_dir / filename
            rel_path = f"evidence/review/{filename}"

            ok = cv2.imwrite(
                str(abs_path), annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if not ok or not abs_path.exists():
                return None

            self._evidence_counter += 1
            return EvidenceReference(
                evidence_id=f"ev_rev_{self._evidence_counter:04d}",
                media_type="review",
                file_path=rel_path,
                timestamp_seconds=timestamp_seconds,
                formatted_timestamp=format_seconds_to_timestamp(timestamp_seconds),
                frame_index=frame_index,
                sha256=self._compute_sha256(abs_path),
                file_size_bytes=abs_path.stat().st_size,
                is_validated=True,
            )
        except Exception as exc:
            LOGGER.warning("Review snapshot write failed for %s: %s", event_id, exc)
            return None

    def attach_evidence_to_event(
        self,
        event: EventRecord,
        frame: np.ndarray | None,
        frame_index: int,
        timestamp_seconds: float,
        bbox: tuple[int, int, int, int] | None = None,
        label: str = "evidence",
    ) -> bool:
        """Capture visual evidence frame and crop, attaching them to the target event."""
        if frame is None or frame.size == 0:
            event.status = EventStatus.EVIDENCE_FAILED
            event.metadata["evidence_failed"] = True
            event.metadata["evidence_error"] = "Input frame was empty or None"
            return False

        try:
            # 1. Capture full key frame
            ev_frame = self.capture_frame(frame, frame_index, timestamp_seconds)
            if ev_frame is not None:
                event.evidence.append(ev_frame)

            # 2. Crop target bbox if present
            if bbox is not None:
                ev_crop = self.crop_region_of_interest(
                    frame=frame,
                    bbox=bbox,
                    event_id=event.event_id,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    label=label,
                )
                if ev_crop is not None:
                    event.evidence.append(ev_crop)

            if len(event.evidence) > 0:
                event.status = EventStatus.EVIDENCE_CAPTURED
                return True
            event.status = EventStatus.EVIDENCE_FAILED
            event.metadata["evidence_failed"] = True
            event.metadata["evidence_error"] = "Disk write failed for both frame and crop"
            return False
        except Exception as e:
            event.status = EventStatus.EVIDENCE_FAILED
            event.metadata["evidence_failed"] = True
            event.metadata["evidence_error"] = f"Exception during evidence capture: {str(e)}"
            return False

    def validate_evidence(self, ev_ref: EvidenceReference) -> EvidenceValidationResult:
        """Thoroughly validate a saved evidence file on disk."""
        abs_path = self.package_dir / ev_ref.file_path
        if not abs_path.exists():
            return EvidenceValidationResult(
                is_valid=False,
                evidence_id=ev_ref.evidence_id,
                file_path=ev_ref.file_path,
                file_exists=False,
                can_open=False,
                error_message=f"File not found on disk: {abs_path}",
            )

        file_size = abs_path.stat().st_size
        if file_size == 0:
            return EvidenceValidationResult(
                is_valid=False,
                evidence_id=ev_ref.evidence_id,
                file_path=ev_ref.file_path,
                file_exists=True,
                can_open=False,
                file_size_bytes=0,
                error_message="Evidence file exists but is 0 bytes (empty)",
            )

        img = cv2.imread(str(abs_path))
        if img is None or img.size == 0:
            return EvidenceValidationResult(
                is_valid=False,
                evidence_id=ev_ref.evidence_id,
                file_path=ev_ref.file_path,
                file_exists=True,
                can_open=False,
                file_size_bytes=file_size,
                error_message="Corrupted image: cv2.imread failed to decode",
            )

        calc_sha = self._compute_sha256(abs_path)
        checksum_match = (calc_sha == ev_ref.sha256) if ev_ref.sha256 else True

        # Check bbox validity if present
        bbox_valid = True
        if ev_ref.bbox is not None:
            x1, y1, x2, y2 = ev_ref.bbox
            if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
                bbox_valid = False

        is_valid = checksum_match and bbox_valid
        err_msg = None if is_valid else "Checksum mismatch or invalid bounding box"

        return EvidenceValidationResult(
            is_valid=is_valid,
            evidence_id=ev_ref.evidence_id,
            file_path=ev_ref.file_path,
            file_exists=True,
            can_open=True,
            image_shape=img.shape,
            file_size_bytes=file_size,
            calculated_sha256=calc_sha,
            expected_sha256=ev_ref.sha256,
            checksum_match=checksum_match,
            bbox_valid=bbox_valid,
            error_message=err_msg,
        )

    def validate_all_event_evidence(
        self,
        events: list[EventRecord],
        prune_invalid: bool = True,
    ) -> dict[str, Any]:
        """Validate every evidence reference and, optionally, detach the bad ones.

        This is workflow stage 9.  Each reference is re-read from disk, decoded and
        re-hashed, so a file that was truncated, overwritten or lost after capture is
        caught here rather than by the proctor who opens the package.

        With ``prune_invalid`` set, references that fail validation are removed from
        the event.  The workflow calls for invalid or incomplete data to be removed,
        and a dangling reference is worse than no reference: it invites a proctor to
        conclude evidence was withheld.  Nothing is deleted silently — every pruned
        reference is listed under the event's ``discarded_evidence`` metadata and in
        the returned report, so the removal itself remains auditable.

        An event whose evidence is entirely pruned falls back to
        ``EVIDENCE_FAILED``: the observation still stands on the temporal record,
        but the package no longer claims to hold a picture of it.
        """
        total_references = 0
        valid_references = 0
        failed_references = 0
        pruned_references = 0
        validation_reports: list[dict[str, Any]] = []

        for ev in events:
            surviving: list[EvidenceReference] = []
            discarded: list[dict[str, Any]] = []

            for ref in ev.evidence:
                total_references += 1
                res = self.validate_evidence(ref)
                ref.is_validated = res.is_valid
                ref.validation_error = res.error_message
                validation_reports.append(res.to_dict())

                if res.is_valid:
                    valid_references += 1
                    surviving.append(ref)
                    continue

                failed_references += 1
                if prune_invalid:
                    pruned_references += 1
                    discarded.append(
                        {
                            "evidence_id": ref.evidence_id,
                            "file_path": ref.file_path,
                            "reason": res.error_message,
                        }
                    )
                else:
                    surviving.append(ref)

            ev.evidence = surviving
            if discarded:
                ev.metadata["discarded_evidence"] = discarded

            # Promote to VALIDATED only when the event still cites at least one
            # asset and every remaining asset verified cleanly.
            if ev.status == EventStatus.EVIDENCE_CAPTURED:
                if ev.evidence and all(r.is_validated for r in ev.evidence):
                    ev.status = EventStatus.VALIDATED
                elif not ev.evidence:
                    ev.status = EventStatus.EVIDENCE_FAILED
                    ev.metadata["evidence_error"] = (
                        "All captured evidence failed on-disk validation"
                    )

        return {
            "total_references": total_references,
            "valid_references": valid_references,
            "failed_references": failed_references,
            "pruned_references": pruned_references,
            "all_passed": failed_references == 0,
            "reports": validation_reports,
        }
