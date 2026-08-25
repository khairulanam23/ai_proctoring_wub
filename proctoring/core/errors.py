"""Robust error boundaries, failure diagnosis, and distinction between detection results and system errors."""

import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    """Categorical classification of failures."""

    CAMERA_UNAVAILABLE = "CAMERA_UNAVAILABLE"
    CORRUPT_FRAME = "CORRUPT_FRAME"
    MODEL_INFERENCE_FAILURE = "MODEL_INFERENCE_FAILURE"
    MISSING_REFERENCE_TEMPLATE = "MISSING_REFERENCE_TEMPLATE"
    EVIDENCE_WRITE_FAILURE = "EVIDENCE_WRITE_FAILURE"
    DETECTOR_INITIALIZATION_FAILURE = "DETECTOR_INITIALIZATION_FAILURE"
    UNEXPECTED_SYSTEM_EXCEPTION = "UNEXPECTED_SYSTEM_EXCEPTION"


class ObservationType(str, Enum):
    """Explicit distinction between system errors and AI detection outcomes."""

    DETECTION_RESULT = "DETECTION_RESULT"  # Normal AI observation (e.g. NO_FACE, PHONE_DETECTED)
    SYSTEM_ERROR = "SYSTEM_ERROR"  # Technical/hardware/IO failure
    SUSPICIOUS_EVENT = "SUSPICIOUS_EVENT"  # Qualified proctoring event for human review


class PipelineException(Exception):
    """Base exception for AI proctoring pipeline failures."""

    def __init__(
        self, category: ErrorCategory, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.category = category
        self.message = message
        self.details = details or {}
        self.timestamp_utc = datetime.now(timezone.utc).isoformat()


@dataclass
class DiagnosticRecord:
    """Diagnostic telemetry recording a handled system exception."""

    category: ErrorCategory
    message: str
    timestamp_seconds: float
    frame_index: int | None = None
    stack_trace: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    is_fatal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "message": self.message,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "frame_index": self.frame_index,
            "stack_trace": self.stack_trace,
            "details": self.details,
            "is_fatal": self.is_fatal,
        }


class PipelineErrorHandler:
    """Centralized failure handler ensuring pipeline gracefully degrades on faults."""

    def __init__(self, session_id: str = "default_session") -> None:
        self.session_id = session_id
        self.diagnostics: list[DiagnosticRecord] = []

    def handle_exception(
        self,
        error: Exception,
        category: ErrorCategory,
        timestamp_seconds: float,
        frame_index: int | None = None,
        details: dict[str, Any] | None = None,
        is_fatal: bool = False,
    ) -> DiagnosticRecord:
        """Record and format a system exception without crashing the pipeline."""
        tb = traceback.format_exc()
        record = DiagnosticRecord(
            category=category,
            message=str(error),
            timestamp_seconds=timestamp_seconds,
            frame_index=frame_index,
            stack_trace=tb,
            details=details or {},
            is_fatal=is_fatal,
        )
        self.diagnostics.append(record)
        return record

    def validate_frame(
        self,
        frame: Any,
        timestamp_seconds: float,
        frame_index: int,
    ) -> tuple[bool, str | None]:
        """Validate candidate video frame integrity."""
        if frame is None:
            self.handle_exception(
                error=ValueError("Received None frame buffer"),
                category=ErrorCategory.CORRUPT_FRAME,
                timestamp_seconds=timestamp_seconds,
                frame_index=frame_index,
            )
            return False, "Frame buffer is None"

        if not hasattr(frame, "shape") or not hasattr(frame, "size"):
            self.handle_exception(
                error=TypeError(f"Frame is not a numpy array: {type(frame)}"),
                category=ErrorCategory.CORRUPT_FRAME,
                timestamp_seconds=timestamp_seconds,
                frame_index=frame_index,
            )
            return False, "Frame is not a valid numpy array"

        if frame.size == 0:
            self.handle_exception(
                error=ValueError("Frame has zero size"),
                category=ErrorCategory.CORRUPT_FRAME,
                timestamp_seconds=timestamp_seconds,
                frame_index=frame_index,
            )
            return False, "Frame has 0 pixels"

        if len(frame.shape) < 2 or frame.shape[0] < 10 or frame.shape[1] < 10:
            self.handle_exception(
                error=ValueError(f"Frame dimensions too small: {frame.shape}"),
                category=ErrorCategory.CORRUPT_FRAME,
                timestamp_seconds=timestamp_seconds,
                frame_index=frame_index,
            )
            return False, f"Frame dimensions too small: {frame.shape}"

        return True, None

    def get_summary(self) -> dict[str, Any]:
        """Summarize all diagnostics recorded during session."""
        counts: dict[str, int] = {}
        for d in self.diagnostics:
            counts[d.category.value] = counts.get(d.category.value, 0) + 1

        return {
            "total_errors": len(self.diagnostics),
            "fatal_errors": sum(1 for d in self.diagnostics if d.is_fatal),
            "error_counts_by_category": counts,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }
