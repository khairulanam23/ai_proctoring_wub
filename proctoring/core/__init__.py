"""Shared vocabulary of the proctoring workflow: event schema and error handling."""

from proctoring.core.errors import (
    DiagnosticRecord,
    ErrorCategory,
    ObservationType,
    PipelineErrorHandler,
    PipelineException,
)
from proctoring.core.events import (
    DetectorInfo,
    EventRecord,
    EventSeverity,
    EventStatus,
    EventType,
    EvidenceReference,
    ObservationDetail,
    coerce_event_type,
    coerce_severity,
    format_seconds_to_timestamp,
)

__all__ = [
    "DiagnosticRecord",
    "ErrorCategory",
    "ObservationType",
    "PipelineErrorHandler",
    "PipelineException",
    "DetectorInfo",
    "EventRecord",
    "EventSeverity",
    "EventStatus",
    "EventType",
    "EvidenceReference",
    "ObservationDetail",
    "coerce_event_type",
    "coerce_severity",
    "format_seconds_to_timestamp",
]
