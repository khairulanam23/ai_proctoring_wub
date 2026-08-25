"""Evaluation harness wrapping :class:`ProctoringEngine` with lifecycle bookkeeping.

The phase 9-11 evaluation suites measure how well the pipeline suppresses false
positives, which needs counters the live pipeline has no reason to carry: how many
raw detections were seen, how many became candidates, how many survived temporal
qualification, and how many were discarded as transient.

That accounting lives here rather than in the engine deliberately.  A production
session should not pay for statistics only an offline study reads, and the engine
stays a single implementation of the workflow rather than growing a second mode.
"""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from proctoring.config import SessionConfig
from proctoring.detection.face_detector import FaceDetector
from proctoring.detection.face_verifier import FaceVerifier
from proctoring.detection.object_detector import ObjectDetector
from proctoring.engine import ProctoringEngine, SessionSummary
from proctoring.evidence.quality import EvidenceQualityValidator
from proctoring.temporal.lifecycle import (
    CandidateEventRecord,
    CandidateObservation,
    EventLifecycleState,
    ValidationConfig,
)


@dataclass
class ValidationSessionStatistics:
    """False-positive filtering and latency metrics for one evaluated session."""

    total_frames_processed: int = 0
    raw_detections_count: int = 0
    candidate_events_created: int = 0
    validated_events_count: int = 0
    discarded_candidates_count: int = 0
    evidence_capture_failures: int = 0
    valid_evidence_count: int = 0
    mean_inference_latency_ms: float = 0.0
    mean_validation_latency_ms: float = 0.0
    mean_evidence_latency_ms: float = 0.0
    mean_total_latency_ms: float = 0.0
    effective_fps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_frames_processed": self.total_frames_processed,
            "raw_detections_count": self.raw_detections_count,
            "candidate_events_created": self.candidate_events_created,
            "validated_events_count": self.validated_events_count,
            "discarded_candidates_count": self.discarded_candidates_count,
            "evidence_capture_failures": self.evidence_capture_failures,
            "valid_evidence_count": self.valid_evidence_count,
            "mean_inference_latency_ms": round(self.mean_inference_latency_ms, 2),
            "mean_validation_latency_ms": round(self.mean_validation_latency_ms, 2),
            "mean_evidence_latency_ms": round(self.mean_evidence_latency_ms, 2),
            "mean_total_latency_ms": round(self.mean_total_latency_ms, 2),
            "effective_fps": round(self.effective_fps, 2),
        }


class ValidatedSessionHarness:
    """Runs a proctoring session while tracking the candidate-event lifecycle.

    Presents the same surface the phase 9-11 suites were written against:
    ``process_frame`` per frame, then ``finalize_session`` returning
    ``(summary, statistics)``.
    """

    def __init__(
        self,
        config: SessionConfig,
        validation_config: ValidationConfig | None = None,
        face_detector: FaceDetector | None = None,
        face_verifier: FaceVerifier | None = None,
        object_detector: ObjectDetector | None = None,
    ) -> None:
        self.config = config
        self.validation_config = validation_config or ValidationConfig()
        self.engine = ProctoringEngine(
            config=config,
            face_detector=face_detector,
            face_verifier=face_verifier,
            object_detector=object_detector,
        )
        self.engine.start_session()

        self.active_candidates: dict[str, CandidateEventRecord] = {}
        self.closed_candidates: list[CandidateEventRecord] = []
        self.discarded_candidates: list[CandidateEventRecord] = []

        self._candidate_seq = 0
        self._raw_detection_count = 0
        self._validation_latencies: list[float] = []
        self._evidence_latencies: list[float] = []
        self._valid_evidence_count = 0
        self._failed_evidence_count = 0

    # Kept so suites that reach through to the engine keep working.
    @property
    def temporal_aggregator(self):
        return self.engine.temporal_aggregator

    def process_frame(
        self,
        frame: np.ndarray | None,
        frame_index: int,
        timestamp_seconds: float,
    ) -> dict[str, Any]:
        """Run one frame through the engine, then advance the candidate state machine."""
        import time

        t_start = time.perf_counter()
        self.engine.process_frame(
            frame, frame_index=frame_index, timestamp_seconds=timestamp_seconds
        )
        inference_ms = (time.perf_counter() - t_start) * 1000.0

        t_val = time.perf_counter()
        for incident in self.engine.temporal_aggregator.active_incidents.values():
            # Only incidents that were touched by *this* frame advance their candidate.
            if abs(incident.last_seen_timestamp - timestamp_seconds) > 0.001:
                continue

            self._raw_detection_count += 1
            key = incident.event_type.value

            if key not in self.active_candidates:
                self._candidate_seq += 1
                self.active_candidates[key] = CandidateEventRecord(
                    candidate_id=f"cand_{self.config.session_id}_{self._candidate_seq:04d}",
                    session_id=self.config.session_id,
                    event_type=incident.event_type,
                    start_timestamp=incident.start_timestamp,
                    last_seen_timestamp=timestamp_seconds,
                )

            candidate = self.active_candidates[key]
            candidate.add_observation(
                CandidateObservation(
                    timestamp=timestamp_seconds,
                    frame_index=frame_index,
                    confidence=incident.best_confidence,
                    bbox=incident.representative_bbox,
                    description=f"{incident.event_type.value} observed at frame {frame_index}.",
                )
            )

            if (
                candidate.evaluate_state(timestamp_seconds, self.validation_config)
                == EventLifecycleState.VALIDATED
            ):
                t_ev = time.perf_counter()
                check = EvidenceQualityValidator.validate_frame(
                    image=frame, timestamp_seconds=timestamp_seconds, event_type=incident.event_type
                )
                self._evidence_latencies.append((time.perf_counter() - t_ev) * 1000.0)

                if check.is_valid:
                    candidate.state = EventLifecycleState.EVIDENCE_CAPTURED
                    candidate.evidence_references.append(f"sha256:{check.sha256_checksum}")
                    self._valid_evidence_count += 1
                else:
                    self._failed_evidence_count += 1
                    candidate.validation_reason += f" [Evidence Warning: {check.error_reason}]"

        # Retire candidates that closed or expired.
        for key in list(self.active_candidates):
            candidate = self.active_candidates[key]
            state = candidate.evaluate_state(timestamp_seconds, self.validation_config)
            if state == EventLifecycleState.CLOSED:
                self.closed_candidates.append(self.active_candidates.pop(key))
            elif state == EventLifecycleState.DISCARDED:
                self.discarded_candidates.append(self.active_candidates.pop(key))

        validation_ms = (time.perf_counter() - t_val) * 1000.0
        self._validation_latencies.append(validation_ms)

        return {
            "frame_index": frame_index,
            "timestamp_seconds": timestamp_seconds,
            "active_candidates": len(self.active_candidates),
            "inference_time_ms": inference_ms,
            "validation_time_ms": validation_ms,
        }

    def finalize_session(self) -> tuple[SessionSummary, ValidationSessionStatistics]:
        """Seal the evidence package and return it alongside lifecycle statistics."""
        for candidate in self.active_candidates.values():
            if candidate.state in (
                EventLifecycleState.VALIDATED,
                EventLifecycleState.EVIDENCE_CAPTURED,
            ):
                candidate.state = EventLifecycleState.CLOSED
                self.closed_candidates.append(candidate)
            else:
                candidate.state = EventLifecycleState.DISCARDED
                self.discarded_candidates.append(candidate)
        self.active_candidates.clear()

        summary = self.engine.finalize_session()

        validation_ms = (
            float(np.mean(self._validation_latencies)) if self._validation_latencies else 0.0
        )
        evidence_ms = float(np.mean(self._evidence_latencies)) if self._evidence_latencies else 0.0
        inference_ms = summary.telemetry.latency_overall.mean_ms

        statistics = ValidationSessionStatistics(
            total_frames_processed=summary.processed_frames,
            raw_detections_count=self._raw_detection_count,
            candidate_events_created=self._candidate_seq,
            validated_events_count=len(self.closed_candidates),
            discarded_candidates_count=len(self.discarded_candidates),
            evidence_capture_failures=self._failed_evidence_count,
            valid_evidence_count=self._valid_evidence_count,
            mean_inference_latency_ms=inference_ms,
            mean_validation_latency_ms=validation_ms,
            mean_evidence_latency_ms=evidence_ms,
            mean_total_latency_ms=inference_ms + validation_ms + evidence_ms,
            effective_fps=summary.telemetry.effective_fps,
        )

        with open(summary.package_dir / "validation_stats.json", "w", encoding="utf-8") as f:
            json.dump(statistics.to_dict(), f, indent=2)

        return summary, statistics
