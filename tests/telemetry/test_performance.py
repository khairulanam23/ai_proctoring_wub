"""Tests for PipelineTelemetryTracker, latency percentiles, and resource profiling."""

from proctoring.telemetry.performance import FrameTimingRecord, PipelineTelemetryTracker


def test_telemetry_latency_statistics():
    """Verify latency percentiles (P50, P90, P95, P99, Max) and stage timings."""
    tracker = PipelineTelemetryTracker(session_id="telemetry_test")

    # Simulate 20 frames with varying latencies
    for i in range(20):
        # Latencies from 10ms to 48ms
        lat = 10.0 + i * 2.0
        timing = FrameTimingRecord(
            frame_index=i + 1,
            timestamp_seconds=i * 0.25,
            capture_decode_ms=2.0,
            preprocessing_ms=1.0,
            face_detector_ms=lat * 0.5,
            object_detector_ms=lat * 0.4,
            temporal_postprocess_ms=0.5,
            evidence_io_ms=0.5,
            total_frame_ms=lat,
        )
        tracker.record_frame(
            timing, per_model_times={"face_detector": lat * 0.5, "object_detector": lat * 0.4}
        )

    report = tracker.generate_report()
    assert report.total_frames == 20
    assert report.processed_frames == 20
    assert report.skipped_frames == 0
    assert report.effective_fps > 0

    # Verify statistical calculations
    assert report.latency_overall.count == 20
    assert report.latency_overall.min_ms == 10.0
    assert report.latency_overall.max_ms == 48.0
    assert report.latency_overall.median_p50_ms == 29.0
    assert report.latency_overall.p95_ms > report.latency_overall.median_p50_ms

    # Verify stage breakdown
    assert "face_detector" in report.stage_latencies_mean
    assert "object_detector" in report.stage_latencies_mean
    assert report.stage_latencies_mean["face_detector"] > 0
