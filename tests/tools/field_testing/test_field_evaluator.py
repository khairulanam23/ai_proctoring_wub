"""Tests for field benchmark evaluator, comparative metrics, and duration accuracy."""

from tools.field_testing.evaluator import FieldBenchmarkEvaluator
from tools.field_testing.schema import AnnotatorAgreementMetrics


def test_field_benchmark_evaluator_metrics():
    """Verify precision, recall, F1, and Controlled vs Field comparative metrics."""
    outcomes = [
        {
            "session_id": "s1",
            "frame_count": 10,
            "duration_seconds": 2.5,
            "expected_events": [],
            "observed_events": [],
            "expected_duration": 0.0,
            "observed_duration": 0.0,
        },
        {
            "session_id": "s2",
            "frame_count": 10,
            "duration_seconds": 2.5,
            "expected_events": ["NO_FACE"],
            "observed_events": ["NO_FACE"],
            "expected_duration": 2.0,
            "observed_duration": 2.0,
        },
        {
            "session_id": "s3",
            "frame_count": 10,
            "duration_seconds": 2.5,
            "expected_events": ["PHONE_DETECTED"],
            "observed_events": ["PHONE_DETECTED"],
            "expected_duration": 1.5,
            "observed_duration": 1.5,
        },
    ]

    agreement = AnnotatorAgreementMetrics(3, 3, 0, 100.0, 1.0)
    field_m = FieldBenchmarkEvaluator.evaluate_field_sessions(outcomes, agreement)

    assert field_m.total_sessions == 3
    assert field_m.tp == 2
    assert field_m.tn == 1
    assert field_m.fp == 0
    assert field_m.fn == 0
    assert field_m.precision == 1.0
    assert field_m.recall == 1.0
    assert field_m.f1_score == 1.0
    assert field_m.event_duration_mae_seconds == 0.0

    comparison = FieldBenchmarkEvaluator.compare_controlled_vs_field(
        field_m, latency_p50_field_ms=25.0
    )
    assert comparison.f1_diff == 0.0
    assert comparison.precision_field == 1.0
