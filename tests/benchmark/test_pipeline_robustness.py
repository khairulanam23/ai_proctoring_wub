"""Unit tests for pipeline perturbation benchmarking."""

import numpy as np

from tools.benchmark.robustness import (
    PipelinePerturbationEvaluator,
    PipelineRobustnessSummary,
    VisualCondition,
)


def test_pipeline_perturbation_evaluator_low_light():
    """Verify PipelinePerturbationEvaluator runs end-to-end on synthetic frames under low light."""
    frames = [np.random.randint(60, 200, (480, 640, 3), dtype=np.uint8) for _ in range(5)]

    summary = PipelinePerturbationEvaluator.evaluate_stream(
        frames=frames,
        condition=VisualCondition.LOW_LIGHT,
        intensity=0.8,
        enable_objects=False,
    )

    assert isinstance(summary, PipelineRobustnessSummary)
    assert summary.condition == VisualCondition.LOW_LIGHT
    assert summary.total_frames == 5
    assert summary.accepted_frames == 5
    assert summary.effective_fps > 0.0
