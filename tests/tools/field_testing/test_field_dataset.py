"""Tests for field dataset builder and scenario generation."""

from tools.field_testing.dataset_generator import FieldTestDatasetBuilder
from tools.field_testing.schema import FieldScenarioType


def test_field_dataset_builder_scenarios():
    """Verify field test dataset builds realistic sessions across Scenarios A through F."""
    dataset = FieldTestDatasetBuilder.build_field_dataset(
        samples_dir="data/samples", sampling_fps=4.0
    )
    assert len(dataset.sessions) >= 8
    assert dataset.dataset_name == "Phase6_Real_World_Field_Test_Dataset"

    summary = dataset.summary()
    assert summary["total_sessions"] == len(dataset.sessions)
    assert summary["total_frames"] > 0
    assert summary["total_duration_seconds"] > 0

    # Verify normal occlusion sessions exist
    hand_sessions = dataset.get_by_scenario(FieldScenarioType.NORMAL_OCCLUSION_HAND)
    assert len(hand_sessions) > 0
    assert len(hand_sessions[0].expected_events) == 0  # Normal occlusion should not flag violations
