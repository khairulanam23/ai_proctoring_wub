"""Tests for field testing schema and pseudonymous participant data structures."""

from tools.field_testing.schema import (
    CameraMetadata,
    EnvironmentMetadata,
    FieldParticipant,
    LightingCondition,
)


def test_field_participant_and_metadata_serialization():
    """Verify participant pseudonymization and environment metadata serialization."""
    part = FieldParticipant(
        participant_id="P001",
        pseudonym="Candidate_Alice",
        has_glasses=True,
        has_facial_hair=False,
        consent_hash="sha256:abcd1234",
    )
    p_dict = part.to_dict()
    assert p_dict["participant_id"] == "P001"
    assert p_dict["pseudonym"] == "Candidate_Alice"
    assert p_dict["has_glasses"] is True
    assert p_dict["consent_verified"] is True

    env = EnvironmentMetadata(
        room_setting="HOME_DESK",
        lighting_condition=LightingCondition.DAYLIGHT,
        estimated_lux=400,
    )
    e_dict = env.to_dict()
    assert e_dict["lighting_condition"] == "DAYLIGHT"
    assert e_dict["estimated_lux"] == 400

    cam = CameraMetadata(
        device_model="HD Webcam",
        resolution=(1280, 720),
        frame_rate_fps=4.0,
    )
    c_dict = cam.to_dict()
    assert c_dict["resolution"] == [1280, 720]
