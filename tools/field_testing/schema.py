"""Field-testing schema, pseudonymous participant data structures, environment metadata, and ground-truth annotation definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from proctoring.core.events import EventType


class LightingCondition(str, Enum):
    """Environmental illumination conditions encountered in remote examination."""

    DAYLIGHT = "DAYLIGHT"  # Natural indirect sunlight
    ARTIFICIAL_STANDARD = "ARTIFICIAL_STANDARD"  # Standard indoor overhead lighting
    LOW_LIGHT = "LOW_LIGHT"  # Dim room / night study lamp
    BACKLIT = "BACKLIT"  # Bright window behind candidate
    UNEVEN_GLARE = "UNEVEN_GLARE"  # Side lamp causing harsh shadows


class FieldScenarioType(str, Enum):
    """Realistic examination behavior scenarios."""

    NORMAL_READING_TYPING = "NORMAL_READING_TYPING"  # Scenario A: Candidate focused on exam
    NATURAL_HEAD_MOVEMENT = "NATURAL_HEAD_MOVEMENT"  # Scenario A: Looking around desk, stretch
    ENVIRONMENTAL_VARIATION = "ENVIRONMENTAL_VARIATION"  # Scenario B: Lighting & background changes
    OFF_AXIS_CAMERA = "OFF_AXIS_CAMERA"  # Scenario C: Low/tilted laptop webcam angle
    STUDENT_VARIATION = "STUDENT_VARIATION"  # Scenario D: Glasses, facial hair, distance
    NORMAL_OCCLUSION_HAND = "NORMAL_OCCLUSION_HAND"  # Scenario E: Hand near chin / thinking posture
    NORMAL_OCCLUSION_DRINK = "NORMAL_OCCLUSION_DRINK"  # Scenario E: Drinking water / cup
    NORMAL_OCCLUSION_GLASSES = "NORMAL_OCCLUSION_GLASSES"  # Scenario E: Adjusting eyeglasses
    CANDIDATE_LEAVES_FRAME = (
        "CANDIDATE_LEAVES_FRAME"  # Scenario F: Candidate steps away from camera
    )
    SECOND_PERSON_ENTERS = "SECOND_PERSON_ENTERS"  # Scenario F: Secondary person in background
    PROHIBITED_PHONE_VISIBLE = (
        "PROHIBITED_PHONE_VISIBLE"  # Scenario F: Cell phone displayed in view
    )
    PROHIBITED_BOOK_VISIBLE = "PROHIBITED_BOOK_VISIBLE"  # Scenario F: Reference textbook on desk
    IMPOSTOR_SUBSTITUTION = "IMPOSTOR_SUBSTITUTION"  # Scenario F: Different person taking exam
    CAMERA_LENS_COVERED = "CAMERA_LENS_COVERED"  # Scenario F: Physical lens obstruction


@dataclass
class CameraMetadata:
    """Hardware and stream capture specifications."""

    device_model: str = "Laptop Integrated Webcam"
    resolution: tuple[int, int] = (640, 480)
    frame_rate_fps: float = 4.0
    field_of_view_deg: float = 75.0
    is_external_usb: bool = False
    mounting_position: str = "top_of_screen"

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_model": self.device_model,
            "resolution": list(self.resolution),
            "frame_rate_fps": self.frame_rate_fps,
            "field_of_view_deg": self.field_of_view_deg,
            "is_external_usb": self.is_external_usb,
            "mounting_position": self.mounting_position,
        }


@dataclass
class EnvironmentMetadata:
    """Room, lighting, and physical setting characteristics."""

    room_setting: str = "HOME_DESK"
    background_type: str = "NEUTRAL_WALL"
    lighting_condition: LightingCondition = LightingCondition.ARTIFICIAL_STANDARD
    estimated_lux: int = 300
    ambient_noise_level: str = "QUIET"

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_setting": self.room_setting,
            "background_type": self.background_type,
            "lighting_condition": self.lighting_condition.value,
            "estimated_lux": self.estimated_lux,
            "ambient_noise_level": self.ambient_noise_level,
        }


@dataclass
class FieldParticipant:
    """Pseudonymous participant record respecting data minimization and privacy principles."""

    participant_id: str  # e.g. "P001"
    pseudonym: str  # e.g. "Candidate_Alpha"
    has_glasses: bool = False
    has_facial_hair: bool = False
    hair_style: str = "short"
    approx_distance_cm: int = 60
    consent_hash: str = ""  # Cryptographic hash of digital consent form

    def to_dict(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "pseudonym": self.pseudonym,
            "has_glasses": self.has_glasses,
            "has_facial_hair": self.has_facial_hair,
            "hair_style": self.hair_style,
            "approx_distance_cm": self.approx_distance_cm,
            "consent_verified": bool(self.consent_hash),
        }


@dataclass
class IndependentAnnotation:
    """Independent ground-truth event annotation created by human annotators."""

    annotation_id: str
    annotator_id: str  # e.g. "REVIEWER_A"
    session_id: str
    scenario_type: FieldScenarioType
    expected_event: EventType | None
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    notes: str = ""
    is_suspicious: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "annotator_id": self.annotator_id,
            "session_id": self.session_id,
            "scenario_type": self.scenario_type.value,
            "expected_event": self.expected_event.value if self.expected_event else None,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.duration_seconds,
            "notes": self.notes,
            "is_suspicious": self.is_suspicious,
        }


@dataclass
class AnnotatorAgreementMetrics:
    """Measures human reviewer inter-annotator agreement on ground-truth events."""

    total_intervals_evaluated: int
    concordant_intervals: int
    discordant_intervals: int
    raw_agreement_percentage: float
    cohen_kappa_estimate: float
    disagreement_summary: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_intervals_evaluated": self.total_intervals_evaluated,
            "concordant_intervals": self.concordant_intervals,
            "discordant_intervals": self.discordant_intervals,
            "raw_agreement_percentage": round(self.raw_agreement_percentage, 2),
            "cohen_kappa_estimate": round(self.cohen_kappa_estimate, 4),
            "disagreement_summary": self.disagreement_summary,
        }


@dataclass
class FieldSessionRecord:
    """Complete specification of a field-test exam session."""

    session_id: str
    participant: FieldParticipant
    scenario: FieldScenarioType
    environment: EnvironmentMetadata
    camera: CameraMetadata
    duration_seconds: float
    frame_count: int
    expected_events: list[EventType] = field(default_factory=list)
    human_annotations: list[IndependentAnnotation] = field(default_factory=list)
    media_frames: list[np.ndarray] = field(default_factory=list)
    mock_detected_objects_timeline: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    browser_events_timeline: list[tuple[float, EventType, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "participant": self.participant.to_dict(),
            "scenario": self.scenario.value,
            "environment": self.environment.to_dict(),
            "camera": self.camera.to_dict(),
            "duration_seconds": self.duration_seconds,
            "frame_count": self.frame_count,
            "expected_events": [e.value for e in self.expected_events],
            "human_annotations": [a.to_dict() for a in self.human_annotations],
            "metadata": self.metadata,
        }
