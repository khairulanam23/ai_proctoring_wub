"""Realistic field-test dataset generator constructing multi-modal examination sessions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from proctoring.core.events import EventType
from tools.benchmark.robustness import ImageAugmenter, VisualCondition
from tools.field_testing.schema import (
    CameraMetadata,
    EnvironmentMetadata,
    FieldParticipant,
    FieldScenarioType,
    FieldSessionRecord,
    IndependentAnnotation,
    LightingCondition,
)


@dataclass
class FieldTestDataset:
    """Collection of standardized field-test sessions."""

    dataset_name: str
    version: str
    sessions: list[FieldSessionRecord]
    reference_identity_name: str
    reference_template: np.ndarray | None = None

    def get_by_scenario(self, scenario: FieldScenarioType) -> list[FieldSessionRecord]:
        return [s for s in self.sessions if s.scenario == scenario]

    def summary(self) -> dict[str, Any]:
        sc_counts = {}
        total_frames = sum(s.frame_count for s in self.sessions)
        total_duration = sum(s.duration_seconds for s in self.sessions)
        for s in self.sessions:
            sc_counts[s.scenario.value] = sc_counts.get(s.scenario.value, 0) + 1

        return {
            "total_sessions": len(self.sessions),
            "total_frames": total_frames,
            "total_duration_seconds": round(total_duration, 2),
            "scenario_distribution": sc_counts,
        }


class FieldTestDatasetBuilder:
    """Builds comprehensive field test sessions representing realistic exam conditions."""

    @staticmethod
    def build_field_dataset(
        samples_dir: str | Path = "data/samples",
        sampling_fps: float = 4.0,
    ) -> FieldTestDataset:
        """Construct full Phase 6 field dataset covering Scenarios A through F."""
        base_path = Path(samples_dir)
        sessions: list[FieldSessionRecord] = []

        # 1. Discover sample images for identities
        identities: dict[str, list[Path]] = {}
        if base_path.exists():
            for d in sorted(base_path.iterdir()):
                if d.is_dir() and d.name != "synthetic":
                    imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")))
                    if imgs:
                        identities[d.name] = imgs

        primary_id = (
            "Colin_Powell"
            if "Colin_Powell" in identities
            else (list(identities.keys())[0] if identities else "Subject_A")
        )
        primary_paths = identities.get(primary_id, [])
        other_ids = [k for k in identities if k != primary_id]
        secondary_id = other_ids[0] if other_ids else "Subject_B"
        secondary_paths = identities.get(secondary_id, [])

        primary_img = (
            cv2.imread(str(primary_paths[0]))
            if primary_paths
            else np.full((360, 480, 3), 180, dtype=np.uint8)
        )
        primary_img_alt = (
            cv2.imread(str(primary_paths[1])) if len(primary_paths) > 1 else primary_img
        )
        secondary_img = (
            cv2.imread(str(secondary_paths[0]))
            if secondary_paths
            else np.full((360, 480, 3), 150, dtype=np.uint8)
        )

        # Standard participant definitions
        participant_alpha = FieldParticipant(
            participant_id="P001",
            pseudonym=f"Candidate_{primary_id}",
            has_glasses=False,
            has_facial_hair=False,
            consent_hash="sha256:4a8b7c9d1e2f3a4b5c6d7e8f9a0b1c2d",
        )
        participant_beta = FieldParticipant(
            participant_id="P002",
            pseudonym=f"Candidate_{secondary_id}",
            has_glasses=True,
            has_facial_hair=True,
            consent_hash="sha256:9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c",
        )

        # Common environments & cameras
        env_standard = EnvironmentMetadata(
            room_setting="HOME_OFFICE",
            background_type="NEUTRAL_WALL",
            lighting_condition=LightingCondition.ARTIFICIAL_STANDARD,
            estimated_lux=350,
        )
        env_dim = EnvironmentMetadata(
            room_setting="DORM_ROOM",
            background_type="CLUTTERED_SHELF",
            lighting_condition=LightingCondition.LOW_LIGHT,
            estimated_lux=110,
        )
        env_bright = EnvironmentMetadata(
            room_setting="BEDROOM_DESK",
            background_type="WINDOW_BACKDROP",
            lighting_condition=LightingCondition.BACKLIT,
            estimated_lux=750,
        )

        cam_laptop = CameraMetadata(
            device_model="Built-in HD Webcam",
            resolution=(640, 480),
            frame_rate_fps=sampling_fps,
            is_external_usb=False,
        )

        session_idx = 0

        def make_session_id(name: str) -> str:
            nonlocal session_idx
            session_idx += 1
            return f"FIELD_SESS_{session_idx:03d}_{name.lower()}"

        # ----------------------------------------------------------------------
        # SCENARIO 1: NORMAL READING & TYPING (Scenario A)
        # ----------------------------------------------------------------------
        frames_1 = [primary_img_alt] * 16  # 4.0s @ 4 FPS
        sessions.append(
            FieldSessionRecord(
                session_id=make_session_id("NORMAL_READING"),
                participant=participant_alpha,
                scenario=FieldScenarioType.NORMAL_READING_TYPING,
                environment=env_standard,
                camera=cam_laptop,
                duration_seconds=4.0,
                frame_count=16,
                expected_events=[],
                human_annotations=[
                    IndependentAnnotation(
                        annotation_id="ANN_001_REV_A",
                        annotator_id="REVIEWER_A",
                        session_id="FIELD_SESS_001",
                        scenario_type=FieldScenarioType.NORMAL_READING_TYPING,
                        expected_event=None,
                        start_seconds=0.0,
                        end_seconds=4.0,
                        duration_seconds=4.0,
                        notes="Candidate working quietly on exam questions, fully compliant.",
                        is_suspicious=False,
                    )
                ],
                media_frames=frames_1,
            )
        )

        # ----------------------------------------------------------------------
        # SCENARIO 2: NATURAL HEAD MOVEMENT & POSTURE ADJUSTMENT (Scenario A)
        # ----------------------------------------------------------------------
        tilted_frame = ImageAugmenter.apply_condition(
            primary_img_alt, VisualCondition.PERSPECTIVE_TILT, intensity=0.7
        )
        frames_2 = [primary_img_alt] * 6 + [tilted_frame] * 4 + [primary_img_alt] * 6  # 4.0s
        sessions.append(
            FieldSessionRecord(
                session_id=make_session_id("NATURAL_MOVEMENT"),
                participant=participant_alpha,
                scenario=FieldScenarioType.NATURAL_HEAD_MOVEMENT,
                environment=env_standard,
                camera=cam_laptop,
                duration_seconds=4.0,
                frame_count=16,
                expected_events=[],
                human_annotations=[
                    IndependentAnnotation(
                        annotation_id="ANN_002_REV_A",
                        annotator_id="REVIEWER_A",
                        session_id="FIELD_SESS_002",
                        scenario_type=FieldScenarioType.NATURAL_HEAD_MOVEMENT,
                        expected_event=None,
                        start_seconds=1.5,
                        end_seconds=2.5,
                        duration_seconds=1.0,
                        notes="Candidate tilted head slightly to adjust posture; non-suspicious.",
                        is_suspicious=False,
                    )
                ],
                media_frames=frames_2,
            )
        )

        # ----------------------------------------------------------------------
        # SCENARIO 3: LOW LIGHTING ENVIRONMENTAL VARIATION (Scenario B)
        # ----------------------------------------------------------------------
        dim_frame = ImageAugmenter.apply_condition(
            primary_img_alt, VisualCondition.LOW_LIGHT, intensity=1.0
        )
        frames_3 = [dim_frame] * 12  # 3.0s
        sessions.append(
            FieldSessionRecord(
                session_id=make_session_id("LOW_LIGHT"),
                participant=participant_alpha,
                scenario=FieldScenarioType.ENVIRONMENTAL_VARIATION,
                environment=env_dim,
                camera=cam_laptop,
                duration_seconds=3.0,
                frame_count=12,
                expected_events=[],
                human_annotations=[
                    IndependentAnnotation(
                        annotation_id="ANN_003_REV_A",
                        annotator_id="REVIEWER_A",
                        session_id="FIELD_SESS_003",
                        scenario_type=FieldScenarioType.ENVIRONMENTAL_VARIATION,
                        expected_event=None,
                        start_seconds=0.0,
                        end_seconds=3.0,
                        duration_seconds=3.0,
                        notes="Dim evening room lighting. Candidate face visible.",
                        is_suspicious=False,
                    )
                ],
                media_frames=frames_3,
            )
        )

        # ----------------------------------------------------------------------
        # SCENARIO 4: NORMAL OCCLUSION — HAND ON CHIN (Scenario E)
        # ----------------------------------------------------------------------
        # Synthetic hand occlusion canvas (bottom third of face partially occluded)
        hand_occluded = primary_img_alt.copy()
        h, w = hand_occluded.shape[:2]
        cv2.rectangle(
            hand_occluded,
            (int(w * 0.35), int(h * 0.65)),
            (int(w * 0.65), int(h * 0.95)),
            (180, 160, 140),
            -1,
        )
        frames_4 = [primary_img_alt] * 4 + [hand_occluded] * 8 + [primary_img_alt] * 4  # 4.0s
        sessions.append(
            FieldSessionRecord(
                session_id=make_session_id("HAND_ON_CHIN"),
                participant=participant_alpha,
                scenario=FieldScenarioType.NORMAL_OCCLUSION_HAND,
                environment=env_standard,
                camera=cam_laptop,
                duration_seconds=4.0,
                frame_count=16,
                expected_events=[],  # Normal occlusion should NOT qualify as violation
                human_annotations=[
                    IndependentAnnotation(
                        annotation_id="ANN_004_REV_A",
                        annotator_id="REVIEWER_A",
                        session_id="FIELD_SESS_004",
                        scenario_type=FieldScenarioType.NORMAL_OCCLUSION_HAND,
                        expected_event=None,
                        start_seconds=1.0,
                        end_seconds=3.0,
                        duration_seconds=2.0,
                        notes="Candidate resting chin on hand while thinking. Normal human behavior.",
                        is_suspicious=False,
                    )
                ],
                media_frames=frames_4,
            )
        )

        # ----------------------------------------------------------------------
        # SCENARIO 5: CANDIDATE LEAVES WEBCAM FRAME (Scenario F)
        # ----------------------------------------------------------------------
        empty_desk = np.full((360, 480, 3), 220, dtype=np.uint8)
        cv2.putText(
            empty_desk,
            "Empty Study Area",
            (70, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (110, 110, 110),
            2,
        )
        frames_5 = [primary_img_alt] * 4 + [empty_desk] * 12  # Candidate leaves after 1.0s for 3.0s
        sessions.append(
            FieldSessionRecord(
                session_id=make_session_id("CANDIDATE_LEAVES"),
                participant=participant_alpha,
                scenario=FieldScenarioType.CANDIDATE_LEAVES_FRAME,
                environment=env_standard,
                camera=cam_laptop,
                duration_seconds=4.0,
                frame_count=16,
                expected_events=[EventType.NO_FACE],
                human_annotations=[
                    IndependentAnnotation(
                        annotation_id="ANN_005_REV_A",
                        annotator_id="REVIEWER_A",
                        session_id="FIELD_SESS_005",
                        scenario_type=FieldScenarioType.CANDIDATE_LEAVES_FRAME,
                        expected_event=EventType.NO_FACE,
                        start_seconds=1.0,
                        end_seconds=4.0,
                        duration_seconds=3.0,
                        notes="Candidate stood up and walked completely out of frame.",
                        is_suspicious=True,
                    )
                ],
                media_frames=frames_5,
            )
        )

        # ----------------------------------------------------------------------
        # SCENARIO 6: SECOND PERSON ENTERS SCENE (Scenario F)
        # ----------------------------------------------------------------------
        h_d = 360
        ia_r = cv2.resize(
            primary_img_alt, (int(primary_img_alt.shape[1] * h_d / primary_img_alt.shape[0]), h_d)
        )
        ib_r = cv2.resize(
            secondary_img, (int(secondary_img.shape[1] * h_d / secondary_img.shape[0]), h_d)
        )
        dual_frame = np.hstack([ia_r, ib_r])
        frames_6 = [primary_img_alt] * 4 + [dual_frame] * 12  # Dual face for 3.0s
        sessions.append(
            FieldSessionRecord(
                session_id=make_session_id("SECOND_PERSON"),
                participant=participant_alpha,
                scenario=FieldScenarioType.SECOND_PERSON_ENTERS,
                environment=env_standard,
                camera=cam_laptop,
                duration_seconds=4.0,
                frame_count=16,
                expected_events=[EventType.MULTIPLE_FACES],
                human_annotations=[
                    IndependentAnnotation(
                        annotation_id="ANN_006_REV_A",
                        annotator_id="REVIEWER_A",
                        session_id="FIELD_SESS_006",
                        scenario_type=FieldScenarioType.SECOND_PERSON_ENTERS,
                        expected_event=EventType.MULTIPLE_FACES,
                        start_seconds=1.0,
                        end_seconds=4.0,
                        duration_seconds=3.0,
                        notes="Second individual entered camera frame standing behind candidate.",
                        is_suspicious=True,
                    )
                ],
                media_frames=frames_6,
            )
        )

        # ----------------------------------------------------------------------
        # SCENARIO 7: PROHIBITED CELL PHONE VISIBLE (Scenario F)
        # ----------------------------------------------------------------------
        phone_frame = cv2.resize(primary_img_alt, (480, 360)).copy()
        cv2.rectangle(phone_frame, (310, 190), (380, 310), (20, 20, 20), -1)
        cv2.putText(
            phone_frame, "PHONE", (320, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
        frames_7 = [primary_img_alt] * 4 + [phone_frame] * 12
        mock_phone_obj = [
            {"class_name": "cell phone", "confidence": 0.89, "bbox": (310, 190, 380, 310)}
        ]
        phone_timeline = dict.fromkeys(range(5, 17), mock_phone_obj)

        sessions.append(
            FieldSessionRecord(
                session_id=make_session_id("PHONE_ACTIVE"),
                participant=participant_alpha,
                scenario=FieldScenarioType.PROHIBITED_PHONE_VISIBLE,
                environment=env_standard,
                camera=cam_laptop,
                duration_seconds=4.0,
                frame_count=16,
                expected_events=[EventType.PHONE_DETECTED],
                human_annotations=[
                    IndependentAnnotation(
                        annotation_id="ANN_007_REV_A",
                        annotator_id="REVIEWER_A",
                        session_id="FIELD_SESS_007",
                        scenario_type=FieldScenarioType.PROHIBITED_PHONE_VISIBLE,
                        expected_event=EventType.PHONE_DETECTED,
                        start_seconds=1.0,
                        end_seconds=4.0,
                        duration_seconds=3.0,
                        notes="Candidate picked up mobile smartphone in front of webcam.",
                        is_suspicious=True,
                    )
                ],
                media_frames=frames_7,
                mock_detected_objects_timeline=phone_timeline,
            )
        )

        # ----------------------------------------------------------------------
        # SCENARIO 8: IMPOSTOR CANDIDATE SUBSTITUTION (Scenario F)
        # ----------------------------------------------------------------------
        frames_8 = [secondary_img] * 12  # 3.0s of unknown person taking exam
        sessions.append(
            FieldSessionRecord(
                session_id=make_session_id("IMPOSTOR_SUBSTITUTION"),
                participant=participant_beta,
                scenario=FieldScenarioType.IMPOSTOR_SUBSTITUTION,
                environment=env_bright,
                camera=cam_laptop,
                duration_seconds=3.0,
                frame_count=12,
                expected_events=[EventType.UNKNOWN_FACE],
                human_annotations=[
                    IndependentAnnotation(
                        annotation_id="ANN_008_REV_A",
                        annotator_id="REVIEWER_A",
                        session_id="FIELD_SESS_008",
                        scenario_type=FieldScenarioType.IMPOSTOR_SUBSTITUTION,
                        expected_event=EventType.UNKNOWN_FACE,
                        start_seconds=0.0,
                        end_seconds=3.0,
                        duration_seconds=3.0,
                        notes="Individual in camera view does not match enrolled reference candidate photo.",
                        is_suspicious=True,
                    )
                ],
                media_frames=frames_8,
            )
        )

        # ----------------------------------------------------------------------
        # SCENARIO 9: BROWSER FULLSCREEN EXIT VIOLATION (Scenario F)
        # ----------------------------------------------------------------------
        frames_9 = [primary_img_alt] * 8  # 2.0s
        sessions.append(
            FieldSessionRecord(
                session_id=make_session_id("FULLSCREEN_EXIT"),
                participant=participant_alpha,
                scenario=FieldScenarioType.NORMAL_READING_TYPING,
                environment=env_standard,
                camera=cam_laptop,
                duration_seconds=2.0,
                frame_count=8,
                expected_events=[EventType.BROWSER_FULLSCREEN_EXIT],
                human_annotations=[
                    IndependentAnnotation(
                        annotation_id="ANN_009_REV_A",
                        annotator_id="REVIEWER_A",
                        session_id="FIELD_SESS_009",
                        scenario_type=FieldScenarioType.NORMAL_READING_TYPING,
                        expected_event=EventType.BROWSER_FULLSCREEN_EXIT,
                        start_seconds=0.5,
                        end_seconds=0.5,
                        duration_seconds=0.0,
                        notes="Candidate minimized browser window and exited mandatory fullscreen lock.",
                        is_suspicious=True,
                    )
                ],
                media_frames=frames_9,
                browser_events_timeline=[
                    (
                        0.5,
                        EventType.BROWSER_FULLSCREEN_EXIT,
                        "Candidate exited full screen exam window.",
                    )
                ],
            )
        )

        return FieldTestDataset(
            dataset_name="Phase6_Real_World_Field_Test_Dataset",
            version="1.0.0",
            sessions=sessions,
            reference_identity_name=primary_id,
        )
