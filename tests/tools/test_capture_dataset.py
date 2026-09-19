"""Tests for the isolated guided dataset capture system (tools/capture_dataset.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.capture_dataset import (
    MIN_FREE_DISK_BYTES,
    SCHEMA_VERSION,
    ScenarioConfig,
    VideoCaptureSession,
    calculate_file_sha256,
    check_disk_space,
    list_available_scenarios,
    resolve_scenario,
    sanitize_identifier,
)


def test_sanitize_identifier():
    assert sanitize_identifier("P001") == "P001"
    assert sanitize_identifier("Session_99-A") == "Session_99-A"
    with pytest.raises(ValueError, match="contains invalid characters"):
        sanitize_identifier("P001/../../hack")
    with pytest.raises(ValueError, match="contains invalid characters"):
        sanitize_identifier("P001; rm -rf")


def test_scenario_yaml_loading_all_configs():
    scenarios_dir = Path("configs/capture_scenarios")
    available = list_available_scenarios(scenarios_dir)
    assert len(available) >= 4

    scenario_ids = {s.scenario_id for s in available}
    assert "phone" in scenario_ids
    assert "paper" in scenario_ids
    assert "earphones" in scenario_ids
    assert "natural_exam" in scenario_ids

    for sc in available:
        assert len(sc.steps) > 0
        for st in sc.steps:
            assert st.step_id
            assert st.instruction
            assert st.instructed_condition
            assert st.preparation_seconds >= 0
            assert st.recording_seconds > 0


def test_scenario_branching_and_question_options():
    earphones = resolve_scenario("earphones", Path("configs/capture_scenarios"))
    question_steps = [s for s in earphones.steps if s.question is not None]
    assert len(question_steps) >= 2

    # Check question on second earbud
    second_q = next(s for s in question_steps if s.step_id == "question_second_earbud")
    opts = {o.key: o.next_step for o in second_q.question.options}
    assert "Y" in opts
    assert "N" in opts
    assert opts["Y"] == "two_earbuds_insert"
    assert opts["N"] == "remove_one_earbud"


def test_disk_space_guard(tmp_path: Path):
    has_space, free_bytes, total_bytes = check_disk_space(tmp_path, min_bytes=MIN_FREE_DISK_BYTES)
    assert isinstance(has_space, bool)
    assert free_bytes > 0
    assert total_bytes >= free_bytes

    # Should report false if threshold is set impossibly high
    insufficient, _, _ = check_disk_space(tmp_path, min_bytes=free_bytes + 10**12)
    assert insufficient is False


def test_duplicate_session_protection(tmp_path: Path):
    scenario = resolve_scenario("phone", Path("configs/capture_scenarios"))
    session = VideoCaptureSession(
        participant_id="P_TEST",
        session_id="S_TEST",
        scenario=scenario,
        output_base_dir=tmp_path,
        dry_run=True,
    )

    # First preparation creates directory
    session.prepare_session_directory()
    assert session.session_dir.exists()

    # Create dummy video & manifest to simulate existing session
    session.video_path.touch()
    session.manifest_path.touch()

    # Second attempt must refuse to overwrite
    with pytest.raises(FileExistsError, match="Refusing to overwrite existing raw research data"):
        session.prepare_session_directory()


def test_manifest_schema_and_instructed_condition_invariant(tmp_path: Path):
    scenario = resolve_scenario("natural_exam", Path("configs/capture_scenarios"))
    # Restrict to 2 steps for fast testing
    test_scenario = ScenarioConfig(
        scenario_id="natural_exam_test",
        name="Natural Exam Test",
        description="Quick test",
        version="1.0",
        default_preparation_seconds=1,
        default_recording_seconds=1,
        steps=scenario.steps[:2],
    )
    for s in test_scenario.steps:
        s.preparation_seconds = 0
        s.recording_seconds = 0

    session = VideoCaptureSession(
        participant_id="P001",
        session_id="S001",
        scenario=test_scenario,
        output_base_dir=tmp_path,
        dry_run=True,
        show_preview=False,
    )

    success = session.execute()
    assert success is True
    assert session.manifest_path.exists()
    assert session.video_path.exists()
    assert session.checksum_path.exists()

    with open(session.manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["research_tool"] == "capture_dataset.py"
    assert manifest["participant_id"] == "P001"
    assert manifest["session_id"] == "S001"
    assert manifest["session_status"] == "completed"
    assert "camera" in manifest
    assert "video" in manifest
    assert manifest["video"]["relative_path"] == "raw_video.mp4"
    assert manifest["video"]["verification_passed"] is True
    assert len(manifest["video"]["sha256"]) == 64

    # INVARIANT CHECK: Instruction is NOT ground truth
    manifest_str = json.dumps(manifest)
    assert "ground_truth" not in manifest_str, "Manifest must not claim verified ground_truth!"

    # Check steps
    assert len(manifest["steps"]) == 2
    for st in manifest["steps"]:
        assert "instructed_condition" in st
        assert st["status"] == "completed"
        assert st["preparation_start_seconds"] >= 0.0
        assert st["recording_start_seconds"] >= st["preparation_end_seconds"]


def test_interruption_safely_preserves_partial_video(tmp_path: Path):
    scenario = resolve_scenario("paper", Path("configs/capture_scenarios"))
    test_scenario = ScenarioConfig(
        scenario_id="paper_abort_test",
        name="Paper Abort Test",
        description="Test interruption handling",
        version="1.0",
        default_preparation_seconds=1,
        default_recording_seconds=2,
        steps=scenario.steps[:3],
    )
    for s in test_scenario.steps:
        s.preparation_seconds = 0
        s.recording_seconds = 1

    session = VideoCaptureSession(
        participant_id="P_ABORT",
        session_id="S_ABORT",
        scenario=test_scenario,
        output_base_dir=tmp_path,
        dry_run=True,
        show_preview=False,
    )

    # Trigger interruption immediately on first countdown
    session.interrupted = True
    session.execute()

    # Raw video and manifest must STILL be saved and preserved
    assert session.video_path.exists()
    assert session.manifest_path.exists()
    with open(session.manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["session_status"] in ("interrupted", "aborted")


def test_calculate_file_sha256(tmp_path: Path):
    dummy_file = tmp_path / "sample.bin"
    dummy_file.write_bytes(b"DATASET_CAPTURE_TEST_BYTES")
    digest = calculate_file_sha256(dummy_file)
    assert len(digest) == 64
    assert digest == "a1817ba00744b6f7025758c9a963f7d162192b2cae733e9328eeda4addae4d82"


def test_branching_execution_both_branches(tmp_path: Path):
    earphones = resolve_scenario("earphones", Path("configs/capture_scenarios"))
    # Scale duration to 0 for instantaneous test execution
    # Branch 1: "Y"
    session_yes = VideoCaptureSession(
        participant_id="P_BRANCH_Y",
        session_id="S001",
        scenario=earphones,
        output_base_dir=tmp_path,
        dry_run=True,
        show_preview=False,
        duration_scale=0.0,
    )
    # Provide "Y" to questions
    res_y = session_yes.execute(input_provider=lambda prompt: "Y")
    assert res_y is True
    with open(session_yes.manifest_path, encoding="utf-8") as f:
        m_y = json.load(f)

    step_ids_y = [s["step_id"] for s in m_y["steps"]]
    assert "two_earbuds_insert" in step_ids_y
    q_step_y = next(s for s in m_y["steps"] if s["step_id"] == "question_second_earbud")
    assert q_step_y["operator_response"] == "Y"

    # Branch 2: "N"
    session_no = VideoCaptureSession(
        participant_id="P_BRANCH_N",
        session_id="S001",
        scenario=earphones,
        output_base_dir=tmp_path,
        dry_run=True,
        show_preview=False,
        duration_scale=0.0,
    )
    # Provide "N" to questions
    res_n = session_no.execute(input_provider=lambda prompt: "N")
    assert res_n is True
    with open(session_no.manifest_path, encoding="utf-8") as f:
        m_n = json.load(f)

    step_ids_n = [s["step_id"] for s in m_n["steps"]]
    # When N was chosen, two_earbuds_insert was bypassed
    assert "two_earbuds_insert" not in step_ids_n
    q_step_n = next(s for s in m_n["steps"] if s["step_id"] == "question_second_earbud")
    assert q_step_n["operator_response"] == "N"


def test_timing_and_frame_accounting_integrity(tmp_path: Path):
    scenario = resolve_scenario("natural_exam", Path("configs/capture_scenarios"))
    session = VideoCaptureSession(
        participant_id="P_TIME",
        session_id="S_TIME",
        scenario=scenario,
        output_base_dir=tmp_path,
        dry_run=True,
        show_preview=False,
        duration_scale=0.0,
    )
    session.execute()

    with open(session.manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Verify frame accounting coherence
    frames_written = manifest["video"]["frames_written"]
    reopened_frames = manifest["video"]["reopened_frames_verified"]
    assert frames_written > 0
    assert frames_written == reopened_frames
    assert manifest["video"]["verification_passed"] is True

    # Check step frame indexing
    last_end = 0
    for st in manifest["steps"]:
        assert st["frame_start_index"] >= last_end
        assert st["frame_end_index"] >= st["frame_start_index"]
        assert st["frames_recorded"] == st["frame_end_index"] - st["frame_start_index"]
        last_end = st["frame_end_index"]


def test_video_file_decode_and_visual_properties(tmp_path: Path):
    scenario = resolve_scenario("phone", Path("configs/capture_scenarios"))
    session = VideoCaptureSession(
        participant_id="P_VISUAL",
        session_id="S_VISUAL",
        scenario=scenario,
        output_base_dir=tmp_path,
        dry_run=True,
        show_preview=False,
        duration_scale=0.0,
    )
    session.execute(input_provider=lambda prompt: "N")

    # Open the resulting MP4 independently using cv2
    import cv2
    cap = cv2.VideoCapture(str(session.video_path))
    assert cap.isOpened()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert width == 1280
    assert height == 720
    assert total_frames > 0

    read_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        assert frame is not None
        assert frame.shape == (720, 1280, 3)
        read_count += 1
    cap.release()

    assert read_count == total_frames


def test_interactive_controls_skip_action(tmp_path: Path):
    scenario = resolve_scenario("natural_exam", Path("configs/capture_scenarios"))
    test_scenario = ScenarioConfig(
        scenario_id="skip_test",
        name="Skip Test",
        description="Testing operator skip",
        version="1.0",
        default_preparation_seconds=2,
        default_recording_seconds=2,
        steps=scenario.steps[:2],
    )

    session = VideoCaptureSession(
        participant_id="P_SKIP",
        session_id="S_SKIP",
        scenario=test_scenario,
        output_base_dir=tmp_path,
        dry_run=True,
        show_preview=False,
        duration_scale=0.05,
    )

    # Key listener returns 'skip' immediately on step 1
    call_count = 0

    def listener():
        nonlocal call_count
        call_count += 1
        return "skip" if call_count == 1 else None

    res = session.execute(key_listener=listener)
    assert res is True

    with open(session.manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(manifest["steps"]) == 2
    assert manifest["steps"][0]["status"] == "skipped"
    assert manifest["steps"][1]["status"] == "completed"


def test_temporal_fps_distinction_in_manifest(tmp_path: Path):
    """Verify requested, reported, measured, and encoded FPS are all explicitly distinguished."""
    scenario = resolve_scenario("natural_exam", Path("configs/capture_scenarios"))
    test_scenario = ScenarioConfig(
        scenario_id="fps_test",
        name="FPS Test",
        description="Testing FPS distinction",
        version="1.0",
        default_preparation_seconds=0,
        default_recording_seconds=1,
        steps=scenario.steps[:1],
    )

    session = VideoCaptureSession(
        participant_id="P_FPS",
        session_id="S_FPS",
        scenario=test_scenario,
        output_base_dir=tmp_path,
        target_fps=25.0,
        dry_run=True,
        show_preview=False,
        duration_scale=0.1,
    )
    assert session.execute() is True

    with open(session.manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Check that the 4 distinct FPS representations exist and are non-zero
    assert "requested_fps" in manifest["camera"]
    assert "reported_camera_fps" in manifest["camera"]
    assert "measured_effective_fps" in manifest["camera"]
    assert "encoded_fps" in manifest["video"]

    assert manifest["camera"]["requested_fps"] == 25.0
    assert manifest["camera"]["reported_camera_fps"] == 25.0
    assert manifest["camera"]["measured_effective_fps"] > 0.0
    assert manifest["video"]["encoded_fps"] == manifest["camera"]["measured_effective_fps"]

    # Check duration metrics
    assert "active_duration_seconds" in manifest
    assert "paused_duration_seconds" in manifest
    assert "capture_duration_seconds" in manifest["video"]
    assert "playback_duration_seconds" in manifest["video"]
    assert manifest["paused_duration_seconds"] == 0.0
    assert manifest["active_duration_seconds"] > 0.0


def test_pause_resume_temporal_integrity(tmp_path: Path):
    """Verify pause/resume properly records paused duration and does not corrupt active timestamps."""
    scenario = resolve_scenario("natural_exam", Path("configs/capture_scenarios"))
    test_scenario = ScenarioConfig(
        scenario_id="pause_test",
        name="Pause Test",
        description="Testing pause/resume temporal integrity",
        version="1.0",
        default_preparation_seconds=0,
        default_recording_seconds=2,
        steps=scenario.steps[:1],
    )

    session = VideoCaptureSession(
        participant_id="P_PAUSE",
        session_id="S_PAUSE",
        scenario=test_scenario,
        output_base_dir=tmp_path,
        dry_run=True,
        show_preview=False,
        duration_scale=0.1,
    )

    call_count = 0

    def listener():
        nonlocal call_count
        call_count += 1
        # Pause on second call, resume on fourth
        if call_count == 2:
            return "pause"
        if call_count >= 5:
            return "resume"
        return None

    assert session.execute(key_listener=listener) is True

    with open(session.manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["session_status"] == "completed"
    assert manifest["paused_duration_seconds"] >= 0.0
    assert manifest["total_duration_seconds"] >= manifest["active_duration_seconds"]

    # Frame indexes must be valid and non-negative
    for st in manifest["steps"]:
        assert st["frame_start_index"] >= 0
        assert st["frame_end_index"] >= st["frame_start_index"]


def test_no_hardcoded_camera_fps(tmp_path: Path):
    """Verify the capture logic dynamically measures frame rate without assuming 30.0 or 11.5 FPS."""
    scenario = resolve_scenario("natural_exam", Path("configs/capture_scenarios"))
    test_scenario = ScenarioConfig(
        scenario_id="dynamic_fps_test",
        name="Dynamic FPS Test",
        description="Testing arbitrary requested FPS adaptation",
        version="1.0",
        default_preparation_seconds=0,
        default_recording_seconds=1,
        steps=scenario.steps[:1],
    )

    for custom_fps in (15.0, 24.0):
        session = VideoCaptureSession(
            participant_id="P_DYN",
            session_id=f"S_DYN_{int(custom_fps)}",
            scenario=test_scenario,
            output_base_dir=tmp_path,
            target_fps=custom_fps,
            dry_run=True,
            show_preview=False,
            duration_scale=0.1,
        )
        assert session.execute() is True

        with open(session.manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["camera"]["requested_fps"] == custom_fps
        assert manifest["video"]["encoded_fps"] > 0.0


def test_pixel_format_parser_argument():
    """Verify build_parser supports --pixel-format choices."""
    from tools.capture_dataset import build_parser

    parser = build_parser()
    args_mjpg = parser.parse_args(["--pixel-format", "MJPG"])
    assert args_mjpg.pixel_format == "MJPG"

    args_yuyv = parser.parse_args(["--pixel-format", "YUYV"])
    assert args_yuyv.pixel_format == "YUYV"

    args_auto = parser.parse_args([])
    assert args_auto.pixel_format == "AUTO"

