"""Tests for the command-line entry points.

The CLI is how the pipeline is exercised by hand, so its argument surface and its
behaviour when models or a camera are missing both matter. These tests never open a
camera or load a model.
"""

import pytest

from proctoring.cli.__main__ import main as dispatcher_main
from proctoring.cli.live import build_parser

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [[], ["--help"], ["help"]])
def test_dispatcher_help_exits_cleanly(argv, capsys):
    assert dispatcher_main(argv) == 0
    output = capsys.readouterr().out
    assert "live" in output and "video" in output


def test_dispatcher_rejects_an_unknown_command(capsys):
    assert dispatcher_main(["nonsense"]) == 1
    assert "Unknown command" in capsys.readouterr().out


def test_video_command_reports_a_missing_file(capsys):
    assert dispatcher_main(["video", "/no/such/recording.mp4"]) == 1
    assert "not found" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# Live argument surface
# ---------------------------------------------------------------------------


def test_live_defaults_are_conservative():
    """Defaults must not silently enable the expensive or unreliable stages."""
    args = build_parser().parse_args([])
    assert args.strictness == "STANDARD"
    assert args.detect_wearables is False, "wearable detection must be opt-in"
    assert args.enroll is False
    assert args.fps == 4.0
    assert args.no_behaviour is False


def test_live_accepts_every_strictness_level():
    for level in ("STANDARD", "STRICT", "MAXIMUM"):
        assert build_parser().parse_args(["--strictness", level]).strictness == level


def test_live_rejects_an_unknown_strictness_level():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--strictness", "PARANOID"])


def test_live_flags_map_to_the_intended_settings():
    args = build_parser().parse_args(
        [
            "--enroll",
            "--enroll-samples",
            "7",
            "--strictness",
            "MAXIMUM",
            "--detect-wearables",
            "--no-objects",
            "--fps",
            "2.5",
            "--zip",
            "--session-id",
            "exam_42",
            "--student",
            "A. Candidate",
        ]
    )
    assert args.enroll is True and args.enroll_samples == 7
    assert args.strictness == "MAXIMUM"
    assert args.detect_wearables is True and args.no_objects is True
    assert args.fps == 2.5 and args.zip is True
    assert args.session_id == "exam_42" and args.student == "A. Candidate"


def test_live_session_id_defaults_to_something_unique():
    first = build_parser().parse_args([]).session_id
    assert first.startswith("live_") and len(first) > 6


# ---------------------------------------------------------------------------
# Model loading degradation
# ---------------------------------------------------------------------------


def test_missing_models_disable_stages_rather_than_crashing(tmp_path, capsys):
    """A user without model files must get a clear message, not a traceback."""
    from proctoring.cli.live import load_detectors

    face_detector, face_verifier, object_detector = load_detectors(tmp_path, enable_objects=False)
    assert face_detector is None and face_verifier is None and object_detector is None
    output = capsys.readouterr().out
    assert "MISS" in output
    assert "download_models" in output
