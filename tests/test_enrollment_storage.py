"""Targeted tests for the enrolment, session-storage and cache-cleanup lifecycle.

The behaviours here are the ones a student and an invigilator would notice: a
student is asked to sit through enrolment exactly once, one examination never
overwrites another, and nothing disposable is left lying on disk afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from proctoring.integration.session_store import SessionStore
from proctoring.storage import (
    ENROLLMENT_IMAGE_COUNT,
    EnrollmentState,
    ProctoringStorage,
)


@pytest.fixture
def storage(tmp_path: Path) -> ProctoringStorage:
    return ProctoringStorage(tmp_path / "data")


def _images(count: int = ENROLLMENT_IMAGE_COUNT) -> list[np.ndarray]:
    """Visually distinct frames, so a mix-up between them would be detectable."""
    return [np.full((120, 160, 3), 20 + index * 40, dtype=np.uint8) for index in range(count)]


def _templates(count: int = ENROLLMENT_IMAGE_COUNT) -> list[np.ndarray]:
    return [np.full(128, 0.1 * (index + 1), dtype=np.float32) for index in range(count)]


def _enrol(storage: ProctoringStorage, student: str = "Anam"):
    return storage.save_enrollment(student, _images(), _templates(), model={"verifier": "SFace"})


# ---------------------------------------------------------------------------
# 1-3. Enrol once, reuse thereafter, refuse to trust a damaged set
# ---------------------------------------------------------------------------


def test_first_enrolment_stores_exactly_five_reference_images(storage):
    record = _enrol(storage)

    assert record.state is EnrollmentState.VALID
    assert len(record.image_paths) == ENROLLMENT_IMAGE_COUNT
    assert len(record.templates) == ENROLLMENT_IMAGE_COUNT

    stored = sorted(p.name for p in record.directory.glob("image_*.jpg"))
    assert stored == [f"image_{i:02d}.jpg" for i in range(1, ENROLLMENT_IMAGE_COUNT + 1)]


def test_an_already_enrolled_student_is_recognised_without_recapturing(storage):
    """The reported bug: `--enroll` captured five photographs on every run.

    The decision the CLI makes is ``is_enrolled``; if that reports True the capture
    loop is never entered, so this is the behaviour worth pinning.
    """
    first = _enrol(storage)
    first_hashes = [p.read_bytes() for p in sorted(first.image_paths)]

    assert storage.is_enrolled("Anam") is True

    reloaded = storage.load_enrollment("Anam")
    assert reloaded.is_usable is True
    assert len(reloaded.templates) == ENROLLMENT_IMAGE_COUNT
    # The very same files — nothing was re-captured or rewritten.
    assert [p.read_bytes() for p in sorted(reloaded.image_paths)] == first_hashes


def test_a_missing_enrolment_reports_missing(storage):
    record = storage.load_enrollment("NeverSeen")
    assert record.state is EnrollmentState.MISSING
    assert record.is_usable is False
    assert storage.is_enrolled("NeverSeen") is False


def test_an_incomplete_enrolment_is_not_used_for_verification(storage):
    """A deleted reference image must invalidate the set, not silently shrink it."""
    record = _enrol(storage)
    record.image_paths[0].unlink()

    reloaded = storage.load_enrollment("Anam")
    assert reloaded.state is EnrollmentState.INCOMPLETE
    assert reloaded.is_usable is False
    assert storage.load_templates("Anam") == []


def test_an_altered_reference_image_is_detected_as_corrupt(storage):
    """Verifying against a tampered reference would produce confident mismatches."""
    record = _enrol(storage)
    target = record.image_paths[0]
    target.write_bytes(target.read_bytes() + b"tampered")

    reloaded = storage.load_enrollment("Anam")
    assert reloaded.state is EnrollmentState.CORRUPT
    assert reloaded.is_usable is False
    assert any("altered" in problem for problem in reloaded.problems)


def test_missing_templates_make_an_enrolment_unusable(storage):
    record = _enrol(storage)
    (record.directory / "templates.npz").unlink()

    reloaded = storage.load_enrollment("Anam")
    assert reloaded.is_usable is False
    assert any("template" in problem for problem in reloaded.problems)


def test_an_unreadable_manifest_is_reported_as_corrupt(storage):
    record = _enrol(storage)
    (record.directory / "enrollment.json").write_text("{ this is not json")

    reloaded = storage.load_enrollment("Anam")
    assert reloaded.state is EnrollmentState.CORRUPT
    assert reloaded.is_usable is False


def test_re_enrolment_replaces_rather_than_accumulates(storage):
    """A replacement must not leave a mixture of old and new references."""
    _enrol(storage)
    storage.save_enrollment("Anam", _images(3), _templates(3))

    directory = storage.enrollment_dir("Anam")
    assert sorted(p.name for p in directory.glob("image_*.jpg")) == [
        "image_01.jpg",
        "image_02.jpg",
        "image_03.jpg",
    ]


# ---------------------------------------------------------------------------
# 4-6. Session directories
# ---------------------------------------------------------------------------


def test_a_session_directory_is_named_for_the_student_and_start_time(storage):
    directory = storage.session_dir("Anam")

    assert directory.exists()
    assert directory.parent == storage.sessions_root
    assert directory.name.startswith("Anam_")
    # Anam_YYYY-MM-DD_HH-MM-SS-mmm
    stamp = directory.name[len("Anam_") :]
    assert len(stamp) == len("2026-08-27_11-35-42-123")


def test_two_sessions_never_share_a_directory(storage):
    directories = [storage.session_dir("Anam") for _ in range(5)]

    assert len({d.name for d in directories}) == 5
    for directory in directories:
        assert directory.exists()


def test_a_new_session_does_not_overwrite_an_earlier_one(storage):
    first = storage.session_dir("Anam")
    (first / "events.json").write_text('[{"marker": "first"}]')

    second = storage.session_dir("Anam")
    (second / "events.json").write_text('[{"marker": "second"}]')

    assert first != second
    assert json.loads((first / "events.json").read_text())[0]["marker"] == "first"


def test_enrolment_images_are_untouched_by_running_sessions(storage):
    record = _enrol(storage)
    before = {p.name: p.read_bytes() for p in record.image_paths}

    for _ in range(3):
        session = storage.session_dir("Anam")
        (session / "events.json").write_text("[]")
        (session / "evidence").mkdir()

    after = storage.load_enrollment("Anam")
    assert after.state is EnrollmentState.VALID
    assert {p.name: p.read_bytes() for p in after.image_paths} == before


def test_enrolment_and_session_data_live_in_separate_trees(storage):
    record = _enrol(storage)
    session = storage.session_dir("Anam")

    assert storage.students_root in record.directory.parents
    assert storage.sessions_root in session.parents
    assert storage.students_root not in session.parents
    assert storage.sessions_root not in record.directory.parents


# ---------------------------------------------------------------------------
# 7. Reference reuse by identity verification
# ---------------------------------------------------------------------------


def test_stored_templates_load_back_unchanged_for_verification(storage):
    originals = _templates()
    storage.save_enrollment("Anam", _images(), originals)

    loaded = storage.load_templates("Anam")

    assert len(loaded) == len(originals)
    for stored, original in zip(loaded, originals, strict=True):
        assert np.allclose(stored, original)


def test_the_session_store_shares_the_single_identity_mechanism(tmp_path):
    """The service and the CLI must not write enrolments to two different places."""
    store = SessionStore(tmp_path / "data")

    assert store.save_enrolment("Anam", _templates(), images=_images()) is True

    shared = ProctoringStorage(tmp_path / "data")
    assert shared.is_enrolled("Anam") is True
    assert len(store.load_enrolment("Anam")) == ENROLLMENT_IMAGE_COUNT

    assert store.delete_enrolment("Anam") is True
    assert shared.is_enrolled("Anam") is False


def test_a_templates_only_enrolment_is_still_supported(tmp_path):
    """Hosts that hold no images must still be able to register an identity."""
    store = SessionStore(tmp_path / "data")
    assert store.save_enrolment("Anam", _templates()) is True
    assert len(store.load_enrolment("Anam")) == ENROLLMENT_IMAGE_COUNT


# ---------------------------------------------------------------------------
# 8-10. Temporary artefacts
# ---------------------------------------------------------------------------


def test_successful_enrolment_leaves_no_staged_copies(storage):
    for index, image in enumerate(_images(), start=1):
        storage.stage_image("Anam", index, image)
    assert storage.staging_dir("Anam").exists()

    record = _enrol(storage)

    assert not storage.staging_dir("Anam").exists()
    # Only the permanent set remains — no duplicates alongside it.
    remaining = sorted(p.name for p in record.directory.iterdir())
    assert remaining == [
        "enrollment.json",
        *[f"image_{i:02d}.jpg" for i in range(1, 6)],
        "templates.npz",
    ]


def test_a_cancelled_enrolment_cleans_up_its_staged_images(storage):
    for index, image in enumerate(_images(3), start=1):
        storage.stage_image("Anam", index, image)

    removed = storage.cleanup_staging("Anam")

    assert removed == 3
    assert not storage.staging_dir("Anam").exists()
    # Cancelling must not have created a half-enrolment.
    assert storage.load_enrollment("Anam").state is EnrollmentState.MISSING


def test_repeated_attempts_do_not_accumulate_staged_images(storage):
    for _ in range(3):
        for index, image in enumerate(_images(), start=1):
            storage.stage_image("Anam", index, image)
        storage.cleanup_staging("Anam")

    assert not storage.staging_dir("Anam").exists()


def test_purging_staging_leaves_permanent_enrolments_alone(storage):
    record = _enrol(storage)
    storage.stage_image("Anam", 1, _images(1)[0])
    storage.stage_image("Bilal", 1, _images(1)[0])

    removed = storage.purge_all_staging()

    assert removed == 2
    assert storage.load_enrollment("Anam").state is EnrollmentState.VALID
    assert len(list(record.directory.glob("image_*.jpg"))) == ENROLLMENT_IMAGE_COUNT


def test_cleanup_never_touches_finalised_session_evidence(storage):
    _enrol(storage)
    session = storage.session_dir("Anam")
    evidence = session / "evidence" / "frames"
    evidence.mkdir(parents=True)
    (evidence / "frame_000001.jpg").write_bytes(b"sealed evidence")
    (session / "manifest.json").write_text("{}")

    storage.purge_all_staging()
    storage.cleanup_staging("Anam")

    assert (evidence / "frame_000001.jpg").read_bytes() == b"sealed evidence"
    assert (session / "manifest.json").exists()


def test_cleanup_is_safe_when_nothing_was_staged(storage):
    assert storage.cleanup_staging("NeverSeen") == 0
    assert storage.purge_all_staging() == 0


def test_deleting_an_enrolment_removes_images_and_templates(storage):
    record = _enrol(storage)
    assert record.directory.exists()

    assert storage.delete_enrollment("Anam") is True
    assert not record.directory.exists()
    assert storage.delete_enrollment("Anam") is False


# ---------------------------------------------------------------------------
# 11. Filesystem safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    ["../../../etc/passwd", "..", "/absolute/path", "a/b/c", "", "   ", "..\\..\\windows"],
)
def test_hostile_student_names_stay_inside_the_data_root(storage, hostile):
    directory = storage.enrollment_dir(hostile)
    session = storage.session_dir(hostile)

    root = storage.data_root.resolve()
    assert root in directory.resolve().parents
    assert root in session.resolve().parents
    assert ".." not in directory.parts


def test_different_students_never_share_a_directory(storage):
    assert storage.enrollment_dir("Anam") != storage.enrollment_dir("Bilal")

    _enrol(storage, "Anam")
    _enrol(storage, "Bilal")

    assert storage.load_enrollment("Anam").state is EnrollmentState.VALID
    assert storage.load_enrollment("Bilal").state is EnrollmentState.VALID


def test_cleanup_refuses_to_delete_outside_the_data_root(storage, tmp_path):
    outside = tmp_path / "not_ours"
    outside.mkdir()

    with pytest.raises(ValueError, match="Refusing to remove"):
        storage._assert_inside_data_root(outside)


def test_session_names_are_filesystem_safe(storage):
    session = storage.session_dir("Anam O'Brien / Group 2")

    assert "/" not in session.name
    assert "'" not in session.name
    assert session.parent == storage.sessions_root
    assert session.exists()
