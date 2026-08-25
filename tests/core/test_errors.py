"""Tests for failure handling, corrupt input boundaries, and error categories."""

import numpy as np

from proctoring.core.errors import ErrorCategory, PipelineErrorHandler


def test_corrupt_frame_validation():
    """Verify handler catches None, non-array, and empty frames without raising uncaught exceptions."""
    handler = PipelineErrorHandler(session_id="err_test")

    # 1. None frame
    ok, msg = handler.validate_frame(None, timestamp_seconds=0.0, frame_index=1)
    assert ok is False
    assert "None" in msg

    # 2. Non-array
    ok, msg = handler.validate_frame("not an image", timestamp_seconds=0.25, frame_index=2)
    assert ok is False

    # 3. 0-size array
    empty = np.array([])
    ok, msg = handler.validate_frame(empty, timestamp_seconds=0.50, frame_index=3)
    assert ok is False

    # 4. Valid image
    valid = np.full((100, 100, 3), 128, dtype=np.uint8)
    ok, msg = handler.validate_frame(valid, timestamp_seconds=0.75, frame_index=4)
    assert ok is True
    assert msg is None

    summary = handler.get_summary()
    assert summary["total_errors"] == 3
    assert summary["error_counts_by_category"].get(ErrorCategory.CORRUPT_FRAME.value) == 3
