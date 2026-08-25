"""AI examination proctoring pipeline.

Public entry point for the whole workflow.  A minimal offline session::

    from proctoring import ProctoringEngine, SessionConfig
    from proctoring.detection import FaceDetector

    engine = ProctoringEngine(
        config=SessionConfig(session_id="exam_001", student_name="A. Candidate"),
        face_detector=FaceDetector(),
    )
    summary = engine.run_video("exam_recording.mp4")
    print(summary.package_dir, summary.integrity_verified)

The engine records observations for a human proctor to review.  It does not
compute suspicion scores and does not decide whether misconduct occurred.
"""

from proctoring.config import SessionConfig
from proctoring.engine import (
    FaceStatus,
    FrameObservation,
    ProctoringEngine,
    SessionSummary,
)

__version__ = "1.0.0"

__all__ = [
    "SessionConfig",
    "ProctoringEngine",
    "FrameObservation",
    "FaceStatus",
    "SessionSummary",
    "__version__",
]
