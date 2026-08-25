"""LMS integration boundary — the stable surface a Moodle plugin talks to.

Everything a host application needs is here: a transport-agnostic service class,
versioned JSON contracts, and a pluggable session store.  Nothing below this
package should be imported by a host, so the pipeline stays free to change.

    from proctoring.integration import ProctoringService, StartSessionRequest

    service = ProctoringService(output_dir="/var/moodledata/proctoring")
    handle = service.start_session(StartSessionRequest(
        attempt_id="4471", user_id="82", candidate_name="A. Candidate",
        strictness="STRICT",
    ))
    ack = service.ingest_frame(handle.session_id, data_url_from_browser)
    result = service.finalize_session(handle.session_id)
    review = service.get_review(handle.session_id)
"""

from proctoring.integration.schemas import (
    CONTRACT_VERSION,
    DEFAULT_REVIEWER_GUIDANCE,
    RELIABILITY_NOTES,
    FrameAck,
    ObservationSummary,
    ReviewPayload,
    SessionHandle,
    SessionResult,
    SessionState,
    StartSessionRequest,
)
from proctoring.integration.service import ProctoringService, ProctoringServiceError
from proctoring.integration.session_store import SessionRecord, SessionStore

__all__ = [
    "CONTRACT_VERSION",
    "DEFAULT_REVIEWER_GUIDANCE",
    "RELIABILITY_NOTES",
    "FrameAck",
    "ObservationSummary",
    "ReviewPayload",
    "SessionHandle",
    "SessionResult",
    "SessionState",
    "StartSessionRequest",
    "ProctoringService",
    "ProctoringServiceError",
    "SessionRecord",
    "SessionStore",
]
