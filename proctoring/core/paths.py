"""Filesystem-safety helpers for identifiers that arrive from outside the system.

Session, attempt and enrolment identifiers are supplied by the host LMS and end up
as path segments under the evidence directory.  Untrusted text reaching a path is
how candidate images and biometric templates get written somewhere they should not
be, so everything that builds a path from such a value funnels through here.

This module deliberately imports nothing from the rest of the package: it sits
below configuration, evidence and storage alike, all of which need it.
"""

import re
from pathlib import Path

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitise_identifier(value: str, fallback: str = "session") -> str:
    """Reduce an untrusted identifier to something safe to use as a path segment.

    Path separators and traversal sequences collapse to underscores, leading dots
    are stripped so nothing becomes a hidden file, and the result is length-capped
    to stay well inside filesystem limits.  An identifier that sanitises to nothing
    falls back rather than producing an empty path segment.
    """
    cleaned = _UNSAFE.sub("_", str(value)).strip("._")
    cleaned = re.sub(r"\.{2,}", "_", cleaned)
    return cleaned[:120] or fallback


def resolve_within(base: str | Path, *segments: str) -> Path:
    """Join ``segments`` under ``base``, refusing any result that escapes it.

    Belt and braces alongside :func:`sanitise_identifier`: even if a path is built
    by some other route, one that resolves outside the intended directory is
    rejected rather than written to.
    """
    base_path = Path(base).resolve()
    candidate = base_path.joinpath(*segments).resolve()
    if candidate != base_path and base_path not in candidate.parents:
        raise ValueError(f"Refusing to operate outside {base_path}: {candidate}")
    return candidate
