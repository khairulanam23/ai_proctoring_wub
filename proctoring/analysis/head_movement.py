"""Temporal patterns in head orientation: sustained turns, repeated glances, rapid movement.

A single threshold on yaw answers only one question — *is the head turned right
now* — and that is not the question a proctor has. A candidate who turns to answer
the door once and a candidate who checks the same spot to their left every twenty
seconds produce the same instantaneous reading, and the previous pipeline reported
both identically or, if each glance fell short of the qualification duration, not
at all.

This module keeps a short rolling window of pose readings and derives the patterns
that carry meaning:

* **Sustained deviation** — how long the current turn has been held.
* **Repeated looking away** — separate deviation episodes within the window.
* **Rapid repeated movement** — direction reversals above a velocity floor, which is
  what looking back and forth between the screen and something else looks like.

Nothing here decides anything about a candidate. It produces descriptive features
that the exam policy may turn into an observation, which the temporal aggregator
then qualifies and the evidence stage illustrates. The angular thresholds
themselves are **not** duplicated here: whether a given pose counts as deviating is
decided by :class:`~proctoring.analysis.policy.ExamPolicy`, which is what keeps
``PHYSICAL_PAPER`` mode honest about a candidate looking down at their paper.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class HeadMovementPattern:
    """Temporal description of head movement over the tracker's window."""

    deviating: bool = False
    """Whether the most recent reading is outside the permitted viewing zone."""

    sustained_seconds: float = 0.0
    """How long the current uninterrupted deviation has lasted. Zero when centred."""

    episode_count: int = 0
    """Completed deviation episodes within the window, each at least
    ``episode_min_seconds`` long. Momentary threshold crossings are not episodes."""

    rapid_reversals: int = 0
    """Direction changes within the window where the head was moving faster than the
    configured velocity floor on at least one side of the turn."""

    peak_velocity_deg_per_s: float = 0.0

    dominant_direction: str = "CENTRE"
    """``LEFT``, ``RIGHT``, ``UP``, ``DOWN`` or ``CENTRE`` — the axis and sign of the
    largest current deviation, for the observation text a proctor reads."""

    repeated_look_away: bool = False
    """Enough separate episodes in the window to describe the movement as repeated."""

    rapid_repeated_movement: bool = False
    """Enough fast reversals in the window to describe the movement as rapid."""

    @property
    def is_pattern(self) -> bool:
        """True when the movement forms a reportable pattern rather than one glance."""
        return self.repeated_look_away or self.rapid_repeated_movement

    def to_dict(self) -> dict[str, Any]:
        return {
            "deviating": self.deviating,
            "sustained_seconds": round(self.sustained_seconds, 2),
            "episode_count": self.episode_count,
            "rapid_reversals": self.rapid_reversals,
            "peak_velocity_deg_per_s": round(self.peak_velocity_deg_per_s, 1),
            "dominant_direction": self.dominant_direction,
            "repeated_look_away": self.repeated_look_away,
            "rapid_repeated_movement": self.rapid_repeated_movement,
        }


class HeadMovementTracker:
    """Rolling-window feature extractor over head pose readings.

    Stateful and session-scoped: one instance per session, fed frames in order.
    """

    _MAX_SAMPLES = 600
    """Hard cap on retained samples, so a caller passing a stalled or non-monotonic
    clock cannot grow the window without bound."""

    def __init__(
        self,
        window_seconds: float = 20.0,
        episode_min_seconds: float = 0.5,
        repeat_episode_count: int = 3,
        reversal_velocity_deg_per_s: float = 45.0,
        min_reversals: int = 4,
    ) -> None:
        self.window_seconds = max(1.0, float(window_seconds))
        self.episode_min_seconds = max(0.0, float(episode_min_seconds))
        self.repeat_episode_count = max(1, int(repeat_episode_count))
        self.reversal_velocity_deg_per_s = max(1.0, float(reversal_velocity_deg_per_s))
        self.min_reversals = max(1, int(min_reversals))

        # (timestamp, yaw, pitch, deviating)
        self._samples: deque[tuple[float, float, float, bool]] = deque(maxlen=self._MAX_SAMPLES)

    def reset(self) -> None:
        """Clear the window between sessions."""
        self._samples.clear()

    def update(
        self,
        timestamp_seconds: float,
        yaw: float,
        pitch: float,
        deviating: bool,
    ) -> HeadMovementPattern:
        """Add one pose reading and describe the movement across the window.

        ``deviating`` is supplied by the caller rather than computed here, so the
        exam policy remains the single place angular limits live and a paper exam's
        allowance for looking down is honoured automatically.
        """
        if self._samples and timestamp_seconds < self._samples[-1][0]:
            # A clock that went backwards means the frame source restarted; the old
            # window describes a different stretch of time and must not be mixed in.
            self._samples.clear()

        self._samples.append((float(timestamp_seconds), float(yaw), float(pitch), bool(deviating)))
        self._prune(timestamp_seconds)

        pattern = HeadMovementPattern(deviating=bool(deviating))
        pattern.dominant_direction = self._direction(yaw, pitch) if deviating else "CENTRE"
        pattern.sustained_seconds = self._sustained_seconds()

        episodes, open_episode = self._episodes()
        pattern.episode_count = episodes
        pattern.repeated_look_away = (
            episodes + (1 if open_episode else 0)
        ) >= self.repeat_episode_count

        reversals, peak = self._reversals()
        pattern.rapid_reversals = reversals
        pattern.peak_velocity_deg_per_s = peak
        pattern.rapid_repeated_movement = reversals >= self.min_reversals

        return pattern

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    @staticmethod
    def _direction(yaw: float, pitch: float) -> str:
        """Name the axis and sign of the larger deviation."""
        if abs(yaw) >= abs(pitch):
            return "LEFT" if yaw > 0 else "RIGHT"
        return "UP" if pitch > 0 else "DOWN"

    def _sustained_seconds(self) -> float:
        """Duration of the deviation run ending at the most recent sample."""
        if not self._samples or not self._samples[-1][3]:
            return 0.0
        end = self._samples[-1][0]
        start = end
        for timestamp, _yaw, _pitch, deviating in reversed(self._samples):
            if not deviating:
                break
            start = timestamp
        return max(0.0, end - start)

    def _episodes(self) -> tuple[int, bool]:
        """Count completed deviation episodes; report whether one is still open.

        An episode must span at least ``episode_min_seconds``. A head merely passing
        through a wide angle on its way somewhere else crosses the threshold for one
        or two frames, and counting that as a glance is how a fidgeting candidate
        turns into a pattern that was never there.
        """
        completed = 0
        run_start: float | None = None
        run_end: float | None = None
        open_episode = False

        for timestamp, _yaw, _pitch, deviating in self._samples:
            if deviating:
                if run_start is None:
                    run_start = timestamp
                run_end = timestamp
                continue
            if run_start is not None and run_end is not None:
                if (run_end - run_start) >= self.episode_min_seconds:
                    completed += 1
                run_start = run_end = None

        if run_start is not None and run_end is not None:
            open_episode = (run_end - run_start) >= self.episode_min_seconds

        return completed, open_episode

    def _reversals(self) -> tuple[int, float]:
        """Count fast direction changes in yaw, and report the peak angular speed.

        Only velocities above the floor are considered, and sign changes are counted
        across *that* filtered sequence. Slow postural sway is excluded by the floor
        rather than by being averaged away, and — importantly — the still moments
        between two fast movements no longer hide the reversal between them. Comparing
        raw consecutive velocities missed exactly that case: a head snapping left,
        pausing, then snapping right reads as fast-slow-slow-fast, and no adjacent
        pair has opposite signs even though the direction plainly reversed.
        """
        velocities: list[float] = []
        samples = list(self._samples)
        for (t0, yaw0, _p0, _d0), (t1, yaw1, _p1, _d1) in zip(samples, samples[1:], strict=False):
            dt = t1 - t0
            if dt <= 1e-6:
                continue
            velocities.append((yaw1 - yaw0) / dt)

        if not velocities:
            return 0, 0.0

        peak = max(abs(v) for v in velocities)
        fast = [v for v in velocities if abs(v) >= self.reversal_velocity_deg_per_s]
        reversals = sum(
            1 for previous, current in zip(fast, fast[1:], strict=False) if previous * current < 0
        )
        return reversals, peak
