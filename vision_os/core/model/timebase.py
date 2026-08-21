"""The time model (02_VISION_OBJECT_MODEL §5).

Multi-camera perception is impossible without a rigorous, honest time model.
Most systems have one timestamp and quietly lie with it.

Two rules are load-bearing here:

* **Every instant carries its uncertainty.** A timestamp without uncertainty is
  a claim to precision the system does not have. With ``+/-20 ms`` an ordering
  across cameras is sound; with ``+/-800 ms`` (common for cheap RTSP over a
  congested network) it is unknowable, and a platform that returns an ordering
  anyway has manufactured a fact.
* **Duration is computed from capture time, never processing time.** A dwell of
  45 s means the object was present for 45 s in the world, regardless of whether
  the platform was keeping up.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ClockQuality(enum.Enum):
    """How much to trust ``t_capture`` (02_VOM §5.1).

    Ordered best-to-worst; ``typical_uncertainty_ms`` is the default assumption
    when a source cannot report better.
    """

    PTP_LOCKED = ("ptp_locked", 1)
    NTP_SYNCED = ("ntp_synced", 10)
    RTCP_DERIVED = ("rtcp_derived", 50)
    ESTIMATED = ("estimated", 500)
    UNKNOWN = ("unknown", 5_000)

    def __init__(self, label: str, typical_uncertainty_ms: int) -> None:
        self.label = label
        self.typical_uncertainty_ms = typical_uncertainty_ms

    @property
    def fusable(self) -> bool:
        """Whether observations under this quality may be fused across cameras.

        ``UNKNOWN`` never fuses: ``t_capture`` equals arrival time and carries no
        relationship to when photons actually arrived (02_VOM §5.1).
        """
        return self is not ClockQuality.UNKNOWN


@dataclass(frozen=True, slots=True, order=True)
class Instant:
    """A UTC instant in integer nanoseconds since the Unix epoch.

    Integer nanoseconds rather than float seconds: at 2026 epoch values a float64
    holds roughly microsecond resolution, which silently destroys sub-millisecond
    timing on PTP-locked deployments.
    """

    ns: int

    @classmethod
    def from_seconds(cls, seconds: float) -> Instant:
        return cls(int(seconds * 1_000_000_000))

    @classmethod
    def from_millis(cls, millis: int) -> Instant:
        return cls(millis * 1_000_000)

    @property
    def seconds(self) -> float:
        return self.ns / 1_000_000_000

    @property
    def millis(self) -> int:
        return self.ns // 1_000_000

    def plus(self, duration: Duration) -> Instant:
        return Instant(self.ns + duration.ns)

    def minus(self, duration: Duration) -> Instant:
        return Instant(self.ns - duration.ns)

    def since(self, earlier: Instant) -> Duration:
        return Duration(self.ns - earlier.ns)

    def __str__(self) -> str:
        return f"Instant({self.ns}ns)"


@dataclass(frozen=True, slots=True, order=True)
class Duration:
    """A signed duration in integer nanoseconds."""

    ns: int

    @classmethod
    def from_seconds(cls, seconds: float) -> Duration:
        return cls(int(seconds * 1_000_000_000))

    @classmethod
    def from_millis(cls, millis: float) -> Duration:
        return cls(int(millis * 1_000_000))

    @property
    def seconds(self) -> float:
        return self.ns / 1_000_000_000

    @property
    def millis(self) -> float:
        return self.ns / 1_000_000

    def __add__(self, other: Duration) -> Duration:
        return Duration(self.ns + other.ns)

    def __str__(self) -> str:
        return f"{self.millis:.3f}ms"


ZERO_DURATION = Duration(0)


@dataclass(frozen=True, slots=True)
class FrameTime:
    """The four timestamps every frame carries (02_VOM §5.1).

    Attributes:
        pts: Source presentation timestamp, monotonic within a stream epoch.
            Used for ordering *within* a camera.
        t_capture: Best estimate of when photons arrived. Used for ordering
            *across* cameras, and the sole basis for computing durations.
        t_capture_uncertainty: Mandatory. Never a default zero.
        t_ingest: When the platform received the frame.
        t_decoded: When the frame became pixels.
        clock_quality: The basis on which ``t_capture`` was derived.
    """

    pts: int
    t_capture: Instant
    t_capture_uncertainty: Duration
    t_ingest: Instant
    t_decoded: Instant
    clock_quality: ClockQuality

    def __post_init__(self) -> None:
        if self.t_capture_uncertainty.ns < 0:
            raise ValueError("t_capture_uncertainty must be non-negative")

    @property
    def ingest_latency(self) -> Duration:
        """How long between the estimated capture instant and platform receipt."""
        return self.t_ingest.since(self.t_capture)

    @property
    def decode_latency(self) -> Duration:
        return self.t_decoded.since(self.t_ingest)

    def may_fuse_with(self, other: FrameTime, phenomenon_scale: Duration) -> bool:
        """Whether two frames' timelines may be compared for ordering.

        The site layer refuses to fuse timelines whose combined uncertainty
        exceeds the phenomenon's timescale, and says so, rather than producing a
        confident wrong answer (02_VOM §5.2 rule 2).
        """
        if not (self.clock_quality.fusable and other.clock_quality.fusable):
            return False
        combined = self.t_capture_uncertainty.ns + other.t_capture_uncertainty.ns
        return combined <= phenomenon_scale.ns
