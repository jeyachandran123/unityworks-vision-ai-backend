"""P4 ClockSyncPort — capture-time estimation adapters.

An adapter that cannot do better must return ``ClockQuality.UNKNOWN`` rather than
a confident guess. A timestamp without honest uncertainty is a claim to precision
the system does not have, and cross-camera fusion built on it produces answers
that are wrong in ways nobody can detect (02_VOM §5.2).
"""

from __future__ import annotations

from ...core.model.timebase import ClockQuality, Duration, Instant
from ...core.ports.acquisition import CaptureEstimate, SourcePacket

#: Default transport latency assumed when no better signal exists.
_DEFAULT_LATENCY = Duration.from_millis(50)
_DEFAULT_UNCERTAINTY = Duration.from_millis(100)
_UNIX_EPOCH = Instant(0)
_PTS_UNCERTAINTY = Duration.from_millis(1)


class ArrivalTimeClockSync:
    """Estimate capture time as arrival minus a modelled transport latency.

    The weakest honest estimator: it reports ``ESTIMATED`` quality and an
    uncertainty wide enough to cover the latency it is guessing at.
    """

    __slots__ = ("_modelled_latency", "_uncertainty")

    def __init__(
        self,
        *,
        modelled_latency: Duration = _DEFAULT_LATENCY,
        uncertainty: Duration = _DEFAULT_UNCERTAINTY,
    ) -> None:
        self._modelled_latency = modelled_latency
        self._uncertainty = uncertainty

    def estimate(self, packet: SourcePacket, ingest: Instant) -> CaptureEstimate:
        return CaptureEstimate(
            t_capture=ingest.minus(self._modelled_latency),
            uncertainty=self._uncertainty,
            quality=ClockQuality.ESTIMATED,
        )

    def reset(self) -> None:
        return None


class WallclockHintClockSync:
    """Use an out-of-band capture timestamp when the transport supplies one.

    An RTSP RTCP sender report or drone telemetry timestamp lifts quality from
    ``ESTIMATED`` to ``RTCP_DERIVED``. Falls back honestly when the hint is
    absent rather than silently reusing a stale offset.
    """

    __slots__ = ("_fallback", "_quality", "_uncertainty")

    def __init__(
        self,
        *,
        quality: ClockQuality = ClockQuality.RTCP_DERIVED,
        uncertainty: Duration | None = None,
        fallback: ArrivalTimeClockSync | None = None,
    ) -> None:
        self._quality = quality
        self._uncertainty = uncertainty or Duration.from_millis(
            quality.typical_uncertainty_ms
        )
        self._fallback = fallback or ArrivalTimeClockSync()

    def estimate(self, packet: SourcePacket, ingest: Instant) -> CaptureEstimate:
        if packet.wallclock_hint is None:
            return self._fallback.estimate(packet, ingest)
        return CaptureEstimate(
            t_capture=packet.wallclock_hint,
            uncertainty=self._uncertainty,
            quality=self._quality,
        )

    def reset(self) -> None:
        self._fallback.reset()


class PtsClockSync:
    """Derive capture time from presentation timestamps against a fixed epoch.

    For archival sources, where "when photons arrived" is a property of the
    recording rather than of the network. Deterministic, which is what makes
    replay reproducible (invariant V13).
    """

    __slots__ = ("_epoch", "_uncertainty")

    def __init__(
        self,
        *,
        epoch: Instant = _UNIX_EPOCH,
        uncertainty: Duration = _PTS_UNCERTAINTY,
    ) -> None:
        self._epoch = epoch
        self._uncertainty = uncertainty

    def estimate(self, packet: SourcePacket, ingest: Instant) -> CaptureEstimate:
        offset_ns = int(packet.pts_seconds * 1_000_000_000)
        return CaptureEstimate(
            t_capture=Instant(self._epoch.ns + offset_ns),
            uncertainty=self._uncertainty,
            quality=ClockQuality.PTP_LOCKED,
        )

    def reset(self) -> None:
        return None


class UnknownClockSync:
    """No basis for a capture estimate.

    Reports ``UNKNOWN``, which the platform refuses to fuse across cameras. That
    refusal is the point: admitting ignorance is better than a confident wrong
    ordering.
    """

    __slots__ = ()

    def estimate(self, packet: SourcePacket, ingest: Instant) -> CaptureEstimate:
        return CaptureEstimate(
            t_capture=ingest,
            uncertainty=Duration.from_millis(ClockQuality.UNKNOWN.typical_uncertainty_ms),
            quality=ClockQuality.UNKNOWN,
        )

    def reset(self) -> None:
        return None
