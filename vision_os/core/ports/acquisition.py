"""Acquisition ports — P1 Source, P2 Decoder, P3 PrivacyMask, P4 ClockSync.

Owner: M2 Video Source Manager (P1-P4), M1 Camera Manager (calibration policy).

These four ports are what make "RTSP today, WebRTC tomorrow, drone streams later"
an adapter change rather than a platform change. Drone and mobile sources arrive
here as ``SourcePort`` implementations plus time-varying calibration; no platform
module changes (03_MODULES M2 extension points).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..model.camera import Camera, SourceSemantics
from ..model.frame import DecodeQuality, FrameDimensions, PrivacyState
from ..model.ids import PrivacyPolicyId
from ..model.timebase import ClockQuality, Duration, Instant
from .buffer import WritableSlot

# --- P1 SourcePort ------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SourcePacket:
    """One unit delivered by a source, encoded or raw per ``codec``.

    ``wallclock_hint`` carries an out-of-band capture-time signal when the
    transport provides one (an RTSP RTCP sender report, a drone telemetry
    timestamp). Its presence is what lifts ``ClockQuality`` above ``ESTIMATED``.
    """

    payload: bytes
    pts: int
    pts_timebase_hz: int
    is_keyframe: bool
    codec: str
    arrival: Instant
    wallclock_hint: Instant | None = None
    sequence_hint: int | None = None

    @property
    def pts_seconds(self) -> float:
        if self.pts_timebase_hz <= 0:
            raise ValueError(f"invalid pts_timebase_hz: {self.pts_timebase_hz}")
        return self.pts / self.pts_timebase_hz


@dataclass(frozen=True, slots=True)
class SourceCapabilities:
    """Declared honestly — adapter obligation A1 (06_PORTS §3)."""

    semantics: SourceSemantics
    codecs: tuple[str, ...]
    seekable: bool = False
    provides_wallclock: bool = False
    max_bytes_per_packet: int = 8 * 1024 * 1024


@runtime_checkable
class SourceHandle(Protocol):
    """An open source. Owned by exactly one source actor."""

    @property
    def is_open(self) -> bool: ...

    async def close(self) -> None: ...


@runtime_checkable
class SourcePort(Protocol):
    """P1 — connect to a stream and deliver packets.

    Implementations: RTSP, RTMP, WebRTC, file, image directory, USB/CSI, ONVIF,
    VMS integration, cloud object storage, future drone and mobile uplinks.
    """

    def capabilities(self) -> SourceCapabilities: ...

    async def open(self, camera: Camera, credential: str | None) -> SourceHandle:
        """Open the stream.

        Args:
            credential: Already resolved from the secret provider. Adapters never
                read secrets themselves.

        Raises:
            ConnectFailedError: transient; the caller backs off and retries.
            UnsupportedCodecError: persistent; fail provisioning loudly.
        """
        ...

    def packets(self, handle: SourceHandle) -> AsyncIterator[SourcePacket]:
        """Yield packets until the stream ends or fails.

        Terminating normally means end-of-stream (archival). Raising
        ``StreamLostError`` means the connection died and should be retried.
        """
        ...

    async def seek(self, handle: SourceHandle, position: Duration) -> None:
        """Archival sources only. Raises ``NotSeekableError`` otherwise."""
        ...


# --- P2 DecoderPort ------------------------------------------------------ #


@dataclass(frozen=True, slots=True)
class DecodeOutcome:
    """Result of decoding one packet into a pooled slot."""

    dimensions: FrameDimensions
    bytes_written: int
    decode_quality: DecodeQuality
    blur: float = 0.0
    exposure: str = "ok"


@dataclass(frozen=True, slots=True)
class DecoderCapabilities:
    codecs: tuple[str, ...]
    hardware_accelerated: bool
    max_width: int = 7680
    max_height: int = 4320
    colour_space: str = "bgr24"


@runtime_checkable
class DecoderPort(Protocol):
    """P2 — turn packets into pixels, writing into pooled memory.

    Implementations: NVDEC, QSV, VAAPI, software, passthrough (for sources that
    already deliver raw frames).
    """

    def capabilities(self) -> DecoderCapabilities: ...

    def decode_into(self, packet: SourcePacket, slot: WritableSlot) -> DecodeOutcome:
        """Decode ``packet`` into ``slot``.

        Raises:
            DecodeError: transient; the frame is dropped and counted, and the
                stream continues. Adapters never return a fabricated image.
        """
        ...

    def reset(self) -> None:
        """Discard decoder state. Called on epoch advance so that a reconnected
        stream never carries reference frames across the discontinuity."""
        ...

    def close(self) -> None: ...


# --- P3 PrivacyMaskPort -------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MaskOutcome:
    state: PrivacyState
    regions_masked: int = 0


@runtime_checkable
class PrivacyMaskPort(Protocol):
    """P3 — apply privacy policy at the earliest point pixels exist.

    Applied in-place on the slot, immediately post-decode and before the frame is
    published, so that **no component ever sees unmasked pixels** (12_SECURITY
    §2.1). Implementations: static polygon mask, detection-driven face/plate
    blur, full-frame encryption for regulated sites.
    """

    @property
    def policy_id(self) -> PrivacyPolicyId | None:
        """``None`` means no masking policy is configured for this camera."""
        ...

    def apply(self, slot: WritableSlot, dimensions: FrameDimensions) -> MaskOutcome:
        """Mask in place.

        Raises:
            PrivacyMaskError: the frame must be dropped, never emitted. This is
                the platform's only fail-closed path.
        """
        ...


# --- P4 ClockSyncPort ---------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CaptureEstimate:
    """An honest estimate of when photons arrived, with its uncertainty."""

    t_capture: Instant
    uncertainty: Duration
    quality: ClockQuality


@runtime_checkable
class ClockSyncPort(Protocol):
    """P4 — derive ``t_capture`` and its uncertainty from available signals.

    Implementations: PTP, NTP + RTCP sender reports, arrival-time estimation with
    a modelled latency. An adapter that cannot do better must return
    ``ClockQuality.UNKNOWN`` rather than a confident guess (02_VOM §5.2).
    """

    def estimate(self, packet: SourcePacket, ingest: Instant) -> CaptureEstimate: ...

    def reset(self) -> None:
        """Discard offset state on epoch advance."""
        ...
