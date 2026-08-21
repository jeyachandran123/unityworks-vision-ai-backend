"""P1/P2 — in-memory raw source and passthrough decoder adapters.

These are the platform's *reference* acquisition adapters: fully deterministic,
dependency-free, and therefore usable in CI without a network, a codec library,
or a camera. They are the archival/discrete path of 01_LAYERED §5.3 and the basis
of deterministic replay (invariant V13).

An RTSP/WebRTC source and an NVDEC/QSV/VAAPI decoder are sibling adapters behind
the same ports. **No platform module changes to add them** — that is the whole
point of P1 and P2.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from ...core.errors import (
    ConnectFailedError,
    DecodeError,
    NotSeekableError,
    StreamLostError,
)
from ...core.model.camera import Camera, SourceSemantics
from ...core.model.frame import DecodeQuality, FrameDimensions
from ...core.model.timebase import Duration, Instant
from ...core.ports.acquisition import (
    DecodeOutcome,
    DecoderCapabilities,
    SourceCapabilities,
    SourcePacket,
)
from ...core.ports.buffer import WritableSlot
from ...core.ports.clock import Clock

RAW_CODEC = "raw_bgr24"


@dataclass(slots=True)
class RawFrameSpec:
    """One raw frame to be delivered by the in-memory source."""

    payload: bytes
    width: int
    height: int
    pts: int
    is_keyframe: bool = True
    wallclock_hint: Instant | None = None


class _RawHandle:
    __slots__ = ("_open", "_camera_id")

    def __init__(self, camera_id: str) -> None:
        self._open = True
        self._camera_id = camera_id

    @property
    def is_open(self) -> bool:
        return self._open

    async def close(self) -> None:
        self._open = False


class InMemoryRawSource:
    """Deliver a scripted sequence of raw frames.

    Args:
        frames: The sequence to deliver.
        semantics: ``ARCHIVAL`` (default) protects completeness and is
            reproducible; ``REALTIME`` permits dropping.
        loop: Repeat the sequence forever, for soak and cadence tests.
        fail_on_open: Raise ``ConnectFailedError`` this many times before
            succeeding, to exercise reconnect and backoff.
        fail_after: Raise ``StreamLostError`` after this many packets in each
            session, simulating a connection dropped mid-stream. This is the
            path that exercises epoch advance, since a *clean* end-of-stream
            correctly stops the actor instead of reconnecting.
        interpacket: Delay between packets on the injected clock.
    """

    def __init__(
        self,
        frames: Sequence[RawFrameSpec],
        *,
        clock: Clock,
        semantics: SourceSemantics = SourceSemantics.ARCHIVAL,
        loop: bool = False,
        fail_on_open: int = 0,
        fail_after: int = 0,
        interpacket: Duration | None = None,
        pts_timebase_hz: int = 1000,
    ) -> None:
        self._frames = list(frames)
        self._clock = clock
        self._semantics = semantics
        self._loop = loop
        self._remaining_open_failures = fail_on_open
        self._fail_after = fail_after
        self._interpacket = interpacket
        self._pts_timebase_hz = pts_timebase_hz
        self.open_calls = 0

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            semantics=self._semantics,
            codecs=(RAW_CODEC,),
            seekable=False,
            provides_wallclock=any(f.wallclock_hint is not None for f in self._frames),
        )

    async def open(self, camera: Camera, credential: str | None) -> _RawHandle:
        self.open_calls += 1
        if self._remaining_open_failures > 0:
            self._remaining_open_failures -= 1
            raise ConnectFailedError(
                f"synthetic connect failure for '{camera.camera_id}'",
                camera_id=str(camera.camera_id),
            )
        return _RawHandle(str(camera.camera_id))

    async def packets(self, handle: _RawHandle) -> AsyncIterator[SourcePacket]:
        emitted = 0
        while True:
            for spec in self._frames:
                if not handle.is_open:
                    return
                if self._fail_after and emitted >= self._fail_after:
                    raise StreamLostError(
                        f"synthetic mid-stream loss after {emitted} packets"
                    )
                if self._interpacket is not None:
                    await self._clock.sleep(self._interpacket)
                else:
                    await asyncio.sleep(0)
                if not handle.is_open:
                    return
                emitted += 1
                yield SourcePacket(
                    payload=spec.payload,
                    pts=spec.pts,
                    pts_timebase_hz=self._pts_timebase_hz,
                    is_keyframe=spec.is_keyframe,
                    codec=RAW_CODEC,
                    arrival=self._clock.now(),
                    wallclock_hint=spec.wallclock_hint,
                )
            if not self._loop:
                return

    async def seek(self, handle: _RawHandle, position: Duration) -> None:
        raise NotSeekableError("InMemoryRawSource does not support seeking")


class PassthroughDecoder:
    """Copy raw packet payload into the pooled slot.

    Zero-copy in spirit: the payload is written *into* buffer-pool memory rather
    than the decoder allocating its own image for the platform to copy.

    Args:
        dimensions: The frame geometry every packet decodes to.
        fail_every: Raise ``DecodeError`` on every Nth packet, to exercise the
            decode-error ladder. 0 disables.
        poison_payloads: Payloads that always fail — the *poison* failure class,
            which must be quarantined without stopping the stream.
    """

    def __init__(
        self,
        *,
        dimensions: FrameDimensions,
        fail_every: int = 0,
        poison_payloads: frozenset[bytes] = frozenset(),
    ) -> None:
        self._dimensions = dimensions
        self._fail_every = fail_every
        self._poison = poison_payloads
        self._count = 0
        self.reset_calls = 0

    def capabilities(self) -> DecoderCapabilities:
        return DecoderCapabilities(
            codecs=(RAW_CODEC,),
            hardware_accelerated=False,
            colour_space=self._dimensions.colour_space,
        )

    def decode_into(self, packet: SourcePacket, slot: WritableSlot) -> DecodeOutcome:
        self._count += 1
        if packet.payload in self._poison:
            raise DecodeError("poison payload", codec=packet.codec)
        if self._fail_every and self._count % self._fail_every == 0:
            raise DecodeError(f"synthetic decode failure at packet {self._count}")

        payload = packet.payload
        if len(payload) > slot.capacity:
            raise DecodeError(
                f"decoded frame ({len(payload)}B) exceeds slot capacity ({slot.capacity}B)"
            )
        memory = slot.memory()
        memory[: len(payload)] = payload
        return DecodeOutcome(
            dimensions=self._dimensions,
            bytes_written=len(payload),
            decode_quality=(
                DecodeQuality.KEYFRAME if packet.is_keyframe else DecodeQuality.DELTA
            ),
        )

    def reset(self) -> None:
        self.reset_calls += 1
        self._count = 0

    def close(self) -> None:
        return None
