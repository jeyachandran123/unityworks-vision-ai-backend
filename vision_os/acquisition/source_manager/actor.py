"""The per-source actor (03_MODULES M2, 08_RUNTIME §3).

**One connection actor per source.** Each source is a single-threaded logical
actor owning its socket, decoder session, and counters, so there are no locks on
the hot path and *a failing camera never affects another camera* — fault
containment as a structural consequence of the concurrency model rather than as
an aspiration.

The actor is the only place ``FrameSeq`` is assigned, which preserves strict
per-camera ordering even when decode is offloaded.
"""

from __future__ import annotations

import asyncio
import enum
import random
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ...core.errors import (
    ConnectFailedError,
    DecodeError,
    PoolExhaustedError,
    PrivacyMaskError,
    StreamLostError,
    UnsupportedCodecError,
)
from ...core.model.camera import Camera, CameraStatus
from ...core.model.frame import (
    Frame,
    FrameQuality,
    PrivacyState,
    SourceMeta,
)
from ...core.model.health import HealthState, ObservabilityReason
from ...core.model.ids import CameraId, FrameRef, FrameSeq, StreamEpoch
from ...core.model.timebase import ClockQuality, Duration, FrameTime
from ...core.ports.acquisition import (
    ClockSyncPort,
    DecoderPort,
    PrivacyMaskPort,
    SourceHandle,
    SourcePacket,
    SourcePort,
)
from ...core.ports.clock import Clock
from ...kernel.config.schema import SourceSection
from ...kernel.events import (
    ClockQualityChanged,
    DecodeFailed,
    EpochAdvanced,
    EventBus,
    MaskFailure,
    StreamConnected,
    StreamLost,
)
from ...kernel.health import HealthMonitor
from ...kernel.metrics import MetricName, MetricsEngine
from ..buffer import FrameBuffer
from .epoch import EpochAllocator

#: How long an archival source waits before re-attempting a full buffer.
_BACKPRESSURE_POLL = Duration.from_millis(2)


@runtime_checkable
class FrameSink(Protocol):
    """Where an actor hands off a published frame.

    Flow 1 wires this to the Frame Scheduler's admission path. Later flows
    replace the sink without the actor changing — the actor's responsibility ends
    at "a trustworthy frame exists".
    """

    async def __call__(self, frame: Frame) -> None: ...


class ActorState(enum.Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    BACKOFF = "backoff"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(slots=True)
class StreamStats:
    packets: int = 0
    frames_published: int = 0
    decode_errors: int = 0
    consecutive_decode_errors: int = 0
    mask_failures: int = 0
    reconnects: int = 0
    connect_failures: int = 0
    stalls: int = 0
    current_epoch: StreamEpoch = StreamEpoch(0)
    clock_quality: ClockQuality = ClockQuality.UNKNOWN
    last_packet_monotonic_ns: int = 0
    drops_by_reason: dict[str, int] = field(default_factory=dict)


class SourceActor:
    """Owns one camera's connection, decode session, and frame numbering."""

    def __init__(
        self,
        *,
        camera: Camera,
        clock: Clock,
        bus: EventBus,
        metrics: MetricsEngine,
        health: HealthMonitor,
        buffer: FrameBuffer,
        source: SourcePort,
        decoder: DecoderPort,
        privacy: PrivacyMaskPort,
        clock_sync: ClockSyncPort,
        epochs: EpochAllocator,
        config: SourceSection,
        credential: str | None,
        on_frame: FrameSink,
    ) -> None:
        self._camera = camera
        self._clock = clock
        self._bus = bus
        self._metrics = metrics
        self._health = health
        self._buffer = buffer
        self._source = source
        self._decoder = decoder
        self._privacy = privacy
        self._clock_sync = clock_sync
        self._epochs = epochs
        self._config = config
        self._credential = credential
        self._on_frame = on_frame

        self._state = ActorState.IDLE
        self._epoch = StreamEpoch(0)
        self._seq = 0
        self._stats = StreamStats()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._handle: SourceHandle | None = None
        self._connect_attempts = 0

    # --- lifecycle ---------------------------------------------------------- #

    @property
    def camera_id(self) -> CameraId:
        return self._camera.camera_id

    @property
    def state(self) -> ActorState:
        return self._state

    @property
    def stats(self) -> StreamStats:
        return self._stats

    def start(self) -> asyncio.Task[None]:
        if self._task is not None:
            return self._task
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name=f"uwv-source-{self.camera_id}")
        return self._task

    async def stop(self, timeout: Duration | None = None) -> None:
        self._state = ActorState.STOPPING
        self._stop.set()
        if self._handle is not None:
            try:
                await self._handle.close()
            except Exception:  # noqa: BLE001, S110 - shutdown must not raise
                pass
        if self._task is not None:
            seconds = timeout.seconds if timeout else 5.0
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=seconds)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        self._state = ActorState.STOPPED
        self._health.set_observability(
            self.camera_id,
            HealthState.DRAINING,
            ObservabilityReason.DRAINING,
            effective_rate=0.0,
            detail="actor stopped",
        )

    # --- main loop ----------------------------------------------------------- #

    async def _run(self) -> None:
        backoff_ms = float(self._config.reconnect_backoff_initial_ms)
        while not self._stop.is_set():
            try:
                await self._connect()
            except UnsupportedCodecError as exc:
                self._fail_persistent(f"unsupported codec: {exc.message}")
                return
            except ConnectFailedError as exc:
                self._connect_attempts += 1
                self._stats.connect_failures += 1
                self._metrics.counter(
                    MetricName.CONNECT_FAILURES, camera_id=str(self.camera_id)
                ).increment()
                self._publish_lost(f"connect failed: {exc.message}")
                if (
                    self._config.max_connect_attempts
                    and self._connect_attempts >= self._config.max_connect_attempts
                ):
                    self._fail_persistent(
                        f"giving up after {self._connect_attempts} connect attempts"
                    )
                    return
                backoff_ms = await self._backoff(backoff_ms)
                continue

            backoff_ms = float(self._config.reconnect_backoff_initial_ms)
            self._connect_attempts = 0

            try:
                ended_cleanly = await self._stream()
            except StreamLostError as exc:
                self._publish_lost(exc.message)
                ended_cleanly = False
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a source must never kill the platform
                self._publish_lost(f"unexpected source failure: {exc}")
                ended_cleanly = False
            finally:
                await self._close_handle()

            if ended_cleanly or self._stop.is_set():
                self._state = ActorState.STOPPED
                self._health.set_observability(
                    self.camera_id,
                    HealthState.DRAINING,
                    ObservabilityReason.DRAINING,
                    effective_rate=0.0,
                    detail="end of stream",
                )
                return

            self._stats.reconnects += 1
            self._metrics.counter(
                MetricName.RECONNECTS, camera_id=str(self.camera_id)
            ).increment()
            backoff_ms = await self._backoff(backoff_ms)

    async def _connect(self) -> None:
        self._state = ActorState.CONNECTING
        self._health.set_observability(
            self.camera_id,
            HealthState.STARTING,
            ObservabilityReason.STARTING_UP,
            effective_rate=0.0,
        )
        self._handle = await self._source.open(self._camera, self._credential)

        # Every (re)connection mints a new epoch and resets stateful collaborators,
        # so a reconnected stream never carries reference frames or clock offsets
        # across the discontinuity.
        self._epoch = self._epochs.next_epoch(self.camera_id)
        self._seq = 0
        self._stats.current_epoch = self._epoch
        self._decoder.reset()
        self._clock_sync.reset()

        self._state = ActorState.STREAMING
        now = self._clock.now()
        self._bus.publish(
            EpochAdvanced(
                occurred_at=now,
                partition_key=str(self.camera_id),
                camera_id=self.camera_id,
                stream_epoch=self._epoch,
            )
        )
        self._bus.publish(
            StreamConnected(
                occurred_at=now,
                partition_key=str(self.camera_id),
                camera_id=self.camera_id,
                stream_epoch=self._epoch,
            )
        )
        self._health.set_observability(
            self.camera_id, HealthState.HEALTHY, ObservabilityReason.NORMAL, effective_rate=1.0
        )

    async def _stream(self) -> bool:
        """Consume packets. Returns True on clean end-of-stream."""
        assert self._handle is not None
        self._stats.last_packet_monotonic_ns = self._clock.monotonic().ns
        watchdog = asyncio.create_task(self._watchdog(), name=f"uwv-watchdog-{self.camera_id}")
        try:
            async for packet in self._source.packets(self._handle):
                if self._stop.is_set():
                    return True
                self._stats.last_packet_monotonic_ns = self._clock.monotonic().ns
                self._stats.packets += 1
                self._metrics.counter(
                    MetricName.PACKETS_RECEIVED, camera_id=str(self.camera_id)
                ).increment()
                await self._handle_packet(packet)
            return True
        finally:
            watchdog.cancel()

    async def _watchdog(self) -> None:
        """Detect a stream that stalls with the socket still open.

        The most common real-world RTSP failure, and the one naive
        implementations miss entirely (03_MODULES M2).
        """
        interval = Duration.from_millis(max(1, self._config.stall_watchdog_ms // 2))
        threshold_ns = self._config.stall_watchdog_ms * 1_000_000
        while not self._stop.is_set():
            await self._clock.sleep(interval)
            idle = self._clock.monotonic().ns - self._stats.last_packet_monotonic_ns
            if idle > threshold_ns:
                self._stats.stalls += 1
                self._metrics.counter(
                    MetricName.STREAM_STALLS, camera_id=str(self.camera_id)
                ).increment()
                self._health.set_observability(
                    self.camera_id,
                    HealthState.BLIND,
                    ObservabilityReason.STREAM_DISCONNECTED,
                    effective_rate=0.0,
                    detail=f"no packets for {idle / 1_000_000:.0f}ms",
                )
                await self._close_handle()
                return

    # --- per-packet path ------------------------------------------------------ #

    async def _acquire_slot(self):
        """Take a buffer slot, honouring this source's backpressure semantics.

        ``realtime`` drops the frame and counts it — latency is protected over
        completeness. ``archival`` and ``discrete`` **wait** for capacity —
        completeness is protected over latency (01_LAYERED §5.3). A recorded
        stream must never lose frames merely because a downstream stage is
        briefly behind.
        """
        semantics = self._camera.source_semantics
        while not self._stop.is_set():
            try:
                return self._buffer.acquire_slot(self.camera_id, semantics)
            except PoolExhaustedError:
                self._record_drop("pool_pressure")
                if semantics.may_drop_frames:
                    self._metrics.counter(
                        MetricName.FRAMES_DROPPED,
                        camera_id=str(self.camera_id),
                        reason="pool_exhausted",
                    ).increment()
                    return None
                # Reclaim anything already past its retention horizon, then wait.
                self._buffer.sweep()
                await self._clock.sleep(_BACKPRESSURE_POLL)
        return None

    async def _handle_packet(self, packet: SourcePacket) -> None:
        ingest = self._clock.now()
        slot = await self._acquire_slot()
        if slot is None:
            return

        try:
            with self._metrics.timer(
                MetricName.DECODE_DURATION_MS, camera_id=str(self.camera_id)
            ):
                outcome = self._decoder.decode_into(packet, slot)
        except DecodeError as exc:
            self._buffer.discard_slot(slot)
            self._on_decode_error(exc.message)
            return
        except Exception as exc:  # noqa: BLE001 - adapters must not crash the actor
            self._buffer.discard_slot(slot)
            self._on_decode_error(f"decoder adapter raised {type(exc).__name__}: {exc}")
            return

        self._stats.consecutive_decode_errors = 0

        # --- privacy: the earliest point pixels exist, and the only fail-closed
        # --- path in the platform (12_SECURITY §2.1).
        try:
            with self._metrics.timer(
                MetricName.MASK_DURATION_MS, camera_id=str(self.camera_id)
            ):
                mask = self._privacy.apply(slot, outcome.dimensions)
        except PrivacyMaskError as exc:
            self._buffer.discard_slot(slot)
            self._on_mask_failure(exc.message)
            return
        except Exception as exc:  # noqa: BLE001 - any masking failure fails closed
            self._buffer.discard_slot(slot)
            self._on_mask_failure(f"privacy adapter raised {type(exc).__name__}: {exc}")
            return

        if not mask.state.emittable:
            self._buffer.discard_slot(slot)
            self._on_mask_failure("adapter reported MASK_FAILED")
            return

        frame = self._publish_frame(packet, slot, outcome, mask.state, ingest)
        self._metrics.counter(
            MetricName.FRAMES_RECEIVED, camera_id=str(self.camera_id)
        ).increment()
        self._metrics.histogram(
            MetricName.INGEST_LATENCY_MS, camera_id=str(self.camera_id)
        ).record(frame.time.ingest_latency.millis)
        await self._on_frame(frame)

    def _publish_frame(
        self,
        packet: SourcePacket,
        slot,  # FrameSlot
        outcome,  # DecodeOutcome
        privacy_state: PrivacyState,
        ingest,
    ) -> Frame:
        estimate = self._clock_sync.estimate(packet, ingest)
        if estimate.quality is not self._stats.clock_quality:
            self._stats.clock_quality = estimate.quality
            self._bus.publish(
                ClockQualityChanged(
                    occurred_at=self._clock.now(),
                    partition_key=str(self.camera_id),
                    camera_id=self.camera_id,
                    quality=estimate.quality.label,
                )
            )

        frame_ref = FrameRef(self.camera_id, self._epoch, FrameSeq(self._seq))
        self._seq += 1

        frame = self._buffer.publish(
            slot,
            frame_ref=frame_ref,
            time=FrameTime(
                pts=packet.pts,
                t_capture=estimate.t_capture,
                t_capture_uncertainty=estimate.uncertainty,
                t_ingest=ingest,
                t_decoded=self._clock.now(),
                clock_quality=estimate.quality,
            ),
            dimensions=outcome.dimensions,
            privacy_state=privacy_state,
            bytes_written=outcome.bytes_written,
            quality=FrameQuality(
                blur=outcome.blur,
                exposure=outcome.exposure,
                decode_quality=outcome.decode_quality,
            ),
            source_meta=SourceMeta(codec=packet.codec),
        )
        self._stats.frames_published += 1
        return frame

    # --- failure handling ------------------------------------------------------ #

    def _on_decode_error(self, reason: str) -> None:
        self._stats.decode_errors += 1
        self._stats.consecutive_decode_errors += 1
        self._metrics.counter(
            MetricName.DECODE_ERRORS, camera_id=str(self.camera_id)
        ).increment()
        self._bus.publish(
            DecodeFailed(
                occurred_at=self._clock.now(),
                partition_key=str(self.camera_id),
                camera_id=self.camera_id,
                reason=reason,
            )
        )
        if self._stats.consecutive_decode_errors >= self._config.max_consecutive_decode_errors:
            self._health.set_observability(
                self.camera_id,
                HealthState.DEGRADED,
                ObservabilityReason.DECODE_FAILING,
                effective_rate=0.0,
                detail=f"{self._stats.consecutive_decode_errors} consecutive decode errors",
            )

    def _on_mask_failure(self, reason: str) -> None:
        """Fails closed: the frame is dropped and the camera degrades.

        A masking failure that proceeds is a compliance incident regardless of
        intent, so this is the one path that stops rather than degrades content.
        """
        self._stats.mask_failures += 1
        self._metrics.counter(
            MetricName.MASK_FAILURES, camera_id=str(self.camera_id)
        ).increment()
        self._bus.publish(
            MaskFailure(
                occurred_at=self._clock.now(),
                partition_key=str(self.camera_id),
                camera_id=self.camera_id,
                reason=reason,
            )
        )
        self._health.set_observability(
            self.camera_id,
            HealthState.BLIND,
            ObservabilityReason.PRIVACY_MASK_FAILED,
            effective_rate=0.0,
            detail=reason,
        )

    def _publish_lost(self, reason: str) -> None:
        self._state = ActorState.BACKOFF
        self._bus.publish(
            StreamLost(
                occurred_at=self._clock.now(),
                partition_key=str(self.camera_id),
                camera_id=self.camera_id,
                stream_epoch=self._epoch,
                reason=reason,
            )
        )
        self._health.set_observability(
            self.camera_id,
            HealthState.BLIND,
            ObservabilityReason.STREAM_DISCONNECTED,
            effective_rate=0.0,
            detail=reason,
        )

    def _fail_persistent(self, reason: str) -> None:
        self._state = ActorState.FAILED
        self._health.set_observability(
            self.camera_id,
            HealthState.FAILED,
            ObservabilityReason.STREAM_DISCONNECTED,
            effective_rate=0.0,
            detail=reason,
        )
        self._bus.publish(
            StreamLost(
                occurred_at=self._clock.now(),
                partition_key=str(self.camera_id),
                camera_id=self.camera_id,
                stream_epoch=self._epoch,
                reason=reason,
            )
        )

    def _record_drop(self, reason: str) -> None:
        self._stats.drops_by_reason[reason] = self._stats.drops_by_reason.get(reason, 0) + 1

    async def _backoff(self, current_ms: float) -> float:
        jitter = 1.0 + random.uniform(  # noqa: S311 - jitter, not cryptography
            -self._config.reconnect_backoff_jitter, self._config.reconnect_backoff_jitter
        )
        await self._clock.sleep(Duration.from_millis(current_ms * jitter))
        return min(current_ms * 2.0, float(self._config.reconnect_backoff_max_ms))

    async def _close_handle(self) -> None:
        if self._handle is None:
            return
        try:
            await self._handle.close()
        except Exception:  # noqa: BLE001, S110 - close is best-effort
            pass
        finally:
            self._handle = None

    def status(self) -> CameraStatus:
        return {
            ActorState.IDLE: CameraStatus.PROVISIONED,
            ActorState.CONNECTING: CameraStatus.CONNECTING,
            ActorState.STREAMING: CameraStatus.STREAMING,
            ActorState.BACKOFF: CameraStatus.DEGRADED,
            ActorState.STOPPING: CameraStatus.DEGRADED,
            ActorState.STOPPED: CameraStatus.PROVISIONED,
            ActorState.FAILED: CameraStatus.BLIND,
        }[self._state]


__all__ = ["ActorState", "FrameSink", "SourceActor", "StreamStats"]
