"""The session — one processing boundary for every source.

    FrameSource ──► sampler ──► bounded queue ──► session loop ──► Vision OS
      (replay or live, identical from here on)

There is exactly one loop. `LiveSession` and `ReplaySession` differ in what they
*declare* — seekable, bounded, expected to end — and not in how a frame is
processed. A replay that took a different path would validate a system nobody
runs.

### Producer and consumer are separate tasks

The producer drains the source into the queue as fast as the source delivers.
The consumer takes frames and drives the pipeline. They are decoupled on purpose:
a slow model call must not stall the decoder, because a stalled decoder becomes a
dropped TCP connection and then a camera outage — a processing problem promoted
into a connection problem.

The queue between them is bounded and drops oldest, so the decoupling costs
memory that is capped rather than memory that grows.

### `started` is not `streaming`

A session that has started has a source and a loop. A session that is
**streaming** has received a genuine frame. Only the second sets `streaming` on
the WebSocket, and it is derived here rather than set anywhere by hand.
"""

from __future__ import annotations

import asyncio
import enum
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from app.vision.analysis_loop import ANALYSIS
from app.vision.frames import (
    DEFAULT_QUEUE_CAPACITY,
    DropReason,
    FrameSampler,
    LiveFrame,
    LiveFrameQueue,
)
from app.vision.ledger import FrameLedger, frame_ref_for
from app.vision.sources.base import CameraHealth, FrameSource, SourceKind, SourceState

#: Called for each admitted frame. Returns nothing; failures are logged and the
#: loop continues, because one bad frame must not end a night's monitoring.
FrameHandler = Callable[[LiveFrame], Awaitable[None]]


class SessionState(enum.Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"

    @property
    def is_active(self) -> bool:
        return self in (SessionState.RUNNING, SessionState.PAUSED)


@dataclass(slots=True)
class SessionStats:
    frames_received: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    processing_errors: int = 0
    first_frame_at_ns: int | None = None
    last_frame_at_ns: int | None = None
    total_processing_ms: float = 0.0

    @property
    def mean_processing_ms(self) -> float:
        return self.total_processing_ms / self.frames_processed if self.frames_processed else 0.0

    def to_wire(self) -> dict[str, Any]:
        return {
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "processing_errors": self.processing_errors,
            "first_frame_at_ns": self.first_frame_at_ns,
            "last_frame_at_ns": self.last_frame_at_ns,
            "mean_processing_ms": round(self.mean_processing_ms, 2),
        }


@dataclass(slots=True)
class SessionSpec:
    """What a session is, before it exists."""

    camera_id: str
    tenant_id: str
    site_id: str = ""
    queue_capacity: int = DEFAULT_QUEUE_CAPACITY
    analysis_fps: float = 4.0
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


class VisionSession:
    """Base session. Owns the loop, the queue, the sampler and the lifecycle."""

    #: Overridden by subclasses. Reported verbatim; a replay never says LIVE.
    kind: SourceKind = SourceKind.REPLAY
    #: A live stream cannot be scrubbed. Claiming otherwise would let the
    #: platform try to protect completeness it can never deliver.
    seekable: bool = False
    #: Whether the source is expected to end by itself.
    bounded: bool = True

    def __init__(
        self,
        spec: SessionSpec,
        source: FrameSource,
        *,
        handler: FrameHandler | None = None,
        ledger: FrameLedger | None = None,
    ) -> None:
        if source.camera_id != spec.camera_id:
            # Camera identity comes from configuration and must agree end to end.
            # A mismatch here silently merges two kitchens downstream.
            raise ValueError(
                f"source camera '{source.camera_id}' does not match session "
                f"camera '{spec.camera_id}'"
            )
        self.spec = spec
        self.source = source
        self._handler = handler
        # Observation only. The ledger never gates, delays or alters a frame —
        # it records that one existed. There is still exactly one processing
        # boundary, and this is not a second one.
        self._ledger = ledger
        self._queue = LiveFrameQueue(spec.queue_capacity)
        self._sampler = FrameSampler(spec.analysis_fps)
        self._state = SessionState.CREATED
        self._stats = SessionStats()
        self._producer: asyncio.Task[None] | None = None
        self._consumer: asyncio.Task[None] | None = None
        self._paused = False
        self._error = ""

    # ── observable state ─────────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        return self.spec.session_id

    @property
    def camera_id(self) -> str:
        return self.spec.camera_id

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def stats(self) -> SessionStats:
        return self._stats

    @property
    def streaming(self) -> bool:
        """**The** definition of streaming, for the whole system.

        Genuine frames have arrived *and* the source is still producing. Never
        true because a socket opened, never set by the frontend, and false the
        moment the source stops producing.
        """
        return self._state is SessionState.RUNNING and self.source.status.producing

    @property
    def health(self) -> CameraHealth:
        return self.source.status.health

    def to_wire(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "kind": self.kind.value,
            "camera_id": self.camera_id,
            "tenant_id": self.spec.tenant_id,
            "state": self._state.value,
            "streaming": self.streaming,
            "seekable": self.seekable,
            "bounded": self.bounded,
            "analysis_fps": self.spec.analysis_fps,
            "error": self._error,
            "source": self.source.status.to_wire(),
            "queue": self._queue.stats.to_wire(),
            "stats": self._stats.to_wire(),
        }

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Begin producing and consuming. Idempotent."""
        if self._state.is_active or self._state is SessionState.STARTING:
            return

        self._state = SessionState.STARTING
        self._error = ""
        self._producer = asyncio.create_task(self._produce(), name=f"produce-{self.session_id}")
        self._consumer = asyncio.create_task(self._consume(), name=f"consume-{self.session_id}")
        self._state = SessionState.RUNNING
        logger.info(
            "session {} started — camera={} kind={} analysis_fps={}",
            self.session_id,
            self.camera_id,
            self.kind.value,
            self.spec.analysis_fps,
        )

    def pause(self) -> None:
        """Stop *processing* while the source keeps running.

        Meaningful for live: the stream stays connected (reconnecting on resume
        would take longer than the pause) and frames continue to arrive and be
        dropped by the bounded queue, which is the correct outcome — a paused
        operator does not want a backlog of stale video when they resume.
        """
        if self._state is SessionState.RUNNING:
            self._paused = True
            self._state = SessionState.PAUSED

    def resume(self) -> None:
        if self._state is SessionState.PAUSED:
            self._paused = False
            self._sampler.reset()
            self._state = SessionState.RUNNING

    async def stop(self) -> None:
        """Stop cleanly. No orphan task, queue, or open source.

        Idempotent and safe to call from a shutdown handler that does not know
        whether the session ever started.
        """
        if self._state in (SessionState.STOPPED, SessionState.CREATED):
            self._state = SessionState.STOPPED
            return

        self._state = SessionState.STOPPING
        self.source.stop()
        self._queue.close()

        for task in (self._producer, self._consumer):
            if task is None or task.done():
                continue
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutdown
                pass

        await self.source.aclose()
        self._queue.clear()
        self._producer = None
        self._consumer = None
        self._state = SessionState.STOPPED
        logger.info(
            "session {} stopped — received={} processed={} dropped={}",
            self.session_id,
            self._stats.frames_received,
            self._stats.frames_processed,
            self._stats.frames_dropped,
        )

    # ── the loop ─────────────────────────────────────────────────────────────

    def _note_frame(self, frame: LiveFrame, elapsed_ms: float, *, error: str = "") -> None:
        """Mark a frame processed. Timing only — never a count.

        Counts belong to whatever actually ran the perception path, and it
        annotates them itself. Writing zeroes here would make every frame in an
        uninstrumented deployment look like a frame that genuinely saw nothing.
        """
        if self._ledger is None:
            return
        self._ledger.annotate(
            frame_ref_for(self.camera_id, frame.epoch, frame.sequence),
            processing_ms=elapsed_ms,
            error=error or None,
        )

    async def _produce(self) -> None:
        """Drain the source into the bounded queue, sampling on the way in."""
        try:
            async for frame in self.source.frames():
                self._stats.frames_received += 1

                # Sampling happens once, here. The pipeline never sees a frame it
                # was not asked to analyse, and nothing downstream samples again.
                if not self._sampler.accepts(frame.captured_at_ns):
                    self._queue.record_sampled_out()
                    self._stats.frames_dropped += 1
                    continue

                dropped = self._queue.put(frame)
                if dropped is DropReason.QUEUE_FULL:
                    self._stats.frames_dropped += 1
                    # Debug, not warning: under sustained load this is the policy
                    # working, and a warning per dropped frame would bury the
                    # events that do need attention.
                    logger.debug(
                        "camera {} dropped an old frame (queue full at {})",
                        self.camera_id,
                        self._queue.stats.capacity,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported as session state
            self._error = f"{type(exc).__name__}: {exc}"
            self._state = SessionState.FAILED
            logger.error("session {} producer failed: {}", self.session_id, self._error)
        finally:
            self._queue.close()

    async def _consume(self) -> None:
        """Take frames and drive the pipeline."""
        try:
            while True:
                frame = await self._queue.get()
                if frame is None:
                    break

                if self._paused:
                    # Drop rather than buffer. A paused session that queued
                    # everything would resume into a backlog of stale video.
                    self._stats.frames_dropped += 1
                    continue

                # Recorded *before* the handler runs, so a frame that crashes
                # the pipeline still appears in the timeline. A frame missing
                # from the ledger would look like one the source never emitted,
                # which is the opposite of what happened.
                if self._ledger is not None:
                    self._ledger.record(
                        camera_id=self.camera_id,
                        sequence=frame.sequence,
                        epoch=frame.epoch,
                        captured_at_ns=frame.captured_at_ns,
                        received_at_ns=frame.received_at_ns,
                        width=getattr(frame, "width", 0),
                        height=getattr(frame, "height", 0),
                        source_kind=self.kind.value,
                    )

                started = time.perf_counter()
                try:
                    if self._handler is not None:
                        # Off the API event loop. The handler is CPU-bound
                        # (ingest + YOLO) and awaiting it here is what took
                        # `/health` to 31s and `/auth/login` to 58s while four
                        # cameras ran. `ANALYSIS.run` executes it on one
                        # dedicated worker loop and awaits the result here, so
                        # ordering, identity and timestamps are unchanged —
                        # this session still finishes one frame before it
                        # starts the next.
                        await ANALYSIS.run(self._handler(frame))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - one frame, not the night
                    self._stats.processing_errors += 1
                    logger.warning(
                        "camera {} frame {} failed: {}: {}",
                        self.camera_id,
                        frame.sequence,
                        type(exc).__name__,
                        exc,
                    )
                    self._note_frame(
                        frame,
                        (time.perf_counter() - started) * 1000,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    continue

                elapsed_ms = (time.perf_counter() - started) * 1000
                self._note_frame(frame, elapsed_ms)
                self._stats.frames_processed += 1
                self._stats.total_processing_ms += elapsed_ms
                self._stats.last_frame_at_ns = frame.captured_at_ns
                if self._stats.first_frame_at_ns is None:
                    self._stats.first_frame_at_ns = frame.captured_at_ns
        except asyncio.CancelledError:
            raise
        finally:
            if self._state not in (SessionState.STOPPING, SessionState.STOPPED):
                self._state = SessionState.FAILED if self._error else SessionState.STOPPED


class LiveSession(VisionSession):
    """A camera. Unbounded, unseekable, expected to run for months."""

    kind = SourceKind.LIVE
    seekable = False
    bounded = False


class ReplaySession(VisionSession):
    """Recorded media. Finite and ends by itself — and **never labelled live**."""

    kind = SourceKind.REPLAY
    seekable = False
    bounded = True


def session_for(source: FrameSource, spec: SessionSpec, **kwargs) -> VisionSession:
    """Build the session that matches the source. One decision, one place."""
    if source.kind is SourceKind.LIVE:
        return LiveSession(spec, source, **kwargs)
    return ReplaySession(spec, source, **kwargs)


__all__ = [
    "FrameHandler",
    "LiveSession",
    "ReplaySession",
    "SessionSpec",
    "SessionState",
    "SessionStats",
    "VisionSession",
    "session_for",
    "SourceState",
]
