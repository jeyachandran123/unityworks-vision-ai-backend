"""Replay and synthetic sources — the development path, on the live boundary.

The restaurant's RTSP port is closed, so this is how the live runtime is
developed and tested. Two rules keep that honest:

**It enters the same session.** `ReplayFrameSource` is a `FrameSource` like
`LiveRtspSource`, feeds the same bounded queue, obeys the same sampler and drives
the same pipeline. A development path that bypassed any of that would validate a
system nobody is going to run.

**It is never called live.** `kind` is `REPLAY`, every status carries it, and the
frontend renders it as replay. Calling recorded footage "LIVE" is the single most
misleading thing this system could do.

### Paced, not dumped

Frames are emitted **over time**, at the source's declared rate, rather than
handed over as a list. That is what makes the session unbounded and the
backpressure real: a source that dumps 456 frames instantly tests a list, not a
stream.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from app.vision.frames import LiveFrame
from app.vision.sources.base import FrameSource, SourceKind


class SyntheticFrameSource(FrameSource):
    """Generated frames at a fixed rate. Deterministic, dependency-free.

    Used by tests that need continuous-source semantics without a codec. The
    pixels are meaningless; the **timing, ordering and boundedness** are the
    point.
    """

    def __init__(
        self,
        *,
        camera_id: str,
        width: int = 64,
        height: int = 64,
        fps: float = 25.0,
        count: int | None = None,
        interval_override_s: float | None = None,
    ) -> None:
        super().__init__(
            camera_id=camera_id,
            kind=SourceKind.REPLAY,
            redacted_uri=f"synthetic://{camera_id}?fps={fps}",
        )
        self._width = width
        self._height = height
        self._fps = fps
        #: `None` runs forever — the unbounded case a live camera actually is.
        self._count = count
        #: Lets a test run the producer far faster than real time without
        #: pretending the frame rate itself changed.
        self._interval = interval_override_s if interval_override_s is not None else 1.0 / fps

    async def _produce(self) -> AsyncIterator[LiveFrame]:
        pixel_bytes = self._width * self._height * 3
        capture_step_ns = int(1_000_000_000 / self._fps)
        # Capture time advances by the frame interval regardless of how fast the
        # producer actually runs. A test that emits 1,000 frames in a
        # millisecond still produces a timeline the sampler and freshness can
        # reason about.
        base_ns = time.time_ns()

        emitted = 0
        while self._count is None or emitted < self._count:
            if self._stopping:
                return

            captured_at = base_ns + emitted * capture_step_ns
            yield LiveFrame(
                camera_id=self.camera_id,
                sequence=emitted,
                epoch=self._status.epoch,
                captured_at_ns=captured_at,
                received_at_ns=time.time_ns(),
                width=self._width,
                height=self._height,
                payload=bytes([emitted % 251]) * pixel_bytes,
            )
            emitted += 1

            if self._interval > 0:
                await asyncio.sleep(self._interval)
            else:
                # Yield to the loop so a zero-interval producer cannot starve the
                # consumer — which is exactly the backpressure scenario, and it
                # must be observable rather than a hang.
                await asyncio.sleep(0)


class ReplayFrameSource(FrameSource):
    """A recorded video file, paced at its own frame rate.

    Decoding is PyAV. Absent PyAV the source reports a capability gap rather than
    silently producing nothing — "no decoder" and "no frames" are different
    facts, and only one of them is a configuration error.
    """

    def __init__(
        self,
        *,
        camera_id: str,
        path: Path | str,
        loop: bool = False,
        speed: float = 1.0,
    ) -> None:
        resolved = Path(path)
        super().__init__(
            camera_id=camera_id,
            kind=SourceKind.REPLAY,
            redacted_uri=f"file://{resolved.name}",
        )
        self._path = resolved
        #: Looping makes a finite file behave like an unbounded source, which is
        #: what a long-running development session needs. The epoch increments
        #: on each pass so tracking never associates across the seam.
        self._loop = loop
        self._speed = max(0.0, speed)

    async def _produce(self) -> AsyncIterator[LiveFrame]:
        if not self._path.is_file():
            raise FileNotFoundError(f"replay media not found: {self._path}")

        try:
            import av  # noqa: PLC0415 - optional, adapter-scoped
        except ImportError as exc:
            raise RuntimeError(
                "replay requires a decoder; install the 'video' extra (PyAV). "
                "Reported rather than degraded: a source that silently yields "
                "nothing is indistinguishable from a camera that sees nothing"
            ) from exc

        sequence = 0
        while True:
            with av.open(str(self._path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                time_base = float(stream.time_base or 0) or 1 / 25
                base_ns = time.time_ns()

                for decoded in container.decode(stream):
                    if self._stopping:
                        return

                    # Presentation timestamp from the container is the file's own
                    # capture timeline. Preferred over arrival time for exactly
                    # the reason live prefers it: freshness must age against when
                    # the picture was taken.
                    pts = decoded.pts
                    offset_ns = int((pts * time_base) * 1_000_000_000) if pts is not None else 0
                    image = decoded.to_ndarray(format="bgr24")

                    yield LiveFrame(
                        camera_id=self.camera_id,
                        sequence=sequence,
                        epoch=self._status.epoch,
                        captured_at_ns=base_ns + offset_ns,
                        received_at_ns=time.time_ns(),
                        width=int(image.shape[1]),
                        height=int(image.shape[0]),
                        payload=image.tobytes(),
                    )
                    sequence += 1

                    if self._speed > 0:
                        await asyncio.sleep(1.0 / (25.0 * self._speed))
                    else:
                        await asyncio.sleep(0)

            if not self._loop or self._stopping:
                return
            # A new pass is a new epoch: the same scene, but discontinuous in
            # time, and tracking must treat it as such.
            self._status.epoch += 1
            sequence = 0


def frames_from(source: FrameSource, limit: int) -> AsyncIterator[LiveFrame]:
    """First `limit` frames. A test helper, not a runtime path."""

    async def _bounded() -> AsyncIterator[LiveFrame]:
        taken = 0
        async for frame in source.frames():
            yield frame
            taken += 1
            if taken >= limit:
                source.stop()
                return

    return _bounded()


__all__ = ["ReplayFrameSource", "SyntheticFrameSource", "frames_from"]
