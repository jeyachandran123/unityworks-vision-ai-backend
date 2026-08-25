"""The camera wall: live viewing, and nothing else.

### This path never touches Vision OS

A tile on a monitoring wall is a person watching a camera. It must not cost a
detection, a track, a crop or a model call, and nothing here can cause one —
there is no import of the perception stack in this module and no call into it.
Viewing and analysing are separate concerns that happen to share a DVR, and
coupling them would mean opening a wall tile started spending model budget.

### One authoritative session per camera

Sixteen tiles in four browsers must not become sixty-four RTSP connections to a
DVR that has sixteen channels. So each camera gets exactly one `LiveRtspSource`,
owned here, and every viewer reads the **latest frame** it produced. A second
viewer costs a JPEG encode, not a connection.

That also makes viewer count irrelevant to the DVR: it sees one client per
channel whether nobody or everybody is watching.

### Why MJPEG

The DVR speaks H.265. No browser decodes H.265 reliably in `<video>`, and the
alternatives all mean transcoding — a second media pipeline, a codec dependency,
and a per-camera transcoder process. The frames are *already* decoded to raw
pixels by the existing source, so encoding one JPEG and pushing it down a
`multipart/x-mixed-replace` response reuses what is there and needs nothing on
the client but an `<img>` tag.

The cost is bandwidth and CPU per encode, which is why the wall runs at a low
frame rate per tile and the detail view asks for a higher one. That trade is
explicit and configurable rather than hidden.

### Live is derived, never declared

A camera is `LIVE` because a frame arrived and is recent. Not because it is
enabled, not because a socket opened, and not because the UI would like it to
be. `STALE_AFTER_S` is what turns silence back into `RECONNECTING`.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

#: Frames per second encoded for a wall tile. Sixteen tiles at four fps is
#: sixty-four JPEG encodes a second; the wall is for noticing movement, not for
#: reading a clock.
DEFAULT_WALL_FPS = 4.0
#: The detail view gets more, because somebody is actually looking at it.
DEFAULT_DETAIL_FPS = 12.0
#: JPEG quality. Low enough to keep sixteen tiles affordable on a LAN.
DEFAULT_QUALITY = 70
#: No frame for this long and the camera is no longer live, whatever the socket
#: believes.
STALE_AFTER_S = 6.0
#: No frame for this long and the source is torn down and re-dialled. Matches
#: the platform source config's own `stall_watchdog_ms` (10s) rather than
#: inventing a second number for the same idea.
STALL_WATCHDOG_S = 10.0


class StreamState:
    """What a viewer is told. A closed set, and every value is earned."""

    DISABLED = "disabled"
    CONNECTING = "connecting"
    LIVE = "live"
    RECONNECTING = "reconnecting"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass(slots=True)
class CameraStreamStats:
    frames_decoded: int = 0
    frames_encoded: int = 0
    decode_errors: int = 0
    reconnects: int = 0
    stalls: int = 0
    viewers: int = 0
    first_frame_at: float | None = None
    last_frame_at: float | None = None
    last_error: str = ""
    width: int = 0
    height: int = 0
    source_fps: float = 0.0

    def to_wire(self) -> dict[str, Any]:
        return {
            "frames_decoded": self.frames_decoded,
            "frames_encoded": self.frames_encoded,
            "decode_errors": self.decode_errors,
            "reconnects": self.reconnects,
            "stalls": self.stalls,
            "viewers": self.viewers,
            "first_frame_at": self.first_frame_at,
            "last_frame_at": self.last_frame_at,
            "last_error": self.last_error,
            "width": self.width,
            "height": self.height,
            "source_fps": self.source_fps,
        }


class CameraStream:
    """One camera. One RTSP connection. Many viewers.

    ### Decode and encode run off the event loop

    RTSP receive, H.265 decode and JPEG encode are synchronous, CPU-bound
    Python calls. Phase 6B.2 measured what happens when they run as asyncio
    coroutines on the server's single event-loop thread: sixteen cameras
    converged on an identical, starved ~3.3 fps each, `/health` rose to
    seconds, and viewer count made no difference because the cost was never
    per-viewer — it was the always-on decode loop itself.

    Each camera instead gets one dedicated OS thread (`_thread`), running its
    own private asyncio loop via `asyncio.run()`. That reuses `LiveRtspSource`
    and its reconnect state machine completely unmodified — only *where* the
    loop runs changes. A private loop per thread also makes container
    ownership trivially safe: the `LiveRtspSource` and the PyAV container it
    opens are created, iterated and closed entirely on one thread and are
    never touched from another, so no operation is ever called concurrently
    on the same decoder (Phase 6B.3 §5's rule).

    The server's own event loop crosses into a camera's thread in exactly one
    place, `latest()`, guarded by `_condition` — the same design the wall
    already used for viewers before this phase, now also carrying the
    boundary between the two threads.
    """

    __slots__ = (
        "_condition",
        "_frame_seq",
        "_jpeg",
        "_last_publish_at",
        "_quality",
        "_secret_environment",
        "_state",
        "_stop_event",
        "_thread",
        "camera",
        "stats",
    )

    def __init__(
        self,
        camera: Any,
        *,
        quality: int = DEFAULT_QUALITY,
        secret_environment: dict[str, str] | None = None,
    ) -> None:
        self.camera = camera
        self._secret_environment = secret_environment
        self.stats = CameraStreamStats()
        self._quality = quality
        self._state = StreamState.DISABLED
        self._jpeg: bytes | None = None
        self._frame_seq = 0
        self._last_publish_at: float | None = None
        # Notifies waiting viewers that a new frame exists, so a tile pushes at
        # the camera's rate instead of polling on a timer. Also the one place
        # the server's event-loop thread and this camera's worker thread touch
        # the same memory.
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # -- observable state -----------------------------------------------------

    @property
    def state(self) -> str:
        """Derived from frame arrival, never from configuration."""
        if self._state in (StreamState.DISABLED, StreamState.ERROR, StreamState.OFFLINE):
            return self._state
        last = self.stats.last_frame_at
        if last is None:
            return StreamState.CONNECTING
        if (time.monotonic() - last) > STALE_AFTER_S:
            return StreamState.RECONNECTING
        return StreamState.LIVE

    def to_wire(self) -> dict[str, Any]:
        last = self.stats.last_frame_at
        return {
            "camera_id": self.camera.camera_key,
            "name": self.camera.name,
            "channel": self.camera.channel,
            "stream_type": self.camera.stream_type,
            "enabled": self.camera.enabled,
            "state": self.state,
            "width": self.stats.width,
            "height": self.stats.height,
            "source_fps": self.stats.source_fps,
            "viewers": self.stats.viewers,
            "reconnects": self.stats.reconnects,
            "frames_decoded": self.stats.frames_decoded,
            "seconds_since_frame": (
                round(time.monotonic() - last, 2) if last is not None else None
            ),
            "first_frame_latency_s": (
                round(self.stats.first_frame_at, 2)
                if self.stats.first_frame_at is not None
                else None
            ),
            "last_error": self.stats.last_error,
        }

    # -- lifecycle ------------------------------------------------------------

    async def start(self, config: Any, reconnect: Any) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._state = StreamState.CONNECTING
        self._thread = threading.Thread(
            target=self._thread_main,
            args=(config, reconnect),
            name=f"wall-{self.camera.camera_key}",
            # A worker that outlives an unclean shutdown must not keep the
            # process alive; the daemon flag is the backstop, not the plan —
            # `stop()` still joins every thread explicitly.
            daemon=True,
        )
        self._thread.start()

    async def stop(self) -> None:
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            # The join is genuinely blocking (an OS thread, not a coroutine),
            # so it runs on the default executor rather than the event loop —
            # the same pattern §10's responsiveness gate requires of the
            # viewer read path.
            await asyncio.to_thread(thread.join, 10.0)
            if thread.is_alive():
                logger.error(
                    "wall camera {} worker thread did not stop within 10s",
                    self.camera.camera_key,
                )
        self._state = StreamState.DISABLED
        # Release any viewer blocked waiting for a frame that will not come.
        with self._condition:
            self._condition.notify_all()

    def _thread_main(self, config: Any, reconnect: Any) -> None:
        """The camera's dedicated thread. Nothing here touches the server's
        event loop, and nothing on the server's event loop touches this
        thread's `LiveRtspSource` or PyAV container — only `latest()`, guarded
        by `_condition`, crosses between the two.
        """
        try:
            asyncio.run(self._run_async(config, reconnect))
        except Exception as exc:  # noqa: BLE001 - a worker thread must not die silently
            self.stats.last_error = f"{type(exc).__name__}: {exc}"
            self._state = StreamState.ERROR
            logger.error(
                "wall camera {} worker thread crashed: {}: {}",
                self.camera.camera_key,
                type(exc).__name__,
                exc,
            )

    async def _run_async(self, config: Any, reconnect: Any) -> None:
        """Decode this camera until told to stop. **One camera, one thread,
        one private loop.**

        This is `LiveRtspSource`, unmodified, driven exactly as it was when it
        ran on the shared server loop — the reconnect state machine, the
        backoff policy and the epoch bookkeeping are all reused verbatim.
        Failures are absorbed and retried here so that a camera going down is
        a tile changing colour rather than an exception reaching the wall.
        """
        from app.vision.secrets import EnvironmentSecretProvider
        from app.vision.sources.rtsp import LiveRtspSource

        # The deployment's configured secrets, layered over the process
        # environment. A bare `EnvironmentSecretProvider()` reads only
        # `os.environ`, which pydantic-settings never writes to — so a correct
        # password in `.env` produced sixteen cameras stuck at CONNECTING.
        secrets = EnvironmentSecretProvider(self._secret_environment)
        started_at = time.monotonic()

        while not self._stop_event.is_set():
            source = LiveRtspSource(config, secrets=secrets, reconnect=reconnect)
            watchdog = asyncio.create_task(self._watch_for_stall(source))
            try:
                async for frame in source.frames():
                    if self._stop_event.is_set():
                        break
                    self._publish(frame, started_at)
            except Exception as exc:  # noqa: BLE001 - one camera, not the wall
                self.stats.decode_errors += 1
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
                self._state = StreamState.RECONNECTING
                logger.warning(
                    "wall camera {} stream failed: {}: {}",
                    self.camera.camera_key,
                    type(exc).__name__,
                    exc,
                )
            finally:
                watchdog.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watchdog
                with contextlib.suppress(Exception):
                    source.stop()

            if self._stop_event.is_set():
                break
            # The source's own reconnect policy governs retries within a
            # session; this is the outer restart after it gave up. Slept in
            # small increments so a stop request lands in ~100ms rather than
            # waiting out the full 2s — this thread has nothing else to do,
            # so there is no cost to checking often.
            self.stats.reconnects += 1
            self._state = StreamState.RECONNECTING
            for _ in range(20):
                if self._stop_event.is_set():
                    break
                await asyncio.sleep(0.1)

    async def _watch_for_stall(self, source: Any) -> None:
        """Ask the source to stop when frames stop arriving.

        **Belt and braces, not the primary fix.** A read that blocks forever is
        bounded at the socket by FFmpeg's `timeout` option — see
        `app/vision/sources/rtsp.py`, where the pre-rename `stimeout` spelling
        was measured to be silently ignored and to leave a camera dark for 38
        minutes with zero reconnect attempts.

        This exists because that is one cause of a stall and not the only
        possible one: a source that returns frames the decoder cannot use, or a
        camera that goes quiet without the socket noticing, produces the same
        symptom and the same useless tile. Asking the source to stop makes the
        outer loop tear it down and dial again.

        It cannot run while the thread is blocked inside a synchronous decode —
        nothing on this loop can. It runs the moment control returns, which the
        socket timeout now guarantees happens.
        """
        # Checked several times per window rather than on a fixed 1s tick, so
        # the detection delay stays proportional to the window instead of
        # swamping a short one. Capped at a second because polling a ten-second
        # window faster than that buys nothing.
        interval = min(1.0, max(0.01, STALL_WATCHDOG_S / 4))

        while not self._stop_event.is_set():
            await asyncio.sleep(interval)
            last = self.stats.last_frame_at
            if last is None:
                # Still connecting. `first_frame` latency is a different
                # problem from a stall and has its own reporting.
                continue
            if (time.monotonic() - last) > STALL_WATCHDOG_S:
                self.stats.stalls += 1
                self.stats.last_error = (
                    f"stalled: no frame for {STALL_WATCHDOG_S:.0f}s"
                )
                self._state = StreamState.RECONNECTING
                logger.warning(
                    "wall camera {} stalled — no frame for {:.0f}s; "
                    "tearing the source down to reconnect",
                    self.camera.camera_key,
                    STALL_WATCHDOG_S,
                )
                with contextlib.suppress(Exception):
                    source.stop()
                return

    def _publish(self, frame: Any, started_at: float) -> None:
        """Encode the newest frame and wake every viewer."""
        payload = getattr(frame, "payload", None)
        width = int(getattr(frame, "width", 0) or 0)
        height = int(getattr(frame, "height", 0) or 0)
        if not payload or not width or not height:
            return

        try:
            jpeg = _encode_jpeg(payload, width, height, self._quality)
        except Exception as exc:  # noqa: BLE001 - a bad frame is not a dead camera
            self.stats.decode_errors += 1
            self.stats.last_error = f"encode: {type(exc).__name__}"
            return

        now = time.monotonic()
        self.stats.frames_decoded += 1
        self.stats.frames_encoded += 1
        self.stats.width, self.stats.height = width, height
        if self.stats.first_frame_at is None:
            self.stats.first_frame_at = now - started_at
        if self._last_publish_at is not None:
            interval = now - self._last_publish_at
            if interval > 0:
                instantaneous = 1.0 / interval
                # An EMA rather than the raw instantaneous value: single-frame
                # gaps (a GC pause, a slow encode) would otherwise make the
                # reported rate as noisy as the thing an operator is trying to
                # read past. alpha=0.3 settles in a handful of frames.
                self.stats.source_fps = (
                    instantaneous
                    if self.stats.source_fps <= 0
                    else 0.3 * instantaneous + 0.7 * self.stats.source_fps
                )
        self._last_publish_at = now
        self.stats.last_frame_at = now
        self._state = StreamState.LIVE

        with self._condition:
            self._jpeg = jpeg
            self._frame_seq += 1
            self._condition.notify_all()

    # -- viewing --------------------------------------------------------------

    def latest(self, after_seq: int, timeout: float) -> tuple[int, bytes | None]:
        """Block until a frame newer than `after_seq`, or time out.

        Waiting rather than polling means a viewer receives frames at the
        camera's rate without a timer, and a stalled camera costs a sleeping
        thread instead of a busy loop.
        """
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._frame_seq <= after_seq and not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._frame_seq, None
                self._condition.wait(remaining)
            return self._frame_seq, self._jpeg


def _encode_jpeg(payload: bytes, width: int, height: int, quality: int) -> bytes:
    """BGR bytes to JPEG. The only pixel work the viewing path does."""
    import io

    import numpy as np
    from PIL import Image

    array = np.frombuffer(payload, dtype=np.uint8)
    needed = width * height * 3
    if array.size < needed:
        raise ValueError(f"frame is {array.size} bytes, expected {needed}")
    # BGR to RGB without a copy of the whole buffer.
    image = Image.fromarray(array[:needed].reshape(height, width, 3)[:, :, ::-1])
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


class CameraWall:
    """Every camera stream for the process. Owns their lifecycle."""

    __slots__ = ("_lock", "_quality", "_secret_environment", "_settings", "_streams")

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._streams: dict[str, CameraStream] = {}
        self._lock = asyncio.Lock()
        self._quality = DEFAULT_QUALITY
        reader = getattr(settings, "secret_environment", None)
        self._secret_environment = reader() if callable(reader) else None

    @property
    def streams(self) -> dict[str, CameraStream]:
        return dict(self._streams)

    def get(self, camera_id: str) -> CameraStream | None:
        return self._streams.get(camera_id)

    async def start_cameras(self, cameras: list[Any]) -> int:
        """Open one stream per camera. **A failure starts the others anyway.**

        Section 15: one camera failing must never take the wall down, and that
        begins here — a camera whose configuration will not even build is
        recorded as ERROR and the loop continues.
        """
        from app.domain.cameras import to_rtsp_config
        from app.vision.sources.rtsp import ReconnectPolicy

        reconnect = ReconnectPolicy(
            initial_ms=self._settings.cctv_reconnect_initial_ms,
            max_ms=self._settings.cctv_reconnect_max_ms,
            max_attempts=self._settings.cctv_reconnect_max_attempts,
        )

        started = 0
        async with self._lock:
            for camera in cameras:
                if camera.camera_key in self._streams:
                    continue
                stream = CameraStream(
                    camera,
                    quality=self._quality,
                    secret_environment=self._secret_environment,
                )
                self._streams[camera.camera_key] = stream
                if not camera.enabled:
                    # Present on the wall, and honest about why it is dark.
                    continue
                try:
                    await stream.start(to_rtsp_config(camera), reconnect)
                    started += 1
                except Exception as exc:  # noqa: BLE001
                    stream._state = StreamState.ERROR  # noqa: SLF001
                    stream.stats.last_error = f"{type(exc).__name__}: {exc}"
                    logger.error(
                        "wall camera {} would not start: {}: {}",
                        camera.camera_key,
                        type(exc).__name__,
                        exc,
                    )
        logger.info("camera wall started {} of {} stream(s)", started, len(cameras))
        return started

    async def stop_all(self) -> None:
        async with self._lock:
            streams = list(self._streams.values())
            self._streams.clear()
        await asyncio.gather(*(s.stop() for s in streams), return_exceptions=True)
        logger.info("camera wall stopped {} stream(s)", len(streams))

    def summary(self) -> dict[str, Any]:
        by_state: dict[str, int] = {}
        for stream in self._streams.values():
            by_state[stream.state] = by_state.get(stream.state, 0) + 1
        return {
            "cameras": len(self._streams),
            "by_state": by_state,
            "live": by_state.get(StreamState.LIVE, 0),
            "viewers": sum(s.stats.viewers for s in self._streams.values()),
        }


__all__ = [
    "DEFAULT_DETAIL_FPS",
    "DEFAULT_QUALITY",
    "DEFAULT_WALL_FPS",
    "STALE_AFTER_S",
    "CameraStream",
    "CameraWall",
    "StreamState",
]
