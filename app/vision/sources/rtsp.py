"""The live RTSP source.

Same boundary as replay, same queue, same sampler, same pipeline. What it adds is
a network, a credential, and the certainty that both will fail sometimes.

### The credential never becomes a string this module keeps

The password is resolved from a `SecretProvider` at connect time, used to build
one URL, handed to the decoder, and dropped. `redacted_uri` — the only URL that
is stored, logged, exported or returned — is built without it:

    rtsp://***:***@gayatri.freemyip.com:554/cam/realmonitor?channel=1&subtype=1

`_redact()` additionally scrubs the live value out of every exception message,
because decoder libraries habitually quote the URL they failed to open.

### Reconnection

Bounded exponential backoff, 1 s → 60 s, capped attempts. Each reconnect
increments the **epoch**, so a frame sequence never appears to continue across a
gap it did not survive. Reconnect is `RECONNECTING`, not `ERROR`: one is expected
to recover and the other needs a human, and an operator should not be paged for
the first.

Failures that retrying cannot fix — bad credentials, unknown stream path — go
straight to `ERROR`. Retrying a rejected password counts toward DVR account
lockout.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import quote

from app.vision.frames import LiveFrame
from app.vision.secrets import MissingSecretError, SecretProvider
from app.vision.sources.base import FrameSource, SourceKind, SourceState

REDACTED = "***"

#: Dahua's path for the XVR5116HS family. **Unverified against this device** —
#: TCP 554 has never been reachable, so no handshake has confirmed it. Kept in
#: one place so a single edit corrects every camera.
DAHUA_PATH = "/cam/realmonitor?channel={channel}&subtype={subtype}"

STREAM_SUBTYPE = {"main": 0, "sub": 1}


class RtspAuthenticationError(RuntimeError):
    """Credentials were rejected. **Not retried** — see the module docstring."""


class RtspStreamNotFoundError(RuntimeError):
    """The path or channel does not exist on this device. Retrying cannot help."""


@dataclass(frozen=True, slots=True)
class ReconnectPolicy:
    """Bounded. There is no setting that means "retry forever, immediately"."""

    initial_ms: float = 1_000.0
    multiplier: float = 2.0
    max_ms: float = 60_000.0
    #: 0 means retry indefinitely — with the delay still capped at `max_ms`, so
    #: an unreachable camera costs one attempt a minute rather than a spin loop.
    max_attempts: int = 0

    def delay_for(self, attempt: int) -> float:
        if attempt <= 1:
            return self.initial_ms
        # The exponent is clamped before it is used. `2.0 ** 9998` raised
        # OverflowError in Phase 7A, turning a patient reconnect into a crash.
        exponent = min(attempt - 1, 32)
        return min(self.initial_ms * (self.multiplier**exponent), self.max_ms)

    def should_retry(self, attempt: int) -> bool:
        return self.max_attempts <= 0 or attempt <= self.max_attempts


@dataclass(frozen=True, slots=True)
class RtspCameraConfig:
    """One camera, as configuration. **Holds a reference, never a password.**"""

    camera_id: str
    host: str
    channel: int = 1
    port: int = 554
    stream_type: str = "sub"
    username: str = ""
    #: `env:CCTV_PASSWORD`, `file:/run/secrets/dvr`, … Resolved at connect time.
    credential_ref: str = ""
    #: Independent of the camera's own frame rate: a 25 fps stream must not
    #: become 25 fps of detection and VLM work.
    analysis_fps: float = 4.0
    enabled: bool = True
    #: Overrides the Dahua default when a device wants a different path.
    path_template: str = DAHUA_PATH

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("a camera must have an id")
        if not self.host:
            raise ValueError("a camera must name a host")
        if self.channel < 1:
            raise ValueError("channel numbering starts at 1")
        if self.stream_type not in STREAM_SUBTYPE:
            raise ValueError(f"stream_type must be one of {sorted(STREAM_SUBTYPE)}")
        if self.analysis_fps <= 0:
            raise ValueError("analysis_fps must be positive")

    @property
    def subtype(self) -> int:
        return STREAM_SUBTYPE[self.stream_type]

    def path(self) -> str:
        return self.path_template.format(channel=self.channel, subtype=self.subtype)

    def redacted_uri(self) -> str:
        """Safe everywhere. Still diagnosable: host, port and path are visible."""
        credential = f"{REDACTED}:{REDACTED}@" if self.username or self.credential_ref else ""
        return f"rtsp://{credential}{self.host}:{self.port}{self.path()}"

    def dial_uri(self, password: str) -> str:
        """The real URL. Built at the moment of use and never stored."""
        user = quote(self.username, safe="")
        secret = quote(password, safe="")
        return f"rtsp://{user}:{secret}@{self.host}:{self.port}{self.path()}"


class LiveRtspSource(FrameSource):
    """A live camera. One instance per camera, so one failing never stops another."""

    def __init__(
        self,
        config: RtspCameraConfig,
        *,
        secrets: SecretProvider,
        reconnect: ReconnectPolicy | None = None,
        opener=None,
    ) -> None:
        super().__init__(
            camera_id=config.camera_id,
            kind=SourceKind.LIVE,
            redacted_uri=config.redacted_uri(),
        )
        self._config = config
        self._secrets = secrets
        self._reconnect = reconnect or ReconnectPolicy()
        #: Injected in tests. Production resolves PyAV lazily so the module
        #: imports on a host with no codec installed.
        self._opener = opener
        self._password: str | None = None

    @property
    def config(self) -> RtspCameraConfig:
        return self._config

    def _redact(self, text: str) -> str:
        """Scrub the live credential out of anything this source can emit.

        Decoder errors routinely quote the URL they failed to open, password
        included, and that string then reaches a log file and a status endpoint.
        """
        cleaned = text
        if self._password:
            cleaned = cleaned.replace(self._password, REDACTED)
            cleaned = cleaned.replace(quote(self._password, safe=""), REDACTED)
        if self._config.username:
            cleaned = cleaned.replace(f"{self._config.username}:", f"{REDACTED}:")
        return cleaned

    def _resolve_password(self) -> str:
        try:
            password = self._secrets.resolve(self._config.credential_ref)
        except MissingSecretError as exc:
            # Distinct from a rejected password. Nothing is broken; something is
            # absent, and dialling with an empty credential would count toward
            # DVR account lockout for no possible benefit.
            raise RtspAuthenticationError(
                f"no credential available for camera '{self._config.camera_id}': {exc}"
            ) from exc
        self._password = password
        return password

    async def _produce(self) -> AsyncIterator[LiveFrame]:
        attempt = 0

        while not self._stopping:
            password = self._resolve_password()
            dial = self._config.dial_uri(password)

            try:
                async for frame in self._stream(dial):
                    attempt = 0  # a delivered frame resets the backoff
                    yield frame

                if self._stopping:
                    return
                # The stream ended without an error. For a live camera that is a
                # disconnection, not an end of file.
                reason = "stream ended"

            except (RtspAuthenticationError, RtspStreamNotFoundError):
                # Retrying cannot fix either. Straight to ERROR.
                raise
            except Exception as exc:  # noqa: BLE001 - transport failures are expected
                reason = self._redact(f"{type(exc).__name__}: {exc}")
                self._record_error(reason)

            attempt += 1
            if not self._reconnect.should_retry(attempt):
                raise RuntimeError(f"giving up after {attempt - 1} reconnect attempts: {reason}")

            self._status.reconnects += 1
            # A new epoch: the sequence restarts, and tracking must not associate
            # across a gap the stream did not survive.
            self._status.epoch += 1
            self._transition(SourceState.RECONNECTING, reason)

            delay = self._reconnect.delay_for(attempt) / 1000.0
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return

    async def _stream(self, dial_uri: str) -> AsyncIterator[LiveFrame]:
        """One connection's worth of frames. Raises on transport failure."""
        opener = self._opener or _open_with_pyav
        container = await asyncio.to_thread(opener, dial_uri)

        sequence = 0
        try:
            for image, pts_ns in _iterate(container):
                if self._stopping:
                    return
                height, width = image.shape[0], image.shape[1]
                yield LiveFrame(
                    camera_id=self.camera_id,
                    sequence=sequence,
                    epoch=self._status.epoch,
                    # Capture time from the stream. Arrival time would make every
                    # observation look newer than it is (§11).
                    captured_at_ns=pts_ns if pts_ns is not None else time.time_ns(),
                    received_at_ns=time.time_ns(),
                    width=int(width),
                    height=int(height),
                    payload=image.tobytes(),
                )
                sequence += 1
                await asyncio.sleep(0)
        finally:
            close = getattr(container, "close", None)
            if callable(close):
                close()

    async def aclose(self) -> None:
        await super().aclose()
        # The resolved password does not outlive the source.
        self._password = None


def _open_with_pyav(uri: str):
    """Open an RTSP stream. Errors are classified, not just propagated."""
    try:
        import av  # noqa: PLC0415 - optional, adapter-scoped
    except ImportError as exc:
        raise RuntimeError(
            "live RTSP requires a decoder; install the 'video' extra (PyAV)"
        ) from exc

    try:
        return av.open(
            uri,
            options={
                # TCP, not UDP: a kitchen DVR behind NAT rarely delivers UDP, and
                # partial UDP frames become decode errors that look like a broken
                # camera.
                "rtsp_transport": "tcp",
                "stimeout": "10000000",  # 10 s, microseconds
                "max_delay": "500000",
            },
        )
    except Exception as exc:  # noqa: BLE001 - classified below
        message = str(exc).lower()
        if "401" in message or "unauthor" in message:
            raise RtspAuthenticationError("the DVR rejected these credentials") from exc
        if "404" in message or "not found" in message:
            raise RtspStreamNotFoundError("the DVR has no stream at this path or channel") from exc
        raise


def _iterate(container):
    """Decode frames to BGR24 with their presentation timestamps."""
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    time_base = float(stream.time_base or 0) or 1 / 25
    base_ns = time.time_ns()

    for decoded in container.decode(stream):
        pts = decoded.pts
        pts_ns = base_ns + int(pts * time_base * 1_000_000_000) if pts is not None else None
        yield decoded.to_ndarray(format="bgr24"), pts_ns


__all__ = [
    "DAHUA_PATH",
    "REDACTED",
    "LiveRtspSource",
    "ReconnectPolicy",
    "RtspAuthenticationError",
    "RtspCameraConfig",
    "RtspStreamNotFoundError",
]
