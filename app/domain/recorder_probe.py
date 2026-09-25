"""A connection test that says *which* thing is wrong.

`app.domain.connectivity.probe` opens a TCP socket and closes it, and it is
honest about what that proves: something is listening. It cannot tell a wrong
password from a wrong channel from a recorder that accepts the connection and
never sends a picture, and those three need three different fixes.

This test goes the rest of the way. It asks for one decoded frame through the
**same code the runtime uses to dial** — `RtspCameraConfig.dial_uri` and the
PyAV opener that already classifies an RTSP 401 and 404 — so a test that passes
means the camera will start, and a test that fails fails for the reason the
camera would.

### Why resolving the password here is acceptable now

`connectivity.py` explains why it would not resolve a credential for a
diagnostic: the value could end up in a log or an error body. Two things are
different here. The password is either the one the person just typed into this
same request, or the recorder's own sealed one, which the runtime is about to
use for the same connection anyway. And the value never leaves this module:
the URL that carries it is built and consumed inside one function, every
exception message is scrubbed of it before it is logged, and the result carries
an outcome and a sentence, never a URL or an error string from the decoder.

### Three steps, each answering a different question

1. **Is anything listening?** A TCP connect. Wrong address, closed port,
   switched-off recorder — answered in seconds.
2. **Is it a video port?** An RTSP `OPTIONS` request, **with no credentials**.
   Recorders listen on several ports — video, web page, phone app — and the one
   printed in a recorder's phone app is frequently not the video port. Every
   RTSP server answers `OPTIONS`, even with "401 Unauthorized"; a port that
   stays silent or answers in another protocol is not the one to stream from.
   Sending no credentials is deliberate: this step can never count as a failed
   sign-in, so it cannot contribute to an account lockout.
3. **Does a picture arrive?** Signed in, through the runtime's own dial path.

### Outcomes

A fixed set, each with one sentence written for the person looking at the
screen and one hint about what to check. The decoder's own error text goes to
the server log, scrubbed, and nowhere else.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from loguru import logger

from app.domain import connectivity
from app.domain.recorder_brands import Brand
from app.vision.sources.rtsp import (
    RtspAuthenticationError,
    RtspCameraConfig,
    RtspStreamNotFoundError,
)

#: Long enough for a recorder on a slow uplink to send its first keyframe, short
#: enough that somebody watching a spinner does not give up first.
FRAME_TIMEOUT_S = 15.0

#: How long the unauthenticated RTSP question may take. A real RTSP server
#: answers `OPTIONS` at once; this only has to outlast a slow uplink.
HANDSHAKE_TIMEOUT_S = 6.0

#: `(outcome) -> (ok, message, hint)`. `{address}` and `{port}` are filled in
#: with what was tested, which the person typed or chose.
OUTCOMES: dict[str, tuple[bool, str, str]] = {
    "connected": (
        True,
        "Recorder connected successfully.",
        "",
    ),
    "authentication_failed": (
        False,
        "The recorder rejected the username or password.",
        "Check the username and password you use to sign in to the recorder itself.",
    ),
    "channel_not_found": (
        False,
        "The recorder is reachable, but this channel could not be opened.",
        "Check the channel number, and that a camera is plugged into it.",
    ),
    "stream_unavailable": (
        False,
        "The recorder answered, but no picture came back from this channel.",
        "Check that the camera on this channel is powered and showing a picture on the recorder.",
    ),
    "timeout": (
        False,
        "The recorder could not be reached.",
        "Check the address and port, and that the recorder is switched on and on the network.",
    ),
    "unreachable": (
        False,
        "The recorder is unreachable from this server.",
        "Check the address and port. A firewall or router may be blocking this server.",
    ),
    "refused": (
        False,
        "The recorder is on the network but refused the connection on this port.",
        "Check the port — RTSP is usually 554 — and that RTSP is enabled on the recorder.",
    ),
    "address_not_found": (
        False,
        "That address could not be found.",
        "Check the address for a typo.",
    ),
    "not_rtsp": (
        False,
        "Something answered on port {port}, but it is not the recorder's video (RTSP) port.",
        "Recorders use different ports for video, their web page and their phone app, and "
        "the port shown in the phone app is often not the video one. The video port is "
        "usually 554. If you connect over the internet, that port must also be forwarded "
        "on the site's router.",
    ),
    "decoder_unavailable": (
        False,
        "This server cannot decode video, so it cannot test the stream.",
        "Whoever runs the server needs to install its video support (the 'av' package).",
    ),
}


#: Failures that mean "nothing answered at that address".
_NETWORK_FAILURES = frozenset({"timeout", "unreachable", "refused"})

_LOCAL_ADDRESS_HINT = (
    "{address} is a local network address. It works only when this server is on the same "
    "network as the recorder. From anywhere else, connect using the recorder's domain."
)


def _is_local(address: str) -> bool:
    try:
        return ipaddress.IPv4Address(address).is_private
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class RecorderTestResult:
    """What a connection test learned, in words for a person."""

    outcome: str
    channel: int
    elapsed_ms: int
    width: int | None = None
    height: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    #: What was tested. Values the person typed or chose; never a URL.
    address: str = ""
    port: int = 0

    @property
    def ok(self) -> bool:
        return OUTCOMES[self.outcome][0]

    def as_dict(self) -> dict[str, Any]:
        ok, message, hint = OUTCOMES[self.outcome]
        # The most useful thing to say about a private address nobody reached
        # is that this server is probably not on that network.
        if self.outcome in _NETWORK_FAILURES and _is_local(self.address):
            hint = _LOCAL_ADDRESS_HINT
        values = {"address": self.address or "this address", "port": self.port or "this"}
        return {
            "ok": ok,
            "outcome": self.outcome,
            "message": message.format(**values),
            "hint": hint.format(**values),
            "address": self.address,
            "port": self.port,
            "channel": self.channel,
            "elapsed_ms": self.elapsed_ms,
            # Stated when known, because "1920 × 1080 arrived" is the most
            # convincing thing a test can say.
            "resolution": (f"{self.width} × {self.height}" if self.width and self.height else None),
        }


#: `(uri) -> (width, height)`. Replaced in tests; production opens the stream
#: with PyAV and decodes one frame.
FrameReader = Callable[[str], tuple[int, int]]


def _first_frame_with_pyav(uri: str) -> tuple[int, int]:
    """Open the stream exactly as the runtime does, and decode one frame."""
    from app.vision.sources.rtsp import _open_with_pyav  # noqa: PLC0415 - the runtime's own

    container = _open_with_pyav(uri)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        for frame in container.decode(stream):
            return int(frame.width), int(frame.height)
        raise RuntimeError("the stream ended before sending a picture")
    finally:
        close = getattr(container, "close", None)
        if callable(close):
            close()


async def _rtsp_handshake(host: str, port: int, *, timeout_s: float = HANDSHAKE_TIMEOUT_S) -> str:
    """Ask the port whether it speaks RTSP. **Sends no credentials.**

    Returns `rtsp` (it answered as an RTSP server, whatever the status), `other`
    (it answered in another protocol), `silent` (connected, never answered) or
    `closed` (the connection failed or ended without a word — not conclusive,
    so the caller carries on and lets the decoder decide).
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, int(port)), timeout=timeout_s
        )
    except (OSError, TimeoutError):
        return "closed"
    try:
        request = (
            f"OPTIONS rtsp://{host}:{int(port)}/ RTSP/1.0\r\n"
            "CSeq: 1\r\nUser-Agent: UnityWorks-Vision\r\n\r\n"
        )
        writer.write(request.encode("ascii"))
        await writer.drain()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        head = b""
        while len(head) < 5:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return "silent" if not head else "other"
            try:
                chunk = await asyncio.wait_for(reader.read(64), timeout=remaining)
            except TimeoutError:
                return "silent" if not head else "other"
            if not chunk:
                return "closed" if not head else "other"
            head += chunk
        return "rtsp" if head.startswith(b"RTSP/") else "other"
    except OSError:
        return "closed"
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


async def check_connection(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    brand: Brand,
    channel: int = 1,
    stream_type: str = "sub",
    reader: FrameReader | None = None,
    frame_timeout_s: float = FRAME_TIMEOUT_S,
) -> RecorderTestResult:
    """Try to receive one picture from one channel of a recorder.

    Raises `ValidationError` only for an address that must never be dialled
    (loopback, link-local and the like — see `connectivity`). Every network or
    device failure is a result, not an exception: the caller asked whether it
    works, and "no, because the password was rejected" is the useful answer.
    """
    started = time.monotonic()
    tested = {"address": host, "port": int(port)}

    def elapsed() -> int:
        return int((time.monotonic() - started) * 1000)

    # Step one is the cheap one. It refuses reserved addresses before anything
    # is dialled, and turns the common failures — wrong address, wrong port,
    # recorder switched off — into an answer in a few seconds rather than after
    # the full decoder timeout.
    reached = await connectivity.probe(host, int(port))
    if not reached.reachable:
        outcome = {
            "timeout": "timeout",
            "dns_timeout": "timeout",
            "dns_failed": "address_not_found",
            "refused": "refused",
        }.get(reached.outcome, "unreachable")
        return RecorderTestResult(outcome, channel, elapsed(), **tested)

    # Step two: is this the video port at all? Answered without a password, so
    # a wrong port never becomes a failed sign-in on the recorder.
    spoken = await _rtsp_handshake(host, int(port))
    if spoken in {"other", "silent"}:
        return RecorderTestResult("not_rtsp", channel, elapsed(), **tested)

    config = RtspCameraConfig(
        camera_id="connection-test",
        host=host,
        channel=int(channel),
        port=int(port),
        stream_type=stream_type,
        username=username,
        credential_ref="(connection test)",
        path_template=brand.path_template,
        stream_values=(brand.main, brand.sub),
    )
    read = reader or _first_frame_with_pyav

    try:
        width, height = await asyncio.wait_for(
            asyncio.to_thread(read, config.dial_uri(password)),
            timeout=frame_timeout_s,
        )
    except RtspAuthenticationError:
        return RecorderTestResult("authentication_failed", channel, elapsed(), **tested)
    except RtspStreamNotFoundError:
        return RecorderTestResult("channel_not_found", channel, elapsed(), **tested)
    except TimeoutError:
        return RecorderTestResult("stream_unavailable", channel, elapsed(), **tested)
    except Exception as exc:  # noqa: BLE001 - classified, scrubbed, logged
        text = _scrub(f"{type(exc).__name__}: {exc}", username=username, password=password)
        if "decoder" in text.lower() and "install" in text.lower():
            return RecorderTestResult("decoder_unavailable", channel, elapsed(), **tested)
        # The decoder's own words go to the server log only. They routinely
        # quote the URL, which is why they are scrubbed first.
        logger.warning(
            "recorder connection test to {}:{} channel {} failed: {}",
            host,
            port,
            channel,
            text,
        )
        return RecorderTestResult("stream_unavailable", channel, elapsed(), **tested)

    return RecorderTestResult("connected", channel, elapsed(), width=width, height=height, **tested)


def _scrub(text: str, *, username: str, password: str) -> str:
    """Remove the credential from anything that is about to be logged."""
    cleaned = text
    for secret in (password, quote(password, safe="")):
        if secret:
            cleaned = cleaned.replace(secret, "***")
    if username:
        cleaned = cleaned.replace(f"{username}:", "***:")
        cleaned = cleaned.replace(f"{quote(username, safe='')}:", "***:")
    return cleaned


class ProbeThrottle:
    """At most one test per recorder every few seconds.

    A test opens an outbound connection with a stored password. Without a limit
    the button is a way to hammer a device — or, on a recorder that locks an
    account after repeated failures, a way to lock the account the cameras use.
    """

    __slots__ = ("_interval", "_last")

    def __init__(self, interval_s: float = 3.0) -> None:
        self._interval = interval_s
        self._last: dict[str, float] = {}

    def check(self, key: str) -> float:
        """Seconds to wait before this key may test again; 0 means go ahead."""
        now = time.monotonic()
        last = self._last.get(key)
        if last is not None and now - last < self._interval:
            return round(self._interval - (now - last), 1)
        self._last[key] = now
        # Bounded: a long-running process sees many keys and needs none of
        # the old ones.
        if len(self._last) > 1024:
            cutoff = now - self._interval
            self._last = {k: v for k, v in self._last.items() if v >= cutoff}
        return 0.0


__all__ = [
    "FRAME_TIMEOUT_S",
    "HANDSHAKE_TIMEOUT_S",
    "OUTCOMES",
    "RecorderTestResult",
    "ProbeThrottle",
    "check_connection",
]
