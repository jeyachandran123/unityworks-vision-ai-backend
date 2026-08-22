"""The frame-source boundary. **A source is a source.**

Replay and live RTSP implement the same interface and enter the same session.
There is no replay pipeline and no CCTV pipeline — there is one pipeline, and
two things that feed it. A replay whose downstream path differs from live is a
replay that validates nothing.

The one honest difference is `kind`: a replay says `REPLAY` and a camera says
`LIVE`, and **nothing ever labels a replay as live**.

### States

    CREATED → CONNECTING → RUNNING → STOPPING → STOPPED
                  ↑           ↓
                  └── RECONNECTING
                              ↓
                            ERROR

Every transition is recorded with a timestamp and a reason, and is observable
through `SourceStatus`. A source that changes state silently is a camera whose
outage nobody can explain afterwards.
"""

from __future__ import annotations

import abc
import enum
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.vision.frames import LiveFrame


class SourceKind(enum.Enum):
    """What is feeding the pipeline. Rendered verbatim; never relabelled."""

    REPLAY = "replay"
    LIVE = "live"


class SourceState(enum.Enum):
    """Truthful, not decorative. Every value is a state an operator can act on."""

    CREATED = "created"
    """Configured. No socket, no file handle, no thread."""

    CONNECTING = "connecting"
    """Opening. Not yet producing."""

    RUNNING = "running"
    """Connected **and** producing frames."""

    RECONNECTING = "reconnecting"
    """Was running, lost the stream, backing off. Distinct from ERROR: this one
    is expected to recover, and the UI should not raise an incident for it."""

    STOPPING = "stopping"
    STOPPED = "stopped"
    """Deliberately ended. Not a fault."""

    ERROR = "error"
    """Cannot produce and will not retry — retries exhausted, or a failure that
    retrying cannot fix (bad credentials, unknown stream path)."""

    @property
    def is_producing(self) -> bool:
        return self is SourceState.RUNNING

    @property
    def is_terminal(self) -> bool:
        return self in (SourceState.STOPPED, SourceState.ERROR)


class CameraHealth(enum.Enum):
    """Source state as an operator reads it.

    A separate vocabulary because the two audiences differ: `RECONNECTING` is a
    precise engineering state and `DEGRADED` is what a restaurant manager needs
    to see. Derived, never stored — one truth, two renderings.
    """

    CONNECTING = "connecting"
    ONLINE = "online"
    """Producing frames now."""

    DEGRADED = "degraded"
    """Reconnecting, or connected and producing nothing. **Never shown as
    online**, and never shown as a frozen last frame."""

    OFFLINE = "offline"
    """Deliberately stopped, or not started."""

    ERROR = "error"
    """Failed and not retrying. Needs a human."""

    @classmethod
    def of(cls, state: SourceState, *, stale: bool = False) -> CameraHealth:
        if state is SourceState.RUNNING:
            # Connected but silent is not online. A camera whose last frame is
            # minutes old must never render as healthy — that is the frozen-frame
            # failure, and it is the most dangerous default in CCTV software.
            return cls.DEGRADED if stale else cls.ONLINE
        if state is SourceState.CONNECTING:
            return cls.CONNECTING
        if state is SourceState.RECONNECTING:
            return cls.DEGRADED
        if state is SourceState.ERROR:
            return cls.ERROR
        return cls.OFFLINE


@dataclass(frozen=True, slots=True)
class StateTransition:
    at_ns: int
    from_state: SourceState
    to_state: SourceState
    reason: str

    def to_wire(self) -> dict[str, Any]:
        return {
            "at_ns": self.at_ns,
            "from": self.from_state.value,
            "to": self.to_state.value,
            "reason": self.reason,
        }


@dataclass(slots=True)
class SourceStatus:
    """Everything observable about a source. Contains no credential, ever."""

    camera_id: str
    kind: SourceKind
    state: SourceState = SourceState.CREATED
    #: Safe to show anywhere: `rtsp://***:***@host:554/...`
    redacted_uri: str = ""
    epoch: int = 0
    frames_produced: int = 0
    last_frame_captured_ns: int | None = None
    last_frame_received_ns: int | None = None
    reconnects: int = 0
    errors: int = 0
    #: Redacted before it is stored. Never a raw exception carrying a URL.
    last_error: str = ""
    transitions: list[StateTransition] = field(default_factory=list)

    #: Beyond this with no frame, a RUNNING source is reported DEGRADED rather
    #: than ONLINE. Ten seconds is ~40 missed frames at the 4 fps analysis rate:
    #: long enough not to flap on a hiccup, short enough that an operator is not
    #: looking at a minute-old world believing it is current.
    stale_after_ms: float = 10_000.0

    @property
    def stale(self) -> bool:
        if self.state is not SourceState.RUNNING:
            return False
        if self.last_frame_received_ns is None:
            # Connected and has never produced. Not online.
            return True
        return (time.time_ns() - self.last_frame_received_ns) / 1_000_000 > self.stale_after_ms

    @property
    def health(self) -> CameraHealth:
        return CameraHealth.of(self.state, stale=self.stale)

    @property
    def producing(self) -> bool:
        """Whether genuine frames have arrived **and** the source still runs.

        This is what `streaming` on the WebSocket is derived from. It is never
        set by hand and never true merely because a socket opened.
        """
        return self.state.is_producing and self.frames_produced > 0

    def to_wire(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "kind": self.kind.value,
            "state": self.state.value,
            "health": self.health.value,
            "uri": self.redacted_uri,
            "epoch": self.epoch,
            "frames_produced": self.frames_produced,
            "last_frame_captured_ns": self.last_frame_captured_ns,
            "last_frame_received_ns": self.last_frame_received_ns,
            "reconnects": self.reconnects,
            "errors": self.errors,
            "last_error": self.last_error,
            "producing": self.producing,
            "stale": self.stale,
            "transitions": [t.to_wire() for t in self.transitions[-10:]],
        }


class FrameSource(abc.ABC):
    """A thing that produces frames for exactly one camera.

    Subclasses implement `_produce()`. Everything else — state machine,
    transition log, counters — is shared, so no source can invent its own
    reporting and no source can change state without being observed.
    """

    def __init__(self, *, camera_id: str, kind: SourceKind, redacted_uri: str = "") -> None:
        if not camera_id:
            raise ValueError("a source must name its camera; identity is never inferred")
        self._status = SourceStatus(camera_id=camera_id, kind=kind, redacted_uri=redacted_uri)
        self._stopping = False

    @property
    def status(self) -> SourceStatus:
        return self._status

    @property
    def camera_id(self) -> str:
        return self._status.camera_id

    @property
    def kind(self) -> SourceKind:
        return self._status.kind

    # ── state ────────────────────────────────────────────────────────────────

    def _transition(self, to_state: SourceState, reason: str = "") -> None:
        previous = self._status.state
        if previous is to_state:
            return
        self._status.state = to_state
        self._status.transitions.append(
            StateTransition(
                at_ns=time.time_ns(),
                from_state=previous,
                to_state=to_state,
                reason=self._redact(reason),
            )
        )
        # Bounded. A source reconnecting all night must not accumulate a
        # transition log that outgrows the frames it failed to deliver.
        if len(self._status.transitions) > 200:
            del self._status.transitions[:-100]

    def _record_frame(self, frame: LiveFrame) -> None:
        self._status.frames_produced += 1
        self._status.last_frame_captured_ns = frame.captured_at_ns
        self._status.last_frame_received_ns = frame.received_at_ns

    def _record_error(self, message: str) -> None:
        self._status.errors += 1
        self._status.last_error = self._redact(message)

    def _redact(self, text: str) -> str:
        """Overridden by sources that hold a credential. Base case: nothing to hide."""
        return text

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def frames(self) -> AsyncIterator[LiveFrame]:
        """Yield frames until stopped or exhausted.

        The state machine lives here rather than in each source, so `RUNNING`
        means the same thing for a file and for a camera.
        """
        self._stopping = False
        self._transition(SourceState.CONNECTING, "starting")
        try:
            async for frame in self._produce():
                if self._stopping:
                    break
                self._transition(SourceState.RUNNING, "frame received")
                self._record_frame(frame)
                yield frame
        except Exception as exc:  # noqa: BLE001 - reported as state, not raised
            self._record_error(f"{type(exc).__name__}: {exc}")
            self._transition(SourceState.ERROR, type(exc).__name__)
            return

        self._transition(SourceState.STOPPED, "stopped" if self._stopping else "source exhausted")

    def stop(self) -> None:
        """Ask the source to finish. Idempotent, and never blocks."""
        if self._status.state.is_terminal:
            return
        self._stopping = True
        self._transition(SourceState.STOPPING, "stop requested")

    @abc.abstractmethod
    def _produce(self) -> AsyncIterator[LiveFrame]:
        """Produce frames. Raise on unrecoverable failure; return on end of stream."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release everything held. Called on every shutdown path."""
        self.stop()
        self._transition(SourceState.STOPPED, "closed")


__all__ = [
    "CameraHealth",
    "FrameSource",
    "SourceKind",
    "SourceState",
    "SourceStatus",
    "StateTransition",
]
