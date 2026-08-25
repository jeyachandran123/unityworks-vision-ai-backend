"""Per-layer observability, read from the platform's own surfaces.

The rule this module inherits from the validation console, and the reason the
console could stay useful across a year of platform evolution:

    Every layer of Vision OS must be inspectable **without modifying the
    Vision OS**. There are exactly three surfaces, all of which the platform
    already offers to anyone:

        1. The Event Bus  (M19) — 62 typed events, bounded subscriptions
        2. The Metrics Engine (M21) — a closed metric vocabulary
        3. The Health Monitor (M20) — component and coverage state

    There is no fourth surface, and in particular there is **no reaching into
    a module**.

A tap that read `TrackingRuntime._table` would be measuring an implementation
detail, would break at the next refactor, and would make DevTools a coupling
point that stops Vision OS evolving — the opposite of what a permanent
engineering tool is for. Everything below goes through `bus.subscribe`,
`metrics.snapshot()` and `health.components()`, all of which are public.

### Bounded, always

Each channel is a ring buffer with a hard cap. An unbounded tap in a long
soak grows fastest exactly when the platform is busiest, which is when an
engineer most needs the tool to still be up.

### The pull model is the platform's, not ours

`EventBus.subscribe()` returns a `Subscription` that buffers and is drained by
the reader. Nothing here calls back into the platform, holds its lock, or slows
a publisher: a slow DevTools reader drops its own events and says so through
`dropped`, and the pipeline never notices. That is why a tap cannot become a
back-pressure source on the perception path.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from loguru import logger

#: One channel per architectural layer, and one DevTools screen each.
CHANNELS: tuple[str, ...] = (
    "acquisition",
    "detection",
    "tracking",
    "registry",
    "cropping",
    "understanding",
    "synthesis",
    "state",
    "health",
    "platform",
)

#: Event-type prefix to channel. Mapping by *prefix* rather than by an explicit
#: table of 62 entries means a new event type added to Vision OS lands on the
#: right channel with no change here — which is the difference between a tap
#: that survives platform evolution and one that silently stops covering it.
_PREFIX_CHANNEL: tuple[tuple[str, str], ...] = (
    ("stream.", "acquisition"),
    ("privacy.", "acquisition"),
    ("scheduler.", "acquisition"),
    ("buffer.", "acquisition"),
    ("camera.", "acquisition"),
    ("detection.", "detection"),
    ("tracking.", "tracking"),
    ("registry.", "registry"),
    ("cropping.", "cropping"),
    ("understanding.", "understanding"),
    ("synthesis.", "synthesis"),
    ("state.", "state"),
    ("health.", "health"),
    ("config.", "platform"),
    ("plugin.", "platform"),
    ("model.", "platform"),
    ("runtime.", "platform"),
    ("bus.", "platform"),
)

#: Per-channel ring capacity. Small enough that ten channels of full buffers are
#: a few megabytes, large enough that a developer stepping through a replay sees
#: the whole run.
DEFAULT_CHANNEL_CAPACITY = 512


def _delivery_policy() -> Any:
    """A deep, drop-oldest subscription. Falls back to the bus default."""
    try:
        from vision_os.kernel.events.bus import DeliveryPolicy, OverflowPolicy

        return DeliveryPolicy(
            capacity=8192,
            # Newest events are the ones being investigated; an overflowing
            # debugging tool should lose history, not the present.
            overflow=OverflowPolicy.DROP_OLDEST,
        )
    except Exception:  # noqa: BLE001 - the bus default is a valid policy
        return None


def channel_for(event_type: str) -> str:
    """Which DevTools channel an event belongs to.

    An unrecognised prefix lands on `platform` rather than being dropped. A tap
    that silently discarded an event type nobody had mapped yet would create the
    hardest kind of observability gap: one that looks like silence from the
    platform.
    """
    for prefix, channel in _PREFIX_CHANNEL:
        if event_type.startswith(prefix):
            return channel
    return "platform"


@dataclass(frozen=True, slots=True)
class TapRecord:
    """One observed fact, with the sequence number that proves nothing was lost.

    `seq` is monotonic across all channels. A reader that sees 41 then 43 knows
    it missed one — which is a different and much more useful state than a
    reader that simply never saw 42.
    """

    seq: int
    at_ns: int
    channel: str
    event_type: str
    payload: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "at_ns": self.at_ns,
            "channel": self.channel,
            "event_type": self.event_type,
            "payload": self.payload,
        }


class TapBus:
    """Fan-in from the Vision OS event bus, fan-out to DevTools readers.

    Thread-safe because the platform publishes from the pipeline thread while
    HTTP handlers read from the event loop's executor. The lock is held only for
    the deque operations, never across a platform call.
    """

    __slots__ = ("_capacity", "_channels", "_dropped", "_lock", "_seq", "_subscription")

    def __init__(self, capacity: int = DEFAULT_CHANNEL_CAPACITY) -> None:
        self._capacity = capacity
        self._lock = threading.Lock()
        self._seq = 0
        self._dropped = 0
        self._subscription: Any = None
        # `maxlen` makes each channel a ring: the oldest record is discarded to
        # make room, which is the right loss for a debugging tool — the most
        # recent events are the ones being investigated.
        self._channels: dict[str, deque[TapRecord]] = {
            name: deque(maxlen=capacity) for name in CHANNELS
        }

    # -- attachment -----------------------------------------------------------

    def attach(self, bus: Any) -> bool:
        """Subscribe to every event type on the platform bus.

        Returns False rather than raising when the bus refuses or is absent: a
        deployment with no assembled platform is a valid deployment, and DevTools
        being unable to observe it is a fact to report, not a reason to fail
        start-up.
        """
        if bus is None:
            return False
        try:
            # `None` subscribes to everything. The bus applies its own
            # `DeliveryPolicy` bound, so this subscription cannot grow without
            # limit even if nothing ever drains it.
            # A busy tracking layer publishes several events per frame per
            # track; the bus's default subscription depth overflowed within
            # seconds and reported `bus.gap`. Asking for a deeper buffer is the
            # supported way to keep up — and if it still overflows, the gap is
            # still reported rather than hidden.
            self._subscription = bus.subscribe(
                None,
                policy=_delivery_policy(),
                subscription_id="devtools-tap",
            )
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            logger.warning(
                "DevTools taps could not attach to the event bus: {}: {}",
                type(exc).__name__,
                exc,
            )
            return False
        logger.info("DevTools taps attached to the Vision OS event bus")
        return True

    def detach(self, bus: Any) -> None:
        if self._subscription is None:
            return
        with _suppressed():
            bus.unsubscribe(self._subscription)
        self._subscription = None

    @property
    def attached(self) -> bool:
        return self._subscription is not None

    # -- draining -------------------------------------------------------------

    def pump(self, limit: int = 2000) -> int:
        """Move whatever the bus has buffered into the channel rings.

        Called on read rather than on a timer. A tap that ran its own thread
        would consume CPU in every deployment whether or not anyone had DevTools
        open; pumping on read means an idle deployment pays nothing.
        """
        if self._subscription is None:
            return 0

        try:
            events = self._subscription.drain(limit)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            logger.warning("DevTools tap drain failed: {}: {}", type(exc).__name__, exc)
            return 0

        moved = 0
        for event in events:
            record = self._record_for(event)
            if record is None:
                continue
            with self._lock:
                self._channels[record.channel].append(record)
            moved += 1

        # The bus's own drop count. Surfaced rather than hidden: a reader that
        # fell behind must know it did.
        with _suppressed():
            self._dropped = int(getattr(self._subscription, "dropped", 0) or 0)

        return moved

    def _record_for(self, event: Any) -> TapRecord | None:
        try:
            event_type = str(getattr(type(event), "event_type", "event"))
            payload = dict(event.payload()) if hasattr(event, "payload") else {}
            # `Event.payload()` merges only `detail`, so an event's own typed
            # fields — `DetectionFailed.reason`, `TrackCreated.track_id` — do not
            # appear in it. Those fields are the whole content of the event, and
            # without them a tap shows sixty `detection.failed` records that all
            # say nothing about why.
            #
            # These are public fields on a public frozen dataclass, so reading
            # them is not the "reaching into a module" this file forbids: the
            # event *is* the published value.
            payload.update(_typed_fields(event))
        except Exception:  # noqa: BLE001 - one malformed event, not the stream
            return None

        with self._lock:
            self._seq += 1
            seq = self._seq

        return TapRecord(
            seq=seq,
            at_ns=int(payload.get("occurred_at_ns", time.time_ns())),
            channel=channel_for(event_type),
            event_type=event_type,
            payload=_shallow(payload),
        )

    # -- reading --------------------------------------------------------------

    def records(
        self,
        *,
        channel: str | None = None,
        since_seq: int = 0,
        limit: int = 200,
    ) -> list[TapRecord]:
        """Records newest-last, optionally for one channel and after a sequence.

        `since_seq` is how a poller resumes without re-reading: it asks for what
        came after what it already has, rather than for "the last N" and
        de-duplicating on the client.
        """
        self.pump()
        limit = min(max(limit, 1), 1000)

        with self._lock:
            if channel is not None:
                source: list[TapRecord] = list(self._channels.get(channel, ()))
            else:
                source = [r for ring in self._channels.values() for r in ring]

        if since_seq:
            source = [r for r in source if r.seq > since_seq]
        source.sort(key=lambda r: r.seq)
        return source[-limit:]

    def stats(self) -> dict[str, Any]:
        """Channel depths and the drop count. Cheap; safe to call on every read."""
        self.pump()
        with self._lock:
            by_channel = {name: len(ring) for name, ring in self._channels.items()}
            seq = self._seq
        return {
            "attached": self.attached,
            "sequence": seq,
            "capacity_per_channel": self._capacity,
            "by_channel": by_channel,
            "observed_total": sum(by_channel.values()),
            # Non-zero means this tap fell behind the platform. It is reported
            # because a gap a reader does not know about is worse than a gap.
            "dropped_by_bus": self._dropped,
        }

    def clear(self) -> None:
        with self._lock:
            for ring in self._channels.values():
                ring.clear()


#: Fields every event carries, already in `payload()`. Excluded so the typed
#: overlay adds only what is specific to the event type.
_BASE_FIELDS = frozenset({"occurred_at", "detail", "partition_key"})


def _typed_fields(event: Any) -> dict[str, Any]:
    """An event subclass's own declared fields, JSON-safe."""
    names = getattr(type(event), "__dataclass_fields__", None)
    if not names:
        return {}
    out: dict[str, Any] = {}
    for name in names:
        if name in _BASE_FIELDS:
            continue
        value = getattr(event, name, None)
        if value is None or value == "":
            continue
        if isinstance(value, str | int | float | bool):
            out[name] = value
        else:
            # `CameraId`, `StreamEpoch` and friends are NewType-like wrappers
            # that render as their underlying value.
            out[name] = str(getattr(value, "value", value))
    return out


def _shallow(payload: dict[str, Any]) -> dict[str, Any]:
    """One level of JSON-safe coercion, bounded.

    Event details are already flat by contract. This guards the case where a
    future event carries a rich object: the tap renders it as text rather than
    failing to serialise, because a DevTools screen that 500s on one event type
    is worse than one that shows that event as a string.
    """
    out: dict[str, Any] = {}
    for key, value in list(payload.items())[:40]:
        if isinstance(value, str | int | float | bool) or value is None:
            out[key] = value
        elif isinstance(value, list | tuple):
            out[key] = [str(v) for v in list(value)[:20]]
        elif isinstance(value, dict):
            out[key] = {str(k): str(v) for k, v in list(value.items())[:20]}
        else:
            out[key] = str(value)
    return out


class _suppressed:
    """Best-effort block for reads that must never break a diagnostics page."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return exc_type is not None


# -- metrics ------------------------------------------------------------------


def metrics_view(platform: Any) -> dict[str, Any]:
    """The platform's metric snapshot, flattened for the wire.

    Reads `MetricsEngine.snapshot()` — the public surface. The names come from
    the platform's closed vocabulary in `kernel/metrics/names.py`, so this
    function declares no metric names of its own and cannot drift from them.
    """
    engine = getattr(platform, "metrics", None)
    if engine is None:
        return {"available": False, "reason": "no metrics engine on this platform"}

    try:
        snapshot = engine.snapshot()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    counters = _flatten(getattr(snapshot, "counters", {}))
    gauges = _flatten(getattr(snapshot, "gauges", {}))
    histograms = {
        _key_name(key): _summarise(values)
        for key, values in getattr(snapshot, "histograms", {}).items()
    }

    return {
        "available": True,
        "counters": counters,
        "gauges": gauges,
        "histograms": histograms,
        # A non-zero cardinality violation count means some label is unbounded,
        # which is the metric bug that takes a store down. Surfaced, not buried.
        "cardinality_violations": int(getattr(snapshot, "cardinality_violations", 0) or 0),
    }


def counter_total(metrics: dict[str, Any], name: str) -> float:
    """Total of one counter, without double-counting a breakdown.

    The platform publishes the same counter at more than one label depth:
    `detection.emitted` appears once per camera *and* once per camera-and-class.
    Adding every series together would report 437 + 383 + 19 + 24 detections when
    437 were emitted — and the economy figures downstream would inherit the
    error, silently and by a factor that changes with the number of classes.

    So the series are grouped by their label **names**, and only the shallowest
    group — the aggregate — is summed. Cameras are still added together, because
    those are genuinely different series of the same measurement.
    """
    if not metrics.get("available"):
        return 0.0

    by_depth: dict[tuple[str, ...], float] = {}
    for key, value in metrics.get("counters", {}).items():
        base, _, rendered = key.partition("|")
        if base != name:
            continue
        signature = tuple(sorted(part.split("=", 1)[0] for part in rendered.split(",") if part))
        by_depth[signature] = by_depth.get(signature, 0.0) + float(value)

    if not by_depth:
        return 0.0
    # The **largest** group total, not the shallowest series.
    #
    # "Shallowest wins" was wrong, and wrong in the direction that hides a
    # working system. The platform registers some counters at a shallow label
    # depth with `increment(0)` — so the series exists before anything happens —
    # and counts them at a deeper depth. `synthesis.built` does exactly this, and
    # the shallow rule reported 0 while 140 observations had genuinely been
    # built.
    #
    # A breakdown of a counter can never exceed its aggregate, so taking the
    # maximum is right for both idioms: it never double-counts, and it never
    # reads a registration placeholder as the answer.
    return max(by_depth.values())


def counter_breakdown(metrics: dict[str, Any], name: str, label: str) -> dict[str, float]:
    """One counter split by a single label — `{class_id: count}`.

    Read straight from the recorded series, never derived. Returns `{}` when the
    platform did not label that counter, which is a fact worth showing rather
    than a zero to invent.
    """
    if not metrics.get("available"):
        return {}
    prefix = f"{label}="
    out: dict[str, float] = {}
    for key, value in metrics.get("counters", {}).items():
        base, _, rendered = key.partition("|")
        if base != name or not rendered:
            continue
        for part in rendered.split(","):
            if part.startswith(prefix):
                out[part[len(prefix) :]] = out.get(part[len(prefix) :], 0.0) + float(value)
    return out


def histogram_p50(metrics: dict[str, Any], name: str) -> float | None:
    """Median of one histogram, or `None` when nothing was recorded.

    `None` rather than `0.0`: a latency of zero and a latency never measured are
    different facts, and a dashboard showing `0 ms` for an unmeasured model is
    the kind of confident wrong number this product refuses to print.
    """
    if not metrics.get("available"):
        return None
    for key, summary in metrics.get("histograms", {}).items():
        if key == name or key.startswith(f"{name}|"):
            if summary.get("count"):
                return summary.get("p50")
    return None


def _flatten(values: dict[Any, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in values.items():
        out[_key_name(key)] = float(value)
    return out


def _split_key(key: Any) -> tuple[str, tuple[tuple[str, str], ...]]:
    """A metric key as `(name, sorted labels)`.

    The engine keys metrics by the tuple `(name, (('label', 'value'), …))`.
    Handling the attribute form too keeps this working if that ever becomes an
    object, without asserting either shape.
    """
    if isinstance(key, tuple) and len(key) == 2:
        name, labels = key
        try:
            return str(name), tuple(sorted((str(k), str(v)) for k, v in labels or ()))
        except Exception:  # noqa: BLE001
            return str(name), ()
    name = str(getattr(key, "name", key))
    labels = getattr(key, "labels", None) or ()
    try:
        return name, tuple(sorted((str(k), str(v)) for k, v in dict(labels).items()))
    except Exception:  # noqa: BLE001
        return name, ()


def _key_name(key: Any) -> str:
    """A stable wire string: `name|label=value,label=value`."""
    name, labels = _split_key(key)
    rendered = ",".join(f"{k}={v}" for k, v in labels)
    return f"{name}|{rendered}" if rendered else name


def _summarise(values: Any) -> dict[str, Any]:
    """Count, min, p50, p95, max. Never a mean.

    A mean over a latency distribution with a cold start in it reports a number
    no request ever experienced. Percentiles over the recorded samples describe
    what actually happened.
    """
    samples = sorted(float(v) for v in values or ())
    if not samples:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(samples),
        "min": samples[0],
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "max": samples[-1],
    }


def _percentile(sorted_samples: list[float], fraction: float) -> float:
    if not sorted_samples:
        return 0.0
    index = min(len(sorted_samples) - 1, int(fraction * len(sorted_samples)))
    return sorted_samples[index]


# -- health -------------------------------------------------------------------


def health_view(platform: Any) -> dict[str, Any]:
    """Component and coverage health, read from the Health Monitor.

    Coverage is the signal an operator cannot get anywhere else: a camera that is
    connected but not observable is the failure mode that looks exactly like a
    compliant kitchen.
    """
    monitor = getattr(platform, "health", None)
    if monitor is None:
        return {"available": False, "reason": "no health monitor on this platform"}

    components: list[dict[str, Any]] = []
    with _suppressed():
        for component in monitor.components():
            components.append(
                {
                    "component_id": str(getattr(component, "component_id", "")),
                    "state": _name_of(getattr(component, "state", "")),
                    "detail": str(getattr(component, "detail", "") or ""),
                }
            )

    site: dict[str, Any] = {}
    with _suppressed():
        summary = monitor.site_health()
        site = {
            "observable_fraction": float(getattr(summary, "observable_fraction", 0.0) or 0.0),
            "cameras_total": int(getattr(summary, "cameras_total", 0) or 0),
            "cameras_observable": int(getattr(summary, "cameras_observable", 0) or 0),
        }

    ready, ready_reasons = _pair(monitor, "readiness")
    live, live_reasons = _pair(monitor, "liveness")

    return {
        "available": True,
        "components": components,
        "site": site,
        "ready": ready,
        "ready_blockers": list(ready_reasons),
        "live": live,
        "live_blockers": list(live_reasons),
    }


def _pair(monitor: Any, name: str) -> tuple[bool, tuple[str, ...]]:
    reader = getattr(monitor, name, None)
    if not callable(reader):
        return False, ()
    try:
        ok, reasons = reader()
        return bool(ok), tuple(str(r) for r in reasons or ())
    except Exception as exc:  # noqa: BLE001
        return False, (f"{type(exc).__name__}: {exc}",)


def _name_of(value: Any) -> str:
    return str(getattr(value, "value", value))


__all__ = [
    "CHANNELS",
    "DEFAULT_CHANNEL_CAPACITY",
    "TapBus",
    "TapRecord",
    "channel_for",
    "counter_breakdown",
    "counter_total",
    "health_view",
    "histogram_p50",
    "metrics_view",
]
