"""P29/P30 — event transport and metrics export adapters.

Both are **observational, never load-bearing**. Every adapter here absorbs its
own failures: a transport outage or a scrape endpoint being down degrades remote
visibility, never local perception (05_KERNEL kernel summary).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from ...core.ports.observability import MetricsSnapshotView


class NullEventTransport:
    """Local-only delivery. The default for single-process deployments."""

    __slots__ = ()

    @property
    def transport_id(self) -> str:
        return "null"

    def deliver(self, event_type: str, partition_key: str, payload: dict[str, Any]) -> None:
        return None

    def flush(self) -> None:
        return None


@dataclass(slots=True)
class DeliveredEvent:
    event_type: str
    partition_key: str
    payload: dict[str, Any]


class RecordingEventTransport:
    """Retain delivered events in a bounded ring.

    Bounded deliberately: an unbounded recorder is a memory leak with a delayed
    fuse, which is exactly what the platform forbids elsewhere.
    """

    def __init__(self, *, capacity: int = 4096) -> None:
        self._capacity = capacity
        self._lock = threading.Lock()
        self._events: list[DeliveredEvent] = []

    @property
    def transport_id(self) -> str:
        return "recording"

    def deliver(self, event_type: str, partition_key: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(DeliveredEvent(event_type, partition_key, payload))
            if len(self._events) > self._capacity:
                del self._events[: len(self._events) - self._capacity]

    def flush(self) -> None:
        return None

    def events(self, event_type: str | None = None) -> tuple[DeliveredEvent, ...]:
        with self._lock:
            if event_type is None:
                return tuple(self._events)
            return tuple(e for e in self._events if e.event_type == event_type)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class FailingEventTransport:
    """Always raises. Proves transport failure never reaches perception."""

    __slots__ = ()

    @property
    def transport_id(self) -> str:
        return "failing"

    def deliver(self, event_type: str, partition_key: str, payload: dict[str, Any]) -> None:
        raise RuntimeError("synthetic transport failure")

    def flush(self) -> None:
        raise RuntimeError("synthetic transport failure")


@dataclass(slots=True)
class InMemoryMetricsExporter:
    """Keep the most recent snapshot. Useful for assertions and local dashboards."""

    exports: int = 0
    last: MetricsSnapshotView | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def exporter_id(self) -> str:
        return "in-memory"

    def export(self, snapshot: MetricsSnapshotView) -> None:
        with self._lock:
            self.exports += 1
            self.last = snapshot


class OpenMetricsTextExporter:
    """Render a snapshot as OpenMetrics/Prometheus text exposition.

    Formatting only: serving the text over HTTP is a deployment concern, not a
    platform one.
    """

    __slots__ = ("_last_text", "_lock")

    def __init__(self) -> None:
        self._last_text = ""
        self._lock = threading.Lock()

    @property
    def exporter_id(self) -> str:
        return "openmetrics-text"

    def export(self, snapshot: MetricsSnapshotView) -> None:
        lines: list[str] = []
        for (name, labels), value in sorted(snapshot.counters.items()):
            lines.append(f"{_render(name, labels)} {value}")
        for (name, labels), value in sorted(snapshot.gauges.items()):
            lines.append(f"{_render(name, labels)} {value}")
        for (name, labels), samples in sorted(snapshot.histograms.items()):
            if not samples:
                continue
            ordered = sorted(samples)
            lines.append(f"{_render(name + '_count', labels)} {len(ordered)}")
            lines.append(f"{_render(name + '_sum', labels)} {sum(ordered)}")
            lines.append(
                f"{_render(name + '_p95', labels)} {ordered[int(len(ordered) * 0.95) - 1]}"
            )
        with self._lock:
            self._last_text = "\n".join(lines)

    def text(self) -> str:
        with self._lock:
            return self._last_text


def _render(name: str, labels: tuple[tuple[str, str], ...]) -> str:
    metric = name.replace(".", "_")
    if not labels:
        return metric
    rendered = ",".join(f'{k}="{v}"' for k, v in labels)
    return f"{metric}{{{rendered}}}"
