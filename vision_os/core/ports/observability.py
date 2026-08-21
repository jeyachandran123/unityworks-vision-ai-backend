"""Observability ports — P29 ``EventTransportPort``, P30 ``MetricsExportPort``.

Owners: M19 Event Bus, M21 Metrics Engine.

Both are **observational, never load-bearing**: the platform must keep perceiving
even when it has lost the ability to describe how well it is perceiving.
Observability that can take down the system it observes is a liability
(05_KERNEL kernel summary).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EventTransportPort(Protocol):
    """P29 — carry control-plane notifications beyond the local process.

    Implementations: in-process (no-op), shared memory, NATS, Kafka, cloud
    pub/sub.

    **No frame, crop, or tensor ever travels on the bus.** Payloads are bounded
    by contract; anything large becomes a reference to storage (05_KERNEL M19).
    """

    @property
    def transport_id(self) -> str: ...

    def deliver(self, event_type: str, partition_key: str, payload: dict[str, Any]) -> None:
        """Deliver one event. Must not block the caller.

        Failure is absorbed and counted: a transport outage degrades remote
        delivery, never local perception.
        """
        ...

    def flush(self) -> None: ...


@runtime_checkable
class MetricsExportPort(Protocol):
    """P30 — publish aggregated telemetry to an external system.

    Implementations: Prometheus/OpenMetrics, OpenTelemetry, StatsD, cloud
    monitoring, local files for air-gapped edge deployments.
    """

    @property
    def exporter_id(self) -> str: ...

    def export(self, snapshot: MetricsSnapshotView) -> None:
        """Publish a snapshot. Never blocks the platform to record a metric."""
        ...


class MetricsSnapshotView(Protocol):
    """The read-only shape an exporter consumes."""

    @property
    def counters(self) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]: ...

    @property
    def gauges(self) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]: ...

    @property
    def histograms(self) -> dict[tuple[str, tuple[tuple[str, str], ...]], tuple[float, ...]]: ...
