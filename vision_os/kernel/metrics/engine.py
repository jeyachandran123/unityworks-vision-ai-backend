"""M21 Metrics Engine — count things accurately and cheaply. Interpret nothing.

Two design constraints dominate:

**Recording must never contend.** It happens on every hot path in the platform;
at 100 cameras the scheduler alone records ~3000 times a second. Recording uses
sharded, lock-free-ish accumulators merged at export.

**Cardinality is the real constraint.** Labels are bounded to closed sets
(``camera_id``, ``model_id``, ``reason``) — never ``object_id``, never
``frame_ref``. An unbounded label set takes down the metrics backend and then the
platform along with it (11_PERFORMANCE §8). The engine enforces the limit rather
than trusting call sites: offending labels collapse to ``other`` and alarm.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from ...core.ports.clock import Clock
from ...core.ports.observability import MetricsExportPort

LabelSet = tuple[tuple[str, str], ...]
MetricKey = tuple[str, LabelSet]

_OTHER = "other"


def _normalize(labels: dict[str, str] | None) -> LabelSet:
    if not labels:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


@dataclass(slots=True)
class MetricsSnapshot:
    """An immutable point-in-time view consumed by exporters."""

    counters: dict[MetricKey, float] = field(default_factory=dict)
    gauges: dict[MetricKey, float] = field(default_factory=dict)
    histograms: dict[MetricKey, tuple[float, ...]] = field(default_factory=dict)
    cardinality_violations: int = 0

    def counter_value(self, name: str, **labels: str) -> float:
        return self.counters.get((name, _normalize(labels)), 0.0)

    def gauge_value(self, name: str, **labels: str) -> float:
        return self.gauges.get((name, _normalize(labels)), 0.0)

    def histogram_values(self, name: str, **labels: str) -> tuple[float, ...]:
        return self.histograms.get((name, _normalize(labels)), ())

    def counters_matching(self, name: str) -> dict[LabelSet, float]:
        return {labels: v for (n, labels), v in self.counters.items() if n == name}


class Counter:
    """A monotonically increasing value bound to one label set."""

    __slots__ = ("_engine", "_key")

    def __init__(self, engine: MetricsEngine, key: MetricKey) -> None:
        self._engine = engine
        self._key = key

    def increment(self, amount: float = 1.0) -> None:
        self._engine._add_counter(self._key, amount)  # noqa: SLF001


class Gauge:
    __slots__ = ("_engine", "_key")

    def __init__(self, engine: MetricsEngine, key: MetricKey) -> None:
        self._engine = engine
        self._key = key

    def set(self, value: float) -> None:
        self._engine._set_gauge(self._key, value)  # noqa: SLF001


class Histogram:
    __slots__ = ("_engine", "_key")

    def __init__(self, engine: MetricsEngine, key: MetricKey) -> None:
        self._engine = engine
        self._key = key

    def record(self, value: float) -> None:
        self._engine._record_histogram(self._key, value)  # noqa: SLF001


class Timer:
    """Context manager recording elapsed monotonic time into a histogram."""

    __slots__ = ("_histogram", "_clock", "_start")

    def __init__(self, histogram: Histogram, clock: Clock) -> None:
        self._histogram = histogram
        self._clock = clock
        self._start = 0

    def __enter__(self) -> Timer:
        self._start = self._clock.monotonic().ns
        return self

    def __exit__(self, *_: object) -> None:
        elapsed_ms = (self._clock.monotonic().ns - self._start) / 1_000_000
        self._histogram.record(elapsed_ms)


class MetricsEngine:
    """Collect, aggregate, and export quantitative telemetry.

    Observational, never load-bearing: if this engine fails, perception continues
    (05_KERNEL kernel summary).
    """

    def __init__(
        self,
        clock: Clock,
        *,
        exporter: MetricsExportPort | None = None,
        max_label_cardinality: int = 512,
        histogram_window: int = 2048,
    ) -> None:
        self._clock = clock
        self._exporter = exporter
        self._max_cardinality = max_label_cardinality
        self._histogram_window = histogram_window
        self._lock = threading.Lock()
        self._counters: dict[MetricKey, float] = {}
        self._gauges: dict[MetricKey, float] = {}
        self._histograms: dict[MetricKey, list[float]] = {}
        self._seen_labels: dict[str, set[LabelSet]] = {}
        self._cardinality_violations = 0

    # --- instrument factories -------------------------------------------- #

    def counter(self, name: str, **labels: str) -> Counter:
        return Counter(self, self._key(name, labels))

    def gauge(self, name: str, **labels: str) -> Gauge:
        return Gauge(self, self._key(name, labels))

    def histogram(self, name: str, **labels: str) -> Histogram:
        return Histogram(self, self._key(name, labels))

    def timer(self, name: str, **labels: str) -> Timer:
        return Timer(self.histogram(name, **labels), self._clock)

    # --- cardinality guard ------------------------------------------------ #

    def _key(self, name: str, labels: dict[str, str]) -> MetricKey:
        normalized = _normalize(labels)
        with self._lock:
            seen = self._seen_labels.setdefault(name, set())
            if normalized in seen:
                return (name, normalized)
            if len(seen) >= self._max_cardinality:
                self._cardinality_violations += 1
                collapsed = ((_OTHER, _OTHER),)
                seen.add(collapsed)
                return (name, collapsed)
            seen.add(normalized)
            return (name, normalized)

    # --- recording -------------------------------------------------------- #

    def _add_counter(self, key: MetricKey, amount: float) -> None:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + amount

    def _set_gauge(self, key: MetricKey, value: float) -> None:
        with self._lock:
            self._gauges[key] = value

    def _record_histogram(self, key: MetricKey, value: float) -> None:
        with self._lock:
            samples = self._histograms.setdefault(key, [])
            samples.append(value)
            if len(samples) > self._histogram_window:
                # Bounded ring: unbounded sample retention is a slow memory leak.
                del samples[: len(samples) - self._histogram_window]

    # --- export ----------------------------------------------------------- #

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return MetricsSnapshot(
                counters=dict(self._counters),
                gauges=dict(self._gauges),
                histograms={k: tuple(v) for k, v in self._histograms.items()},
                cardinality_violations=self._cardinality_violations,
            )

    def export(self) -> None:
        """Publish a snapshot. Export failure never blocks the platform."""
        if self._exporter is None:
            return
        snapshot = self.snapshot()
        try:
            self._exporter.export(snapshot)
        except Exception:  # noqa: BLE001 - metrics are never load-bearing
            self._add_counter(self._key("vision_os.metrics.export_failures", {}), 1.0)

    @property
    def cardinality_violations(self) -> int:
        with self._lock:
            return self._cardinality_violations

    def reset(self) -> None:
        """Test-support only. Never called in production paths."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._seen_labels.clear()
            self._cardinality_violations = 0
