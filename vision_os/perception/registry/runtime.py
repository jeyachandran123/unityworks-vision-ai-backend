"""The Registry Runtime — the ``TrackingConsumer`` seam, one actor per camera.

> **Single responsibility:** *Own the registry's lifecycle and serialize each
> camera's writes. Decide no identity yourself.*

``08_RUNTIME`` section 2 places M7 in the actor table and section 4.1 of
``07_STATE`` states the rule: *"The camera is the partition. Each partition has
exactly one writer."* This runtime is what makes that true regardless of how the
caller behaves — one lock per camera, so two updates for one camera can never
interleave while two cameras never contend at all.

It also owns the two schedules M7 needs that no frame drives:

**Expiry.** Horizons must advance for a camera that has gone quiet. Without a
tick, a camera that stops sending frames would freeze its objects in whatever
state they were last in, and a dormant object would never become departed.

**Persistence.** ``07_STATE`` section 9.3 requires object identity to survive
restart. Writes are batched and asynchronous (section M7 Performance): the hot
path updates memory and the flush runs on its own schedule, so ingestion never
blocks on I/O.
"""

from __future__ import annotations

import asyncio

from ...core.errors import ObjectStoreError
from ...core.model.health import ComponentHealth, HealthState
from ...core.model.ids import CameraId, ModuleId
from ...core.model.timebase import Duration, Instant
from ...core.model.track import TrackUpdate
from ...core.ports.clock import Clock
from ...core.ports.registry import ObjectStorePort, PartitionSnapshot
from ...kernel.config.schema import RegistrySection
from ...kernel.health import HealthMonitor
from ...kernel.metrics import MetricName, MetricsEngine
from .engine import ObjectRegistry, RegistryUpdate

REGISTRY_RUNTIME_ID = ModuleId("registry_runtime")

_DEFAULT_REPORT_INTERVAL = Duration.from_millis(1_000)


class RegistryRuntimeStats:
    """Mutable counters. Updated on the hot path, never published as a value."""

    __slots__ = (
        "frames_consumed",
        "frames_failed",
        "objects_emitted",
        "persist_failures",
        "persists",
        "sink_failures",
        "updates_applied",
    )

    def __init__(self) -> None:
        self.frames_consumed = 0
        self.updates_applied = 0
        self.frames_failed = 0
        self.objects_emitted = 0
        self.sink_failures = 0
        self.persists = 0
        self.persist_failures = 0

    @property
    def failure_rate(self) -> float:
        return self.frames_failed / self.frames_consumed if self.frames_consumed else 0.0


class RegistryRuntime:
    """Implements the tracking-to-registry seam; owns registry lifecycle."""

    def __init__(
        self,
        *,
        clock: Clock,
        metrics: MetricsEngine,
        health: HealthMonitor,
        registry: ObjectRegistry,
        config: RegistrySection,
        store: ObjectStorePort | None = None,
        sink=None,
        report_interval: Duration = _DEFAULT_REPORT_INTERVAL,
    ) -> None:
        self._clock = clock
        self._metrics = metrics
        self._health = health
        self._registry = registry
        self._config = config
        self._store = store
        self._sink = sink
        self._report_interval = report_interval

        self._stats = RegistryRuntimeStats()
        self._locks: dict[CameraId, asyncio.Lock] = {}
        self._dirty: set[CameraId] = set()
        self._started = False
        self._last_report_ns = 0
        self._last_expiry_ns = 0
        self._last_persist_ns = 0

    # --- lifecycle -------------------------------------------------------------- #

    async def start(self) -> None:
        """Reload durable objects, then accept traffic.

        Reload happens before the first frame so a track arriving immediately
        after a restart can re-bind to the object it belonged to rather than
        minting a duplicate.
        """
        if self._store is not None and self._config.persistence_enabled:
            self._reload()
        self._started = True
        self._report_health(HealthState.HEALTHY, "registry ready")

    async def stop(self) -> None:
        """Flush before shutting down.

        A clean shutdown that discarded unflushed objects would make restart
        behaviour depend on whether the last flush happened to have run.
        """
        if self._store is not None and self._config.persistence_enabled:
            self._flush(force=True)
        self._started = False
        self._report_health(HealthState.DRAINING, "stopped")

    # --- the seam ---------------------------------------------------------------- #

    async def on_tracked(self, camera_id: CameraId, update: TrackUpdate) -> None:
        """Consume one camera's tracks. **Never raises.**

        A registry failure may not stop tracking, which may not stop detection,
        which may not stop acquisition (invariant V9).
        """
        if not self._started:
            return

        self._stats.frames_consumed += 1
        lock = self._locks.get(camera_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[camera_id] = lock

        try:
            async with lock:
                result = self._registry.ingest(camera_id, update)
                self._dirty.add(camera_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the seam is a firewall
            self._stats.frames_failed += 1
            self._metrics.counter(
                MetricName.REGISTRY_FAILURES,
                camera_id=str(camera_id),
                reason="runtime_guard",
            ).increment()
            self._report_health(
                HealthState.DEGRADED, f"unhandled {type(exc).__name__}: {exc}"
            )
            return

        self._publish(result)
        self._maintain()

    def _publish(self, result: RegistryUpdate) -> None:
        if result.failed:
            self._stats.frames_failed += 1
        else:
            self._stats.updates_applied += 1
            self._stats.objects_emitted += result.count

        if self._sink is not None:
            try:
                self._sink(result)
            except Exception:  # noqa: BLE001 - a bad sink must not break the registry
                self._stats.sink_failures += 1

    # --- schedules ---------------------------------------------------------------- #

    def _maintain(self) -> None:
        """Advance horizons and flush, on their own cadences."""
        now = self._clock.monotonic().ns

        if now - self._last_expiry_ns >= self._config.expiry_interval_ms * 1_000_000:
            self._last_expiry_ns = now
            self._expire()

        if (
            self._store is not None
            and self._config.persistence_enabled
            and now - self._last_persist_ns
            >= self._config.persistence_interval_ms * 1_000_000
        ):
            self._last_persist_ns = now
            self._flush()

        self._maybe_report()

    def _expire(self) -> None:
        try:
            removed = self._registry.expire_stale(self._clock.now())
        except Exception:  # noqa: BLE001 - maintenance never stops ingestion
            self._metrics.counter(
                MetricName.REGISTRY_FAILURES, reason="expiry"
            ).increment()
            return
        if removed:
            self._dirty.update(self._registry.partitions)

    def _flush(self, *, force: bool = False) -> None:
        """Persist dirty partitions. Failures degrade durability, not ingestion."""
        if self._store is None:
            return
        targets = tuple(self._dirty) if not force else self._registry.partitions
        for camera_id in targets:
            started = self._clock.monotonic().ns
            try:
                self._store.save(self._snapshot(camera_id))
            except ObjectStoreError:
                self._stats.persist_failures += 1
                self._metrics.counter(
                    MetricName.OBJECT_STORE_FAILURES, camera_id=str(camera_id)
                ).increment()
                continue
            except Exception:  # noqa: BLE001 - a store must not stop the platform
                self._stats.persist_failures += 1
                self._metrics.counter(
                    MetricName.OBJECT_STORE_FAILURES, camera_id=str(camera_id)
                ).increment()
                continue
            self._stats.persists += 1
            self._metrics.counter(
                MetricName.OBJECT_STORE_WRITES, camera_id=str(camera_id)
            ).increment()
            self._metrics.histogram(
                MetricName.OBJECT_STORE_LATENCY_MS, camera_id=str(camera_id)
            ).record((self._clock.monotonic().ns - started) / 1_000_000)
            self._dirty.discard(camera_id)

    def _snapshot(self, camera_id: CameraId) -> PartitionSnapshot:
        stats = self._registry.partition_stats(camera_id)
        return PartitionSnapshot(
            camera_id=camera_id,
            site_id=self._registry.site_id,
            version=stats.version if stats else 0,
            taken_at=self._clock.now(),
            objects=self._registry.objects(camera_id),
            next_local_sequence=stats.ids_minted if stats else 0,
        )

    def _reload(self) -> None:
        """Restore durable objects after a restart.

        07_STATE section 9.3: *"object identity survives, tracks do not"*. Every
        reloaded object is unbound, so the first track to match it re-binds with
        an ``EPOCH_REBIND`` method and explicitly reduced confidence.
        """
        if self._store is None:
            return
        for camera_id in self._registry.partitions:
            try:
                snapshot = self._store.load(camera_id)
            except ObjectStoreError:
                self._metrics.counter(
                    MetricName.OBJECT_STORE_FAILURES, camera_id=str(camera_id)
                ).increment()
                continue
            if snapshot is None:
                continue
            restored = self._registry.restore(snapshot)
            self._metrics.counter(
                MetricName.OBJECTS_RELOADED, camera_id=str(camera_id)
            ).increment(restored)

    def restore_from(self, camera_id: CameraId) -> int:
        """Reload one partition explicitly. Used at boot and in recovery tests."""
        if self._store is None:
            return 0
        snapshot = self._store.load(camera_id)
        if snapshot is None:
            return 0
        restored = self._registry.restore(snapshot)
        self._metrics.counter(
            MetricName.OBJECTS_RELOADED, camera_id=str(camera_id)
        ).increment(restored)
        return restored

    # --- health -------------------------------------------------------------------- #

    def _maybe_report(self) -> None:
        now = self._clock.monotonic().ns
        if now - self._last_report_ns < self._report_interval.ns:
            return
        self._last_report_ns = now
        health = self._registry.health()
        self._report_health(health.state, health.detail)
        self._metrics.gauge(MetricName.REGISTRY_PARTITIONS).set(
            float(len(self._registry.partitions))
        )

    def _report_health(self, state: HealthState, detail: str) -> None:
        self._health.report(
            ComponentHealth(
                component_id=REGISTRY_RUNTIME_ID,
                state=state,
                reported_at=self._clock.now(),
                detail=detail,
                metrics={
                    "frames_consumed": float(self._stats.frames_consumed),
                    "failure_rate": self._stats.failure_rate,
                    "persist_failures": float(self._stats.persist_failures),
                },
            )
        )

    # --- access ---------------------------------------------------------------------- #

    @property
    def stats(self) -> RegistryRuntimeStats:
        return self._stats

    @property
    def started(self) -> bool:
        return self._started

    @property
    def cameras_seen(self) -> int:
        return len(self._locks)

    @property
    def dirty_partitions(self) -> int:
        return len(self._dirty)

    def health(self) -> ComponentHealth:
        return self._registry.health()

    def forget(self, camera_id: CameraId) -> None:
        """Release a camera's lock after it detaches.

        Without this the lock table grows with every camera the process has ever
        seen — a slow leak visible only on long-lived nodes with churning camera
        sets, which is exactly where it is hardest to diagnose.
        """
        self._locks.pop(camera_id, None)
        self._dirty.discard(camera_id)

    async def flush_now(self) -> None:
        """Force a persistence pass. For operators and recovery tests."""
        self._flush(force=True)

    def expire_now(self) -> tuple:
        """Force a horizon pass. For operators and tests on a virtual clock."""
        return self._registry.expire_stale(self._clock.now())


def registry_expiry_due(last_ns: int, now_ns: int, interval_ms: int) -> bool:
    """Whether an expiry pass is due. Extracted so the schedule is testable."""
    return now_ns - last_ns >= interval_ms * 1_000_000


def snapshot_instant(clock: Clock) -> Instant:
    return clock.now()
