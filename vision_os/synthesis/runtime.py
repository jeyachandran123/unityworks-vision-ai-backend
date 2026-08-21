"""The Synthesis Runtime — the ``UnderstandingConsumer`` seam and the state edge.

> **Single responsibility:** *Own the builder's lifecycle, serialize each
> camera's synthesis, and hand facts to state. Decide nothing yourself.*

08_RUNTIME §1 places M11 in two rows at once — *actor* for its order-dependent
suppression state, *worker pool* for assembly, which is a pure function. This
runtime honours both: one lock per camera around the suppression memory, and
nothing shared between cameras at all.

**The two queue policies that meet here.** 08_RUNTIME §5.2:

| Edge | Policy | Why |
|---|---|---|
| `Crop → Understanding` | `drop_oldest` | *"losing an enrichment is acceptable"* |
| `Builder → State` | **`block`** | *"Observations must not be lost — this is the system of record (V5)"* |

> *"That asymmetry is the whole philosophy of the platform expressed as queue
> configuration."*

So this runtime **never drops an observation**. When state cannot accept one, the
runtime surfaces the failure rather than shedding — because on this edge, unlike
every edge before it, silence is the unacceptable outcome.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from ..core.errors import (
    CommitFailedError,
    PartitionDegradedError,
    ValidationFailedError,
)
from ..core.model.health import ComponentHealth, HealthState
from ..core.model.ids import CameraId, ModuleId, ObjectId
from ..core.model.observation import (
    LifecycleTransition,
    MeasurementBasis,
    ObservabilityReason,
    ObservabilityStatus,
    Observation,
)
from ..core.model.timebase import ClockQuality, Duration, Instant
from ..core.model.understanding import UnderstandingResult
from ..core.model.visual_object import VisualObject
from ..core.ports.clock import Clock
from ..core.ports.synthesis import ObservationSinkPort
from ..kernel.config.schema import SynthesisSection
from ..kernel.health import HealthMonitor
from ..kernel.metrics import MetricName, MetricsEngine
from .builder.engine import BuildContext, ObservationBuilder

SYNTHESIS_RUNTIME_ID = ModuleId("synthesis_runtime")

_DEFAULT_REPORT_INTERVAL = Duration.from_millis(1_000)


class SynthesisRuntimeStats:
    """Mutable counters. Updated on the hot path, never published as a value."""

    __slots__ = (
        "attribute_results",
        "commit_failures",
        "coverage_published",
        "objects_seen",
        "observations_built",
        "registry_updates",
        "rejected",
        "sink_failures",
        "suppressed",
    )

    def __init__(self) -> None:
        self.registry_updates = 0
        self.attribute_results = 0
        self.objects_seen = 0
        self.observations_built = 0
        self.suppressed = 0
        self.rejected = 0
        self.coverage_published = 0
        self.commit_failures = 0
        self.sink_failures = 0

    @property
    def suppression_rate(self) -> float:
        total = self.observations_built + self.suppressed
        return self.suppressed / total if total else 0.0


class SynthesisRuntime:
    """Implements the understanding-to-synthesis seam; owns builder lifecycle."""

    def __init__(
        self,
        *,
        clock: Clock,
        metrics: MetricsEngine,
        health: HealthMonitor,
        builder: ObservationBuilder,
        config: SynthesisSection,
        state=None,
        sinks: Sequence[ObservationSinkPort] = (),
        taxonomy_version: str = "",
        report_interval: Duration = _DEFAULT_REPORT_INTERVAL,
    ) -> None:
        self._clock = clock
        self._metrics = metrics
        self._health = health
        self._builder = builder
        self._config = config
        self._state = state
        self._sinks = tuple(sinks)
        self._taxonomy_version = taxonomy_version
        self._report_interval = report_interval

        self._stats = SynthesisRuntimeStats()
        self._locks: dict[CameraId, asyncio.Lock] = {}
        self._lifecycles: dict[ObjectId, str] = {}
        self._started = False
        self._last_report_ns = 0

    # --- lifecycle -------------------------------------------------------------- #

    async def start(self) -> None:
        self._started = True
        self._report_health(HealthState.HEALTHY, "synthesis ready")

    async def stop(self) -> None:
        self._started = False
        self._report_health(HealthState.DRAINING, "stopped")

    # --- the seams ---------------------------------------------------------------- #

    async def on_registered(self, update) -> None:
        """Consume a registry update — presence, spatial and lifecycle facts.

        `01_LAYERED` §3.1's **dotted edges**: *"detection, tracking, and registry
        results become observations without passing through understanding.
        Understanding is enrichment, not a toll gate."* This seam is that
        sentence: a presence observation is published whether or not any model
        ever ran.

        **Never raises.** A synthesis failure may not stop the registry, which may
        not stop tracking, which may not stop detection (V9).
        """
        if not self._started or not self._config.enabled:
            return
        if getattr(update, "failed", False):
            # A broken registry must not read as "nothing was here": evaluating
            # its empty object list would manufacture an empty-scene conclusion
            # from an upstream fault (V8).
            return

        self._stats.registry_updates += 1
        camera_id = update.camera_id
        lock = self._locks.setdefault(camera_id, asyncio.Lock())

        try:
            async with lock:
                built = self._build_from_registry(update)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the seam is a firewall
            self._report_health(
                HealthState.DEGRADED, f"unhandled {type(exc).__name__}: {exc}"
            )
            return

        self._commit(camera_id, built)
        self._maybe_report()

    async def on_understood(self, results: Sequence[UnderstandingResult]) -> None:
        """Consume understanding results — attribute facts.

        M9's results arrive here, not at state: 07_STATE §1.1 makes the
        observation the only write path, so an attribute becomes state only by
        first becoming a published fact.
        """
        if not self._started or not self._config.enabled or not results:
            return

        by_camera: dict[CameraId, list[UnderstandingResult]] = {}
        for result in results:
            by_camera.setdefault(result.camera_id, []).append(result)

        for camera_id, batch in by_camera.items():
            lock = self._locks.setdefault(camera_id, asyncio.Lock())
            try:
                async with lock:
                    built = self._build_from_understanding(camera_id, batch)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the seam is a firewall
                self._report_health(
                    HealthState.DEGRADED, f"unhandled {type(exc).__name__}: {exc}"
                )
                continue
            self._commit(camera_id, built)

        self._maybe_report()

    def publish_coverage(
        self,
        camera_id: CameraId,
        *,
        status: ObservabilityStatus,
        reason: ObservabilityReason,
        since: Instant,
        effective_rate: float = 1.0,
        context: BuildContext | None = None,
    ) -> Observation | None:
        """Emit a coverage observation. **Never suppressed.**

        02_VOM §11.2: *"the difference between a platform that is honest about
        its limits and one that is dangerously silent."* Called by the runtime
        supervisor on a stream loss, a scheduler alarm, a budget exhaustion or a
        restart — every mechanism 07_STATE §7.2 enumerates.
        """
        if not self._config.enabled:
            return None
        built = self._builder.build_coverage(
            context or self._coverage_context(camera_id),
            status=status,
            reason=reason,
            since=since,
            effective_rate=effective_rate,
        )
        self._stats.coverage_published += 1
        self._commit(camera_id, [built])
        return built

    # --- assembly ------------------------------------------------------------------ #

    def _build_from_registry(self, update) -> list[Observation]:
        """Presence, spatial and lifecycle observations for one frame."""
        context = self._context_for(update)
        built: list[Observation] = []

        for obj in update.objects:
            self._stats.objects_seen += 1
            built.extend(self._object_observations(obj, context))

        for change in getattr(update, "lifecycle_changes", ()):
            object_id, previous, current = change
            obj = next((o for o in update.objects if o.object_id == object_id), None)
            if obj is None or previous == current:
                continue
            observation = self._safely(
                self._builder.build_lifecycle,
                obj,
                LifecycleTransition(previous=previous, current=current),
                context,
            )
            if observation is not None:
                built.append(observation)

        return built

    def _object_observations(
        self, obj: VisualObject, context: BuildContext
    ) -> list[Observation]:
        basis = (
            MeasurementBasis.MEASURED
            if obj.last_confirmed.ns >= obj.last_seen.ns
            else MeasurementBasis.PREDICTED
        )
        built: list[Observation] = []
        for build in (self._builder.build_presence, self._builder.build_spatial):
            observation = self._safely(build, obj, context, basis=basis)
            if observation is not None:
                built.append(observation)
            else:
                self._stats.suppressed += 1
        return built

    def _build_from_understanding(
        self, camera_id: CameraId, results: Sequence[UnderstandingResult]
    ) -> list[Observation]:
        built: list[Observation] = []
        for result in results:
            self._stats.attribute_results += 1
            obj = self._object_for(result)
            if obj is None:
                continue
            context = BuildContext(
                camera_id=camera_id,
                tenant_id=result.tenant_id,
                site_id=result.site_id,
                frame_ref=result.evidence.frame_ref,
                t_capture=self._capture_of(result),
                clock_quality=ClockQuality.UNKNOWN,
                taxonomy_version=self._taxonomy_version,
            )
            built.extend(
                self._safely_many(
                    self._builder.build_attribute, obj, result, context
                )
            )
        return built

    def _object_for(self, result: UnderstandingResult) -> VisualObject | None:
        """The subject an attribute result describes — a **carrier, not a fact**.

        Reconstructed from the result rather than fetched from M7: reaching into
        the registry would give L5 a read dependency on L2 that the layered
        dependency law does not grant, and the result already carries everything
        the envelope needs about its subject. The ``ObjectId`` is M7's, arriving
        by way of the crop and the request; nothing here mints an identity.

        **What it deliberately does not carry is confidence.** M11 does not know
        how sure M7 is that this track is this object, and stamping 1.0 would
        publish a certainty nobody measured — *"never fabricate certainty"*. The
        placeholder below is the weakest value the type permits and it never
        reaches an observation: ``build_attribute`` is called with an explicit
        ``confidence=None``, because an attribute observation's confidence lives
        on each attribute, where M9 measured it.
        """
        from ..core.model.confidence import Confidence, ConfidenceSemantics
        from ..core.model.space import FrameOfReference, SpatialInfo
        from ..core.model.visual_object import LifecycleState

        if result.object_id is None:
            return None
        return VisualObject(
            object_id=result.object_id,
            tenant_id=result.tenant_id,
            site_id=result.site_id,
            camera_id=result.camera_id,
            class_id=result.class_id,
            confidence=Confidence.uncalibrated(0.0, ConfidenceSemantics.IDENTITY),
            lifecycle=LifecycleState.ACTIVE,
            class_history=(),
            track_bindings=(),
            current_spatial=SpatialInfo(frame_of_reference=FrameOfReference.NORMALIZED),
            spatial_history=(),
            attributes={},
            first_seen=self._capture_of(result),
            last_seen=self._capture_of(result),
            last_confirmed=self._capture_of(result),
            observation_count=0,
            provenance=result.provenance,
        )

    @staticmethod
    def _capture_of(result: UnderstandingResult) -> Instant:
        for attribute in result.attributes:
            return attribute.observed_at
        return Instant(0)

    def _context_for(self, update) -> BuildContext:
        first = update.objects[0] if update.objects else None
        return BuildContext(
            camera_id=update.camera_id,
            tenant_id=first.tenant_id if first else "",
            site_id=first.site_id if first else "",
            frame_ref=update.frame_ref,
            t_capture=first.last_seen if first else self._clock.now(),
            clock_quality=ClockQuality.UNKNOWN,
            taxonomy_version=self._taxonomy_version,
        )

    def _coverage_context(self, camera_id: CameraId) -> BuildContext:
        from ..core.model.ids import FrameRef, FrameSeq, StreamEpoch

        return BuildContext(
            camera_id=camera_id,
            tenant_id="",
            site_id="",
            frame_ref=FrameRef(camera_id, StreamEpoch(0), FrameSeq(0)),
            t_capture=self._clock.now(),
            clock_quality=ClockQuality.UNKNOWN,
            taxonomy_version=self._taxonomy_version,
        )

    def _safely(self, build, *args, **kwargs) -> Observation | None:
        """Run one builder, converting a refusal into a counted non-event.

        A rejected observation must not stop the rest of the frame: section M11
        rejects *that* envelope, and the object's other observations are
        unaffected.

        Arguments are forwarded rather than closed over, so a builder called in a
        loop binds the current iteration's values — a lambda here would capture
        the variable and every deferred call would use the last object.
        """
        try:
            return build(*args, **kwargs)
        except ValidationFailedError:
            self._stats.rejected += 1
            return None

    def _safely_many(self, build, *args, **kwargs) -> list[Observation]:
        try:
            return list(build(*args, **kwargs))
        except ValidationFailedError:
            self._stats.rejected += 1
            return []

    # --- the state edge -------------------------------------------------------------- #

    def _commit(self, camera_id: CameraId, observations: Sequence[Observation]) -> None:
        """Hand facts to state and to every sink. **Never drops.**

        08_RUNTIME §5.2 gives this edge `block`, not `drop_oldest`, because
        *"observations must not be lost — this is the system of record (V5)."* A
        commit failure is surfaced and counted; it is never a silent shed.
        """
        if not observations:
            return

        self._stats.observations_built += len(observations)
        self._metrics.counter(
            MetricName.OBSERVATIONS_BUILT, camera_id=str(camera_id)
        ).increment(0)

        for sink in self._sinks:
            try:
                sink.emit(observations)
            except Exception:  # noqa: BLE001 - a bad sink must not break synthesis
                self._stats.sink_failures += 1

        if self._state is None:
            return
        try:
            self._state.append(observations)
        except (CommitFailedError, PartitionDegradedError):
            # Loudly, not silently. The partition is halted and says so; the
            # facts are not quietly discarded to keep the pipeline moving.
            self._stats.commit_failures += 1
            self._report_health(
                HealthState.DEGRADED,
                f"state refused {len(observations)} observation(s) for "
                f"'{camera_id}'; facts are not being recorded",
            )
        except Exception:  # noqa: BLE001 - state must not take synthesis down
            self._stats.commit_failures += 1

    # --- observability ---------------------------------------------------------------- #

    def _maybe_report(self) -> None:
        now = self._clock.monotonic().ns
        if now - self._last_report_ns < self._report_interval.ns:
            return
        self._last_report_ns = now
        health = self._builder.health()
        self._report_health(health.state, health.detail)
        self._metrics.gauge(MetricName.OBSERVATIONS_SUPPRESSED).set(
            float(self._builder.suppression.suppressed)
        )

    def _report_health(self, state: HealthState, detail: str) -> None:
        self._health.report(
            ComponentHealth(
                component_id=SYNTHESIS_RUNTIME_ID,
                state=state,
                reported_at=self._clock.now(),
                detail=detail,
                metrics={
                    "observations_built": float(self._stats.observations_built),
                    "suppression_rate": self._stats.suppression_rate,
                    "rejected": float(self._stats.rejected),
                    "commit_failures": float(self._stats.commit_failures),
                },
            )
        )

    # --- access -------------------------------------------------------------------------- #

    @property
    def stats(self) -> SynthesisRuntimeStats:
        return self._stats

    @property
    def started(self) -> bool:
        return self._started

    @property
    def builder(self) -> ObservationBuilder:
        return self._builder

    @property
    def cameras_seen(self) -> int:
        return len(self._locks)

    def health(self) -> ComponentHealth:
        return self._builder.health()

    def forget(self, camera_id: CameraId) -> None:
        """Release a camera's lock and suppression state after it detaches."""
        self._locks.pop(camera_id, None)
        self._builder.forget_camera(camera_id)


@dataclass(frozen=True, slots=True)
class SynthesisReport:
    """What one synthesis pass produced. For operators and tests."""

    observations: tuple[Observation, ...] = ()
    suppressed: int = 0
    rejected: int = 0

    @property
    def published(self) -> int:
        return len(self.observations)

    @property
    def suppression_rate(self) -> float:
        total = self.published + self.suppressed
        return self.suppressed / total if total else 0.0
