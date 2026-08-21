"""The whole platform, assembled — L0 through L7 (Flow 8).

> `08_RUNTIME` §7.1's boot sequence, executed in order:
>
> 1. load + validate config · 2. initialize kernel · 3. discover + validate
> plugins · 4. register models · 5. load + warm models · 6. load prompt packs ·
> 7. initialize storage, recover state · 8. **construct object graph** ·
> 9. start state writers + API · 10. attach camera pipelines · 11. signal readiness

Two details from §7.1 that matter more than they appear, and which this module
implements literally:

**Step 9 precedes step 10.** *"the API serves recovered state before cameras
attach, so consumers reconnecting after a deployment get valid (if briefly stale)
answers rather than errors."* `VisionSystem.boot` starts the API, *then* attaches
pipelines.

**Step 7 precedes step 8.** State is recovered from the log before the object
graph is constructed, so the projection a consumer reads at step 9 is the one the
log describes rather than an empty one that fills in later.

**No bypasses.** Every seam below is the one its own flow declared:

```
kernel → acquisition → detection → tracking → registry → cropping
                                                  ↓          ↓
                                            synthesis ← understanding
                                                  ↓
                                            vision state → observation API
```

Each arrow is a callable one module holds and the other assigned. No module
imports the module above it, and the assembly below is the only place that knows
the whole shape.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field

from .bootstrap import VisionPlatform
from .core.model.api import CapabilitySummary
from .core.model.ids import CameraId
from .core.model.observation import ObservabilityReason, ObservabilityStatus
from .core.model.timebase import Instant
from .exposure import ObservationApi
from .exposure_bootstrap import ExposureLayer, build_exposure_layer
from .state import VisionStateManager
from .state.replay import ReplayReport, ReplayVerifier
from .synthesis import SynthesisRuntime
from .synthesis_bootstrap import SynthesisLayer, build_synthesis_layer


@dataclass(frozen=True, slots=True)
class VisionSystem:
    """Every layer, assembled and wired.

    Returned as a value rather than hidden behind a facade so that an operator or
    a test can reach any single module — the practical expression of *"every
    module is independently testable"* (`01_LAYERED` §2.3).
    """

    platform: VisionPlatform
    synthesis: SynthesisLayer
    exposure: ExposureLayer
    registry_layer: object = None
    cropping: object = None
    understanding: object = None
    detection: object = None
    tracking: object = None
    _started: list[bool] = field(default_factory=lambda: [False])
    _layers: list[str] = field(default_factory=list)

    # --- lifecycle -------------------------------------------------------------- #

    async def boot(self) -> None:
        """08_RUNTIME §7.1, steps 9 through 11.

        The API starts **before** cameras attach. A consumer reconnecting after a
        deployment gets a valid, briefly-stale answer rather than an error, which
        is the difference between a rolling restart and an outage from the
        consumer's point of view.
        """
        await self.synthesis.runtime.start()
        self._layers.append("synthesis")

        for name, layer in (
            ("understanding", self.understanding),
            ("cropping", self.cropping),
            ("registry", self.registry_layer),
            ("tracking", self.tracking),
        ):
            runtime = getattr(layer, "runtime", None)
            if runtime is not None and hasattr(runtime, "start"):
                await runtime.start()
                self._layers.append(name)

        # Step 9 — the API is serving here, with recovered state and no cameras.
        self._started[0] = True

        # Step 10 — cameras attach last, staggered by the runtime.
        await self.platform.boot()

    async def shutdown(self, *, graceful: bool = True):
        """Drain in reverse. Cameras first, the API last.

        Reverse because a consumer should keep receiving answers for as long as
        there is state to answer from. Stopping the API first would blind every
        consumer while the platform was still recording facts they could have
        read.
        """
        result = await self.platform.shutdown(graceful=graceful)
        for layer in (self.registry_layer, self.cropping, self.understanding):
            runtime = getattr(layer, "runtime", None)
            if runtime is not None and hasattr(runtime, "stop"):
                await runtime.stop()
        await self.synthesis.runtime.stop()
        self.hub.close_all()
        self._started[0] = False
        return result

    async def record_restart_gap(
        self, cameras: Sequence[CameraId] = (), *, down_since: Instant | None = None
    ) -> int:
        """Publish a coverage observation for the restart window (07_STATE §9.3).

        > *"The restart gap is **recorded as a coverage observation**, so consumers
        > see the discontinuity as data rather than inferring it from a suspicious
        > silence. Deployments are thereby visible in the record, which is exactly
        > what an operator investigating an anomaly needs."*

        The status is **BLIND**, not observing. A restart is a window during which
        the platform could not see, and the coverage model refuses to let it be
        recorded otherwise — an ``OBSERVING`` window carrying reason ``RESTART``
        is rejected at construction, because a camera that is observing has no
        observability problem to name. That refusal caught this method claiming
        the opposite of what it meant.

        Called by an operator or a supervisor after boot rather than
        automatically, because a first-ever start is not a restart and recording
        a gap before there was anything to interrupt would put a fiction in the
        log.
        """
        published = 0
        now = self.platform.clock.now()
        began = down_since if down_since is not None else now
        for camera_id in cameras or self.state.partitions:
            observation = self.synthesis.runtime.publish_coverage(
                camera_id,
                status=ObservabilityStatus.BLIND,
                reason=ObservabilityReason.RESTART,
                since=began,
                effective_rate=0.0,
            )
            if observation is not None:
                published += 1
        return published

    # --- verification ------------------------------------------------------------ #

    def verify_replay(self, cameras: Sequence[CameraId] = ()) -> tuple[ReplayReport, ...]:
        """Prove every partition rebuilds identically from the log (V13).

        Runs the same projection the live path ran, over the same log, and diffs
        the result field by field. §9.1 places every *"no data loss"* guarantee on
        this property holding.
        """
        verifier = ReplayVerifier(
            clock=self.platform.clock,
            metrics=self.platform.metrics,
            log=self.synthesis.log,
            bounds=self.state._bounds,  # noqa: SLF001 - the bounds the live path used
        )
        snapshot = self.state.snapshot(list(cameras) or None)
        return verifier.verify_all(
            [(camera_id, partition) for camera_id, partition in snapshot.partitions.items()]
        )

    def health(self) -> dict[str, str]:
        """One line per layer, for an operator and for a readiness probe."""
        report = {
            "platform": "healthy" if self._started[0] else "stopped",
            "layers": ",".join(self._layers) or "none",
            "synthesis": self.synthesis.runtime.health().state.value,
            "state": self.state.health().state.value,
            "api": "serving" if self._started[0] else "stopped",
            "subscriptions": str(self.hub.count),
        }
        for name, layer in (
            ("registry", self.registry_layer),
            ("cropping", self.cropping),
            ("understanding", self.understanding),
        ):
            runtime = getattr(layer, "runtime", None)
            if runtime is not None and hasattr(runtime, "health"):
                report[name] = runtime.health().state.value
        return report

    # --- access -------------------------------------------------------------------- #

    @property
    def api(self) -> ObservationApi:
        return self.exposure.api

    @property
    def state(self) -> VisionStateManager:
        return self.synthesis.state

    @property
    def hub(self):
        return self.exposure.hub

    @property
    def runtime(self) -> SynthesisRuntime:
        return self.synthesis.runtime

    @property
    def started(self) -> bool:
        return self._started[0]

    @property
    def started_layers(self) -> tuple[str, ...]:
        """Which layers ``boot`` actually started.

        Reported because a layer this system was never given is a layer that
        silently does nothing: `assemble` takes each as optional so a deployment
        can run L1-L5 without exposure, and the cost of that flexibility is that
        omitting one by accident looks exactly like omitting it on purpose. This
        turns the difference into something an operator — or a test — can see.
        """
        return tuple(self._layers)


def assemble(
    platform: VisionPlatform,
    *,
    registry_layer,
    cropping=None,
    understanding=None,
    detection=None,
    tracking=None,
    log=None,
    sinks: Sequence = (),
    attributes=None,
    authorizer=None,
    evidence=None,
    grants: Sequence = (),
    capabilities: CapabilitySummary | None = None,
    audit_sinks: Sequence = (),
) -> VisionSystem:
    """Build L5–L7 over an already-assembled L0–L4 and wire every seam.

    Takes the lower layers rather than building them, because each has its own
    composition root with its own adapter choices, and collapsing eight bootstraps
    into one would put every deployment decision in a single function.

    **The Observation API is wired to fan out published observations**, and this
    is the one seam Flow 8 adds to the write path. It is not a write: the
    observations already exist and are already recorded; the hub delivers copies
    to whoever subscribed. §M14 owns *"subscription sessions"*, and this is where
    they are fed.
    """
    synthesis = build_synthesis_layer(
        platform,
        registry_layer,
        attributes=attributes,
        log=log,
        sinks=sinks,
        attach=True,
    )

    exposure = build_exposure_layer(
        platform,
        synthesis.state,
        authorizer=authorizer,
        evidence=evidence,
        audit_sinks=audit_sinks,
        demand_registry=getattr(cropping, "demands", None),
        capabilities=capabilities,
        grants=grants,
    )

    if understanding is not None:
        from .synthesis_bootstrap import attach_understanding

        attach_understanding(understanding, synthesis.runtime)

    _attach_api_fanout(synthesis.runtime, exposure.api)

    return VisionSystem(
        platform=platform,
        synthesis=synthesis,
        exposure=exposure,
        registry_layer=registry_layer,
        cropping=cropping,
        understanding=understanding,
        detection=detection,
        tracking=tracking,
    )


def _attach_api_fanout(runtime: SynthesisRuntime, api: ObservationApi) -> None:
    """Deliver published observations to subscribers.

    Wrapped rather than replacing the existing sinks: M11 may already be fanning
    out to a data lake or a message bus through P19, and the API is one more
    consumer rather than the only one.

    **Never blocks and never raises into the write path.** A subscriber that
    cannot keep up degrades itself — its own bounded queue fills and it receives
    a `Gap`. 09_API §3.4: *"Never: stall the platform."*
    """
    existing = getattr(runtime, "_sinks", ())

    class _ApiSink:
        """A P19 sink that fans out to subscriptions.

        Declares itself **not durable** (obligation K5): delivering to a live
        subscriber is not recording a fact, and a deployment must never mistake a
        connected dashboard for a system of record.
        """

        __slots__ = ()

        @property
        def sink_id(self) -> str:
            return "sink.api_fanout"

        @property
        def durable(self) -> bool:
            return False

        def emit(self, observations):
            from .core.ports.synthesis import SinkResult

            try:
                delivered = api.publish(observations)
            except Exception:  # noqa: BLE001 - fan-out must not break publication
                return SinkResult(accepted=0, failed=len(observations))
            return SinkResult(accepted=delivered)

    runtime._sinks = (*existing, _ApiSink())  # noqa: SLF001 - the declared seam


async def drain(system: VisionSystem, *, ticks: int = 20) -> None:
    """Let scheduled seam tasks complete.

    The registry and understanding seams schedule their hand-off rather than
    awaiting it, so that synthesis latency never lands on a lower layer's
    critical path (V9). A caller that needs the results *now* — a test, or a
    shutdown — drains the loop.
    """
    for _ in range(ticks):
        await asyncio.sleep(0)
