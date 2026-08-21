"""Camera to Vision State, through the real Flow 1-7 platform.

The proof that Flow 7 attaches at documented seams and nowhere else: a full
platform boots, frames flow from an in-memory source through decode, masking,
buffering and admission; detection resumes that path; tracking resumes
detection's; the registry consumes tracking; the Crop Manager consumes the
registry — and the Observation Builder consumes the registry too, publishing
facts that reach Vision State without any model having run.

Also exercises the composition root, which is the only module that selects a
suppression policy, an observation log or a sink.
"""

from __future__ import annotations

import asyncio

import pytest

from vision_os.adapters.registry import InMemoryObjectStore
from vision_os.adapters.synthesis import (
    CollectingSink,
    FileObservationLog,
    InMemoryObservationLog,
)
from vision_os.core.errors import ObservationError, StateError
from vision_os.core.model.ids import CameraId
from vision_os.core.model.observation import ObservationType
from vision_os.core.model.vision_state import ConsistencyLevel
from vision_os.registry_bootstrap import build_registry_layer
from vision_os.synthesis_bootstrap import (
    build_suppression_policy,
    build_synthesis_layer,
)

from ...cropping.integration.test_end_to_end import (
    build_stack,
    cropping_document,
    make_platform,
    pump,
)
from ...registry.integration.test_end_to_end import _TrackingToRegistry

CAMERA = CameraId("cam-01")


def synthesis_document(**overrides) -> dict:
    """Flow 1-7 configuration."""
    document = cropping_document()
    document["synthesis"] = {
        "enabled": True,
        "suppression_policy": "suppression.exact",
        "heartbeat_ms": 30_000,
        **overrides,
    }
    document["state"] = {"enabled": True, "max_objects_per_partition": 64}
    return document


async def run_full_pipeline(clock, document, *, log=None, sinks=()):
    """Boot every flow including synthesis, and pump frames through all of them."""
    from vision_os.perception.detection import DetectionRuntime

    platform, detection, tracking, registry_layer, cropping = await build_stack(
        clock, document
    )
    synthesis = build_synthesis_layer(
        platform,
        registry_layer,
        attributes=cropping.capabilities_registry
        if hasattr(cropping, "capabilities_registry")
        else None,
        log=log or InMemoryObservationLog(),
        sinks=sinks,
        attach=True,
    )

    bridge = _TrackingToRegistry(registry_layer.runtime)
    tracking.runtime._sink = bridge  # noqa: SLF001 - the Flow 3/4 seam

    runtime = DetectionRuntime(
        clock=platform.clock,
        bus=platform.bus,
        metrics=platform.metrics,
        health=platform.health,
        engine=detection.engine,
        consumer=tracking.runtime,
    )
    platform.runtime._admitted_consumer = runtime  # noqa: SLF001
    await detection.start()
    await runtime.start()
    await tracking.runtime.start()
    await registry_layer.runtime.start()
    await cropping.runtime.start()
    await synthesis.runtime.start()
    await platform.boot()

    await pump(clock, lambda: len(bridge.pending) >= 5)
    await bridge.drain(tracking)
    for _ in range(40):
        await asyncio.sleep(0)

    await detection.stop()
    await platform.shutdown()
    return platform, registry_layer, synthesis


class TestTheCompositionRoot:
    def test_the_layer_assembles(self, clock) -> None:
        platform = make_platform(clock, synthesis_document())
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        layer = build_synthesis_layer(platform, registry_layer, attach=False)
        assert layer.policy_id == "suppression.exact"
        assert layer.state is not None

    def test_a_disabled_synthesis_layer_refuses_to_build(self, clock) -> None:
        """A site that does not want published facts should not build a layer
        that publishes nothing — the second is harder to diagnose.
        """
        document = synthesis_document()
        document["synthesis"]["enabled"] = False
        platform = make_platform(clock, document)
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        with pytest.raises(ObservationError, match="synthesis.enabled is false"):
            build_synthesis_layer(platform, registry_layer, attach=False)

    def test_a_disabled_state_layer_refuses_to_build(self, clock) -> None:
        """Observations with nowhere to go are worse than no observations."""
        document = synthesis_document()
        document["state"]["enabled"] = False
        platform = make_platform(clock, document)
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        with pytest.raises(StateError, match="state.enabled is false"):
            build_synthesis_layer(platform, registry_layer, attach=False)

    def test_an_unknown_suppression_policy_is_refused_by_name(self, clock) -> None:
        """Refusing beats defaulting.

        A typo falling back to ``always`` would multiply output volume by 10-50x
        with no signal but the storage bill.
        """
        platform = make_platform(
            clock, synthesis_document(suppression_policy="suppression.typo")
        )
        with pytest.raises(ObservationError, match="unknown suppression policy"):
            build_suppression_policy(platform)

    def test_every_shipped_policy_is_selectable_by_name(self, clock) -> None:
        for name in ("suppression.exact", "suppression.threshold", "suppression.always"):
            platform = make_platform(
                clock, synthesis_document(suppression_policy=name)
            )
            assert build_suppression_policy(platform).policy_id == name

    def test_an_adapter_failing_its_kit_is_refused(self, clock) -> None:
        """V3 as a gate rather than an aspiration.

        A non-idempotent log would corrupt the record on every recovery, which is
        exactly the failure a restart is supposed to survive.
        """
        platform = make_platform(clock, synthesis_document())
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        with pytest.raises(ObservationError, match="failed conformance"):
            build_synthesis_layer(
                platform, registry_layer, log=_NonIdempotentLog(), attach=False
            )


class TestTheWholePipeline:
    async def test_observations_reach_vision_state(self, clock) -> None:
        _, _, synthesis = await run_full_pipeline(clock, synthesis_document())
        snapshot = synthesis.state.snapshot()
        assert snapshot.partitions, "no partition was ever written"
        assert any(p.objects for p in snapshot.partitions.values())

    async def test_facts_are_published_without_any_model_running(
        self, clock
    ) -> None:
        """The dotted edge, end to end.

        No understander is bound in this stack. Presence still reaches state,
        which is what *"understanding is enrichment, not a toll gate"* means in
        practice.
        """
        _, _, synthesis = await run_full_pipeline(clock, synthesis_document())
        logged = tuple(synthesis.log.read(CAMERA, limit=500))
        assert logged
        assert {o.observation_type for o in logged} <= {
            ObservationType.PRESENCE,
            ObservationType.SPATIAL,
            ObservationType.LIFECYCLE,
            ObservationType.COVERAGE,
            ObservationType.QUALITY,
        }
        assert not any(o.attributes for o in logged)

    async def test_the_state_matches_the_log(self, clock) -> None:
        """§9.1: the log is authoritative, and a rebuild must prove it."""
        _, _, synthesis = await run_full_pipeline(clock, synthesis_document())
        before = synthesis.state.snapshot().partitions[CAMERA]
        synthesis.state.rebuild(CAMERA)
        after = synthesis.state.snapshot().partitions[CAMERA]
        assert after.objects.keys() == before.objects.keys()

    async def test_a_sink_receives_what_the_log_received(self, clock) -> None:
        sink = CollectingSink()
        _, _, synthesis = await run_full_pipeline(
            clock, synthesis_document(), sinks=(sink,)
        )
        logged = tuple(synthesis.log.read(CAMERA, limit=500))
        assert len(sink.observations) == len(logged)

    async def test_earlier_flows_never_learn_synthesis_exists(self, clock) -> None:
        """The registry holds a callable and never learns what implements it.

        If it imported M11, the dependency would run downward and the registry
        could no longer be deployed without the layer above it.
        """
        _, registry_layer, _ = await run_full_pipeline(clock, synthesis_document())
        source = type(registry_layer.runtime).__module__
        module = __import__(source, fromlist=["__file__"])
        text = open(module.__file__, encoding="utf-8").read()
        assert "synthesis" not in text
        assert "VisionStateManager" not in text

    async def test_a_single_camera_snapshot_is_strongly_consistent(
        self, clock
    ) -> None:
        _, _, synthesis = await run_full_pipeline(clock, synthesis_document())
        snapshot = synthesis.state.snapshot()
        assert len(snapshot.partitions) == 1, "the document declares one camera"
        assert snapshot.consistency is ConsistencyLevel.STRONG


class TestDurableLogRoundTrip:
    async def test_a_file_log_survives_a_restart(self, clock, tmp_path) -> None:
        """The point of binding a durable P20 adapter.

        A restart that lost the log would lose the record, and 07_STATE §9.1's
        *"replay from the last committed log position"* would have nothing to
        replay from.
        """
        log = FileObservationLog(tmp_path)
        _, _, synthesis = await run_full_pipeline(
            clock, synthesis_document(), log=log
        )
        written = len(tuple(log.read(CAMERA, limit=500)))
        assert written, "the pipeline published nothing; the seam is broken"

        reopened = FileObservationLog(tmp_path)
        assert len(tuple(reopened.read(CAMERA, limit=500))) == written

    async def test_reappending_the_same_observation_is_a_no_op(
        self, clock, tmp_path
    ) -> None:
        """Idempotency by observation id is what makes recovery safe.

        Without it, every replay after a crash would double-count, and the
        permanent record would drift a little further from the truth each time.
        """
        log = FileObservationLog(tmp_path)
        _, _, synthesis = await run_full_pipeline(
            clock, synthesis_document(), log=log
        )
        existing = tuple(log.read(CAMERA, limit=500))
        assert existing, "the pipeline published nothing; the seam is broken"

        log.append(CAMERA, existing)
        assert len(tuple(log.read(CAMERA, limit=500))) == len(existing)


class _NonIdempotentLog:
    """A P20 adapter that appends the same observation twice.

    The defect the conformance kit exists to catch, implemented deliberately so
    the gate can be shown to close.
    """

    __slots__ = ("_records",)

    def __init__(self) -> None:
        self._records: dict[CameraId, list] = {}

    @property
    def adapter_id(self) -> str:
        return "log.non_idempotent"

    @property
    def durable(self) -> bool:
        return False

    def append(self, camera_id: CameraId, observations):
        from vision_os.core.model.ids import LogPosition
        from vision_os.core.ports.synthesis import LogAppendResult

        held = self._records.setdefault(camera_id, [])
        held.extend(observations)
        return LogAppendResult(
            appended=len(observations), position=LogPosition(len(held))
        )

    def read(self, camera_id: CameraId, *, start=None, end=None, limit: int = 1000):
        return tuple(self._records.get(camera_id, ())[:limit])

    def position(self, camera_id: CameraId):
        from vision_os.core.model.ids import LogPosition

        return LogPosition(len(self._records.get(camera_id, ())))

    def trim(self, camera_id: CameraId, *, before) -> int:
        return 0
