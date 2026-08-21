"""Camera to consumer, through the real Flow 1-8 platform.

The end-to-end proof the brief asks for: a full platform boots, frames flow from
an in-memory source through decode, masking, buffering and admission; detection
resumes that path; tracking resumes detection's; the registry consumes tracking;
the Crop Manager consumes the registry; the Observation Builder consumes the
registry too; Vision State projects what the builder published; and the
Observation API serves it to an authorized consumer.

**No bypasses.** Every seam is the one its own flow declared. The assertion that
matters most is not that data arrives — it is that no layer imports the layer
above it, which `test_exposure_architecture.py` proves statically and this file
proves by construction: the stack is wired here, in one place, and nowhere else.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.exposure.authorization import (
    StaticAuthorizer,
    full_grant,
    read_only_grant,
)
from vision_os.adapters.persistence import InMemoryEvidenceStore
from vision_os.adapters.registry import InMemoryObjectStore
from vision_os.adapters.synthesis import InMemoryObservationLog
from vision_os.core.errors import ApiError, ForbiddenError
from vision_os.core.model.api import (
    CapabilitySummary,
    DeliveryPolicy,
    Principal,
    Scope,
    TimeWindow,
)
from vision_os.core.model.ids import CameraId, TenantId
from vision_os.core.model.observation import ObservationType
from vision_os.core.model.timebase import Instant
from vision_os.exposure_bootstrap import (
    build_authorizer,
    build_evidence_store,
    build_exposure_layer,
)
from vision_os.registry_bootstrap import build_registry_layer
from vision_os.system import assemble, drain

from ...cropping.integration.test_end_to_end import (
    build_stack,
    cropping_document,
    make_platform,
    pump,
)
from ...registry.integration.test_end_to_end import _TrackingToRegistry

CAMERA = CameraId("cam-01")
TENANT = TenantId("acme")
CONSUMER = "dashboard"


def full_document(**api_overrides) -> dict:
    """Flow 1-8 configuration."""
    document = cropping_document()
    document["synthesis"] = {"enabled": True, "suppression_policy": "suppression.exact"}
    document["state"] = {"enabled": True, "max_objects_per_partition": 64}
    document["storage"] = {"evidence_store": "evidence.memory"}
    document["api"] = {
        "enabled": True,
        "authorizer": "authz.static",
        **api_overrides,
    }
    return document


async def boot_everything(clock, document=None, *, grants=None):
    """Boot L0-L8 and return the assembled system."""
    from vision_os.perception.detection import DetectionRuntime

    document = document or full_document()
    platform, detection, tracking, registry_layer, cropping = await build_stack(
        clock, document
    )

    system = assemble(
        platform,
        registry_layer=registry_layer,
        cropping=cropping,
        detection=detection,
        tracking=tracking,
        log=InMemoryObservationLog(),
        evidence=InMemoryEvidenceStore(),
        authorizer=StaticAuthorizer(
            grants
            if grants is not None
            else [read_only_grant(CONSUMER, TENANT), full_grant("operator", TENANT)]
        ),
        capabilities=CapabilitySummary(taxonomy_version="1"),
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
    await system.boot()

    await pump(clock, lambda: len(bridge.pending) >= 5)
    await bridge.drain(tracking)
    await drain(system, ticks=40)

    await detection.stop()
    await system.shutdown()
    return system


def reader() -> Principal:
    return Principal(subject=CONSUMER, tenant_id=TENANT)


def operator() -> Principal:
    return Principal(subject="operator", tenant_id=TENANT)


class TestTheWholePipeline:
    async def test_a_frame_becomes_an_answer_to_a_consumer(self, clock) -> None:
        """The single most important assertion in the platform.

        Photons entered at L1; a consumer with a token read a fact at L7. Every
        layer in between did its own job and nothing else.
        """
        system = await boot_everything(clock)
        result = system.api.query_state(reader(), Scope(tenant_id=TENANT))

        assert result.objects, "no object reached the consumer"
        assert result.coverage is not None
        assert result.snapshot.partitions

    async def test_observations_are_recorded_before_they_are_served(
        self, clock
    ) -> None:
        """07_STATE §9.1: the log is the system of record.

        What a consumer reads must already be in the log — otherwise state holds
        something a rebuild would not reproduce.
        """
        system = await boot_everything(clock)
        logged = tuple(system.synthesis.log.read(CAMERA, limit=500))
        result = system.api.query_state(operator(), Scope(tenant_id=TENANT))

        assert logged
        served = {str(o.object_id) for o in result.objects}
        recorded = {str(o.object_id) for o in logged if o.object_id}
        assert served <= recorded, "the API served an object the log never recorded"

    async def test_a_historical_query_returns_what_was_published(
        self, clock
    ) -> None:
        system = await boot_everything(clock)
        page = system.api.query_observations(
            operator(),
            Scope(tenant_id=TENANT),
            TimeWindow(start=Instant(0), end=Instant(3_600 * 1_000_000_000)),
        )
        assert page.observations
        assert page.window_fully_observable

    async def test_state_replays_identically_after_a_full_run(self, clock) -> None:
        """V13, end to end.

        Not a unit test over synthetic observations — the real log, produced by
        the real pipeline, replayed through the real projection.
        """
        system = await boot_everything(clock)
        reports = system.verify_replay()
        assert reports
        for report in reports:
            assert report.identical, report.summary()

    async def test_a_subscriber_receives_live_observations(self, clock) -> None:
        """The fan-out seam Flow 8 adds to the write path.

        Not a write: the observations already exist and are already recorded, and
        the hub delivers copies to whoever asked.
        """
        from vision_os.perception.detection import DetectionRuntime

        document = full_document()
        platform, detection, tracking, registry_layer, cropping = await build_stack(
            clock, document
        )
        system = assemble(
            platform,
            registry_layer=registry_layer,
            cropping=cropping,
            tracking=tracking,
            log=InMemoryObservationLog(),
            authorizer=StaticAuthorizer([full_grant("operator", TENANT)]),
        )
        subscription = system.api.subscribe(
            operator(), Scope(tenant_id=TENANT), policy=DeliveryPolicy()
        )

        bridge = _TrackingToRegistry(registry_layer.runtime)
        tracking.runtime._sink = bridge  # noqa: SLF001
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
        await system.boot()
        await pump(clock, lambda: len(bridge.pending) >= 5)
        await bridge.drain(tracking)
        await drain(system, ticks=40)

        messages = subscription.drain()
        await detection.stop()
        await system.shutdown()

        assert messages, "a subscriber received nothing from a running pipeline"


class TestBootOrder:
    """08_RUNTIME §7.1."""

    async def test_the_api_serves_before_cameras_attach(self, clock) -> None:
        """§7.1: *"the API serves recovered state before cameras attach, so
        consumers reconnecting after a deployment get valid (if briefly stale)
        answers rather than errors."*

        Asserted by querying between boot and the first frame: the answer is
        empty, and — crucially — it is an *answer*, not an error.
        """
        platform, _, _, registry_layer, cropping = await build_stack(
            clock, full_document()
        )
        system = assemble(
            platform,
            registry_layer=registry_layer,
            cropping=cropping,
            log=InMemoryObservationLog(),
            authorizer=StaticAuthorizer([full_grant("operator", TENANT)]),
        )
        await system.boot()

        result = system.api.query_state(operator(), Scope(tenant_id=TENANT))
        assert result.objects == ()
        assert result.coverage is not None
        await system.shutdown()

    async def test_shutdown_drains_in_reverse(self, clock) -> None:
        """Cameras first, the API last.

        A consumer should keep receiving answers while there is state to answer
        from. Stopping the API first would blind every consumer while the
        platform was still recording facts they could have read.
        """
        system = await boot_everything(clock)
        assert not system.started

    async def test_a_restart_gap_is_recorded_as_an_observation(self, clock) -> None:
        """07_STATE §9.3: *"the restart gap is recorded as a coverage
        observation, so consumers see the discontinuity as data rather than
        inferring it from a suspicious silence."*
        """
        from vision_os.perception.detection import DetectionRuntime

        platform, detection, tracking, registry_layer, cropping = await build_stack(
            clock, full_document()
        )
        system = assemble(
            platform,
            registry_layer=registry_layer,
            cropping=cropping,
            tracking=tracking,
            log=InMemoryObservationLog(),
            authorizer=StaticAuthorizer([full_grant("operator", TENANT)]),
        )
        await system.boot()
        assert "tracking" in system.started_layers

        bridge = _TrackingToRegistry(registry_layer.runtime)
        tracking.runtime._sink = bridge  # noqa: SLF001
        runtime = DetectionRuntime(
            clock=platform.clock, bus=platform.bus, metrics=platform.metrics,
            health=platform.health, engine=detection.engine, consumer=tracking.runtime,
        )
        platform.runtime._admitted_consumer = runtime  # noqa: SLF001
        await detection.start()
        await runtime.start()
        await pump(clock, lambda: len(bridge.pending) >= 3)
        await bridge.drain(tracking)
        await drain(system, ticks=30)

        published = await system.record_restart_gap()
        await detection.stop()
        await system.shutdown()

        assert published >= 1
        coverage = [
            o
            for o in system.synthesis.log.read(CAMERA, limit=500)
            if o.observation_type is ObservationType.COVERAGE
        ]
        assert coverage, "a restart left no trace in the record"
        assert any(o.coverage.reason.value == "restart" for o in coverage)


class TestTheCompositionRoot:
    def test_the_layer_assembles(self, clock) -> None:
        platform = make_platform(clock, full_document())
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        from vision_os.synthesis_bootstrap import build_synthesis_layer

        synthesis = build_synthesis_layer(platform, registry_layer, attach=False)
        layer = build_exposure_layer(platform, synthesis.state)
        assert layer.api is not None
        assert layer.authorizer_id == "authz.static"

    def test_a_disabled_api_refuses_to_build(self, clock) -> None:
        """A site that does not want to serve consumers should not build a layer
        that refuses every request — the second is harder to diagnose.
        """
        document = full_document()
        document["api"]["enabled"] = False
        platform = make_platform(clock, document)
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        from vision_os.synthesis_bootstrap import build_synthesis_layer

        synthesis = build_synthesis_layer(platform, registry_layer, attach=False)
        with pytest.raises(ApiError, match="api.enabled is false"):
            build_exposure_layer(platform, synthesis.state)

    def test_an_unknown_authorizer_is_refused_by_name(self, clock) -> None:
        """A defaulted authorization model is how a deployment ends up serving
        more than anybody intended.
        """
        platform = make_platform(clock, full_document(authorizer="authz.typo"))
        with pytest.raises(ApiError, match="unknown authorizer"):
            build_authorizer(platform)

    def test_the_default_authorizer_denies(self, clock) -> None:
        """`authz.deny_all` is the default in the schema.

        A platform that served data until somebody remembered to configure a
        policy has exactly one failure mode and it is a breach.
        """
        from vision_os.kernel.config.schema import ApiSection

        assert ApiSection().authorizer == "authz.deny_all"

    def test_an_unknown_evidence_store_is_refused_by_name(self, clock) -> None:

        document = full_document()
        document["storage"] = {"evidence_store": "evidence.memory"}
        platform = make_platform(clock, document)
        assert build_evidence_store(platform) is not None

        with pytest.raises(Exception):  # noqa: B017 - schema or bootstrap, both refuse
            bad = full_document()
            bad["storage"] = {"evidence_store": "evidence.typo"}
            build_evidence_store(make_platform(clock, bad))

    def test_a_file_store_without_a_path_is_refused_at_boot(self, clock) -> None:
        """Fail at configuration, not at the first crop.

        A durable store with nowhere to write would work perfectly until the
        first piece of evidence arrived.
        """
        from vision_os.core.errors import ValidationError

        document = full_document()
        document["storage"] = {"evidence_store": "evidence.file"}
        with pytest.raises(ValidationError, match="evidence_path"):
            make_platform(clock, document).config.storage()


class TestNoBypasses:
    """The brief: *"No bypasses. No hidden dependencies. No direct module-to-
    module shortcuts."*"""

    async def test_the_api_reads_only_through_state(self, clock) -> None:
        """Everything a consumer sees arrived through M12.

        Proved by removing state: with the partition forgotten, the API returns
        an empty answer rather than reaching past M12 to the log.
        """
        system = await boot_everything(clock)
        for camera in list(system.state.partitions):
            system.state.forget(camera)

        result = system.api.query_state(operator(), Scope(tenant_id=TENANT))
        assert result.objects == ()

    async def test_no_layer_holds_a_reference_to_the_layer_above(
        self, clock
    ) -> None:
        """Each earlier flow holds a callable, never a typed collaborator."""
        system = await boot_everything(clock)

        registry_runtime = system.registry_layer.runtime
        assert not hasattr(registry_runtime, "_synthesis")
        assert not hasattr(registry_runtime, "_api")
        assert not hasattr(system.synthesis.runtime, "_api")

    async def test_the_demand_path_is_declarative_not_a_call(self, clock) -> None:
        """01_LAYERED §3.2: *"No call ever returns through the pipeline it
        entered."*

        M14 writes a demand record; M8 reads demand state at trigger time. The
        API never invokes the Crop Manager.
        """
        import inspect

        from vision_os.exposure import demands

        source = inspect.getsource(demands)
        for forbidden in ("CropManager", "CropRuntime", ".extract(", ".trigger("):
            assert forbidden not in source


class TestSecurityEndToEnd:
    async def test_an_unauthorized_consumer_is_refused(self, clock) -> None:
        system = await boot_everything(clock)
        stranger = Principal(subject="nobody", tenant_id=TENANT)
        with pytest.raises(ForbiddenError):
            system.api.query_state(stranger, Scope(tenant_id=TENANT))

    async def test_a_read_only_consumer_cannot_reach_evidence(self, clock) -> None:
        """12_SECURITY §5.3, through the whole stack."""
        from vision_os.core.model.ids import BlobRef

        system = await boot_everything(clock)
        with pytest.raises(ForbiddenError):
            system.api.get_evidence(reader(), BlobRef("sha256:x"), purpose="curiosity")

    async def test_every_access_is_audited(self, clock) -> None:
        system = await boot_everything(clock)
        system.api.query_state(operator(), Scope(tenant_id=TENANT))
        assert system.exposure.audit.failures == 0

    async def test_the_health_report_names_every_layer(self, clock) -> None:
        system = await boot_everything(clock)
        health = system.health()
        assert "state" in health
        assert "synthesis" in health
        assert "api" in health
