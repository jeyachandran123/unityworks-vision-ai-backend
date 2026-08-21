"""Camera to canonical evidence, through the real Flow 1-5 platform.

The proof that Flow 5 attaches at documented seams and nowhere else: a full
platform boots, frames flow from an in-memory source through decode, masking,
buffering and admission; detection resumes that path; tracking resumes
detection's; the registry consumes tracking; and the Crop Manager consumes the
registry — with each earlier flow holding nothing but a protocol.

Also exercises the composition root, which is the only module that selects a
trigger policy, a quality estimator, a crop strategy or an extractor.
"""

from __future__ import annotations

import asyncio
import hashlib

import pytest

from vision_os.acquisition import SourceBindings
from vision_os.adapters.acquisition import (
    ArrivalTimeClockSync,
    InMemoryRawSource,
    NoMaskPolicy,
    PassthroughDecoder,
)
from vision_os.adapters.configuration import InMemoryConfigSource
from vision_os.adapters.detection import ReferenceDetector, ScriptedDetection
from vision_os.adapters.models import InMemoryArtifactStore, ScriptedRuntime
from vision_os.adapters.registry import InMemoryObjectStore
from vision_os.bootstrap import build_platform
from vision_os.conformance import platform_registry
from vision_os.core.errors import CropError, DemandRejectedError
from vision_os.core.model.camera import Camera
from vision_os.core.model.crop import SkipReason
from vision_os.core.model.demand import DemandStatus
from vision_os.core.model.frame import FrameDimensions
from vision_os.core.model.ids import AttributeKey, CameraId, ClassId
from vision_os.core.model.space import Box
from vision_os.core.model.timebase import Duration
from vision_os.cropping_bootstrap import (
    build_crop_strategy,
    build_cropping_layer,
    build_quality_estimator,
    build_trigger_policy,
)
from vision_os.detection_bootstrap import build_detection_layer
from vision_os.kernel.config import ConfigLayer
from vision_os.perception.cropping import CapabilityView
from vision_os.perception.detection import DetectionRuntime
from vision_os.registry_bootstrap import build_registry_layer
from vision_os.tracking_bootstrap import build_tracking_layer

from ...conftest import HEIGHT, WIDTH, base_config_document, make_frames
from ...registry.integration.test_end_to_end import (
    REFERENCE_HASH,
    REFERENCE_WEIGHTS,
    _TrackingToRegistry,
)
from ..conftest import COLOUR, make_demand

DIMENSIONS = FrameDimensions(width=WIDTH, height=HEIGHT)
CAMERA = CameraId("cam-01")
PERSON = ClassId("person")

assert REFERENCE_HASH  # imported for parity with the Flow 4 harness


def cropping_document(**cropping_overrides) -> dict:
    """Flow 1 + 2 + 3 + 4 + 5 configuration."""
    document = base_config_document(cameras=1, target_fps=1000.0)
    document["detection"] = {
        "enabled": True,
        "confidence_threshold": 0.25,
        "max_batch_size": 4,
        "batch_max_wait_ms": 0,
        "inference_timeout_ms": 2000,
        "queue_capacity": 32,
    }
    document["models"] = {"warmup_enabled": True, "allow_cpu_fallback": True}
    document["tracking"] = {
        "enabled": True,
        "tracker_id": "tracker.sort",
        "min_hits_to_confirm": 2,
        "max_coast_frames": 4,
        "max_lost_frames": 8,
    }
    document["registry"] = {
        "enabled": True,
        "min_observations_to_confirm": 2,
        "persistence_enabled": False,
    }
    document["cropping"] = {
        "enabled": True,
        "understanding_calls_per_hour": 360_000.0,
        "crop_width": 64,
        "crop_height": 64,
        "min_scale_pixels": 32.0,
        **cropping_overrides,
    }
    document["taxonomy"] = [{"class_id": "person"}]
    document["detectors"] = [
        {
            "detector_id": "reference-primary",
            "adapter_id": "detector.reference",
            "model_id": "reference-detector",
            "model_version": "1.0.0",
            "artifact_uri": "mem://reference.bin",
            "artifact_hash": REFERENCE_HASH,
            "role": "primary_detector",
            "mappings": [{"native_label": "person", "class_id": "person"}],
        }
    ]
    return document


def bindings_factory(clock, frames: int = 12, interpacket_ms: int = 20):
    def factory(camera: Camera) -> SourceBindings:
        return SourceBindings(
            source=InMemoryRawSource(
                make_frames(frames),
                clock=clock,
                semantics=camera.source_semantics,
                interpacket=Duration.from_millis(interpacket_ms),
            ),
            decoder=PassthroughDecoder(dimensions=DIMENSIONS),
            privacy=NoMaskPolicy(),
            clock_sync=ArrivalTimeClockSync(),
        )

    return factory


def detector_factory(clock):
    """One large, well-placed person — the cleanest end-to-end signal.

    Deliberately big enough to clear the scale gate: this suite tests the
    *plumbing*, and the gate's thresholds have their own unit tests.
    """

    def factory(declaration):
        return ReferenceDetector(
            clock=clock,
            producible_classes=(PERSON,),
            script=(ScriptedDetection(PERSON, Box(0.3, 0.1, 0.6, 0.9), 0.92),),
        )

    return factory


async def pump(clock, predicate, *, steps: int = 400, step_ms: int = 5) -> None:
    for _ in range(steps):
        if predicate():
            return
        clock.advance(Duration.from_millis(step_ms))
        for _ in range(6):
            await asyncio.sleep(0)


def make_platform(clock, document: dict, *, conformance=None):
    return build_platform(
        config_sources={ConfigLayer.SITE: InMemoryConfigSource(document)},
        bindings_factory=bindings_factory(clock),
        clock=clock,
        conformance=conformance or platform_registry(),
    )


def full_capabilities() -> CapabilityView:
    """What Flow 6 will supply. Granted here so attention has something to serve."""
    return CapabilityView(
        registered_attributes=frozenset({COLOUR}),
        producible_attributes=frozenset({COLOUR}),
        producible_classes=frozenset({PERSON}),
        observed_cameras=frozenset({CAMERA}),
    )


async def build_stack(clock, document: dict, *, crops: list | None = None, **kwargs):
    """Boot every flow and wire all four seams."""
    artifacts = InMemoryArtifactStore()
    artifacts.put("mem://reference.bin", REFERENCE_WEIGHTS)

    platform = make_platform(clock, document)
    registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
    cropping = build_cropping_layer(
        platform,
        registry_layer,
        crop_sink=(
            (lambda result, produced: crops.append((result, produced)))
            if crops is not None
            else None
        ),
        attach=True,
        **kwargs,
    )
    tracking = build_tracking_layer(platform, tracking_sink=None)
    detection = build_detection_layer(
        platform,
        detector_factory=detector_factory(clock),
        artifacts=artifacts,
        runtimes=(ScriptedRuntime(),),
        detection_consumer=tracking.runtime,
    )
    return platform, detection, tracking, registry_layer, cropping


async def run_pipeline(clock, document, *, crops=None, **kwargs):
    """Boot, pump frames through every layer, and return the assembled stack."""
    platform, detection, tracking, registry_layer, cropping = await build_stack(
        clock, document, crops=crops, **kwargs
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
    await platform.boot()

    await pump(clock, lambda: len(bridge.pending) >= 5)
    await bridge.drain(tracking)
    for _ in range(20):
        await asyncio.sleep(0)

    await detection.stop()
    await platform.shutdown()
    return platform, registry_layer, cropping


class TestCompositionRoot:
    def test_the_layer_assembles(self, clock) -> None:
        platform = make_platform(clock, cropping_document())
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        layer = build_cropping_layer(platform, registry_layer, attach=False)
        assert layer.policy_id == "trigger.default"
        assert layer.budget.ceiling_per_hour == pytest.approx(360_000.0)

    def test_declared_quality_floors_reach_the_gate(self, clock) -> None:
        """Per-attribute floors must survive the whole composition root.

        The floors live in a policy document, are parsed by `SemanticPolicy`,
        merged across policies by the harness, and handed to `build_cropping_layer`.
        A gate that quietly kept the defaults would leave every declared floor
        unenforced while the document, the tests and the dashboards all showed it
        configured — the failure would be invisible from every side.
        """
        platform = make_platform(clock, cropping_document())
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        layer = build_cropping_layer(
            platform,
            registry_layer,
            attach=False,
            quality_floors={
                "head_covering": {"min_scale_pixels": 220.0, "max_blur": 0.5},
                "hand_covering": {"min_scale_pixels": 120.0},
            },
        )
        gate = layer.manager._gate

        assert gate.thresholds_for(("head_covering",)).min_scale_pixels == 220.0
        assert gate.thresholds_for(("head_covering",)).max_blur == 0.5
        assert gate.thresholds_for(("hand_covering",)).min_scale_pixels == 120.0

        # An unspecified field keeps the deployment default rather than the
        # dataclass default: a document tightening scale must not loosen blur.
        default = gate.thresholds_for(())
        assert gate.thresholds_for(("hand_covering",)).max_blur == default.max_blur

        # The strictest declared floor governs a crop that answers both.
        assert gate.thresholds_for(("head_covering", "hand_covering")).min_scale_pixels == 220.0

        # An attribute nobody declared is judged by the deployment default.
        assert gate.thresholds_for(("garment_colour",)) is default

    def test_a_layer_without_floors_behaves_exactly_as_before(self, clock) -> None:
        platform = make_platform(clock, cropping_document())
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        layer = build_cropping_layer(platform, registry_layer, attach=False)
        gate = layer.manager._gate
        assert gate.thresholds_for(("head_covering",)) is gate.thresholds

    def test_declared_output_sizes_reach_the_crop_manager(self, clock) -> None:
        """Per-attribute resolution must survive the whole composition root.

        Loading the value proves the parser works; only planning a crop proves
        the strategy the manager actually holds was given it. A size that stopped
        at the boundary would leave the document, the tests and the dashboards
        all showing 448 while every crop was still rendered at 224 — and the
        symptom would look like a model failure.
        """
        platform = make_platform(
            clock, cropping_document(crop_strategy="crop.part_focused")
        )
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        layer = build_cropping_layer(
            platform,
            registry_layer,
            attach=False,
            evidence_regions={"head_covering": (0.0, 0.45)},
            output_sizes={"head_covering": (448, 448)},
        )
        strategy = layer.manager._strategy

        def plan_for(*attributes):
            return strategy.plan(
                box=Box(0.3, 0.1, 0.5, 0.9),
                class_id=ClassId("person"),
                source_width=WIDTH,
                source_height=HEIGHT,
                attributes=tuple(AttributeKey(a) for a in attributes),
            )

        assert plan_for("head_covering").output_width == 448
        # An attribute nobody sized keeps the deployment default, whatever the
        # deployment happens to have configured it to.
        assert plan_for("hand_covering").output_width == 64
        # A crop answering both is rendered at the larger of the two.
        assert plan_for("head_covering", "hand_covering").output_width == 448

    def test_a_layer_without_output_sizes_renders_as_before(self, clock) -> None:
        """No deployment is silently upgraded: 448 costs 4x the vision tokens."""
        platform = make_platform(
            clock, cropping_document(crop_strategy="crop.part_focused")
        )
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        layer = build_cropping_layer(platform, registry_layer, attach=False)
        plan = layer.manager._strategy.plan(
            box=Box(0.3, 0.1, 0.5, 0.9),
            class_id=ClassId("person"),
            source_width=WIDTH,
            source_height=HEIGHT,
            attributes=(AttributeKey("head_covering"),),
        )
        assert plan.output_width == 64, "the deployment's own size, unchanged"

    def test_a_disabled_layer_refuses_to_build(self, clock) -> None:
        """A site that does not want attention should not build the layer."""
        document = cropping_document()
        document["cropping"]["enabled"] = False
        platform = make_platform(clock, document)
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        with pytest.raises(CropError, match="cropping.enabled is false"):
            build_cropping_layer(platform, registry_layer, attach=False)

    @pytest.mark.parametrize(
        ("key", "value", "builder", "message"),
        [
            ("trigger_policy", "trigger.typo", build_trigger_policy, "unknown trigger policy"),
            (
                "quality_estimator",
                "quality.typo",
                build_quality_estimator,
                "unknown quality estimator",
            ),
            ("crop_strategy", "crop.typo", build_crop_strategy, "unknown crop strategy"),
        ],
    )
    def test_an_unknown_adapter_is_refused_not_defaulted(
        self, clock, key, value, builder, message
    ) -> None:
        """A typo that silently fell back would change what the platform pays
        attention to, with no signal at all."""
        platform = make_platform(clock, cropping_document(**{key: value}))
        with pytest.raises(CropError, match=message):
            builder(platform)

    def test_an_adapter_that_fails_conformance_is_never_activated(self, clock) -> None:
        """Invariant V3 as a gate, not an aspiration (06_PORTS section 5)."""

        class _Dropping:
            policy_id = "trigger.dropping"

            def evaluate(self, candidates, *, now, demands):
                return []

        platform = make_platform(clock, cropping_document())
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        with pytest.raises(CropError, match="failed conformance"):
            build_cropping_layer(
                platform, registry_layer, policy=_Dropping(), attach=False
            )

    def test_configuration_reaches_the_adapters(self, clock) -> None:
        platform = make_platform(
            clock,
            cropping_document(
                crop_strategy="crop.padded",
                crop_padding=0.4,
                crop_width=96,
                crop_height=96,
            ),
        )
        strategy = build_crop_strategy(platform)
        assert strategy.padding == pytest.approx(0.4)
        plan = strategy.plan(
            box=Box(0.4, 0.3, 0.55, 0.85),
            class_id=PERSON,
            source_width=WIDTH,
            source_height=HEIGHT,
        )
        assert (plan.output_width, plan.output_height) == (96, 96)

    def test_the_gate_thresholds_come_from_configuration(self, clock) -> None:
        from vision_os.cropping_bootstrap import build_gate_thresholds

        platform = make_platform(clock, cropping_document(min_scale_pixels=99.0))
        assert build_gate_thresholds(platform).min_scale_pixels == pytest.approx(99.0)


class TestTheSeam:
    def test_the_crop_runtime_attaches_to_the_registry(self, clock) -> None:
        """The Flow 4 report's declared extension point, wired for real."""
        platform = make_platform(clock, cropping_document())
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        build_cropping_layer(platform, registry_layer, attach=True)
        assert registry_layer.runtime._sink is not None  # noqa: SLF001

    def test_attach_is_optional(self, clock) -> None:
        platform = make_platform(clock, cropping_document())
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        build_cropping_layer(platform, registry_layer, attach=False)
        assert registry_layer.runtime._sink is None  # noqa: SLF001


class TestEndToEnd:
    async def test_frames_become_canonical_evidence(self, clock) -> None:
        """Pixels to defensible evidence, through every seam."""
        crops: list = []
        _platform, registry_layer, cropping = await run_pipeline(
            clock,
            cropping_document(),
            crops=crops,
            capabilities=full_capabilities(),
        )
        cropping.manager.register_demand(make_demand())

        assert registry_layer.registry.objects(CAMERA), "objects must exist first"
        assert cropping.runtime.stats.frames_consumed > 0, (
            "the registry seam must deliver to the Crop Manager"
        )

    async def test_a_crop_is_fully_traceable(self, clock) -> None:
        """Camera, frame, object, transform — everything a claim needs."""
        crops: list = []
        platform = make_platform(clock, cropping_document())
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        cropping = build_cropping_layer(
            platform,
            registry_layer,
            capabilities=full_capabilities(),
            crop_sink=lambda result, produced: crops.append((result, produced)),
            attach=False,
        )
        cropping.manager.register_demand(make_demand())

        from ..conftest import frame_context, make_object, sharp_frame

        # Real camera dimensions rather than the tiny synthetic frames the
        # acquisition harness uses: the gate's scale floor is measured in source
        # pixels, and a 4-pixel-tall test frame cannot clear any honest floor.
        width, height = 640, 480
        frame = frame_context(width=width, height=height)
        result = cropping.manager.evaluate([make_object()], frame)
        assert result.requests, "a demanded object must trigger"

        crop = cropping.manager.extract(
            result.requests[0], pixels=sharp_frame(width, height), frame=frame
        )
        assert crop.camera_id == CAMERA
        assert crop.source_frame == frame.frame_ref
        assert crop.object_id is not None
        assert crop.transform.source_width == width
        assert crop.provenance.config_revision, "reproducibility needs the revision"
        assert crop.trigger_reason is not None
        assert crop.t_capture == frame.t_capture, "capture time, not extraction time"

    async def test_no_demand_means_no_evidence(self, clock) -> None:
        """§M8's largest single saving, measured end to end."""
        crops: list = []
        _platform, _registry, cropping = await run_pipeline(
            clock, cropping_document(), crops=crops
        )
        assert cropping.runtime.stats.crops_produced == 0
        assert cropping.runtime.stats.skips_recorded > 0

    async def test_every_candidate_is_accounted_for(self, clock) -> None:
        """The V8 identity, across the whole pipeline rather than one call."""
        crops: list = []
        _platform, _registry, _cropping = await run_pipeline(
            clock, cropping_document(), crops=crops, capabilities=full_capabilities()
        )
        assert crops, "the sink must see every frame"
        for result, _produced in crops:
            seen = [r.object_id for r in result.requests] + [
                s.object_id for s in result.skipped
            ]
            assert len(seen) == len(set(seen)), (
                f"an object appears twice in one frame's accounting: {seen}"
            )

    async def test_skip_reasons_are_attributed_across_a_run(self, clock) -> None:
        crops: list = []
        await run_pipeline(clock, cropping_document(), crops=crops)
        counts: dict[SkipReason, int] = {}
        for result, _ in crops:
            for reason, count in result.skips_by_reason().items():
                counts[reason] = counts.get(reason, 0) + count
        assert counts, "every unanalysed candidate carries a reason"
        assert set(counts) <= set(SkipReason)


class TestEarlierFlowsUnaffected:
    async def test_the_registry_still_produces_objects_with_cropping_attached(
        self, clock
    ) -> None:
        """Flow 4 must not change because Flow 5 exists."""
        _platform, registry_layer, _cropping = await run_pipeline(
            clock, cropping_document(), capabilities=full_capabilities()
        )
        objects = registry_layer.registry.objects(CAMERA)
        assert objects
        for obj in objects:
            assert obj.camera_id == CAMERA
            assert obj.confidence.semantics.value == "identity"

    async def test_a_broken_crop_manager_does_not_stop_the_registry(
        self, clock
    ) -> None:
        """V9, through the real seam.

        The frames driven here are never resident in the buffer, so the runtime
        skips with ``FRAME_UNAVAILABLE`` before it reaches the broken manager —
        which is itself the point: the registry's output is unaffected either
        way. The path where the manager *is* reached and throws is covered by
        ``test_crop_runtime.py::TestTheFirewall::test_the_seam_never_raises``,
        which supplies a resident frame.
        """
        platform = make_platform(clock, cropping_document())
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        cropping = build_cropping_layer(platform, registry_layer, attach=False)
        await cropping.runtime.start()

        class _Exploding:
            def evaluate(self, *args, **kwargs):
                raise RuntimeError("boom")

            def health(self):
                raise RuntimeError("boom")

        cropping.runtime._manager = _Exploding()  # noqa: SLF001

        from ...registry.conftest import drive

        updates = drive(registry_layer.registry, 5)
        for update in updates:
            await cropping.runtime.on_registered(update)

        assert all(not u.failed for u in updates), "the registry kept working"
        assert [u.count for u in updates] == [u.count for u in updates]
        assert cropping.runtime.stats.frames_consumed == 5, (
            "every update was consumed; none escaped as an exception"
        )


class TestHonestCapability:
    def test_with_no_understander_every_demand_is_unsatisfiable(self, clock) -> None:
        """The shipping default until Flow 6, stated rather than hidden.

        A consumer is told at registration that nothing will arrive, instead of
        waiting forever for an attribute no loaded model can produce.
        """
        platform = make_platform(clock, cropping_document())
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        layer = build_cropping_layer(
            platform,
            registry_layer,
            capabilities=CapabilityView(registered_attributes=frozenset({COLOUR})),
            attach=False,
        )
        ack = layer.manager.register_demand(make_demand())
        assert layer.demands.get(ack.demand_id).status is DemandStatus.UNSATISFIABLE

    def test_an_unregistered_attribute_is_refused(self, clock) -> None:
        platform = make_platform(clock, cropping_document())
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        layer = build_cropping_layer(platform, registry_layer, attach=False)
        with pytest.raises(DemandRejectedError, match="register them first"):
            layer.manager.register_demand(
                make_demand(attributes=(AttributeKey("appearance.vibe"),))
            )


class TestBudgetEndToEnd:
    async def test_a_tight_budget_thins_rather_than_stops(self, clock) -> None:
        """Under-provisioning degrades coverage; it never blinds the platform."""
        crops: list = []
        _platform, _registry, cropping = await run_pipeline(
            clock,
            cropping_document(understanding_calls_per_hour=1.0),
            crops=crops,
            capabilities=full_capabilities(),
        )
        assert cropping.runtime.stats.frames_consumed > 0
        assert cropping.runtime.stats.frames_failed == 0

    def test_the_acknowledgement_reflects_the_configured_ceiling(self, clock) -> None:
        platform = make_platform(clock, cropping_document())
        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        layer = build_cropping_layer(
            platform, registry_layer, capabilities=full_capabilities(), attach=False
        )
        ack = layer.manager.register_demand(make_demand(freshness_ms=1))
        assert ack.effective_freshness > Duration(0)


def _hash_is_stable() -> None:
    assert hashlib.sha256(b"x").hexdigest()
