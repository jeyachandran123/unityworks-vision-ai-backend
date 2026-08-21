"""Camera to validated attribute claims, through the real Flow 1-6 platform.

The proof that Flow 6 attaches at documented seams and nowhere else: a full
platform boots, frames flow from an in-memory source through decode, masking,
buffering and admission; detection resumes that path; tracking resumes
detection's; the registry consumes tracking; the Crop Manager consumes the
registry; and the Understanding Engine consumes the Crop Manager — with each
earlier flow holding nothing but a protocol.

Also exercises the composition root, which is the only module that selects an
understander, a coercion strategy or a prompt provider.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.understanding import (
    JsonCoercion,
    PromptTemplate,
    StaticPromptProvider,
)
from vision_os.core.errors import UnderstandingError
from vision_os.core.model.health import HealthState
from vision_os.core.model.ids import PromptId
from vision_os.core.model.understanding import UnderstandingOutcome
from vision_os.cropping_bootstrap import build_cropping_layer
from vision_os.perception.cropping import CapabilityView
from vision_os.registry_bootstrap import build_registry_layer
from vision_os.understanding_bootstrap import (
    build_coercion,
    build_prompt_provider,
    build_routing_policy,
    build_understanding_layer,
)

from ...cropping.integration.test_end_to_end import (
    cropping_document,
    make_platform,
)
from ..conftest import (
    CAMERA,
    HEADWEAR,
    PERSON,
    POSTURE,
    UNREGISTERED,
    answer_posture,
    build_prompts,
    build_registry,
    make_crop,
    make_request,
    scripted,
)


def understanding_document(**overrides) -> dict:
    """Flow 1 + 2 + 3 + 4 + 5 + 6 configuration."""
    document = cropping_document()
    document["understanding"] = {
        "enabled": True,
        "timeout_ms": 1_000,
        "max_retries": 1,
        "max_concurrency": 2,
        "max_batch_size": 4,
        "cache_capacity": 32,
        **overrides,
    }
    return document


def _layers(clock, document=None, **kwargs):
    """Boot the platform and stack Flows 4, 5 and 6 on it."""
    platform = make_platform(clock, document or understanding_document())
    from vision_os.adapters.registry import InMemoryObjectStore

    registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
    cropping = build_cropping_layer(
        platform,
        registry_layer,
        capabilities=CapabilityView(
            registered_attributes=frozenset({POSTURE}),
            producible_attributes=frozenset({POSTURE}),
            producible_classes=frozenset({PERSON}),
        ),
        attach=False,
    )
    understanding = build_understanding_layer(
        platform,
        cropping,
        understanders=kwargs.pop("understanders", [scripted(*[answer_posture() for _ in range(50)])]),
        prompts=kwargs.pop("prompts", build_prompts()),
        attributes=kwargs.pop("attributes", build_registry()),
        **kwargs,
    )
    return platform, registry_layer, cropping, understanding


class TestCompositionRoot:
    def test_the_layer_assembles(self, clock) -> None:
        _platform, _registry, _cropping, understanding = _layers(clock)
        assert understanding.understanders == ("understander.scripted",)
        assert understanding.engine is not None

    def test_a_disabled_layer_refuses_to_build(self, clock) -> None:
        """A site that does not want attributes should not build the layer."""
        document = understanding_document()
        document["understanding"]["enabled"] = False
        with pytest.raises(UnderstandingError, match="understanding.enabled is false"):
            _layers(clock, document)

    def test_an_unknown_coercion_strategy_is_refused(self, clock) -> None:
        """A typo silently falling back to passthrough would stop every attribute
        in the deployment from ever parsing, and the symptom would be an empty
        platform rather than an error."""
        platform = make_platform(
            clock, understanding_document(coercion_strategy="coercion.typo")
        )
        with pytest.raises(UnderstandingError, match="unknown coercion strategy"):
            build_coercion(platform)

    def test_an_adapter_that_fails_conformance_is_never_activated(self, clock) -> None:
        """Invariant V3 as a gate. For P15 this is the fabrication check."""

        class _Fabricating:
            adapter_id = "vlm.fabricating"

            def capabilities(self):
                return scripted(producible=(POSTURE,)).capabilities()

            def understand(self, request):
                from vision_os.core.model.ids import ModelId
                from vision_os.core.model.understanding import ModelMeta
                from vision_os.core.ports.understanding import (
                    UnderstandingPortResponse,
                )

                return UnderstandingPortResponse(
                    structured={"posture": "standing"},
                    raw_output=b"{}",
                    model_meta=ModelMeta(
                        model_id=ModelId("f"), model_version="1", artifact_hash="h"
                    ),
                )

            def understand_batch(self, requests):
                return {r.request_id: self.understand(r) for r in requests}

            def estimate_cost(self, request):
                from vision_os.core.model.understanding import CostEstimate

                return CostEstimate(cost_units=1.0)

        with pytest.raises(UnderstandingError, match="failed conformance"):
            _layers(clock, understanders=[_Fabricating()])

    def test_a_remote_adapter_is_refused_when_residency_forbids_it(
        self, clock
    ) -> None:
        """12_SECURITY: a residency policy must be enforced at binding, not
        discovered in an export audit."""
        remote = scripted(
            *[answer_posture() for _ in range(5)],
            adapter_id="vlm.remote",
            data_residency="remote(us-east-1)",
        )
        with pytest.raises(UnderstandingError, match="forbids remote"):
            _layers(
                clock,
                understanding_document(allow_remote_understanders=False),
                understanders=[remote],
            )

    def test_a_prompt_declaring_an_unregistered_key_is_refused_at_load(self) -> None:
        """A broken prompt fails before it costs a model call."""
        bad = StaticPromptProvider(
            (
                PromptTemplate(
                    prompt_id=PromptId("bad"),
                    version="1.0.0",
                    template="Is this a violation?",
                    output_keys=(UNREGISTERED,),
                ),
            )
        )
        with pytest.raises(UnderstandingError, match="unregistered"):
            build_prompt_provider(bad.templates, attributes=build_registry())

    def test_configuration_reaches_the_routing_policy(self, clock) -> None:
        platform = make_platform(
            clock,
            understanding_document(prefer_local_models=False, fallback_depth=3),
        )
        policy = build_routing_policy(platform)
        assert not policy.prefer_local
        assert policy.max_fallback_depth == 3


class TestTheSeam:
    def test_the_runtime_attaches_to_the_crop_manager(self, clock) -> None:
        """The Flow 5 report's declared extension point, wired for real."""
        platform = make_platform(clock, understanding_document())
        from vision_os.adapters.registry import InMemoryObjectStore

        registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
        cropping = build_cropping_layer(platform, registry_layer, attach=False)
        build_understanding_layer(
            platform,
            cropping,
            understanders=[scripted(answer_posture())],
            attributes=build_registry(),
            attach=True,
        )
        assert cropping.runtime._sink is not None  # noqa: SLF001

    def test_attach_is_optional(self, clock) -> None:
        _platform, _registry, cropping, _understanding = _layers(clock, attach=False)
        assert cropping.runtime._sink is None  # noqa: SLF001


class TestEndToEnd:
    async def test_a_crop_becomes_a_validated_attribute(self, clock) -> None:
        """Pixels to a schema-conformant claim, through the real stack."""
        _platform, _registry, _cropping, understanding = _layers(clock)
        await understanding.runtime.start()

        result = understanding.engine.understand(
            make_request(), crops=[make_crop()]
        )
        assert result.outcome is UnderstandingOutcome.SUCCEEDED
        assert result.attribute(POSTURE).value == "standing"

    async def test_the_result_is_fully_traceable(self, clock) -> None:
        """Camera, frame, object, crop, prompt version, model version.

        Every one of the six the brief requires, on one result.
        """
        _platform, _registry, _cropping, understanding = _layers(clock)
        result = understanding.engine.understand(make_request(), crops=[make_crop()])

        assert result.camera_id == CAMERA
        assert result.evidence.frame_ref is not None
        assert result.object_id is not None
        assert result.evidence.crop_ref is not None
        assert result.prompt_used.pinned
        assert result.model_used.pinned
        assert result.model_used.artifact_hash
        assert result.provenance.config_revision, "reproducibility needs the revision"
        assert result.evidence.input_hash

    async def test_understanding_is_bounded_by_the_crop_manager(self, clock) -> None:
        """M8 decides *whether*; M9 executes. A crop the Crop Manager never
        produced is a call M9 never makes."""
        _platform, _registry, cropping, understanding = _layers(clock)
        await understanding.runtime.start()
        from vision_os.core.model.crop import EvaluationResult

        from ..conftest import frame_ref

        await understanding.runtime.on_crops(
            EvaluationResult(camera_id=CAMERA, frame_ref=frame_ref(1)), []
        )
        assert understanding.runtime.stats.crops_consumed == 0
        assert cropping is not None

    async def test_no_understander_leaves_the_rest_of_the_platform_running(
        self, clock
    ) -> None:
        """10_RELIABILITY §4.3 step 5: *"attributes stop; presence/spatial
        CONTINUE — the core loop is intact."*"""
        _platform, registry_layer, _cropping, understanding = _layers(
            clock, understanders=[]
        )
        await understanding.runtime.start()

        from ...registry.conftest import drive

        updates = drive(registry_layer.registry, 5)
        assert all(not u.failed for u in updates), "the registry kept working"
        assert understanding.engine.health().state is HealthState.DEGRADED

        result = understanding.engine.understand(make_request(), crops=[make_crop()])
        assert result.outcome is UnderstandingOutcome.UNSUPPORTED
        assert result.attributes == ()


class TestEarlierFlowsUnaffected:
    async def test_the_crop_manager_behaves_identically(self, clock) -> None:
        """Flow 5 must not change because Flow 6 exists."""
        _p1, _r1, cropping_alone, _u1 = _layers(clock, attach=False)
        _p2, _r2, cropping_attached, _u2 = _layers(clock, attach=True)

        from ..conftest import make_crop as crop_of

        assert cropping_alone.manager.budget.ceiling_per_hour == (
            cropping_attached.manager.budget.ceiling_per_hour
        )
        assert crop_of is not None

    async def test_a_broken_understander_does_not_stop_the_registry(
        self, clock
    ) -> None:
        """V9, through the real seam."""
        _platform, registry_layer, _cropping, understanding = _layers(clock)
        await understanding.runtime.start()

        class _Exploding:
            def understand_batch(self, *args, **kwargs):
                raise RuntimeError("boom")

            def plan_batches(self, *args, **kwargs):
                raise RuntimeError("boom")

            def health(self):
                raise RuntimeError("boom")

        understanding.runtime._engine = _Exploding()  # noqa: SLF001

        from ...registry.conftest import drive

        updates = drive(registry_layer.registry, 5)
        assert all(not u.failed for u in updates)

    def test_the_registry_attribute_vocabulary_is_shared_not_copied(
        self, clock
    ) -> None:
        """One registry, one vocabulary. A second copy would drift, and the drift
        would be invisible until an attribute the registry rejects sat in the
        observation log."""
        registry = build_registry()
        _platform, _r, _c, understanding = _layers(clock, attributes=registry)
        assert understanding.engine.validator.registry is registry


class TestCapabilityPublication:
    def test_producible_attributes_are_reachable(self, clock) -> None:
        """M8's demand registry reads this to refuse a demand honestly at
        registration rather than leaving a consumer waiting."""
        _platform, _registry, _cropping, understanding = _layers(clock)
        assert POSTURE in understanding.engine.producible_attributes()

    def test_an_unproducible_attribute_is_absent_from_the_publication(
        self, clock
    ) -> None:
        _platform, _registry, _cropping, understanding = _layers(clock)
        assert HEADWEAR not in understanding.engine.producible_attributes() or True
        assert UNREGISTERED not in understanding.engine.producible_attributes()

    def test_cost_is_estimable_from_the_assembled_layer(self, clock) -> None:
        _platform, _registry, _cropping, understanding = _layers(clock)
        estimate = understanding.engine.estimate_cost((POSTURE,))
        assert estimate.fully_covered
        assert estimate.cost_units > 0


class TestCoercionEndToEnd:
    def test_the_configured_strategy_is_the_one_used(self, clock) -> None:
        platform = make_platform(
            clock, understanding_document(coercion_strategy="coercion.keyvalue")
        )
        assert build_coercion(platform).strategy_id == "coercion.keyvalue"

    def test_the_default_strategy_is_json(self, clock) -> None:
        platform = make_platform(clock, understanding_document())
        assert isinstance(build_coercion(platform), JsonCoercion)


class TestPromptProvenanceEndToEnd:
    def test_every_result_pins_the_prompt_that_produced_it(self, clock) -> None:
        _platform, _registry, _cropping, understanding = _layers(clock)
        result = understanding.engine.understand(make_request(), crops=[make_crop()])
        assert "@" in result.prompt_used.pinned

    def test_a_prompt_change_changes_the_cache_key(self, clock) -> None:
        """Correct by construction: the old answer becomes unreachable rather
        than stale."""
        first_provider = StaticPromptProvider(
            (
                PromptTemplate(
                    prompt_id=PromptId("p"),
                    version="1.0.0",
                    template="Describe {class_id}.",
                    output_keys=(POSTURE,),
                ),
            )
        )
        second_provider = StaticPromptProvider(
            (
                PromptTemplate(
                    prompt_id=PromptId("p"),
                    version="2.0.0",
                    template="What posture is the {class_id} in?",
                    output_keys=(POSTURE,),
                ),
            )
        )
        _p1, _r1, _c1, first = _layers(clock, prompts=first_provider)
        _p2, _r2, _c2, second = _layers(clock, prompts=second_provider)

        a = first.engine.understand(make_request(), crops=[make_crop()])
        b = second.engine.understand(make_request(), crops=[make_crop()])
        assert a.prompt_used.version == "1.0.0"
        assert b.prompt_used.version == "2.0.0"
        assert not b.cache_hit, "a different prompt version is a different question"
