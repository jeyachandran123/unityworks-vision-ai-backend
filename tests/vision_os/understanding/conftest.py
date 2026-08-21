"""Fixtures for the Flow 6 understanding suite.

Built from real modules and reference adapters — no mocks at a module boundary.
The understander is **scripted** rather than real, and that is the point: the
property under test is *"does the platform handle what a model said"*, and a real
model would make every assertion non-deterministic for no benefit.

The attribute registry here declares four attributes spanning the value types
that matter — an enum with a domain, a bool, a scalar with a range, and a
multi-valued relation — because the schema gate's job is to refuse values, and it
cannot be tested against a vocabulary of one shape.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.understanding import (
    JsonCoercion,
    PromptTemplate,
    ScriptedAnswer,
    ScriptedUnderstander,
    StaticAttributeHead,
    StaticPromptProvider,
    UnavailableUnderstander,
)
from vision_os.core.model.confidence import Confidence, ConfidenceSemantics
from vision_os.core.model.crop import (
    Crop,
    CropTransform,
    GateResult,
    PrivacyClass,
    RetentionMode,
    TriggerReason,
)
from vision_os.core.model.detection import QualityGrades, QualityLevel
from vision_os.core.model.ids import (
    AttributeKey,
    CameraId,
    ClassId,
    ConfigRevision,
    CropId,
    FrameRef,
    FrameSeq,
    ModuleId,
    ObjectId,
    PromptId,
    RequestId,
    SiteId,
    StreamEpoch,
    TenantId,
)
from vision_os.core.model.provenance import Provenance
from vision_os.core.model.space import Box
from vision_os.core.model.timebase import Duration, Instant
from vision_os.core.model.understanding import UnderstandingRequest
from vision_os.core.ports.understanding import OutputSchema, RenderedPrompt
from vision_os.kernel.config.schema import UnderstandingSection
from vision_os.perception.registry.attributes import (
    AttributeRegistry,
    AttributeSchema,
    AttributeValueType,
    Cardinality,
)
from vision_os.perception.understanding import (
    BoundUnderstander,
    CapabilityRouter,
    ResponseCache,
    RoutingPolicy,
    UnderstandingEngine,
    UnderstandingRuntime,
)

CAMERA = CameraId("cam-01")
OTHER_CAMERA = CameraId("cam-02")
TENANT = TenantId("acme")
OTHER_TENANT = TenantId("globex")
SITE = SiteId("site-sg-01")
PERSON = ClassId("person")
VEHICLE = ClassId("vehicle")

POSTURE = AttributeKey("posture")
HEADWEAR = AttributeKey("headwear_present")
HEIGHT = AttributeKey("apparent_height_ratio")
CARRYING = AttributeKey("carrying")
UNREGISTERED = AttributeKey("is_violation")

FRAME_INTERVAL_MS = 200


def at(seq: int) -> Instant:
    return Instant(seq * FRAME_INTERVAL_MS * 1_000_000)


def frame_ref(seq: int = 0, *, camera: CameraId = CAMERA) -> FrameRef:
    return FrameRef(camera, StreamEpoch(1), FrameSeq(seq))


# --- the attribute vocabulary ---------------------------------------------------- #


def build_registry() -> AttributeRegistry:
    """Four attributes spanning the value types the gate must police.

    Every justification names visible evidence, because the registry refuses
    anything else — these had to pass the same neutrality gate a real deployment's
    would.
    """
    registry = AttributeRegistry()
    registry.register(
        AttributeSchema(
            key=POSTURE,
            value_type=AttributeValueType.ENUM,
            domain=("standing", "sitting", "lying", "crouching"),
            neutrality_justification="Body configuration is directly visible",
            applies_to=(PERSON,),
            validity=Duration.from_millis(60_000),
        )
    )
    registry.register(
        AttributeSchema(
            key=HEADWEAR,
            value_type=AttributeValueType.BOOL,
            neutrality_justification="Head region shows a covering",
            applies_to=(PERSON,),
        )
    )
    registry.register(
        AttributeSchema(
            key=HEIGHT,
            value_type=AttributeValueType.SCALAR,
            domain=("0:1",),
            neutrality_justification="Ratio of visible object height to frame height",
        )
    )
    registry.register(
        AttributeSchema(
            key=CARRYING,
            value_type=AttributeValueType.RELATION,
            domain=("bag", "box", "tool"),
            cardinality=Cardinality.MULTI,
            neutrality_justification="An object is visibly supported by the person",
            applies_to=(PERSON,),
        )
    )
    return registry


@pytest.fixture
def attribute_registry() -> AttributeRegistry:
    return build_registry()


# --- prompts ---------------------------------------------------------------------- #


def build_prompts() -> StaticPromptProvider:
    return StaticPromptProvider(
        (
            PromptTemplate(
                prompt_id=PromptId("person.appearance"),
                version="1.0.0",
                template="Describe the {class_id}: {requested_attributes}",
                output_keys=(POSTURE, HEADWEAR, CARRYING),
                applies_to=(PERSON,),
            ),
            PromptTemplate(
                prompt_id=PromptId("person.posture"),
                version="1.0.0",
                template="What posture is the {class_id} in?",
                output_keys=(POSTURE,),
                applies_to=(PERSON,),
            ),
            PromptTemplate(
                prompt_id=PromptId("generic.geometry"),
                version="2.1.0",
                template="Report the apparent height ratio for {class_id}.",
                output_keys=(HEIGHT,),
            ),
        )
    )


@pytest.fixture
def prompts() -> StaticPromptProvider:
    return build_prompts()


def rendered(
    *, fields: tuple[AttributeKey, ...] = (POSTURE,), version: str = "1.0.0"
) -> RenderedPrompt:
    return RenderedPrompt(
        prompt_id=PromptId("test.prompt"),
        version=version,
        text="Describe the subject.",
        output_schema=OutputSchema(fields=fields),
        content_hash="sha256:test",
    )


# --- crops ------------------------------------------------------------------------ #


def make_crop(
    *,
    crop_id: str = "crop-1",
    object_id: str = "obj-1",
    seq: int = 3,
    camera: CameraId = CAMERA,
    tenant: TenantId = TENANT,
    trigger: TriggerReason = TriggerReason.FIRST_SIGHT,
    with_pixels: bool = True,
) -> Crop:
    transform = CropTransform(
        source_width=640,
        source_height=480,
        output_width=64,
        output_height=64,
        crop_x=100,
        crop_y=100,
        crop_width=64,
        crop_height=64,
    )
    return Crop(
        crop_id=CropId(crop_id),
        tenant_id=tenant,
        site_id=SITE,
        camera_id=camera,
        source_frame=frame_ref(seq, camera=camera),
        object_id=ObjectId(object_id),
        source_box=Box(0.4, 0.3, 0.55, 0.85),
        padding_applied=0.15,
        output_size=(64, 64),
        transform=transform,
        quality=QualityGrades(scale_pixels=264.0, overall=QualityLevel.GOOD),
        gate_result=GateResult.accept(),
        retention=RetentionMode.EPHEMERAL,
        privacy_class=PrivacyClass.C1_IMAGERY,
        t_capture=at(seq),
        trigger_reason=trigger,
        provenance=Provenance(
            producer_module=ModuleId("crop_manager"),
            producer_version="1.0.0",
            config_revision=ConfigRevision("test"),
        ),
        pixels=memoryview(bytes(64 * 64 * 3)) if with_pixels else None,
    )


def make_request(
    *,
    request_id: str = "req-1",
    object_id: str = "obj-1",
    crop_id: str = "crop-1",
    attributes: tuple[AttributeKey, ...] = (POSTURE,),
    class_id: ClassId = PERSON,
    camera: CameraId = CAMERA,
    tenant: TenantId = TENANT,
    seq: int = 3,
    trigger: TriggerReason = TriggerReason.FIRST_SIGHT,
    crops: int = 1,
) -> UnderstandingRequest:
    return UnderstandingRequest(
        request_id=RequestId(request_id),
        tenant_id=tenant,
        site_id=SITE,
        camera_id=camera,
        object_id=ObjectId(object_id),
        class_id=class_id,
        crop_ids=tuple(CropId(f"{crop_id}-{i}" if crops > 1 else crop_id) for i in range(crops)),
        frame_ref=frame_ref(seq, camera=camera),
        requested_attributes=attributes,
        trigger_reason=trigger,
        t_capture=at(seq),
    )


def attribute_confidence(value: float = 0.9) -> Confidence:
    return Confidence.uncalibrated(value, ConfidenceSemantics.ATTRIBUTE)


# --- understanders ----------------------------------------------------------------- #


def scripted(
    *answers: ScriptedAnswer,
    adapter_id: str = "understander.scripted",
    producible: tuple[AttributeKey, ...] = (POSTURE, HEADWEAR, CARRYING),
    cost_class: float = 1.0,
    deterministic: bool = True,
    data_residency: str = "local",
    supports_batching: bool = True,
) -> ScriptedUnderstander:
    from vision_os.core.model.ids import ModelId

    return ScriptedUnderstander(
        adapter_id=adapter_id,
        model_id=ModelId(f"model-{adapter_id}"),
        producible=producible,
        answers=answers,
        cost_class=cost_class,
        deterministic=deterministic,
        data_residency=data_residency,
        supports_batching=supports_batching,
    )


def answer_posture(value: str = "standing", **kwargs) -> ScriptedAnswer:
    return ScriptedAnswer(fields={str(POSTURE): value}, **kwargs)


class LeakyUnderstander:
    """An adapter that **violates U1** by returning undeclared fields.

    Not a bug in the test suite — a model of the real world. A VLM asked for
    ``posture`` routinely volunteers ``{"posture": "standing", "is_violation":
    true}``, and a naive adapter passes the whole object through.

    This exists because the engine's schema gate is the platform's defence
    against exactly that, and a defence can only be tested against the attack.
    The shipped ``ScriptedUnderstander`` is U1-compliant and filters undeclared
    fields itself, so it can never exercise the gate.
    """

    __slots__ = ("_calls", "_capabilities", "_fields", "_id")

    def __init__(
        self,
        *,
        fields: dict,
        producible: tuple[AttributeKey, ...] = (POSTURE,),
        adapter_id: str = "vlm.leaky",
    ) -> None:
        from vision_os.core.model.ids import ModelId
        from vision_os.core.ports.understanding import UnderstanderCapabilities

        self._id = adapter_id
        self._fields = fields
        self._calls = 0
        self._capabilities = UnderstanderCapabilities(
            producible_attributes=producible,
            model_id=ModelId(f"model-{adapter_id}"),
            supports_structured_output=True,
            deterministic=True,
        )

    @property
    def adapter_id(self) -> str:
        return self._id

    @property
    def calls(self) -> int:
        return self._calls

    def capabilities(self):
        return self._capabilities

    def understand(self, request):
        import json

        from vision_os.core.model.understanding import ModelMeta, Timing
        from vision_os.core.ports.understanding import UnderstandingPortResponse

        self._calls += 1
        return UnderstandingPortResponse(
            structured=dict(self._fields),  # every field, declared or not
            raw_output=json.dumps(self._fields, sort_keys=True, default=str).encode(),
            model_meta=ModelMeta(
                model_id=self._capabilities.model_id,
                model_version="1.0.0",
                artifact_hash="leaky:no-weights",
                adapter_id=self._id,
            ),
            timing=Timing(inference_ms=1.0, total_ms=1.0),
        )

    def understand_batch(self, requests):
        return {r.request_id: self.understand(r) for r in requests}

    def estimate_cost(self, request):
        from vision_os.core.model.understanding import CostEstimate

        return CostEstimate(cost_units=1.0, model_id=self._capabilities.model_id)


def universal_prompts() -> StaticPromptProvider:
    """Prompts with no class restriction, for tests about the value gate.

    A class-scoped prompt makes an unmatched class fail at *resolution*, which is
    correct but tests a different thing — this provider lets a request reach the
    validator so the class-applicability rule can be exercised there.
    """
    return StaticPromptProvider(
        (
            PromptTemplate(
                prompt_id=PromptId("universal.all"),
                version="1.0.0",
                template="Describe {class_id}.",
                output_keys=(POSTURE, HEADWEAR, CARRYING, HEIGHT),
            ),
        )
    )


@pytest.fixture
def understander() -> ScriptedUnderstander:
    return scripted(*[answer_posture() for _ in range(50)])


@pytest.fixture
def head() -> StaticAttributeHead:
    """A specialized head — the `attr.*` adapter 06_PORTS calls the point of P15."""
    return StaticAttributeHead(attribute=HEADWEAR, value=True, cost_class=0.01)


@pytest.fixture
def unavailable() -> UnavailableUnderstander:
    return UnavailableUnderstander(
        producible=(POSTURE, HEADWEAR), adapter_id="understander.gone"
    )


def bind(adapter, *, is_fallback: bool = False) -> BoundUnderstander:
    return BoundUnderstander(
        adapter=adapter, capabilities=adapter.capabilities(), is_fallback=is_fallback
    )


@pytest.fixture
def router(understander) -> CapabilityRouter:
    return CapabilityRouter([bind(understander)], policy=RoutingPolicy())


# --- the engine ---------------------------------------------------------------------- #


@pytest.fixture
def understanding_config() -> UnderstandingSection:
    return UnderstandingSection(
        enabled=True,
        timeout_ms=1_000,
        max_retries=1,
        circuit_breaker_threshold=2,
        circuit_breaker_cooldown_ms=5_000,
        fallback_depth=2,
        max_concurrency=4,
        remote_concurrency=2,
        max_batch_size=4,
        cache_capacity=64,
        cache_ttl_ms=60_000,
        schema_drift_window=4,
        schema_drift_threshold=0.5,
    )


@pytest.fixture
def understanding_provenance() -> Provenance:
    return Provenance(
        producer_module=ModuleId("understanding_engine"),
        producer_version="1.0.0",
        config_revision=ConfigRevision("test"),
        deterministic=True,
    )


@pytest.fixture
def engine(
    clock,
    metrics,
    bus,
    understanding_config,
    router,
    prompts,
    attribute_registry,
    understanding_provenance,
) -> UnderstandingEngine:
    return UnderstandingEngine(
        clock=clock,
        metrics=metrics,
        events=bus,
        config=understanding_config,
        router=router,
        prompts=prompts,
        coercion=JsonCoercion(),
        attributes=attribute_registry,
        provenance=understanding_provenance,
        cache=ResponseCache(capacity=64, ttl=Duration.from_millis(60_000)),
    )


def build_engine(
    clock,
    metrics,
    bus,
    config,
    *,
    understanders=(),
    fallbacks=(),
    prompt_provider=None,
    registry=None,
    coercion=None,
    provenance=None,
) -> UnderstandingEngine:
    """An engine with an explicit set of understanders. For routing tests."""
    bound = [bind(a) for a in understanders] + [bind(a, is_fallback=True) for a in fallbacks]
    # ``is None`` rather than ``or``: an empty ``StaticPromptProvider`` defines
    # ``__len__`` and is therefore falsy, so ``prompt_provider or build_prompts()``
    # would silently substitute the full catalogue for the empty one a test
    # deliberately passed — and the no-prompt path would never be exercised.
    return UnderstandingEngine(
        clock=clock,
        metrics=metrics,
        events=bus,
        config=config,
        router=CapabilityRouter(
            bound, policy=RoutingPolicy(max_fallback_depth=config.fallback_depth)
        ),
        prompts=build_prompts() if prompt_provider is None else prompt_provider,
        coercion=coercion or JsonCoercion(),
        attributes=build_registry() if registry is None else registry,
        provenance=provenance
        or Provenance(
            producer_module=ModuleId("understanding_engine"),
            producer_version="1.0.0",
            config_revision=ConfigRevision("test"),
        ),
        cache=ResponseCache(capacity=config.cache_capacity),
    )


@pytest.fixture
def understanding_runtime(
    clock, metrics, health, engine, understanding_config
) -> UnderstandingRuntime:
    return UnderstandingRuntime(
        clock=clock,
        metrics=metrics,
        health=health,
        engine=engine,
        config=understanding_config,
    )
