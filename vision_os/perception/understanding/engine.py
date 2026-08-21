"""M9 Vision Understanding Engine — pixels to schema-conformant claims.

> **Single responsibility:** *Ask a model what is true of these pixels, and return
> only what fits the declared schema.*

04_MODULES §M9 calls this *"the platform's only semantically ambitious component,
and therefore the one most tightly constrained."* Both halves matter: it is the
only place an open-ended question is asked, and the constraint is what keeps the
answer usable.

The public API is §M9's, implemented verbatim::

    understand(crop, requested_attributes, context) -> UnderstandingResult
    understand_batch(requests)                      -> Map<RequestId, Result>
    capabilities()                                  -> UnderstanderCapabilities
    estimate_cost(requested_attributes)             -> CostEstimate
    health()                                        -> ComponentHealth

``understand`` **never raises**. §M9 states the governing property:

> *understanding failure is **never** pipeline failure. Detection, tracking,
> identity, and spatial observations continue unaffected; only enrichment is
> lost.*

Every failure therefore comes back as a `UnderstandingResult` with a failure
outcome and an intact decision path, not as an exception the caller must catch.

**What this module does not do**, and why each absence is load-bearing:

*It never decides whether the call was worth making.* §M9 lists this explicitly
as a non-responsibility. M8 decided; M9 executes and reports the cost.

*It never builds an observation.* `01_LAYERED` §1.2: a synthesis layer owning
schema and ceiling enforcement *"is the only durable defense of V1 and V4"*. M9
produces candidate claims; M11 decides what is published.

*It never writes an attribute anywhere.* M7 is the only writer of Vision Objects.
Returning a value is not storing one.

*It never fabricates.* Port obligation U2. A timeout produces zero attributes and
says so — never a plausible default.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from ...core.errors import (
    OutputCoercionError,
    PromptUnavailableError,
    UnderstanderTimeoutError,
    UnderstanderUnavailableError,
)
from ...core.model.crop import Crop
from ...core.model.health import ComponentHealth, HealthState
from ...core.model.ids import (
    AttributeKey,
    BlobRef,
    EvidenceId,
    ModelId,
    RequestId,
    new_ulid,
)
from ...core.model.provenance import Provenance
from ...core.model.timebase import Duration
from ...core.model.understanding import (
    CostEstimate,
    DecisionRecord,
    ModelMeta,
    PromptMeta,
    Timing,
    UnderstandingEvidence,
    UnderstandingOutcome,
    UnderstandingRequest,
    UnderstandingResult,
    UnderstandingStep,
    bounded_note,
)
from ...core.ports.clock import Clock
from ...core.ports.understanding import (
    CropView,
    OutputSchema,
    PromptProvider,
    RenderedPrompt,
    UnderstanderCapabilities,
    UnderstandingPortRequest,
    UnderstandingPortResponse,
)
from ...kernel.config.schema import UnderstandingSection
from ...kernel.events import (
    CapabilityGap,
    EventBus,
    ModelFallbackEngaged,
    SchemaDriftSuspected,
    UnderstandingFailed,
)
from ...kernel.metrics import MetricName, MetricsEngine
from ..registry.attributes import AttributeRegistry
from .cache import ResponseCache, cache_key, group_for_batching
from .routing import BoundUnderstander, CapabilityRouter, CircuitBreaker
from .validation import AttributeValidator

UNDERSTANDING_ENGINE_ID = "understanding_engine"
UNDERSTANDING_ENGINE_VERSION = "1.0.0"


@dataclass(slots=True)
class _Attempt:
    """Mutable working state for one request. Never published as a value."""

    records: list[DecisionRecord]
    started_ns: int
    queued_ms: float = 0.0

    def note(self, step: UnderstandingStep, detail: str = "", *, at_ms: float = 0.0) -> None:
        self.records.append(DecisionRecord(step=step, detail=detail, at_ms=at_ms))

    @property
    def path(self) -> tuple[DecisionRecord, ...]:
        return tuple(self.records)


class UnderstandingEngine:
    """M9. Turns canonical crops into validated, evidenced attribute claims."""

    __slots__ = (
        "_breakers",
        "_cache",
        "_clock",
        "_coercion",
        "_config",
        "_events",
        "_failures",
        "_metrics",
        "_prompts",
        "_provenance",
        "_requests",
        "_rejection_window",
        "_router",
        "_semaphores",
        "_validator",
    )

    def __init__(
        self,
        *,
        clock: Clock,
        metrics: MetricsEngine,
        events: EventBus,
        config: UnderstandingSection,
        router: CapabilityRouter,
        prompts: PromptProvider,
        coercion,
        attributes: AttributeRegistry,
        provenance: Provenance,
        cache: ResponseCache | None = None,
    ) -> None:
        self._clock = clock
        self._metrics = metrics
        self._events = events
        self._config = config
        self._router = router
        self._prompts = prompts
        self._coercion = coercion
        self._validator = AttributeValidator(attributes)
        self._provenance = provenance
        self._cache = (
            ResponseCache(
                capacity=config.cache_capacity,
                ttl=Duration.from_millis(config.cache_ttl_ms),
            )
            if cache is None
            else cache
        )

        self._breakers: dict[ModelId, CircuitBreaker] = {}
        self._semaphores: dict[str, object] = {}
        self._requests = 0
        self._failures = 0
        self._rejection_window: list[bool] = []

    # --- public API: understand ---------------------------------------------- #

    def understand(
        self, request: UnderstandingRequest, *, crops: Sequence[Crop] = ()
    ) -> UnderstandingResult:
        """Answer one request. **Never raises.**

        Every documented failure — no capable model, no prompt, timeout, refusal,
        unparseable output — comes back as a result whose ``outcome`` names it and
        whose ``decision_path`` shows how the engine got there.
        """
        self._requests += 1
        attempt = _Attempt(records=[], started_ns=self._clock.monotonic().ns)
        try:
            return self._understand(request, crops, attempt)
        except Exception as exc:  # noqa: BLE001 - the engine is a firewall
            self._failures += 1
            self._metrics.counter(
                MetricName.UNDERSTANDING_FAILURES,
                camera_id=str(request.camera_id),
                reason="engine_guard",
            ).increment()
            attempt.note(UnderstandingStep.QUARANTINED, f"unhandled {type(exc).__name__}")
            return self._failed(
                request,
                UnderstandingOutcome.UNAVAILABLE,
                attempt,
                detail=f"{type(exc).__name__}: {exc}",
            )

    def _understand(
        self,
        request: UnderstandingRequest,
        crops: Sequence[Crop],
        attempt: _Attempt,
    ) -> UnderstandingResult:
        camera = str(request.camera_id)

        # 1. Route. No capable model is a capability gap, not a failure.
        decision = self._router.route(request.requested_attributes)
        if not decision.has_route:
            attempt.note(UnderstandingStep.NO_CAPABLE_MODEL, decision.reason)
            self._publish_capability_gap(request, decision.uncovered)
            return self._failed(
                request, UnderstandingOutcome.UNSUPPORTED, attempt, detail=decision.reason
            )
        attempt.note(
            UnderstandingStep.MODEL_SELECTED,
            f"{decision.selected.adapter_id} for {len(decision.covered)} attribute(s)",
        )
        if decision.uncovered:
            self._publish_capability_gap(request, decision.uncovered)

        # 2. Resolve and render the prompt. M9 consumes; it never authors.
        try:
            prompt = self._prompt_for(request, decision.selected, attempt)
        except PromptUnavailableError as exc:
            attempt.note(UnderstandingStep.PROMPT_UNAVAILABLE, exc.message)
            self._publish_capability_gap(request, request.requested_attributes)
            return self._failed(
                request, UnderstandingOutcome.UNSUPPORTED, attempt, detail=exc.message
            )

        # 3. The cache, keyed exactly as documented.
        key = cache_key(
            tenant_id=request.tenant_id,
            crop_id=request.primary_crop,
            prompt_version=prompt.pinned,
            model_version=decision.selected.capabilities.model_id,
            attributes=decision.covered,
        )
        cached = self._cache.get(key, self._clock.now())
        if cached is not None:
            attempt.note(UnderstandingStep.CACHE_HIT, prompt.pinned)
            self._metrics.counter(
                MetricName.UNDERSTANDING_CACHE_HITS, camera_id=camera
            ).increment()
            return replace(
                cached,
                request_id=request.request_id,
                cache_hit=True,
                cost_units=0.0,
                evidence=replace(
                    cached.evidence,
                    evidence_id=EvidenceId(self._new_id()),
                    decision_path=attempt.path,
                ),
            )
        attempt.note(UnderstandingStep.CACHE_MISS)
        self._metrics.counter(
            MetricName.UNDERSTANDING_CACHE_MISSES, camera_id=camera
        ).increment()

        # 4. Invoke, with retry and the fallback chain.
        result = self._invoke_chain(request, crops, decision, prompt, attempt)
        if not result.outcome.is_failure:
            self._cache.put(key, result, self._clock.now())
        return result

    # --- the reliability ladder ------------------------------------------------ #

    def _invoke_chain(
        self,
        request: UnderstandingRequest,
        crops: Sequence[Crop],
        decision,
        prompt: RenderedPrompt,
        attempt: _Attempt,
    ) -> UnderstandingResult:
        """Primary, then retry, then each fallback, then explicit unavailability.

        10_RELIABILITY §7.2's two rules, made mechanical: every fallback writes a
        ``FALLBACK_MODEL_USED`` step and publishes an event, so *"a fallback is
        never silent"*; and the chain terminates in ``UNAVAILABLE`` rather than a
        guess, so *"the last link is always explicit unavailability"*.
        """
        chain: list[tuple[BoundUnderstander, bool]] = [(decision.selected, False)]
        chain.extend((fallback, True) for fallback in decision.fallbacks)

        last_detail = "no understander was reachable"
        for bound, is_fallback in chain:
            breaker = self._breaker_for(bound.model_id)
            if breaker.is_open(self._clock.monotonic().ns):
                attempt.note(UnderstandingStep.CIRCUIT_OPEN, bound.adapter_id)
                self._metrics.counter(
                    MetricName.UNDERSTANDING_CIRCUIT_OPEN, model=str(bound.model_id)
                ).increment()
                last_detail = f"circuit open for {bound.adapter_id}"
                continue

            if is_fallback:
                attempt.note(UnderstandingStep.FALLBACK_MODEL_USED, bound.adapter_id)
                self._publish_fallback(request, decision.selected, bound)

            attempts = 1 + max(0, self._config.max_retries)
            for retry_index in range(attempts):
                if retry_index:
                    attempt.note(UnderstandingStep.RETRIED, f"attempt {retry_index + 1}")
                    self._metrics.counter(
                        MetricName.UNDERSTANDING_RETRIES, model=str(bound.model_id)
                    ).increment()
                try:
                    response = self._invoke(bound, request, crops, prompt, attempt)
                except UnderstanderTimeoutError as exc:
                    breaker.record_failure(self._clock.monotonic().ns)
                    last_detail = exc.message
                    attempt.note(UnderstandingStep.TIMED_OUT, bound.adapter_id)
                    self._metrics.counter(
                        MetricName.UNDERSTANDING_TIMEOUTS, model=str(bound.model_id)
                    ).increment()
                    continue
                except UnderstanderUnavailableError as exc:
                    breaker.record_failure(self._clock.monotonic().ns)
                    last_detail = exc.message
                    break
                except Exception as exc:  # noqa: BLE001 - an adapter may do anything
                    breaker.record_failure(self._clock.monotonic().ns)
                    last_detail = f"{type(exc).__name__}: {exc}"
                    self._metrics.counter(
                        MetricName.UNDERSTANDING_ADAPTER_ERRORS,
                        model=str(bound.model_id),
                    ).increment()
                    break

                breaker.record_success()
                return self._interpret(request, bound, prompt, response, attempt)

        attempt.note(UnderstandingStep.QUARANTINED, last_detail)
        self._publish_failure(request, UnderstandingOutcome.UNAVAILABLE, last_detail)
        return self._failed(
            request, UnderstandingOutcome.UNAVAILABLE, attempt, detail=last_detail
        )

    def _invoke(
        self,
        bound: BoundUnderstander,
        request: UnderstandingRequest,
        crops: Sequence[Crop],
        prompt: RenderedPrompt,
        attempt: _Attempt,
    ) -> UnderstandingPortResponse:
        """One adapter call, under the model's concurrency cap."""
        semaphore = self._semaphore_for(bound)
        if not semaphore.try_acquire():
            self._metrics.counter(
                MetricName.UNDERSTANDING_CONCURRENCY_REJECTED,
                model=str(bound.model_id),
            ).increment()
            raise UnderstanderUnavailableError(
                f"understander '{bound.adapter_id}' is at its concurrency cap "
                f"({semaphore.limit} in flight); enrichment is shed rather than "
                f"queued, because a queued call outlives the frame it describes",
                adapter_id=bound.adapter_id,
            )
        try:
            attempt.note(UnderstandingStep.INVOKED, bound.adapter_id)
            return bound.adapter.understand(
                UnderstandingPortRequest(
                    request_id=request.request_id,
                    crops=self._views(request, crops),
                    prompt=prompt,
                    output_schema=prompt.output_schema,
                    context=self._context(request),
                    max_tokens=prompt.max_output_tokens,
                    timeout=Duration.from_millis(self._config.timeout_ms),
                    temperature=self._config.temperature,
                )
            )
        finally:
            semaphore.release()

    # --- interpretation -------------------------------------------------------- #

    def _interpret(
        self,
        request: UnderstandingRequest,
        bound: BoundUnderstander,
        prompt: RenderedPrompt,
        response: UnderstandingPortResponse,
        attempt: _Attempt,
    ) -> UnderstandingResult:
        """Coerce, validate, and account for everything the model said."""
        camera = str(request.camera_id)
        model_meta = response.model_meta or self._meta_for(bound)

        if response.refused:
            attempt.note(UnderstandingStep.MODEL_REFUSED, response.refusal_reason)
            self._metrics.counter(
                MetricName.UNDERSTANDING_REFUSALS, model=str(bound.model_id)
            ).increment()
            return self._completed(
                request,
                UnderstandingOutcome.REFUSED,
                attempt,
                model_meta=model_meta,
                prompt=prompt,
                response=response,
                note=response.refusal_reason or "the model declined to answer",
            )

        structured, unparsed = self._coerce(response, prompt.output_schema, attempt)

        outcome_note = bounded_note(unparsed)
        validation = self._validator.validate(
            structured,
            schema=prompt.output_schema,
            class_id=request.class_id,
            observed_at=request.t_capture,
            producer=self._attribution(model_meta, prompt),
            evidence_ref=str(request.primary_crop),
            field_confidence=response.field_confidence,
        )

        if validation.rejected:
            attempt.note(
                UnderstandingStep.SCHEMA_REJECTED,
                f"{len(validation.rejected)} field(s) refused",
            )
            for rejection in validation.rejected:
                self._metrics.counter(
                    MetricName.ATTRIBUTES_SCHEMA_REJECTED,
                    camera_id=camera,
                    reason=rejection.reason.value,
                ).increment()
        if validation.accepted:
            attempt.note(
                UnderstandingStep.COERCED, f"{len(validation.accepted)} attribute(s)"
            )
            attempt.note(UnderstandingStep.CONFIDENCE_MARKED_SELF_REPORTED)

        self._note_rejection_rate(request, validation)

        if validation.produced_nothing:
            attempt.note(UnderstandingStep.QUARANTINED, "nothing survived the schema")

        outcome = (
            UnderstandingOutcome.SUCCEEDED
            if validation.accepted
            else UnderstandingOutcome.NO_ATTRIBUTES
        )
        self._metrics.counter(
            MetricName.UNDERSTANDING_RESULTS, camera_id=camera, outcome=outcome.value
        ).increment()
        self._metrics.counter(
            MetricName.ATTRIBUTES_PRODUCED, camera_id=camera
        ).increment(len(validation.accepted))
        self._metrics.histogram(
            MetricName.UNDERSTANDING_LATENCY_MS, model=str(bound.model_id)
        ).record(response.timing.total_ms)

        return self._completed(
            request,
            outcome,
            attempt,
            model_meta=model_meta,
            prompt=prompt,
            response=response,
            note=outcome_note,
            attributes=validation.accepted,
            rejected=validation.rejected,
            cost=bound.capabilities.cost_class,
        )

    def _coerce(
        self,
        response: UnderstandingPortResponse,
        schema: OutputSchema,
        attempt: _Attempt,
    ) -> tuple[Mapping[str, object], str | None]:
        """Parse what the adapter returned, preserving what did not parse.

        An adapter with structured output support has already parsed; the
        coercion port runs only over free text. Running it twice would risk a
        second interpretation of an answer that was already unambiguous.
        """
        if response.structured:
            attempt.note(UnderstandingStep.STRUCTURED_PARSE, "adapter-parsed")
            return response.structured, response.unparsed

        if not response.unparsed:
            return {}, None

        try:
            coerced = self._coercion.coerce(response.unparsed, schema=schema)
        except Exception as exc:  # noqa: BLE001 - X4: coercion must never raise
            attempt.note(UnderstandingStep.QUARANTINED, f"coercion raised: {exc}")
            raise OutputCoercionError(
                f"coercion strategy '{getattr(self._coercion, 'strategy_id', '?')}' "
                f"raised {type(exc).__name__}; obligation X4 requires malformed "
                f"text to return an empty parse, never an exception"
            ) from exc

        attempt.note(
            UnderstandingStep.REPARSE_ATTEMPTED if coerced.reparsed else UnderstandingStep.STRUCTURED_PARSE,
            coerced.strategy_used,
        )
        return coerced.parsed, coerced.unparsed

    # --- public API: batch, capabilities, cost, health -------------------------- #

    def understand_batch(
        self,
        requests: Sequence[UnderstandingRequest],
        *,
        crops: Mapping[RequestId, Sequence[Crop]] | None = None,
    ) -> dict[RequestId, UnderstandingResult]:
        """Answer many requests. **Every request id appears in the result.**

        Grouping is by ``(adapter, prompt_version)`` per 08_RUNTIME §1 — only
        compatible requests batch together, because two prompts are two questions
        and answering one while attributing it to both is fabrication with extra
        steps.

        A dropped id would be indistinguishable from a lost one, so the mapping
        is total even when everything failed.
        """
        pixels = crops or {}
        results: dict[RequestId, UnderstandingResult] = {}
        for request in requests:
            results[request.request_id] = self.understand(
                request, crops=pixels.get(request.request_id, ())
            )
        return results

    def plan_batches(self, requests: Sequence[UnderstandingRequest]):
        """How these requests would group. Exposed for tests and telemetry.

        Separated from execution so batch composition is checkable without
        invoking a model — 08_RUNTIME §4.3 requires composition not to depend on
        arrival timing, and that is only testable if it can be observed alone.
        """
        keys: list[tuple[str, str]] = []
        for request in requests:
            decision = self._router.route(request.requested_attributes)
            adapter_id = decision.selected.adapter_id if decision.has_route else ""
            resolved = self._resolve_prompt(request, decision.selected)
            keys.append((adapter_id, f"{resolved[0]}@{resolved[1]}" if resolved else ""))
        return group_for_batching(keys, max_batch_size=self._config.max_batch_size)

    def capabilities(self) -> tuple[UnderstanderCapabilities, ...]:
        """What every bound understander can produce.

        Published *"so capability gaps are visible"* (§M9). M8's demand registry
        reads this to refuse a demand honestly at registration rather than
        leaving a consumer waiting for an attribute nothing can produce.
        """
        return self._router.capabilities()

    def producible_attributes(self) -> frozenset[AttributeKey]:
        return self._router.producible_attributes()

    def estimate_cost(self, attributes: Sequence[AttributeKey]) -> CostEstimate:
        """What answering this would cost, before spending it.

        Port obligation U7 and §M9's *"governed by M8 rather than by itself"*: M9
        answers the question so M8's budget policy can decide, and never acts on
        the answer itself.
        """
        decision = self._router.route(attributes)
        if not decision.has_route:
            return CostEstimate(
                cost_units=0.0,
                attributes_covered=decision.covered,
                attributes_uncovered=decision.uncovered,
            )
        return CostEstimate(
            cost_units=decision.selected.cost_class,
            model_id=decision.selected.model_id,
            estimated_latency=Duration.from_millis(
                int(decision.selected.capabilities.latency_p95_ms) or 0
            ),
            attributes_covered=decision.covered,
            attributes_uncovered=decision.uncovered,
        )

    def health(self) -> ComponentHealth:
        open_circuits = [
            str(model_id)
            for model_id, breaker in self._breakers.items()
            if breaker.is_open(self._clock.monotonic().ns)
        ]
        state = HealthState.HEALTHY
        detail = "understanding nominal"
        if not len(self._router):
            state = HealthState.DEGRADED
            detail = (
                "no understander is bound; attributes stop while detection, "
                "tracking and spatial observations continue (10_RELIABILITY "
                "section 4.3 step 5)"
            )
        elif open_circuits:
            state = HealthState.DEGRADED
            detail = f"circuit open for {', '.join(sorted(open_circuits))}"
        elif self._failures:
            state = HealthState.DEGRADED
            detail = f"{self._failures} engine failures"

        return ComponentHealth(
            component_id=UNDERSTANDING_ENGINE_ID,
            state=state,
            reported_at=self._clock.now(),
            detail=detail,
            metrics={
                "requests": float(self._requests),
                "failures": float(self._failures),
                "cache_hit_rate": self._cache.stats().hit_rate,
                "understanders_bound": float(len(self._router)),
            },
        )

    # --- construction helpers --------------------------------------------------- #

    def _prompt_for(
        self, request: UnderstandingRequest, bound: BoundUnderstander, attempt: _Attempt
    ) -> RenderedPrompt:
        resolved = self._resolve_prompt(request, bound)
        if resolved is None:
            raise PromptUnavailableError(
                f"no prompt covers {sorted(request.requested_attributes)} for class "
                f"'{request.class_id}'; M8 records a capability gap so the "
                f"consumer stops waiting (04_MODULES section M10)",
                attributes=tuple(str(k) for k in request.requested_attributes),
            )
        prompt_id, version = resolved
        prompt = self._prompts.render(prompt_id, version, self._context(request))
        attempt.note(UnderstandingStep.PROMPT_RESOLVED, prompt.pinned)
        return prompt

    def _resolve_prompt(self, request: UnderstandingRequest, bound) -> tuple | None:
        if bound is None:
            return None
        return self._prompts.resolve(
            request.requested_attributes,
            class_id=request.class_id,
            model_family=bound.capabilities.model_id,
        )

    def _context(self, request: UnderstandingRequest) -> dict[str, object]:
        """Rendering context — 04_MODULES §M10 renders with *"object class,
        requested attributes, quality hints, prior values"*.

        Note what is absent: no tenant, no site, no object id, no region label. A
        prompt that could name the subject could be asked about the subject, and
        the ceiling would have a hole in it shaped like a template variable.
        """
        return {
            "class_id": str(request.class_id),
            "requested_attributes": [str(k) for k in request.requested_attributes],
            "prior_attributes": dict(request.prior_attributes),
            "quality": request.quality,
        }

    def _views(
        self, request: UnderstandingRequest, crops: Sequence[Crop]
    ) -> tuple[CropView, ...]:
        views: list[CropView] = []
        for crop in crops:
            if crop.pixels is None:
                continue
            views.append(
                CropView(
                    crop_id=crop.crop_id,
                    pixels=crop.pixels,
                    width=crop.width,
                    height=crop.height,
                    colour_space=crop.transform.colour_space,
                )
            )
        if views:
            return tuple(views)
        raise UnderstanderUnavailableError(
            f"request '{request.request_id}' carries no readable crop pixels; the "
            f"crop was evicted before understanding could run",
            request_id=str(request.request_id),
        )

    def _attribution(self, model: ModelMeta, prompt: RenderedPrompt) -> Provenance:
        """Provenance for every attribute this call produces.

        Carries the model, its exact weights and the prompt version, because
        02_VOM §7.2 and V4 both require that a claim be traceable to the thing
        that made it — not to the module that passed it along.
        """
        return replace(
            self._provenance,
            adapter_id=model.adapter_id or None,
            model_id=model.model_id,
            model_version=model.model_version,
            model_artifact_hash=model.artifact_hash,
            deterministic=model.deterministic,
        )

    def _meta_for(self, bound: BoundUnderstander) -> ModelMeta:
        return ModelMeta(
            model_id=bound.model_id,
            model_version="unknown",
            artifact_hash="unreported",
            adapter_id=bound.adapter_id,
            deterministic=bound.capabilities.deterministic,
        )

    def _completed(
        self,
        request: UnderstandingRequest,
        outcome: UnderstandingOutcome,
        attempt: _Attempt,
        *,
        model_meta: ModelMeta,
        prompt: RenderedPrompt,
        response: UnderstandingPortResponse,
        note: str | None = None,
        attributes: Sequence = (),
        rejected: Sequence = (),
        cost: float = 0.0,
    ) -> UnderstandingResult:
        raw = response.raw_output or b""
        return UnderstandingResult(
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            site_id=request.site_id,
            camera_id=request.camera_id,
            object_id=request.object_id,
            class_id=request.class_id,
            outcome=outcome,
            evidence=self._evidence(request, attempt, prompt, raw, note, response.timing),
            provenance=self._attribution(model_meta, prompt),
            attributes=tuple(attributes),
            rejected_fields=tuple(rejected),
            model_used=model_meta,
            prompt_used=PromptMeta(
                prompt_id=prompt.prompt_id,
                version=prompt.version,
                content_hash=prompt.content_hash,
            ),
            requested_attributes=tuple(request.requested_attributes),
            demand_ids=tuple(request.demand_ids),
            cost_units=cost,
            raw_output=raw or None,
        )

    def _failed(
        self,
        request: UnderstandingRequest,
        outcome: UnderstandingOutcome,
        attempt: _Attempt,
        *,
        detail: str = "",
    ) -> UnderstandingResult:
        """A failure result: intact evidence, **zero** attributes.

        The result exists rather than an exception because §M9 requires the
        pipeline to continue, and because a consumer needs to know the platform
        looked and could not answer — which is a different fact from the platform
        never having looked (V8).
        """
        self._metrics.counter(
            MetricName.UNDERSTANDING_RESULTS,
            camera_id=str(request.camera_id),
            outcome=outcome.value,
        ).increment()
        return UnderstandingResult(
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            site_id=request.site_id,
            camera_id=request.camera_id,
            object_id=request.object_id,
            class_id=request.class_id,
            outcome=outcome,
            evidence=self._evidence(request, attempt, None, b"", detail or None, Timing()),
            provenance=self._provenance,
            requested_attributes=tuple(request.requested_attributes),
            demand_ids=tuple(request.demand_ids),
        )

    def _evidence(
        self,
        request: UnderstandingRequest,
        attempt: _Attempt,
        prompt: RenderedPrompt | None,
        raw: bytes,
        note: str | None,
        timing: Timing,
    ) -> UnderstandingEvidence:
        elapsed = (self._clock.monotonic().ns - attempt.started_ns) / 1_000_000
        return UnderstandingEvidence(
            evidence_id=EvidenceId(self._new_id()),
            trigger_reason=request.trigger_reason,
            input_hash=self._input_hash(request, prompt),
            frame_ref=request.frame_ref,
            crop_ref=request.primary_crop,
            raw_output_ref=blob_ref(raw) if raw else None,
            unstructured_note=bounded_note(note),
            decision_path=attempt.path,
            timing=replace(timing, total_ms=timing.total_ms or elapsed),
            retention=self._config.evidence_retention,
        )

    @staticmethod
    def _input_hash(request: UnderstandingRequest, prompt: RenderedPrompt | None) -> str:
        """Hash of the exact model input — crop content plus rendered prompt.

        02_VOM §10.9. Two results with the same input hash and different answers
        proves the model is non-deterministic, which is a claim worth being able
        to make with evidence rather than by assertion.
        """
        digest = hashlib.sha256()
        for crop_id in request.crop_ids:
            digest.update(str(crop_id).encode())
        if prompt is not None:
            digest.update(prompt.pinned.encode())
            digest.update(prompt.text.encode())
        return digest.hexdigest()

    def _new_id(self) -> str:
        return new_ulid(now_ms=self._clock.now().ns // 1_000_000)

    def _breaker_for(self, model_id: ModelId) -> CircuitBreaker:
        breaker = self._breakers.get(model_id)
        if breaker is None:
            breaker = CircuitBreaker(
                model_id=model_id,
                threshold=self._config.circuit_breaker_threshold,
                cooldown_ns=self._config.circuit_breaker_cooldown_ms * 1_000_000,
            )
            self._breakers[model_id] = breaker
        return breaker

    def _semaphore_for(self, bound: BoundUnderstander):
        from .cache import ModelSemaphore

        semaphore = self._semaphores.get(bound.adapter_id)
        if semaphore is None:
            limit = (
                self._config.remote_concurrency
                if bound.capabilities.is_remote
                else self._config.max_concurrency
            )
            semaphore = ModelSemaphore(max(1, limit))
            self._semaphores[bound.adapter_id] = semaphore
        return semaphore

    # --- alarms ------------------------------------------------------------------ #

    def _note_rejection_rate(self, request: UnderstandingRequest, validation) -> None:
        """Alarm on sustained ceiling violations.

        §M9: *"If the rate is sustained, alarm — this means a prompt has drifted
        beyond its declared schema."* One rejection is a model being creative;
        fifty in a row is a prompt asking for something the registry does not
        hold, and that is a deploy-time problem showing up at inference time.
        """
        window = self._config.schema_drift_window
        self._rejection_window.append(bool(validation.ceiling_violations))
        if len(self._rejection_window) > window:
            del self._rejection_window[:-window]
        if len(self._rejection_window) < window:
            return
        rate = sum(self._rejection_window) / len(self._rejection_window)
        if rate < self._config.schema_drift_threshold:
            return
        self._rejection_window.clear()
        self._events.publish(
            SchemaDriftSuspected(
                occurred_at=self._clock.now(),
                partition_key=str(request.camera_id),
                camera_id=request.camera_id,
                rejection_rate=rate,
                sample_size=window,
                detail=(
                    "sustained unregistered-key rejections; a prompt has drifted "
                    "beyond its declared output schema"
                ),
            )
        )
        self._metrics.counter(MetricName.SCHEMA_DRIFT_ALARMS).increment()

    def _publish_fallback(
        self, request: UnderstandingRequest, primary, fallback
    ) -> None:
        """A fallback is never silent (10_RELIABILITY §7.2 rule 1).

        Without this event a fallback *"becomes permanent, and the platform
        quietly runs on its worst model forever."*
        """
        self._events.publish(
            ModelFallbackEngaged(
                occurred_at=self._clock.now(),
                partition_key=str(request.camera_id),
                camera_id=request.camera_id,
                primary_model=str(primary.model_id),
                fallback_model=str(fallback.model_id),
                detail="the primary understander was unavailable",
            )
        )
        self._metrics.counter(
            MetricName.UNDERSTANDING_FALLBACKS, model=str(fallback.model_id)
        ).increment()

    def _publish_capability_gap(
        self, request: UnderstandingRequest, uncovered: Sequence[AttributeKey]
    ) -> None:
        if not uncovered:
            return
        self._events.publish(
            CapabilityGap(
                occurred_at=self._clock.now(),
                partition_key=str(request.camera_id),
                camera_id=request.camera_id,
                attribute_key=str(uncovered[0]),
                reason="no_capable_model",
                detail=(
                    f"{len(uncovered)} requested attribute(s) cannot be produced "
                    f"by any bound understander"
                ),
            )
        )
        self._metrics.counter(
            MetricName.UNDERSTANDING_UNSUPPORTED, camera_id=str(request.camera_id)
        ).increment()

    def _publish_failure(
        self, request: UnderstandingRequest, outcome: UnderstandingOutcome, detail: str
    ) -> None:
        self._events.publish(
            UnderstandingFailed(
                occurred_at=self._clock.now(),
                partition_key=str(request.camera_id),
                camera_id=request.camera_id,
                outcome=outcome.value,
                detail=detail,
            )
        )
        self._metrics.counter(
            MetricName.UNDERSTANDING_FAILURES,
            camera_id=str(request.camera_id),
            reason=outcome.value,
        ).increment()

    # --- access -------------------------------------------------------------------- #

    @property
    def cache(self) -> ResponseCache:
        return self._cache

    @property
    def router(self) -> CapabilityRouter:
        return self._router

    @property
    def validator(self) -> AttributeValidator:
        return self._validator

    @property
    def requests(self) -> int:
        return self._requests

    @property
    def failures(self) -> int:
        return self._failures

    def breaker(self, model_id: ModelId) -> CircuitBreaker | None:
        return self._breakers.get(model_id)

    def in_flight(self, adapter_id: str) -> int:
        semaphore = self._semaphores.get(adapter_id)
        return semaphore.in_flight if semaphore else 0


def blob_ref(payload: bytes) -> BlobRef:
    """Content-address a byte payload (02_VOM §10.9).

    A hash rather than a path for the same reason ``CropId`` is: identical model
    output stored twice is one blob, the reference survives storage migration, and
    the reference is computable before any store exists. Flow 6 computes it;
    persisting the bytes is M13's job through P22.
    """
    return BlobRef("sha256:" + hashlib.sha256(payload).hexdigest())
