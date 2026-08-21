"""M9's public API, and the reliability ladder underneath it.

Two properties this file exists to defend.

**Understanding failure is never pipeline failure.** §M9 states it, and
``understand`` never raising is how it becomes true rather than aspirational:
every documented failure comes back as a result whose outcome names it and whose
decision path shows how the engine got there.

**Nothing is ever fabricated.** Port obligation U2 calls fabrication *"the single
most dangerous failure mode for a VLM-based system, because fabricated output is
indistinguishable from real output downstream."* Every failure test here asserts
zero attributes, and ``UnderstandingResult.__post_init__`` refuses to construct
the alternative.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.understanding import (
    ScriptedAnswer,
    StaticPromptProvider,
)
from vision_os.core.model.crop import TriggerReason
from vision_os.core.model.ids import (
    ConfigRevision,
    CropId,
    EvidenceId,
    ModelId,
    ModuleId,
    RequestId,
)
from vision_os.core.model.provenance import Provenance
from vision_os.core.model.understanding import (
    ModelMeta,
    RejectionReason,
    UnderstandingEvidence,
    UnderstandingOutcome,
    UnderstandingResult,
    UnderstandingStep,
)

from ..conftest import (
    CAMERA,
    CARRYING,
    HEADWEAR,
    HEIGHT,
    PERSON,
    POSTURE,
    SITE,
    TENANT,
    UNREGISTERED,
    VEHICLE,
    LeakyUnderstander,
    answer_posture,
    at,
    build_engine,
    frame_ref,
    make_crop,
    make_request,
    scripted,
    universal_prompts,
)


def build_result(
    *,
    outcome: UnderstandingOutcome = UnderstandingOutcome.NO_ATTRIBUTES,
    raw_output: bytes | None = None,
) -> UnderstandingResult:
    """A minimal result, for cache and model tests that are not about the engine."""
    model = (
        ModelMeta(
            model_id=ModelId("m"),
            model_version="1.0.0",
            artifact_hash="hash",
        )
        if outcome.produced_a_model_call
        else None
    )
    return UnderstandingResult(
        request_id=RequestId("req-1"),
        tenant_id=TENANT,
        site_id=SITE,
        camera_id=CAMERA,
        object_id=None,
        class_id=PERSON,
        outcome=outcome,
        evidence=UnderstandingEvidence(
            evidence_id=EvidenceId("ev-1"),
            trigger_reason=TriggerReason.FIRST_SIGHT,
            input_hash="hash",
            frame_ref=frame_ref(3),
            crop_ref=CropId("crop-1"),
        ),
        provenance=Provenance(
            producer_module=ModuleId("understanding_engine"),
            producer_version="1.0.0",
            config_revision=ConfigRevision("test"),
        ),
        model_used=model,
        raw_output=raw_output,
    )


class TestTheHappyPath:
    def test_a_registered_attribute_is_produced(self, engine) -> None:
        result = engine.understand(make_request(), crops=[make_crop()])
        assert result.outcome is UnderstandingOutcome.SUCCEEDED
        assert result.attribute_keys == (POSTURE,)
        assert result.attribute(POSTURE).value == "standing"

    def test_the_result_carries_its_model(self, engine) -> None:
        result = engine.understand(make_request(), crops=[make_crop()])
        assert result.model_used is not None
        assert result.model_used.artifact_hash, "the exact weights are mandatory (V4)"

    def test_the_result_carries_its_prompt_version(self, engine) -> None:
        result = engine.understand(make_request(), crops=[make_crop()])
        assert result.prompt_used is not None
        assert result.prompt_used.pinned == "person.posture@1.0.0"

    def test_the_result_carries_its_trigger_reason(self, engine) -> None:
        """*Why this was computed at all* — inherited from the crop, never
        re-derived (02_VOM §10.9)."""
        result = engine.understand(
            make_request(trigger=TriggerReason.ATTRIBUTE_STALE), crops=[make_crop()]
        )
        assert result.evidence.trigger_reason is TriggerReason.ATTRIBUTE_STALE

    def test_the_attribute_is_stamped_with_capture_time(self, engine) -> None:
        """V11. Inference time would make an attribute a measurement of the
        platform rather than of the world."""
        result = engine.understand(make_request(seq=7), crops=[make_crop(seq=7)])
        assert result.attribute(POSTURE).observed_at == at(7)

    def test_requested_attributes_are_recorded(self, engine) -> None:
        """The difference between requested and produced is the coverage gap,
        and a consumer needs both to compute it."""
        engine_with = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[scripted(answer_posture(), producible=(POSTURE, HEADWEAR))],
        )
        result = engine_with.understand(
            make_request(attributes=(POSTURE, HEADWEAR)), crops=[make_crop()]
        )
        assert result.requested_attributes == (POSTURE, HEADWEAR)
        assert result.unsatisfied == (HEADWEAR,)


class TestTheSchemaGate:
    def test_an_unregistered_field_is_rejected_and_recorded(self, engine) -> None:
        """The ceiling, at the engine level. The model said it; the schema refused.

        Run against a **U1-violating** adapter, because that is the case the gate
        exists for: a compliant adapter filters undeclared fields itself, so it
        can never exercise the platform's own defence.
        """
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[
                LeakyUnderstander(
                    fields={str(POSTURE): "standing", str(UNREGISTERED): True}
                )
            ],
        )
        result = e.understand(make_request(), crops=[make_crop()])
        assert result.attribute_keys == (POSTURE,)
        assert result.ceiling_violations, (
            "a volunteered judgment must be recorded, not silently dropped"
        )
        assert result.ceiling_violations[0].field_name == str(UNREGISTERED)

    def test_a_judgment_never_becomes_an_attribute(self, engine) -> None:
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[LeakyUnderstander(fields={str(UNREGISTERED): True})],
        )
        result = e.understand(make_request(), crops=[make_crop()])
        assert not result.attributes
        assert result.outcome is UnderstandingOutcome.NO_ATTRIBUTES

    def test_a_bad_value_is_rejected_with_its_reason(self, engine) -> None:
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[LeakyUnderstander(fields={str(POSTURE): "levitating"})],
        )
        result = e.understand(make_request(), crops=[make_crop()])
        assert result.rejected_fields[0].reason is RejectionReason.OUT_OF_DOMAIN
        assert not result.ceiling_violations, (
            "a bad enum value is a formatting problem, not a ceiling breach; "
            "conflating them would send an operator to the wrong fix"
        )

    def test_the_decision_path_records_the_rejection(self, engine) -> None:
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[LeakyUnderstander(fields={str(POSTURE): "levitating"})],
        )
        result = e.understand(make_request(), crops=[make_crop()])
        assert result.evidence.took(UnderstandingStep.SCHEMA_REJECTED)

    def test_a_class_mismatch_is_rejected(self, engine) -> None:
        """``posture`` is scoped to ``person``; a vehicle cannot carry it.

        Uses a class-agnostic prompt so the request actually reaches the gate. A
        class-scoped prompt refuses earlier, at resolution — a stronger guarantee
        covered by the next test.
        """
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[
                LeakyUnderstander(
                    fields={str(POSTURE): "standing"}, adapter_id="vlm.leaky"
                )
            ],
            prompt_provider=universal_prompts(),
        )
        result = e.understand(make_request(class_id=VEHICLE), crops=[make_crop()])
        assert result.rejected_fields[0].reason is RejectionReason.CLASS_NOT_APPLICABLE

    def test_a_class_scoped_prompt_refuses_earlier(self, engine) -> None:
        """No prompt covers ``posture`` for a vehicle, so nothing is even asked.

        Refusing at resolution is strictly better than refusing at validation: it
        costs no model call at all.
        """
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[scripted(answer_posture(), producible=(POSTURE,))],
        )
        result = e.understand(make_request(class_id=VEHICLE), crops=[make_crop()])
        assert result.outcome is UnderstandingOutcome.UNSUPPORTED


class TestUnstructuredOutput:
    def test_prose_is_preserved_never_promoted(self, engine) -> None:
        """02_VOM §9.3: inspectable, never promoted.

        Not queryable as fact, never entering Vision State, never filterable —
        it exists so a human debugging an odd result can see what the model said.
        """
        adapter = scripted(
            ScriptedAnswer(fields={}, unparsed="I cannot tell from this image."),
            producible=(POSTURE,),
        )
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[adapter],
        )
        result = e.understand(make_request(), crops=[make_crop()])
        assert not result.attributes
        assert result.evidence.unstructured_note == "I cannot tell from this image."
        assert result.outcome is UnderstandingOutcome.NO_ATTRIBUTES

    def test_nothing_coercible_emits_zero_attributes(self, engine) -> None:
        """§M9: *"quarantine to `unstructured_note` and emit **zero**
        attributes."*"""
        adapter = scripted(
            ScriptedAnswer(fields={}, unparsed="the subject seems fine"),
            producible=(POSTURE,),
        )
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[adapter],
        )
        result = e.understand(make_request(), crops=[make_crop()])
        assert result.attributes == ()
        assert result.evidence.took(UnderstandingStep.QUARANTINED)

    def test_a_long_note_is_bounded_and_marked(self, engine) -> None:
        adapter = scripted(
            ScriptedAnswer(fields={}, unparsed="x" * 100_000), producible=(POSTURE,)
        )
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[adapter],
        )
        result = e.understand(make_request(), crops=[make_crop()])
        assert len(result.evidence.unstructured_note) <= 4_096
        assert "truncated" in result.evidence.unstructured_note


class TestNoFabrication:
    def test_a_timeout_produces_zero_attributes(self, engine) -> None:
        """**U2.** Never a plausible default."""
        adapter = scripted(
            ScriptedAnswer(raise_timeout=True),
            ScriptedAnswer(raise_timeout=True),
            producible=(POSTURE,),
        )
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[adapter],
        )
        result = e.understand(make_request(), crops=[make_crop()])
        assert result.attributes == ()
        assert result.outcome is UnderstandingOutcome.UNAVAILABLE

    def test_a_refusal_is_recorded_as_evidence(self, engine) -> None:
        """§M9: *"Record refusal as evidence; emit no attributes; count."*"""
        adapter = scripted(
            ScriptedAnswer(refused=True, refusal_reason="content policy"),
            producible=(POSTURE,),
        )
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[adapter],
        )
        result = e.understand(make_request(), crops=[make_crop()])
        assert result.outcome is UnderstandingOutcome.REFUSED
        assert result.attributes == ()
        assert "content policy" in result.evidence.unstructured_note
        assert result.evidence.took(UnderstandingStep.MODEL_REFUSED)

    def test_a_failed_result_cannot_carry_attributes(self, engine) -> None:
        """The model type itself refuses to represent fabrication.

        Constructed directly rather than via the engine, because the engine can
        never produce this — which is the point. The type is the last line: even
        a future bug cannot express a timeout that produced a value.
        """
        from dataclasses import replace

        succeeded = engine.understand(make_request(), crops=[make_crop()])
        assert succeeded.attributes, "a fixture precondition"
        with pytest.raises(ValueError, match="fabrication"):
            replace(succeeded, outcome=UnderstandingOutcome.TIMED_OUT)

    def test_succeeded_requires_an_attribute(self) -> None:
        """*"The model answered and nothing fit"* is NO_ATTRIBUTES.

        Conflating them would hide a schema drift problem behind a success.
        """
        with pytest.raises(ValueError, match="NO_ATTRIBUTES"):
            build_result(outcome=UnderstandingOutcome.SUCCEEDED)


class TestTheReliabilityLadder:
    def test_a_timeout_is_retried_once(self, engine) -> None:
        """10_RELIABILITY §4.3: *"Retry once with backoff."*"""
        adapter = scripted(
            ScriptedAnswer(raise_timeout=True),
            answer_posture(),
            producible=(POSTURE,),
        )
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[adapter],
        )
        result = e.understand(make_request(), crops=[make_crop()])
        assert result.outcome is UnderstandingOutcome.SUCCEEDED
        assert result.evidence.took(UnderstandingStep.RETRIED)

    def test_a_fallback_answers_when_the_primary_is_gone(self, engine, unavailable) -> None:
        primary = scripted(
            ScriptedAnswer(raise_unavailable=True),
            producible=(POSTURE,),
            adapter_id="vlm.primary",
        )
        fallback = scripted(
            answer_posture("sitting"), producible=(POSTURE,), adapter_id="vlm.fallback"
        )
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[primary], fallbacks=[fallback],
        )
        result = e.understand(make_request(), crops=[make_crop()])
        assert result.outcome is UnderstandingOutcome.SUCCEEDED
        assert result.attribute(POSTURE).value == "sitting"

    def test_a_fallback_is_never_silent(self, engine, bus) -> None:
        """10_RELIABILITY §7.2 rule 1. Without the event a fallback *"becomes
        permanent, and the platform quietly runs on its worst model forever."*"""
        subscription = bus.subscribe(["understanding.fallback_engaged"])
        primary = scripted(
            ScriptedAnswer(raise_unavailable=True),
            producible=(POSTURE,),
            adapter_id="vlm.primary",
        )
        fallback = scripted(
            answer_posture(), producible=(POSTURE,), adapter_id="vlm.fallback"
        )
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[primary], fallbacks=[fallback],
        )
        result = e.understand(make_request(), crops=[make_crop()])
        assert subscription.drain(), "a fallback must publish"
        assert result.evidence.used_fallback, "and must be visible in the evidence"

    def test_the_chain_ends_in_explicit_unavailability(self, engine) -> None:
        """10_RELIABILITY §7.2 rule 2: *"never a guess."*"""
        primary = scripted(
            ScriptedAnswer(raise_unavailable=True),
            producible=(POSTURE,),
            adapter_id="vlm.primary",
        )
        fallback = scripted(
            ScriptedAnswer(raise_unavailable=True),
            producible=(POSTURE,),
            adapter_id="vlm.fallback",
        )
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[primary], fallbacks=[fallback],
        )
        result = e.understand(make_request(), crops=[make_crop()])
        assert result.outcome is UnderstandingOutcome.UNAVAILABLE
        assert result.attributes == ()

    def test_the_circuit_opens_after_repeated_failure(self, engine) -> None:
        adapter = scripted(
            *[ScriptedAnswer(raise_unavailable=True) for _ in range(20)],
            producible=(POSTURE,),
        )
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[adapter],
        )
        for _ in range(4):
            e.understand(make_request(), crops=[make_crop()])
        breaker = e.breaker(adapter.capabilities().model_id)
        assert breaker is not None
        assert breaker.trips >= 1

    def test_an_open_circuit_skips_the_model(self, engine) -> None:
        adapter = scripted(
            *[ScriptedAnswer(raise_unavailable=True) for _ in range(20)],
            producible=(POSTURE,),
        )
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[adapter],
        )
        for _ in range(3):
            e.understand(make_request(), crops=[make_crop()])
        before = adapter.calls
        result = e.understand(make_request(), crops=[make_crop()])
        assert adapter.calls == before, "an open circuit must not call the model"
        assert result.evidence.took(UnderstandingStep.CIRCUIT_OPEN)

    def test_an_adapter_that_explodes_does_not_escape(self, engine) -> None:
        class _Exploding:
            adapter_id = "vlm.exploding"

            def capabilities(self):
                return scripted(producible=(POSTURE,)).capabilities()

            def understand(self, request):
                raise RuntimeError("boom")

            def understand_batch(self, requests):
                raise RuntimeError("boom")

            def estimate_cost(self, request):
                raise RuntimeError("boom")

        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[_Exploding()],
        )
        result = e.understand(make_request(), crops=[make_crop()])
        assert result.outcome is UnderstandingOutcome.UNAVAILABLE
        assert result.attributes == ()


class TestCapabilityGaps:
    def test_an_unproducible_attribute_is_unsupported_not_failed(self, engine) -> None:
        """A capability gap is the honest answer, not a fault."""
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[scripted(answer_posture(), producible=(POSTURE,))],
        )
        result = e.understand(make_request(attributes=(HEIGHT,)), crops=[make_crop()])
        assert result.outcome is UnderstandingOutcome.UNSUPPORTED
        assert result.evidence.took(UnderstandingStep.NO_CAPABLE_MODEL)

    def test_a_capability_gap_is_published(self, engine, bus) -> None:
        subscription = bus.subscribe(["cropping.capability_gap"])
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[scripted(answer_posture(), producible=(POSTURE,))],
        )
        e.understand(make_request(attributes=(HEIGHT,)), crops=[make_crop()])
        assert subscription.drain(), (
            "the consumer must learn the demand cannot be served here (V8)"
        )

    def test_no_understander_bound_is_unsupported_not_a_crash(self, engine) -> None:
        """10_RELIABILITY §4.3 step 5: attributes stop, everything else runs."""
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[],
        )
        result = e.understand(make_request(), crops=[make_crop()])
        assert result.outcome is UnderstandingOutcome.UNSUPPORTED

    def test_no_prompt_is_a_capability_gap(self, engine) -> None:
        """04_MODULES §M10 returns ``NoSuitablePrompt``, and M8 records a gap."""
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[scripted(answer_posture(), producible=(POSTURE,))],
            prompt_provider=StaticPromptProvider(()),
        )
        result = e.understand(make_request(), crops=[make_crop()])
        assert result.outcome is UnderstandingOutcome.UNSUPPORTED
        assert result.evidence.took(UnderstandingStep.PROMPT_UNAVAILABLE)


class TestTheCacheInTheEngine:
    def test_a_repeat_request_hits_the_cache(self, engine) -> None:
        request = make_request()
        crop = make_crop()
        engine.understand(request, crops=[crop])
        second = engine.understand(request, crops=[crop])
        assert second.cache_hit
        assert second.cost_units == 0.0, "a cache hit costs nothing"

    def test_a_cache_hit_does_not_call_the_model(self, engine, understander) -> None:
        request = make_request()
        engine.understand(request, crops=[make_crop()])
        before = understander.calls
        engine.understand(request, crops=[make_crop()])
        assert understander.calls == before

    def test_a_cache_hit_gets_a_fresh_evidence_id(self, engine) -> None:
        """Two results are two events even when the answer is the same, and each
        needs its own evidence record with its own decision path."""
        request = make_request()
        first = engine.understand(request, crops=[make_crop()])
        second = engine.understand(request, crops=[make_crop()])
        assert first.evidence.evidence_id != second.evidence.evidence_id
        assert second.evidence.took(UnderstandingStep.CACHE_HIT)

    def test_a_different_crop_misses(self, engine) -> None:
        engine.understand(make_request(crop_id="crop-1"), crops=[make_crop()])
        result = engine.understand(
            make_request(request_id="req-2", crop_id="crop-2"),
            crops=[make_crop(crop_id="crop-2")],
        )
        assert not result.cache_hit

    def test_a_failure_is_not_cached(self, engine) -> None:
        adapter = scripted(
            ScriptedAnswer(raise_unavailable=True),
            ScriptedAnswer(raise_unavailable=True),
            answer_posture(),
            producible=(POSTURE,),
        )
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[adapter],
        )
        first = e.understand(make_request(), crops=[make_crop()])
        assert first.outcome is UnderstandingOutcome.UNAVAILABLE
        second = e.understand(make_request(), crops=[make_crop()])
        assert not second.cache_hit, (
            "caching a transient failure would extend a one-second blip for the "
            "life of the entry"
        )


class TestCostEstimation:
    def test_cost_is_estimable_before_invocation(self, engine) -> None:
        """Obligation U7 and §M9's *"governed by M8 rather than by itself"*."""
        estimate = engine.estimate_cost((POSTURE,))
        assert estimate.cost_units > 0
        assert estimate.fully_covered

    def test_an_uncoverable_request_estimates_zero_and_names_the_gap(
        self, engine
    ) -> None:
        estimate = engine.estimate_cost((HEIGHT,))
        assert estimate.cost_units == 0.0
        assert estimate.attributes_uncovered == (HEIGHT,)

    def test_estimating_costs_nothing(self, engine, understander) -> None:
        before = understander.calls
        engine.estimate_cost((POSTURE,))
        assert understander.calls == before, "estimation must not invoke a model"


class TestBatching:
    def test_every_request_id_appears_in_the_result(self, engine) -> None:
        requests = [make_request(request_id=f"req-{i}") for i in range(5)]
        crops = {r.request_id: [make_crop()] for r in requests}
        results = engine.understand_batch(requests, crops=crops)
        assert set(results) == {r.request_id for r in requests}

    def test_a_failing_request_still_appears(self, engine) -> None:
        adapter = scripted(
            answer_posture(),
            ScriptedAnswer(raise_unavailable=True),
            ScriptedAnswer(raise_unavailable=True),
            answer_posture(),
            producible=(POSTURE,),
        )
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[adapter],
        )
        requests = [make_request(request_id=f"req-{i}", crop_id=f"c-{i}") for i in range(3)]
        crops = {r.request_id: [make_crop(crop_id=str(r.crop_ids[0]))] for r in requests}
        results = e.understand_batch(requests, crops=crops)
        assert len(results) == 3

    def test_batches_group_by_model_and_prompt(self, engine) -> None:
        requests = [
            make_request(request_id="a", attributes=(POSTURE,)),
            make_request(request_id="b", attributes=(POSTURE,)),
        ]
        groups = engine.plan_batches(requests)
        assert len(groups) == 1, "identical questions batch together"

    def test_different_prompts_do_not_batch(self, engine) -> None:
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[scripted(answer_posture(), producible=(POSTURE, HEADWEAR, CARRYING))],
        )
        groups = e.plan_batches(
            [
                make_request(request_id="a", attributes=(POSTURE,)),
                make_request(request_id="b", attributes=(POSTURE, HEADWEAR, CARRYING)),
            ]
        )
        assert len(groups) == 2


class TestHealth:
    def test_no_understander_bound_reports_degraded(self, engine) -> None:
        from vision_os.core.model.health import HealthState

        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[],
        )
        health = e.health()
        assert health.state is HealthState.DEGRADED
        assert "attributes stop" in health.detail

    def test_a_healthy_engine_reports_healthy(self, engine) -> None:
        from vision_os.core.model.health import HealthState

        assert engine.health().state is HealthState.HEALTHY

    def test_health_reports_the_cache_hit_rate(self, engine) -> None:
        engine.understand(make_request(), crops=[make_crop()])
        engine.understand(make_request(), crops=[make_crop()])
        assert engine.health().metrics["cache_hit_rate"] > 0.0


class TestSchemaDriftAlarm:
    def test_sustained_ceiling_violations_alarm(self, engine, bus) -> None:
        """§M9: *"If the rate is sustained, alarm — this means a prompt has
        drifted beyond its declared schema."*"""
        subscription = bus.subscribe(["understanding.schema_drift_suspected"])
        adapter = LeakyUnderstander(
            fields={str(POSTURE): "standing", str(UNREGISTERED): True}
        )
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[adapter],
        )
        for index in range(6):
            e.understand(make_request(crop_id=f"c-{index}"), crops=[make_crop(crop_id=f"c-{index}")])
        assert subscription.drain(), "sustained drift must be alarmed"

    def test_an_occasional_violation_does_not_alarm(self, engine, bus) -> None:
        """One rejection is a model being creative. Alarming on it would train
        operators to ignore the alarm."""
        subscription = bus.subscribe(["understanding.schema_drift_suspected"])
        answers = [answer_posture() for _ in range(10)]
        answers[0] = ScriptedAnswer(fields={str(UNREGISTERED): True})
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[scripted(*answers, producible=(POSTURE,))],
        )
        for index in range(6):
            e.understand(make_request(crop_id=f"c-{index}"), crops=[make_crop(crop_id=f"c-{index}")])
        assert not subscription.drain()


class TestNoCropPixels:
    def test_a_crop_without_pixels_fails_explicitly(self, engine) -> None:
        result = engine.understand(
            make_request(), crops=[make_crop(with_pixels=False)]
        )
        assert result.outcome is UnderstandingOutcome.UNAVAILABLE
        assert result.attributes == ()

    def test_no_crops_at_all_fails_explicitly(self, engine) -> None:
        result = engine.understand(make_request(), crops=[])
        assert result.outcome is UnderstandingOutcome.UNAVAILABLE
