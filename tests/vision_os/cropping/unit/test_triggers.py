"""The trigger policy — all nine reasons, and the skips.

§M8 lists nine ``TriggerReason``s and seven ``SkipReason``s. A policy that can
only express some of them is not a smaller policy, it is a policy that
misattributes: an object skipped for budget shows up as skipped for freshness,
and the deployment tunes the wrong thing.

The most important test in this file is
``test_every_candidate_produces_exactly_one_decision``. It is the executable form
of invariant V8 and responsibility 7 — a candidate that produces neither a
request nor a skip becomes invisible, and no downstream consumer can tell
"nothing was there" from "we never looked".
"""

from __future__ import annotations

import pytest

from vision_os.adapters.cropping import DefaultTriggerPolicy, ExplicitRequestPolicy
from vision_os.core.model.crop import SkipReason, TriggerReason
from vision_os.core.model.timebase import Duration, Instant
from vision_os.core.ports.cropping import AttributeStatus, TriggerDecision

from ..conftest import COLOUR, GARMENT, at, make_candidate

NOW = at(100)


def wants(freshness_ms: int = 30_000, priority: str = "standard", *attributes):
    """A resolver demanding the given attributes at a freshness."""
    keys = attributes or (COLOUR,)

    def resolver(*, camera_id, class_id, region_ids):
        return {
            key: (Duration.from_millis(freshness_ms), priority, ("demand-1",))
            for key in keys
        }

    return resolver


def wants_nothing(*, camera_id, class_id, region_ids):
    return {}


def observed(key, *, ago_ms: int, confidence: float = 0.9) -> AttributeStatus:
    return AttributeStatus(
        key=key,
        observed_at=Instant(NOW.ns - ago_ms * 1_000_000),
        confidence=confidence,
    )


class TestTheNineTriggerReasons:
    """Each reason must be reachable, and reachable for its own cause."""

    def test_first_sight(self, trigger_policy: DefaultTriggerPolicy) -> None:
        """A never-analysed object with a demand fires FIRST_SIGHT."""
        candidate = make_candidate(last_analysed=None)
        [decision] = trigger_policy.evaluate([candidate], now=NOW, demands=[wants()])
        assert decision.reason is TriggerReason.FIRST_SIGHT
        assert decision.attributes == (COLOUR,)

    def test_attribute_missing(self, trigger_policy: DefaultTriggerPolicy) -> None:
        """Analysed before, but a *new* demand names an attribute never computed.

        Distinct from FIRST_SIGHT: the object is known, the attribute is not.
        Conflating them would hide the moment a consumer's new demand started
        costing money.
        """
        candidate = make_candidate(
            last_analysed=at(90),
            attributes={
                COLOUR: observed(COLOUR, ago_ms=1_000),
                GARMENT: AttributeStatus(key=GARMENT),
            },
        )
        [decision] = trigger_policy.evaluate(
            [candidate], now=NOW, demands=[wants(30_000, "standard", COLOUR, GARMENT)]
        )
        assert decision.reason is TriggerReason.ATTRIBUTE_MISSING
        assert decision.attributes == (GARMENT,)

    def test_attribute_stale(self, trigger_policy: DefaultTriggerPolicy) -> None:
        candidate = make_candidate(
            last_analysed=at(50),
            attributes={COLOUR: observed(COLOUR, ago_ms=90_000)},
        )
        [decision] = trigger_policy.evaluate(
            [candidate], now=NOW, demands=[wants(30_000)]
        )
        assert decision.reason is TriggerReason.ATTRIBUTE_STALE

    def test_appearance_changed(self, trigger_policy: DefaultTriggerPolicy) -> None:
        candidate = make_candidate(
            last_analysed=at(99),
            attributes={COLOUR: observed(COLOUR, ago_ms=200)},
            appearance_delta=0.4,
        )
        [decision] = trigger_policy.evaluate(
            [candidate], now=NOW, demands=[wants(30_000)]
        )
        assert decision.reason is TriggerReason.APPEARANCE_CHANGED

    def test_low_confidence(self, trigger_policy: DefaultTriggerPolicy) -> None:
        candidate = make_candidate(
            last_analysed=at(99),
            attributes={COLOUR: observed(COLOUR, ago_ms=200, confidence=0.2)},
        )
        [decision] = trigger_policy.evaluate(
            [candidate], now=NOW, demands=[wants(30_000)]
        )
        assert decision.reason is TriggerReason.LOW_CONFIDENCE

    def test_quality_improved(self, trigger_policy: DefaultTriggerPolicy) -> None:
        """A previously rejected object retries as soon as conditions allow.

        Checked *before* staleness so recovery is prompt: an object that just
        became gradable should not wait out a freshness window it was never able
        to satisfy.
        """
        candidate = make_candidate(
            last_analysed=at(99),
            attributes={COLOUR: observed(COLOUR, ago_ms=200)},
            last_gate_rejection=True,
        )
        [decision] = trigger_policy.evaluate(
            [candidate], now=NOW, demands=[wants(30_000)]
        )
        assert decision.reason is TriggerReason.QUALITY_IMPROVED

    def test_periodic_refresh(self, trigger_policy: DefaultTriggerPolicy) -> None:
        """The cadence floor fires even when nothing else changed."""
        policy = DefaultTriggerPolicy(refresh_interval=Duration.from_millis(1_000))
        candidate = make_candidate(
            last_analysed=at(0),
            attributes={COLOUR: observed(COLOUR, ago_ms=100)},
        )
        [decision] = policy.evaluate([candidate], now=NOW, demands=[wants(300_000)])
        assert decision.reason is TriggerReason.PERIODIC_REFRESH

    def test_lifecycle_transition(self, trigger_policy: DefaultTriggerPolicy) -> None:
        candidate = make_candidate(
            last_analysed=at(99),
            attributes={COLOUR: observed(COLOUR, ago_ms=200)},
            lifecycle_changed=True,
        )
        [decision] = trigger_policy.evaluate(
            [candidate], now=NOW, demands=[wants(30_000)]
        )
        assert decision.reason is TriggerReason.LIFECYCLE_TRANSITION

    def test_region_entry_is_a_lifecycle_transition(
        self, trigger_policy: DefaultTriggerPolicy
    ) -> None:
        candidate = make_candidate(
            last_analysed=at(99),
            attributes={COLOUR: observed(COLOUR, ago_ms=200)},
            entered_region=True,
        )
        [decision] = trigger_policy.evaluate(
            [candidate], now=NOW, demands=[wants(30_000)]
        )
        assert decision.reason is TriggerReason.LIFECYCLE_TRANSITION

    def test_explicit_request(self, trigger_policy: DefaultTriggerPolicy) -> None:
        """The on-demand API path, composed rather than bypassing the policy."""
        candidate = make_candidate(
            last_analysed=at(99),
            attributes={COLOUR: observed(COLOUR, ago_ms=200)},
        )
        policy = ExplicitRequestPolicy(trigger_policy)
        policy.request(candidate.object_id)
        [decision] = policy.evaluate([candidate], now=NOW, demands=[wants(30_000)])
        assert decision.reason is TriggerReason.EXPLICIT_REQUEST

    def test_explicit_request_is_consumed(self, trigger_policy) -> None:
        """One request, one analysis. §M8: *bounded, rate-limited*.

        A request that persisted would turn a single API call into a standing
        subscription nobody registered and nobody is billed for.
        """
        candidate = make_candidate(
            last_analysed=at(99),
            attributes={COLOUR: observed(COLOUR, ago_ms=200)},
        )
        policy = ExplicitRequestPolicy(trigger_policy)
        policy.request(candidate.object_id)
        policy.evaluate([candidate], now=NOW, demands=[wants(30_000)])
        [second] = policy.evaluate([candidate], now=NOW, demands=[wants(30_000)])
        assert second.reason is not TriggerReason.EXPLICIT_REQUEST

    def test_every_reason_is_reachable_from_some_shipped_policy(self) -> None:
        """No documented reason is decorative.

        Reads the modules rather than the enum, so a reason defined but never
        emitted by any shipped policy is caught.

        Both policy modules are scanned because the platform ships two: the
        default policy answers *"is this worth a fresh look?"* and the
        verification policy answers *"is the detector's claim worth relying
        on?"*. ``IDENTITY_UNVERIFIED`` is deliberately unreachable from the
        default one — a policy that has never been told what the bound detector
        can name has no basis for doubting it, and emitting the reason anyway
        would make it meaningless.
        """
        import inspect

        from vision_os.adapters.cropping import triggers, verification

        source = inspect.getsource(triggers) + inspect.getsource(verification)
        unreachable = [
            reason.name
            for reason in TriggerReason
            if f"TriggerReason.{reason.name}" not in source
        ]
        assert not unreachable, (
            f"no shipped policy can emit {unreachable}; a reason the platform "
            f"cannot produce is a reason a consumer will never see"
        )

    def test_the_default_policy_never_emits_a_verification_reason(self) -> None:
        """The two policies stay separable.

        If the default policy ever learned to emit ``IDENTITY_UNVERIFIED``, a
        deployment with no verification rules would start spending model calls it
        never configured — and the composition seam would have quietly become a
        default.
        """
        import inspect

        from vision_os.adapters.cropping import triggers

        assert "IDENTITY_UNVERIFIED" not in inspect.getsource(triggers)


class TestTheSkipReasons:
    def test_no_demand(self, trigger_policy: DefaultTriggerPolicy) -> None:
        """The largest bucket in a healthy deployment (§M8 Performance)."""
        [decision] = trigger_policy.evaluate(
            [make_candidate()], now=NOW, demands=[wants_nothing]
        )
        assert decision.skip is SkipReason.NO_DEMAND
        assert not decision.fires

    def test_fresh_enough(self, trigger_policy: DefaultTriggerPolicy) -> None:
        """The success case for V7: everything is known and nothing is spent."""
        candidate = make_candidate(
            last_analysed=at(99),
            attributes={COLOUR: observed(COLOUR, ago_ms=200)},
        )
        [decision] = trigger_policy.evaluate(
            [candidate], now=NOW, demands=[wants(30_000)]
        )
        assert decision.skip is SkipReason.FRESH_ENOUGH

    def test_no_resolver_means_no_demand(self, trigger_policy) -> None:
        """A policy handed no demands must skip, never fire.

        Firing without a demand spends money nobody asked for — the exact cost
        that demand-driven analysis exists to eliminate.
        """
        [decision] = trigger_policy.evaluate([make_candidate()], now=NOW, demands=[])
        assert decision.skip is SkipReason.NO_DEMAND


class TestDecisionStructure:
    def test_every_candidate_produces_exactly_one_decision(
        self, trigger_policy: DefaultTriggerPolicy
    ) -> None:
        """Invariant V8, executable.

        A candidate that produces neither a request nor a skip is invisible: the
        consumer sees no attribute and cannot tell whether the platform looked.
        """
        candidates = [make_candidate(object_id=f"obj-{i}") for i in range(20)]
        decisions = trigger_policy.evaluate(candidates, now=NOW, demands=[wants()])
        assert len(decisions) == len(candidates)
        assert [d.object_id for d in decisions] == [c.object_id for c in candidates]

    def test_a_decision_cannot_carry_both(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            TriggerDecision(
                object_id=make_candidate().object_id,
                reason=TriggerReason.FIRST_SIGHT,
                skip=SkipReason.NO_DEMAND,
            )

    def test_a_decision_cannot_carry_neither(self) -> None:
        with pytest.raises(ValueError, match="invisible"):
            TriggerDecision(object_id=make_candidate().object_id)

    def test_decisions_are_deterministic(self, trigger_policy) -> None:
        candidates = [make_candidate(object_id=f"obj-{i}") for i in range(8)]
        first = trigger_policy.evaluate(candidates, now=NOW, demands=[wants()])
        second = trigger_policy.evaluate(candidates, now=NOW, demands=[wants()])
        assert [(d.object_id, d.reason) for d in first] == [
            (d.object_id, d.reason) for d in second
        ]

    def test_demand_ids_travel_for_cost_attribution(self, trigger_policy) -> None:
        [decision] = trigger_policy.evaluate(
            [make_candidate()], now=NOW, demands=[wants()]
        )
        assert decision.demand_ids == ("demand-1",)

    def test_priority_travels_but_is_never_interpreted(self, trigger_policy) -> None:
        """The policy reports the class; it never branches on what it means."""
        [decision] = trigger_policy.evaluate(
            [make_candidate()], now=NOW, demands=[wants(30_000, "☃-unrecognised")]
        )
        assert decision.priority_class == "☃-unrecognised"
        assert decision.fires, "an unknown priority must not suppress a trigger"


class TestPolicyValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"appearance_threshold": 1.5},
            {"appearance_threshold": -0.1},
            {"low_confidence": 2.0},
            {"refresh_interval": Duration(0)},
        ],
    )
    def test_invalid_configuration_is_refused(self, kwargs) -> None:
        with pytest.raises(ValueError):
            DefaultTriggerPolicy(**kwargs)

    def test_appearance_below_threshold_does_not_fire(self) -> None:
        policy = DefaultTriggerPolicy(appearance_threshold=0.5)
        candidate = make_candidate(
            last_analysed=at(99),
            attributes={COLOUR: observed(COLOUR, ago_ms=200)},
            appearance_delta=0.3,
        )
        [decision] = policy.evaluate([candidate], now=NOW, demands=[wants(30_000)])
        assert decision.skip is SkipReason.FRESH_ENOUGH

    def test_unmeasured_appearance_never_fires(self, trigger_policy) -> None:
        """``None`` is *not measured*, and must not be read as *changed*.

        Treating an absent measurement as a change would re-analyse every object
        on its first sighting after a restart, forever.
        """
        candidate = make_candidate(
            last_analysed=at(99),
            attributes={COLOUR: observed(COLOUR, ago_ms=200)},
            appearance_delta=None,
        )
        [decision] = trigger_policy.evaluate(
            [candidate], now=NOW, demands=[wants(30_000)]
        )
        assert not decision.fires


class TestStrictestFreshnessWins:
    def test_two_demands_take_the_stricter(self, demand_registry) -> None:
        """Satisfying the looser demand would under-serve the stricter one.

        This is the same honesty ``effective_freshness`` enforces at
        registration, applied per frame.
        """
        from ..conftest import CAMERA, PERSON, make_demand

        demand_registry.register(
            make_demand(freshness_ms=60_000, priority="background"), now=at(0)
        )
        demand_registry.register(
            make_demand(freshness_ms=5_000, priority="urgent"), now=at(0)
        )
        wanted = demand_registry.required_attributes(
            camera_id=CAMERA, class_id=PERSON, region_ids=()
        )
        freshness, _priority, ids = wanted[COLOUR]
        assert freshness.millis == pytest.approx(5_000)
        assert len(ids) == 2, "both demands must be credited for cost attribution"
