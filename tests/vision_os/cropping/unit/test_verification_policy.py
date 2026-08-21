"""The verification trigger policy — P12 composition, exercised as a policy.

These tests build a ``TriggerCandidate`` and assert one decision. No Crop
Manager, no frames, no engine: the policy is a pure function of a candidate and a
demand resolver, which is exactly what obligation G6 buys and what makes the
whole trust predicate exhaustively testable.

The numbers here are the ones from the architecture brief. ``0.454`` is not an
arbitrary low score — it is what a COCO detector actually returns for a pen it
believes is a toothbrush, comfortably above any threshold that still admits real
detections. A test suite that used ``0.05`` would prove nothing about the failure
this seam exists to catch.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.cropping import (
    DefaultTriggerPolicy,
    VerificationPolicy,
    VerificationRules,
)
from vision_os.core.errors import ConfigurationError
from vision_os.core.model.confidence import Confidence, ConfidenceSemantics
from vision_os.core.model.crop import SkipReason, TriggerReason
from vision_os.core.model.detection import QualityGrades
from vision_os.core.model.ids import AttributeKey, CameraId, ClassId, ObjectId
from vision_os.core.model.space import Box
from vision_os.core.model.timebase import Duration, Instant
from vision_os.core.ports.cropping import (
    CLOSED_SET,
    OPEN_VOCABULARY,
    AttributeStatus,
    LabelSpaceView,
    TriggerCandidate,
)

KIND = AttributeKey("visible_object_kind")
PPE = AttributeKey("headwear_present")
NOW = Instant(10_000_000_000)
FRESHNESS = Duration.from_millis(60_000)


# --- fixtures ------------------------------------------------------------------ #


def rules(**overrides) -> VerificationRules:
    """The representative document, with per-test overrides."""
    require = {
        "detector_label_space": [CLOSED_SET],
        "class_confidence_below": 0.65,
        "class_outside_native_vocabulary": True,
    }
    require.update(overrides.pop("require_when", {}))
    return VerificationRules.from_document(
        {
            "version": "1.0.0",
            "rules": [
                {
                    "rule_id": "identity.sensitive.v1",
                    "attributes": [str(KIND)],
                    "require_when": require,
                    "crop_must": overrides.pop("crop_must", {"min_scale_pixels": 48}),
                    **overrides,
                }
            ],
        }
    )


def candidate(
    *,
    class_id: str = "toothbrush",
    score: float | None = 0.454,
    in_vocabulary: bool | None = True,
    label_space: str = CLOSED_SET,
    scale: float | None = 120.0,
    truncation: float | None = None,
    occlusion: float | None = None,
    alternatives: tuple[tuple[str, float], ...] = (),
    attributes: dict | None = None,
    last_analysed: Instant | None = None,
) -> TriggerCandidate:
    return TriggerCandidate(
        object_id=ObjectId("obj-1"),
        camera_id=CameraId("cam-1"),
        class_id=ClassId(class_id),
        box=Box(0.1, 0.1, 0.3, 0.5),
        lifecycle="active",
        identity_confidence=0.92,
        first_seen=Instant(0),
        last_confirmed=NOW,
        observation_count=7,
        attributes=attributes or {},
        last_analysed=last_analysed,
        estimated_quality=QualityGrades(
            scale_pixels=scale, truncation=truncation, occlusion=occlusion
        ),
        class_confidence=(
            None
            if score is None
            else Confidence.uncalibrated(score, ConfidenceSemantics.CLASSIFICATION)
        ),
        class_alternatives=tuple(
            (ClassId(c), w) for c, w in alternatives
        ),
        label_space_kind=label_space,
        class_in_native_vocabulary=in_vocabulary,
    )


def resolver_for(*keys: AttributeKey):
    """A demand resolver wanting exactly these attributes."""

    def resolve(**_):
        return {key: (FRESHNESS, "", ("demand-1",)) for key in keys}

    return resolve


def decide(policy, cand, *keys: AttributeKey):
    return policy.evaluate(
        [cand], now=NOW, demands=[resolver_for(*(keys or (KIND,)))]
    )[0]


@pytest.fixture
def policy() -> VerificationPolicy:
    return VerificationPolicy(DefaultTriggerPolicy(), rules())


# --- 1. a trustworthy detection is not corroborated ----------------------------- #


class TestTrustedDetectionsAreNotVerified:
    """The subtractive half, and the one that keeps cost under control."""

    def test_high_confidence_in_vocabulary_is_withheld(self, policy) -> None:
        decision = decide(policy, candidate(class_id="person", score=0.94))

        assert not decision.fires
        assert decision.skip is SkipReason.EVIDENCE_SUFFICIENT
        assert "sufficient" in decision.detail

    def test_withheld_decision_still_names_its_attributes(self, policy) -> None:
        """Obligation G1's spirit: a withdrawn request is still accounted for."""
        decision = decide(policy, candidate(class_id="person", score=0.94))

        assert decision.attributes == (KIND,)
        assert decision.demand_ids == ("demand-1",)

    def test_an_open_vocabulary_detector_is_not_scrutinised(self, policy) -> None:
        """The rule names ``closed_set``; an open-vocabulary answer stands.

        Absence from an open-vocabulary detector's answer carries information a
        closed-set absence does not, so the same low score means something
        different and the rule does not apply.
        """
        decision = decide(
            policy,
            candidate(score=0.30, label_space=OPEN_VOCABULARY, in_vocabulary=False),
        )

        assert decision.skip is SkipReason.EVIDENCE_SUFFICIENT

    def test_an_undeclared_label_space_is_not_scrutinised(self, policy) -> None:
        """Undeclared must not read as "outside the vocabulary".

        A deployment that has not wired its detector capability would otherwise
        send every object to a model on a configuration mistake — the brute-force
        pipeline this seam exists to prevent.
        """
        decision = decide(
            policy, candidate(score=0.10, label_space="", in_vocabulary=None)
        )

        assert decision.skip is SkipReason.EVIDENCE_SUFFICIENT


# --- 2-3. ambiguity and capability ---------------------------------------------- #


class TestCorroborationIsRequested:
    def test_the_pen_at_0454_is_corroborated(self, policy) -> None:
        decision = decide(policy, candidate(score=0.454))

        assert decision.fires
        assert decision.reason is TriggerReason.IDENTITY_UNVERIFIED
        assert "0.454" in decision.detail

    def test_a_class_outside_the_detector_vocabulary_is_corroborated(
        self, policy
    ) -> None:
        decision = decide(
            policy, candidate(class_id="pen", score=0.94, in_vocabulary=False)
        )

        assert decision.reason is TriggerReason.IDENTITY_UNVERIFIED
        assert "outside the detector's declared vocabulary" in decision.detail

    def test_class_instability_is_corroborated(self) -> None:
        """A flapping object makes a weaker claim than its top score shows."""
        policy = VerificationPolicy(
            DefaultTriggerPolicy(),
            rules(require_when={"class_instability_above": 0.3}),
        )
        decision = decide(
            policy,
            candidate(score=0.94, alternatives=(("fork", 0.25), ("spoon", 0.20))),
        )

        assert decision.reason is TriggerReason.IDENTITY_UNVERIFIED

    def test_an_explicitly_named_class_is_always_corroborated(self) -> None:
        policy = VerificationPolicy(
            DefaultTriggerPolicy(), rules(require_when={"classes": ["toothbrush"]})
        )
        decision = decide(policy, candidate(score=0.99))

        assert decision.reason is TriggerReason.IDENTITY_UNVERIFIED

    def test_the_reason_is_not_conflated_with_low_confidence(self, policy) -> None:
        """``LOW_CONFIDENCE`` means a prior *claim* was weak. This is not that.

        The distinction matters because this value lands on
        ``Evidence.trigger_reason``, which is where "why did we look?" is
        answered six months later.
        """
        decision = decide(policy, candidate(score=0.454))

        assert decision.reason is not TriggerReason.LOW_CONFIDENCE
        assert decision.reason is TriggerReason.IDENTITY_UNVERIFIED


# --- 4-5. crop quality ----------------------------------------------------------- #


class TestCropQualityGatesCorroboration:
    def test_a_good_crop_is_eligible(self, policy) -> None:
        assert decide(policy, candidate(scale=200.0)).fires

    def test_a_small_crop_is_skipped_with_an_attributed_reason(self, policy) -> None:
        decision = decide(policy, candidate(scale=20.0))

        assert decision.skip is SkipReason.QUALITY_INSUFFICIENT
        assert "20px below the 48px floor" in decision.detail

    def test_a_truncated_crop_is_skipped(self) -> None:
        policy = VerificationPolicy(
            DefaultTriggerPolicy(), rules(crop_must={"max_truncation": 0.4})
        )
        decision = decide(policy, candidate(truncation=0.8))

        assert decision.skip is SkipReason.QUALITY_INSUFFICIENT
        assert "truncation" in decision.detail

    def test_an_unmeasured_grade_never_fails_a_requirement(self, policy) -> None:
        """"Not measured" and "measured as bad" are different claims (Q2).

        Refusing on an unmeasured grade would make an unwired quality estimator
        look like a site full of unusable cameras.
        """
        decision = decide(policy, candidate(scale=None))

        assert decision.fires

    def test_trust_is_assessed_before_quality(self, policy) -> None:
        """A trusted claim on a poor crop is ``EVIDENCE_SUFFICIENT``, not a
        quality rejection — blaming the camera for a call that was never going to
        be made would make the gate statistic unreadable."""
        decision = decide(policy, candidate(class_id="person", score=0.94, scale=5.0))

        assert decision.skip is SkipReason.EVIDENCE_SUFFICIENT


# --- 6-7. freshness and re-verification ------------------------------------------ #


class TestFreshnessGovernsReuse:
    def test_a_fresh_result_produces_no_second_request(self, policy) -> None:
        """Frames 100, 101 and 102 of the same object cost one call, not three.

        The inner policy answers ``FRESH_ENOUGH`` from the attribute's own
        freshness window, and the verification policy does not overrule it: a
        trust rule is not a licence to ignore a freshness SLA.
        """
        fresh = {
            KIND: AttributeStatus(
                key=KIND,
                observed_at=Instant(NOW.ns - 1_000_000_000),
                confidence=0.9,
                valid_until=Instant(NOW.ns + 60_000_000_000),
            )
        }
        decision = decide(
            policy, candidate(attributes=fresh, last_analysed=Instant(NOW.ns - 10))
        )

        assert not decision.fires
        assert decision.skip is SkipReason.FRESH_ENOUGH

    def test_a_stale_result_is_re_corroborated(self, policy) -> None:
        stale = {
            KIND: AttributeStatus(
                key=KIND,
                observed_at=Instant(NOW.ns - 600_000_000_000),
                confidence=0.9,
                valid_until=Instant(NOW.ns - 500_000_000_000),
            )
        }
        decision = decide(
            policy, candidate(attributes=stale, last_analysed=Instant(NOW.ns - 10))
        )

        assert decision.reason is TriggerReason.IDENTITY_UNVERIFIED


# --- composition guarantees ------------------------------------------------------ #


class TestCompositionIsTransparent:
    """Ordinary understanding must be untouched — the §7 requirement."""

    def test_an_ungoverned_attribute_passes_through_unchanged(self, policy) -> None:
        decision = decide(policy, candidate(class_id="person", score=0.94), PPE)
        inner = DefaultTriggerPolicy().evaluate(
            [candidate(class_id="person", score=0.94)],
            now=NOW,
            demands=[resolver_for(PPE)],
        )[0]

        assert decision == inner

    def test_a_mixed_demand_is_narrowed_not_withdrawn(self, policy) -> None:
        """Some attributes governed, some not: serve the rest.

        Withdrawing the whole decision would silently stop serving demands this
        policy has no opinion about.
        """
        decision = decide(policy, candidate(class_id="person", score=0.94), KIND, PPE)

        assert decision.fires
        assert decision.attributes == (PPE,)
        assert "withdrawn" in decision.detail

    def test_no_rules_means_the_inner_policy_verbatim(self) -> None:
        """§27's no-policy scenario, one layer up."""
        inner = DefaultTriggerPolicy()
        wrapped = VerificationPolicy(inner, VerificationRules())
        cand = candidate(score=0.454)

        assert wrapped.evaluate([cand], now=NOW, demands=[resolver_for(KIND)])[
            0
        ] == inner.evaluate([cand], now=NOW, demands=[resolver_for(KIND)])[0]

    def test_no_demand_is_never_overruled(self, policy) -> None:
        """A trust rule cannot manufacture work nobody asked for."""
        decision = policy.evaluate(
            [candidate(score=0.454)], now=NOW, demands=[lambda **_: {}]
        )[0]

        assert decision.skip is SkipReason.NO_DEMAND

    def test_one_decision_per_candidate_in_input_order(self, policy) -> None:
        """Obligation G1 and G3, preserved through composition."""
        cands = [
            candidate(class_id="person", score=0.94),
            candidate(score=0.454),
            candidate(class_id="pen", score=0.9, in_vocabulary=False),
        ]
        decisions = policy.evaluate(cands, now=NOW, demands=[resolver_for(KIND)])

        assert len(decisions) == len(cands)
        assert [d.object_id for d in decisions] == [c.object_id for c in cands]

    def test_evaluation_is_deterministic(self, policy) -> None:
        cand = candidate(score=0.454)
        first = policy.evaluate([cand], now=NOW, demands=[resolver_for(KIND)])
        second = policy.evaluate([cand], now=NOW, demands=[resolver_for(KIND)])

        assert first == second

    def test_policy_id_names_what_it_wraps(self, policy) -> None:
        assert policy.policy_id == "trigger.verification(trigger.default)"


# --- document validation --------------------------------------------------------- #


class TestDocumentIsValidatedAtLoad:
    """A malformed rule found on the first violation is one that has been
    silently not-firing since deployment."""

    def test_a_rule_governing_nothing_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="governs no attribute"):
            VerificationRules.from_document(
                {"rules": [{"rule_id": "r", "attributes": []}]}
            )

    def test_a_rule_with_no_condition_is_refused(self) -> None:
        """Such a rule suppresses every call to the attributes it governs."""
        with pytest.raises(ConfigurationError, match="declares no condition"):
            VerificationRules.from_document(
                {"rules": [{"rule_id": "r", "attributes": ["k"], "require_when": {}}]}
            )

    def test_an_unknown_label_space_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown detector label space"):
            VerificationRules.from_document(
                {
                    "rules": [
                        {
                            "rule_id": "r",
                            "attributes": ["k"],
                            "require_when": {"detector_label_space": ["fuzzy_set"]},
                        }
                    ]
                }
            )

    def test_an_out_of_range_threshold_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match=r"must be in \[0,1\]"):
            VerificationRules.from_document(
                {
                    "rules": [
                        {
                            "rule_id": "r",
                            "attributes": ["k"],
                            "require_when": {"class_confidence_below": 1.4},
                        }
                    ]
                }
            )

    def test_no_domain_vocabulary_reaches_the_policy(self) -> None:
        """Everything domain-shaped is data. This is the whole point of §19."""
        document = {
            "rules": [
                {
                    "rule_id": "anything.at.all",
                    "attributes": ["some_attribute_nobody_has_heard_of"],
                    "subject_classes": ["a_class_that_does_not_exist"],
                    "require_when": {"class_confidence_below": 0.5},
                }
            ]
        }
        loaded = VerificationRules.from_document(document)

        assert loaded.governed_attributes == frozenset(
            {AttributeKey("some_attribute_nobody_has_heard_of")}
        )


# --- capability view -------------------------------------------------------------- #


class TestLabelSpaceView:
    def test_undeclared_reports_cannot_tell(self) -> None:
        assert LabelSpaceView().covers(ClassId("pen")) is None

    def test_hierarchical_coverage(self) -> None:
        view = LabelSpaceView(
            kind=CLOSED_SET, producible_classes=frozenset({ClassId("person")})
        )

        assert view.covers(ClassId("person.child")) is True
        assert view.covers(ClassId("pen")) is False
