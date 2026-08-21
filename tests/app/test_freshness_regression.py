"""Freshness and trigger semantics survived the migration. **Mandatory regression.**

The platform's own suite covers trigger behaviour in depth — 29 tests in
`tests/vision_os/cropping/unit/test_triggers.py`, all migrated and passing. This
file does something narrower and complementary: it names the five reasons the
Phase 6 programme paid for and fails with an explanation if any of them is
removed or renamed.

The distinction matters because those tests assert *behaviour given a policy*,
while this one asserts *the vocabulary still exists at all*. A refactor that
renamed `FRESH_ENOUGH` would keep the platform's tests green by renaming them
too, and would silently end the only mechanism that stops the VLM being asked
the same question on every frame.

### Why this is not an end-to-end freshness test

Phase 1 binds no source and no understanding layer, so there is no frame to
observe and no VLM answer to reuse. Writing an end-to-end freshness test here
would mean building a second, application-owned pipeline to test against — which
is the duplication the whole programme has refused. The end-to-end path is
exercised by the platform suite today and returns in Phase 3 with a real source.

**No `validity_ms` is tuned here, and no performance is optimised.** This phase
proves migration correctness.
"""

from __future__ import annotations

import pytest


class TestTriggerVocabulary:
    """The ten reasons to spend a model call."""

    @pytest.mark.parametrize(
        "reason",
        [
            "FIRST_SIGHT",
            "ATTRIBUTE_MISSING",
            "ATTRIBUTE_STALE",
            "APPEARANCE_CHANGED",
            "LOW_CONFIDENCE",
            "IDENTITY_UNVERIFIED",
            "QUALITY_IMPROVED",
            "PERIODIC_REFRESH",
            "EXPLICIT_REQUEST",
            "LIFECYCLE_TRANSITION",
        ],
    )
    def test_trigger_reason_survived(self, reason: str) -> None:
        from vision_os.core.model.crop import TriggerReason

        assert hasattr(TriggerReason, reason), (
            f"TriggerReason.{reason} is gone. Every reason is a decision the "
            f"platform can explain to an operator asking why it spent a call."
        )

    def test_there_are_exactly_ten(self) -> None:
        from vision_os.core.model.crop import TriggerReason

        assert len(list(TriggerReason)) == 10


class TestSkipVocabulary:
    """The eight reasons to save one. `FRESH_ENOUGH` is the one that saves money."""

    @pytest.mark.parametrize(
        "reason",
        [
            "NO_DEMAND",
            "BUDGET_EXHAUSTED",
            "QUALITY_INSUFFICIENT",
            "FRESH_ENOUGH",
            "EVIDENCE_SUFFICIENT",
            "DEDUPLICATED",
            "PRIORITY_PREEMPTED",
            "FRAME_UNAVAILABLE",
        ],
    )
    def test_skip_reason_survived(self, reason: str) -> None:
        from vision_os.core.model.crop import SkipReason

        assert hasattr(SkipReason, reason)

    def test_there_are_exactly_eight(self) -> None:
        from vision_os.core.model.crop import SkipReason

        assert len(list(SkipReason)) == 8

    def test_fresh_enough_is_the_reuse_mechanism(self) -> None:
        """Phase 6's whole subject.

        Until 6.9 this had never fired: M7 and M9 held different attribute
        registries, so every write-back was rejected, nothing was ever fresh, and
        the platform re-asked the VLM for an answer it already held on every
        frame. Sharing the registry produced 522 of these on the first run.
        """
        from vision_os.core.model.crop import SkipReason

        assert SkipReason.FRESH_ENOUGH.value == "fresh_enough"


class TestStalenessArithmetic:
    """`AttributeStatus` decides freshness. Its arithmetic must not drift."""

    def test_a_recent_attribute_is_fresh(self) -> None:
        from vision_os.core.model.timebase import Duration, Instant
        from vision_os.core.model.ids import AttributeKey
        from vision_os.core.ports.cropping import AttributeStatus

        status = AttributeStatus(key=AttributeKey("head_covering"), observed_at=Instant(ns=0))
        now = Instant(ns=Duration.from_millis(30_000).ns)

        assert not status.is_stale(now, Duration.from_millis(120_000))

    def test_an_old_attribute_is_stale(self) -> None:
        from vision_os.core.model.timebase import Duration, Instant
        from vision_os.core.model.ids import AttributeKey
        from vision_os.core.ports.cropping import AttributeStatus

        status = AttributeStatus(key=AttributeKey("head_covering"), observed_at=Instant(ns=0))
        now = Instant(ns=Duration.from_millis(200_000).ns)

        assert status.is_stale(now, Duration.from_millis(120_000))

    def test_a_never_observed_attribute_has_no_age(self) -> None:
        """Absent is not old. An attribute nobody has looked at yet must trigger
        `ATTRIBUTE_MISSING`, not `ATTRIBUTE_STALE` — the two lead to different
        explanations and the operator-facing difference is real."""
        from vision_os.core.model.ids import AttributeKey
        from vision_os.core.model.timebase import Instant
        from vision_os.core.ports.cropping import AttributeStatus

        status = AttributeStatus(key=AttributeKey("head_covering"), observed_at=None)
        assert status.age(Instant(ns=1_000)) is None


class TestPolicyValidityUnchanged:
    """The configured freshness windows are what Phase 6 measured against.

    Not asserted as *correct* — asserted as *unchanged*. Phase 1 is a migration,
    and a validity window that shifted during the move would invalidate every
    comparison against the Phase 6 numbers without anyone noticing.
    """

    def test_head_and_hand_validity_windows(self) -> None:
        from app.vision.composition import build_attribute_registry, load_policies
        from vision_os.core.model.ids import AttributeKey

        registry = build_attribute_registry(
            load_policies("config/policies/kitchen-safety.example.json")
        )

        head = registry.require(AttributeKey("head_covering"))
        hand = registry.require(AttributeKey("hand_covering"))

        assert head.validity.ns == 120_000 * 1_000_000, "head_covering validity_ms must stay 120000"
        assert hand.validity.ns == 60_000 * 1_000_000, "hand_covering validity_ms must stay 60000"


class TestThreeValuedSemantics:
    """`NOT_VISIBLE` and `UNKNOWN` must never become `ABSENT`."""

    def test_every_declared_attribute_carries_a_refusal_value(self) -> None:
        """A model that could not see the body part must be able to say so.

        Without `not_visible` in the domain, a crop of somebody bending over a pot
        yields a decided answer about hands nobody looked at — and a confident
        violation generated from pixels nobody inspected.
        """
        from app.vision.composition import build_attribute_registry, load_policies

        registry = build_attribute_registry(
            load_policies("config/policies/kitchen-safety.example.json")
        )
        for key, schema in registry.schemas.items():
            assert "not_visible" in schema.domain, f"{key} lost its refusal value"

    def test_compliance_states_remain_distinct(self) -> None:
        from compliance import ComplianceState

        values = {s.name for s in ComplianceState}
        assert {"COMPLIANT", "VIOLATION", "UNKNOWN"} <= values

    def test_unknown_reasons_survived(self) -> None:
        """Why a rule could not decide is part of the answer, not an omission."""
        from compliance import UnknownReason

        assert len(list(UnknownReason)) >= 3
