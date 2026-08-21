"""The P12/P13/P14 conformance kits.

A kit that only ever passes proves nothing. Every kit below is run twice: once
against the shipped adapter, which must pass, and once against an adapter
deliberately built to violate one obligation, which must fail *with that
obligation named*.

That second half is the part that matters. Without it a kit can silently stop
checking — a refactor that turns an assertion into a no-op looks identical to a
green build.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.cropping import (
    DefaultTriggerPolicy,
    HeuristicQualityEstimator,
    PaddedCropStrategy,
    TightCropStrategy,
)
from vision_os.conformance import (
    ALL_CROPPING_KITS,
    CROP_STRATEGY_KIT,
    QUALITY_ESTIMATOR_KIT,
    TRIGGER_POLICY_KIT,
)
from vision_os.conformance.kit import KitSection
from vision_os.core.model.crop import SkipReason, TriggerReason
from vision_os.core.model.detection import QualityGrades, QualityLevel
from vision_os.core.ports.cropping import CropPlan, TriggerDecision
from vision_os.kernel.plugins.manifest import PortCatalogue


class TestShippedAdaptersConform:
    def test_the_default_policy_passes(self) -> None:
        report = TRIGGER_POLICY_KIT.run(DefaultTriggerPolicy())
        assert report.passed, report.failures

    def test_the_heuristic_estimator_passes(self) -> None:
        report = QUALITY_ESTIMATOR_KIT.run(HeuristicQualityEstimator())
        assert report.passed, report.failures

    def test_both_crop_strategies_pass(self) -> None:
        for strategy in (TightCropStrategy(), PaddedCropStrategy()):
            report = CROP_STRATEGY_KIT.run(strategy)
            assert report.passed, f"{strategy.strategy_id}: {report.failures}"

    def test_every_kit_covers_shape_and_semantics(self) -> None:
        for kit in ALL_CROPPING_KITS:
            covered = kit.sections_covered()
            assert KitSection.SHAPE in covered
            assert KitSection.SEMANTICS in covered

    def test_kits_are_registered_against_the_right_ports(self) -> None:
        assert TRIGGER_POLICY_KIT.port_id == PortCatalogue.TRIGGER_POLICY
        assert QUALITY_ESTIMATOR_KIT.port_id == PortCatalogue.QUALITY_ESTIMATOR
        assert CROP_STRATEGY_KIT.port_id == PortCatalogue.CROP_STRATEGY

    def test_the_fast_subset_runs_at_load(self) -> None:
        """The Plugin Manager runs this before activating an adapter."""
        report = TRIGGER_POLICY_KIT.run(DefaultTriggerPolicy(), fast_only=True)
        assert report.passed
        assert report.fast_subset_only
        assert report.executed


# --- deliberately broken adapters --------------------------------------------- #


class _DroppingPolicy:
    """Returns fewer decisions than candidates. Violates G1."""

    policy_id = "trigger.dropping"

    def evaluate(self, candidates, *, now, demands):
        return [
            TriggerDecision(
                object_id=c.object_id, reason=TriggerReason.FIRST_SIGHT
            )
            for c in list(candidates)[:1]
        ]


class _ReorderingPolicy:
    """Returns decisions in a different order. Violates G3."""

    policy_id = "trigger.reordering"

    def evaluate(self, candidates, *, now, demands):
        return [
            TriggerDecision(object_id=c.object_id, reason=TriggerReason.FIRST_SIGHT)
            for c in reversed(list(candidates))
        ]


class _NonDeterministicPolicy:
    """Alternates its answer. Violates G3."""

    policy_id = "trigger.flaky"

    def __init__(self) -> None:
        self._flip = False

    def evaluate(self, candidates, *, now, demands):
        self._flip = not self._flip
        reason = TriggerReason.FIRST_SIGHT if self._flip else None
        return [
            TriggerDecision(
                object_id=c.object_id,
                reason=reason,
                skip=None if reason else SkipReason.NO_DEMAND,
            )
            for c in candidates
        ]


class _EagerPolicy:
    """Fires with no demand. Violates G4 — spends money nobody asked for."""

    policy_id = "trigger.eager"

    def evaluate(self, candidates, *, now, demands):
        return [
            TriggerDecision(object_id=c.object_id, reason=TriggerReason.FIRST_SIGHT)
            for c in candidates
        ]


class _PriorityInterpretingPolicy:
    """Raises on an unrecognised priority class. Violates G5."""

    policy_id = "trigger.opinionated"

    def evaluate(self, candidates, *, now, demands):
        resolver = demands[0] if demands else None
        decisions = []
        for candidate in candidates:
            wanted = (
                resolver(
                    camera_id=candidate.camera_id,
                    class_id=candidate.class_id,
                    region_ids=candidate.region_ids,
                )
                if resolver
                else {}
            )
            for _key, (_freshness, priority, _ids) in wanted.items():
                if priority not in ("urgent", "standard"):
                    raise ValueError(f"unknown priority {priority}")
            decisions.append(
                TriggerDecision(
                    object_id=candidate.object_id,
                    reason=TriggerReason.FIRST_SIGHT if wanted else None,
                    skip=None if wanted else SkipReason.NO_DEMAND,
                )
            )
        return decisions


class _ZeroingEstimator:
    """Reports 0.0 for grades it never measured. Violates Q2."""

    estimator_id = "quality.zeroing"

    def estimate(self, request):
        return QualityGrades(
            scale_pixels=request.box.height * request.source_height,
            truncation=0.0,
            occlusion=0.0,
            blur=0.0,
            crowding=0.0,
            overall=QualityLevel.GOOD,
        )


class _UngradedEstimator:
    """Never sets ``overall``. Violates Q3 — the gate has no input."""

    estimator_id = "quality.ungraded"

    def estimate(self, request):
        return QualityGrades(scale_pixels=100.0)


class _OptimisticEstimator:
    """Grades a sub-pixel object as usable. Violates Q5."""

    estimator_id = "quality.optimistic"

    def estimate(self, request):
        return QualityGrades(
            scale_pixels=request.box.height * max(1, request.source_height),
            overall=QualityLevel.EXCELLENT,
        )


class _EscapingStrategy:
    """Plans a box outside the frame. Violates C1 — reads past the buffer."""

    strategy_id = "crop.escaping"

    def plan(self, *, box, class_id, source_width, source_height, attributes=()):
        from vision_os.core.model.space import Box

        return CropPlan(
            source_box=box,
            padded_box=Box(0.0, 0.0, 2.0, 2.0),
            padding_applied=0.0,
            output_width=64,
            output_height=64,
        )


class _ScalingStrategy:
    """Sizes the output from the object. Violates C2 — upscaling is pure waste."""

    strategy_id = "crop.scaling"

    def plan(self, *, box, class_id, source_width, source_height, attributes=()):
        side = max(8, int(box.height * source_height))
        return CropPlan(
            source_box=box,
            padded_box=box.clamped_to_unit(),
            padding_applied=0.0,
            output_width=side,
            output_height=side,
        )


class _RaisingStrategy:
    """Raises on a degenerate box, hiding the outcome from the statistics."""

    strategy_id = "crop.raising"

    def plan(self, *, box, class_id, source_width, source_height, attributes=()):
        if box.area < 1e-6:
            raise ValueError("too small")
        return CropPlan(
            source_box=box,
            padded_box=box.clamped_to_unit(),
            padding_applied=0.0,
            output_width=64,
            output_height=64,
        )


class TestBrokenAdaptersAreCaught:
    """Every kit must actually fail something, or it is checking nothing."""

    @pytest.mark.parametrize(
        ("adapter", "obligation"),
        [
            (_DroppingPolicy(), "G1"),
            (_ReorderingPolicy(), "G3"),
            (_NonDeterministicPolicy(), "G3"),
            (_EagerPolicy(), "G4"),
            (_PriorityInterpretingPolicy(), "G5"),
        ],
    )
    def test_a_broken_trigger_policy_is_rejected(self, adapter, obligation) -> None:
        report = TRIGGER_POLICY_KIT.run(adapter)
        assert not report.passed, f"{adapter.policy_id} slipped through the kit"
        assert any(obligation in failure for failure in report.failures), (
            f"the kit failed {adapter.policy_id} but not for {obligation}: "
            f"{report.failures}"
        )

    @pytest.mark.parametrize(
        ("adapter", "obligation"),
        [
            (_ZeroingEstimator(), "Q2"),
            (_UngradedEstimator(), "Q3"),
            (_OptimisticEstimator(), "Q5"),
        ],
    )
    def test_a_broken_estimator_is_rejected(self, adapter, obligation) -> None:
        report = QUALITY_ESTIMATOR_KIT.run(adapter)
        assert not report.passed, f"{adapter.estimator_id} slipped through the kit"
        assert any(obligation in failure for failure in report.failures), (
            f"failed for the wrong reason: {report.failures}"
        )

    @pytest.mark.parametrize(
        ("adapter", "obligation"),
        [
            (_EscapingStrategy(), "C1"),
            (_ScalingStrategy(), "C2"),
        ],
    )
    def test_a_broken_strategy_is_rejected(self, adapter, obligation) -> None:
        report = CROP_STRATEGY_KIT.run(adapter)
        assert not report.passed, f"{adapter.strategy_id} slipped through the kit"
        assert any(obligation in failure for failure in report.failures), (
            f"failed for the wrong reason: {report.failures}"
        )

    def test_a_strategy_that_raises_on_degenerate_input_is_rejected(self) -> None:
        report = CROP_STRATEGY_KIT.run(_RaisingStrategy())
        assert not report.passed
        assert any("degenerate" in failure for failure in report.failures)

    def test_a_kit_failure_names_the_check(self) -> None:
        report = TRIGGER_POLICY_KIT.run(_DroppingPolicy())
        assert all("/" in failure for failure in report.failures), (
            "each failure must identify its section and check so an adapter "
            "author knows what to fix"
        )
