"""A departed object must not reach the evaluator as a subject.

The chain this closes, measured on the running product before it was fixed:

    registry            27 objects       ← aged correctly
    Vision State        74 objects, ALL `active`, median last seen 279 s
    ComplianceDriver    99 subjects evaluated, 198 findings

`ComplianceDriver.snapshot` reads `exposure.api.query_state`, so the subject
list *is* the Vision State present set. The registry's scheduled horizon pass
never announced its transitions, so that present set never shrank and a chef
who left nine minutes ago was still being evaluated.

The gating itself was never the bug — `StateFilter`'s default already means
"present things". These tests pin that it stays that way, and that the
evaluator is fed from it rather than from the whole population.
"""

from __future__ import annotations

import pytest

from vision_os.core.model.api import StateFilter
from vision_os.core.model.visual_object import LifecycleState

#: Everything the platform can hold, split by whether a consumer asking
#: "what is here" should be told about it.
PRESENT = (
    LifecycleState.PROVISIONAL,
    LifecycleState.ACTIVE,
    LifecycleState.OCCLUDED,
)
DEPARTED = (
    LifecycleState.DORMANT,
    LifecycleState.DEPARTED,
    LifecycleState.EXPIRED,
    LifecycleState.MERGED_INTO,
)


class TestTheDefaultFilterMeansPresent:
    """`ComplianceDriver.snapshot` passes no filter, so the default *is* the gate."""

    def test_the_default_admits_only_present_lifecycles(self):
        default = StateFilter()
        assert set(default.lifecycle) <= set(PRESENT), (
            f"the default state filter admits {set(default.lifecycle) - set(PRESENT)}, "
            f"which a compliance pass would then evaluate as current subjects"
        )

    @pytest.mark.parametrize("lifecycle", DEPARTED)
    def test_a_departed_lifecycle_is_not_in_the_default(self, lifecycle):
        assert lifecycle not in StateFilter().lifecycle, (
            f"{lifecycle.value} objects would be evaluated as if the person were "
            f"still in the kitchen"
        )

    def test_provisional_is_excluded_as_well(self):
        """Stricter than "present", and deliberately so.

        The default is `(ACTIVE, OCCLUDED)`. An object the registry has not yet
        confirmed is not something a compliance decision may rest on, and
        Phase 6A.4 explicitly rejected widening this to populate a screen.
        Widening it to raise an incident would be worse.
        """
        assert LifecycleState.PROVISIONAL not in StateFilter().lifecycle
        assert set(StateFilter().lifecycle) == {
            LifecycleState.ACTIVE,
            LifecycleState.OCCLUDED,
        }


class TestTheDriverDoesNotWidenIt:
    def test_the_snapshot_passes_no_filter_of_its_own(self):
        """If the driver ever passes a filter, this suite stops guarding it."""
        import inspect

        from app.vision.compliance_driver import ComplianceDriver

        source = inspect.getsource(ComplianceDriver.snapshot)
        assert "filter_" not in source, (
            "ComplianceDriver.snapshot now passes its own StateFilter; the "
            "guarantee these tests give is about the default, so they must be "
            "updated to cover whatever it passes instead"
        )

    def test_the_evaluator_is_fed_the_snapshot_objects(self):
        """Subjects come from the narrowed query, not from the population."""
        import inspect

        from app.vision.compliance_driver import ComplianceDriver

        source = inspect.getsource(ComplianceDriver.evaluate)
        assert "snapshot.objects" in source, (
            "the evaluator is no longer fed from the state query, so the "
            "present-set gate no longer bounds what can raise an incident"
        )
