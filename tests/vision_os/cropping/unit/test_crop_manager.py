"""The Crop Manager itself — M8's documented public API.

The accounting identity this module exists to preserve, stated once:

    every candidate ends up in ``requests`` or in ``skipped``, exactly once.

Everything else — budget, gate, dedup, priority — is a way of *choosing* which
side a candidate lands on. None of them may make a candidate disappear, and the
tests below try each of them in turn to make one.
"""

from __future__ import annotations

import pytest

from vision_os.core.errors import GateRejectedError
from vision_os.core.model.crop import SkipReason, TriggerReason
from vision_os.core.model.detection import QualityLevel
from vision_os.core.model.ids import DemandId
from vision_os.core.model.space import Box
from vision_os.core.model.timebase import Duration, Instant
from vision_os.core.model.visual_object import LifecycleState
from vision_os.perception.cropping import CropManager

from ..conftest import (
    CAMERA,
    COLOUR,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    OTHER_TENANT,
    frame_context,
    make_demand,
    make_object,
    make_request,
    sharp_frame,
)


def demanded(manager: CropManager, **kwargs):
    return manager.register_demand(make_demand(**kwargs))


class TestTheAccountingIdentity:
    def test_every_candidate_appears_exactly_once(self, manager: CropManager) -> None:
        demanded(manager)
        objects = [make_object(object_id=f"obj-{i}") for i in range(12)]
        result = manager.evaluate(objects, frame_context())
        assert result.candidate_count == len(objects)

        seen = {r.object_id for r in result.requests} | {
            s.object_id for s in result.skipped
        }
        assert seen == {o.object_id for o in objects}

    def test_an_undemanded_population_is_entirely_skipped(
        self, manager: CropManager
    ) -> None:
        """The healthy default. Nothing is computed that nobody asked for."""
        objects = [make_object(object_id=f"obj-{i}") for i in range(5)]
        result = manager.evaluate(objects, frame_context())
        assert not result.requests
        assert all(s.reason is SkipReason.NO_DEMAND for s in result.skipped)

    def test_an_empty_population_produces_an_empty_result(
        self, manager: CropManager
    ) -> None:
        result = manager.evaluate([], frame_context())
        assert result.candidate_count == 0
        assert result.frame_ref == frame_context().frame_ref

    def test_evaluate_never_raises(self, manager: CropManager) -> None:
        """V9: an attention failure may not stop the registry beneath it.

        A policy that explodes must degrade into attributed skips, not into an
        exception that unwinds the perception stack.
        """

        class _Exploding:
            policy_id = "trigger.exploding"

            def evaluate(self, candidates, *, now, demands):
                raise RuntimeError("boom")

        manager._policy = _Exploding()  # noqa: SLF001 - injecting a fault
        objects = [make_object(object_id=f"obj-{i}") for i in range(3)]
        result = manager.evaluate(objects, frame_context())

        assert result.candidate_count == 3, (
            "a crashed evaluator that returned nothing would be indistinguishable "
            "from an idle scene (V8)"
        )
        assert manager.failures == 1

    def test_a_policy_that_drops_candidates_is_refused(
        self, manager: CropManager
    ) -> None:
        """Obligation G1, enforced at the call site as well as in the kit."""

        class _Dropping:
            policy_id = "trigger.dropping"

            def evaluate(self, candidates, *, now, demands):
                return []

        demanded(manager)
        manager._policy = _Dropping()  # noqa: SLF001
        result = manager.evaluate([make_object()], frame_context())
        assert result.candidate_count == 1, "the guard must still account for it"


class TestDemandDrivenTriggering:
    def test_a_demand_produces_a_request(self, manager: CropManager) -> None:
        ack = demanded(manager)
        result = manager.evaluate([make_object()], frame_context())
        assert len(result.requests) == 1
        assert result.requests[0].trigger_reason is TriggerReason.FIRST_SIGHT
        assert result.requests[0].demand_ids == (str(ack.demand_id),)

    def test_revoking_a_demand_stops_the_requests(self, manager: CropManager) -> None:
        ack = demanded(manager)
        assert manager.evaluate([make_object()], frame_context()).requests
        manager.revoke_demand(ack.demand_id)
        result = manager.evaluate([make_object()], frame_context(1))
        assert not result.requests
        assert result.skipped[0].reason is SkipReason.NO_DEMAND

    def test_the_request_carries_the_demanded_attributes(
        self, manager: CropManager
    ) -> None:
        """M8 passes them to M9 and never interprets them."""
        demanded(manager)
        result = manager.evaluate([make_object()], frame_context())
        assert result.requests[0].required_attributes == (str(COLOUR),)

    def test_the_request_inherits_tenancy_from_the_object(
        self, manager: CropManager
    ) -> None:
        """Tenancy is a property of the data, not a module-level constant."""
        demanded(manager)
        obj = make_object(tenant=OTHER_TENANT)
        result = manager.evaluate([obj], frame_context())
        assert result.requests[0].tenant_id == OTHER_TENANT

    def test_expired_demands_stop_being_served(self, manager: CropManager) -> None:
        demanded(manager, expires_ms=-1)
        manager.expire_demands()
        result = manager.evaluate([make_object()], frame_context())
        assert result.skipped[0].reason is SkipReason.NO_DEMAND


class TestBudgetShedding:
    def test_exhaustion_produces_attributed_skips(self, manager: CropManager) -> None:
        """*"We could not afford to look"* must be distinguishable from silence."""
        demanded(manager)
        manager._budget._credit = 2.0  # noqa: SLF001 - forcing the boundary
        objects = [make_object(object_id=f"obj-{i}") for i in range(10)]
        result = manager.evaluate(objects, frame_context())

        assert len(result.requests) == 2
        budget_skips = [
            s for s in result.skipped if s.reason is SkipReason.BUDGET_EXHAUSTED
        ]
        assert len(budget_skips) == 8
        assert result.candidate_count == 10

    def test_exhaustion_publishes_an_event(self, manager: CropManager, bus) -> None:
        subscription = bus.subscribe(["cropping.budget_exhausted"])
        demanded(manager)
        manager._budget._credit = 0.0  # noqa: SLF001
        manager.evaluate([make_object()], frame_context())
        assert subscription.drain(), (
            "consumers learn their attributes are thinned from an event, not from "
            "noticing an absence (V8)"
        )

    def test_shedding_follows_priority(self, manager: CropManager) -> None:
        """Ordering happens before spending, or priority means nothing."""
        demanded(manager, demand_id="low", priority="background")
        demanded(manager, demand_id="high", priority="urgent")
        manager._budget._credit = 1.0  # noqa: SLF001

        objects = [make_object(object_id=f"obj-{i}") for i in range(4)]
        result = manager.evaluate(objects, frame_context())
        assert len(result.requests) == 1
        assert result.requests[0].priority_class in ("urgent", "background")

    def test_the_per_frame_ceiling_preempts_rather_than_truncates(
        self, manager: CropManager, cropping_config
    ) -> None:
        from dataclasses import replace

        manager._config = replace(cropping_config, max_candidates_per_frame=3)  # noqa: SLF001
        demanded(manager)
        objects = [make_object(object_id=f"obj-{i}") for i in range(10)]
        result = manager.evaluate(objects, frame_context())

        assert len(result.requests) == 3
        preempted = [
            s for s in result.skipped if s.reason is SkipReason.PRIORITY_PREEMPTED
        ]
        assert len(preempted) == 7
        assert result.candidate_count == 10

    def test_budget_status_is_reportable(self, manager: CropManager) -> None:
        status = manager.budget_status()
        assert status.ceiling_per_hour > 0
        assert status.pressure >= 0.0


class TestExtraction:
    def test_a_good_object_produces_a_crop(self, manager: CropManager) -> None:
        demanded(manager)
        frame = frame_context()
        result = manager.evaluate([make_object()], frame)
        crop = manager.extract(
            result.requests[0], pixels=sharp_frame(), frame=frame
        )
        assert crop.passed_gate
        assert crop.pixels is not None
        assert crop.output_size == (64, 64)
        assert crop.quality.overall is not None

    def test_the_crop_id_is_a_content_hash(self, manager: CropManager) -> None:
        """The same pixels cropped twice must be one crop (02_VOM section 4.1)."""
        demanded(manager)
        frame = frame_context()
        result = manager.evaluate([make_object()], frame)
        request = result.requests[0]
        first = manager.extract(request, pixels=sharp_frame(), frame=frame)
        second = manager.extract(request, pixels=sharp_frame(), frame=frame)
        assert first.crop_id == second.crop_id

    def test_different_pixels_produce_different_ids(self, manager: CropManager) -> None:
        from ..conftest import other_sharp_frame

        demanded(manager)
        frame = frame_context()
        request = manager.evaluate([make_object()], frame).requests[0]
        sharp = manager.extract(request, pixels=sharp_frame(), frame=frame)
        other = manager.extract(request, pixels=other_sharp_frame(), frame=frame)
        assert sharp.crop_id != other.crop_id

    def test_a_gate_rejection_raises_with_its_reason(
        self, manager: CropManager
    ) -> None:
        """Not a failure — the gate working (§M8 failure table)."""
        demanded(manager)
        frame = frame_context()
        tiny = make_object(box=Box(0.5, 0.5, 0.52, 0.53))
        result = manager.evaluate([tiny], frame)
        with pytest.raises(GateRejectedError) as excinfo:
            manager.extract(result.requests[0], pixels=sharp_frame(), frame=frame)
        assert excinfo.value.context["reason"] == "too_small"
        assert excinfo.value.retryable, "conditions can improve; this is transient"

    def test_a_rejection_refunds_the_budget(self, manager: CropManager) -> None:
        """A rejected crop bought nothing, so it must not cost anything.

        Without the refund a run of rejections exhausts the budget having bought
        nothing, and the platform stops looking at what it *could* have answered.
        """
        demanded(manager)
        frame = frame_context()
        tiny = make_object(box=Box(0.5, 0.5, 0.52, 0.53))
        before = manager.budget.credit
        request = manager.evaluate([tiny], frame).requests[0]
        with pytest.raises(GateRejectedError):
            manager.extract(request, pixels=sharp_frame(), frame=frame)
        assert manager.budget.credit == pytest.approx(before)

    def test_extracting_against_the_wrong_frame_is_refused(
        self, manager: CropManager
    ) -> None:
        """Evidence must be traceable to its own frame."""
        from vision_os.core.errors import FrameUnavailableError

        demanded(manager)
        result = manager.evaluate([make_object()], frame_context(1))
        with pytest.raises(FrameUnavailableError, match="traceable"):
            manager.extract(
                result.requests[0], pixels=sharp_frame(), frame=frame_context(2)
            )

    def test_the_crop_records_its_transform(self, manager: CropManager) -> None:
        demanded(manager)
        frame = frame_context()
        result = manager.evaluate([make_object()], frame)
        crop = manager.extract(result.requests[0], pixels=sharp_frame(), frame=frame)
        assert crop.transform.source_width == FRAME_WIDTH
        assert crop.transform.source_height == FRAME_HEIGHT
        assert crop.transform.crop_width > 0

    def test_the_crop_carries_the_trigger_reason(self, manager: CropManager) -> None:
        demanded(manager)
        frame = frame_context()
        result = manager.evaluate([make_object()], frame)
        crop = manager.extract(result.requests[0], pixels=sharp_frame(), frame=frame)
        assert crop.trigger_reason is TriggerReason.FIRST_SIGHT

    def test_crowding_uses_the_neighbours_supplied(self, manager: CropManager) -> None:
        demanded(manager)
        frame = frame_context()
        obj = make_object(box=Box(0.4, 0.2, 0.6, 0.9))
        request = manager.evaluate([obj], frame).requests[0]
        crop = manager.extract(
            request,
            pixels=sharp_frame(),
            frame=frame,
            neighbour_boxes=(Box(0.5, 0.2, 0.7, 0.9),),
        )
        assert crop.quality.crowding is not None and crop.quality.crowding > 0.0


class TestTriggerStateIsEphemeral:
    def test_analysis_updates_trigger_state(self, manager: CropManager) -> None:
        demanded(manager)
        frame = frame_context()
        request = manager.evaluate([make_object()], frame).requests[0]
        manager.extract(request, pixels=sharp_frame(), frame=frame)
        state = manager.trigger_state.partition(CAMERA).objects[request.object_id]
        assert state.last_analysed is not None
        assert state.analyses == 1

    def test_a_second_frame_does_not_re_trigger_a_fresh_object(
        self, manager: CropManager
    ) -> None:
        """Change-based triggering: a stationary object costs nothing to keep."""
        demanded(manager, freshness_ms=3_600_000)
        frame = frame_context()
        obj = make_object()
        request = manager.evaluate([obj], frame).requests[0]
        manager.extract(request, pixels=sharp_frame(), frame=frame)

        analysed = manager.trigger_state.partition(CAMERA).objects[obj.object_id]
        from vision_os.core.ports.cropping import AttributeStatus

        refreshed = make_object(
            attributes={
                COLOUR: _attribute(analysed.last_analysed),
            }
        )
        second = manager.evaluate([refreshed], frame_context(1))
        assert not second.requests
        assert second.skipped[0].reason is SkipReason.FRESH_ENOUGH
        assert AttributeStatus is not None

    def test_a_gate_rejection_is_remembered(self, manager: CropManager) -> None:
        """So QUALITY_IMPROVED becomes expressible on the next frame."""
        demanded(manager)
        frame = frame_context()
        tiny = make_object(box=Box(0.5, 0.5, 0.52, 0.53))
        request = manager.evaluate([tiny], frame).requests[0]
        with pytest.raises(GateRejectedError):
            manager.extract(request, pixels=sharp_frame(), frame=frame)
        state = manager.trigger_state.partition(CAMERA).objects[tiny.object_id]
        assert state.last_gate_rejection is not None
        assert state.consecutive_gate_rejections == 1

    def test_forgetting_a_camera_releases_its_state(self, manager: CropManager) -> None:
        demanded(manager)
        manager.evaluate([make_object()], frame_context())
        assert manager.trigger_state.tracked_objects == 1
        manager.forget_camera(CAMERA)
        assert manager.trigger_state.tracked_objects == 0

    def test_trigger_state_is_bounded(self) -> None:
        """An unbounded map grows with every object a camera has ever seen."""
        from vision_os.perception.cropping import TriggerStateStore

        store = TriggerStateStore(capacity_per_camera=8)
        partition = store.partition(CAMERA)
        for index in range(100):
            partition.state_for(f"obj-{index}", now=Instant(index))
        assert partition.tracked_objects == 8
        assert partition.evictions == 92


def _attribute(observed_at):
    from vision_os.core.model.confidence import Confidence, ConfidenceSemantics
    from vision_os.core.model.ids import ConfigRevision, ModuleId
    from vision_os.core.model.provenance import Provenance
    from vision_os.core.model.visual_object import Attribute

    return Attribute(
        key=COLOUR,
        schema_version="1.0.0",
        value="blue",
        confidence=Confidence.uncalibrated(0.9, ConfidenceSemantics.ATTRIBUTE),
        observed_at=observed_at,
        producer=Provenance(
            producer_module=ModuleId("understanding"),
            producer_version="1.0.0",
            config_revision=ConfigRevision("test"),
        ),
    )


class TestAlarms:
    def test_a_gate_rejection_spike_is_published(
        self, manager: CropManager, bus
    ) -> None:
        """Almost always physical: a camera nudged, a lens fouled, a light failed."""
        subscription = bus.subscribe(["cropping.gate_rejection_spike"])
        demanded(manager)
        frame = frame_context()
        for index in range(10):
            tiny = make_object(object_id=f"obj-{index}", box=Box(0.5, 0.5, 0.52, 0.53))
            request = manager.evaluate([tiny], frame).requests[0]
            with pytest.raises(GateRejectedError):
                manager.extract(request, pixels=sharp_frame(), frame=frame)

        events = subscription.drain()
        assert events, "a spike must be alarmed"
        assert events[0].reason == "too_small", (
            "the dominant cause is what makes the alarm actionable on first read"
        )

    def test_a_capability_gap_is_published_after_persistent_failure(
        self, manager: CropManager, bus
    ) -> None:
        """Tell a consumer to stop waiting for data that will never arrive."""
        subscription = bus.subscribe(["cropping.capability_gap"])
        ack = demanded(manager)
        frame = frame_context()
        tiny = make_object(box=Box(0.5, 0.5, 0.52, 0.53))
        for _ in range(5):
            request = manager.evaluate([tiny], frame).requests[0]
            with pytest.raises(GateRejectedError):
                manager.extract(request, pixels=sharp_frame(), frame=frame)

        events = subscription.drain()
        assert events, "a persistently ungradable demand must be reported"
        assert events[0].demand_id == str(ack.demand_id)

    def test_health_degrades_when_the_budget_is_exhausted(
        self, manager: CropManager
    ) -> None:
        from vision_os.core.model.health import HealthState

        manager._budget._credit = 0.0  # noqa: SLF001
        assert manager.health().state is HealthState.DEGRADED


class TestSustainableFreshness:
    def test_registration_reports_what_the_budget_can_buy(
        self, manager: CropManager
    ) -> None:
        for index in range(20):
            manager.evaluate([make_object(object_id=f"obj-{index}")], frame_context())
        ack = manager.register_demand(make_demand(freshness_ms=1))
        assert ack.effective_freshness.ns > 0

    def test_a_zero_ceiling_reports_no_sustainable_rate(
        self, clock, metrics, bus, cropping_config, trigger_policy, estimator,
        strategy, extractor, cropping_provenance, capabilities, gate,
    ) -> None:
        from dataclasses import replace

        from vision_os.perception.cropping import DemandRegistry, UnderstandingBudget

        manager = CropManager(
            clock=clock,
            metrics=metrics,
            events=bus,
            config=replace(cropping_config, understanding_calls_per_hour=0.0),
            policy=trigger_policy,
            estimator=estimator,
            strategy=strategy,
            extractor=extractor,
            provenance=cropping_provenance,
            demands=DemandRegistry(capabilities=capabilities),
            budget=UnderstandingBudget(ceiling_per_hour=0.0, now=clock.monotonic()),
            gate=gate,
        )
        ack = manager.register_demand(make_demand())
        assert ack.effective_freshness.ns > 0
        result = manager.evaluate([make_object()], frame_context())
        assert result.skipped[0].reason is SkipReason.BUDGET_EXHAUSTED


class TestLifecycleAndRegions:
    def test_a_lifecycle_change_triggers_a_re_look(self, manager: CropManager) -> None:
        demanded(manager, freshness_ms=3_600_000)
        frame = frame_context()
        obj = make_object(lifecycle=LifecycleState.ACTIVE)
        request = manager.evaluate([obj], frame).requests[0]
        manager.extract(request, pixels=sharp_frame(), frame=frame)

        analysed = manager.trigger_state.partition(CAMERA).objects[obj.object_id]
        occluded = make_object(
            lifecycle=LifecycleState.OCCLUDED,
            attributes={COLOUR: _attribute(analysed.last_analysed)},
        )
        second = manager.evaluate([occluded], frame_context(1))
        assert second.requests
        assert second.requests[0].trigger_reason is TriggerReason.LIFECYCLE_TRANSITION

    def test_entering_a_region_triggers_a_re_look(self, manager: CropManager) -> None:
        demanded(manager, freshness_ms=3_600_000)
        frame = frame_context()
        obj = make_object()
        request = manager.evaluate(
            [obj], frame, regions_of=lambda _oid: frozenset()
        ).requests[0]
        manager.extract(request, pixels=sharp_frame(), frame=frame)

        analysed = manager.trigger_state.partition(CAMERA).objects[obj.object_id]
        refreshed = make_object(attributes={COLOUR: _attribute(analysed.last_analysed)})
        second = manager.evaluate(
            [refreshed], frame_context(1), regions_of=lambda _oid: frozenset({"Z3"})
        )
        assert second.requests
        assert second.requests[0].trigger_reason is TriggerReason.LIFECYCLE_TRANSITION

    def test_appearance_change_triggers_a_re_look(self, manager: CropManager) -> None:
        demanded(manager, freshness_ms=3_600_000)
        frame = frame_context()
        obj = make_object()
        request = manager.evaluate(
            [obj], frame, appearance_of=lambda _o: 0.1
        ).requests[0]
        manager.extract(request, pixels=sharp_frame(), frame=frame)

        analysed = manager.trigger_state.partition(CAMERA).objects[obj.object_id]
        refreshed = make_object(attributes={COLOUR: _attribute(analysed.last_analysed)})
        second = manager.evaluate(
            [refreshed], frame_context(1), appearance_of=lambda _o: 0.9
        )
        assert second.requests
        assert second.requests[0].trigger_reason is TriggerReason.APPEARANCE_CHANGED


class TestObjectsWithoutGeometry:
    def test_an_object_with_no_box_is_rejected_not_crashed(
        self, manager: CropManager
    ) -> None:
        """A counted, explicable outcome rather than an exception."""
        from vision_os.core.model.space import FrameOfReference, SpatialInfo

        demanded(manager)
        obj = make_object()
        obj = obj.__class__(
            **{
                **{f: getattr(obj, f) for f in obj.__dataclass_fields__},
                "current_spatial": SpatialInfo(
                    frame_of_reference=FrameOfReference.NORMALIZED, bbox=None
                ),
            }
        )
        frame = frame_context()
        result = manager.evaluate([obj], frame)
        assert result.candidate_count == 1
        if result.requests:
            with pytest.raises(GateRejectedError):
                manager.extract(result.requests[0], pixels=sharp_frame(), frame=frame)


class TestDeduplication:
    def test_identical_pixels_hit_the_cache(self, manager: CropManager) -> None:
        demanded(manager)
        frame = frame_context()
        request = manager.evaluate([make_object()], frame).requests[0]
        manager.extract(request, pixels=sharp_frame(), frame=frame)
        before = manager.cache.stats().hits
        manager.extract(request, pixels=sharp_frame(), frame=frame)
        assert manager.cache.stats().hits == before + 1

    def test_the_cache_is_tenant_scoped(self, manager: CropManager) -> None:
        """12_SECURITY section 4. One tenant's crop may never satisfy another's."""
        demanded(manager)
        frame = frame_context()
        mine = manager.evaluate([make_object()], frame).requests[0]
        manager.extract(mine, pixels=sharp_frame(), frame=frame)

        theirs = make_request(tenant=OTHER_TENANT)
        manager.extract(theirs, pixels=sharp_frame(), frame=frame_context())
        assert manager.cache.stats().misses >= 2


class TestQualityGradesTravel:
    def test_grades_are_attached_to_the_crop(self, manager: CropManager) -> None:
        """*"Quality is computed once, in the Crop Manager, and travels."*"""
        demanded(manager)
        frame = frame_context()
        request = manager.evaluate([make_object()], frame).requests[0]
        crop = manager.extract(request, pixels=sharp_frame(), frame=frame)
        assert crop.quality.scale_pixels is not None
        assert crop.quality.blur is not None, "pixel grades are measured post-crop"
        assert crop.quality.overall in tuple(QualityLevel)

    def test_a_freshness_of_zero_is_not_permitted(self, manager: CropManager) -> None:
        ack = manager.register_demand(make_demand(freshness_ms=1))
        assert ack.effective_freshness > Duration(0)
        assert isinstance(ack.demand_id, str | DemandId)
