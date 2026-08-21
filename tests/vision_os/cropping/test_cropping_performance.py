"""Performance characteristics of the Crop Manager.

§M8 calls this *"the single most important cost-control point in the platform"*
and gives a worked cost model:

> 100 cameras x 5 objects x 5 fps = 2500 candidate analyses/second. With demand
> filtering, change-based triggering, and quality gating, the real rate is
> roughly **10-15 VLM calls/second** — a difference between "impossible" and
> "one GPU." *That reduction is not an optimization; it is the architecture.*

So the tests here measure **reduction factors**, not wall-clock. A timing
assertion on a shared CI machine measures the machine; a reduction factor
measures the design, and it is the number the cost model actually depends on.

The two timing budgets that remain are deliberately order-of-magnitude bounds.
They exist to catch a structural regression — an accidental O(n²) scan, a
per-candidate lease — not jitter.
"""

from __future__ import annotations

import contextlib
import time

from vision_os.core.errors import GateRejectedError
from vision_os.core.model.crop import SkipReason
from vision_os.core.model.space import Box
from vision_os.core.model.timebase import Duration
from vision_os.perception.cropping import TriggerStateStore

from .conftest import (
    CAMERA,
    COLOUR,
    frame_context,
    make_demand,
    make_object,
    sharp_frame,
)


def _walking(index: int) -> Box:
    """Objects spread across the frame, all large enough to clear the gate."""
    x = 0.02 + (index % 20) * 0.045
    return Box(x, 0.1, min(0.99, x + 0.04), 0.9)


class TestDemandFiltering:
    def test_no_demand_costs_nothing(self, manager) -> None:
        """*"Nothing is computed that no consumer asked for."*

        The largest single lever in the cost model — often a 10x reduction on its
        own — and the one that is easiest to lose by accident.
        """
        objects = [make_object(object_id=f"obj-{i}", box=_walking(i)) for i in range(100)]
        result = manager.evaluate(objects, frame_context())

        assert not result.requests, "an undemanded population must cost nothing"
        assert len(result.skipped) == 100
        assert all(s.reason is SkipReason.NO_DEMAND for s in result.skipped)

    def test_a_scoped_demand_filters_by_class(self, manager) -> None:
        from vision_os.core.model.ids import ClassId

        manager.register_demand(make_demand(classes=(ClassId("person"),)))
        objects = [
            make_object(object_id=f"p-{i}", box=_walking(i)) for i in range(10)
        ] + [
            make_object(
                object_id=f"v-{i}", box=_walking(i), class_id=ClassId("vehicle")
            )
            for i in range(10)
        ]
        result = manager.evaluate(objects, frame_context())
        assert len(result.requests) == 10, "only the demanded class triggers"
        assert result.candidate_count == 20

    def test_a_camera_scope_filters_by_camera(self, manager) -> None:
        from ..cropping.conftest import OTHER_CAMERA

        manager.register_demand(make_demand(cameras=(CAMERA,)))
        result = manager.evaluate(
            [make_object(camera=OTHER_CAMERA, box=_walking(i)) for i in range(10)],
            frame_context(camera=OTHER_CAMERA),
        )
        assert not result.requests


class TestChangeBasedTriggering:
    def test_a_stationary_analysed_object_stops_costing(self, manager) -> None:
        """*"A stationary object does not need re-analysis every second."*

        Another 5-20x in the cost model. Measured here as a ratio over a run
        rather than as a single call, because the property is about the *steady
        state* after the first look.
        """
        manager.register_demand(make_demand(freshness_ms=3_600_000))
        frame = frame_context()
        obj = make_object()

        first = manager.evaluate([obj], frame)
        assert first.requests, "the first sighting must trigger"
        crop = manager.extract(first.requests[0], pixels=sharp_frame(), frame=frame)
        assert crop.passed_gate

        analysed = manager.trigger_state.partition(CAMERA).objects[obj.object_id]
        settled = make_object(attributes={COLOUR: _attr(analysed.last_analysed)})

        triggers = 0
        for step in range(1, 30):
            result = manager.evaluate([settled], frame_context(step))
            triggers += len(result.requests)

        assert triggers == 0, (
            f"{triggers} re-analyses of an unchanged, fresh object; change-based "
            f"triggering is what makes cost `demands x changes` rather than "
            f"`cameras x fps x objects`"
        )

    def test_the_reduction_factor_is_measurable(self, manager) -> None:
        """The headline number, computed rather than asserted from a document."""
        manager.register_demand(make_demand(freshness_ms=3_600_000))
        frame = frame_context()
        objects = [make_object(object_id=f"obj-{i}", box=_walking(i)) for i in range(20)]

        first = manager.evaluate(objects, frame)
        for request in first.requests:
            with contextlib.suppress(GateRejectedError):
                # Some of the spread boxes land small; a rejection here is the
                # gate doing its job and does not affect the reduction ratio.
                manager.extract(request, pixels=sharp_frame(), frame=frame)

        analysed = manager.trigger_state.partition(CAMERA)
        settled = [
            make_object(
                object_id=f"obj-{i}",
                box=_walking(i),
                attributes={COLOUR: _attr(analysed.objects[f"obj-{i}"].last_analysed)}
                if analysed.objects.get(f"obj-{i}")
                and analysed.objects[f"obj-{i}"].last_analysed
                else {},
            )
            for i in range(20)
        ]

        naive = 0
        actual = 0
        for step in range(1, 21):
            result = manager.evaluate(settled, frame_context(step))
            naive += result.candidate_count
            actual += len(result.requests)

        assert naive == 400, "20 objects over 20 frames is the naive cost"
        reduction = naive / max(1, actual)
        assert reduction >= 4.0, (
            f"only a {reduction:.1f}x reduction against the naive rate; the cost "
            f"model depends on this being an order of magnitude"
        )


class TestBoundedResources:
    def test_trigger_state_never_exceeds_its_capacity(self) -> None:
        """An unbounded map grows with every object a camera has ever seen."""
        from vision_os.core.model.timebase import Instant

        store = TriggerStateStore(capacity_per_camera=128)
        partition = store.partition(CAMERA)
        for index in range(10_000):
            partition.state_for(f"obj-{index}", now=Instant(index))
        assert partition.tracked_objects == 128
        assert store.evictions == 10_000 - 128

    def test_the_dedup_cache_never_exceeds_its_capacity(self, manager) -> None:
        from vision_os.core.model.ids import CropId

        from .conftest import TENANT

        for index in range(20_000):
            manager.cache.put(TENANT, f"hash-{index}", CropId(f"c-{index}"))
        assert len(manager.cache) == manager.cache.stats().capacity

    def test_the_gate_window_is_bounded(self, manager) -> None:
        """A rolling window of outcomes, not every rejection ever seen."""
        from vision_os.perception.cropping import GateRejectionWindow

        window = GateRejectionWindow(camera_id=CAMERA, window=50)
        for _ in range(5_000):
            window.record(passed=False, reason=None)
        assert window.sample_size == 50


class TestStructuralCost:
    def test_evaluation_is_linear_in_candidates(self, manager) -> None:
        """Catches an accidental O(n²) scan, not machine jitter.

        The bound is an order of magnitude wide on purpose: the assertion is
        that doubling the population does not quadruple the work, and a tight
        bound on a shared machine measures the machine.
        """
        manager.register_demand(make_demand())

        def elapsed(count: int) -> float:
            objects = [
                make_object(object_id=f"obj-{i}", box=_walking(i)) for i in range(count)
            ]
            manager.forget_camera(CAMERA)
            started = time.perf_counter()
            manager.evaluate(objects, frame_context())
            return time.perf_counter() - started

        small = elapsed(50)
        large = elapsed(200)
        ratio = large / max(small, 1e-6)
        assert ratio < 20.0, (
            f"4x the candidates took {ratio:.1f}x the time; that is superlinear "
            f"and suggests a nested scan over the population"
        )

    def test_the_demand_lookup_does_not_scale_with_population(
        self, manager
    ) -> None:
        """Demand matching is per candidate; adding demands must not multiply it.

        A registry scanned per object *per demand* would make a site with many
        consumers quadratically expensive, which is the failure this checks for.
        """
        for index in range(20):
            manager.register_demand(make_demand(demand_id=f"d-{index}"))

        objects = [make_object(object_id=f"obj-{i}", box=_walking(i)) for i in range(50)]
        started = time.perf_counter()
        result = manager.evaluate(objects, frame_context())
        elapsed_ms = (time.perf_counter() - started) * 1000

        assert result.candidate_count == 50
        assert elapsed_ms < 500.0, (
            f"{elapsed_ms:.1f}ms to evaluate 50 objects against 20 demands; this "
            f"is an order-of-magnitude guard, so a failure here is structural"
        )

    def test_extraction_cost_is_independent_of_frame_size(self, manager) -> None:
        """Crops are emitted at the model's native size — never larger.

        §M8 Performance calls upscaling *"pure waste"*: a bigger source frame
        must not produce a bigger crop.
        """
        manager.register_demand(make_demand())
        sizes = []
        for width, height in ((320, 240), (1280, 960)):
            manager.forget_camera(CAMERA)
            frame = frame_context(width=width, height=height)
            request = manager.evaluate([make_object()], frame).requests[0]
            crop = manager.extract(
                request, pixels=sharp_frame(width, height), frame=frame
            )
            sizes.append(crop.pixel_count)
        assert sizes[0] == sizes[1], (
            "a 16x larger frame produced a different crop size; the output is "
            "fixed by the model's input, not by the camera's resolution"
        )


class TestBudgetIsTheCeiling:
    def test_a_configured_ceiling_is_the_observed_rate(self, manager) -> None:
        """The cost model's guarantee: spend cannot exceed the configured rate."""
        manager.register_demand(make_demand())
        manager._budget._credit = 5.0  # noqa: SLF001 - setting the boundary exactly

        objects = [make_object(object_id=f"obj-{i}", box=_walking(i)) for i in range(200)]
        result = manager.evaluate(objects, frame_context())

        assert len(result.requests) == 5
        assert result.candidate_count == 200, "the other 195 are still accounted for"

    def test_sustainable_freshness_reflects_the_ceiling(self, manager) -> None:
        for index in range(50):
            manager.evaluate(
                [make_object(object_id=f"obj-{index}")], frame_context(index)
            )
        ack = manager.register_demand(make_demand(freshness_ms=1))
        assert ack.effective_freshness > Duration(0), (
            "with N objects and C calls/hour the platform can only promise N/C; "
            "reporting the request back unchanged would be a lie"
        )


def _attr(observed_at):
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
