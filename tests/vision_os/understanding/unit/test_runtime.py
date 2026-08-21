"""The Understanding Runtime — the seam, the queue, and the firewall.

08_RUNTIME §5.2 gives the crop-to-understanding queue `drop_oldest` *"because
losing an enrichment is acceptable"*, and contrasts it with `Builder → State`'s
`block`. That asymmetry is, in the document's words, *"the whole philosophy of
the platform expressed as queue configuration"* — and
``test_the_queue_drops_oldest_under_pressure`` is where it becomes real.
"""

from __future__ import annotations

import asyncio

import pytest

from vision_os.core.model.crop import EvaluationResult
from vision_os.core.model.health import HealthState
from vision_os.core.model.understanding import UnderstandingOutcome
from vision_os.perception.understanding import (
    UnderstandingBatchReport,
    UnderstandingRuntime,
)

from ..conftest import (
    CAMERA,
    PERSON,
    POSTURE,
    SITE,
    TENANT,
    frame_ref,
    make_crop,
)


def crop_request(
    *, object_id: str = "obj-1", seq: int = 3, attributes=(POSTURE,)
):
    from vision_os.core.model.crop import CropRequest, TriggerReason
    from vision_os.core.model.ids import CropId, ObjectId
    from vision_os.core.model.space import Box

    assert CropId is not None
    return CropRequest(
        object_id=ObjectId(object_id),
        camera_id=CAMERA,
        frame_ref=frame_ref(seq),
        source_box=Box(0.4, 0.3, 0.55, 0.85),
        trigger_reason=TriggerReason.FIRST_SIGHT,
        tenant_id=TENANT,
        site_id=SITE,
        class_id=PERSON,
        required_attributes=tuple(str(a) for a in attributes),
    )


def evaluation(*requests, seq: int = 3) -> EvaluationResult:
    return EvaluationResult(
        camera_id=CAMERA, frame_ref=frame_ref(seq), requests=tuple(requests)
    )


@pytest.fixture
def runtime(clock, metrics, health, engine, understanding_config):
    return UnderstandingRuntime(
        clock=clock,
        metrics=metrics,
        health=health,
        engine=engine,
        config=understanding_config,
    )


class TestLifecycle:
    async def test_nothing_is_consumed_before_start(self, runtime) -> None:
        await runtime.on_crops(evaluation(crop_request()), [make_crop()])
        assert runtime.stats.frames_consumed == 0

    async def test_start_reports_healthy(self, runtime, health) -> None:
        await runtime.start()
        reported = {c.component_id: c for c in health.components()}
        assert reported["understanding_runtime"].state is HealthState.HEALTHY

    async def test_stop_drains_before_shutting_down(self, runtime) -> None:
        """A clean shutdown that discarded queued crops would make the last few
        seconds of a run depend on when the operator pressed stop."""
        await runtime.start()
        await runtime.on_crops(evaluation(crop_request()), [make_crop()])
        await runtime.stop()
        assert runtime.queue_depth == 0
        assert runtime.stats.results_produced >= 1

    async def test_a_disabled_config_consumes_nothing(
        self, clock, metrics, health, engine, understanding_config
    ) -> None:
        from dataclasses import replace

        runtime = UnderstandingRuntime(
            clock=clock,
            metrics=metrics,
            health=health,
            engine=engine,
            config=replace(understanding_config, enabled=False),
        )
        await runtime.start()
        await runtime.on_crops(evaluation(crop_request()), [make_crop()])
        assert runtime.stats.frames_consumed == 0


class TestTheFirewall:
    async def test_the_seam_never_raises(self, runtime) -> None:
        """V9. An understanding failure may not stop the Crop Manager, which may
        not stop the registry, which may not stop tracking."""

        class _Exploding:
            def understand_batch(self, *args, **kwargs):
                raise RuntimeError("boom")

            def plan_batches(self, *args, **kwargs):
                raise RuntimeError("boom")

            def health(self):
                raise RuntimeError("boom")

        await runtime.start()
        runtime._engine = _Exploding()  # noqa: SLF001 - injecting a fault
        await runtime.on_crops(evaluation(crop_request()), [make_crop()])
        assert runtime.stats.frames_failed == 1

    async def test_a_broken_sink_does_not_break_understanding(
        self, clock, metrics, health, engine, understanding_config
    ) -> None:
        def _bad_sink(results):
            raise RuntimeError("subscriber exploded")

        runtime = UnderstandingRuntime(
            clock=clock, metrics=metrics, health=health, engine=engine,
            config=understanding_config, sink=_bad_sink,
        )
        await runtime.start()
        await runtime.on_crops(evaluation(crop_request()), [make_crop()])
        assert runtime.stats.sink_failures == 1
        assert runtime.stats.frames_failed == 0

    async def test_an_empty_crop_list_is_not_an_error(self, runtime) -> None:
        """M8 correctly produced nothing, and its skips already explain why."""
        await runtime.start()
        await runtime.on_crops(evaluation(), [])
        assert runtime.stats.frames_consumed == 1
        assert runtime.stats.frames_failed == 0


class TestTheQueue:
    async def test_the_queue_drops_oldest_under_pressure(
        self, clock, metrics, health, engine, understanding_config
    ) -> None:
        """08_RUNTIME §5.2, and §5.1's *"NEVER silent"* — the drop is counted.

        Losing an enrichment is acceptable; losing it *quietly* is not.
        """
        runtime = UnderstandingRuntime(
            clock=clock, metrics=metrics, health=health, engine=engine,
            config=understanding_config, queue_capacity=2,
        )
        await runtime.start()
        requests = [crop_request(object_id=f"obj-{i}") for i in range(6)]
        crops = [make_crop(object_id=f"obj-{i}", crop_id=f"c-{i}") for i in range(6)]

        runtime._enqueue(evaluation(*requests), crops)  # noqa: SLF001
        assert runtime.queue_depth == 2, "the queue is bounded"
        assert runtime.stats.dropped_on_overflow == 4, "and every drop is counted"

    async def test_a_crop_without_a_matching_request_is_ignored(
        self, runtime
    ) -> None:
        """A crop M8 produced for a demand M9 was not told about has no attribute
        set, and guessing one would spend money on a question nobody asked."""
        await runtime.start()
        await runtime.on_crops(
            evaluation(crop_request(object_id="obj-1")),
            [make_crop(object_id="obj-other", crop_id="c-other")],
        )
        assert runtime.stats.crops_consumed == 0


class TestOutput:
    async def test_results_reach_the_sink(
        self, clock, metrics, health, engine, understanding_config
    ) -> None:
        recorded: list = []
        runtime = UnderstandingRuntime(
            clock=clock, metrics=metrics, health=health, engine=engine,
            config=understanding_config, sink=recorded.append,
        )
        await runtime.start()
        await runtime.on_crops(evaluation(crop_request()), [make_crop()])
        assert recorded
        assert recorded[0][0].outcome is UnderstandingOutcome.SUCCEEDED

    async def test_raw_output_is_stripped_at_the_seam(
        self, clock, metrics, health, engine, understanding_config
    ) -> None:
        """`01_LAYERED` §3.2 sizes this edge at ~3 KB: *"structured claims + raw
        output reference"*. The reference travels; the payload does not (V12)."""
        recorded: list = []
        runtime = UnderstandingRuntime(
            clock=clock, metrics=metrics, health=health, engine=engine,
            config=understanding_config, sink=recorded.append,
        )
        await runtime.start()
        await runtime.on_crops(evaluation(crop_request()), [make_crop()])
        result = recorded[0][0]
        assert result.raw_output is None
        assert result.evidence.raw_output_ref is not None, (
            "the reference must survive so the bytes are still findable"
        )

    async def test_the_request_inherits_the_crops_trigger_reason(
        self, clock, metrics, health, engine, understanding_config
    ) -> None:
        """The reason the platform looked was M8's decision; M9 never re-derives
        it."""
        from vision_os.core.model.crop import TriggerReason

        recorded: list = []
        runtime = UnderstandingRuntime(
            clock=clock, metrics=metrics, health=health, engine=engine,
            config=understanding_config, sink=recorded.append,
        )
        await runtime.start()
        await runtime.on_crops(
            evaluation(crop_request()),
            [make_crop(trigger=TriggerReason.APPEARANCE_CHANGED)],
        )
        assert (
            recorded[0][0].evidence.trigger_reason
            is TriggerReason.APPEARANCE_CHANGED
        )

    async def test_attributes_are_counted(self, runtime) -> None:
        await runtime.start()
        await runtime.on_crops(evaluation(crop_request()), [make_crop()])
        assert runtime.stats.attributes_produced == 1


class TestBatchReport:
    def test_it_counts_by_outcome(self, engine) -> None:
        from ..conftest import make_request

        results = [
            engine.understand(make_request(request_id=f"r-{i}", crop_id=f"c-{i}"), crops=[make_crop(crop_id=f"c-{i}")])
            for i in range(3)
        ]
        report = UnderstandingBatchReport.of(results)
        assert report.succeeded == 3
        assert report.attributes == 3

    def test_an_empty_report_is_coherent(self) -> None:
        report = UnderstandingBatchReport.of([])
        assert report.succeeded == 0
        assert report.attributes == 0


class TestConcurrentDelivery:
    async def test_concurrent_deliveries_do_not_corrupt_the_queue(
        self, runtime
    ) -> None:
        await runtime.start()
        await asyncio.gather(
            *(
                runtime.on_crops(
                    evaluation(crop_request(object_id=f"obj-{i}"), seq=i),
                    [make_crop(object_id=f"obj-{i}", crop_id=f"c-{i}", seq=i)],
                )
                for i in range(10)
            )
        )
        assert runtime.stats.frames_consumed == 10
        assert runtime.stats.frames_failed == 0
        assert runtime.queue_depth == 0

    async def test_every_delivered_crop_produces_a_result(self, runtime) -> None:
        """Nothing is silently lost between the seam and the engine."""
        await runtime.start()
        for index in range(5):
            await runtime.on_crops(
                evaluation(crop_request(object_id=f"obj-{index}"), seq=index),
                [make_crop(object_id=f"obj-{index}", crop_id=f"c-{index}", seq=index)],
            )
        assert runtime.stats.results_produced == runtime.stats.crops_consumed == 5
