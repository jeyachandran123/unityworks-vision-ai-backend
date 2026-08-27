"""The crop path's P33 seam — what it does, and what it must not do.

Three properties, in descending order of how badly a regression would hurt:

1. **Absent a producer, nothing changes.** A seam nobody has wired must not
   quietly restrict a running deployment.
2. **A refused region withholds the crop**, so no attribute is produced and the
   rule reaches UNKNOWN through the path it already has. The gate never invents
   a value; it declines to buy one.
3. **A broken producer is ignored, not fatal.** An observability signal is an
   improvement to the crop path, never a new way for it to fail (V9).
"""

from __future__ import annotations

import pytest

from vision_os.core.errors import GateRejectedError
from vision_os.core.model.crop import GateRejection
from vision_os.core.model.ids import AttributeKey
from vision_os.core.model.region_observability import RegionState, RegionVerdict
from vision_os.perception.cropping.engine import CropManager

from ..conftest import frame_context, make_request, sharp_frame

HEAD = AttributeKey("head_covering")
FACE = AttributeKey("face_covering")


class Producer:
    """Answers with one state for everything it is asked."""

    def __init__(self, state: RegionState, *, detail: str = "scripted") -> None:
        self.state = state
        self.detail = detail
        self.calls = 0

    def capabilities(self):
        from vision_os.core.ports.region_observability import (
            RegionObservabilityCapabilities,
        )

        return RegionObservabilityCapabilities(producer_id="scripted")

    def assess(self, request):
        self.calls += 1
        return tuple(
            RegionVerdict(
                attribute=key,
                state=self.state,
                producer_id="scripted",
                detail=self.detail if self.state.is_refusal else "",
            )
            for key in request.attributes
        )


class Exploding:
    def capabilities(self):
        raise RuntimeError("producer is broken")

    def assess(self, request):
        raise RuntimeError("producer is broken")


@pytest.fixture
def build(
    clock, metrics, bus, cropping_config, trigger_policy, estimator, strategy,
    extractor, cropping_provenance, demand_registry, budget, gate,
):
    def _build(observability=None) -> CropManager:
        return CropManager(
            clock=clock, metrics=metrics, events=bus, config=cropping_config,
            policy=trigger_policy, estimator=estimator, strategy=strategy,
            extractor=extractor, provenance=cropping_provenance,
            demands=demand_registry, budget=budget, gate=gate,
            observability=observability,
        )

    return _build


def _extract(manager: CropManager):
    context = frame_context()
    return manager.extract(
        make_request(attributes=(HEAD,)),
        pixels=sharp_frame(),
        frame=context,
    )


class TestTheDefault:
    def test_absent_a_producer_a_good_crop_is_still_taken(self, build) -> None:
        """The property that makes this change safe to merge."""
        crop = _extract(build())
        assert crop is not None

    def test_the_default_is_permissive_on_the_very_input_a_producer_would_refuse(
        self, build
    ) -> None:
        """The contrast that makes the default meaningful.

        Same request, same pixels: refused when a producer says the region is
        missing, taken when none is bound. Asserting only the second half would
        pass even if the seam were never reached.
        """
        with pytest.raises(GateRejectedError):
            _extract(build(Producer(RegionState.NOT_LOCATED, detail="no keypoints")))
        assert _extract(build()) is not None


class TestRefusal:
    def test_an_unlocatable_region_withholds_the_crop(self, build) -> None:
        with pytest.raises(GateRejectedError) as raised:
            _extract(build(Producer(RegionState.NOT_LOCATED, detail="no keypoints")))
        assert raised.value.context["reason"] == GateRejection.REGION_NOT_OBSERVABLE.value

    def test_a_low_confidence_region_also_withholds(self, build) -> None:
        """The producer said it was guessing. Acting on a guess is the failure
        this whole port exists to prevent."""
        with pytest.raises(GateRejectedError):
            _extract(build(Producer(RegionState.LOW_CONFIDENCE, detail="0.31 < 0.50")))

    def test_the_refusal_names_the_attribute_and_the_reason(self, build) -> None:
        with pytest.raises(GateRejectedError) as raised:
            _extract(build(Producer(RegionState.NOT_LOCATED, detail="head is behind a pot")))
        assert "head_covering" in raised.value.message
        assert "head is behind a pot" in raised.value.message

    def test_it_is_not_recorded_as_a_quality_problem(self, build) -> None:
        """Attribution matters operationally. 'Quality insufficient' sends someone
        to clean a lens; the head was simply bent over a pot."""
        with pytest.raises(GateRejectedError) as raised:
            _extract(build(Producer(RegionState.NOT_LOCATED, detail="x")))
        reason = raised.value.context["reason"]
        assert reason not in {
            GateRejection.TOO_BLURRY.value,
            GateRejection.TOO_SMALL.value,
            GateRejection.TOO_TRUNCATED.value,
        }


class TestPermission:
    def test_a_located_region_proceeds(self, build) -> None:
        producer = Producer(RegionState.LOCATED)
        assert _extract(build(producer)) is not None
        assert producer.calls == 1

    def test_an_unsupported_attribute_proceeds(self, build) -> None:
        """O2. Binding a producer that covers heads must not blind hands."""
        producer = Producer(RegionState.UNSUPPORTED)
        assert _extract(build(producer)) is not None


class TestFailureIsNotFatal:
    def test_a_broken_producer_does_not_break_the_crop_path(self, build) -> None:
        """V9 — degrade, never die.

        The failure mode this avoids is the ugly one: an observability producer
        added to reduce false alerts becoming the reason no crop is ever taken.
        """
        assert _extract(build(Exploding())) is not None
