"""The P15 and P16 conformance kits, run against broken adapters.

A kit that only ever passes proves nothing. Every kit runs twice: against the
shipped adapters, which must pass, and against adapters built to violate one
obligation, which must fail **with that obligation named**.

``_FabricatingUnderstander`` is the one that matters. 10_RELIABILITY §2.1 lists
the ``NO_FABRICATION_ON_FAILURE`` conformance test as a primary defence against
Byzantine failure, and U2 says why:

> *Fabricated output is indistinguishable from real output downstream.*

An adapter that returns a confident default on a corrupt crop passes every other
test anyone would think to write.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.understanding import (
    JsonCoercion,
    KeyValueCoercion,
    PassthroughCoercion,
    StaticAttributeHead,
)
from vision_os.conformance import (
    ALL_UNDERSTANDING_KITS,
    OUTPUT_COERCION_KIT,
    UNDERSTANDER_KIT,
)
from vision_os.conformance.kit import KitSection
from vision_os.core.model.ids import ModelId
from vision_os.core.model.understanding import CostEstimate, ModelMeta, Timing
from vision_os.core.ports.understanding import (
    CoercionResult,
    UnderstanderCapabilities,
    UnderstandingPortResponse,
)
from vision_os.kernel.plugins.manifest import PortCatalogue

from ..conftest import HEADWEAR, POSTURE, scripted


class TestShippedAdaptersConform:
    def test_the_scripted_understander_passes(self) -> None:
        report = UNDERSTANDER_KIT.run(scripted(producible=(POSTURE,)))
        assert report.passed, report.failures

    def test_a_specialized_head_passes(self) -> None:
        """The same kit, for a 2 MB classifier and a 7B VLM alike."""
        report = UNDERSTANDER_KIT.run(
            StaticAttributeHead(attribute=HEADWEAR, value=True)
        )
        assert report.passed, report.failures

    def test_every_coercion_strategy_passes(self) -> None:
        for strategy in (JsonCoercion(), KeyValueCoercion(), PassthroughCoercion()):
            report = OUTPUT_COERCION_KIT.run(strategy)
            assert report.passed, f"{strategy.strategy_id}: {report.failures}"

    def test_the_kits_cover_shape_semantics_and_failure(self) -> None:
        for kit in ALL_UNDERSTANDING_KITS:
            covered = kit.sections_covered()
            assert KitSection.SHAPE in covered
            assert KitSection.SEMANTICS in covered
            assert KitSection.FAILURE in covered, (
                "a port whose failure behaviour is untested is a port whose most "
                "dangerous property is untested"
            )

    def test_kits_are_registered_against_the_right_ports(self) -> None:
        assert UNDERSTANDER_KIT.port_id == PortCatalogue.UNDERSTANDER
        assert OUTPUT_COERCION_KIT.port_id == PortCatalogue.OUTPUT_COERCION

    def test_the_fast_subset_runs_at_load(self) -> None:
        report = UNDERSTANDER_KIT.run(scripted(producible=(POSTURE,)), fast_only=True)
        assert report.passed
        assert report.fast_subset_only


# --- deliberately broken adapters --------------------------------------------- #


def _capabilities(**overrides) -> UnderstanderCapabilities:
    payload = {
        "producible_attributes": (POSTURE,),
        "model_id": ModelId("broken"),
        "deterministic": True,
    }
    payload.update(overrides)
    return UnderstanderCapabilities(**payload)


def _meta() -> ModelMeta:
    return ModelMeta(
        model_id=ModelId("broken"), model_version="1.0.0", artifact_hash="hash"
    )


class _LeakingUnderstander:
    """Returns fields the schema did not declare. Violates U1."""

    adapter_id = "vlm.leaking"

    def capabilities(self):
        return _capabilities()

    def understand(self, request):
        return UnderstandingPortResponse(
            structured={"posture": "standing", "is_violation": True},
            raw_output=b"{}",
            model_meta=_meta(),
        )

    def understand_batch(self, requests):
        return {r.request_id: self.understand(r) for r in requests}

    def estimate_cost(self, request):
        return CostEstimate(cost_units=1.0)


class _FabricatingUnderstander:
    """**Answers confidently from an unreadable crop. Violates U2.**

    The adapter this whole kit exists to catch. It passes any test that only ever
    supplies valid input, and poisons the observation log forever.
    """

    adapter_id = "vlm.fabricating"

    def capabilities(self):
        return _capabilities()

    def understand(self, request):
        return UnderstandingPortResponse(
            structured={"posture": "standing"},
            raw_output=b'{"posture": "standing"}',
            model_meta=_meta(),
            timing=Timing(inference_ms=1.0, total_ms=1.0),
        )

    def understand_batch(self, requests):
        return {r.request_id: self.understand(r) for r in requests}

    def estimate_cost(self, request):
        return CostEstimate(cost_units=1.0)


class _AmnesiacUnderstander:
    """Loses raw output. Violates U3 — V4 becomes theoretical."""

    adapter_id = "vlm.amnesiac"

    def capabilities(self):
        return _capabilities()

    def understand(self, request):
        return UnderstandingPortResponse(
            structured={"posture": "standing"}, raw_output=b"", model_meta=_meta()
        )

    def understand_batch(self, requests):
        return {r.request_id: self.understand(r) for r in requests}

    def estimate_cost(self, request):
        return CostEstimate(cost_units=1.0)


class _StatefulUnderstander:
    """Answers differently on the second identical request. Violates U5.

    *"Or caching and replay both break."* — an adapter carrying conversation
    state makes every cache hit silently wrong.
    """

    adapter_id = "vlm.stateful"

    def __init__(self) -> None:
        self._seen = 0

    def capabilities(self):
        return _capabilities(deterministic=True)

    def understand(self, request):
        self._seen += 1
        value = "standing" if self._seen % 2 else "sitting"
        return UnderstandingPortResponse(
            structured={"posture": value}, raw_output=b"{}", model_meta=_meta()
        )

    def understand_batch(self, requests):
        return {r.request_id: self.understand(r) for r in requests}

    def estimate_cost(self, request):
        return CostEstimate(cost_units=1.0)


class _DroppingUnderstander:
    """Loses request ids in a batch. A dropped id is a lost answer in disguise."""

    adapter_id = "vlm.dropping"

    def capabilities(self):
        return _capabilities()

    def understand(self, request):
        return UnderstandingPortResponse(
            structured={"posture": "standing"}, raw_output=b"{}", model_meta=_meta()
        )

    def understand_batch(self, requests):
        return {requests[0].request_id: self.understand(requests[0])}

    def estimate_cost(self, request):
        return CostEstimate(cost_units=1.0)


class _MuteUnderstander:
    """Declares no producible attributes — can never be routed to."""

    adapter_id = "vlm.mute"

    def capabilities(self):
        return object()

    def understand(self, request):
        return UnderstandingPortResponse()

    def understand_batch(self, requests):
        return {}

    def estimate_cost(self, request):
        return CostEstimate(cost_units=0.0)


class _InventingCoercion:
    """Produces a field the text never mentioned. Violates X1."""

    strategy_id = "coercion.inventing"

    def coerce(self, raw, *, schema):
        return CoercionResult(
            parsed={str(field): "standing" for field in schema.fields},
            strategy_used=self.strategy_id,
        )


class _DiscardingCoercion:
    """Drops what it cannot parse. Violates X2 and 02_VOM §9.3."""

    strategy_id = "coercion.discarding"

    def coerce(self, raw, *, schema):
        import json

        try:
            parsed = json.loads(raw)
        except ValueError:
            return CoercionResult(strategy_used=self.strategy_id)
        if not isinstance(parsed, dict):
            return CoercionResult(strategy_used=self.strategy_id)
        declared = {str(f) for f in schema.fields}
        return CoercionResult(
            parsed={k: v for k, v in parsed.items() if k in declared},
            strategy_used=self.strategy_id,
        )


class _RaisingCoercion:
    """Raises on malformed text. Violates X4 — malformed is the normal case."""

    strategy_id = "coercion.raising"

    def coerce(self, raw, *, schema):
        import json

        return CoercionResult(parsed=json.loads(raw), strategy_used=self.strategy_id)


class _PromotingCoercion:
    """Returns undeclared fields as parsed values. Violates X1.

    Precisely how a model's judgment becomes a platform fact.
    """

    strategy_id = "coercion.promoting"

    def coerce(self, raw, *, schema):
        import json

        try:
            parsed = json.loads(raw)
        except ValueError:
            return CoercionResult(unparsed=raw, strategy_used=self.strategy_id)
        if not isinstance(parsed, dict):
            return CoercionResult(unparsed=raw, strategy_used=self.strategy_id)
        return CoercionResult(parsed=parsed, strategy_used=self.strategy_id)


class TestBrokenUnderstandersAreCaught:
    @pytest.mark.parametrize(
        ("adapter", "obligation"),
        [
            (_LeakingUnderstander(), "U1"),
            (_FabricatingUnderstander(), "U2"),
            (_AmnesiacUnderstander(), "U3"),
            (_StatefulUnderstander(), "U5"),
        ],
    )
    def test_an_obligation_violation_is_named(self, adapter, obligation) -> None:
        report = UNDERSTANDER_KIT.run(adapter)
        assert not report.passed, f"{adapter.adapter_id} slipped through the kit"
        assert any(obligation in failure for failure in report.failures), (
            f"the kit failed {adapter.adapter_id} but not for {obligation}: "
            f"{report.failures}"
        )

    def test_a_dropping_batch_is_rejected(self) -> None:
        report = UNDERSTANDER_KIT.run(_DroppingUnderstander())
        assert not report.passed
        assert any("request ids" in failure for failure in report.failures)

    def test_an_adapter_with_no_capabilities_is_rejected(self) -> None:
        report = UNDERSTANDER_KIT.run(_MuteUnderstander())
        assert not report.passed

    def test_the_fabrication_check_is_in_the_fast_subset(self) -> None:
        """It must run at plugin load, not only in a full suite.

        An adapter that fabricates is worse than one that fails, and the platform
        must refuse to activate it before a single real crop is processed.
        """
        report = UNDERSTANDER_KIT.run(_FabricatingUnderstander(), fast_only=True)
        assert not report.passed, (
            "fabrication-on-failure must be caught by the fast subset the Plugin "
            "Manager runs at load"
        )


class TestBrokenCoercionIsCaught:
    @pytest.mark.parametrize(
        ("adapter", "obligation"),
        [
            (_InventingCoercion(), "X1"),
            (_DiscardingCoercion(), "X2"),
            (_RaisingCoercion(), "X4"),
            (_PromotingCoercion(), "X1"),
        ],
    )
    def test_an_obligation_violation_is_named(self, adapter, obligation) -> None:
        report = OUTPUT_COERCION_KIT.run(adapter)
        assert not report.passed, f"{adapter.strategy_id} slipped through the kit"
        assert any(obligation in failure for failure in report.failures), (
            f"failed for the wrong reason: {report.failures}"
        )

    def test_a_failure_names_its_check(self) -> None:
        report = OUTPUT_COERCION_KIT.run(_InventingCoercion())
        assert all("/" in failure for failure in report.failures), (
            "each failure must identify its section and check so an adapter "
            "author knows what to fix"
        )
