"""``kit.region_observability`` — the P33 conformance kit.

Two producers can implement ``RegionObservabilityPort`` perfectly and still break
the platform on swap. The checks that earn their keep here:

``one_verdict_per_attribute``
    Silence is the dangerous answer. A producer that returns verdicts only for
    what it assessed leaves the caller unable to distinguish *observable* from
    *unassessed* — and the safe reading of those two is opposite.

``never_expresses_a_covering``
    The Semantic Ceiling, enforced on this port. A producer that could say
    "uncovered" would be a second attribute source outside the registry's
    neutrality gate, which is the one thing this platform refuses everywhere
    else. The state enum offers no such value; this check proves an adapter has
    not smuggled one into ``detail``.

``degenerate_geometry_does_not_raise``
    The crop path must degrade, never die (V9). ``Box`` refuses zero area at
    construction, so the extreme a producer can actually be handed is a sliver a
    fraction of a pixel across — and that must be a refusal with a reason, not an
    exception that takes the frame down.

``refusal_names_its_reason``
    Same rule the quality gate keeps: *"the VLM never answers for far-away
    people"* must be a statistic with a name rather than a mystery.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.model.ids import AttributeKey, CameraId
from ..core.model.region_observability import RegionState
from ..core.model.space import Box
from ..core.ports.region_observability import RegionObservabilityRequest
from ..kernel.plugins.manifest import PortCatalogue
from .kit import ConformanceCheck, ConformanceKit, KitSection

_WIDTH = 64
_HEIGHT = 64

#: The smallest box the platform's own geometry type will construct.
#:
#: ``Box`` rejects zero area at construction, so a truly degenerate box cannot
#: reach a producer through the type system. This is the legal extreme instead —
#: a sliver a fraction of a pixel across, which is what "legal but extreme
#: geometry" (O5) actually means here.
_SLIVER = Box(0.5, 0.5, 0.5 + 1e-6, 0.5 + 1e-6)

#: Words an adapter must not use to smuggle a judgment through ``detail``.
_VERDICT_WORDS = (
    "uncovered",
    "bare",
    "no hairnet",
    "not wearing",
    "non-compliant",
    "violation",
    "compliant",
)


def _request(
    adapter, *, box: Box | None = None, attributes: Sequence[str] = ()
) -> RegionObservabilityRequest:
    keys = tuple(AttributeKey(k) for k in attributes) or tuple(
        adapter.capabilities().assessable_attributes
    ) or (AttributeKey("head_covering"),)
    return RegionObservabilityRequest(
        camera_id=CameraId("kit-cam"),
        box=box if box is not None else Box(0.25, 0.1, 0.6, 0.9),
        attributes=keys,
        source_width=_WIDTH,
        source_height=_HEIGHT,
        pixels=memoryview(bytes(_WIDTH * _HEIGHT * 3)),
        frame_key="kit-frame",
    )


# --- shape ---------------------------------------------------------------------- #


def _declares_capabilities(adapter) -> None:
    caps = adapter.capabilities()
    assert caps.producer_id, "a producer must name itself"


def _capabilities_are_stable(adapter) -> None:
    assert adapter.capabilities() == adapter.capabilities(), (
        "capabilities are captured at binding; a value that changes under a "
        "running route makes two identical requests take different paths"
    )


def _one_verdict_per_attribute(adapter) -> None:
    keys = (AttributeKey("head_covering"), AttributeKey("kit_unknown_attribute"))
    verdicts = adapter.assess(_request(adapter, attributes=[str(k) for k in keys]))
    assert len(verdicts) == len(keys), (
        f"expected {len(keys)} verdicts, got {len(verdicts)} — a caller must "
        f"never have to guess whether silence meant observable or unassessed (O1)"
    )
    assert tuple(v.attribute for v in verdicts) == keys, "verdicts must keep request order"


def _unassessable_is_unsupported(adapter) -> None:
    verdicts = adapter.assess(
        _request(adapter, attributes=["kit_definitely_not_assessable"])
    )
    assert verdicts[0].state is RegionState.UNSUPPORTED, (
        "an attribute outside declared coverage must be UNSUPPORTED, never a "
        "guess and never a refusal (O2)"
    )
    assert verdicts[0].state.is_observable, (
        "UNSUPPORTED must not restrict: binding a partial producer cannot be "
        "allowed to silently blind every attribute it does not cover"
    )


# --- semantics ------------------------------------------------------------------ #


def _never_expresses_a_covering(adapter) -> None:
    states = {s.value for s in RegionState}
    assert not (states & {"present", "absent", "covered", "uncovered"}), (
        "this port reports where a body part is, never what is on it (O3)"
    )
    for verdict in adapter.assess(_request(adapter)):
        lowered = verdict.detail.lower()
        for word in _VERDICT_WORDS:
            assert word not in lowered, (
                f"detail said {word!r}; a judgment reaching the platform through "
                f"this port would bypass the registry's neutrality gate (O3)"
            )


def _refusal_names_its_reason(adapter) -> None:
    for verdict in adapter.assess(_request(adapter, box=_SLIVER)):
        if verdict.state.is_refusal:
            assert verdict.detail, (
                "a refused region must name why; an unattributed refusal is a "
                "statistic nobody can act on"
            )


def _deterministic(adapter) -> None:
    first = adapter.assess(_request(adapter))
    second = adapter.assess(_request(adapter))
    assert [(v.attribute, v.state) for v in first] == [
        (v.attribute, v.state) for v in second
    ], "the same frame and box must produce the same verdict, or a refusal cannot be replayed (O4)"


# --- failure -------------------------------------------------------------------- #


def _degenerate_geometry_does_not_raise(adapter) -> None:
    for box in (_SLIVER, Box(0.0, 0.0, 1.0, 1.0)):
        verdicts = adapter.assess(_request(adapter, box=box))
        assert verdicts, "a degenerate box is a refusal with a reason, not an empty answer"


def _absent_pixels_are_not_a_guess(adapter) -> None:
    if not adapter.capabilities().requires_pixels:
        return
    request = RegionObservabilityRequest(
        camera_id=CameraId("kit-cam"),
        box=Box(0.25, 0.1, 0.6, 0.9),
        attributes=(AttributeKey("head_covering"),),
        source_width=_WIDTH,
        source_height=_HEIGHT,
        pixels=None,
    )
    verdicts = adapter.assess(request)
    assert verdicts[0].state is RegionState.UNSUPPORTED, (
        "a producer that needs pixels and was given none must say it could not "
        "assess, never refuse — refusing would convert a plumbing gap into a "
        "perception result"
    )


REGION_OBSERVABILITY_KIT = ConformanceKit(
    port_id=PortCatalogue.REGION_OBSERVABILITY,
    version="1.0.0",
    checks=(
        ConformanceCheck(
            "declares_capabilities", KitSection.SHAPE, _declares_capabilities, "O2"
        ),
        ConformanceCheck(
            "capabilities_are_stable", KitSection.SHAPE, _capabilities_are_stable, "O2"
        ),
        ConformanceCheck(
            "one_verdict_per_attribute", KitSection.SHAPE, _one_verdict_per_attribute, "O1"
        ),
        ConformanceCheck(
            "unassessable_is_unsupported", KitSection.SHAPE, _unassessable_is_unsupported, "O2"
        ),
        ConformanceCheck(
            "never_expresses_a_covering", KitSection.SEMANTICS, _never_expresses_a_covering, "O3"
        ),
        ConformanceCheck(
            "refusal_names_its_reason", KitSection.SEMANTICS, _refusal_names_its_reason
        ),
        ConformanceCheck("deterministic", KitSection.SEMANTICS, _deterministic, "O4"),
        ConformanceCheck(
            "degenerate_geometry_does_not_raise",
            KitSection.FAILURE,
            _degenerate_geometry_does_not_raise,
            "O5",
        ),
        ConformanceCheck(
            "absent_pixels_are_not_a_guess", KitSection.FAILURE, _absent_pixels_are_not_a_guess
        ),
    ),
)


def region_observability_kit_checks() -> Sequence[str]:
    """Check names, for the conformance report."""
    return tuple(check.qualified_name for check in REGION_OBSERVABILITY_KIT.checks)
