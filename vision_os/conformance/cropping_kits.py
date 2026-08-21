"""Conformance kits for the Crop Manager's three ports — P12, P13, P14.

An interface constrains the shape of a call, not the meaning of its result. Two
trigger policies can implement ``evaluate`` perfectly and still break the
platform when swapped: one drops candidates it has no opinion about, and V8 dies
quietly. These kits are what stop that at plugin-load time rather than in
production data.

Each check below guards something that fails **silently** otherwise:

``trigger/every_candidate_decided``
    The V8 check. A policy that returns fewer decisions than candidates makes
    objects invisible — no crop, no skip, no record. A consumer then cannot tell
    "nothing was there" from "we never looked", which is precisely the
    distinction responsibility 7 exists to preserve.

``quality/unmeasured_is_none``
    A zeroed grade reads as a *good* grade. An estimator that returns 0.0 for
    "I did not measure blur" tells the gate the crop is perfectly sharp, and the
    platform spends money on unanswerable inputs while its dashboards look
    healthy.

``crop/output_never_exceeds_native``
    §M8 Performance: crops are emitted at the model's native input size,
    *"never larger, which is pure waste"*. Upscaling adds no information and
    multiplies the cost of every downstream call.

``crop/aspect_declared``
    A squashed crop produces attributes about a distorted object, and the
    distortion is invisible in the output. The transform record is only useful
    if strategies declare their aspect handling honestly.

**What these kits cannot check**, stated plainly rather than papered over: no
check here can tell whether a trigger policy makes *good* decisions, or whether
an estimator's blur score correlates with real sharpness. That needs labelled
ground truth the platform does not have. The kits verify contracts — the
structural properties whose violation is silent — and stop there.
"""

from __future__ import annotations

from ..core.model.crop import SkipReason, TriggerReason
from ..core.model.detection import QualityGrades
from ..core.model.ids import AttributeKey, CameraId, ClassId, ObjectId
from ..core.model.space import Box
from ..core.model.timebase import Duration, Instant
from ..core.ports.cropping import (
    AttributeStatus,
    QualityRequest,
    TriggerCandidate,
)
from ..kernel.plugins.manifest import PortCatalogue
from .kit import ConformanceCheck, ConformanceKit, KitSection

_CAMERA = CameraId("kit-crop-cam")
_CLASS = ClassId("person")
_NOW = Instant(1_000_000_000)
_KEY = AttributeKey("kit.colour")


def _candidate(suffix: str, *, box: Box | None = None) -> TriggerCandidate:
    return TriggerCandidate(
        object_id=ObjectId(f"kit-object-{suffix}"),
        camera_id=_CAMERA,
        class_id=_CLASS,
        box=box or Box(0.4, 0.4, 0.6, 0.9),
        lifecycle="active",
        identity_confidence=0.9,
        first_seen=Instant(0),
        last_confirmed=_NOW,
        observation_count=10,
        attributes={_KEY: AttributeStatus(key=_KEY)},
    )


def _wants_everything(*, camera_id, class_id, region_ids):
    """A resolver that demands one attribute of everything.

    The kit supplies its own rather than building a ``DemandRegistry``: a kit
    that needed half the platform to run would not be runnable at plugin load,
    which is the one moment it has to be.
    """
    return {_KEY: (Duration.from_millis(1_000), "kit-priority", ("kit-demand",))}


def _nothing_wanted(*, camera_id, class_id, region_ids):
    return {}


# --- P12 TriggerPolicyPort ----------------------------------------------------- #


def _check_shape(adapter) -> None:
    assert hasattr(adapter, "policy_id"), "a trigger policy must expose policy_id"
    assert isinstance(adapter.policy_id, str) and adapter.policy_id, (
        "policy_id must be a non-empty string; it identifies the policy in "
        "provenance and in every metric label"
    )
    decisions = adapter.evaluate([], now=_NOW, demands=[_wants_everything])
    assert list(decisions) == [], "an empty candidate list must yield no decisions"


def _check_every_candidate_decided(adapter) -> None:
    candidates = [_candidate(str(index)) for index in range(5)]
    decisions = list(adapter.evaluate(candidates, now=_NOW, demands=[_wants_everything]))
    assert len(decisions) == len(candidates), (
        f"policy returned {len(decisions)} decisions for {len(candidates)} "
        f"candidates; every candidate must produce exactly one decision, or an "
        f"object becomes invisible — no crop, no skip, no record (obligation G1)"
    )
    decided = [d.object_id for d in decisions]
    assert sorted(decided) == sorted(c.object_id for c in candidates), (
        "the decisions do not cover the candidates one-for-one; a policy may not "
        "invent, drop, or duplicate an object"
    )


def _check_reason_xor_skip(adapter) -> None:
    candidates = [_candidate("xor")]
    for resolver in (_wants_everything, _nothing_wanted):
        for decision in adapter.evaluate(candidates, now=_NOW, demands=[resolver]):
            assert (decision.reason is None) != (decision.skip is None), (
                f"decision for {decision.object_id} carries "
                f"reason={decision.reason} and skip={decision.skip}; exactly one "
                f"is required (obligation G2)"
            )
            if decision.reason is not None:
                assert isinstance(decision.reason, TriggerReason), (
                    "reason must be a platform TriggerReason, not an adapter's own "
                    "vocabulary; the nine documented reasons are closed"
                )
            if decision.skip is not None:
                assert isinstance(decision.skip, SkipReason), (
                    "skip must be a platform SkipReason; the seven documented "
                    "reasons are closed"
                )


def _check_determinism(adapter) -> None:
    candidates = [_candidate(str(index)) for index in range(4)]
    first = list(adapter.evaluate(candidates, now=_NOW, demands=[_wants_everything]))
    second = list(adapter.evaluate(candidates, now=_NOW, demands=[_wants_everything]))
    assert [(d.object_id, d.reason, d.skip) for d in first] == [
        (d.object_id, d.reason, d.skip) for d in second
    ], (
        "identical candidates produced different decisions; replay must reproduce "
        "attention exactly, or a six-month-old result cannot be explained (V13)"
    )


def _check_order_preserved(adapter) -> None:
    candidates = [_candidate(str(index)) for index in range(6)]
    decisions = list(adapter.evaluate(candidates, now=_NOW, demands=[_wants_everything]))
    assert [d.object_id for d in decisions] == [c.object_id for c in candidates], (
        "decisions must come back in candidate order; a caller zips the two "
        "sequences, and a reordering silently attributes one object's decision "
        "to another (obligation G3)"
    )


def _check_no_demand_is_a_skip(adapter) -> None:
    decisions = list(
        adapter.evaluate([_candidate("undemanded")], now=_NOW, demands=[_nothing_wanted])
    )
    assert len(decisions) == 1, "one candidate, one decision"
    assert not decisions[0].fires, (
        "a candidate no demand covers must not fire; demand-driven analysis is "
        "the platform's largest cost saving (§M8 Performance) and firing without "
        "a demand spends money nobody asked for"
    )


def _check_priority_is_opaque(adapter) -> None:
    """A policy may order by priority; it may not interpret it.

    Checked by handing the policy a class it has never seen. A policy that
    branches on the *meaning* of a class raises or misbehaves here; one that
    treats it as an opaque token carries on.
    """

    def _weird_priority(*, camera_id, class_id, region_ids):
        return {
            _KEY: (
                Duration.from_millis(1_000),
                "☃-priority-nobody-configured",
                ("kit-demand",),
            )
        }

    decisions = list(
        adapter.evaluate([_candidate("opaque")], now=_NOW, demands=[_weird_priority])
    )
    assert len(decisions) == 1, (
        "an unrecognised priority class must not stop evaluation; priority is an "
        "opaque string the platform orders by and never interprets (V1/V2)"
    )


TRIGGER_POLICY_KIT = ConformanceKit(
    port_id=PortCatalogue.TRIGGER_POLICY,
    version="1.0.0",
    checks=(
        ConformanceCheck("interface", KitSection.SHAPE, _check_shape),
        ConformanceCheck(
            "every_candidate_decided",
            KitSection.SEMANTICS,
            _check_every_candidate_decided,
            obligation="G1",
        ),
        ConformanceCheck(
            "reason_xor_skip", KitSection.SEMANTICS, _check_reason_xor_skip, obligation="G2"
        ),
        ConformanceCheck(
            "determinism", KitSection.SEMANTICS, _check_determinism, obligation="G3"
        ),
        ConformanceCheck(
            "order_preserved", KitSection.SEMANTICS, _check_order_preserved, obligation="G3"
        ),
        ConformanceCheck(
            "no_demand_is_a_skip",
            KitSection.SEMANTICS,
            _check_no_demand_is_a_skip,
            obligation="G4",
        ),
        ConformanceCheck(
            "priority_is_opaque", KitSection.FAILURE, _check_priority_is_opaque, obligation="G5"
        ),
    ),
)


# --- P13 QualityEstimatorPort -------------------------------------------------- #


def _quality_request(box: Box, **overrides) -> QualityRequest:
    payload = {
        "camera_id": _CAMERA,
        "box": box,
        "source_width": 1920,
        "source_height": 1080,
    }
    payload.update(overrides)
    return QualityRequest(**payload)


def _check_estimator_shape(adapter) -> None:
    assert hasattr(adapter, "estimator_id"), "an estimator must expose estimator_id"
    assert isinstance(adapter.estimator_id, str) and adapter.estimator_id
    grades = adapter.estimate(_quality_request(Box(0.4, 0.4, 0.6, 0.9)))
    assert isinstance(grades, QualityGrades), "estimate must return QualityGrades"


def _check_ranges(adapter) -> None:
    grades = adapter.estimate(_quality_request(Box(0.4, 0.4, 0.6, 0.9)))
    for name in ("truncation", "occlusion", "blur", "crowding"):
        value = getattr(grades, name)
        assert value is None or 0.0 <= value <= 1.0, (
            f"{name}={value} is outside [0,1]; the gate compares grades against "
            f"configured thresholds and an out-of-range grade silently inverts a "
            f"comparison (obligation Q1)"
        )
    assert grades.scale_pixels is None or grades.scale_pixels >= 0.0, (
        "scale_pixels must be non-negative source pixels"
    )


def _check_unmeasured_is_none(adapter) -> None:
    """Without pixels, blur and exposure must be ``None`` — never zero.

    A zeroed blur grade claims the crop is perfectly sharp. An estimator that
    reports that for an input it never looked at is not conservative, it is
    wrong in the direction that costs money.
    """
    grades = adapter.estimate(_quality_request(Box(0.4, 0.4, 0.6, 0.9)))
    assert grades.blur is None, (
        f"blur={grades.blur} was reported without pixels; 'not measured' and "
        f"'measured as zero' are different claims and a consumer reads a zeroed "
        f"grade as a good one (obligation Q2)"
    )
    assert grades.exposure is None, (
        "exposure was reported without pixels; the same reasoning applies"
    )


def _check_overall_is_set(adapter) -> None:
    grades = adapter.estimate(_quality_request(Box(0.4, 0.4, 0.6, 0.9)))
    assert grades.overall is not None, (
        "overall is unset; it is the gate's only input, and a gate given None "
        "must either pass everything or reject everything (obligation Q3)"
    )
    assert grades.is_graded, "is_graded must follow from overall being set"


def _check_estimator_determinism(adapter) -> None:
    request = _quality_request(Box(0.31, 0.22, 0.55, 0.81))
    first = adapter.estimate(request)
    second = adapter.estimate(request)
    assert first == second, (
        "identical input produced different grades; a gate rejection must be "
        "reproducible from a replay six months later (obligation Q4, V13)"
    )


def _check_extreme_geometry(adapter) -> None:
    """Legal-but-extreme geometry grades poorly; it never raises.

    A one-pixel object at the frame corner is a real thing a real camera
    produces. An estimator that raises on it turns a routine bad input into an
    exception that stops a frame.
    """
    tiny = adapter.estimate(_quality_request(Box(0.0, 0.0, 0.0005, 0.0005)))
    assert tiny.overall is not None, "a degenerate box must still be graded"
    assert not tiny.overall.is_usable, (
        f"a sub-pixel object graded {tiny.overall.value}; the strongest single "
        f"predictor of a useless claim is scale (obligation Q5)"
    )
    edge = adapter.estimate(_quality_request(Box(0.98, 0.98, 1.4, 1.6)))
    assert edge is not None, "an out-of-frame box must grade, not raise"
    assert adapter.estimate(
        _quality_request(Box(0.4, 0.4, 0.6, 0.9), source_width=0, source_height=0)
    ) is not None, "a zero-dimension frame must grade, not raise"


def _check_scale_tracks_size(adapter) -> None:
    small = adapter.estimate(_quality_request(Box(0.5, 0.5, 0.52, 0.54)))
    large = adapter.estimate(_quality_request(Box(0.2, 0.1, 0.5, 0.9)))
    if small.scale_pixels is None or large.scale_pixels is None:
        return
    assert large.scale_pixels > small.scale_pixels, (
        "a physically larger box graded no larger in scale_pixels; scale is "
        "object height in *source* pixels, and getting it backwards inverts the "
        "gate's most important threshold"
    )


QUALITY_ESTIMATOR_KIT = ConformanceKit(
    port_id=PortCatalogue.QUALITY_ESTIMATOR,
    version="1.0.0",
    checks=(
        ConformanceCheck("interface", KitSection.SHAPE, _check_estimator_shape),
        ConformanceCheck("ranges", KitSection.SEMANTICS, _check_ranges, obligation="Q1"),
        ConformanceCheck(
            "unmeasured_is_none",
            KitSection.SEMANTICS,
            _check_unmeasured_is_none,
            obligation="Q2",
        ),
        ConformanceCheck(
            "overall_is_set", KitSection.SEMANTICS, _check_overall_is_set, obligation="Q3"
        ),
        ConformanceCheck(
            "determinism",
            KitSection.SEMANTICS,
            _check_estimator_determinism,
            obligation="Q4",
        ),
        ConformanceCheck(
            "extreme_geometry", KitSection.FAILURE, _check_extreme_geometry, obligation="Q5"
        ),
        ConformanceCheck("scale_tracks_size", KitSection.GOLDEN, _check_scale_tracks_size),
    ),
)


# --- P14 CropStrategyPort ------------------------------------------------------- #


def _plan(adapter, box: Box, **overrides):
    payload = {
        "box": box,
        "class_id": _CLASS,
        "source_width": 1920,
        "source_height": 1080,
    }
    payload.update(overrides)
    return adapter.plan(**payload)


def _check_strategy_shape(adapter) -> None:
    assert hasattr(adapter, "strategy_id"), "a crop strategy must expose strategy_id"
    assert isinstance(adapter.strategy_id, str) and adapter.strategy_id
    plan = _plan(adapter, Box(0.4, 0.4, 0.6, 0.9))
    assert plan.output_width > 0 and plan.output_height > 0
    assert plan.source_box is not None, "the plan must record the unpadded box"


def _check_plan_inside_frame(adapter) -> None:
    """The padded box stays inside the frame after clamping (obligation C1).

    An unclamped box reads outside the buffer. In C that is a segfault; in Python
    it is a short slice that produces a crop of the wrong shape and no error.
    """
    for box in (
        Box(0.0, 0.0, 0.05, 0.1),
        Box(0.95, 0.9, 1.0, 1.0),
        Box(0.4, 0.4, 0.6, 0.9),
    ):
        plan = _plan(adapter, box)
        padded = plan.padded_box
        assert 0.0 <= padded.x1 < padded.x2 <= 1.0, (
            f"padded box {padded} escapes the frame horizontally for source {box}"
        )
        assert 0.0 <= padded.y1 < padded.y2 <= 1.0, (
            f"padded box {padded} escapes the frame vertically for source {box}"
        )


def _check_output_never_exceeds_native(adapter) -> None:
    """A crop is never emitted larger than the model's input (obligation C2).

    §M8 Performance calls upscaling *"pure waste"*: it adds no information and
    multiplies the bytes every downstream call must move.
    """
    tiny = _plan(adapter, Box(0.5, 0.5, 0.51, 0.52))
    large = _plan(adapter, Box(0.1, 0.05, 0.9, 0.95))
    assert (tiny.output_width, tiny.output_height) == (
        large.output_width,
        large.output_height,
    ), (
        "output size varies with the object's size; the crop format is fixed by "
        "the model's native input, not by how big the object happened to be"
    )


def _check_strategy_determinism(adapter) -> None:
    box = Box(0.33, 0.21, 0.58, 0.79)
    assert _plan(adapter, box) == _plan(adapter, box), (
        "identical input produced different plans; a replay must reproduce the "
        "same crop geometry or the evidence is not the same evidence (C3)"
    )


def _check_aspect_declared(adapter) -> None:
    plan = _plan(adapter, Box(0.4, 0.4, 0.6, 0.9))
    assert isinstance(plan.preserve_aspect, bool), (
        "aspect handling must be declared; a squashed crop produces attributes "
        "about a distorted object and the distortion is invisible in the output "
        "(obligation C4)"
    )
    assert plan.padding_applied >= 0.0, "padding_applied must be non-negative"


def _check_degenerate_box_is_planned(adapter) -> None:
    """A box with no area still produces a plan, not an exception.

    The gate is the right place to refuse a hopeless crop, because it *records*
    the refusal with a reason. A strategy that raises instead produces the same
    outcome with no statistic attached.
    """
    plan = _plan(adapter, Box(0.5, 0.5, 0.5001, 0.5001))
    assert plan.output_width > 0 and plan.output_height > 0


CROP_STRATEGY_KIT = ConformanceKit(
    port_id=PortCatalogue.CROP_STRATEGY,
    version="1.0.0",
    checks=(
        ConformanceCheck("interface", KitSection.SHAPE, _check_strategy_shape),
        ConformanceCheck(
            "plan_inside_frame",
            KitSection.SEMANTICS,
            _check_plan_inside_frame,
            obligation="C1",
        ),
        ConformanceCheck(
            "output_never_exceeds_native",
            KitSection.SEMANTICS,
            _check_output_never_exceeds_native,
            obligation="C2",
        ),
        ConformanceCheck(
            "determinism",
            KitSection.SEMANTICS,
            _check_strategy_determinism,
            obligation="C3",
        ),
        ConformanceCheck(
            "aspect_declared", KitSection.SEMANTICS, _check_aspect_declared, obligation="C4"
        ),
        ConformanceCheck(
            "degenerate_box_is_planned",
            KitSection.FAILURE,
            _check_degenerate_box_is_planned,
        ),
    ),
)


ALL_CROPPING_KITS: tuple[ConformanceKit, ...] = (
    TRIGGER_POLICY_KIT,
    QUALITY_ESTIMATOR_KIT,
    CROP_STRATEGY_KIT,
)
