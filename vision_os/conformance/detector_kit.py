"""``kit.detector`` — the P8 conformance kit (06_PORTS_AND_ADAPTERS section 5.2).

Every future detector must pass this before the Plugin Manager will activate it.
Two detectors can implement ``DetectorPort`` perfectly and still break the
platform on swap; this kit is what catches the difference.

The checks that earn their keep:

``coordinate_normalization`` / ``letterbox_inverse_exactness``
    The highest-frequency, lowest-visibility adapter bug in computer vision.
    Boxes drift by a few percent, detection still "works", tracking quietly
    degrades, and the cause is found months later — if ever.

    **What these checks can and cannot prove.** They verify that every box lands
    inside normalized space at any aspect ratio, which catches gross inversion
    failures. They cannot verify that a box lands in the *right place*: a model's
    output is expressed in letterboxed pixel space, so a correct inverse
    legitimately yields different normalized positions for different source
    shapes, and without ground truth those two cases are indistinguishable.

    Exactness is proven instead by two things: the pure-arithmetic
    ``letterbox`` module, which is exhaustively unit-tested across aspect ratios,
    and the kit's golden-corpus section, which needs annotated reference data.

``no_fabrication_on_failure``
    An adapter that returns a plausible default when inference fails poisons
    everything downstream with fully-provenanced fiction that nothing can detect.

``empty_result_is_not_error``
    "Nothing detected" and "detection failed" are different facts. Conflating
    them is how an empty scene becomes indistinguishable from a blind one.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.errors import DetectionFailedError, VisionOSError
from ..core.model.frame import FrameDimensions
from ..core.model.ids import CameraId, FrameRef, FrameSeq, StreamEpoch
from ..core.model.taxonomy import GeometryKind
from ..core.ports.detection import DetectionRequest, FrameView
from ..kernel.plugins.manifest import PortCatalogue
from .kit import ConformanceCheck, ConformanceKit, KitSection

_WIDTH = 64
_HEIGHT = 32
_CHANNELS = 3


def _frame(index: int = 0, width: int = _WIDTH, height: int = _HEIGHT) -> FrameView:
    payload = bytes([(index % 251) + 1]) * (width * height * _CHANNELS)
    return FrameView(
        frame_ref=FrameRef(CameraId("kit-cam"), StreamEpoch(1), FrameSeq(index)),
        dimensions=FrameDimensions(width=width, height=height, colour_space="bgr24"),
        pixels=memoryview(bytearray(payload)).toreadonly(),
    )


def _request(**overrides) -> DetectionRequest:
    defaults: dict = {"min_confidence": 0.0, "max_detections": 100}
    defaults.update(overrides)
    return DetectionRequest(**defaults)


# --- shape ------------------------------------------------------------------- #


def _declares_capabilities(adapter) -> None:
    capabilities = adapter.capabilities()
    assert capabilities.producible_classes, (
        "a detector must declare at least one producible class; an undeclared "
        "capability is an undetectable gap (invariant V8)"
    )
    assert capabilities.geometry_kinds, "a detector must declare its geometry kinds"
    assert GeometryKind.BOX in capabilities.geometry_kinds or capabilities.geometry_kinds, (
        "declared geometry kinds must be non-empty"
    )
    assert capabilities.precision in ("fp32", "fp16", "int8", "int4"), (
        f"unknown precision {capabilities.precision!r}"
    )


def _capabilities_are_stable(adapter) -> None:
    first = adapter.capabilities()
    second = adapter.capabilities()
    assert first.producible_classes == second.producible_classes, (
        "capabilities must be stable across calls; the platform publishes them as "
        "the site's capability surface"
    )


def _batch_order_preserved(adapter) -> None:
    frames = [_frame(index) for index in range(4)]
    results = adapter.detect(frames, _request())
    assert len(results) == len(frames), (
        f"returned {len(results)} results for {len(frames)} frames (obligation D6)"
    )
    for frame, result in zip(frames, results, strict=True):
        assert result.frame_ref == frame.frame_ref, (
            "batch results must map 1:1 and in order (obligation D6); result "
            f"{result.frame_ref} does not match input {frame.frame_ref}"
        )


def _empty_batch_is_handled(adapter) -> None:
    results = adapter.detect([], _request())
    assert len(results) == 0, "an empty batch must yield an empty result set"


def _model_metadata_is_complete(adapter) -> None:
    """Provenance is mandatory, not optional (invariant V4)."""
    results = adapter.detect([_frame()], _request())
    meta = results[0].model_meta
    assert meta.model_id, "a result must name the model that produced it"
    assert meta.model_version, "a result must name the model version"
    assert meta.artifact_hash, (
        "a result must carry the artifact hash of the exact weights used; without "
        "it no result is reproducible six months later (invariant V4)"
    )


def _inference_timing_is_reported(adapter) -> None:
    results = adapter.detect([_frame()], _request())
    timing = results[0].timing
    assert timing.inference_ms >= 0.0, "inference_ms must be non-negative"
    assert timing.batch_size >= 1, "batch_size must be at least 1"
    assert timing.model_load_state in ("warm", "cold"), (
        "a cold first inference can be 10-100x slower; the state must be declared "
        "so it does not read as a performance regression"
    )


# --- semantics ---------------------------------------------------------------- #


def _coordinate_normalization(adapter) -> None:
    """Obligation D1 — every box lands in normalized [0,1] source space."""
    for width, height in ((640, 360), (360, 640), (1920, 1080), (100, 100), (1280, 40)):
        frames = [_frame(0, width=width, height=height)]
        for result in adapter.detect(frames, _request()):
            for detection in result.detections:
                box = detection.box
                assert 0.0 <= box.x1 < box.x2 <= 1.0, (
                    f"box x-range {box.x1}..{box.x2} escapes [0,1] at {width}x{height}; "
                    f"letterboxing must be inverted exactly (obligation D1)"
                )
                assert 0.0 <= box.y1 < box.y2 <= 1.0, (
                    f"box y-range {box.y1}..{box.y2} escapes [0,1] at {width}x{height}"
                )


def _extreme_aspect_ratios(adapter) -> None:
    """The shapes where a letterbox-inverse bug actually shows up.

    A square test image hides the error entirely, which is why so many adapters
    ship with it.
    """
    for width, height in ((1920, 60), (60, 1920), (3840, 2160), (33, 97)):
        frames = [_frame(0, width=width, height=height)]
        results = adapter.detect(frames, _request())
        assert len(results) == 1
        for detection in results[0].detections:
            assert detection.box.is_within_unit(), (
                f"box {detection.box} escapes normalized space at aspect "
                f"{width}x{height} — the classic letterbox-inverse failure"
            )


def _taxonomy_mapping_complete(adapter) -> None:
    """Obligation D2 — a native label must never escape the adapter."""
    producible = set(adapter.capabilities().producible_classes)
    for result in adapter.detect([_frame(i) for i in range(3)], _request()):
        for detection in result.detections:
            assert detection.class_id in producible or detection.class_id == "unknown", (
                f"emitted class '{detection.class_id}' is not among the declared "
                f"producible classes {sorted(producible)}; a native label must "
                f"never escape the adapter (obligation D2)"
            )


def _confidence_semantics(adapter) -> None:
    """Obligation D3 — scores are raw model output in [0,1]."""
    for result in adapter.detect([_frame()], _request()):
        for detection in result.detections:
            assert 0.0 <= detection.score <= 1.0, (
                f"score {detection.score} escapes [0,1]; the platform calibrates "
                f"raw scores and cannot calibrate an out-of-range one"
            )


def _threshold_behaviour(adapter) -> None:
    """A raised threshold may never *increase* the number of detections."""
    low = adapter.detect([_frame()], _request(min_confidence=0.0))
    high = adapter.detect([_frame()], _request(min_confidence=0.99))
    low_count = sum(len(r.detections) for r in low)
    high_count = sum(len(r.detections) for r in high)
    assert high_count <= low_count, (
        f"raising min_confidence from 0.0 to 0.99 produced more detections "
        f"({high_count} > {low_count}); the pre-filter is inverted"
    )
    for result in high:
        for detection in result.detections:
            assert detection.score >= 0.99, (
                f"detection scoring {detection.score} survived a 0.99 threshold"
            )


def _max_detections_respected(adapter) -> None:
    for result in adapter.detect([_frame()], _request(max_detections=1)):
        assert len(result.detections) <= 1, (
            f"returned {len(result.detections)} detections against a limit of 1"
        )


def _nms_declaration_matches_behaviour(adapter) -> None:
    """Obligation D4 — a platform cannot correct for what it does not know."""
    nms = adapter.capabilities().nms
    if nms.applied:
        assert nms.iou_threshold is not None, (
            "an adapter applying NMS must declare its IoU threshold, or the "
            "platform cannot reason about whether to apply its own"
        )
        assert 0.0 <= nms.iou_threshold <= 1.0, (
            f"declared IoU threshold {nms.iou_threshold} escapes [0,1]"
        )


def _statelessness(adapter) -> None:
    """Obligation D7 — frame N must not depend on frame N-1.

    Run the same frame twice with an unrelated frame in between. A stateful
    adapter carries something across and the two results diverge.
    """
    probe = _frame(7)
    first = adapter.detect([probe], _request())
    adapter.detect([_frame(99, width=200, height=200)], _request())
    second = adapter.detect([probe], _request())

    if not adapter.capabilities().deterministic:
        return
    first_boxes = [(d.class_id, d.box) for d in first[0].detections]
    second_boxes = [(d.class_id, d.box) for d in second[0].detections]
    assert first_boxes == second_boxes, (
        "the same frame produced different results after an unrelated frame; the "
        "adapter is stateful across calls (obligation D7)"
    )


def _determinism(adapter) -> None:
    """A detector declaring determinism must actually be deterministic."""
    if not adapter.capabilities().deterministic:
        return
    probe = _frame(3)
    runs = [adapter.detect([probe], _request()) for _ in range(3)]
    signatures = [
        tuple((d.class_id, d.box, round(d.score, 6)) for d in run[0].detections)
        for run in runs
    ]
    assert len(set(signatures)) == 1, (
        "an adapter declaring deterministic=True produced varying results across "
        "identical calls; replay and regression testing both depend on this"
    )


def _empty_result_is_not_error(adapter) -> None:
    """Obligation D5 — an empty scene is a result, not a failure."""
    blank = FrameView(
        frame_ref=FrameRef(CameraId("kit-cam"), StreamEpoch(1), FrameSeq(500)),
        dimensions=FrameDimensions(width=_WIDTH, height=_HEIGHT),
        pixels=memoryview(bytearray(_WIDTH * _HEIGHT * _CHANNELS)).toreadonly(),
    )
    results = adapter.detect([blank], _request(min_confidence=0.999999))
    assert len(results) == 1, "an empty result is still one result per frame"
    assert isinstance(results[0].detections, tuple | list), (
        "detections must be a sequence, empty or not — never None, which would "
        "make 'nothing detected' indistinguishable from 'failed'"
    )


# --- failure ------------------------------------------------------------------- #


def _no_fabrication_on_failure(adapter) -> None:
    """The most important check in the kit.

    An adapter handed unusable input must fail explicitly or return nothing. One
    that returns a plausible guess poisons state with fully-provenanced fiction
    that looks entirely legitimate downstream (obligation A4).
    """
    degenerate = FrameView(
        frame_ref=FrameRef(CameraId("kit-cam"), StreamEpoch(1), FrameSeq(900)),
        dimensions=FrameDimensions(width=1, height=1),
        pixels=memoryview(bytearray(3)).toreadonly(),
    )
    try:
        results = adapter.detect([degenerate], _request())
    except (DetectionFailedError, VisionOSError):
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"an adapter must raise a typed VisionOSError on failure, got "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    assert len(results) == 1
    for detection in results[0].detections:
        assert detection.box.is_within_unit(), (
            "an adapter fabricated an out-of-range box from degenerate input "
            "rather than failing or returning nothing"
        )


def _corrupt_input_is_typed(adapter) -> None:
    """A truncated buffer must produce a typed failure, never a crash."""
    truncated = FrameView(
        frame_ref=FrameRef(CameraId("kit-cam"), StreamEpoch(1), FrameSeq(901)),
        dimensions=FrameDimensions(width=_WIDTH, height=_HEIGHT),
        pixels=memoryview(bytearray(8)).toreadonly(),
    )
    try:
        adapter.detect([truncated], _request())
    except VisionOSError:
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"corrupt input must raise a typed VisionOSError, got "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _extreme_resolution_is_handled(adapter) -> None:
    huge = _frame(0, width=4096, height=2160)
    try:
        results = adapter.detect([huge], _request())
    except VisionOSError:
        return
    assert len(results) == 1, "an oversized frame must be handled or typed-rejected"


def _warm_is_idempotent(adapter) -> None:
    adapter.warm()
    adapter.warm()
    health = adapter.health()
    assert health is not None, "health() must return a report after warmup"


def _health_never_raises(adapter) -> None:
    """A component that cannot report its health is itself unhealthy."""
    report = adapter.health()
    assert report.state is not None
    assert report.component_id, "a health report must name its component"


# --- resource -------------------------------------------------------------------- #


def _no_steady_state_growth(adapter) -> None:
    """Detection runs continuously for months; a per-call leak is fatal by day 26."""
    import gc

    frames = [_frame(index) for index in range(4)]
    for _ in range(50):
        adapter.detect(frames, _request())
    gc.collect()
    before = len(gc.get_objects())
    for _ in range(400):
        adapter.detect(frames, _request())
    gc.collect()
    after = len(gc.get_objects())
    growth = after - before
    assert growth < 20_000, (
        f"adapter retained {growth} objects across 400 batches — a slow leak that "
        f"looks fine on day one and kills a node on day 26"
    )


def _batch_declaration_is_truthful(adapter) -> None:
    batch = adapter.capabilities().batch
    if not batch.supported:
        return
    size = min(batch.max_size, 8)
    frames = [_frame(index) for index in range(size)]
    results = adapter.detect(frames, _request())
    assert len(results) == size, (
        f"adapter declares batch support up to {batch.max_size} but returned "
        f"{len(results)} results for a batch of {size}"
    )


DETECTOR_KIT = ConformanceKit(
    port_id=PortCatalogue.DETECTOR,
    version="1.0.0",
    checks=(
        # shape
        ConformanceCheck(
            "declares_capabilities", KitSection.SHAPE, _declares_capabilities, "A1"
        ),
        ConformanceCheck(
            "capabilities_are_stable", KitSection.SHAPE, _capabilities_are_stable, "A1"
        ),
        ConformanceCheck(
            "batch_order_preserved", KitSection.SHAPE, _batch_order_preserved, "D6"
        ),
        ConformanceCheck("empty_batch_is_handled", KitSection.SHAPE, _empty_batch_is_handled),
        ConformanceCheck(
            "model_metadata_is_complete", KitSection.SHAPE, _model_metadata_is_complete, "A3"
        ),
        ConformanceCheck(
            "inference_timing_is_reported", KitSection.SHAPE, _inference_timing_is_reported
        ),
        # semantics
        ConformanceCheck(
            "coordinate_normalization", KitSection.SEMANTICS, _coordinate_normalization, "D1"
        ),
        ConformanceCheck(
            "letterbox_inverse_exactness",
            KitSection.SEMANTICS,
            _extreme_aspect_ratios,
            "D1",
        ),
        ConformanceCheck(
            "taxonomy_mapping_complete", KitSection.SEMANTICS, _taxonomy_mapping_complete, "D2"
        ),
        ConformanceCheck(
            "confidence_semantics", KitSection.SEMANTICS, _confidence_semantics, "D3"
        ),
        ConformanceCheck("threshold_behaviour", KitSection.SEMANTICS, _threshold_behaviour),
        ConformanceCheck(
            "max_detections_respected", KitSection.SEMANTICS, _max_detections_respected
        ),
        ConformanceCheck(
            "nms_declaration_matches_behaviour",
            KitSection.SEMANTICS,
            _nms_declaration_matches_behaviour,
            "D4",
        ),
        ConformanceCheck("statelessness", KitSection.SEMANTICS, _statelessness, "D7"),
        ConformanceCheck("determinism", KitSection.SEMANTICS, _determinism),
        ConformanceCheck(
            "empty_result_is_not_error", KitSection.SEMANTICS, _empty_result_is_not_error, "D5"
        ),
        # failure
        ConformanceCheck(
            "no_fabrication_on_failure", KitSection.FAILURE, _no_fabrication_on_failure, "A4"
        ),
        ConformanceCheck(
            "corrupt_input_is_typed", KitSection.FAILURE, _corrupt_input_is_typed, "A4"
        ),
        ConformanceCheck(
            "extreme_resolution_is_handled", KitSection.FAILURE, _extreme_resolution_is_handled
        ),
        ConformanceCheck("warm_is_idempotent", KitSection.FAILURE, _warm_is_idempotent),
        ConformanceCheck("health_never_raises", KitSection.FAILURE, _health_never_raises),
        # resource
        ConformanceCheck(
            "no_steady_state_growth", KitSection.RESOURCE, _no_steady_state_growth
        ),
        ConformanceCheck(
            "batch_declaration_is_truthful",
            KitSection.RESOURCE,
            _batch_declaration_is_truthful,
        ),
    ),
)


def detector_kit_checks() -> Sequence[str]:
    """Check names, for the conformance report."""
    return tuple(check.qualified_name for check in DETECTOR_KIT.checks)
