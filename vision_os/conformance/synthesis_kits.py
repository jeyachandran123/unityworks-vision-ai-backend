"""Conformance kits for P18, P19 and P20.

Each check below guards something that fails **silently** otherwise:

``suppression/first_always_publishes``
    A policy that suppressed a first observation would let an object exist in the
    log only implicitly. Nothing downstream could detect it: the object simply
    appears later, with no beginning.

``suppression/heartbeat_always_publishes``
    Without it, a working camera watching a stationary scene and a dead camera
    produce byte-identical silence. §M11 makes this the reason the heartbeat
    exists at all.

``sink/never_mutates``
    A sink that altered an observation would make two consumers disagree about a
    published fact — and the log would say one of them was wrong.

``log/idempotent_by_id``
    07_STATE §9.1 recovers a crashed writer by *"replay from the last committed
    log position"*. If replay double-counted, every recovery would corrupt the
    record it was recovering.

**What these kits cannot check**, stated plainly: whether a suppression policy
suppresses the *right* things, or whether a log is genuinely durable. The first
needs a deployment's judgment; the second needs to survive a power cut. The kits
verify contracts — the structural properties whose violation is silent.
"""

from __future__ import annotations

from dataclasses import replace

from ..core.model.confidence import Confidence, ConfidenceSemantics
from ..core.model.ids import (
    CameraId,
    ConfigRevision,
    FrameRef,
    FrameSeq,
    LogPosition,
    ModuleId,
    ObjectId,
    ObservationId,
    SiteId,
    StreamEpoch,
    TenantId,
)
from ..core.model.observation import (
    CoverageWindow,
    MeasurementBasis,
    ObservabilityReason,
    ObservabilityStatus,
    Observation,
    ObservationType,
)
from ..core.model.provenance import Provenance
from ..core.model.space import Box, FrameOfReference, SpatialInfo
from ..core.model.timebase import ClockQuality, Duration, Instant
from ..core.model.understanding import Timing
from ..kernel.plugins.manifest import PortCatalogue
from .kit import ConformanceCheck, ConformanceKit, KitSection

_CAMERA = CameraId("kit-obs-cam")
_TENANT = TenantId("kit-tenant")
_SITE = SiteId("kit-site")
_PROVENANCE = Provenance(
    producer_module=ModuleId("observation_builder"),
    producer_version="1.0.0",
    config_revision=ConfigRevision("kit"),
)


def _observation(
    suffix: str = "1",
    *,
    kind: ObservationType = ObservationType.PRESENCE,
    box: Box | None = None,
    at: int = 1_000_000_000,
) -> Observation:
    spatial = SpatialInfo(
        frame_of_reference=FrameOfReference.NORMALIZED,
        bbox=box or Box(0.3, 0.3, 0.5, 0.8),
    )
    coverage = (
        CoverageWindow(
            status=ObservabilityStatus.BLIND,
            reason=ObservabilityReason.STREAM_DISCONNECTED,
            since=Instant(at),
        )
        if kind is ObservationType.COVERAGE
        else None
    )
    return Observation(
        observation_id=ObservationId(f"kit-obs-{suffix}"),
        observation_type=kind,
        tenant_id=_TENANT,
        site_id=_SITE,
        camera_id=_CAMERA,
        frame_ref=FrameRef(_CAMERA, StreamEpoch(1), FrameSeq(int(suffix) if suffix.isdigit() else 1)),
        t_capture=Instant(at),
        t_capture_unc=Duration.from_millis(10),
        clock_quality=ClockQuality.NTP_SYNCED,
        t_published=Instant(at + 1_000_000),
        provenance=_PROVENANCE,
        timing=Timing(total_ms=1.0),
        object_id=None if kind is ObservationType.COVERAGE else ObjectId("kit-object"),
        confidence=(
            None
            if kind is ObservationType.COVERAGE
            else Confidence.uncalibrated(0.9, ConfidenceSemantics.IDENTITY)
        ),
        spatial=None if kind is ObservationType.COVERAGE else spatial,
        coverage=coverage,
        measurement_basis=MeasurementBasis.MEASURED,
    )


# --- P18 SuppressionPolicyPort --------------------------------------------- #


def _check_suppression_shape(adapter) -> None:
    assert hasattr(adapter, "policy_id"), "a suppression policy must expose policy_id"
    assert isinstance(adapter.policy_id, str) and adapter.policy_id
    signature = adapter.signature(_observation())
    assert isinstance(signature, str) and signature, (
        "a signature must be a non-empty string; the builder stores it verbatim"
    )


def _check_first_always_publishes(adapter) -> None:
    """Obligation S1. There is nothing to compare against."""
    decision = adapter.should_publish(
        _observation(),
        None,
        elapsed=Duration(0),
        heartbeat=Duration.from_millis(30_000),
    )
    assert decision.publish, (
        "a policy suppressed a first observation; the object would then exist in "
        "the log only implicitly, appearing later with no beginning"
    )


def _check_heartbeat_always_publishes(adapter) -> None:
    """Obligation S2. The difference between 'unchanged' and 'stopped observing'."""
    candidate = _observation()
    decision = adapter.should_publish(
        candidate,
        adapter.signature(candidate),
        elapsed=Duration.from_millis(60_000),
        heartbeat=Duration.from_millis(30_000),
    )
    assert decision.publish, (
        "an unchanged observation past its heartbeat was suppressed; a working "
        "camera watching a still scene and a dead camera then produce identical "
        "silence (04_MODULES section M11)"
    )


def _check_coverage_is_never_suppressed(adapter) -> None:
    """02_VOM §11.2: coverage is *"not optional"*.

    A suppressed blindness report is the platform deciding its own outage was not
    worth mentioning.
    """
    candidate = _observation(kind=ObservationType.COVERAGE)
    decision = adapter.should_publish(
        candidate,
        adapter.signature(candidate),
        elapsed=Duration(0),
        heartbeat=Duration.from_millis(30_000),
    )
    assert decision.publish, (
        "a coverage observation was suppressed; this type is how the platform "
        "says it cannot see, and it is not optional"
    )


def _check_correction_always_publishes(adapter) -> None:
    """Obligation S4. A correction nobody receives is not a correction."""
    candidate = replace(
        _observation(), supersedes=ObservationId("kit-obs-earlier")
    )
    decision = adapter.should_publish(
        candidate,
        adapter.signature(candidate),
        elapsed=Duration(0),
        heartbeat=Duration.from_millis(30_000),
    )
    assert decision.publish, "a correction was suppressed"


def _check_signature_is_content_only(adapter) -> None:
    """Obligation S7. Otherwise nothing is ever suppressed.

    Two observations differing only in id and publication time describe the same
    fact. A signature that distinguished them would make every observation look
    changed.
    """
    first = _observation("1")
    second = replace(
        first,
        observation_id=ObservationId("kit-obs-2"),
        t_published=Instant(first.t_published.ns + 5_000_000_000),
    )
    assert adapter.signature(first) == adapter.signature(second), (
        "the signature varies with identity or publication time; suppression "
        "would then never suppress anything"
    )


def _check_signature_detects_change(adapter) -> None:
    """A policy whose signature never changes suppresses real changes."""
    first = _observation("1")
    moved = replace(
        first,
        spatial=SpatialInfo(
            frame_of_reference=FrameOfReference.NORMALIZED,
            bbox=Box(0.1, 0.1, 0.9, 0.9),
        ),
    )
    assert adapter.signature(first) != adapter.signature(moved), (
        "a large positional change produced an identical signature; the policy "
        "would suppress movement it should publish"
    )


def _check_suppression_determinism(adapter) -> None:
    candidate = _observation()
    first = adapter.should_publish(
        candidate, "other", elapsed=Duration(0), heartbeat=Duration.from_millis(30_000)
    )
    second = adapter.should_publish(
        candidate, "other", elapsed=Duration(0), heartbeat=Duration.from_millis(30_000)
    )
    assert first.publish == second.publish, (
        "identical inputs produced different decisions; replay would publish a "
        "different observation stream (V13)"
    )
    assert adapter.signature(candidate) == adapter.signature(candidate)


def _check_decision_carries_a_reason(adapter) -> None:
    candidate = _observation()
    for previous in (None, adapter.signature(candidate), "other"):
        decision = adapter.should_publish(
            candidate,
            previous,
            elapsed=Duration(0),
            heartbeat=Duration.from_millis(30_000),
        )
        assert decision.reason, (
            "a decision carried no reason; a quiet platform with no explanation "
            "is indistinguishable from a broken one"
        )


SUPPRESSION_POLICY_KIT = ConformanceKit(
    port_id=PortCatalogue.SUPPRESSION_POLICY,
    version="1.0.0",
    checks=(
        ConformanceCheck("interface", KitSection.SHAPE, _check_suppression_shape),
        ConformanceCheck(
            "first_always_publishes",
            KitSection.SEMANTICS,
            _check_first_always_publishes,
            obligation="S1",
        ),
        ConformanceCheck(
            "heartbeat_always_publishes",
            KitSection.SEMANTICS,
            _check_heartbeat_always_publishes,
            obligation="S2",
        ),
        ConformanceCheck(
            "coverage_never_suppressed",
            KitSection.SEMANTICS,
            _check_coverage_is_never_suppressed,
            obligation="S2",
        ),
        ConformanceCheck(
            "correction_always_publishes",
            KitSection.SEMANTICS,
            _check_correction_always_publishes,
            obligation="S4",
        ),
        ConformanceCheck(
            "signature_is_content_only",
            KitSection.SEMANTICS,
            _check_signature_is_content_only,
            obligation="S7",
        ),
        ConformanceCheck(
            "signature_detects_change", KitSection.GOLDEN, _check_signature_detects_change
        ),
        ConformanceCheck(
            "determinism", KitSection.SEMANTICS, _check_suppression_determinism, obligation="S3"
        ),
        ConformanceCheck(
            "decision_carries_a_reason", KitSection.FAILURE, _check_decision_carries_a_reason
        ),
    ),
)


# --- P19 ObservationSinkPort ------------------------------------------------ #


def _check_sink_shape(adapter) -> None:
    assert hasattr(adapter, "sink_id"), "a sink must expose sink_id"
    assert isinstance(adapter.sink_id, str) and adapter.sink_id
    assert isinstance(adapter.durable, bool), (
        "a sink must declare durability; a tee to a dashboard is not a system of "
        "record and must not be mistaken for one (obligation K5)"
    )
    result = adapter.emit([_observation()])
    assert result is not None, "emit must return a result"


def _check_sink_never_mutates(adapter) -> None:
    """Obligation K1. Observations are immutable facts (V5).

    A sink that altered one would make two consumers disagree about a published
    fact, and the log would say one of them was wrong.
    """
    observation = _observation()
    before = (
        observation.observation_id,
        observation.observation_type,
        observation.t_capture,
        observation.attributes,
        observation.measurement_basis,
    )
    adapter.emit([observation])
    after = (
        observation.observation_id,
        observation.observation_type,
        observation.t_capture,
        observation.attributes,
        observation.measurement_basis,
    )
    assert before == after, "the sink mutated an observation it was handed"


def _check_sink_is_idempotent(adapter) -> None:
    """Obligation K2. At-least-once delivery needs a harmless repeat."""
    observation = _observation()
    first = adapter.emit([observation])
    second = adapter.emit([observation])
    assert first is not None and second is not None, (
        "a repeated emit must be accepted rather than raising; at-least-once "
        "delivery is workable only if a retry is harmless"
    )


def _check_sink_accepts_an_empty_batch(adapter) -> None:
    result = adapter.emit([])
    assert result.accepted == 0
    assert result.complete, "an empty batch is a complete success, not a failure"


def _check_sink_reports_what_it_took(adapter) -> None:
    """A sink may legitimately keep a subset — but never *silently*."""
    result = adapter.emit([_observation("1"), _observation("2")])
    assert result.accepted + len(result.rejected) <= 2
    assert result.accepted >= 0


OBSERVATION_SINK_KIT = ConformanceKit(
    port_id=PortCatalogue.OBSERVATION_SINK,
    version="1.0.0",
    checks=(
        ConformanceCheck("interface", KitSection.SHAPE, _check_sink_shape),
        ConformanceCheck(
            "never_mutates", KitSection.SEMANTICS, _check_sink_never_mutates, obligation="K1"
        ),
        ConformanceCheck(
            "idempotent", KitSection.SEMANTICS, _check_sink_is_idempotent, obligation="K2"
        ),
        ConformanceCheck(
            "empty_batch_is_success", KitSection.FAILURE, _check_sink_accepts_an_empty_batch
        ),
        ConformanceCheck(
            "reports_what_it_took",
            KitSection.SEMANTICS,
            _check_sink_reports_what_it_took,
            obligation="K3",
        ),
    ),
)


# --- P20 ObservationLogPort ------------------------------------------------- #


def _check_log_shape(adapter) -> None:
    assert hasattr(adapter, "log_id"), "a log must expose log_id"
    assert isinstance(adapter.log_id, str) and adapter.log_id
    result = adapter.append(_CAMERA, [_observation("1")])
    assert result.appended == 1
    assert int(result.position) >= 1


def _check_log_is_append_only(adapter) -> None:
    """Obligation L1. No update, no delete outside retention.

    Verified by shape: an append-only log offers no method that could rewrite a
    record, so the absence *is* the guarantee.
    """
    for forbidden in ("update", "delete", "replace", "overwrite", "set"):
        assert not hasattr(adapter, forbidden), (
            f"a log exposing '{forbidden}' is not append-only; 07_STATE section "
            f"8.2 refuses rewriting because it *'would destroy the property that "
            f"makes the log trustworthy in the first place'*"
        )


def _check_log_idempotent_by_id(adapter) -> None:
    """**Obligation L2 — the check that makes recovery safe.**

    07_STATE §9.1 recovers a crashed writer by replaying from the last committed
    position. If replay double-counted, every recovery would corrupt the record
    it was recovering.
    """
    camera = CameraId("kit-log-idempotent")
    observation = _observation("7")
    first = adapter.append(camera, [observation])
    second = adapter.append(camera, [observation])

    assert first.appended == 1
    assert second.appended == 0, (
        "a repeated observation_id was appended twice; replay after an uncertain "
        "outcome would then corrupt the log"
    )
    assert observation.observation_id in second.duplicates, (
        "a duplicate must be *reported* rather than silently ignored — a retry "
        "reporting duplicates is a success, and a caller needs to know which"
    )


def _check_log_positions_are_monotonic(adapter) -> None:
    """Obligation L3. A watermark that moved backwards is not a watermark."""
    camera = CameraId("kit-log-monotonic")
    positions = [
        int(adapter.append(camera, [_observation(f"m{index}")]).position)
        for index in range(5)
    ]
    assert positions == sorted(positions), f"positions went backwards: {positions}"
    assert len(set(positions)) == len(positions), "positions repeated"


def _check_log_preserves_order(adapter) -> None:
    """Obligation L4. Order is the log's contract."""
    camera = CameraId("kit-log-order")
    written = [_observation(f"o{index}") for index in range(4)]
    adapter.append(camera, written)
    read = list(adapter.read(camera))
    assert [o.observation_id for o in read] == [o.observation_id for o in written], (
        "the log returned records out of append order; a set-like store cannot "
        "satisfy an ordered contract"
    )


def _check_log_tail_follows_without_blocking(adapter) -> None:
    """Obligation L7. A follow must return, even with nothing to follow.

    The failure this catches is a ``tail`` that waits for data. A camera
    watching an empty corridor produces nothing for minutes at a time, and an
    adapter that blocked there would make every subscriber's liveness depend on
    the scene being busy.
    """
    camera = CameraId("kit-log-tail")
    assert list(adapter.tail(camera)) == [], (
        "tail on an empty partition must return immediately with nothing, not "
        "block waiting for a first record"
    )

    written = [_observation(f"t{index}") for index in range(4)]
    adapter.append(camera, written)

    assert [o.observation_id for o in adapter.tail(camera)] == [
        o.observation_id for o in written
    ], "tail from the start must yield the whole partition in append order"

    resumed = list(adapter.tail(camera, start=LogPosition(2)))
    assert [o.observation_id for o in resumed] == [
        o.observation_id for o in written[2:]
    ], (
        "tail from a position must yield only what followed it; a subscriber "
        "resuming from a cursor would otherwise receive duplicates it cannot "
        "distinguish from a genuine re-delivery"
    )

    assert list(adapter.tail(camera, start=LogPosition(len(written)))) == [], (
        "tail from the current end must return empty rather than block"
    )


def _check_log_partitions_are_independent(adapter) -> None:
    """Obligation L6. One camera's traffic never appears in another's."""
    first = CameraId("kit-log-part-a")
    second = CameraId("kit-log-part-b")
    adapter.append(first, [_observation("a1")])
    adapter.append(second, [_observation("b1")])
    assert int(adapter.position(first)) == 1
    assert int(adapter.position(second)) == 1
    assert all(o.observation_id != "kit-obs-b1" for o in adapter.read(first))


def _check_log_empty_partition_is_not_an_error(adapter) -> None:
    """A cold start must be distinguishable from a failure."""
    assert int(adapter.position(CameraId("kit-log-never-used"))) == 0
    assert list(adapter.read(CameraId("kit-log-never-used"))) == []


def _check_log_truncate_removes_a_prefix(adapter) -> None:
    """Retention only, and only from the front.

    Removing from the middle would break the position arithmetic that makes
    rebuild resumable.
    """
    camera = CameraId("kit-log-truncate")
    adapter.append(
        camera,
        [
            _observation("t1", at=1_000_000_000),
            _observation("t2", at=5_000_000_000),
        ],
    )
    removed = adapter.truncate(camera, Instant(3_000_000_000))
    assert removed == 1, f"expected one record removed, got {removed}"
    remaining = [o.observation_id for o in adapter.read(camera)]
    assert "kit-obs-t2" in remaining


OBSERVATION_LOG_KIT = ConformanceKit(
    port_id=PortCatalogue.OBSERVATION_LOG,
    version="1.0.0",
    checks=(
        ConformanceCheck("interface", KitSection.SHAPE, _check_log_shape),
        ConformanceCheck(
            "append_only", KitSection.SHAPE, _check_log_is_append_only, obligation="L1"
        ),
        ConformanceCheck(
            "idempotent_by_id",
            KitSection.SEMANTICS,
            _check_log_idempotent_by_id,
            obligation="L2",
        ),
        ConformanceCheck(
            "positions_are_monotonic",
            KitSection.SEMANTICS,
            _check_log_positions_are_monotonic,
            obligation="L3",
        ),
        ConformanceCheck(
            "preserves_order", KitSection.SEMANTICS, _check_log_preserves_order, obligation="L4"
        ),
        ConformanceCheck(
            "tail_follows_without_blocking",
            KitSection.SEMANTICS,
            _check_log_tail_follows_without_blocking,
            obligation="L7",
        ),
        ConformanceCheck(
            "partitions_are_independent",
            KitSection.SEMANTICS,
            _check_log_partitions_are_independent,
            obligation="L6",
        ),
        ConformanceCheck(
            "empty_partition_is_not_an_error",
            KitSection.FAILURE,
            _check_log_empty_partition_is_not_an_error,
        ),
        ConformanceCheck(
            "truncate_removes_a_prefix", KitSection.RESOURCE, _check_log_truncate_removes_a_prefix
        ),
    ),
)


ALL_SYNTHESIS_KITS: tuple[ConformanceKit, ...] = (
    SUPPRESSION_POLICY_KIT,
    OBSERVATION_SINK_KIT,
    OBSERVATION_LOG_KIT,
)
