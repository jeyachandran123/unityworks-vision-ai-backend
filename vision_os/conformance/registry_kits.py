"""Conformance kits for the registry ports — the object store, and P11.

``ObjectStorePort`` carries less semantic weight than a detector or a tracker,
but each check below guards something that fails silently otherwise:

``store/decode_failure_is_loud``
    A store that returns an empty partition when its data is corrupt reports
    data loss as a fresh start. Every object the platform had asserted vanishes,
    and the only symptom is that identity mysteriously restarted.

``store/round_trip_preserves_identity``
    The whole point. ``07_STATE`` section 9.3 promises *"object identity
    survives"*; a store that loses ``object_id``, lineage, or ``merged_into``
    breaks V5 across every restart.

``store/absence_is_not_an_error``
    A cold start must be distinguishable from a failure, or the platform cannot
    tell "first boot" from "your disk is gone".

**The P11 kit is deliberately not registered.** No resolver ships in Phase 1, and
registering a kit for a port with no implementations would suggest one is
expected. The checks exist so that a Phase 2 resolver has an executable contract
waiting for it.
"""

from __future__ import annotations

from ..core.errors import ObjectStoreError
from ..core.model.confidence import Confidence, ConfidenceSemantics
from ..core.model.ids import (
    CameraId,
    ClassId,
    ConfigRevision,
    LocalTrackId,
    ModuleId,
    ObjectId,
    SiteId,
    TenantId,
    TrackerEpoch,
    TrackId,
)
from ..core.model.provenance import Provenance
from ..core.model.space import Box, FrameOfReference, SpatialInfo
from ..core.model.timebase import Instant
from ..core.model.visual_object import (
    BindingId,
    BindingMethod,
    ClassObservation,
    LifecycleState,
    TrackBinding,
    VisualObject,
)
from ..core.ports.registry import PartitionSnapshot, ResolutionRequest
from ..kernel.plugins.manifest import PortCatalogue
from .kit import ConformanceCheck, ConformanceKit, KitSection

_CAMERA = CameraId("kit-store-cam")
_TENANT = TenantId("kit-tenant")
_SITE = SiteId("kit-site")


def _object(
    object_id: str = "KIT0000000000000000000001",
    *,
    lifecycle: LifecycleState = LifecycleState.ACTIVE,
    merged_into: str | None = None,
) -> VisualObject:
    return VisualObject(
        object_id=ObjectId(object_id),
        tenant_id=_TENANT,
        site_id=_SITE,
        camera_id=_CAMERA,
        class_id=ClassId("person"),
        confidence=Confidence.uncalibrated(0.9, ConfidenceSemantics.IDENTITY),
        lifecycle=lifecycle,
        class_history=(
            ClassObservation(
                class_id=ClassId("person"),
                observed_at=Instant(1_000_000_000),
                confidence=Confidence.uncalibrated(
                    0.9, ConfidenceSemantics.CLASSIFICATION
                ),
            ),
        ),
        track_bindings=(
            TrackBinding(
                binding_id=BindingId("KITBINDING0000000000000001"),
                track_id=TrackId(_CAMERA, TrackerEpoch(1), LocalTrackId(7)),
                bound_from=Instant(1_000_000_000),
                confidence=Confidence.uncalibrated(0.8, ConfidenceSemantics.IDENTITY),
                method=BindingMethod.FIRST_SIGHT,
            ),
        ),
        current_spatial=SpatialInfo(
            frame_of_reference=FrameOfReference.NORMALIZED, bbox=Box(0.1, 0.2, 0.3, 0.6)
        ),
        spatial_history=(
            (
                Instant(1_000_000_000),
                SpatialInfo(
                    frame_of_reference=FrameOfReference.NORMALIZED,
                    bbox=Box(0.1, 0.2, 0.3, 0.6),
                ),
            ),
        ),
        attributes={},
        first_seen=Instant(1_000_000_000),
        last_seen=Instant(2_000_000_000),
        last_confirmed=Instant(2_000_000_000),
        observation_count=5,
        provenance=Provenance(
            producer_module=ModuleId("object_registry"),
            producer_version="1.0.0",
            config_revision=ConfigRevision("kit"),
            deterministic=True,
        ),
        merged_into=ObjectId(merged_into) if merged_into else None,
        lineage=(ObjectId("KIT0000000000000000000099"),),
    )


def _snapshot(*objects: VisualObject, camera: CameraId = _CAMERA) -> PartitionSnapshot:
    return PartitionSnapshot(
        camera_id=camera,
        site_id=_SITE,
        version=3,
        taken_at=Instant(2_500_000_000),
        objects=objects or (_object(),),
        next_local_sequence=11,
    )


# --- shape -------------------------------------------------------------------- #


def _store_declares_id(adapter) -> None:
    assert adapter.store_id, "an object store must declare a store_id"


def _absence_is_not_an_error(adapter) -> None:
    """A cold start must be distinguishable from a failure (obligation S2)."""
    result = adapter.load(CameraId("kit-never-written"))
    assert result is None, (
        f"loading an unwritten partition returned {type(result).__name__}; "
        f"absence is not an error and must not be one"
    )


# --- semantics ----------------------------------------------------------------- #


def _round_trip_preserves_identity(adapter) -> None:
    """The promise of 07_STATE section 9.3: object identity survives."""
    original = _object()
    adapter.save(_snapshot(original))
    reloaded = adapter.load(_CAMERA)

    assert reloaded is not None, "a saved partition must reload"
    assert reloaded.count == 1
    restored = reloaded.objects[0]
    assert restored.object_id == original.object_id, "object_id was not preserved"
    assert restored.camera_id == original.camera_id
    assert restored.tenant_id == original.tenant_id, (
        "tenancy was not preserved; a reloaded object in the wrong tenant crosses "
        "the platform's hard isolation boundary"
    )
    assert restored.first_seen == original.first_seen, (
        "first_seen was not preserved; an object present for 20 minutes would "
        "reload as new, which is exactly what durable identity exists to prevent"
    )
    assert restored.lifecycle is original.lifecycle
    assert restored.lineage == original.lineage


def _merged_objects_survive(adapter) -> None:
    """V5: an observation referencing a merged id must stay resolvable.

    If a store drops terminal objects to save space, history stays resolvable
    only until the next restart.
    """
    merged = _object(
        object_id="KIT0000000000000000000002",
        lifecycle=LifecycleState.MERGED_INTO,
        merged_into="KIT0000000000000000000001",
    )
    adapter.save(_snapshot(_object(), merged))
    reloaded = adapter.load(_CAMERA)

    assert reloaded is not None
    by_id = {o.object_id: o for o in reloaded.objects}
    survivor = by_id.get(ObjectId("KIT0000000000000000000002"))
    assert survivor is not None, (
        "a merged object was dropped on reload; observations referencing it are "
        "now unresolvable (invariant V5)"
    )
    assert survivor.merged_into == ObjectId("KIT0000000000000000000001")


def _bindings_survive_with_their_method(adapter) -> None:
    """Provenance of an identity claim must outlive a restart (V4)."""
    adapter.save(_snapshot())
    reloaded = adapter.load(_CAMERA)
    assert reloaded is not None
    bindings = reloaded.objects[0].track_bindings
    assert bindings, "track bindings were not preserved"
    assert bindings[0].method is BindingMethod.FIRST_SIGHT
    assert bindings[0].confidence is not None, (
        "binding confidence was dropped; an identity claim without its confidence "
        "is an assumed truth, which 02_VOM section 4.2 forbids"
    )


def _overwrite_replaces_not_appends(adapter) -> None:
    """A second save must replace the partition, not accumulate it."""
    adapter.save(_snapshot(_object()))
    adapter.save(_snapshot(_object(), _object(object_id="KIT0000000000000000000003")))
    reloaded = adapter.load(_CAMERA)
    assert reloaded is not None
    assert reloaded.count == 2, f"expected 2 objects after overwrite, got {reloaded.count}"


def _partitions_are_independent(adapter) -> None:
    """The camera is the partition; writing one must not disturb another."""
    other = CameraId("kit-store-cam-2")
    adapter.save(_snapshot(_object()))
    adapter.save(_snapshot(_object(object_id="KIT0000000000000000000004"), camera=other))

    first = adapter.load(_CAMERA)
    second = adapter.load(other)
    assert first is not None and second is not None
    assert first.objects[0].object_id != second.objects[0].object_id
    assert first.camera_id == _CAMERA
    assert second.camera_id == other


def _snapshot_version_is_preserved(adapter) -> None:
    """The version is how a reader detects a stale or concurrent write."""
    adapter.save(_snapshot())
    reloaded = adapter.load(_CAMERA)
    assert reloaded is not None
    assert reloaded.version == 3


# --- failure --------------------------------------------------------------------- #


def _decode_failure_is_loud(adapter) -> None:
    """Obligation S3 — never present data loss as a fresh start.

    Only meaningful for a store with an inspectable backing medium; a memory
    store cannot become corrupt and correctly skips.
    """
    corrupt = getattr(adapter, "_path", None)
    if corrupt is None:
        return

    adapter.save(_snapshot())
    path = corrupt(_CAMERA)
    path.write_text("{ this is not json", encoding="utf-8")

    try:
        adapter.load(_CAMERA)
    except ObjectStoreError:
        return
    raise AssertionError(
        "a corrupt partition loaded without error; silently returning an empty "
        "partition reports data loss as a fresh start (obligation S3)"
    )


def _forget_removes_the_partition(adapter) -> None:
    adapter.save(_snapshot())
    assert adapter.load(_CAMERA) is not None
    adapter.forget(_CAMERA)
    assert adapter.load(_CAMERA) is None


def _forget_of_unknown_partition_is_safe(adapter) -> None:
    adapter.forget(CameraId("kit-never-existed"))


def _empty_partition_round_trips(adapter) -> None:
    """A camera with no objects is a valid state, not a missing one."""
    adapter.save(
        PartitionSnapshot(
            camera_id=CameraId("kit-empty"),
            site_id=_SITE,
            version=1,
            taken_at=Instant(1),
            objects=(),
        )
    )
    reloaded = adapter.load(CameraId("kit-empty"))
    assert reloaded is not None, "an empty partition must reload as empty, not absent"
    assert reloaded.count == 0


# --- resource ---------------------------------------------------------------------- #


def _large_partition_round_trips(adapter) -> None:
    """A busy camera's whole population must survive, not a truncated prefix."""
    objects = tuple(
        _object(object_id=f"KIT{index:023d}") for index in range(1, 121)
    )
    adapter.save(_snapshot(*objects))
    reloaded = adapter.load(_CAMERA)
    assert reloaded is not None
    assert reloaded.count == 120, (
        f"saved 120 objects, reloaded {reloaded.count}; a store that truncates "
        f"loses identity for exactly the cameras that need it most"
    )


OBJECT_STORE_KIT = ConformanceKit(
    port_id=PortCatalogue.STATE_STORE,
    version="1.0.0",
    checks=(
        ConformanceCheck("declares_id", KitSection.SHAPE, _store_declares_id, "A1"),
        ConformanceCheck(
            "absence_is_not_an_error", KitSection.SHAPE, _absence_is_not_an_error, "S2"
        ),
        ConformanceCheck(
            "round_trip_preserves_identity",
            KitSection.SEMANTICS,
            _round_trip_preserves_identity,
            "S1",
        ),
        ConformanceCheck(
            "merged_objects_survive", KitSection.SEMANTICS, _merged_objects_survive, "V5"
        ),
        ConformanceCheck(
            "bindings_survive_with_their_method",
            KitSection.SEMANTICS,
            _bindings_survive_with_their_method,
            "V4",
        ),
        ConformanceCheck(
            "overwrite_replaces_not_appends",
            KitSection.SEMANTICS,
            _overwrite_replaces_not_appends,
            "S1",
        ),
        ConformanceCheck(
            "partitions_are_independent",
            KitSection.SEMANTICS,
            _partitions_are_independent,
            "S1",
        ),
        ConformanceCheck(
            "snapshot_version_is_preserved",
            KitSection.SEMANTICS,
            _snapshot_version_is_preserved,
        ),
        ConformanceCheck(
            "decode_failure_is_loud", KitSection.FAILURE, _decode_failure_is_loud, "S3"
        ),
        ConformanceCheck(
            "forget_removes_the_partition", KitSection.FAILURE, _forget_removes_the_partition
        ),
        ConformanceCheck(
            "forget_of_unknown_partition_is_safe",
            KitSection.FAILURE,
            _forget_of_unknown_partition_is_safe,
        ),
        ConformanceCheck(
            "empty_partition_round_trips", KitSection.FAILURE, _empty_partition_round_trips
        ),
        ConformanceCheck(
            "large_partition_round_trips", KitSection.RESOURCE, _large_partition_round_trips
        ),
    ),
)


# --- P11, specified but unregistered ------------------------------------------------ #


def _resolver_declares_id(adapter) -> None:
    assert adapter.resolver_id, "a resolver must declare a resolver_id"


def _resolver_is_advisory(adapter) -> None:
    """I1 — a resolver proposes; it never mutates and never mints."""
    request = ResolutionRequest(
        camera_id=_CAMERA,
        site_id=_SITE,
        track_id=TrackId(_CAMERA, TrackerEpoch(1), LocalTrackId(1)),
        observed_at=Instant(1_000_000_000),
        spatial=SpatialInfo(
            frame_of_reference=FrameOfReference.NORMALIZED, bbox=Box(0.1, 0.1, 0.2, 0.4)
        ),
        class_id="person",
        candidates=(_object(),),
    )
    before = request.candidates[0]
    adapter.resolve(request)
    assert request.candidates[0] is before, "a resolver mutated its candidate set"


def _resolver_stays_within_candidates(adapter) -> None:
    """I4 — minting identity is the registry's sole authority."""
    candidate = _object()
    request = ResolutionRequest(
        camera_id=_CAMERA,
        site_id=_SITE,
        track_id=TrackId(_CAMERA, TrackerEpoch(1), LocalTrackId(1)),
        observed_at=Instant(1_000_000_000),
        spatial=SpatialInfo(
            frame_of_reference=FrameOfReference.NORMALIZED, bbox=Box(0.1, 0.1, 0.2, 0.4)
        ),
        class_id="person",
        candidates=(candidate,),
    )
    result = adapter.resolve(request)
    allowed = {candidate.object_id}
    for ranked in result.ranked:
        assert ranked.object_id in allowed, (
            f"resolver proposed '{ranked.object_id}', which was not a candidate; "
            f"only the registry may mint an object identity (01_LAYERED section 8)"
        )


def _resolver_abstention_is_explicit(adapter) -> None:
    """I3 — "no basis to answer" differs from "no candidate matches" (V8)."""
    request = ResolutionRequest(
        camera_id=_CAMERA,
        site_id=_SITE,
        track_id=TrackId(_CAMERA, TrackerEpoch(1), LocalTrackId(1)),
        observed_at=Instant(1_000_000_000),
        spatial=SpatialInfo(
            frame_of_reference=FrameOfReference.NORMALIZED, bbox=Box(0.1, 0.1, 0.2, 0.4)
        ),
        class_id="person",
        candidates=(),
    )
    result = adapter.resolve(request)
    assert isinstance(result.abstained, bool)
    if not result.ranked:
        assert result.reason or result.abstained, (
            "an empty ranking with no reason is indistinguishable from abstention"
        )


def _resolver_is_deterministic(adapter) -> None:
    """I5 — identical input yields identical ranking, including tie order."""
    request = ResolutionRequest(
        camera_id=_CAMERA,
        site_id=_SITE,
        track_id=TrackId(_CAMERA, TrackerEpoch(1), LocalTrackId(1)),
        observed_at=Instant(1_000_000_000),
        spatial=SpatialInfo(
            frame_of_reference=FrameOfReference.NORMALIZED, bbox=Box(0.1, 0.1, 0.2, 0.4)
        ),
        class_id="person",
        candidates=(_object(), _object(object_id="KIT0000000000000000000005")),
    )
    first = adapter.resolve(request)
    second = adapter.resolve(request)
    assert [c.object_id for c in first.ranked] == [c.object_id for c in second.ranked]


#: Executable contract for P11, **not registered** in ``platform_registry()``.
#:
#: 15_ROADMAP section 3: no implementations ship in Phase 1. Registering a kit
#: for a port with no implementations would suggest one is expected; these checks
#: exist so a Phase 2 resolver has a contract waiting rather than one written
#: after the fact.
IDENTITY_RESOLVER_KIT = ConformanceKit(
    port_id=PortCatalogue.IDENTITY_RESOLVER,
    version="1.0.0",
    checks=(
        ConformanceCheck("declares_id", KitSection.SHAPE, _resolver_declares_id, "A1"),
        ConformanceCheck("advisory_only", KitSection.SEMANTICS, _resolver_is_advisory, "I1"),
        ConformanceCheck(
            "stays_within_candidates",
            KitSection.SEMANTICS,
            _resolver_stays_within_candidates,
            "I4",
        ),
        ConformanceCheck(
            "abstention_is_explicit",
            KitSection.SEMANTICS,
            _resolver_abstention_is_explicit,
            "I3",
        ),
        ConformanceCheck(
            "determinism", KitSection.SEMANTICS, _resolver_is_deterministic, "I5"
        ),
    ),
)
