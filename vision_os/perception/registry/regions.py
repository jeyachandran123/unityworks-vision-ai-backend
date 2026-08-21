"""Region membership and dwell — **pure geometry** (03_MODULES M7 responsibility 5).

> **Single responsibility:** *Compute which objects are inside which regions, and
> for how long. Interpret nothing.*

This module is where the Semantic Ceiling is most tempting to breach, and
07_STATE section 3.3 says so explicitly:

> *`occupancy` is a count. `dwell_stats` are descriptive statistics over
> durations. There is no `is_crowded`, no `exceeds_capacity`, no `queue_forming`
> — each of those requires a threshold or a definition that only a consumer
> possesses (V1).*

Two properties are load-bearing:

**Dwell is computed from ``t_capture``, never from processing time** (V11,
14_TESTING section 4). A dwell of 45 s means the object was present for 45 s *in
the world*, regardless of whether the platform was keeping up. Computing it from
wall time would make the platform's measurements a function of its own load.

**Membership uses a precomputed spatial index.** Section M7 is explicit that
polygon tests "must not be naive at 100 objects x 20 regions" — 2,000 ray-casts
per frame. Bounding-box rejection reduces that to the handful of genuinely
plausible pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.model.ids import ObjectId, RegionId
from ...core.model.region import ContainmentMethod, MembershipState, Region
from ...core.model.space import Box, Point
from ...core.model.timebase import Duration, Instant


@dataclass(frozen=True, slots=True)
class RegionMembership:
    """One object's standing in one region.

    ``entered_at`` uses capture time, so dwell is a claim about the world rather
    than about the pipeline.
    """

    region_id: RegionId
    geometry_version: str
    state: MembershipState
    method: ContainmentMethod
    entered_at: Instant
    last_confirmed: Instant

    def dwell(self, now: Instant) -> Duration:
        """Elapsed presence, from capture time (V11)."""
        return Duration(max(0, now.ns - self.entered_at.ns))


@dataclass(frozen=True, slots=True)
class RegionTransition:
    """An entry or exit. Descriptive; the platform draws no conclusion from it."""

    object_id: ObjectId
    region_id: RegionId
    geometry_version: str
    entered: bool
    at: Instant
    dwell: Duration = Duration(0)
    """Populated on exit. Zero on entry."""

    method: ContainmentMethod = ContainmentMethod.BBOX_BOTTOM_CENTRE

    @property
    def exited(self) -> bool:
        return not self.entered


@dataclass(frozen=True, slots=True)
class RegionOccupancy:
    """``RegionState`` as 07_STATE section 3.3 defines it — counting only.

    Note what is absent: no ``is_crowded``, no ``exceeds_capacity``, no
    ``queue_forming``. Each needs a threshold only a consumer possesses.
    """

    region_id: RegionId
    geometry_version: str
    occupancy: dict[str, int] = field(default_factory=dict)
    """Count per class id. Pure counting, no interpretation."""

    present_objects: tuple[ObjectId, ...] = ()
    dwell_current_max: Duration = Duration(0)
    dwell_current_mean: Duration = Duration(0)
    last_transition: Instant = Instant(0)

    @property
    def total(self) -> int:
        return len(self.present_objects)


@dataclass(frozen=True, slots=True)
class _IndexedRegion:
    """A region with its precomputed bounding box."""

    region: Region
    bounds: Box


class RegionIndex:
    """Precomputed spatial index over a camera's regions.

    Bounding-box rejection before the ray-cast. At 100 objects x 20 regions the
    naive form is 2,000 polygon tests per frame; almost all of them are answered
    by four float comparisons instead.

    Rebuilt when geometry changes, never mutated in place — a half-updated index
    would silently produce wrong membership.
    """

    __slots__ = ("_regions", "_version")

    def __init__(self, regions: tuple[Region, ...] = ()) -> None:
        self._regions: tuple[_IndexedRegion, ...] = tuple(
            _IndexedRegion(region=r, bounds=r.geometry.bounds) for r in regions
        )
        self._version = 0

    @property
    def version(self) -> int:
        """Increments on every rebuild, so a caller can detect a geometry change."""
        return self._version

    @property
    def regions(self) -> tuple[Region, ...]:
        return tuple(indexed.region for indexed in self._regions)

    def __len__(self) -> int:
        return len(self._regions)

    def rebuild(self, regions: tuple[Region, ...]) -> None:
        self._regions = tuple(
            _IndexedRegion(region=r, bounds=r.geometry.bounds) for r in regions
        )
        self._version += 1

    def containing(self, point: Point) -> tuple[Region, ...]:
        """Regions whose polygon contains the point.

        The bounds check is not an approximation of the answer — it is an exact
        rejection of pairs that cannot possibly match, so the result is identical
        to testing every polygon.
        """
        hits: list[Region] = []
        for indexed in self._regions:
            bounds = indexed.bounds
            if not (bounds.x1 <= point.x <= bounds.x2 and bounds.y1 <= point.y <= bounds.y2):
                continue
            if indexed.region.geometry.contains(point):
                hits.append(indexed.region)
        return tuple(hits)


def containment_point(box: Box, method: ContainmentMethod) -> Point:
    """The point tested for membership.

    Which point is used matters: containment from a bounding box's bottom edge
    and from a projected ground point disagree substantially at range, and a
    consumer comparing dwell across cameras deserves to know which was used —
    which is why the method travels with the membership record.
    """
    if method is ContainmentMethod.BBOX_BOTTOM_CENTRE:
        return box.bottom_centre
    if method is ContainmentMethod.GROUND_POINT:
        # Ground projection needs a calibration the registry does not hold; the
        # caller supplies an already-projected point via SpatialInfo when it has
        # one. Falling back to the bottom centre is the honest approximation,
        # and the recorded method says which was used.
        return box.bottom_centre
    return box.centre


class RegionTracker:
    """Per-camera membership and dwell state. Single-writer, owned by a partition.

    Holds membership keyed by object, **not** on the ``VisualObject`` record:
    02_VOM section 10.6 has no ``regions`` field, and M7 owns region membership
    as separate partition state (section M7 State Ownership).
    """

    __slots__ = ("_index", "_memberships", "_method")

    def __init__(
        self,
        *,
        regions: tuple[Region, ...] = (),
        method: ContainmentMethod = ContainmentMethod.BBOX_BOTTOM_CENTRE,
    ) -> None:
        self._index = RegionIndex(regions)
        self._memberships: dict[ObjectId, dict[RegionId, RegionMembership]] = {}
        self._method = method

    @property
    def index(self) -> RegionIndex:
        return self._index

    @property
    def method(self) -> ContainmentMethod:
        return self._method

    def set_regions(self, regions: tuple[Region, ...], *, now: Instant) -> tuple[RegionTransition, ...]:
        """Adopt new geometry.

        Section M7: *"Existing dwell accumulations are closed out and new ones
        opened against the new region version; both are published with their
        version."* Carrying an accumulation across a geometry change would
        attribute time spent in the old shape to the new one.
        """
        closed: list[RegionTransition] = []
        for object_id, by_region in self._memberships.items():
            for membership in by_region.values():
                closed.append(
                    RegionTransition(
                        object_id=object_id,
                        region_id=membership.region_id,
                        geometry_version=membership.geometry_version,
                        entered=False,
                        at=now,
                        dwell=membership.dwell(now),
                        method=membership.method,
                    )
                )
        self._memberships.clear()
        self._index.rebuild(regions)
        return tuple(closed)

    def update(
        self, object_id: ObjectId, box: Box, *, at: Instant
    ) -> tuple[RegionTransition, ...]:
        """Recompute one object's membership. Returns entries and exits.

        ``at`` is capture time. Passing processing time here would silently make
        every dwell a measurement of the platform rather than of the world.
        """
        point = containment_point(box, self._method)
        inside = {r.region_id: r for r in self._index.containing(point)}
        previous = self._memberships.get(object_id, {})
        transitions: list[RegionTransition] = []

        for region_id, region in inside.items():
            existing = previous.get(region_id)
            if existing is None or existing.geometry_version != region.version:
                if existing is not None:
                    transitions.append(
                        RegionTransition(
                            object_id=object_id,
                            region_id=region_id,
                            geometry_version=existing.geometry_version,
                            entered=False,
                            at=at,
                            dwell=existing.dwell(at),
                            method=existing.method,
                        )
                    )
                transitions.append(
                    RegionTransition(
                        object_id=object_id,
                        region_id=region_id,
                        geometry_version=region.version,
                        entered=True,
                        at=at,
                        method=self._method,
                    )
                )

        current: dict[RegionId, RegionMembership] = {}
        for region_id, region in inside.items():
            existing = previous.get(region_id)
            entered_at = (
                existing.entered_at
                if existing is not None and existing.geometry_version == region.version
                else at
            )
            current[region_id] = RegionMembership(
                region_id=region_id,
                geometry_version=region.version,
                state=MembershipState.INSIDE,
                method=self._method,
                entered_at=entered_at,
                last_confirmed=at,
            )

        for region_id, membership in previous.items():
            if region_id in inside:
                continue
            transitions.append(
                RegionTransition(
                    object_id=object_id,
                    region_id=region_id,
                    geometry_version=membership.geometry_version,
                    entered=False,
                    at=at,
                    dwell=membership.dwell(at),
                    method=membership.method,
                )
            )

        if current:
            self._memberships[object_id] = current
        else:
            self._memberships.pop(object_id, None)
        return tuple(transitions)

    def forget(self, object_id: ObjectId, *, at: Instant) -> tuple[RegionTransition, ...]:
        """Close out an object's memberships when it leaves the population."""
        by_region = self._memberships.pop(object_id, {})
        return tuple(
            RegionTransition(
                object_id=object_id,
                region_id=membership.region_id,
                geometry_version=membership.geometry_version,
                entered=False,
                at=at,
                dwell=membership.dwell(at),
                method=membership.method,
            )
            for membership in by_region.values()
        )

    def membership(self, object_id: ObjectId) -> dict[RegionId, RegionMembership]:
        return dict(self._memberships.get(object_id, {}))

    def occupancy(
        self, *, classes: dict[ObjectId, str], now: Instant
    ) -> tuple[RegionOccupancy, ...]:
        """Per-region counts and descriptive dwell statistics.

        ``classes`` maps object to class id; the caller supplies it because the
        region tracker holds geometry, not objects.
        """
        by_region: dict[RegionId, list[tuple[ObjectId, RegionMembership]]] = {}
        for object_id, memberships in self._memberships.items():
            for region_id, membership in memberships.items():
                by_region.setdefault(region_id, []).append((object_id, membership))

        reports: list[RegionOccupancy] = []
        for region in self._index.regions:
            present = by_region.get(region.region_id, [])
            counts: dict[str, int] = {}
            dwells: list[int] = []
            for object_id, membership in present:
                class_id = classes.get(object_id, "unknown")
                counts[class_id] = counts.get(class_id, 0) + 1
                dwells.append(membership.dwell(now).ns)
            reports.append(
                RegionOccupancy(
                    region_id=region.region_id,
                    geometry_version=region.version,
                    occupancy=counts,
                    present_objects=tuple(sorted(o for o, _ in present)),
                    dwell_current_max=Duration(max(dwells) if dwells else 0),
                    dwell_current_mean=Duration(
                        sum(dwells) // len(dwells) if dwells else 0
                    ),
                    last_transition=now,
                )
            )
        return tuple(reports)

    @property
    def tracked_objects(self) -> int:
        return len(self._memberships)
