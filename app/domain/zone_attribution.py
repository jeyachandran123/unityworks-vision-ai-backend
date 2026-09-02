"""Where a past event happened, answered from the past rather than the present.

`cameras.zone_id` is current state. Every durable record in this application
that describes something a camera saw — an observation in Vision OS's log, an
`Incident`, an `EvidenceRecord`, a `FrameRecord` — carries a `camera_key` and a
time, and none of them carries a zone. The obvious way to add one is to join
`cameras.zone_id` at read time, and it is wrong: reassign a camera and every
reading it has ever produced silently moves with it.

`CameraZoneAssignment` records the mapping as a series of closed intervals
instead, so the question "which zone was camera 3 in at 14:32 last Tuesday" has
an answer that does not change when camera 3 moves tomorrow.

### What "no answer" means here

`resolve` returns `None` for an instant with no covering interval, and callers
must render that as *unrecorded* rather than as a zone or as an empty string.
Intervals begin the first time a camera's zone is written after this table
existed; nothing before that point was recorded, and inferring it from today's
mapping is exactly the error being avoided. A honest gap is the correct output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import CameraZoneAssignment, Zone


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes. Compare in UTC or not at all."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ZoneAttribution:
    """Where something happened, as recorded at the time it happened."""

    zone_id: str | None
    zone_name: str
    restaurant_id: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "restaurant_id": self.restaurant_id,
        }


async def record_assignment(
    session: AsyncSession,
    *,
    organization_id: str,
    camera_key: str,
    zone_id: str | None,
    restaurant_id: str | None = None,
    assigned_by: str = "",
    at: datetime | None = None,
) -> CameraZoneAssignment | None:
    """Open an interval for this camera's zone, closing any interval in force.

    Returns the new row, or `None` when the camera was already assigned to this
    zone — repeating an assignment is not a move, and a second interval for the
    same zone would make the history look like one.

    The zone's **name** is captured here rather than joined later, for the same
    reason the zone id is: renaming "Prep line" to "Prep line 1" must not
    relabel a year of history.
    """
    moment = at or _now()

    open_row = (
        await session.execute(
            select(CameraZoneAssignment)
            .where(
                CameraZoneAssignment.organization_id == organization_id,
                CameraZoneAssignment.camera_key == camera_key,
                CameraZoneAssignment.effective_to.is_(None),
            )
            .order_by(CameraZoneAssignment.effective_from.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    if open_row is not None and open_row.zone_id == zone_id:
        return None

    zone_name = ""
    if zone_id:
        zone = (
            await session.execute(select(Zone).where(Zone.id == zone_id))
        ).scalar_one_or_none()
        if zone is not None:
            zone_name = zone.name
            restaurant_id = restaurant_id or zone.restaurant_id

    if open_row is not None:
        # Closed, never rewritten. The old row keeps its zone forever, which is
        # the whole point of the table.
        open_row.effective_to = moment

    assignment = CameraZoneAssignment(
        organization_id=organization_id,
        camera_key=camera_key,
        zone_id=zone_id,
        zone_name=zone_name,
        restaurant_id=restaurant_id,
        effective_from=moment,
        assigned_by=assigned_by,
    )
    session.add(assignment)
    return assignment


class ZoneHistory:
    """Every recorded interval for a set of cameras, loaded once.

    Built for a page that resolves hundreds of observations: one query, then an
    in-memory interval lookup, rather than a query per row.
    """

    __slots__ = ("_by_camera",)

    def __init__(self, rows: list[CameraZoneAssignment]) -> None:
        by_camera: dict[str, list[CameraZoneAssignment]] = {}
        for row in rows:
            by_camera.setdefault(row.camera_key, []).append(row)
        for intervals in by_camera.values():
            intervals.sort(key=lambda r: _aware(r.effective_from) or datetime.min.replace(tzinfo=UTC))
        self._by_camera = by_camera

    @classmethod
    async def load(
        cls,
        session: AsyncSession,
        *,
        organization_id: str,
        camera_keys: tuple[str, ...],
    ) -> ZoneHistory:
        if not camera_keys:
            return cls([])
        rows = (
            (
                await session.execute(
                    select(CameraZoneAssignment).where(
                        CameraZoneAssignment.organization_id == organization_id,
                        CameraZoneAssignment.camera_key.in_(camera_keys),
                    )
                )
            )
            .scalars()
            .all()
        )
        return cls(list(rows))

    def resolve(self, camera_key: str, when: datetime | None) -> ZoneAttribution | None:
        """The zone in force for this camera at this instant, or `None`.

        `None` is returned for an unknown camera, for an instant with no
        covering interval, and for a caller that has no time to ask about —
        three different ways of not knowing, all of which must render as
        *unrecorded* rather than as a zone.
        """
        if when is None:
            return None
        intervals = self._by_camera.get(camera_key)
        if not intervals:
            return None

        moment = _aware(when)
        if moment is None:  # pragma: no cover - `when` is aware by construction
            return None

        for interval in intervals:
            start = _aware(interval.effective_from)
            end = _aware(interval.effective_to)
            if start is not None and moment < start:
                continue
            # Half-open: an assignment made at 14:00 covers 14:00 and the
            # closing instant belongs to the interval that follows it.
            if end is not None and moment >= end:
                continue
            return ZoneAttribution(
                zone_id=interval.zone_id,
                zone_name=interval.zone_name,
                restaurant_id=interval.restaurant_id,
            )
        return None

    def resolve_ns(self, camera_key: str, captured_ns: int | None) -> ZoneAttribution | None:
        """`resolve`, for the nanosecond instants the platform reports time in."""
        if captured_ns is None:
            return None
        return self.resolve(camera_key, datetime.fromtimestamp(captured_ns / 1_000_000_000, tz=UTC))


__all__ = ["ZoneAttribution", "ZoneHistory", "record_assignment"]
