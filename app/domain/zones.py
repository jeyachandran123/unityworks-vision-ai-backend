"""Zones — the estate's one placement level.

A zone belongs to an organization and holds cameras. It replaced the site in
2026-09-23: the estate was organization → site → zone → camera, and the people
running a restaurant describe where a camera is in one step, not two.

### Deleting versus deactivating

A zone that has never held a camera is configuration, and deleting it loses
nothing. A zone that holds cameras is part of the record of what was watched and
where, so it deactivates instead — the same rule the rest of the product
follows, stated once here and enforced in `delete()` rather than in a route that
could forget it.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import Camera, Zone
from app.errors import ConflictError, NotFoundError


class ZoneService:
    """Every query here is constructed already narrowed to one organization."""

    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list(
        self,
        *,
        organization_id: str,
        q: str = "",
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[list[Zone], int]:
        """One page of zones, and how many there are in total.

        Paginated from the start. A restaurant with five zones does not need it;
        the same code with five hundred does, and the version that fetches every
        row works right up until the day it does not.
        """
        statement = select(Zone).where(Zone.organization_id == organization_id)
        if q.strip():
            statement = statement.where(func.lower(Zone.name).like(f"%{q.strip().lower()}%"))

        total = int(
            (
                await self._session.execute(select(func.count()).select_from(statement.subquery()))
            ).scalar_one()
        )
        rows = (
            (await self._session.execute(statement.order_by(Zone.name).limit(limit).offset(offset)))
            .scalars()
            .all()
        )
        return list(rows), total

    async def camera_counts(self, *, organization_id: str) -> dict[str, int]:
        """How many cameras each zone holds, in one query."""
        rows = (
            await self._session.execute(
                select(Camera.zone_id, func.count())
                .where(Camera.organization_id == organization_id)
                .group_by(Camera.zone_id)
            )
        ).all()
        return {zone_id: int(count) for zone_id, count in rows if zone_id}

    async def get(self, *, organization_id: str, zone_id: str) -> Zone:
        """A zone in this organization, or `NotFoundError`.

        Another organization's zone is *not found* rather than forbidden:
        existence across a tenant boundary is itself a disclosure.
        """
        zone = (
            await self._session.execute(
                select(Zone).where(Zone.id == zone_id, Zone.organization_id == organization_id)
            )
        ).scalar_one_or_none()
        if zone is None:
            raise NotFoundError(f"no zone '{zone_id}'")
        return zone

    async def create(self, *, organization_id: str, name: str) -> Zone:
        cleaned = name.strip()
        if not cleaned:
            raise ConflictError("a zone needs a name")
        existing = (
            await self._session.execute(
                select(Zone.id).where(
                    Zone.organization_id == organization_id,
                    func.lower(Zone.name) == cleaned.lower(),
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            raise ConflictError(
                f"this organization already has a zone called '{cleaned}'. Two zones "
                "with one name cannot be told apart when placing a camera."
            )
        zone = Zone(organization_id=organization_id, name=cleaned, is_active=True)
        self._session.add(zone)
        await self._session.flush()
        return zone

    async def rename(self, *, organization_id: str, zone_id: str, name: str) -> Zone:
        zone = await self.get(organization_id=organization_id, zone_id=zone_id)
        cleaned = name.strip()
        if cleaned:
            zone.name = cleaned
        return zone

    async def set_active(self, *, organization_id: str, zone_id: str, active: bool) -> Zone:
        zone = await self.get(organization_id=organization_id, zone_id=zone_id)
        zone.is_active = active
        return zone

    async def delete(self, *, organization_id: str, zone_id: str) -> Zone:
        """Remove a zone that has never held a camera.

        Refused otherwise, and the refusal says how many cameras are in the way
        and what to do about them. Deleting a zone with cameras would either
        orphan them or delete them silently, and both are worse than an answer.
        """
        zone = await self.get(organization_id=organization_id, zone_id=zone_id)
        held = int(
            (
                await self._session.execute(
                    select(func.count()).select_from(Camera).where(Camera.zone_id == zone_id)
                )
            ).scalar_one()
        )
        if held:
            raise ConflictError(
                f"'{zone.name}' holds {held} camera(s). Move them to another zone, or "
                "retire them, before deleting it. Deactivating the zone keeps its "
                "cameras and their history.",
                details={"zone_id": zone_id, "camera_count": held},
            )
        await self._session.delete(zone)
        return zone


def to_wire(zone: Zone, *, camera_count: int = 0) -> dict[str, Any]:
    return {
        "id": zone.id,
        "name": zone.name,
        "is_active": bool(zone.is_active),
        "camera_count": camera_count,
        "created_at": zone.created_at.isoformat() if zone.created_at else None,
    }


__all__ = ["ZoneService", "to_wire"]
