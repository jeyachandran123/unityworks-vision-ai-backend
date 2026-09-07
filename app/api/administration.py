"""The administration API — restaurants, zones, and who may see them.

Three groups, and a deliberate asymmetry between them.

### Restaurants and zones are fully writable

They are organisational structure: a name, a timezone, an area of a kitchen.
Getting one wrong is an inconvenience, and the blast radius of a mistake is a
mislabelled row.

Each domain is gated on its own permission — `VIEW_SITES` / `MANAGE_SITES` for
restaurants, `VIEW_ZONES` / `MANAGE_ZONES` for zones — and each write is
audited, because renaming the site an incident is attributed to changes how that
incident reads six months later.

These were once gated on `VIEW_USERS` for reads and `MANAGE_ORGANIZATION` for
writes, which made two unrelated questions the same grant: "may see who works
here" also meant "may read every site", and "may rename a zone" also meant "may
reconfigure the organisation". The product needs those separated — two managers
holding one role, where one may edit the estate and the other may only read
it — and no arrangement of a blanket permission expresses that.

### Users are read-only here, and that is a decision rather than an omission

Listing who holds which role is administration. *Creating* an account is
identity: it mints a credential, and every safe way to do that — an invitation
with a signed single-use token, a password-reset channel, an SSO assertion —
needs a delivery mechanism this backend does not yet have. `app/auth` hashes
passwords and issues tokens; it has no route that provisions a user, and the
only shape that would fit in this phase is an admin-sets-a-password form, which
puts a plaintext credential in a request body and in an admin's clipboard.

So this module reads users and stops. The frontend renders the same boundary:
the list is real, and the write path says plainly that it is not connected. A
half-built invite flow would be worse than an honest absence, because the half
that is missing is the half that keeps the credential secret.

### Scoping

Every query is constructed already narrowed to the caller's tenant, the same
discipline `product.py` documents. Nothing here accepts an organization id from
the request — tenancy comes from the authenticated session and nowhere else.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import CurrentAccess, DbSession, requires
from app.authorization.model import AccessDecision, Permission
from app.domain.audit import AuditAction, AuditTrail
from app.domain.models import Camera, Restaurant, Zone
from app.errors import NotFoundError, ValidationError
from app.users.models import RoleAssignment, User

router = APIRouter(prefix="/api/v1", tags=["administration"])


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def _roles(access: AccessDecision) -> tuple[str, ...]:
    return tuple(sorted(r.value for r in access.roles))


def _text(payload: dict[str, Any], key: str, *, required: bool = False) -> str:
    value = str(payload.get(key, "") or "").strip()
    if required and not value:
        raise ValidationError(f"'{key}' is required")
    return value


def _slugify(name: str) -> str:
    """A URL-safe slug from a name. Not clever, and deliberately not unique-ified.

    A collision raises through the table's own unique constraint rather than
    being silently suffixed: two restaurants called the same thing in one
    organisation is a question for a person, not something to paper over with
    `-2`.
    """
    kept = [c.lower() if c.isalnum() else "-" for c in name.strip()]
    slug = "".join(kept)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:128]


# ── Restaurants ──────────────────────────────────────────────────────────────


def restaurant_to_wire(restaurant: Restaurant, *, zone_count: int, camera_count: int) -> dict[str, Any]:
    return {
        "id": restaurant.id,
        "name": restaurant.name,
        "slug": restaurant.slug,
        "timezone": restaurant.timezone,
        "is_active": bool(restaurant.is_active),
        "created_at": restaurant.created_at.isoformat() if restaurant.created_at else None,
        # Counts rather than nested collections: an administration list wants to
        # know a site has four zones, not to carry all four on every row.
        "zone_count": zone_count,
        "camera_count": camera_count,
    }


async def _counts(session: AsyncSession, organization_id: str) -> tuple[dict[str, int], dict[str, int]]:
    """Zone and camera counts per restaurant, in two queries rather than 2N."""
    zones = (
        await session.execute(
            select(Zone.restaurant_id).join(
                Restaurant, Restaurant.id == Zone.restaurant_id
            ).where(Restaurant.organization_id == organization_id)
        )
    ).scalars().all()
    cameras = (
        await session.execute(
            select(Camera.restaurant_id).where(Camera.organization_id == organization_id)
        )
    ).scalars().all()

    zone_counts: dict[str, int] = {}
    for restaurant_id in zones:
        zone_counts[restaurant_id] = zone_counts.get(restaurant_id, 0) + 1
    camera_counts: dict[str, int] = {}
    for restaurant_id in cameras:
        if restaurant_id:
            camera_counts[restaurant_id] = camera_counts.get(restaurant_id, 0) + 1
    return zone_counts, camera_counts


@router.get("/restaurants", dependencies=[Depends(requires(Permission.VIEW_SITES))])
async def list_restaurants(
    access: CurrentAccess,
    session: DbSession,
    q: Annotated[str | None, Query(max_length=200)] = None,
    is_active: Annotated[bool | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    """Sites in the caller's organisation.

    Gated on `VIEW_SITES`: a restaurant manager needs to read the structure
    their incidents are attributed to, and reading it grants no ability to
    change it. Writing is `MANAGE_SITES`, a permission this role does not hold
    by default.

    Searchable and paginated. A chain with four restaurants does not need it;
    the same code with four hundred does, and the version that fetches all of
    them works perfectly right up until the day it does not.
    """
    statement = select(Restaurant).where(Restaurant.organization_id == access.tenant_id)
    if q:
        needle = f"%{q.strip().lower()}%"
        statement = statement.where(
            func.lower(Restaurant.name).like(needle)
            | func.lower(Restaurant.slug).like(needle)
        )
    if is_active is not None:
        statement = statement.where(Restaurant.is_active.is_(is_active))

    total = int(
        (await session.execute(select(func.count()).select_from(statement.subquery()))).scalar_one()
    )
    found = (
        (
            await session.execute(
                statement.order_by(Restaurant.name).limit(limit).offset(offset)
            )
        )
        .scalars()
        .all()
    )
    zone_counts, camera_counts = await _counts(session, access.tenant_id)
    return {
        "restaurants": [
            restaurant_to_wire(
                r,
                zone_count=zone_counts.get(r.id, 0),
                camera_count=camera_counts.get(r.id, 0),
            )
            for r in found
        ],
        "count": len(found),
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/restaurants/{restaurant_id}", dependencies=[Depends(requires(Permission.VIEW_SITES))])
async def get_restaurant(
    restaurant_id: str, access: CurrentAccess, session: DbSession
) -> dict[str, Any]:
    """One site, for its own page.

    A detail route rather than "find it in the list": a site page is now a real
    surface with zones and cameras of its own, and asking a client to fetch
    every site to render one is how a list becomes a hidden dependency of a
    detail view.
    """
    restaurant = await _restaurant_in_tenant(session, access.tenant_id, restaurant_id)
    zone_counts, camera_counts = await _counts(session, access.tenant_id)
    return restaurant_to_wire(
        restaurant,
        zone_count=zone_counts.get(restaurant.id, 0),
        camera_count=camera_counts.get(restaurant.id, 0),
    )


@router.post("/restaurants", dependencies=[Depends(requires(Permission.MANAGE_SITES))])
async def create_restaurant(
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    name = _text(payload, "name", required=True)
    slug = _slugify(_text(payload, "slug") or name)
    if not slug:
        raise ValidationError("'name' must contain at least one alphanumeric character")

    restaurant = Restaurant(
        # Never from the request. Tenancy is identity, not a field.
        organization_id=access.tenant_id,
        name=name,
        slug=slug,
        timezone=_text(payload, "timezone") or "UTC",
        is_active=bool(payload.get("is_active", True)),
    )
    session.add(restaurant)
    await session.flush()

    await AuditTrail(session).record(
        action=AuditAction.RESTAURANT_CREATED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="restaurant",
        resource_id=restaurant.id,
        request_id=_request_id(request),
        detail={"name": restaurant.name, "slug": restaurant.slug},
    )
    return restaurant_to_wire(restaurant, zone_count=0, camera_count=0)


@router.patch(
    "/restaurants/{restaurant_id}",
    dependencies=[Depends(requires(Permission.MANAGE_SITES))],
)
async def update_restaurant(
    restaurant_id: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    restaurant = await _restaurant_in_tenant(session, access.tenant_id, restaurant_id)

    changed: list[str] = []
    if "name" in payload:
        restaurant.name = _text(payload, "name", required=True)
        changed.append("name")
    if "timezone" in payload:
        restaurant.timezone = _text(payload, "timezone") or "UTC"
        changed.append("timezone")
    if "is_active" in payload:
        restaurant.is_active = bool(payload["is_active"])
        changed.append("is_active")
    # `slug` is deliberately not editable: it is the stable handle other rows
    # and URLs are formed from, and renaming it silently orphans them.

    await session.flush()
    zone_counts, camera_counts = await _counts(session, access.tenant_id)

    await AuditTrail(session).record(
        action=AuditAction.RESTAURANT_UPDATED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="restaurant",
        resource_id=restaurant.id,
        request_id=_request_id(request),
        detail={"fields": sorted(changed)},
    )
    return restaurant_to_wire(
        restaurant,
        zone_count=zone_counts.get(restaurant.id, 0),
        camera_count=camera_counts.get(restaurant.id, 0),
    )


async def _restaurant_in_tenant(
    session: AsyncSession, organization_id: str, restaurant_id: str
) -> Restaurant:
    """Fetch already narrowed to the tenant.

    A tenant mismatch is a 404 rather than a 403: telling a caller that a
    restaurant exists but belongs to someone else is itself a disclosure.
    """
    found = (
        await session.execute(
            select(Restaurant).where(
                Restaurant.id == restaurant_id,
                Restaurant.organization_id == organization_id,
            )
        )
    ).scalar_one_or_none()
    if found is None:
        raise NotFoundError(f"no restaurant '{restaurant_id}'")
    return found


# ── Zones ────────────────────────────────────────────────────────────────────


def zone_to_wire(zone: Zone, *, camera_count: int = 0) -> dict[str, Any]:
    return {
        "id": zone.id,
        "restaurant_id": zone.restaurant_id,
        "name": zone.name,
        "created_at": zone.created_at.isoformat() if zone.created_at else None,
        "camera_count": camera_count,
    }


@router.get("/zones", dependencies=[Depends(requires(Permission.VIEW_ZONES))])
async def list_zones(
    access: CurrentAccess,
    session: DbSession,
    restaurant_id: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    """Zones, optionally for one site. Always narrowed to the caller's tenant.

    The join onto `Restaurant` is what enforces tenancy: `zones` carries no
    organization column of its own, so filtering on the parent is the only
    construction that cannot leak another organisation's areas.
    """
    statement = (
        select(Zone)
        .join(Restaurant, Restaurant.id == Zone.restaurant_id)
        .where(Restaurant.organization_id == access.tenant_id)
        .order_by(Zone.name)
    )
    if restaurant_id:
        statement = statement.where(Zone.restaurant_id == restaurant_id)

    found = (await session.execute(statement)).scalars().all()

    camera_rows = (
        await session.execute(
            select(Camera.zone_id).where(Camera.organization_id == access.tenant_id)
        )
    ).scalars().all()
    per_zone: dict[str, int] = {}
    for zone_id in camera_rows:
        if zone_id:
            per_zone[zone_id] = per_zone.get(zone_id, 0) + 1

    return {
        "zones": [zone_to_wire(z, camera_count=per_zone.get(z.id, 0)) for z in found],
        "count": len(found),
    }


@router.post("/zones", dependencies=[Depends(requires(Permission.MANAGE_ZONES))])
async def create_zone(
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    restaurant_id = _text(payload, "restaurant_id", required=True)
    # Checked before insert so a zone can never be attached to another
    # organisation's restaurant by naming its id.
    await _restaurant_in_tenant(session, access.tenant_id, restaurant_id)

    zone = Zone(restaurant_id=restaurant_id, name=_text(payload, "name", required=True))
    session.add(zone)
    await session.flush()

    await AuditTrail(session).record(
        action=AuditAction.ZONE_CREATED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="zone",
        resource_id=zone.id,
        request_id=_request_id(request),
        detail={"name": zone.name, "restaurant_id": restaurant_id},
    )
    return zone_to_wire(zone)


@router.patch("/zones/{zone_id}", dependencies=[Depends(requires(Permission.MANAGE_ZONES))])
async def update_zone(
    zone_id: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    zone = (
        await session.execute(
            select(Zone)
            .join(Restaurant, Restaurant.id == Zone.restaurant_id)
            .where(Zone.id == zone_id, Restaurant.organization_id == access.tenant_id)
        )
    ).scalar_one_or_none()
    if zone is None:
        raise NotFoundError(f"no zone '{zone_id}'")

    if "name" in payload:
        zone.name = _text(payload, "name", required=True)
    await session.flush()

    await AuditTrail(session).record(
        action=AuditAction.ZONE_UPDATED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="zone",
        resource_id=zone.id,
        request_id=_request_id(request),
        detail={"name": zone.name},
    )
    return zone_to_wire(zone)


# ── Users ────────────────────────────────────────────────────────────────────


@router.get("/users", dependencies=[Depends(requires(Permission.VIEW_USERS))])
async def list_users(access: CurrentAccess, session: DbSession) -> dict[str, Any]:
    """Who holds which role in this organisation.

    **Read-only, and no credential material of any kind.** `password_hash` is
    never selected, never rendered, and is not part of this response shape — a
    hash is not a password but it is still the thing an offline attack is run
    against, and an administration screen has no use for one.

    `write_available` is `false` and says why. The frontend shows the same
    sentence rather than a disabled button with no explanation.
    """
    users = (
        (
            await session.execute(
                select(User)
                .where(User.organization_id == access.tenant_id)
                .order_by(User.email)
            )
        )
        .scalars()
        .all()
    )

    assignments = (
        await session.execute(
            select(RoleAssignment.user_id, RoleAssignment.role)
            .join(User, User.id == RoleAssignment.user_id)
            .where(User.organization_id == access.tenant_id)
        )
    ).all()
    roles_by_user: dict[str, list[str]] = {}
    for user_id, role in assignments:
        roles_by_user.setdefault(user_id, []).append(role)

    return {
        "users": [
            {
                "id": user.id,
                "email": user.email,
                "display_name": user.display_name,
                "is_active": bool(user.is_active),
                "roles": sorted(roles_by_user.get(user.id, [])),
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "last_login_at": user.last_login_at.isoformat()
                if user.last_login_at
                else None,
            }
            for user in users
        ],
        "count": len(users),
        # Stated in the payload rather than assumed by the client, so the reason
        # travels with the capability and one place decides it.
        "write_available": False,
        "write_unavailable_reason": (
            "Creating an account issues a credential, and this deployment has no "
            "invitation or password-reset delivery channel yet. Accounts are "
            "provisioned directly until one exists."
        ),
    }


__all__ = ["router"]
