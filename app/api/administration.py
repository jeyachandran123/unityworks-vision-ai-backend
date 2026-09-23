"""The administration API — zones, and who may see them.

Three groups, and a deliberate asymmetry between them.

### Zones are fully writable

They are organizational structure: the name of an area of a kitchen.
Getting one wrong is an inconvenience, and the blast radius of a mistake is a
mislabelled row.

Each domain is gated on its own permission — `VIEW_ZONES` / `MANAGE_ZONES` for
restaurants, `VIEW_ZONES` / `MANAGE_ZONES` for zones — and each write is
audited, because renaming the site an incident is attributed to changes how that
incident reads six months later.

These were once gated on `VIEW_USERS` for reads and `MANAGE_ORGANIZATION` for
writes, which made two unrelated questions the same grant: "may see who works
here" also meant "may read every site", and "may rename a zone" also meant "may
reconfigure the organization". The product needs those separated — two managers
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
from sqlalchemy import select

from app.api.dependencies import CurrentAccess, DbSession, requires
from app.authorization.model import AccessDecision, Permission
from app.domain import zones as zone_domain
from app.domain.audit import AuditAction, AuditTrail
from app.errors import ValidationError
from app.users.models import OrganizationMembership, RoleAssignment, User

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
    organization is a question for a person, not something to paper over with
    `-2`.
    """
    kept = [c.lower() if c.isalnum() else "-" for c in name.strip()]
    slug = "".join(kept)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:128]


# ── zones ────────────────────────────────────────────────────────────────────
#
# The estate's one placement level. Sites were folded into zones on 2026-09-23
# (`b4c8e1a37d90`): a zone belongs to the organization and holds cameras, which
# is how the people running a restaurant describe where a camera is.


@router.get("/zones", dependencies=[Depends(requires(Permission.VIEW_ZONES))])
async def list_zones(
    access: CurrentAccess,
    session: DbSession,
    q: Annotated[str | None, Query(max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    """Zones in the caller's organization, with how many cameras each holds.

    Gated on `VIEW_ZONES`: somebody reading incidents needs to know the places
    they are attributed to, and reading grants no ability to change them.
    """
    service = zone_domain.ZoneService(session)
    zones, total = await service.list(
        organization_id=access.tenant_id, q=q or "", limit=limit, offset=offset
    )
    counts = await service.camera_counts(organization_id=access.tenant_id)
    return {
        "zones": [zone_domain.to_wire(z, camera_count=counts.get(z.id, 0)) for z in zones],
        "count": len(zones),
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/zones/{zone_id}", dependencies=[Depends(requires(Permission.VIEW_ZONES))])
async def get_zone(zone_id: str, access: CurrentAccess, session: DbSession) -> dict[str, Any]:
    service = zone_domain.ZoneService(session)
    zone = await service.get(organization_id=access.tenant_id, zone_id=zone_id)
    counts = await service.camera_counts(organization_id=access.tenant_id)
    return zone_domain.to_wire(zone, camera_count=counts.get(zone.id, 0))


@router.post("/zones", dependencies=[Depends(requires(Permission.MANAGE_ZONES))])
async def create_zone(
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Create a zone: a name, and nothing else to decide.

    Everything a zone needs is its name — kitchen, dine hall, billing. The
    organization comes from the session, never from the request.
    """
    zone = await zone_domain.ZoneService(session).create(
        organization_id=access.tenant_id, name=_text(payload, "name", required=True)
    )
    await AuditTrail(session).record(
        action=AuditAction.ZONE_CREATED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="zone",
        resource_id=zone.id,
        request_id=_request_id(request),
        detail={"name": zone.name},
    )
    return zone_domain.to_wire(zone, camera_count=0)


@router.patch("/zones/{zone_id}", dependencies=[Depends(requires(Permission.MANAGE_ZONES))])
async def update_zone(
    zone_id: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Rename a zone, or take it out of use.

    Renaming does not rewrite history: `CameraZoneAssignment` froze the name a
    past reading was labelled with, deliberately.
    """
    service = zone_domain.ZoneService(session)
    zone = await service.get(organization_id=access.tenant_id, zone_id=zone_id)
    before = {"name": zone.name, "is_active": zone.is_active}

    if "name" in payload:
        zone = await service.rename(
            organization_id=access.tenant_id, zone_id=zone_id, name=_text(payload, "name")
        )
    if "is_active" in payload:
        zone = await service.set_active(
            organization_id=access.tenant_id,
            zone_id=zone_id,
            active=bool(payload.get("is_active")),
        )

    await AuditTrail(session).record(
        action=AuditAction.ZONE_UPDATED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="zone",
        resource_id=zone.id,
        request_id=_request_id(request),
        detail={"before": before, "after": {"name": zone.name, "is_active": zone.is_active}},
    )
    counts = await service.camera_counts(organization_id=access.tenant_id)
    return zone_domain.to_wire(zone, camera_count=counts.get(zone.id, 0))


@router.delete("/zones/{zone_id}", dependencies=[Depends(requires(Permission.MANAGE_ZONES))])
async def delete_zone(
    zone_id: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
) -> dict[str, Any]:
    """Delete a zone that holds no cameras.

    A zone with cameras is refused with the count and the alternative, because
    it is part of the record of what was watched and where. Deactivating keeps
    that record; deleting would lose it.
    """
    zone = await zone_domain.ZoneService(session).delete(
        organization_id=access.tenant_id, zone_id=zone_id
    )
    await AuditTrail(session).record(
        action=AuditAction.ZONE_DELETED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="zone",
        resource_id=zone_id,
        request_id=_request_id(request),
        detail={"name": zone.name},
    )
    return {"zone_id": zone_id, "deleted": True}


@router.get("/users", dependencies=[Depends(requires(Permission.VIEW_USERS))])
async def list_users(access: CurrentAccess, session: DbSession) -> dict[str, Any]:
    """Who holds which role in this organization.

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
                .where(
                    # Members, not accounts created here: one person may hold
                    # several organizations.
                    User.id.in_(
                        select(OrganizationMembership.user_id).where(
                            OrganizationMembership.organization_id == access.tenant_id
                        )
                    )
                )
                .order_by(User.email)
            )
        )
        .scalars()
        .all()
    )

    assignments = (
        await session.execute(
            select(RoleAssignment.user_id, RoleAssignment.role).where(
                RoleAssignment.organization_id == access.tenant_id
            )
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
                "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
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
