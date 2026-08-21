"""The Phase 1 HTTP surface.

Four groups, and no product endpoints:

    /health, /health/ready     unauthenticated liveness and readiness
    /api/v1/auth/*             login, refresh, me
    /api/v1/status             operator-facing status, authenticated
    /api/v1/devtools/*         engineering surface, permission-gated

There is no restaurant, camera, incident, notification or report route. Those
arrive in Phase 4 with the domain they describe; a route created before its
feature is an API contract nobody has thought through.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Request, Response

from app.auth.cookies import clear_refresh_cookie, read_refresh_cookie, set_refresh_cookie
from app.api.devtools import router as devtools_router
from app.api.dependencies import (
    CurrentAccess,
    DbSession,
    auth_of,
    cache_of,
    database_of,
    requires,
    settings_of,
    vision_of,
)
from app.authorization.model import Permission
from app.errors import AuthenticationError

# ── health ───────────────────────────────────────────────────────────────────
#
# Unauthenticated, and therefore deliberately uninformative. A health endpoint
# that reports versions, component names or camera counts is reconnaissance for
# anyone who can reach the port.

health_router = APIRouter(tags=["health"])


@health_router.get("/health")
async def liveness() -> dict[str, str]:
    """The process is up. Nothing more is claimed and nothing more is disclosed."""
    return {"status": "ok"}


@health_router.get("/health/ready")
async def readiness(request: Request, response: Response) -> dict[str, Any]:
    """Whether backing dependencies answer. Booleans only, never detail."""
    database = database_of(request)
    cache = cache_of(request)
    vision = vision_of(request)

    db_ok, _ = await database.healthy()
    cache_ok, _ = await cache.healthy()

    # Vision OS is reported but does not gate readiness in Phase 1: no source
    # adapter is bound yet, so an unassembled platform is the expected state
    # rather than a fault.
    ready = db_ok and cache_ok
    response.status_code = 200 if ready else 503
    return {
        "ready": ready,
        "database": db_ok,
        "cache": cache_ok,
        "vision_os": vision.assembled,
    }


# ── authentication ───────────────────────────────────────────────────────────

auth_router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@auth_router.post("/login")
async def login(
    request: Request,
    response: Response,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Exchange an email and password for a session.

    **The refresh token is never in the body.** It is set as an httpOnly,
    Secure, SameSite=Strict cookie scoped to `/api/v1/auth`, so page JavaScript
    cannot read it and no other request carries it. The access token is returned
    in the body deliberately: it is short-lived and belongs in memory, where a
    page reload discards it.
    """
    auth = auth_of(request)
    email = str(payload.get("email", "")).strip()
    password = str(payload.get("password", ""))

    decision = await auth.authenticate(session, email=email, password=password)
    issued = auth.issue(decision)

    set_refresh_cookie(response, issued.refresh_token, settings_of(request))

    return {
        "access_token": issued.access_token,
        "token_type": "bearer",
        "expires_at": issued.expires_at.isoformat(),
        "user": _identity(decision),
    }


@auth_router.post("/refresh")
async def refresh(
    request: Request,
    response: Response,
    session: DbSession,
) -> dict[str, Any]:
    """Trade the refresh cookie for a new access token, and rotate the cookie.

    Two properties:

    **The decision is rebuilt from the database**, not read from the refresh
    token — so a role revoked five minutes ago is not reissued for another
    fifteen.

    **The refresh token is rotated on every use.** A token that has been
    exchanged is no longer held by the client, which turns a stolen-and-replayed
    token into a visible anomaly rather than a silent second session.
    """
    from app.auth.service import decision_for_claims

    auth = auth_of(request)
    token = read_refresh_cookie(request)
    if not token:
        # "No session" rather than "bad session". A client that was simply never
        # logged in should go to the login screen, not be told its credential
        # failed.
        raise AuthenticationError("no active session")

    claims = auth.verify_refresh(token)
    decision = await decision_for_claims(session, claims)
    issued = auth.issue(decision)

    set_refresh_cookie(response, issued.refresh_token, settings_of(request))

    return {
        "access_token": issued.access_token,
        "token_type": "bearer",
        "expires_at": issued.expires_at.isoformat(),
        "user": _identity(decision),
    }


@auth_router.post("/logout")
async def logout(request: Request, response: Response) -> dict[str, bool]:
    """End the session by clearing the refresh cookie.

    Unauthenticated on purpose. Logging out must work when the access token has
    already expired — requiring a valid one would mean the only users who cannot
    log out are the ones whose session is in the worst state.

    Idempotent: logging out twice, or without a session, succeeds.
    """
    clear_refresh_cookie(response, settings_of(request))
    return {"ok": True}


@auth_router.get("/me")
async def me(access: CurrentAccess) -> dict[str, Any]:
    """Who the caller is and what they may reach.

    The frontend uses this to decide what to render. It is a convenience, not a
    control — every capability listed here is separately enforced server-side on
    the route that provides it.
    """
    return _identity(access)


def _identity(decision) -> dict[str, Any]:
    return {
        "subject": decision.subject,
        "display_name": decision.display_name,
        "tenant_id": decision.tenant_id,
        "roles": sorted(r.value for r in decision.roles),
        "permissions": sorted(p.value for p in decision.permissions),
        "camera_scope": {
            "breadth": decision.cameras.breadth.value,
            "camera_ids": list(decision.cameras.camera_ids),
        },
        "site_ids": list(decision.site_ids),
    }


# ── operator status ──────────────────────────────────────────────────────────

status_router = APIRouter(prefix="/api/v1", tags=["status"])


@status_router.get("/status")
async def status(request: Request, access: CurrentAccess) -> dict[str, Any]:
    """Operator-facing status, in terms an operator can act on.

    Phase 1 answers the two signals that exist yet. Camera health, coverage and
    incidents arrive with the features that produce them — reporting a hardcoded
    zero for any of them now would be the exact failure this product must never
    commit: an empty answer that reads as a clean result.
    """
    vision = vision_of(request)
    database = database_of(request)
    db_ok, _ = await database.healthy()

    return {
        "service": {"ok": db_ok},
        "vision_os": vision.status().to_wire(),
        "tenant_id": access.tenant_id,
        "not_yet_reported": ["cameras", "coverage", "incidents"],
    }


def build_router() -> APIRouter:
    """Assemble the unauthenticated and authenticated groups. DevTools is separate."""
    root = APIRouter()
    root.include_router(health_router)
    root.include_router(auth_router)
    root.include_router(status_router)
    return root


__all__ = [
    "auth_router",
    "build_router",
    "devtools_router",
    "health_router",
    "status_router",
]
