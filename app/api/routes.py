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

from fastapi import APIRouter, Body, Request, Response
from loguru import logger

from app.api.dependencies import (
    CurrentAccess,
    DbSession,
    auth_of,
    cache_of,
    database_of,
    settings_of,
    vision_of,
)
from app.api.devtools import router as devtools_router
from app.auth.cookies import clear_refresh_cookie, read_refresh_cookie, set_refresh_cookie
from app.domain.audit import AuditAction, AuditOutcome, AuditTrail
from app.errors import AppError, NoSessionError

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

    trail = AuditTrail(session)
    try:
        decision = await auth.authenticate(session, email=email, password=password)
    except AppError as exc:
        # A failed login is the row that shows a password-spraying attempt. It
        # records the email attempted — an identifier the caller already sent —
        # and never the password, which `_scrub` would strip anyway.
        #
        # The tenant is resolved from the account when there is one, so an
        # admin investigating attempts against their own staff finds them in
        # their own trail. An unknown email has no tenant to belong to and falls
        # back to the configured default, where it is still visible rather than
        # discarded.
        await trail.record(
            action=AuditAction.LOGIN_FAILED,
            organization_id=await _tenant_for_email(session, email)
            or settings_of(request).default_tenant_id,
            actor=email,
            outcome=AuditOutcome.DENIED,
            resource_type="session",
            request_id=getattr(request.state, "request_id", ""),
            detail={"reason": exc.code},
        )
        # Committed before re-raising: the session rolls back on the way out,
        # and losing the record of a failed login because the login failed is
        # exactly the wrong way round.
        await session.commit()
        raise

    issued = auth.issue(decision)

    await trail.record(
        action=AuditAction.LOGIN,
        organization_id=decision.tenant_id,
        actor=decision.subject,
        actor_roles=tuple(sorted(r.value for r in decision.roles)),
        resource_type="session",
        request_id=getattr(request.state, "request_id", ""),
    )

    set_refresh_cookie(response, issued.refresh_token, settings_of(request))

    # A successful login is worth recording; the token it issued is not. A
    # bearer token on stdout is a bearer token in the container log, in the log
    # aggregator, and in any screen recording of a terminal — and it is valid
    # for fifteen minutes to whoever reads it there.
    logger.info("login succeeded for {}", email)

    return {
        "access_token": issued.access_token,
        "token_type": "bearer",
        "expires_at": issued.expires_at.isoformat(),
        "user": _identity(decision),
    }


async def _tenant_for_email(session, email: str) -> str:
    """The organization an email belongs to, or `""` if no account has it.

    Never raises and never reveals anything to the caller: it exists so a failed
    login is filed where somebody will look for it, not to tell the client
    whether the account exists.
    """
    from app.auth.service import load_user_by_email

    if not email:
        return ""
    try:
        user = await load_user_by_email(session, email)
    except Exception:  # noqa: BLE001 - attribution is best effort
        return ""
    return user.organization_id if user else ""


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
        # failed — and this must not log like a failure, because every page load
        # produces one.
        raise NoSessionError("no active session")

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
async def logout(request: Request, response: Response, session: DbSession) -> dict[str, bool]:
    """End the session by clearing the refresh cookie.

    Unauthenticated on purpose. Logging out must work when the access token has
    already expired — requiring a valid one would mean the only users who cannot
    log out are the ones whose session is in the worst state.

    Idempotent: logging out twice, or without a session, succeeds.
    """
    # Audited only when the cookie identifies somebody. Clearing a cookie that
    # was never set is not an event, and inventing an actor for it would put a
    # row in the trail that names nobody.
    token = read_refresh_cookie(request)
    if token:
        try:
            claims = auth_of(request).verify_refresh(token)
        except AppError:
            claims = None
        if claims is not None:
            await AuditTrail(session).record(
                action=AuditAction.LOGOUT,
                organization_id=claims.tenant_id,
                actor=claims.subject,
                actor_roles=claims.roles,
                resource_type="session",
                request_id=getattr(request.state, "request_id", ""),
            )

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
async def status(request: Request, access: CurrentAccess, session: DbSession) -> dict[str, Any]:
    """Operator-facing status, in terms an operator can act on.

    Camera health and camera configuration are real. Coverage is not, and is
    named in `not_yet_reported` rather than zeroed — reporting a hardcoded zero
    would be the exact failure this product must never commit: an empty answer
    that reads as a clean result.
    """
    vision = vision_of(request)
    database = database_of(request)
    db_ok, _ = await database.healthy()

    from app.api.dependencies import live_of

    live = live_of(request)
    summary = live.summary()
    sessions = live.visible(tenant_id=access.tenant_id, camera_ids=_scope_cameras(access))

    from app.domain.cameras import CameraService

    rows = await CameraService(session).list(
        organization_id=access.tenant_id, camera_keys=_scope_cameras(access)
    )
    registered = len(rows)
    enabled = sum(1 for row in rows if row.enabled)

    return {
        "service": {"ok": db_ok},
        "vision_os": vision.status().to_wire(),
        "tenant_id": access.tenant_id,
        # Real camera health, derived from real source state. A camera that is
        # not producing is reported as such — never as a frozen last frame, and
        # never as online.
        "cameras": {
            "configured": len(live.describe_cameras()),
            "sessions": len(sessions),
            "streaming": summary.streaming_sessions,
            "health": [
                {"camera_id": s.camera_id, "health": s.health.value, "kind": s.kind.value}
                for s in sessions
            ],
        },
        "live_runtime": summary.to_wire(),
        # Cameras now come from the database rather than an environment
        # variable, so this is what the runtime would restore on a restart.
        "cameras_registered": registered,
        "cameras_enabled": enabled,
        # Still named rather than zeroed. `incidents` left this list in Phase 5:
        # the store exists, so the count is reported at /api/v1/incidents.
        # `coverage` remains, because reporting 0 uncovered zones from a system
        # that cannot compute coverage is the failure this product must never
        # commit.
        "not_yet_reported": ["coverage"],
    }


def _scope_cameras(access):
    from app.authorization.model import ScopeBreadth

    if access.cameras.breadth is ScopeBreadth.ALL_IN_TENANT:
        return None
    return access.cameras.camera_ids


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
