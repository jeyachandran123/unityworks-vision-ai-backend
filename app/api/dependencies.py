"""FastAPI dependencies — the only door into request-scoped identity.

Every authenticated route depends on ``current_access``. Nothing constructs an
``AccessDecision`` anywhere else, so there is exactly one code path where a
request becomes an identity, and exactly one place to audit.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.service import AuthService, decision_for_claims
from app.authorization.model import AccessDecision, Permission
from app.configuration.settings import Settings
from app.errors import AuthenticationError
from app.infrastructure.cache import Cache
from app.infrastructure.database import Database
from app.vision.runtime import VisionRuntime


def settings_of(request: Request) -> Settings:
    return request.app.state.settings


def database_of(request: Request) -> Database:
    return request.app.state.database


def cache_of(request: Request) -> Cache:
    return request.app.state.cache


def auth_of(request: Request) -> AuthService:
    return request.app.state.auth


def vision_of(request: Request) -> VisionRuntime:
    return request.app.state.vision


async def db_session(request: Request) -> AsyncIterator[AsyncSession]:
    database: Database = request.app.state.database
    async with database.session_scope() as session:
        yield session


def bearer_token(request: Request) -> str:
    """Extract the bearer token, or fail with the reason counted.

    Deliberately not FastAPI's ``HTTPBearer``: that returns a bare 403 with no
    envelope, which would be the one response in the application that does not
    match the error contract.
    """
    from app.infrastructure.observability import AUTH_FAILURES

    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        AUTH_FAILURES.labels("missing_bearer").inc()
        raise AuthenticationError("a bearer token is required")
    return token.strip()


async def current_access(
    request: Request,
    token: Annotated[str, Depends(bearer_token)],
    session: Annotated[AsyncSession, Depends(db_session)],
) -> AccessDecision:
    """The caller's identity and reach, rebuilt from the database each request.

    Rebuilt rather than read from the token so that a revoked role or a disabled
    account takes effect immediately rather than at the next token expiry.
    """
    from app.infrastructure.observability import AUTH_FAILURES

    auth: AuthService = request.app.state.auth
    try:
        claims = auth.verify_access(token)
    except AuthenticationError:
        AUTH_FAILURES.labels("invalid_token").inc()
        raise

    decision = await decision_for_claims(session, claims)
    request.state.subject = decision.subject
    request.state.tenant_id = decision.tenant_id
    return decision


def requires(permission: Permission):
    """Build a dependency enforcing one permission.

    Server-side, on the route. The frontend also hides what a user cannot reach,
    and that is a courtesy — this is the control. Hiding a link is not closing a
    door, and a DevTools route that relies on the former has no authorization at
    all.
    """

    async def _guard(
        decision: Annotated[AccessDecision, Depends(current_access)],
    ) -> AccessDecision:
        from app.auth.service import require

        return require(decision, permission)

    return _guard


CurrentAccess = Annotated[AccessDecision, Depends(current_access)]
DbSession = Annotated[AsyncSession, Depends(db_session)]


__all__ = [
    "CurrentAccess",
    "DbSession",
    "auth_of",
    "bearer_token",
    "cache_of",
    "current_access",
    "database_of",
    "db_session",
    "requires",
    "settings_of",
    "vision_of",
]
