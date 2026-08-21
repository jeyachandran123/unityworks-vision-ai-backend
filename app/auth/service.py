"""Authentication service and the request-scoped identity dependencies.

### The order of operations matters

    credential → User row → AccessDecision → Principal + Grant + Scope

Identity is established first, and *then* scope is derived from the stored
record. At no point does request input contribute to the tenant. That is what
makes cross-tenant access structurally impossible rather than a filtering
discipline — and reversing the order, by reading a tenant from a header or a
path parameter, would undo it in a single line.

### Failing uniformly

A wrong password and an unknown email produce the same error, the same status
and — because the password is verified against a dummy hash when the user is
absent — approximately the same timing. Distinguishing them turns the login form
into an account-enumeration oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth.passwords import hash_password, verify_password
from app.auth.tokens import TokenClaims, TokenService, TokenType
from app.authorization.model import AccessDecision, Permission
from app.authorization.resolver import decide
from app.errors import AuthenticationError, InvalidCredentialsError, ScopeError
from app.users.models import User

#: Verified against when no user matches, so a missing account costs the same
#: bcrypt work as a wrong password. Computed once at import.
_DUMMY_HASH = hash_password("not-a-real-password-placeholder", min_length=0)


@dataclass(frozen=True, slots=True)
class IssuedSession:
    access_token: str
    refresh_token: str
    expires_at: datetime
    decision: AccessDecision


class AuthService:
    """Authenticates credentials and issues sessions."""

    __slots__ = ("_tokens",)

    def __init__(self, tokens: TokenService) -> None:
        self._tokens = tokens

    async def authenticate(
        self, session: AsyncSession, *, email: str, password: str
    ) -> AccessDecision:
        """Verify a password and return the caller's full access decision.

        Raises:
            InvalidCredentialsError: no such user, wrong password, or the account
                is inactive — deliberately indistinguishable to the caller.
        """
        user = await load_user_by_email(session, email)

        if user is None:
            verify_password(password, _DUMMY_HASH)
            raise InvalidCredentialsError("email or password is incorrect")

        if not verify_password(password, user.password_hash):
            raise InvalidCredentialsError("email or password is incorrect")

        if not user.is_active or not user.organization.is_active:
            raise InvalidCredentialsError("email or password is incorrect")

        user.last_login_at = datetime.now(UTC)
        return decide(user)

    def issue(self, decision: AccessDecision) -> IssuedSession:
        roles = tuple(sorted(r.value for r in decision.roles))
        access, expires = self._tokens.issue_access(
            subject=decision.subject, tenant_id=decision.tenant_id, roles=roles
        )
        refresh, _ = self._tokens.issue_refresh(
            subject=decision.subject, tenant_id=decision.tenant_id
        )
        return IssuedSession(
            access_token=access,
            refresh_token=refresh,
            expires_at=expires,
            decision=decision,
        )

    def verify_access(self, token: str) -> TokenClaims:
        return self._tokens.verify(token, expect=TokenType.ACCESS)

    def verify_refresh(self, token: str) -> TokenClaims:
        return self._tokens.verify(token, expect=TokenType.REFRESH)


async def load_user_by_email(session: AsyncSession, email: str) -> User | None:
    """Load a user with the relationships an access decision needs.

    Eager-loaded, because building the decision lazily inside an async request
    triggers implicit IO on attribute access and fails under asyncio.
    """
    result = await session.execute(
        select(User)
        .where(User.email == email.strip().lower())
        .options(
            selectinload(User.role_assignments),
            selectinload(User.access_grants),
            selectinload(User.organization),
        )
    )
    return result.scalar_one_or_none()


async def decision_for_claims(session: AsyncSession, claims: TokenClaims) -> AccessDecision:
    """Rebuild the access decision from the database on every request.

    **Not** from the token's own claims. A token is a proof of authentication,
    not a cache of authorization: reading roles out of it would mean a revoked
    role stayed in force until the token expired, and a disabled account kept
    working for fifteen minutes.
    """
    user = await load_user_by_email(session, claims.subject)
    if user is None or not user.is_active:
        raise AuthenticationError("the account is no longer active")

    decision = decide(user)
    if decision.tenant_id != claims.tenant_id:
        # The user moved organizations, or the token was minted elsewhere.
        # Either way the safe reading is that this token no longer applies.
        raise AuthenticationError("the token's tenant no longer matches the account")
    return decision


def require(decision: AccessDecision, permission: Permission) -> AccessDecision:
    """Enforce a permission, recording the denial.

    Every denial is counted by the permission that was missing, because a spike
    in one permission is either a misconfigured role or somebody probing, and
    both are worth seeing.
    """
    if not decision.has(permission):
        from app.infrastructure.observability import AUTHZ_DENIALS

        AUTHZ_DENIALS.labels(permission.value).inc()
        raise ScopeError(
            f"this account does not hold '{permission.value}'",
            details={"required": permission.value},
        )
    return decision


__all__ = [
    "AuthService",
    "IssuedSession",
    "decision_for_claims",
    "load_user_by_email",
    "require",
]
