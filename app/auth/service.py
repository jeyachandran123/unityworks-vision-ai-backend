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
from app.authorization.model import AccessDecision, OrganizationStatus, Permission
from app.authorization.platform import (
    ACTING_AS_PLATFORM_OPERATOR,
    entry_decision,
    resolve_operator,
)
from app.authorization.resolver import (
    accessible_organization_ids,
    decide,
    membership_for,
    parse_organization_status,
)
from app.errors import AuthenticationError, InvalidCredentialsError, ScopeError
from app.users.models import Organization, OrganizationMembership, User

#: Verified against when no user matches, so a missing account costs the same
#: bcrypt work as a wrong password. Computed once at import.
_DUMMY_HASH = hash_password("not-a-real-password-placeholder", min_length=0)


@dataclass(frozen=True, slots=True)
class Authentication:
    """The result of a successful login, before a token exists.

    Carries the whole answer to "where may this person work", not only the one
    organization the session is about to be bound to — the login response has to
    tell the frontend whether an organization *choice* is owed, and asking again
    with a second round trip would let the two answers disagree.
    """

    user: User
    #: The organization this session starts in. See `AuthService._opening`.
    decision: AccessDecision
    #: Every organization the account may enter, sorted. Always contains
    #: `decision.tenant_id`.
    organizations: tuple[str, ...]

    @property
    def must_select(self) -> bool:
        """Whether the frontend owes the user an organization choice.

        One organization is not a choice, and presenting it as one is the
        specific misfeature this flow exists to avoid: a single-organization
        administrator should reach their Command Center, not a page with one
        card on it.
        """
        return len(self.organizations) > 1


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
    ) -> Authentication:
        """Verify a password and return where this person may work.

        Raises:
            InvalidCredentialsError: no such user, wrong password, the account is
                inactive, or every organization they belong to is archived —
                deliberately indistinguishable to the caller.
        """
        user = await load_user_by_email(session, email)

        if user is None:
            verify_password(password, _DUMMY_HASH)
            raise InvalidCredentialsError("email or password is incorrect")

        if not verify_password(password, user.password_hash):
            raise InvalidCredentialsError("email or password is incorrect")

        if not user.is_active:
            raise InvalidCredentialsError("email or password is incorrect")

        opening = self._opening(user)
        if opening is None:
            # Every organization they may enter is archived or disabled. Reported
            # as bad credentials for the same reason an unknown email is: which
            # of a deployment's customers have been archived is not something a
            # login form should confirm.
            raise InvalidCredentialsError("email or password is incorrect")

        user.last_login_at = datetime.now(UTC)
        return Authentication(
            user=user,
            decision=decide(user, organization_id=opening),
            organizations=accessible_organization_ids(user),
        )

    @staticmethod
    def _opening(user: User) -> str | None:
        """Which organization a fresh session starts in, or ``None`` to refuse.

        The home organization when it is usable, because that is where the
        account lives and it is what every single-organization user has always
        got. Otherwise the first usable organization they are a member of, so
        that somebody whose home tenant has been archived is not locked out of a
        customer they still legitimately work for.

        "Usable" is `is_active` and not ARCHIVED — the same two checks login has
        always made, now asked per organization instead of once. SUSPENDED is
        deliberately usable: it narrows what a session may do rather than
        whether one may exist, and that distinction is the whole difference
        between suspension and archival.
        """

        def usable(organization: Organization | None) -> bool:
            if organization is None or not organization.is_active:
                return False
            return parse_organization_status(organization.status) is not OrganizationStatus.ARCHIVED

        by_id: dict[str, Organization] = {}
        for membership in user.memberships or ():
            if membership.organization is not None:
                by_id[membership.organization_id] = membership.organization

        home = by_id.get(user.organization_id)
        if usable(home):
            return user.organization_id

        for organization_id in sorted(by_id):
            if usable(by_id[organization_id]):
                return organization_id
        return None

    def issue(self, decision: AccessDecision) -> IssuedSession:
        roles = tuple(sorted(r.value for r in decision.roles))
        # `acting_as` travels on both tokens. On the access token because every
        # request has to be resolvable from it alone; on the refresh token
        # because a refresh must not quietly launder an audited read-only
        # operator entry into an ordinary session with the roles the account
        # happens to hold in that organization.
        access, expires = self._tokens.issue_access(
            subject=decision.subject,
            tenant_id=decision.tenant_id,
            roles=roles,
            acting_as=decision.acting_as,
        )
        refresh, _ = self._tokens.issue_refresh(
            subject=decision.subject,
            tenant_id=decision.tenant_id,
            acting_as=decision.acting_as,
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
            selectinload(User.permission_overrides),
            selectinload(User.organization),
            # The memberships *and* the organizations behind them. Loading only
            # the membership rows would answer "may they enter" but not "is that
            # organization archived", and the second question is asked on every
            # single request — one join here rather than a query per call.
            selectinload(User.memberships).selectinload(OrganizationMembership.organization),
        )
    )
    return result.scalar_one_or_none()


async def organization_of(
    session: AsyncSession, user: User, organization_id: str
) -> Organization | None:
    """The organization row for a tenant, preferring the one already loaded.

    A membership's organization arrived with the user, so the common path costs
    nothing. The query is only reached for a tenant the user is *not* a member
    of, which in practice means a platform-operator entry session.
    """
    membership = membership_for(user, organization_id)
    if membership is not None and membership.organization is not None:
        return membership.organization
    if user.organization is not None and user.organization_id == organization_id:
        return user.organization
    return (
        await session.execute(select(Organization).where(Organization.id == organization_id))
    ).scalar_one_or_none()


async def user_for_claims(session: AsyncSession, claims: TokenClaims) -> User:
    """The authenticated user behind a token, with the account checks applied.

    Split out of `decision_for_claims` so the platform-operator door
    (`app.api.dependencies.current_operator`) can reuse exactly these checks
    without building an `AccessDecision` it has no use for — an operator is not
    scoped to a tenant, and manufacturing a tenant-scoped decision just to
    throw it away would put a misleading identity into the request.

    ### The tenant check is now a membership check

    It used to be ``user.organization_id != claims.tenant_id``, which was the
    whole of multi-tenancy while an account had exactly one organization. Now
    that a token may legitimately name any organization the account belongs to,
    the equality is replaced by the table — and it is read on every request, so
    a membership revoked a minute ago refuses the next call rather than the
    call after the token expires. A token for organization A still cannot reach
    organization B: it names A, and only A's rows are ever resolved from it.

    ### An entry token is checked against the grant instead

    A token carrying ``act: platform_operator`` names an organization the
    account is deliberately *not* a member of. It is validated against the
    thing that actually authorised it — the operator grant, re-read from the
    database here rather than trusted from the token — so revoking the grant
    ends every entry session in flight.

    The ARCHIVED refusal deliberately applies to operators too, and is asked of
    the organization the token names rather than of the account's home. An
    operator's login lives in some organization like anyone else's, and if the
    organization they are working in has been archived, that session is over.
    """
    user = await load_user_by_email(session, claims.subject)
    if user is None or not user.is_active:
        raise AuthenticationError("the account is no longer active")

    organization = await organization_of(session, user, claims.tenant_id)
    if organization is None:
        raise AuthenticationError("the token's tenant no longer exists")
    if parse_organization_status(organization.status) is OrganizationStatus.ARCHIVED:
        raise AuthenticationError("the organization is archived")

    if claims.acting_as == ACTING_AS_PLATFORM_OPERATOR:
        # Raises ScopeError when the grant is gone, which is the correct
        # outcome — but the caller asked "who is this token", so it is reported
        # as an authentication failure and the session ends rather than the
        # request 403ing inside an organization the account can no longer reach.
        try:
            await resolve_operator(session, user)
        except ScopeError as exc:
            raise AuthenticationError(
                "this session was a platform-operator entry and the grant is gone"
            ) from exc
        return user

    if membership_for(user, claims.tenant_id) is None:
        raise AuthenticationError("this account is not a member of the token's organization")
    return user


async def decision_for_claims(session: AsyncSession, claims: TokenClaims) -> AccessDecision:
    """Rebuild the access decision from the database on every request.

    **Not** from the token's own claims. A token is a proof of authentication,
    not a cache of authorization: reading roles out of it would mean a revoked
    role stayed in force until the token expired, and a disabled account kept
    working for fifteen minutes. The same now goes for the organization — the
    tenant is *named* by the token and *authorised* by `user_for_claims`, which
    re-reads the membership every time.

    ARCHIVED refuses every API call, not only login, and is asked of the tenant
    the token names. SUSPENDED is not checked here — it narrows `decide()`'s
    permissions instead, so the existing per-permission checks refuse writes
    with no change at the call site, while reads and this call itself keep
    working.
    """
    user = await user_for_claims(session, claims)

    if claims.acting_as == ACTING_AS_PLATFORM_OPERATOR:
        # `user_for_claims` has already re-checked the grant against the
        # database. The reach is stated, never resolved: this account has no
        # roles in the organization it is standing in, and building the decision
        # from the rows it does not have is exactly what must not happen.
        operator = await resolve_operator(session, user)
        return entry_decision(operator, claims.tenant_id)

    return decide(user, organization_id=claims.tenant_id)


async def accessible_organizations(session: AsyncSession, user: User) -> list[Organization]:
    """The organizations this user may enter, as rows, ordered by name.

    Membership is the only source. There is no branch here for administrators,
    for operators, or for the home organization — a surface that answers "which
    customers may I open" with anything other than the membership table is a
    second authorization system, and the second one is always the one that is
    wrong.
    """
    ids = accessible_organization_ids(user)
    if not ids:
        return []
    rows = (
        (
            await session.execute(
                select(Organization).where(Organization.id.in_(ids)).order_by(Organization.name)
            )
        )
        .scalars()
        .all()
    )
    return list(rows)


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
    "Authentication",
    "AuthService",
    "IssuedSession",
    "accessible_organizations",
    "decision_for_claims",
    "load_user_by_email",
    "organization_of",
    "require",
    "user_for_claims",
]
