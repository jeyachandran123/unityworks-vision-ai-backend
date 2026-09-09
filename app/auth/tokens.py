"""JWT issuance and verification.

PyJWT rather than python-jose, which the reference backend used: python-jose has
been effectively unmaintained since 2021 and has carried algorithm-confusion
CVEs. This is a recreation, not a copy, and the safer library is the one to
recreate onto.

### The two-token split

An **access token** is short-lived (15 min) and travels on every request. A
**refresh token** is long-lived (7 days), travels only to the refresh endpoint,
and — in the frontend from Phase 2 — lives in an httpOnly cookie the page's
JavaScript cannot read.

The split exists so that a stolen access token expires on its own, and a stolen
refresh token is hard to steal in the first place. Both properties are lost if
the access token is given a long life "for convenience".

### The `act` claim

An ordinary token names a tenant the account is a member of. A token minted by
a platform-operator entry names one it is deliberately *not* a member of, and
``act`` is how the two are told apart on the way back in. It grants nothing:
`user_for_claims` re-reads the operator grant from the database on every
request that carries it, so the claim decides which check applies, never
whether the check passes.

### Why the token type is inside the payload

``typ`` is checked on every verification. Without it, a refresh token is a
perfectly valid access token — same signature, same issuer — and the 15-minute
expiry becomes 7 days for anyone who thinks to try it.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jwt

from app.configuration.settings import Settings
from app.errors import AuthenticationError, ConfigurationInvalidError, TokenExpiredError


class TokenType(enum.Enum):
    ACCESS = "access"
    REFRESH = "refresh"


@dataclass(frozen=True, slots=True)
class TokenClaims:
    """A verified token's contents. Constructed only after signature checks pass."""

    subject: str
    tenant_id: str
    token_type: TokenType
    roles: tuple[str, ...]
    token_id: str
    issued_at: datetime
    expires_at: datetime
    #: Empty for an ordinary session. ``"platform_operator"`` for one minted by
    #: an audited entry into an organization the account is not a member of
    #: (`app.authorization.platform.ACTING_AS_PLATFORM_OPERATOR`).
    #:
    #: It is a *statement of how this tenant was reached*, never a grant. The
    #: grant is re-read from the database on every request, exactly as roles
    #: are — a claim that authorised anything by its own presence would be a
    #: cross-customer privilege forgeable by anyone who could mint a token, and
    #: the point of putting it here is that it makes the session's provenance
    #: undeniable rather than that it makes it powerful.
    acting_as: str = ""

    @property
    def is_access(self) -> bool:
        return self.token_type is TokenType.ACCESS


class TokenService:
    """Issues and verifies tokens for one configuration.

    A class rather than module functions so that tests can construct one with
    test settings without mutating process state, and so key material is
    resolved once at construction rather than on every request.
    """

    __slots__ = ("_settings", "_signing_key", "_verifying_key")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._signing_key, self._verifying_key = _resolve_keys(settings)

    # ── issuing ──────────────────────────────────────────────────────────────

    def issue_access(
        self,
        *,
        subject: str,
        tenant_id: str,
        roles: tuple[str, ...] = (),
        acting_as: str = "",
    ) -> tuple[str, datetime]:
        return self._issue(
            subject=subject,
            tenant_id=tenant_id,
            roles=roles,
            token_type=TokenType.ACCESS,
            lifetime=timedelta(minutes=self._settings.jwt_access_token_expire_minutes),
            acting_as=acting_as,
        )

    def issue_refresh(
        self, *, subject: str, tenant_id: str, acting_as: str = ""
    ) -> tuple[str, datetime]:
        # No roles in a refresh token. It authorises nothing; it only proves the
        # session is still alive. Carrying roles would let a role revocation take
        # up to seven days to bite.
        #
        # `acting_as` is the exception, and it is not a role. It records how the
        # tenant on this token was reached, so that refreshing an operator entry
        # yields another operator entry. Dropping it here would turn a read-only
        # audited session into an ordinary one at the next refresh — silently,
        # and in the direction of more access.
        return self._issue(
            subject=subject,
            tenant_id=tenant_id,
            roles=(),
            token_type=TokenType.REFRESH,
            lifetime=timedelta(days=self._settings.jwt_refresh_token_expire_days),
            acting_as=acting_as,
        )

    def _issue(
        self,
        *,
        subject: str,
        tenant_id: str,
        roles: tuple[str, ...],
        token_type: TokenType,
        lifetime: timedelta,
        acting_as: str = "",
    ) -> tuple[str, datetime]:
        if not subject or not tenant_id:
            raise ValueError("a token must name a subject and a tenant")

        now = datetime.now(UTC)
        expires = now + lifetime
        payload: dict[str, Any] = {
            "sub": subject,
            "ten": tenant_id,
            "typ": token_type.value,
            "iss": self._settings.jwt_issuer,
            "iat": int(now.timestamp()),
            "exp": int(expires.timestamp()),
            "jti": uuid.uuid4().hex,
        }
        if roles:
            payload["rol"] = list(roles)
        if acting_as:
            payload["act"] = acting_as

        token = jwt.encode(payload, self._signing_key, algorithm=self._settings.jwt_algorithm)
        return token, expires

    # ── verifying ────────────────────────────────────────────────────────────

    def verify(self, token: str, *, expect: TokenType) -> TokenClaims:
        """Verify signature, expiry, issuer and **type**.

        Raises:
            TokenExpiredError: valid but past its expiry — the client should
                refresh, not re-authenticate.
            AuthenticationError: anything else.
        """
        try:
            payload = jwt.decode(
                token,
                self._verifying_key,
                algorithms=[self._settings.jwt_algorithm],
                issuer=self._settings.jwt_issuer,
                options={"require": ["exp", "iat", "sub", "iss"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise TokenExpiredError("the token has expired") from exc
        except jwt.InvalidTokenError as exc:
            # Deliberately uniform. Reporting *why* a token is invalid — bad
            # signature vs wrong issuer vs malformed — is a probing oracle.
            raise AuthenticationError("the token is not valid") from exc

        actual = str(payload.get("typ", ""))
        if actual != expect.value:
            raise AuthenticationError(
                f"expected a {expect.value} token",
                details={"expected": expect.value},
            )

        subject = str(payload.get("sub", ""))
        tenant_id = str(payload.get("ten", ""))
        if not subject or not tenant_id:
            raise AuthenticationError("the token names no subject or no tenant")

        return TokenClaims(
            subject=subject,
            tenant_id=tenant_id,
            token_type=TokenType(actual),
            roles=tuple(str(r) for r in payload.get("rol", ())),
            token_id=str(payload.get("jti", "")),
            issued_at=datetime.fromtimestamp(int(payload["iat"]), tz=UTC),
            expires_at=datetime.fromtimestamp(int(payload["exp"]), tz=UTC),
            acting_as=str(payload.get("act", "")),
        )


def _resolve_keys(settings: Settings) -> tuple[str, str]:
    """Signing and verifying key material for the configured algorithm."""
    if settings.jwt_algorithm == "HS256":
        secret = settings.secret_key.get_secret_value()
        if not secret:
            raise ConfigurationInvalidError("SECRET_KEY is required to sign tokens")
        return secret, secret

    private = Path(settings.jwt_private_key_path)
    public = Path(settings.jwt_public_key_path)
    for label, path in (("JWT_PRIVATE_KEY_PATH", private), ("JWT_PUBLIC_KEY_PATH", public)):
        if not path.is_file():
            raise ConfigurationInvalidError(
                f"{label} does not name a readable file",
                details={"setting": label},
            )
    return private.read_text(), public.read_text()


__all__ = ["TokenClaims", "TokenService", "TokenType"]
