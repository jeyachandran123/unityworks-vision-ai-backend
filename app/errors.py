"""The application error envelope.

One shape for every failure, so a client's error handling never depends on which
layer failed:

    {"code": "...", "message": "...", "retryable": bool,
     "details": {...}, "request_id": "..."}

This is the shape Vision OS already uses for its own typed errors (09_API §8), and
matching it means a consumer parses one envelope rather than two.

### What must never cross this boundary

Stack traces, file paths, module names, SQL, credentials, secret values, and the
internal ``str(exc)`` of an unexpected error. An unhandled exception becomes a
generic ``INTERNAL`` with a request id; the detail goes to the log, where an
engineer can find it by that id and a client cannot.

Typed failures are different. ``WindowTooLargeError`` means "narrow your window",
and rendering it as a 500 with a traceback would send an integrator hunting
through platform logs for what is really a documented policy bound.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base for every failure this application raises deliberately.

    ``code`` is the stable identifier a client may branch on. ``message`` is for
    a human and may change; ``code`` may not.
    """

    code: str = "INTERNAL"
    http_status: int = 500
    retryable: bool = False

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_envelope(self, request_id: str = "") -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
            "request_id": request_id,
        }


# ── Authentication ───────────────────────────────────────────────────────────


class AuthenticationError(AppError):
    """No usable credential was presented, or it did not verify."""

    code = "UNAUTHENTICATED"
    http_status = 401


class TokenExpiredError(AuthenticationError):
    """Distinct from a bad token: the client should refresh rather than re-login.

    Collapsing the two forces a client to guess, and the usual guess is to send
    the user back to a login screen they did not need to see.
    """

    code = "TOKEN_EXPIRED"
    retryable = True


class NoSessionError(AuthenticationError):
    """No session cookie was presented at all.

    Distinct from a *failed* credential, and deliberately quiet: every page load
    calls `/auth/refresh` to restore a session, and a first-time visitor has no
    cookie to send. That is the flow working, not a security event, and logging
    it at the same level as a real authentication failure buries the ones that
    matter under routine traffic.

    Still a 401 — the client must know to show the login screen — but the code
    lets a client distinguish "you were never signed in" from "your session was
    rejected", which are different things to tell a user.
    """

    code = "NO_SESSION"


class InvalidCredentialsError(AuthenticationError):
    """Wrong username or password.

    Deliberately does not say which. Distinguishing them turns the login form
    into an account-enumeration oracle.
    """

    code = "INVALID_CREDENTIALS"


# ── Authorization ────────────────────────────────────────────────────────────


class AuthorizationError(AppError):
    """Authenticated, and not permitted.

    Separate from ``AuthenticationError`` because the remedies differ entirely:
    one is "log in", the other is "ask an administrator".
    """

    code = "FORBIDDEN"
    http_status = 403


class ScopeError(AuthorizationError):
    """The principal is not granted the tenant, site or camera it asked for."""

    code = "OUT_OF_SCOPE"


class EvidenceForbiddenError(AuthorizationError):
    """Imagery was requested without the evidence privilege.

    Its own class because it is its own act. 12_SECURITY §5.3: *"Reading 'a
    person was here' and viewing their image are categorically different acts."*
    A distinct code lets an audit query find every attempt.
    """

    code = "EVIDENCE_FORBIDDEN"


# ── Request ──────────────────────────────────────────────────────────────────


class ValidationError(AppError):
    code = "INVALID_REQUEST"
    http_status = 422


class NotFoundError(AppError):
    code = "NOT_FOUND"
    http_status = 404


class ConflictError(AppError):
    code = "CONFLICT"
    http_status = 409


class RateLimitedError(AppError):
    code = "RATE_LIMITED"
    http_status = 429
    retryable = True


# ── Infrastructure ───────────────────────────────────────────────────────────


class ConfigurationInvalidError(AppError):
    """A deployment is misconfigured. Surfaces at startup, not at request time."""

    code = "CONFIGURATION_INVALID"
    http_status = 500


class DependencyUnavailableError(AppError):
    """A backing service is down. Retryable, and it names which one.

    Naming the dependency is safe — that it exists is not a secret — and it is
    the difference between an operator restarting the right thing and the wrong
    thing.
    """

    code = "DEPENDENCY_UNAVAILABLE"
    http_status = 503
    retryable = True


class VisionUnavailableError(AppError):
    """Vision OS is not assembled, or could not answer.

    Distinct from ``DEPENDENCY_UNAVAILABLE`` because "the platform is not
    running" and "the platform saw nothing" must never be reported the same way
    — invariant V8, and the reason the validation console renders a boot failure
    rather than an empty result.
    """

    code = "VISION_UNAVAILABLE"
    http_status = 503
    retryable = True


__all__ = [
    "AppError",
    "AuthenticationError",
    "AuthorizationError",
    "ConfigurationInvalidError",
    "ConflictError",
    "DependencyUnavailableError",
    "EvidenceForbiddenError",
    "InvalidCredentialsError",
    "NoSessionError",
    "NotFoundError",
    "RateLimitedError",
    "ScopeError",
    "TokenExpiredError",
    "ValidationError",
    "VisionUnavailableError",
]
