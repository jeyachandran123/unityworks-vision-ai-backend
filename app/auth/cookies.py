"""Refresh-token cookie handling.

Phase 1 returned the refresh token in the response body, because no frontend
existed to receive it any other way. That is corrected here.

### Why the cookie, and why these exact flags

A refresh token is a seven-day credential. An access token expires in fifteen
minutes and can be held in memory; a refresh token cannot, and anywhere page
JavaScript can read it — `localStorage`, `sessionStorage`, a non-httpOnly
cookie — is one XSS away from a week of impersonation.

| flag | value | why |
|---|---|---|
| `httponly` | always `True` | page JavaScript cannot read it. The whole point. |
| `secure` | `True` in production | never sent over plain HTTP. Off for local development, where there is no TLS to be sent over. |
| `samesite` | `strict` | the browser will not attach it to any cross-site request, so a hostile page cannot silently mint an access token. |
| `path` | `/api/v1/auth` | it travels only to the routes that consume it. Every other request in the application carries one credential fewer. |

`SameSite=Strict` means the frontend must be same-origin with the API. In
development the Vite dev server proxies `/api` and `/ws`, so it is; in
production they sit behind one origin. That is a deployment requirement, not an
accident, and it is documented in the frontend README.

### Rotation

Every refresh issues a **new** refresh token and overwrites the cookie. A token
that has been exchanged is never valid again in practice, because the client no
longer holds it — which turns a stolen-and-replayed token into a visible
anomaly rather than a silent second session.
"""

from __future__ import annotations

from fastapi import Request, Response

from app.configuration.settings import Settings

#: Named for what it is. A generic `session` cookie invites other things to be
#: put in it.
REFRESH_COOKIE = "uwv_refresh"

#: Scoped to the auth routes. Nothing else needs it, so nothing else receives it.
REFRESH_PATH = "/api/v1/auth"


def set_refresh_cookie(response: Response, token: str, settings: Settings) -> None:
    """Attach a rotated refresh token. Overwrites any previous value."""
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=token,
        httponly=True,
        secure=settings.is_production,
        samesite="strict",
        path=REFRESH_PATH,
        max_age=settings.jwt_refresh_token_expire_days * 24 * 60 * 60,
    )


def clear_refresh_cookie(response: Response, settings: Settings) -> None:
    """Remove it on logout.

    The flags must match the ones it was set with, or the browser treats it as a
    different cookie and quietly leaves the original in place — a logout that
    logs nobody out.
    """
    response.delete_cookie(
        key=REFRESH_COOKIE,
        httponly=True,
        secure=settings.is_production,
        samesite="strict",
        path=REFRESH_PATH,
    )


def read_refresh_cookie(request: Request) -> str:
    """The presented refresh token, or an empty string.

    Empty rather than raising: "no session" and "a bad session" are different
    facts, and the caller decides which error each deserves.
    """
    return (request.cookies.get(REFRESH_COOKIE) or "").strip()


__all__ = [
    "REFRESH_COOKIE",
    "REFRESH_PATH",
    "clear_refresh_cookie",
    "read_refresh_cookie",
    "set_refresh_cookie",
]
