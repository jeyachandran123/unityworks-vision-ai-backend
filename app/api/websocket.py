"""Authenticated WebSocket foundation.

### Why the token is not in the query string

A URL is logged. By the browser, by every reverse proxy, by the access log, and
by anything that samples traffic. `?token=…` puts a bearer credential in all of
them permanently, and the usual workarounds (short TTL, one-time tokens) are
compensating controls for a decision that did not need making.

So this handshake is: **connect, then authenticate over the socket.** The client
sends one `authenticate` frame with its access token; nothing else is served
until that frame arrives and verifies, and a connection that fails to
authenticate within the grace period is closed.

The trade is one extra round trip at connect time, which is invisible next to the
lifetime of a monitoring session.

### What it does not do yet

There is no live event source in this phase — no camera is acquiring frames, so
nothing is produced to fan out. This establishes the **authentication contract**
and the connection lifecycle so that Phase 3 attaches a real subscription to a
socket whose security is already settled and tested.

A client that authenticates receives `ready`, then heartbeats. It does **not**
receive fabricated observations: a "live" badge that lights up over invented
traffic is worse than no badge.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()

#: How long a socket may stay unauthenticated. Short, because an unauthenticated
#: connection is an open file descriptor that anyone can create.
AUTH_GRACE_SECONDS = 10.0

#: Server→client liveness. The client's own timer should be a small multiple of
#: this; a socket that stops heart-beating is dead long before TCP notices.
HEARTBEAT_SECONDS = 25.0

# Close codes. 4401/4403 mirror HTTP semantics in the application range so a
# client can distinguish "log in again" from "you will never be allowed".
CLOSE_UNAUTHENTICATED = 4401
CLOSE_FORBIDDEN = 4403
CLOSE_TIMEOUT = 4408


@router.websocket("/ws/v1/live")
async def live(socket: WebSocket) -> None:
    """The live channel. Authenticate first, then subscribe (Phase 3)."""
    await socket.accept()

    try:
        access = await _authenticate(socket)
    except (WebSocketDisconnect, asyncio.CancelledError):
        return
    except _HandshakeFailure as failure:
        with contextlib.suppress(Exception):
            await socket.close(code=failure.code, reason=failure.reason)
        return

    await socket.send_text(
        json.dumps(
            {
                "type": "ready",
                "subject": access.subject,
                "tenant_id": access.tenant_id,
                # Stated rather than implied. A client that knows no stream is
                # attached can render "connected, not yet streaming" instead of
                # an empty view that looks like an outage.
                "streaming": False,
                "note": (
                    "authenticated; no live source is attached before Phase 3, "
                    "so no observation frames will be delivered"
                ),
            }
        )
    )

    try:
        await _heartbeat(socket)
    except (WebSocketDisconnect, asyncio.CancelledError):
        return
    except Exception:  # noqa: BLE001 - a dead socket is not an application error
        logger.debug("live socket closed for {}", access.subject)


class _HandshakeFailure(Exception):
    def __init__(self, code: int, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


async def _authenticate(socket: WebSocket):
    """Read one `authenticate` frame and resolve it to an access decision.

    The decision is rebuilt from the database exactly as it is on an HTTP
    request. A socket held open for hours must not keep the authorization it had
    when it opened.
    """
    from app.auth.service import decision_for_claims
    from app.errors import AuthenticationError
    from app.infrastructure.observability import AUTH_FAILURES

    try:
        raw = await asyncio.wait_for(socket.receive_text(), timeout=AUTH_GRACE_SECONDS)
    except TimeoutError as exc:
        AUTH_FAILURES.labels("ws_handshake_timeout").inc()
        raise _HandshakeFailure(CLOSE_TIMEOUT, "authentication timed out") from exc

    try:
        message = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _HandshakeFailure(CLOSE_UNAUTHENTICATED, "malformed frame") from exc

    if message.get("type") != "authenticate":
        raise _HandshakeFailure(
            CLOSE_UNAUTHENTICATED, "the first frame must be 'authenticate'"
        )

    token = str(message.get("access_token", "")).strip()
    if not token:
        AUTH_FAILURES.labels("ws_missing_token").inc()
        raise _HandshakeFailure(CLOSE_UNAUTHENTICATED, "an access token is required")

    app = socket.app
    try:
        claims = app.state.auth.verify_access(token)
    except AuthenticationError as exc:
        AUTH_FAILURES.labels("ws_invalid_token").inc()
        # Uniform reason. Distinguishing expired from forged over a socket is the
        # same probing oracle it is over HTTP.
        raise _HandshakeFailure(CLOSE_UNAUTHENTICATED, "authentication failed") from exc

    database = app.state.database
    try:
        async with database.session_scope() as session:
            decision = await decision_for_claims(session, claims)
    except AuthenticationError as exc:
        raise _HandshakeFailure(CLOSE_UNAUTHENTICATED, "authentication failed") from exc

    from app.authorization.model import Permission

    if not decision.has(Permission.VIEW_LIVE):
        from app.infrastructure.observability import AUTHZ_DENIALS

        AUTHZ_DENIALS.labels(Permission.VIEW_LIVE.value).inc()
        raise _HandshakeFailure(
            CLOSE_FORBIDDEN, "this account may not view live monitoring"
        )

    return decision


async def _heartbeat(socket: WebSocket) -> None:
    """Send a heartbeat until the client goes away."""
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        await socket.send_text(json.dumps({"type": "heartbeat"}))


def describe_protocol() -> dict[str, Any]:
    """The handshake, as data — so the frontend and its tests agree with the server."""
    return {
        "path": "/ws/v1/live",
        "handshake": "connect, then send {'type':'authenticate','access_token':'…'}",
        "auth_grace_seconds": AUTH_GRACE_SECONDS,
        "heartbeat_seconds": HEARTBEAT_SECONDS,
        "close_codes": {
            "unauthenticated": CLOSE_UNAUTHENTICATED,
            "forbidden": CLOSE_FORBIDDEN,
            "timeout": CLOSE_TIMEOUT,
        },
        "token_in_query_string": False,
    }


__all__ = [
    "AUTH_GRACE_SECONDS",
    "CLOSE_FORBIDDEN",
    "CLOSE_TIMEOUT",
    "CLOSE_UNAUTHENTICATED",
    "HEARTBEAT_SECONDS",
    "describe_protocol",
    "router",
]
