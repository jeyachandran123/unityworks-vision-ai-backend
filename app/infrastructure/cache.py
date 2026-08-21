"""Redis lifecycle and health.

**Degraded, never fatal.** Redis unavailable must not stop the process booting.
The reference backend learned this in production: refusing to start without Redis
took down every route, including the majority that never touch it. Its own
comment records the fix, and the reasoning carries over unchanged —

    "Features that genuinely need Redis now fail at request time, where the error
     names the missing dependency, instead of at boot where it reads as 'the app
     is broken'."

Phase 1 builds no feature on Redis. The lifecycle exists so that Phase 4's
workers and rate limits have something to attach to, and so that its health is
already visible before anything depends on it.
"""

from __future__ import annotations

from typing import Any

from app.configuration.settings import Settings
from app.errors import DependencyUnavailableError


class Cache:
    """Owns one Redis connection pool for the process lifetime."""

    __slots__ = ("_client", "_last_error", "_settings")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None
        self._last_error: str = ""

    @property
    def enabled(self) -> bool:
        return self._settings.redis_enabled

    @property
    def available(self) -> bool:
        """Whether a client exists. Not the same as whether Redis is reachable."""
        return self._client is not None

    async def connect(self) -> bool:
        """Build the pool and ping. Returns success; never raises.

        Returning a bool rather than raising is the whole point: the caller is
        application startup, and startup must continue.
        """
        if not self._settings.redis_enabled:
            self._last_error = "disabled by configuration"
            return False
        try:
            import redis.asyncio as redis

            client = redis.from_url(
                self._settings.redis_url,
                max_connections=self._settings.redis_max_connections,
                decode_responses=True,
            )
            await client.ping()
            self._client = client
            self._last_error = ""
            return True
        except Exception as exc:  # noqa: BLE001 - degradation is the contract
            # The type only. A Redis URL carries a password, and an exception
            # message from a connection failure frequently quotes the URL.
            self._last_error = type(exc).__name__
            self._client = None
            return False

    async def disconnect(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def require(self) -> Any:
        """The client, for a feature that genuinely needs it.

        Raises at *request* time with the dependency named, which is where the
        error is actionable.
        """
        if self._client is None:
            raise DependencyUnavailableError(
                "this feature requires Redis, which is not connected",
                details={"dependency": "redis", "reason": self._last_error or "unknown"},
            )
        return self._client

    async def healthy(self) -> tuple[bool, str]:
        """``(ok, detail)``. Never raises."""
        if not self._settings.redis_enabled:
            return True, "disabled"
        if self._client is None:
            return False, self._last_error or "not connected"
        try:
            await self._client.ping()
            return True, "ok"
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            return False, type(exc).__name__


__all__ = ["Cache"]
