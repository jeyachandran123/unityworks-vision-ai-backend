"""Database engine, session lifecycle and health.

### Tables are created by Alembic, never by the application

``Base.metadata.create_all()`` is not called at startup. A schema that appears
because the application booted is a schema nobody reviewed, and it diverges from
the migration history the moment the two disagree. The one exception is tests,
which build a throwaway SQLite schema and say so explicitly.

### Transaction boundary

One session per request, committed by the caller. ``session_scope`` commits on a
clean exit and rolls back on any exception, so a handler that raises never leaves
a half-written unit of work behind.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.configuration.settings import Settings
from app.errors import DependencyUnavailableError


class Base(DeclarativeBase):
    """Declarative base for every application table.

    Vision OS has no table here and never will: its state lives in its own
    stores, behind its own ports. Copying observations into this schema would
    create a second source of truth for what a camera saw, and the second one
    would drift.
    """


class Database:
    """Owns one engine and its sessionmaker for the process lifetime."""

    __slots__ = ("_engine", "_sessionmaker", "_settings")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def connect(self) -> AsyncEngine:
        """Build the engine. Idempotent; does not open a connection."""
        if self._engine is not None:
            return self._engine

        url = self._settings.database_url
        kwargs: dict = {"echo": self._settings.db_echo, "pool_pre_ping": True}
        # SQLite (tests) has no connection pool to size, and passing pool
        # arguments to its dialect raises rather than being ignored.
        if not url.startswith("sqlite"):
            kwargs["pool_size"] = self._settings.db_pool_size
            kwargs["max_overflow"] = self._settings.db_max_overflow

        self._engine = create_async_engine(url, **kwargs)
        self._sessionmaker = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        return self._engine

    async def disconnect(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise DependencyUnavailableError("the database engine is not connected")
        return self._engine

    # ── sessions ─────────────────────────────────────────────────────────────

    @contextlib.asynccontextmanager
    async def session_scope(self) -> AsyncIterator[AsyncSession]:
        """A session with a transaction boundary tied to the block."""
        if self._sessionmaker is None:
            raise DependencyUnavailableError("the database engine is not connected")
        session = self._sessionmaker()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def session(self) -> AsyncIterator[AsyncSession]:
        """FastAPI dependency form of :meth:`session_scope`."""
        async with self.session_scope() as active:
            yield active

    # ── health ───────────────────────────────────────────────────────────────

    async def healthy(self) -> tuple[bool, str]:
        """``(ok, detail)``. Never raises — a health check that raises is not one.

        The detail names the failure class, not the connection string: a DSN in a
        health response is a credential in a health response.
        """
        from sqlalchemy import text

        if self._engine is None:
            return False, "not connected"
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True, "ok"
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            return False, type(exc).__name__


async def create_all_for_tests(database: Database) -> None:
    """Build the schema directly. **Tests only.**

    Named so that its appearance in application code is obviously wrong. Production
    schema changes go through Alembic, where they can be reviewed and rolled back.
    """
    async with database.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


__all__ = ["Base", "Database", "create_all_for_tests"]
