"""A first-class 'not configured' answer, with one shape for every module.

### Why this is a 200 and not a 404

A 404 says the surface does not exist, which is false and unhelpful: the module
is real, its schema is decided, its permission is enforced and its page is
routed. A 501 says the server does not implement the method, which is also
false. What is true is narrower and more useful — *the thing exists, nothing has
been connected to it, and here is exactly what is missing* — and none of the
status codes says that, so the answer says it in the body.

This mirrors what the dashboard already does. `/status` names `coverage` in
`not_yet_reported` rather than reporting zero uncovered zones, because a zero
from a system that cannot compute the figure is the one failure this product
must never commit. A capability route generalises that: `available: false` with
a reason and a checklist, never an empty list that reads as a clean result.

### `stored_records` is a real count of a real table

It is not a metric and must never be rendered as one. It answers "does the
schema exist and is it empty", which is exactly what an operator asking why a
page is blank needs to know, and it comes from `SELECT count(*)` against the
tenant's rows rather than from a constant — so the day something starts writing,
this stops being zero without anybody changing this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True, slots=True)
class Requirement:
    """One real-world input a module is waiting for.

    `id` is stable and machine-readable so a frontend can key on it; `detail` is
    the sentence a person reads, and it is written here rather than in the
    frontend so there is exactly one copy of each answer.
    """

    id: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"id": self.id, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ModuleCapability:
    """Everything a client needs to render a module that is not connected."""

    module: str
    title: str
    #: What this module would report if it were connected. Present so a page can
    #: describe the product rather than only the absence.
    purpose: str
    reason: str
    requirements: tuple[Requirement, ...]
    #: The tables that hold this module's records, named so an operator can see
    #: the schema exists.
    tables: tuple[str, ...] = ()
    #: Anchor in `docs/architecture/NOT_YET_CONNECTED.md`.
    documentation: str = ""
    #: `not_configured` for everything in this phase. `blocked` is reserved for
    #: a module a *decision* is withholding rather than an input — patron
    #: identification is the only one, and the distinction matters because one
    #: of them is waiting for work and the other is waiting for permission.
    state: str = "not_configured"


async def render(
    capability: ModuleCapability,
    session: AsyncSession,
    *,
    organization_id: str,
    models: tuple[Any, ...] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The wire shape. `available` is always false while nothing is bound.

    `models` are counted for real, scoped to the caller's tenant — constructed
    already narrowed, the same discipline every other query in this application
    follows, so there is no moment at which another tenant's rows exist in
    memory to leak.
    """
    counts: dict[str, int] = {}
    for model in models:
        total = await session.scalar(
            select(func.count())
            .select_from(model)
            .where(model.organization_id == organization_id)
        )
        counts[model.__tablename__] = int(total or 0)

    stored = sum(counts.values())
    return {
        "module": capability.module,
        "title": capability.title,
        "purpose": capability.purpose,
        # False, and it is the field every consumer must branch on before it
        # looks at anything else — exactly as the observation API requires.
        "available": False,
        "state": capability.state,
        "reason": capability.reason,
        "awaiting": [r.as_dict() for r in capability.requirements],
        # The schema is real and empty. Never rendered as a metric: zero rows is
        # a fact about storage, not a reading about a kitchen.
        "storage_ready": True,
        "tables": list(capability.tables),
        "stored_records": stored,
        "records_by_table": counts,
        "documentation": capability.documentation,
        **(extra or {}),
    }


__all__ = ["ModuleCapability", "Requirement", "render"]
