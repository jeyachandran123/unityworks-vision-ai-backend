"""The report engine's core shapes. **Coverage is not optional.**

### The one decision this module encodes

`ReportData` cannot be constructed without a `Coverage`. Not "should carry one",
not "carries one by convention" — the dataclass has no default for it, so a
report type that forgets is a `TypeError` at construction rather than a page of
numbers with nothing saying what they were computed from.

That is the same rule the rest of this product already follows and the reason it
is worth enforcing structurally here: `/api/v1/status` names `coverage` in
`not_yet_reported` rather than returning zero uncovered zones, and
`/api/v1/observations` returns `available: false` with a reason rather than an
empty list. A report is where those disciplines are easiest to lose, because a
table with no rows renders perfectly happily as a clean month.

### Three ways a report can have nothing to say, and they are all different

    the source is not connected      → `SourceCoverage.available = False`, with a reason
    the source was read and is empty → `available = True`, `rows = 0`, and the section says so
    part of the window was unreadable → `complete = False`, with named gaps

A report that rendered all three as an empty table would tell a manager their
kitchen was clean when the truth was that nothing was looking at it.

### Why sections are tabular rather than free-form

Every renderer — JSON, CSV, Excel, PDF — has to produce the same report from the
same data, and a free-form section would mean four renderers each deciding what
it looks like. A `Section` is columns and rows, which every one of them can
express without inventing anything.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime
from typing import Any


class Granularity(enum.Enum):
    """How a period is subdivided. A closed set."""

    TOTAL = "total"
    """One bucket covering the whole window. Always supported."""
    DAY = "day"
    WEEK = "week"
    MONTH = "month"

    @property
    def label(self) -> str:
        return {"total": "Whole period", "day": "Daily", "week": "Weekly", "month": "Monthly"}[
            self.value
        ]


class ExportFormat(enum.Enum):
    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"
    PDF = "pdf"


@dataclass(frozen=True, slots=True)
class Gap:
    """A named part of the window that could not be read, or was not covered.

    A gap is stated, never subtracted. Removing an unreadable hour from a
    denominator makes a partial figure look complete, which is the specific lie
    this type exists to prevent.
    """

    #: `future`, `before_history`, `source_unavailable`, `truncated`.
    kind: str
    detail: str
    since: datetime | None = None
    until: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "since": self.since.isoformat() if self.since else None,
            "until": self.until.isoformat() if self.until else None,
        }


@dataclass(frozen=True, slots=True)
class SourceCoverage:
    """What one data source contributed, and whether it could be read at all.

    `available=False` is emphatically not `rows=0`. The first says the source
    could not be consulted; the second says it was consulted and held nothing.
    Both render, and they never render the same.
    """

    #: `incidents`, `observations`, `cameras`, `audit`, or a Phase 2 module id.
    source: str
    available: bool
    #: Why, when unavailable. Shown to the reader verbatim.
    reason: str = ""
    rows: int = 0
    #: True when a row cap stopped the read before the source was exhausted.
    #: A truncated section is never presented as a complete one.
    truncated: bool = False
    #: The earliest record this source holds, when known. Used to detect a
    #: window that reaches back further than the data does.
    earliest: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "available": self.available,
            "reason": self.reason,
            "rows": self.rows,
            "truncated": self.truncated,
            "earliest": self.earliest.isoformat() if self.earliest else None,
        }


@dataclass(frozen=True, slots=True)
class Coverage:
    """What a report was computed from. Mandatory on every `ReportData`.

    `complete` is the single field a reader needs to know whether the numbers
    above it can be compared with last month's. It is false whenever *any*
    source was unavailable, *any* section was truncated, or the window extends
    past now or before the data — and the reasons are in `gaps`.
    """

    since: datetime
    until: datetime
    #: The IANA zone the period boundaries were computed in.
    timezone: str
    #: Whether that zone actually resolved. False means the boundaries are UTC
    #: and say so, rather than being silently wrong by eight hours.
    timezone_resolved: bool
    granularity: Granularity
    sources: tuple[SourceCoverage, ...] = ()
    gaps: tuple[Gap, ...] = ()
    #: A sentence naming what the figures were computed from, for a reader who
    #: will not expand the coverage panel.
    basis: str = ""

    @property
    def complete(self) -> bool:
        """False if anything at all makes these figures less than the whole story."""
        if any(not source.available for source in self.sources):
            return False
        if any(source.truncated for source in self.sources):
            return False
        return not self.gaps

    def as_dict(self) -> dict[str, Any]:
        return {
            "since": self.since.isoformat(),
            "until": self.until.isoformat(),
            "timezone": self.timezone,
            "timezone_resolved": self.timezone_resolved,
            "granularity": self.granularity.value,
            "complete": self.complete,
            "basis": self.basis,
            "sources": [s.as_dict() for s in self.sources],
            "gaps": [g.as_dict() for g in self.gaps],
        }


@dataclass(frozen=True, slots=True)
class Column:
    """One column of a section. `numeric` right-aligns and uses tabular figures."""

    key: str
    header: str
    numeric: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "header": self.header, "numeric": self.numeric}


@dataclass(frozen=True, slots=True)
class Section:
    """One table in a report.

    `empty_note` is required rather than optional, and it is why: a section with
    no rows has to say *which* kind of nothing it is. "No incident was raised in
    this period" and "the incident store could not be read" are different
    sentences, and a renderer cannot invent either of them.
    """

    key: str
    title: str
    columns: tuple[Column, ...]
    rows: tuple[dict[str, Any], ...]
    empty_note: str
    #: A sentence about this section specifically — what it counts, what it
    #: excludes. Rendered above the table in every format.
    note: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "columns": [c.as_dict() for c in self.columns],
            "rows": list(self.rows),
            "empty_note": self.empty_note,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class ReportData:
    """A rendered-ready report. **`coverage` has no default, deliberately.**

    A report type that forgets it fails at construction rather than producing a
    page of numbers with nothing saying what they were computed from.
    """

    report_id: str
    title: str
    subtitle: str
    coverage: Coverage
    sections: tuple[Section, ...] = ()
    #: `not_configured` / `blocked` for a Phase 2 module report; empty for one
    #: backed by real data. Carried so a client renders the *same* honest state
    #: those modules' own pages already show, from the same shape.
    capability_state: str = ""
    capability_reason: str = ""
    #: The module's own activation checklist, when this is a capability report.
    awaiting: tuple[dict[str, str], ...] = ()
    generated_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "title": self.title,
            "subtitle": self.subtitle,
            "coverage": self.coverage.as_dict(),
            "sections": [s.as_dict() for s in self.sections],
            "capability_state": self.capability_state,
            "capability_reason": self.capability_reason,
            "awaiting": [dict(a) for a in self.awaiting],
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }


@dataclass(frozen=True, slots=True)
class ReportRequest:
    """What the caller asked for, already validated and resolved.

    Constructed once at the route boundary and passed down, so a collector never
    parses a query string and never sees an unvalidated window.
    """

    report_id: str
    since: datetime
    until: datetime
    granularity: Granularity
    timezone: str
    timezone_resolved: bool
    organization_id: str
    #: `None` is a tenant-wide grant; an **empty tuple is none** and matches
    #: nothing — the same three-state rule the rest of the application uses.
    camera_keys: tuple[str, ...] | None
    restaurant_id: str = ""
    #: Hard ceiling per section. A read that hits it reports `truncated`.
    row_limit: int = 5000


#: Per-section row ceiling. A report is a document somebody reads, not a data
#: dump: past a few thousand rows a PDF is unusable and the honest answer is a
#: narrower period. Hitting it sets `truncated`, which sets `complete = False`.
DEFAULT_ROW_LIMIT = 5000

#: The longest window any report will compute. A year and a day, so that "the
#: last twelve months" always fits and "everything since we installed it" does
#: not silently become an unbounded scan.
MAX_WINDOW_DAYS = 366


__all__ = [
    "DEFAULT_ROW_LIMIT",
    "MAX_WINDOW_DAYS",
    "Column",
    "Coverage",
    "ExportFormat",
    "Gap",
    "Granularity",
    "ReportData",
    "ReportRequest",
    "Section",
    "SourceCoverage",
]
