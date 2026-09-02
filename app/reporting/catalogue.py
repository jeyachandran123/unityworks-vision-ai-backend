"""The report catalogue: what exists, who may run it, and how it is built.

### The abstraction, and why adding a report later is small

A `ReportType` declares four things and nothing else:

    what data it needs      → `permissions`, and a `collect` that reads it
    what periods it supports → `granularities`
    how it says "no data"    → its collectors' `empty_note` and `SourceCoverage`
    what it is called        → `title`, `summary`

Connecting a Phase 2 module later is then: write a collector that returns
`(sections, SourceCoverage)`, and change that module's entry in `MODULE_REPORTS`
from `capability_report(...)` to a `ReportType` with that collector. The page,
the export path, the audit rows, the period handling and the coverage panel are
already built and do not move. That is the whole point of this file.

### Permissions compose, they never substitute

**A report type requires its own permission *and* the permission for every
source it reads.** An incident report needs `VIEW_REPORTS` and `VIEW_INCIDENTS`;
an audit report needs `VIEW_REPORTS` and `VIEW_AUDIT`. Without that rule the
reporting page becomes a permission bypass — the single most likely way a
reporting feature undoes an access-control model, because it is the one surface
whose whole purpose is to assemble data from everywhere at once.

`EXPORT_REPORTS` is separate from `VIEW_REPORTS` again, deliberately: a
downloaded file leaves this system entirely. It can be forwarded, it is not
covered by any retention sweep here, and it outlives every policy this
application enforces. Looking at a figure on screen and taking a copy of it away
are different acts.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.authorization.model import Permission
from app.reporting import sources as collectors
from app.reporting.model import (
    Column,
    Coverage,
    Gap,
    Granularity,
    ReportData,
    ReportRequest,
    Section,
    SourceCoverage,
)
from app.reporting.periods import ResolvedZone, gaps_for_window


@dataclass(frozen=True, slots=True)
class CollectContext:
    """Everything a collector may need that is not already in the request."""

    session: AsyncSession
    request: ReportRequest
    zone: ResolvedZone
    #: The caller, for the platform principal an observation query needs.
    access: Any
    #: Vision OS's Observation API, or `None` when synthesis is not assembled.
    exposure_api: Any = None
    vision_reason: str = ""
    #: Retention windows, so a report can say when a window reaches past them.
    observation_retention_days: int = 0
    incident_retention_days: int = 0


Collector = Callable[[CollectContext], Awaitable[tuple[tuple[Section, ...], SourceCoverage]]]


@dataclass(frozen=True, slots=True)
class ReportType:
    """One report a caller can run.

    `permissions` is **every** permission required, including the ones for the
    underlying data. It is a conjunction, not a disjunction.
    """

    id: str
    title: str
    summary: str
    permissions: tuple[Permission, ...]
    granularities: tuple[Granularity, ...]
    collectors: tuple[Collector, ...] = ()
    #: What the figures are computed from, in one sentence for the reader.
    basis: str = ""
    #: Set for a module that has no data source. Carries the Phase 2 shape
    #: through unchanged rather than inventing a second way to say the same
    #: thing.
    capability_module: str = ""
    retention_subject: str = ""
    retention_days_attr: str = ""

    @property
    def is_capability(self) -> bool:
        return bool(self.capability_module)

    def as_dict(self, *, permitted: bool) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "granularities": [g.value for g in self.granularities],
            "requires": [p.value for p in self.permissions],
            # Whether *this caller* may run it. Listed either way, so an
            # operator can see a report exists and that they cannot run it —
            # which is more useful than a menu that silently varies by account.
            "permitted": permitted,
            "kind": "capability" if self.is_capability else "data",
            "capability_module": self.capability_module,
        }


# ── Collector adapters ───────────────────────────────────────────────────────
#
# Thin wrappers so every collector has the same shape, and so a collector's own
# signature stays readable rather than taking a context object it mostly ignores.


async def _incidents(ctx: CollectContext):
    return await collectors.collect_incidents(ctx.session, ctx.request, ctx.zone)


async def _observations(ctx: CollectContext):
    return await collectors.collect_observations(
        ctx.session,
        ctx.request,
        ctx.zone,
        exposure_api=ctx.exposure_api,
        unavailable_reason=ctx.vision_reason
        or "Vision OS is not assembled in this process, so no observation can be read.",
        access=ctx.access,
    )


async def _cameras(ctx: CollectContext):
    return await collectors.collect_cameras(ctx.session, ctx.request, ctx.zone)


async def _audit(ctx: CollectContext):
    return await collectors.collect_audit(ctx.session, ctx.request, ctx.zone)


# ── The reports backed by real data ──────────────────────────────────────────

ALL_GRANULARITIES = (Granularity.TOTAL, Granularity.DAY, Granularity.WEEK, Granularity.MONTH)


DATA_REPORTS: tuple[ReportType, ...] = (
    ReportType(
        id="incident_summary",
        title="Incident summary",
        summary=(
            "Incidents raised in the period, broken down by period, severity, "
            "zone and rule. Zones are read as frozen on each incident."
        ),
        permissions=(Permission.VIEW_REPORTS, Permission.VIEW_INCIDENTS),
        granularities=ALL_GRANULARITIES,
        collectors=(_incidents,),
        basis="Rows in the durable incident store, scoped to your cameras and sites.",
        retention_subject="incidents",
        retention_days_attr="incident_retention_days",
    ),
    ReportType(
        id="hygiene_observations",
        title="Hygiene observations",
        summary=(
            "PPE observations by state and by zone. Four states are kept four, "
            "and no compliance percentage is computed from them."
        ),
        permissions=(Permission.VIEW_REPORTS, Permission.VIEW_OBSERVATIONS),
        granularities=(Granularity.TOTAL,),
        collectors=(_observations,),
        basis="Vision OS's durable observation log, read through its Observation API.",
        retention_subject="observations",
        retention_days_attr="observation_retention_days",
    ),
    ReportType(
        id="camera_estate",
        title="Camera estate and zone history",
        summary=(
            "Which cameras are configured, which are processing, and the frozen "
            "zone-assignment history that past events are attributed to."
        ),
        permissions=(Permission.VIEW_REPORTS, Permission.VIEW_CAMERAS),
        granularities=(Granularity.TOTAL,),
        collectors=(_cameras,),
        basis="The camera table and the zone-assignment history, as configured now.",
    ),
    ReportType(
        id="audit_activity",
        title="Audit activity",
        summary=(
            "Counts of audited actions and the principals that performed them. "
            "Individual rows are deliberately not exported."
        ),
        permissions=(Permission.VIEW_REPORTS, Permission.VIEW_AUDIT),
        granularities=(Granularity.TOTAL,),
        collectors=(_audit,),
        basis="The append-only audit trail for this organisation.",
        retention_subject="the audit trail",
        retention_days_attr="audit_retention_days",
    ),
    ReportType(
        id="operations_overview",
        title="Operations overview",
        summary=(
            "Incidents, hygiene observations and the camera estate in one "
            "document. Each source states separately whether it could be read."
        ),
        permissions=(
            Permission.VIEW_REPORTS,
            Permission.VIEW_INCIDENTS,
            Permission.VIEW_OBSERVATIONS,
            Permission.VIEW_CAMERAS,
        ),
        granularities=ALL_GRANULARITIES,
        collectors=(_incidents, _observations, _cameras),
        basis="Three stores, each reporting its own availability independently.",
        retention_subject="observations",
        retention_days_attr="observation_retention_days",
    ),
)


# ── The seven modules with no data source ────────────────────────────────────
#
# Each is a real, runnable report that produces the module's own Phase 2
# capability answer. Included so a scope claiming to cover "everything" does not
# quietly omit them — an absent section reads as nothing to report, which is the
# exact confusion these modules' pages already exist to prevent.


def capability_report(
    *,
    report_id: str,
    title: str,
    module: str,
    permission: Permission,
) -> ReportType:
    return ReportType(
        id=report_id,
        title=title,
        summary=(
            f"{title} has no data source yet. This report states that, with the "
            "module's own reason and the inputs it is waiting for."
        ),
        permissions=(Permission.VIEW_REPORTS, permission),
        granularities=(Granularity.TOTAL,),
        capability_module=module,
        basis="",
    )


MODULE_REPORTS: tuple[ReportType, ...] = (
    capability_report(
        report_id="module_people_counting",
        title="People counting",
        module="people_counting",
        permission=Permission.VIEW_PEOPLE_COUNT,
    ),
    capability_report(
        report_id="module_demography",
        title="Demography",
        module="demography",
        permission=Permission.VIEW_DEMOGRAPHY,
    ),
    capability_report(
        report_id="module_table_occupancy",
        title="Table occupancy",
        module="table_occupancy",
        permission=Permission.VIEW_TABLE_OCCUPANCY,
    ),
    capability_report(
        report_id="module_cutting_board",
        title="Cutting board compliance",
        module="cutting_board",
        permission=Permission.VIEW_CUTTING_BOARD,
    ),
    capability_report(
        report_id="module_meal_detection",
        title="Meal detection",
        module="meal_detection",
        permission=Permission.VIEW_MEAL_DETECTION,
    ),
    capability_report(
        report_id="module_pos_integration",
        title="POS / ERP integration",
        module="pos_integration",
        permission=Permission.VIEW_POS_INTEGRATION,
    ),
    capability_report(
        report_id="module_patron_id",
        title="Unique patron ID",
        module="patron_id",
        permission=Permission.VIEW_PATRON_ID,
    ),
)


CATALOGUE: tuple[ReportType, ...] = DATA_REPORTS + MODULE_REPORTS

BY_ID: dict[str, ReportType] = {report.id: report for report in CATALOGUE}


def report_for(report_id: str) -> ReportType | None:
    return BY_ID.get(report_id)


def permitted(report: ReportType, access: Any) -> bool:
    """Every permission the report names, held. A conjunction, never a subset."""
    return all(access.has(permission) for permission in report.permissions)


# ── Building the report ──────────────────────────────────────────────────────


async def build(report: ReportType, ctx: CollectContext) -> ReportData:
    """Run a report type's collectors and assemble its coverage.

    Coverage is assembled here rather than by each collector, so a report cannot
    be complete-by-omission: `Coverage.complete` reads every source, and a
    collector that reported unavailable makes the whole report incomplete
    whether or not it wanted to.
    """
    if report.is_capability:
        return await _build_capability(report, ctx)

    sections: list[Section] = []
    coverages: list[SourceCoverage] = []

    for collect in report.collectors:
        produced, coverage = await collect(ctx)
        sections.extend(produced)
        coverages.append(coverage)

    retention_days = 0
    if report.retention_days_attr == "observation_retention_days":
        retention_days = ctx.observation_retention_days
    elif report.retention_days_attr == "incident_retention_days":
        retention_days = ctx.incident_retention_days

    gaps = list(
        gaps_for_window(
            since=ctx.request.since,
            until=ctx.request.until,
            retention_days=retention_days,
            retention_subject=report.retention_subject,
        )
    )

    # A source that could not be read, and a section that was cut short, each
    # become a stated gap as well as a false `complete`. One of them is a
    # machine-readable flag; the other is the sentence a person reads.
    for coverage in coverages:
        if not coverage.available:
            gaps.append(
                Gap(
                    kind="source_unavailable",
                    detail=(
                        f"{coverage.source}: {coverage.reason} Figures below do "
                        "not include this source, and its absence is not a zero."
                    ),
                )
            )
        if coverage.truncated:
            gaps.append(
                Gap(
                    kind="truncated",
                    detail=(
                        f"{coverage.source}: more rows exist than this report "
                        f"will read ({ctx.request.row_limit}). The figures cover "
                        "only what was read — narrow the period for a complete "
                        "answer."
                    ),
                )
            )

    return ReportData(
        report_id=report.id,
        title=report.title,
        subtitle=report.summary,
        coverage=Coverage(
            since=ctx.request.since,
            until=ctx.request.until,
            timezone=ctx.request.timezone,
            timezone_resolved=ctx.request.timezone_resolved,
            granularity=ctx.request.granularity,
            sources=tuple(coverages),
            gaps=tuple(gaps),
            basis=report.basis,
        ),
        sections=tuple(sections),
        generated_at=datetime.now(UTC),
    )


async def _build_capability(report: ReportType, ctx: CollectContext) -> ReportData:
    """A module report, from the Phase 2 capability definition itself.

    Reuses the module's own `ModuleCapability` rather than restating it, so the
    report says exactly what the module's page says, in the module's own words,
    and there is one copy of every answer.
    """
    from app.api import analytics, integrations, patron

    definitions = {
        "people_counting": analytics.PEOPLE_COUNTING,
        "demography": analytics.DEMOGRAPHY,
        "table_occupancy": analytics.TABLE_OCCUPANCY,
        "cutting_board": analytics.CUTTING_BOARD,
        "meal_detection": analytics.MEAL_DETECTION,
        "pos_integration": integrations.POS_INTEGRATION,
        "patron_id": patron.PATRON_ID,
    }
    capability = definitions[report.capability_module]

    awaiting = tuple({"id": r.id, "detail": r.detail} for r in capability.requirements)

    section = Section(
        key="awaiting",
        title="What this module is waiting for",
        columns=(
            Column("id", "Input"),
            Column("detail", "Required before it can be connected"),
        ),
        rows=awaiting,
        note=(
            "Each of these is a real-world input rather than a task. Until they "
            "exist there is nothing to report, and this report will not invent it."
        ),
        empty_note="This module declares no outstanding inputs.",
    )

    return ReportData(
        report_id=report.id,
        title=report.title,
        subtitle=capability.purpose,
        coverage=Coverage(
            since=ctx.request.since,
            until=ctx.request.until,
            timezone=ctx.request.timezone,
            timezone_resolved=ctx.request.timezone_resolved,
            granularity=ctx.request.granularity,
            sources=(
                SourceCoverage(
                    source=capability.module,
                    # False, always. The module has no data source, and this is
                    # the same field a real report uses — so a client that
                    # already branches on availability needs no special case.
                    available=False,
                    reason=capability.reason,
                ),
            ),
            gaps=(
                Gap(
                    kind="source_unavailable",
                    detail=capability.reason,
                    since=ctx.request.since,
                    until=ctx.request.until,
                ),
            ),
            basis="No data source is connected to this module.",
        ),
        sections=(section,),
        capability_state=capability.state,
        capability_reason=capability.reason,
        awaiting=awaiting,
        generated_at=datetime.now(UTC),
    )


__all__ = [
    "CATALOGUE",
    "DATA_REPORTS",
    "MODULE_REPORTS",
    "CollectContext",
    "ReportType",
    "build",
    "permitted",
    "report_for",
]
