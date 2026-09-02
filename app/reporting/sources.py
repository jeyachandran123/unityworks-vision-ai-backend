"""The collectors. Each reads one real store and reports what it could not read.

Every function here returns `(sections, SourceCoverage)`. The coverage is not a
courtesy — it is how the engine knows whether `complete` is true, and a
collector that returned rows without one could not be distinguished from a
collector that found nothing.

### Scoping is constructed, never filtered afterwards

Every query is built already narrowed to the caller's tenant and camera grant,
the same discipline `app/api/product.py` documents. `camera_keys is None` is a
tenant-wide grant; an **empty tuple is none** and matches nothing.

### Frozen attribution is read as frozen

Zone columns on incidents, and the `zone_id`/`zone_name`/`zone_recorded` fields
the observation API returns, are read **as stored**. Nothing here joins
`cameras.zone_id` to find out where a past event happened. That join is exactly
the bug `camera_zone_assignments` was built to fix: it would re-attribute a
quarter of history to wherever a camera happens to sit today, and it would do it
silently, inside a report somebody signs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain import observations as observation_fold
from app.domain.models import AuditEvent, Camera, CameraZoneAssignment, Incident, Restaurant, Zone
from app.reporting.model import Column, ReportRequest, Section, SourceCoverage
from app.reporting.periods import ResolvedZone, buckets

#: What a row's zone reads as when nobody recorded one. Never an empty cell and
#: never "Unassigned": the reader has to be able to tell "this happened in no
#: zone" from "nobody wrote down which zone this happened in".
UNRECORDED_ZONE = "Not recorded"


def _scope_incidents(statement, request: ReportRequest):
    statement = statement.where(Incident.organization_id == request.organization_id)
    if request.camera_keys is not None:
        # An empty tuple matches nothing, which is the correct answer for an
        # account granted no cameras — never a wildcard.
        statement = statement.where(Incident.camera_key.in_(request.camera_keys))
    if request.restaurant_id:
        statement = statement.where(Incident.restaurant_id == request.restaurant_id)
    return statement


# ── Incidents ────────────────────────────────────────────────────────────────


async def collect_incidents(
    session: AsyncSession, request: ReportRequest, zone: ResolvedZone
) -> tuple[tuple[Section, ...], SourceCoverage]:
    """Incidents raised in the window, by period, severity, zone and rule.

    Bucketed on `created_at` — when the organisation was told — rather than on
    `observed_at`. They differ by seconds normally and by much more after an
    outage, and "how much did we raise this month" is a question about the work
    queue, not about the camera. Stated in the section note so a reader is never
    guessing which clock a figure is on.
    """
    rows = (
        (
            await session.execute(
                _scope_incidents(select(Incident), request)
                .where(Incident.created_at >= request.since, Incident.created_at < request.until)
                .order_by(Incident.created_at)
                .limit(request.row_limit + 1)
            )
        )
        .scalars()
        .all()
    )

    truncated = len(rows) > request.row_limit
    if truncated:
        rows = rows[: request.row_limit]

    earliest = await session.scalar(
        _scope_incidents(select(func.min(Incident.created_at)), request)
    )

    coverage = SourceCoverage(
        source="incidents",
        available=True,
        rows=len(rows),
        truncated=truncated,
        earliest=_aware(earliest),
    )

    # Zone **labels** only. The attribution itself is the frozen `zone_id` on
    # each incident and is never recomputed; this looks up what that id is
    # currently called so a reader sees "Prep line" rather than a hex string.
    # The caveat is stated in the section note rather than hidden: `Incident`
    # freezes the id but not the name, so renaming a zone does relabel history.
    # Naming that is honest; silently showing the new name is not.
    zone_labels = dict(
        (
            await session.execute(
                select(Zone.id, Zone.name)
                .join(Restaurant, Restaurant.id == Zone.restaurant_id)
                .where(Restaurant.organization_id == request.organization_id)
            )
        ).all()
    )

    period = _incidents_by_period(rows, request, zone)
    severity = _incidents_by_key(
        rows,
        key=lambda i: (i.severity or "unspecified"),
        column=Column("severity", "Severity"),
        title="By severity",
        note="Severity as frozen on the incident, from the rule that raised it.",
        empty_note="No incident was raised in this period, so there is nothing to break down.",
    )
    zones = _incidents_by_key(
        rows,
        # The id read as stored, then labelled. Never joined against the
        # camera's current zone, which is the re-attribution being avoided.
        key=lambda i: (zone_labels.get(i.zone_id or "", i.zone_id) or UNRECORDED_ZONE),
        column=Column("zone", "Zone"),
        title="By zone",
        note=(
            "The zone recorded on the incident when it was raised, not the "
            "camera's zone today — a camera that has since moved does not move "
            "its history. Names are looked up from the zone's current name: the "
            "incident freezes the zone id but not its label, so a renamed zone "
            "does appear under its new name."
        ),
        empty_note="No incident was raised in this period, so there is nothing to break down.",
    )
    rules = _incidents_by_key(
        rows,
        key=lambda i: (i.rule_id or "unspecified"),
        column=Column("rule", "Rule"),
        title="By rule",
        note="Rule identifiers as frozen on the incident, with the ruleset version that produced them.",
        empty_note="No incident was raised in this period, so there is nothing to break down.",
    )

    return (period, severity, zones, rules), coverage


def _incidents_by_period(
    rows, request: ReportRequest, zone: ResolvedZone
) -> Section:
    windows = buckets(request.since, request.until, request.granularity, zone)
    tallied: list[dict[str, Any]] = []

    for start, end, label in windows:
        in_bucket = [r for r in rows if start <= _aware(r.created_at) < end]
        resolved = [r for r in in_bucket if r.status == "resolved"]
        tallied.append(
            {
                "period": label,
                "raised": len(in_bucket),
                "resolved": len(resolved),
                # Open *at the end of the window*, not open now — the second
                # would change every time the report is regenerated.
                "still_open": len(in_bucket) - len(resolved),
            }
        )

    return Section(
        key="incidents_by_period",
        title="Incidents by period",
        columns=(
            Column("period", "Period"),
            Column("raised", "Raised", numeric=True),
            Column("resolved", "Resolved", numeric=True),
            Column("still_open", "Still open", numeric=True),
        ),
        rows=tuple(tallied),
        note=(
            "Bucketed on when the incident was raised, in the site's own "
            "timezone. 'Resolved' counts incidents raised in that bucket that "
            "have since been resolved, which is why it can change between runs."
        ),
        empty_note=(
            "No incident was raised in this period. The incident store was read "
            "successfully and held nothing for this window — this is a real "
            "reading, not a missing one."
        ),
    )


def _incidents_by_key(rows, *, key, column: Column, title: str, note: str, empty_note: str) -> Section:
    tally: dict[str, int] = {}
    for row in rows:
        tally[key(row)] = tally.get(key(row), 0) + 1

    ordered = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
    return Section(
        key=f"incidents_{column.key}",
        title=title,
        columns=(column, Column("count", "Incidents", numeric=True)),
        rows=tuple({column.key: name, "count": count} for name, count in ordered),
        note=note,
        empty_note=empty_note,
    )


# ── Observations (hygiene) ───────────────────────────────────────────────────


async def collect_observations(
    session: AsyncSession,
    request: ReportRequest,
    zone: ResolvedZone,
    *,
    exposure_api: Any,
    unavailable_reason: str,
    access: Any,
) -> tuple[tuple[Section, ...], SourceCoverage]:
    """PPE observations for the window, summarised by state and by zone.

    ### Availability is not emptiness, in a report as much as on a screen

    A platform that is not assembled produces `available: False` with the
    platform's own reason. It never produces an empty table, because a hygiene
    report showing no observations from a system that was not watching is the
    single most dangerous document this product could generate.

    ### The four states stay four

    Values arrive from the observation API exactly as the platform reported
    them and are resolved here by the same rule the frontend uses. `not_visible`
    is counted in its own column and is never folded into `absent` — a report
    that did so would put a number on an accusation nobody could support.
    """
    if exposure_api is None:
        return (), SourceCoverage(
            source="observations",
            available=False,
            reason=unavailable_reason,
        )

    cameras = request.camera_keys
    if cameras is None:
        registered = (
            (
                await session.execute(
                    select(Camera.camera_key).where(
                        Camera.organization_id == request.organization_id
                    )
                )
            )
            .scalars()
            .all()
        )
        cameras = tuple(registered)

    if not cameras:
        return (), SourceCoverage(
            source="observations",
            available=True,
            reason="This account reaches no camera, so no observation is in scope.",
            rows=0,
        )

    # The canonical fold, from the domain layer. One implementation, shared with
    # `app/api/product.py`: it is the single place an attribute value becomes a
    # subject record, and a second copy would be a second place `not_visible`
    # could be quietly collapsed into `none`. A test calls it through both
    # consumers and asserts identical output.
    subjects, observation_count, _ = observation_fold.query_observations(
        exposure_api, access, cameras, request.since, request.until, request.row_limit
    )

    # Zone attribution, read from the frozen assignment history exactly as the
    # observation API does it — never joined against the camera's zone today.
    from app.domain.zone_attribution import ZoneHistory

    history = await ZoneHistory.load(
        session, organization_id=request.organization_id, camera_keys=tuple(cameras)
    )
    for subject in subjects:
        attribution = history.resolve_ns(
            str(subject.get("camera_key", "")), subject.get("last_seen")
        )
        subject["zone_name"] = attribution.zone_name if attribution else ""
        subject["zone_recorded"] = attribution is not None

    coverage = SourceCoverage(
        source="observations",
        available=True,
        rows=len(subjects),
        truncated=observation_count >= request.row_limit,
    )

    return (
        _observation_states(subjects),
        _observations_by_zone(subjects),
    ), coverage


#: The PPE attributes reported, in a fixed order. Fixed so a missing attribute
#: is a row that says UNKNOWN rather than a column that quietly disappears.
PPE_KEYS = ("head_covering", "face_covering", "hand_covering")


def _resolve_state(value: str | None) -> str:
    """Raw attribute value → one of the four states.

    Deliberately conservative in one direction only: anything unrecognised is
    `unknown`, never `absent`. A value this function has not seen before is a
    reason to say nothing, not a reason to accuse somebody.
    """
    if value is None or value == "":
        return "unknown"
    lowered = str(value).strip().lower()
    if lowered in {"not_visible", "notvisible", "occluded", "obscured"}:
        return "not_visible"
    if lowered in {"unknown", "unclear", "indeterminate"}:
        return "unknown"
    if lowered in {"none", "absent", "no", "false", "missing"}:
        return "absent"
    return "present"


def _observation_states(subjects: list[dict[str, Any]]) -> Section:
    tallies: dict[str, dict[str, int]] = {
        key: {"present": 0, "absent": 0, "not_visible": 0, "unknown": 0} for key in PPE_KEYS
    }

    for subject in subjects:
        found = {a.get("key"): a.get("value") for a in subject.get("attributes", [])}
        for key in PPE_KEYS:
            tallies[key][_resolve_state(found.get(key))] += 1

    return Section(
        key="observation_states",
        title="PPE observations by state",
        columns=(
            Column("item", "Item"),
            Column("present", "Present", numeric=True),
            Column("absent", "Absent", numeric=True),
            Column("not_visible", "Not visible", numeric=True),
            Column("unknown", "Unknown", numeric=True),
        ),
        rows=tuple(
            {
                "item": key.replace("_", " ").capitalize(),
                **tallies[key],
            }
            for key in PPE_KEYS
        ),
        note=(
            "Four states, kept four. 'Not visible' means the camera could not "
            "see the item and 'Unknown' means nothing fresh was observed; "
            "neither is a violation and neither is counted as one. No "
            "compliance percentage is computed from these figures, because a "
            "percentage would have to decide what to do with the last two "
            "columns and there is no honest answer."
        ),
        empty_note="No subject was observed in this period.",
    )


def _observations_by_zone(subjects: list[dict[str, Any]]) -> Section:
    tally: dict[str, int] = {}
    for subject in subjects:
        if subject.get("zone_recorded") and subject.get("zone_name"):
            name = str(subject["zone_name"])
        elif subject.get("zone_recorded"):
            name = "No zone"
        else:
            name = UNRECORDED_ZONE
        tally[name] = tally.get(name, 0) + 1

    ordered = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
    return Section(
        key="observations_by_zone",
        title="Subjects observed by zone",
        columns=(Column("zone", "Zone"), Column("subjects", "Subjects", numeric=True)),
        rows=tuple({"zone": name, "subjects": count} for name, count in ordered),
        note=(
            "The zone the camera belonged to at the moment of observation, from "
            f"the frozen assignment history. '{UNRECORDED_ZONE}' means no "
            "assignment covered that instant — which is different from 'No "
            "zone', and is never filled in from where the camera sits today."
        ),
        empty_note="No subject was observed in this period.",
    )


# ── Cameras and structure ────────────────────────────────────────────────────


async def collect_cameras(
    session: AsyncSession, request: ReportRequest, zone: ResolvedZone
) -> tuple[tuple[Section, ...], SourceCoverage]:
    """The camera estate and its zone assignments.

    ### This is configuration, not history

    Camera *health* lives in the live runtime and is not persisted, so this
    report cannot say what percentage of the period a camera was online. It says
    so in the section note rather than computing an uptime figure from the one
    sample it happens to have — a "98% uptime" derived from the state at the
    moment somebody pressed Generate would be a fabrication of exactly the kind
    this product exists to avoid.
    """
    statement = select(Camera).where(Camera.organization_id == request.organization_id)
    if request.camera_keys is not None:
        statement = statement.where(Camera.camera_key.in_(request.camera_keys))
    if request.restaurant_id:
        statement = statement.where(Camera.restaurant_id == request.restaurant_id)

    cameras = (await session.execute(statement.order_by(Camera.camera_key))).scalars().all()

    zone_names = dict(
        (
            await session.execute(
                select(Zone.id, Zone.name).join(
                    Restaurant, Restaurant.id == Zone.restaurant_id
                ).where(Restaurant.organization_id == request.organization_id)
            )
        ).all()
    )
    site_names = dict(
        (
            await session.execute(
                select(Restaurant.id, Restaurant.name).where(
                    Restaurant.organization_id == request.organization_id
                )
            )
        ).all()
    )

    roster = Section(
        key="camera_roster",
        title="Cameras configured",
        columns=(
            Column("camera_key", "Camera"),
            Column("name", "Name"),
            Column("site", "Site"),
            Column("zone", "Zone (current)"),
            Column("enabled", "Processing"),
        ),
        rows=tuple(
            {
                "camera_key": camera.camera_key,
                "name": camera.name,
                "site": site_names.get(camera.restaurant_id, "—"),
                "zone": zone_names.get(camera.zone_id or "", UNRECORDED_ZONE),
                "enabled": "Yes" if camera.enabled else "No",
            }
            for camera in cameras
        ),
        note=(
            "Current configuration, as of generation. The 'Zone (current)' "
            "column is where each camera sits **now** — it is not what past "
            "incidents or observations are attributed to, which comes from the "
            "assignment history below. A camera that is not processing creates "
            "no session, no decode and no model call."
        ),
        empty_note="No camera is configured for this organisation.",
    )

    assignments = (
        (
            await session.execute(
                select(CameraZoneAssignment)
                .where(CameraZoneAssignment.organization_id == request.organization_id)
                .order_by(
                    CameraZoneAssignment.camera_key, CameraZoneAssignment.effective_from
                )
                .limit(request.row_limit)
            )
        )
        .scalars()
        .all()
    )

    history = Section(
        key="camera_zone_history",
        title="Zone assignment history",
        columns=(
            Column("camera_key", "Camera"),
            Column("zone", "Zone"),
            Column("from", "From"),
            Column("to", "Until"),
            Column("by", "Assigned by"),
        ),
        rows=tuple(
            {
                "camera_key": row.camera_key,
                "zone": row.zone_name or ("No zone" if row.zone_id is None else row.zone_id),
                "from": _iso(row.effective_from),
                "to": _iso(row.effective_to) or "current",
                "by": row.assigned_by or "—",
            }
            for row in assignments
        ),
        note=(
            "What a past event is attributed to. Intervals are closed rather "
            "than edited, so moving a camera never changes where its earlier "
            "readings happened. Intervals begin only from when this history "
            "started being recorded; anything earlier reads as "
            f"'{UNRECORDED_ZONE}' rather than being inferred."
        ),
        empty_note=(
            "No zone assignment has been recorded yet. Observations and "
            "incidents from before the first assignment are attributed to no "
            "zone, and are deliberately not backfilled from current mappings."
        ),
    )

    return (roster, history), SourceCoverage(
        source="cameras", available=True, rows=len(cameras)
    )


# ── Audit ────────────────────────────────────────────────────────────────────


async def collect_audit(
    session: AsyncSession, request: ReportRequest, zone: ResolvedZone
) -> tuple[tuple[Section, ...], SourceCoverage]:
    """Who did what in the window, summarised by action and by actor.

    The rows themselves are not reproduced. An audit report that copied every
    row would become a second, exportable, unretained copy of the trail — a file
    on somebody's laptop recording who looked at imagery of which named
    employee, outliving the 730-day policy that governs the original.
    """
    rows = (
        (
            await session.execute(
                select(AuditEvent.action, AuditEvent.actor, AuditEvent.outcome)
                .where(
                    AuditEvent.organization_id == request.organization_id,
                    AuditEvent.occurred_at >= request.since,
                    AuditEvent.occurred_at < request.until,
                )
                .limit(request.row_limit + 1)
            )
        )
        .all()
    )

    truncated = len(rows) > request.row_limit
    if truncated:
        rows = rows[: request.row_limit]

    by_action: dict[tuple[str, str], int] = {}
    by_actor: dict[str, int] = {}
    for action, actor, outcome in rows:
        by_action[(action, outcome)] = by_action.get((action, outcome), 0) + 1
        by_actor[actor or "—"] = by_actor.get(actor or "—", 0) + 1

    actions = Section(
        key="audit_actions",
        title="Actions recorded",
        columns=(
            Column("action", "Action"),
            Column("outcome", "Outcome"),
            Column("count", "Count", numeric=True),
        ),
        rows=tuple(
            {"action": action, "outcome": outcome, "count": count}
            for (action, outcome), count in sorted(
                by_action.items(), key=lambda kv: (-kv[1], kv[0])
            )
        ),
        note=(
            "Counts only. Individual audit rows are deliberately not exported: "
            "a file listing who viewed imagery of which employee would be a "
            "second copy of the trail, outliving the retention policy that "
            "governs the original."
        ),
        empty_note="No audited action was recorded in this period.",
    )

    actors = Section(
        key="audit_actors",
        title="By actor",
        columns=(Column("actor", "Actor"), Column("count", "Actions", numeric=True)),
        rows=tuple(
            {"actor": actor, "count": count}
            for actor, count in sorted(by_actor.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        note="Every authenticated principal that acted in this period.",
        empty_note="No audited action was recorded in this period.",
    )

    earliest = await session.scalar(
        select(func.min(AuditEvent.occurred_at)).where(
            AuditEvent.organization_id == request.organization_id
        )
    )

    return (actions, actors), SourceCoverage(
        source="audit",
        available=True,
        rows=len(rows),
        truncated=truncated,
        earliest=_aware(earliest),
    )


# ── helpers ──────────────────────────────────────────────────────────────────


def _aware(value: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes. Compare in UTC or not at all."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _iso(value: datetime | None) -> str:
    aware = _aware(value)
    return aware.isoformat() if aware else ""


__all__ = [
    "PPE_KEYS",
    "UNRECORDED_ZONE",
    "collect_audit",
    "collect_cameras",
    "collect_incidents",
    "collect_observations",
]
