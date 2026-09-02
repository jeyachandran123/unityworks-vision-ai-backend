"""Period boundaries, timezones, and the gaps a window implies before any query runs.

### Boundaries are local, storage is UTC

Every timestamp in this application is stored timezone-aware in UTC, and that is
correct. But "September" for a restaurant in Singapore begins at 16:00 UTC on
31 August, and a monthly report computed on UTC boundaries silently attributes
eight hours of every month to the wrong one. So a window is resolved in the
site's own zone — `restaurants.timezone`, which Administration already
maintains — and converted to UTC exactly once, here.

### A zone that will not resolve is reported, not swallowed

`zoneinfo` reads the operating system's tz database, and a Windows host or a
minimal container may not have one. Falling back to UTC silently would produce
boundaries wrong by up to a day with a report that looked completely confident.
So `resolve_timezone` returns *whether it resolved*, that travels into
`Coverage.timezone_resolved`, and every renderer shows it.

`tzdata` is a declared dependency for exactly this reason: the fallback exists
for the deployment that somehow lacks it, not as the normal path.

### The partial-period rule

`gaps_for_window` is where this module earns its place. A window is not complete
merely because a query returned rows:

  * a window whose end is in the future has not finished happening
  * a window whose start predates the data cannot be compared with one that does not
  * a window longer than the retention floor asks for records that were deleted
    on schedule, and their absence is a policy outcome rather than a quiet month

Each of those becomes a `Gap`, `Coverage.complete` becomes false, and the report
says so above the numbers rather than below them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.errors import ValidationError
from app.reporting.model import MAX_WINDOW_DAYS, Gap, Granularity


@dataclass(frozen=True, slots=True)
class ResolvedZone:
    name: str
    resolved: bool
    tz: ZoneInfo | None

    @property
    def effective(self):
        """The tzinfo to compute boundaries in. UTC when the zone did not resolve."""
        return self.tz or UTC


def resolve_timezone(name: str) -> ResolvedZone:
    """Look up an IANA zone, reporting failure rather than raising or guessing."""
    candidate = (name or "").strip() or "UTC"
    if candidate.upper() == "UTC":
        return ResolvedZone(name="UTC", resolved=True, tz=None)
    try:
        return ResolvedZone(name=candidate, resolved=True, tz=ZoneInfo(candidate))
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        # Not an error the caller can fix, and not a reason to refuse a report.
        # It is a reason to say the boundaries are UTC, which the coverage does.
        return ResolvedZone(name=candidate, resolved=False, tz=None)


def parse_instant(raw: str | None) -> datetime | None:
    """An ISO-8601 instant, normalised to aware UTC. `None` passes through."""
    if not raw:
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValidationError(f"'{raw}' is not an ISO-8601 instant") from exc
    # A naive instant is read as UTC rather than as the server's local time,
    # which would make a report mean something different on every host.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def resolve_window(
    *,
    since: datetime | None,
    until: datetime | None,
    zone: ResolvedZone,
    granularity: Granularity,
) -> tuple[datetime, datetime]:
    """Snap a requested window to whole local buckets and validate it.

    Snapping matters: a "monthly" report over 3 September to 3 October is two
    partial months presented as two months. Boundaries are aligned in the site's
    own zone and returned in UTC.
    """
    now = datetime.now(UTC)
    end = until or now
    start = since or (end - timedelta(days=30))

    if end < start:
        raise ValidationError("'since' must not be later than 'until'")
    if (end - start) > timedelta(days=MAX_WINDOW_DAYS):
        raise ValidationError(
            f"a report window may not exceed {MAX_WINDOW_DAYS} days; "
            "narrow the period rather than scanning further",
            details={"max_days": MAX_WINDOW_DAYS},
        )

    if granularity is Granularity.TOTAL:
        return start, end

    tz = zone.effective
    local_start = start.astimezone(tz)
    local_end = end.astimezone(tz)

    aligned_start = _floor(local_start, granularity)
    # The end snaps *up* to the next boundary so the final bucket is a whole
    # one. Whether that bucket has finished happening is a separate question,
    # and `gaps_for_window` answers it rather than this function pretending it
    # cannot arise.
    aligned_end = _ceil(local_end, granularity)

    return aligned_start.astimezone(UTC), aligned_end.astimezone(UTC)


def _floor(moment: datetime, granularity: Granularity) -> datetime:
    midnight = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    if granularity is Granularity.DAY:
        return midnight
    if granularity is Granularity.WEEK:
        # ISO weeks: Monday is day 0. A retail week starting Sunday is a real
        # variant and a real configuration decision; until somebody supplies it,
        # ISO is the standard rather than a guess.
        return midnight - timedelta(days=midnight.weekday())
    return midnight.replace(day=1)


def _ceil(moment: datetime, granularity: Granularity) -> datetime:
    floored = _floor(moment, granularity)
    if floored == moment:
        return moment
    if granularity is Granularity.DAY:
        return floored + timedelta(days=1)
    if granularity is Granularity.WEEK:
        return floored + timedelta(days=7)
    year, month = floored.year, floored.month
    return floored.replace(year=year + 1, month=1) if month == 12 else floored.replace(month=month + 1)


def buckets(
    since: datetime, until: datetime, granularity: Granularity, zone: ResolvedZone
) -> list[tuple[datetime, datetime, str]]:
    """The (start, end, label) buckets a window divides into, in local terms.

    Labels are local dates because that is what a reader recognises; the
    boundaries are UTC because that is what the rows are stored in.
    """
    if granularity is Granularity.TOTAL:
        return [(since, until, "Whole period")]

    tz = zone.effective
    out: list[tuple[datetime, datetime, str]] = []
    cursor = since.astimezone(tz)
    end_local = until.astimezone(tz)

    # Bounded by the window cap above, so this cannot run away: 366 daily
    # buckets is the ceiling.
    while cursor < end_local and len(out) <= MAX_WINDOW_DAYS + 1:
        nxt = _ceil(cursor + timedelta(seconds=1), granularity)
        if nxt > end_local:
            nxt = end_local
        if granularity is Granularity.DAY:
            label = cursor.strftime("%Y-%m-%d")
        elif granularity is Granularity.WEEK:
            label = f"Week of {cursor.strftime('%Y-%m-%d')}"
        else:
            label = cursor.strftime("%Y-%m")
        out.append((cursor.astimezone(UTC), nxt.astimezone(UTC), label))
        cursor = nxt
    return out


def gaps_for_window(
    *,
    since: datetime,
    until: datetime,
    retention_days: int | None = None,
    retention_subject: str = "",
) -> tuple[Gap, ...]:
    """Everything that makes this window less than a whole story, before any query.

    Computed from the window alone, so it holds even for a report whose sources
    all answered perfectly. A month that has not finished is not a month,
    however many rows came back for it.
    """
    now = datetime.now(UTC)
    found: list[Gap] = []

    if until > now:
        found.append(
            Gap(
                kind="future",
                detail=(
                    "This period has not finished. Figures cover it up to now "
                    "and will change; they are not comparable with a completed "
                    "period of the same length."
                ),
                since=now,
                until=until,
            )
        )

    if retention_days and retention_days > 0:
        floor = now - timedelta(days=retention_days)
        if since < floor:
            found.append(
                Gap(
                    kind="before_history",
                    detail=(
                        f"The window starts before the {retention_days}-day "
                        f"retention floor for {retention_subject or 'this data'}. "
                        "Records older than that were deleted on schedule, so "
                        "their absence is a retention outcome and not a quiet "
                        "period."
                    ),
                    since=since,
                    until=floor,
                )
            )

    return tuple(found)


__all__ = [
    "ResolvedZone",
    "buckets",
    "gaps_for_window",
    "parse_instant",
    "resolve_timezone",
    "resolve_window",
]
