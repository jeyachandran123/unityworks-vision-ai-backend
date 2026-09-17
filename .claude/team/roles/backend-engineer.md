# backend-engineer — Vision AI backend brief

## Evidence to read
`app/api/**`, `app/domain/**`, `app/reporting/**`, `app/users/**`, `app/infrastructure/**`,
`app/main.py`, `migrations/versions/**` and the matching `tests/app/**`. Skills: `backend-verify`,
`api-route-change`, `db-migration` (if models or migrations changed), `vision-os-boundaries` (if
`app/vision/` changed).

## What breaks here
- Route ↔ permission mismatch; router not included in `app/main.py` `create_app`; a route without its
  refusal and cross-tenant (404) tests.
- Unbounded reports (capped at 366 days and `row_limit`).
- Computed zeros where the honest answer is `available: false` with a reason (`app/api/capability.py`).
- Zone of a past event read from `cameras.zone_id` instead of `CameraZoneAssignment` intervals.
- Incidents closed by anything other than a grounded clearing observation or an authorised action;
  retention pruning anything but `RESOLVED`.
- Evidence erased by deleting the row instead of the `RETAINED → EXPIRED → DELETED` tombstone.
- CPU-bound work (reportlab, openpyxl, image decode) on the event loop instead of `asyncio.to_thread`;
  shared vision state touched outside the single analysis thread's seams.
- Anything but configuration made fatal at start-up; import-time side effects in `app.main`.

## How to prove it
`.venv/Scripts/python.exe -m pytest <node-id>` with `tests/app/conftest.py` fixtures; SQLite in-memory
only. Full gate = the `gate` lines in `.claude/team.conf`.
