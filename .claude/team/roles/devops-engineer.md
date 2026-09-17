# devops-engineer — Vision AI backend brief

## Evidence to read
`.github/workflows/*.yml`, `pyproject.toml`, `alembic.ini`, `migrations/**`,
`app/configuration/settings.py`, `docs/deployment/README.md`, `docs/configuration/README.md`,
`.env.example` (never `.env`), `scripts/**`. Skills: `prod-readiness`, `db-migration`, `contract-sync`.

## What breaks here
- **Contract:** `/check contract` reproduces the frontend CI pin exactly; the pin must name a pushed
  backend commit.
- **Coverage in `addopts` or CI:** `tests/vision_os/conftest.py` skips timing-budget tests under
  `sys.gettrace()`, so default coverage silently deletes that coverage.
- **Migrations:** more than one head (`.venv/Scripts/python.exe -m alembic heads`); autogenerate
  renames as drop + add; alters without `batch_alter_table`; backfills that widen access; revisions
  touching authorization without a migration test. Exercise only with
  `DATABASE_URL_OVERRIDE=sqlite+aiosqlite:///<temp>/check.db`.
- **Production start-up:** `assert_production_safe()` enforces `SECRET_KEY`, `DB_PASSWORD`,
  `APP_DEBUG`, RS256 key paths — **not** `CORS_ORIGINS`, although `docs/deployment/README.md` implies
  it. `SERVE_FRAMES`, `ALLOW_EVIDENCE`, `FEATURE_DEVTOOLS`, `FEATURE_LIVE_CCTV` default off.
- **Ports:** the pair runs on `8010`; docs saying `8000` are stale.
- **Toolchain rot:** a venv whose editable install or bytecode points at an old checkout path (see
  `backend-verify`'s failure table).
- `gh` is not installed on the main workstation: CI status is unverified unless shown.
