# release-scribe — Vision AI backend brief

## Evidence to read
`README.md`, `CLAUDE.md`, the CLAUDE.md of the directory containing this repository and its frontend,
`docs/deployment/`, `docs/configuration/`, `docs/architecture/NOT_YET_CONNECTED.md`, `.env.example`.

## Known drift to confirm (not rediscover)
- `README.md` says product routes (restaurant, camera, incident, notification, report) do not exist.
- `CLAUDE.md` says there is no CI workflow — `.github/workflows/ci.yml` exists.
- Ports documented as `8000`; the pair runs on `8010`.
- `docs/deployment/README.md` implies start-up enforces `CORS_ORIGINS`; `assert_production_safe()` does not.
- `app/vision/composition.py` cites `tests/app/test_vision_boundary.py`, which does not exist.
- Documentation describing the old `atlas/` workspace layout.

## Safe commands to run
`.venv/Scripts/python.exe -m ruff check app tests`, `… -m black --check app tests`,
`… scripts/export_openapi.py --check`, `… -m alembic heads`. Never a server, a real database or an install.
