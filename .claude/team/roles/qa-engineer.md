# qa-engineer — Vision AI backend brief

## Evidence to read
`tests/**`, `tests/app/conftest.py`, `tests/vision_os/conftest.py`, `.claude/team/invariants.md`.
Skill: `backend-verify`.

## What breaks here
- Coverage: run explicitly and scoped (`.venv/Scripts/python.exe -m pytest --cov=app --cov-report=term-missing <tests>`).
  Never default — timing-budget tests in `tests/vision_os/` skip under a tracer (expected skips).
- Over-granting fixtures: `make_user` / `admit` roles holding more than `permissions_for(role)` in
  `app/authorization/model.py`.
- Missing access shapes per route: allowed, refused (403), cross-tenant (404).
- Missing state shapes: `present` / `absent` / `not_visible` / `unknown`, each distinct.
- Migrations: access preserved across upgrade (`tests/app/test_membership_migration.py` is the pattern).
- Environment leakage: `tests/app/conftest.py` clears env and `env_file`; a result that changes with a
  local `.env` means that isolation broke.
- Flake: wall-clock reads, sleeps, real cameras or network.
