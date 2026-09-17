---
name: backend-verify
description: Use before claiming any backend change is done, when choosing which tests to run for a change, when ruff, black, pytest or export_openapi --check fails, when a test is skipped unexpectedly, or when considering coverage flags in this repository.
---

# Backend Verify

## Overview

The gate is exactly what CI runs (`.github/workflows/ci.yml`), in this order. A change is done when
the whole gate is green — not when the tests you picked are green.

```bash
PY=.venv/Scripts/python.exe          # bare `python` is not on PATH on the main workstation
$PY -m ruff check app tests
$PY -m black --check app tests
$PY -m pytest
$PY scripts/export_openapi.py --check
```

Check exit codes. `addopts = -q --tb=short`, and adding your own `-q` suppresses even the summary line.

## Scoping while iterating

Run the narrowest suite that exercises the change, then the full gate once at the end.

| Changed | Run first |
|---|---|
| `app/api/**`, `app/auth/**`, `app/authorization/**` | `pytest tests/app` |
| `app/vision/**` | `pytest tests/app/test_live_runtime.py tests/app/test_shared_attribute_registry.py tests/app/test_subject_evidence_wiring.py` |
| `app/domain/observations.py`, `app/reporting/**` | `pytest tests/app/test_observations.py tests/app/test_reports.py` |
| `migrations/**`, `app/*/models.py` | `pytest tests/app/test_membership_migration.py tests/app/test_persistence.py` |
| `config/policies/**`, `config/rules/**` | `pytest tests/compliance tests/app/test_freshness_regression.py` |
| `vision_os/**`, `compliance/**` (rare, deliberate) | `pytest tests/vision_os tests/compliance tests/app/test_migration.py` |
| `tools/**` | `pytest tests/tools` |

One test: `pytest tests/app/test_observations.py::test_not_visible_survives_the_fold_unchanged`
By keyword: `pytest -k shared_attribute_registry`. `asyncio_mode = "auto"` — no asyncio marker needed.

## Coverage — never default

```bash
$PY -m pytest --cov=app tests/app      # explicit, on demand
```

Never add `--cov` to `addopts` or CI. `tests/vision_os/conftest.py` skips its timing-budget tests
when `sys.gettrace()` is set, so coverage-by-default silently deletes that coverage instead of
measuring it. The same applies to debuggers: a timing test "skipped" under a debugger is expected.

## Lint scope

`ruff` and `black` exclude `vision_os/`, `compliance/` and `tools/` (migrated verbatim). Never run
`ruff check .`, `ruff --fix` or `black .` across the repo — it rewrites migrated code and breaks
`tests/app/test_migration.py`. For `app tests` only, `ruff check --fix app tests` is safe; review the
diff, then re-run `black`.

## Failure reading

| Symptom | Meaning |
|---|---|
| `No module named 'app.configuration…'` from a script | venv's editable install points at an old checkout path → `$PY -m pip install --no-deps -e .` |
| `ERROR at setup` … `OSError: could not get source code` in `pytest_asyncio/plugin.py` | Stale bytecode from before the repo moved: pytest's rewrite caches match by mtime, not path → `find app compliance vision_os tools tests scripts migrations -type d -name __pycache__ -prune -exec rm -rf {} +` (gitignored, regenerated) |
| `.venv/Scripts/alembic.exe` (or any launcher) fails to start | Launchers embed the venv's original path → always `$PY -m alembic`, `$PY -m pytest` |
| Tests change behaviour after creating `.env` | Should not happen: `tests/app/conftest.py` clears env and `env_file`. If it does, that isolation broke — fix it, don't delete `.env` |
| `export_openapi.py --check` fails | Route surface changed → `contract-sync` |
| Login fails in a new test | Password under 12 characters — the fixture default is `correct-horse-battery` |
| `test_membership_migration.py` slow | Expected; it shells out to alembic four times on real SQLite |

## Reporting

Paste the real final lines of each command. If a step could not run, say which and why — never
report the gate as green with a step missing.
