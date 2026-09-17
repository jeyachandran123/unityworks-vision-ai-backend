---
name: prod-readiness
description: Use when deciding whether unityworks-vision-ai-backend and unityworks-vision-ai-frontend can be released, merged to main, deployed or tagged; when asked "are we ready to ship", "go/no-go", or for a release checklist; and before running /ship.
---

# Production Readiness

## Overview

A go/no-go for the Vision AI pair. Every item is a command with an observable result — nothing is
ticked from memory or from a previous session. Any **blocker** row that is red or could not be run
makes the verdict *no-go*.

Locate both repositories first, exactly as the `contract-sync` skill shows (by `.claude/team.conf` kind,
never by folder name); it sets `$BACKEND` and `$FRONTEND`. Stop if it fails. Run backend commands in
`$BACKEND`, frontend commands in `$FRONTEND`.
Backend python is `.venv/Scripts/python.exe` (bare `python` is not on PATH on the main workstation). `gh` is not installed, so CI status is read from the GitHub web UI
or reported as *unverified* — never assumed green.

## 1. Gates (blocker)

| Check | Command | Pass |
|---|---|---|
| Backend lint | `ruff check app tests` · `black --check app tests` | exit 0 |
| Backend tests | `pytest` | exit 0 (no `--cov`; it disables the timing-budget tests) |
| Contract exported | `python scripts/export_openapi.py --check` | "is up to date" |
| Frontend | `npm run verify` | exit 0 |
| Frontend CI contract | reproduce with the pin — `/check contract`, or see `contract-sync` | exit 0 |

If `python scripts/export_openapi.py` fails with `No module named 'app.configuration…'`, the venv's
editable install points at an old checkout path: `.venv/Scripts/python.exe -m pip install --no-deps -e .`

## 2. Source state (blocker)

```bash
for r in "$BACKEND" "$FRONTEND"; do
  git -C "$r" status --short; git -C "$r" status -sb | head -1; git -C "$r" log --oneline origin/main..HEAD | wc -l   # commits not yet on main (informational)
done
cat "$FRONTEND/.github/backend-schema.sha"; git -C "$BACKEND" rev-parse HEAD
```

- Working trees clean; branches not ahead of their remote (unpushed work is not releasable).
- The pin names a pushed backend commit whose `docs/api/openapi.json` matches the frontend's types.

## 3. Database (blocker)

```bash
alembic heads            # exactly one head
alembic current          # against the TARGET database, with its settings — equals heads
```

- New revisions since the last release are listed and each has been run `upgrade` on a copy of real
  data. Downgrade is not a rollback plan here (`destructive-bash` blocks it); a backup is.
- Observation partitions: if `runtime_identity` changed, `python -m scripts.migrate_observation_partitions --dry-run` was reviewed.

## 4. Production configuration (blocker)

`assert_production_safe()` refuses to boot under `APP_ENV=production` on default `SECRET_KEY` or
`DB_PASSWORD`, `APP_DEBUG=true`, or `RS256` without key paths. It does **not** check the rest —
confirm each deliberately:

| Setting | Expected | Why it is a decision |
|---|---|---|
| `CORS_ORIGINS` | the real frontend origin only | not enforced at boot, despite docs/deployment/README.md implying so |
| `SERVE_FRAMES`, `ALLOW_EVIDENCE` | `false` unless a written decision says otherwise | CCTV imagery of identifiable people leaves the process |
| `FEATURE_DEVTOOLS` | `false` | mounts `/api/v1/devtools/*` at all |
| `FEATURE_LIVE_CCTV`, `VISION_AUTOSTART`, `VISION_BIND_PERCEPTION` | as the deployment intends | a session may dial a DVR |
| Single origin | frontend and `/api`, `/ws` behind one origin | refresh cookie is `SameSite=Strict`; `secure` follows `APP_ENV=production` |
| `.env` | not in the image; mounted as a secret | |

## 5. Invariants (blocker)

The six backend and five frontend invariants hold — proven with `unityworks-team:invariant-audit`,
not by reading their test names. A full suite pass covers most; mutate at least the ones this
release's diff touches.

## 6. Documentation drift (major, not blocker)

- Backend `README.md` still claims product routes do not exist.
- Either repo's `CLAUDE.md` claiming "no CI workflow".
- Ports documented as `8000` where the Vision pair runs on `8010`.

## Verdict

Write it with `unityworks-team:evidence-report` as `docs/reviews/<date>/release.md` in the backend:
**go**, **go after blockers** (list them), or **no-go** — plus every check that could not be run, by
name. An unrun check is never reported as passing.
