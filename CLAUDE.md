# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope of this file

This repo (`unityworks-vision-ai-backend/`) is one of six projects under the `atlas/` workspace.
`../CLAUDE.md` covers the workspace as a whole (the other backend, the four frontends, cross-repo
port collisions); this file covers only what is true inside this tree, in more depth.

`README.md` is the operator-facing entry point and is worth reading, but it froze at **Phase 1** —
it still says "there is no restaurant, camera, incident, notification or report route". Those all
exist now (`app/api/product.py`, `administration.py`, `reports.py`, `wall.py`, …). Treat the README
as authoritative on *install, extras, roles and the six invariants*, and the code as authoritative
on what is routed.

---

## Commands

Python **3.11**. Local venv is `.venv/` (Windows: `.venv\Scripts\python.exe`).

```bash
pip install -e ".[inference,test]"   # minimum to run the app + suite
pip install -e ".[dev]"              # everything, incl. ruff/black

alembic upgrade head
uvicorn app.main:app --host 127.0.0.1 --port 8000   # or: unityworks-backend
```

### Tests

```bash
pytest                                          # everything; addopts = -q --tb=short
pytest tests/app                                # application: API, auth, domain, live runtime
pytest tests/vision_os                          # the platform suite (architecture, conformance, perf)
pytest tests/compliance                         # rule engine + dataset regression
pytest tests/tools                              # the offline eval harness
pytest tests/app/test_observations.py::test_not_visible_survives_the_fold_unchanged
pytest -k "shared_attribute_registry"
pytest --cov=app                                # coverage — explicit, never default
```

`asyncio_mode = "auto"`: async tests need no `@pytest.mark.asyncio`.

**Do not add `--cov` to `addopts`.** `tests/vision_os/conftest.py` skips its timing-budget tests when
`sys.gettrace()` is set, so making coverage the default silently deletes that coverage instead of
measuring it. Same reason `pytest --cov=app` is scoped to `app`.

### Lint

```bash
ruff check app tests
black --check app tests
```

There is **no mypy config and no CI workflow in this repo** (no `.github/`) — the workspace CLAUDE.md's
CI description belongs to `../backend/`. `ruff` and `black` exclude `vision_os/`, `compliance/` and
`tools/` wholesale: that code is migrated verbatim and lint fixes inside it are out of scope.

### Scripts

```bash
python scripts/manage.py list-users | create-user | reset-password | check-password
python scripts/manage.py grant --email you@example.com --cameras all      # camera scope
python scripts/manage.py grant-operator --email you@example.com           # platform operator
python scripts/export_openapi.py            # writes docs/api/openapi.json (committed)
python scripts/export_openapi.py --check    # fails if a route changed without regenerating
python scripts/seed_cameras.py
python scripts/migrate_observation_partitions.py
```

`/openapi.json` mounts only under `APP_DEBUG=true`, so the frontend types are generated from the
**committed export**, not from a live endpoint. Any route change means re-running `export_openapi.py`
and then `npm run types:generate` in `../unityworks-vision-ai-frontend`.

---

## The three trees, and the one-way arrow

```
app/          the application — config, auth, authorization, API, domain, composition
compliance/   the rule engine — RuleSet, ComplianceEvaluator, ObservationReader
vision_os/    the perception platform — 201 files, migrated verbatim
```

`app → compliance → vision_os`, one way, asserted by `tests/compliance/test_boundaries.py` and
`tests/app/test_migration.py`. The governing rule: **if it has a business opinion, it does not belong
in the platform.** "A person's head covering is not visible" is an observation; "that is a hygiene
violation" is a judgment. `tests/vision_os/architecture/test_boundaries.py` enforces the platform half
of that with `TestSemanticCeiling` — it greps for domain vocabulary as identifiers *and* string
literals and fails on it. `hairnet` lives in `config/policies/*.json`, never in code.

`vision_os/` is layered `core/` (contracts, stdlib-only) → `kernel/` L0 → `acquisition/` L1 →
`perception/` L2–L4 → `synthesis/` L5 → `state/` L6 → `exposure/` L7, with all external technology
confined to `adapters/` and executable port conformance kits in `conformance/`. Its own architecture
test suite additionally forbids wall-clock reads, module-level mutable singletons, concrete adapter
names in platform modules, and binding of the unimplemented ports (`EmbeddingPort`,
`IdentityResolverPort`).

---

## `app/vision/` — the composition layer, and why it is where the bugs were

This package is the only thing that joins the application to the platform, and **it is composition,
not logic**: no detection, tracking, cropping, understanding or compliance code lives here. Read the
module docstrings before changing anything in it — each one is a post-mortem of a real, silent
outage, and the shape of the module is the fix.

| module | what it joins |
|---|---|
| `composition.py` | the Vision OS composition root; calls the platform's own `build_*` bootstraps |
| `runtime.py` | one owner of "is the platform assembled?" for the process |
| `manager.py` / `session.py` / `frames.py` | camera sessions, bounded drop-oldest queues |
| `ingest.py` | the seam: a session frame becomes a Vision OS admitted frame |
| `bridges.py` | L3 → L4 — `TrackingOutcome` adapted into `RegistryRuntime.on_tracked` |
| `understanding.py` / `writeback.py` | M9 binding and the M9 → M7 write-back sink |
| `demands.py` | what the application is willing to pay to look at |
| `compliance_driver.py` | timer pass: Vision State → `compliance/` → `app/domain/incidents` |
| `analysis_loop.py` | the one worker thread CPU-bound analysis runs on |
| `wall.py` / `decision_frames.py` | live viewing (never touches perception) and evidence frames |
| `taps.py` | DevTools observability, via the platform's public bus/metrics/health only |

Five facts that will save hours:

1. **One `AttributeRegistry`, checked by identity.** `build_registry_layer(...)` and
   `build_understanding_layer(...)` must receive the *same object*. When they did not, M7 rejected
   308 of 308 attributes M9 produced, `SkipReason.FRESH_ENOUGH` never fired once, and every layer
   reported zero failures. `assert_shared_attribute_registry()` raises at assembly;
   `tests/app/test_shared_attribute_registry.py` is a mandatory regression test. Equality is not
   enough — two registries built from the same documents compare equal and drift on the next reload.
2. **No demands means no model calls, and that is correct.** M8 skips every candidate with
   `no_demand`. If `understanding.results = 0`, check `register_policy_demands` ran before you go
   looking at the VLM. `vision_demand_freshness_ms` is the cost lever.
3. **Never reach into a platform private.** `tracking.runtime._sink = registry_layer.runtime` type-
   mismatched, every call raised `TypeError`, and tracking's invariant-V9 sink guard swallowed it —
   detection and tracking ran, everything above stayed at zero, nothing said why. Adapt shapes in
   `bridges.py` and use declared seams (`AdmittedFrameConsumer`, `on_tracked`, `bus.subscribe`).
4. **Count after the call; never `except Exception: continue`.** `writeback.py` counts every
   rejection by exception class because instrumentation that counted attempts *before* the call
   reported "391 applied" when the true number was zero.
5. **Analysis is one thread, deliberately.** A single registry, tracker, metrics engine and
   `VisionStateManager` are shared by every camera and none is documented thread-safe. Per-camera
   analysis threads would trade an API-starvation bug for a correctness one. The wall is different —
   there each camera owns its decoder outright, so a thread per camera is safe there.

Startup ordering in `app/main.py`: settings → `assert_production_safe()` → logging → database →
cache → analysis thread → Vision OS → ingest → demands → compliance driver. **Configuration is the
only fatal failure**; Redis down warns, an unassembled platform warns, and both surface through
`/health/ready` and `/api/v1/status`. `app.main:app` is built by a module-level `__getattr__`, so
importing the module constructs nothing (hence the `F822` per-file-ignore).

---

## Authorization — read `app/authorization/` before touching any route

Two principals, and **nothing translates between them**:

- `AccessDecision` — tenant-scoped, mandatory `tenant_id`, carries `Permission`s. Built in exactly
  one place, `app/api/dependencies.py::current_access`, rebuilt from the database every request so a
  revoked role takes effect immediately rather than at token expiry. Routes gate with
  `Depends(requires(Permission.X))`.
- `PlatformOperator` — cross-tenant, has no `tenant_id` and no permissions, resolved only by
  `current_operator`, produced only by a row in `platform_operator_grants` written out-of-band via
  `scripts/manage.py grant-operator`. `app/api/platform.py` is the only module that uses it.

Do not "fix" this by widening `super_admin` — it is a *tenant* role, and widening it would silently
convert every existing customer's `super_admin` into a cross-customer superuser.

Effective permissions are `(role permissions ∪ GRANTs) − REVOKEs`, composed in
`app/authorization/resolver.py::decide()`. Overrides are written only through
`app/authorization/overrides.py` and camera scope only through `app/authorization/camera_scope.py` —
both hold the anti-escalation guards (no self-modification; a grantor may not confer reach it does
not hold) so a route cannot forget them.

**Camera scope is three-valued and must stay so.** Vision OS reads an empty `cameras` tuple in a
`Grant` as *every camera in the tenant*. The application's natural value for "no access yet" is an
empty list. So `ScopeBreadth` is `NONE` / `ALL_IN_TENANT` / `LISTED`, `CameraScope` cannot be built
in a confusable state, and `AccessDecision.to_grant()` raises rather than guessing. In query code,
`camera_keys is None` means tenant-wide and an **empty tuple matches nothing**.

Permissions are never implied by one another: `VIEW_EVIDENCE` is not implied by `VIEW_OBSERVATIONS`,
`DELETE_EVIDENCE` is not implied by `VIEW_EVIDENCE`, `VIEW_DEMOGRAPHY` is not implied by
`VIEW_PEOPLE_COUNT`. Each route names the exact permission it needs. Two role assignments are
load-bearing and are not bugs: `kitchen_supervisor` has no evidence access (most likely to be a
shared screen on a kitchen wall); `auditor` has no live access (reviewing the record and watching
people work are different acts).

Cross-tenant lookups return **404, not 403** — existence across a tenant boundary is itself a
disclosure. Every query is *constructed* already narrowed, never filtered after loading.

---

## Domain conventions worth knowing before you write a query

- **Four-valued semantics.** `PRESENT` / `ABSENT` / `NOT_VISIBLE` / `UNKNOWN` stay distinct all the
  way to the UI; only `ABSENT` may become a violation. The platform-observation → product-subject
  fold exists in exactly one place, `app/domain/observations.py`, imported by both `app/api/product.py`
  and `app/reporting/sources.py`, so `not_visible` cannot be collapsed into `none` in two ways.
- **Never render a computed zero for something you cannot compute.** `/status` names `coverage` in
  `not_yet_reported`; `app/api/capability.py` answers `available: false` with a reason and a checklist
  and a real `SELECT count(*)`, as a 200. An empty list that reads as a clean result is the one
  failure this product must not commit.
- **Runtime identity is `organization_id:camera_key`,** never a bare key — `camera_key` is unique only
  per organization, and `cam-01` is the key every customer picks. See `app/domain/runtime_identity.py`.
- **Zone of a past event comes from `CameraZoneAssignment` intervals,** never from joining
  `cameras.zone_id`; `None` renders as *unrecorded*.
- **Audit is append-only** with structural credential scrubbing in `_scrub()`. A refusal is audited
  with the same weight as a success, and committed before the error propagates.
- **Evidence lifecycle is a state, not an absence:** `RETAINED → EXPIRED → DELETED (tombstone)`. The
  row survives erasure so the erasure is provable. Retention has four categories with four clocks
  (`app/domain/retention.py`), and only `RESOLVED` incidents are ever pruned.
- **Incidents close for exactly two reasons:** a later grounded observation that clears the condition,
  or an explicit authorised operator action. A UI refresh is neither.
- **Credentials are references.** `Camera.credential_ref` holds `env:CCTV_PASSWORD`, never a password.
  A database dump must not be a credential dump.

---

## Configuration

`app/configuration/settings.py`. Precedence, last wins: code defaults → `.env` → environment →
`SecretProviderPort` → database. Vision OS adapters read their own `VISION_*` environment at their
own composition root — do not mirror those reads into `Settings`.

Under `APP_ENV=production` the app refuses to start on the default `SECRET_KEY` or `DB_PASSWORD`
(`assert_production_safe()`), and the error names the variable without printing it.

Off by default, and each one is a deployment decision rather than a user setting:

| flag | consequence of turning it on |
|---|---|
| `SERVE_FRAMES`, `ALLOW_EVIDENCE` | CCTV imagery of identifiable people may leave the process |
| `FEATURE_DEVTOOLS` | mounts `/api/v1/devtools/*` at all — off means 404, not 403 |
| `FEATURE_LIVE_CCTV` | a session may dial a DVR |
| `VISION_AUTOSTART` | assemble the platform at boot |
| `VISION_BIND_PERCEPTION` | bind the perception stack |

Starting a camera takes three deliberate acts: `start_configured()` from the lifespan, **and**
`FEATURE_LIVE_CCTV`, **and** a camera row that is `enabled`. Importing this application opens no
socket, and neither does running the test suite.

`tests/app/conftest.py` clears these from the environment and unsets `env_file` for every test —
without it, a developer creating a local `.env` silently changes what the suite asserts.

---

## Invariants a change must not break

Each has a test behind it; breaking one is a design change, not a fix.

1. **Vision OS is migrated, not redesigned.** `vision_os/` is byte-identical to its source and every
   internal import is relative. Lint findings inside it are pre-existing and out of scope.
2. **No `sys.path` manipulation** in `app/`, `vision_os/`, `compliance/`, `tools/` — a test parses
   every file. (`scripts/` is outside that set and one script does use it.)
3. **No sibling-repository dependency** — a test runs `import vision_os` in a subprocess from outside
   this tree. There is no `app/vision_os/`, no second copy, no compatibility shim.
4. **One `AttributeRegistry`,** checked by object identity.
5. **Three-valued compliance** survives to the edge; every declared attribute keeps its `not_visible`.
6. **Deny by default** — no grant, never an empty camera tuple.

---

## Traps

- The `app/vision/composition.py` docstring cites `tests/app/test_vision_boundary.py`; that file does
  not currently exist. The rule it describes still holds.
- `docs/architecture/NOT_YET_CONNECTED.md` is the activation checklist for the seven product modules
  that have a schema, permission, route and page but no data source and no detection logic. Read it
  before "implementing" one — the same wording is served by each module's own route.
- This codebase's characteristic failure is a capability that was fully built and never called from
  the composition root (`compliance/` existed for phases before anything invoked it; understanding
  produced 308 attributes that were all rejected). When something reports zero, suspect the wiring
  in `app/vision/` before suspecting the layer.
- Phase reports in `docs/architecture/`, `docs/production-hardening/` and `docs/forensics/` are
  historical records of decisions and outages; they are not specs for current behaviour.
- Ollama/NVIDIA VLM calls are real and slow (~11 s each in the baseline run). There is deliberately
  no "run evaluation" HTTP endpoint, and `REGISTER_DEMAND` exists as a permission and stays unwired —
  a user who could register a demand could spend the model budget directly.
- Passwords have a 12-character floor enforced at hash time (`app/auth/passwords.py`). A failing login
  with a shorter test password is a bad fixture, not an auth bug.
