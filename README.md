# UnityWorks Vision AI — Backend

The production backend, and the home of the **Vision OS** perception platform.

```
unityworks-vision-ai-backend/
├── vision_os/      the perception platform — migrated verbatim, 201 files
├── compliance/     the rule engine that consumes it — outside the platform
├── app/            the application: config, auth, authorization, API, lifecycle
├── config/         semantic policies and compliance rules — the domain, as data
├── models/         yolov8n.onnx, the production detector
├── tools/          the offline evaluation harness
├── migrations/     Alembic
└── tests/
```

The dependency direction is one-way and asserted by a test:
`app → compliance → vision_os`. The platform never imports the application. The
moment it does, it has acquired a business opinion.

---

## Install

Requires **Python 3.11**.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[inference,test]"
```

Extras, and why they are separate:

| extra | brings | needed when |
|---|---|---|
| *(base)* | fastapi, sqlalchemy, redis, PyJWT, bcrypt, loguru | always |
| `inference` | numpy, onnxruntime | `VISION_DETECTOR_PROVIDER=yolo` |
| `video` | av (PyAV) | decoding video — Phase 3 |
| `eval` | numpy, onnxruntime, av, Pillow | running `tools/vision_eval` |
| `test` | pytest, httpx, aiosqlite | running the suite |
| `dev` | all of the above + ruff, black | development |

`vision_os.core` is stdlib-only by contract, so the base install can import the
platform's object model and every port without a CV stack present. The heavy
libraries arrive only with the adapters that use them.

## Configure

```bash
cp .env.example .env
```

Then set, at minimum, `SECRET_KEY` and `DB_PASSWORD`. **In `APP_ENV=production`
the application refuses to start with default values**, and the error names the
variable without printing it.

Resolution order, later wins:

```
defaults in code → .env → environment → SecretProvider (Phase 2) → database (Phase 4)
```

Two settings decide whether CCTV imagery of identifiable people can leave the
process. Both default to **off**, both are deployment decisions, and neither is a
user setting:

```
SERVE_FRAMES=false
ALLOW_EVIDENCE=false
```

## Run

```bash
alembic upgrade head
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Or `unityworks-backend`, which reads host and port from settings.

### Endpoints in this phase

| route | auth |
|---|---|
| `GET /health` | none — liveness, discloses nothing |
| `GET /health/ready` | none — booleans only |
| `POST /api/v1/auth/login` | none |
| `POST /api/v1/auth/refresh` | none (refresh token) |
| `GET /api/v1/auth/me` | bearer |
| `GET /api/v1/status` | bearer — operator-facing |
| `GET /api/v1/devtools/vision` | bearer + `access_devtools` + `FEATURE_DEVTOOLS` |
| `GET /metrics` | none — expose on an internal network only |

There is no restaurant, camera, incident, notification or report route. Those
arrive in Phase 4 with the domain they describe.

`/docs`, `/redoc` and `/openapi.json` are mounted only when `APP_DEBUG=true`. A
schema is a map of the attack surface.

## Test

```bash
pytest                         # everything
pytest tests/vision_os         # the platform suite
pytest tests/app               # application foundation + migration invariants
pytest --cov=app               # coverage, explicitly
```

Coverage is deliberately **not** in the default `addopts`. The platform suite
carries timing budgets that are meaningless under a trace function and skips them
when `sys.gettrace()` is set — making coverage the default would silently disable
that coverage instead of measuring it.

## Roles

| role | reaches |
|---|---|
| `super_admin` | everything, including DevTools |
| `org_admin` | organization, users, observations, evidence, demands |
| `restaurant_manager` | live, observations, evidence, camera health |
| `kitchen_supervisor` | live, observations, camera health — **no evidence** |
| `hygiene_officer` | observations, evidence, camera health |
| `auditor` | observations, evidence — **no live** |
| `developer` | product surfaces + DevTools + demands |

Two assignments are load-bearing. `kitchen_supervisor` has no evidence access
because it is the role most likely to be a shared screen on a kitchen wall.
`auditor` has no live access because reviewing the record and watching people
work are different acts with different purposes.

## The rules this repository keeps

1. **Vision OS is migrated, not redesigned.** `vision_os/` is byte-identical to
   its source. Every internal import is relative, which is why the rename needed
   no edit inside it.
2. **No `sys.path` manipulation.** A test parses every source file and fails on
   any call to `sys.path.insert/append/extend`.
3. **No sibling-repository dependency.** A test runs `import vision_os` in a
   subprocess from outside this tree.
4. **One AttributeRegistry.** M7 and M9 share one instance, checked by identity
   at assembly and by a mandatory regression test. Phase 6 spent nine sub-phases
   discovering what happens when they do not.
5. **Three-valued compliance.** `PRESENT` / `ABSENT` / `NOT_VISIBLE` / `UNKNOWN`
   stay distinct. Every declared attribute keeps its `not_visible` value.
6. **Deny by default.** A user with no camera grant gets no Vision OS grant at
   all — never an empty camera tuple, which the platform reads as *every camera*.

Details and evidence: [docs/architecture/PHASE_1_IMPLEMENTATION.md](docs/architecture/PHASE_1_IMPLEMENTATION.md).
