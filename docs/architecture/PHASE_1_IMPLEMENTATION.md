# Phase 1 — Production Backend Foundation + Vision OS Migration

**UnityWorks Vision AI · 2026-08-20**

## Result: **PASS**

Every gate in §35 of the brief is met. Numbers below are measured, not estimated.

---

## 1. Executive summary

`atlas/unityworks-vision-ai-backend` exists and owns Vision OS. The platform is
imported as `vision_os` from a real package, in a clean virtual environment, with
no `sys.path` manipulation and no dependency on any sibling repository.

| | |
|---|---|
| **Full suite** | **2,994 tests · 0 failures · 0 errors · 9 skipped · 117.4 s** |
| — platform | 2,809 (94 test files) |
| — compliance | 71 (4 test files) |
| — application + migration | 114 (4 test files) |
| Platform files migrated | **201 — byte-identical** (`diff -rq` reports no difference) |
| Import sites rewritten | 1,102 `app.vision_os` + 9 `app.compliance` → **0 remaining** |
| New application code | 24 Python modules |
| Reference repositories | `frontend`, `vision_os_demo`, `vision_os_validation_console` **clean**; `backend` carries two edits **made by the user during this session** — see §12 |

**The single most valuable finding:** Vision OS uses relative imports throughout.
All 201 files moved without a single content edit, which is why this is a
migration rather than a rewrite.

**The single most urgent finding:** a live NVIDIA API key is present in
`atlas/backend/.env.example`, a git-tracked file. Details in §12. Not introduced
by this phase and not modified by it.

---

## 2. Repository structure

```
unityworks-vision-ai-backend/
├── vision_os/          201 files — the platform, verbatim
├── compliance/           5 files — the rule engine, imports rewritten only
├── app/                 24 files — the application foundation
│   ├── configuration/   settings, production hardening
│   ├── auth/            passwords, tokens, service
│   ├── authorization/   roles, permissions, scope model, resolver
│   ├── users/           Organization, User, RoleAssignment, AccessGrant
│   ├── infrastructure/  database, cache, observability
│   ├── vision/          composition root + runtime  ← the integration layer
│   ├── api/             routes, dependencies
│   ├── errors.py        the error envelope
│   └── main.py          the FastAPI factory
├── config/             policies + rules — the domain, as data
├── models/             yolov8n.onnx
├── tools/vision_eval/  the offline evaluation harness
├── migrations/         Alembic + one revision
├── tests/              102 test files
├── docs/               architecture · configuration · deployment
├── pyproject.toml · alembic.ini · .env.example · .gitignore · README.md
```

---

## 3. Vision OS migration

### The finding that made it cheap

Every internal import inside `vision_os/` is **relative** (`from ...core.model.ids
import CameraId`). Not one absolute self-reference exists in 201 files.

**Consequence: the rename required zero edits inside the platform.** Verified:

```
diff -rq atlas/backend/app/vision_os  ./vision_os  -x __pycache__
→ no differences
```

This was audited before any file was copied, exactly as §5 required. An
uncontrolled global replace would have produced the same result here by luck; the
audit is what turned that into knowledge, and it is also what surfaced the two
problems in §4.

### What was rewritten

| target | occurrences | files |
|---|---|---|
| `app.vision_os` → `vision_os` | 1,102 | 111 |
| `app.compliance` → `compliance` | 9 | 4 |

All in *consumers*: `compliance/` (4 import lines), the migrated test suites, and
`tools/vision_eval`. Verified zero remaining by
`tests/app/test_migration.py::test_no_module_references_the_old_package_name`.

### What came with it

| asset | note |
|---|---|
| `tests/vision_os/` | 94 test files, unchanged except import paths |
| `tests/compliance/` | 4 test files |
| `config/policies/*`, `config/rules/*` | 6 documents, byte-identical |
| `models/yolov8n.onnx` | the production detector, 13 MB |
| `tools/vision_eval/` | 12 modules — the programme's measurement instrument |

---

## 4. Two problems the migration surfaced

Both were found by tests written for this phase, and both are recorded because
they are non-obvious and would have been rediscovered painfully later.

### 4.1 `tests/vision_os/` shadowed the `vision_os` package

The platform suite failed to collect with:

```
ModuleNotFoundError: No module named 'vision_os.conformance.kit'
```

— while `import vision_os.conformance.kit` worked perfectly from a shell.

**Cause.** `tests/vision_os/__init__.py` existed; `tests/__init__.py` did not.
pytest's `prepend` import mode therefore put `tests/` on `sys.path`, and
`import vision_os` resolved to the **test directory**.

**Under `app.vision_os` this could not happen.** The rename created the
collision.

**Fix.** One file: `tests/__init__.py`, carrying the explanation. pytest now walks
to the repository root and imports the suite as `tests.vision_os.*`. Renaming the
directory would also have worked and was rejected — `tests/vision_os` is where
every engineer on this programme expects the platform suite to be.

### 4.2 `tools/vision_eval/pose_observability.py` mutated `sys.path`

```python
sys.path.insert(0, str(args.dataset))
from head_locations import HEAD_BANDS
```

Not a sibling-repository import — it loads a dataset-local ground-truth module.
Still a `sys.path` mutation, and §3 forbids those outright.

**Fix.** `importlib.util.spec_from_file_location`, loading the file by path. Also
strictly better: a dataset directory on `sys.path` shadows every module sharing a
name with a file in it, for the rest of the process.

Caught by `test_no_source_file_mutates_sys_path`, which parses the AST rather
than grepping — so the phrase appearing in a docstring (as it does here) is not a
false positive.

---

## 5. Removing the sibling-repository coupling

### What was removed

```python
# the validation harness — assembly.py:66-84
def ensure_importable(vision_os_root: Path | None = None) -> Path:
    """Put the backend on `sys.path` so `app.vision_os` resolves."""
    sys.path.insert(0, str(resolved))
```

Nothing of that shape exists here. Four tests enforce it:

| test | asserts |
|---|---|
| `test_no_source_file_mutates_sys_path` | no `sys.path.insert/append/extend` in any source file (AST-parsed) |
| `test_no_source_file_names_a_sibling_repository` | no reference to `atlas/backend`, the console, the demo, or `vosvc_harness` |
| `test_the_platform_imports_in_a_subprocess_with_no_extra_path` | `import vision_os` in a subprocess, **cwd outside the repository** |
| `test_there_is_exactly_one_vision_os_package` | no `app/vision_os/`, no second copy, no shim |

### Verified

```
cwd          : C:\Users\Jayachandran\ProjectsAndDocs         ← outside the repo
vision_os    : ...\unityworks-vision-ai-backend\vision_os\__init__.py
sys.path unchanged by import: True
```

---

## 6. Dependencies

Derived from an import audit, not inherited.

### What the platform actually needs

| finding | evidence |
|---|---|
| `vision_os.core` is stdlib-only | asserted by `test_core_imports_no_third_party_library` |
| HTTP is `urllib.request` | `adapters/understanding/nvidia_vl.py:42` — **no httpx/requests anywhere** |
| `numpy`, `onnxruntime` are lazy adapter imports | `adapters/models/runtimes.py:215,386,564` |
| `torch` is an optional CUDA probe | `adapters/models/devices.py:75` — absent torch means "no CUDA" |
| `ultralytics` is never imported | appears in comments and a class name only |

### Declared

| group | packages |
|---|---|
| base | fastapi · uvicorn · python-multipart · pydantic · pydantic-settings · sqlalchemy · alembic · asyncpg · redis · PyJWT · bcrypt · prometheus-client · loguru |
| `[inference]` | numpy · onnxruntime |
| `[video]` | av |
| `[eval]` | numpy · onnxruntime · av · Pillow |
| `[test]` | pytest · pytest-asyncio · pytest-cov · httpx · aiosqlite |
| `[lint]` | ruff · black |

**Phase 0 finding R-05 is closed.** `atlas/backend/requirements.txt` declared none
of numpy, onnxruntime or av while importing all three at runtime — a clean
install could not execute detection. They are now declared.

### Not carried

langgraph · langchain-core · langchain-community · chromadb · seven tree-sitter
packages · gitpython · pypdf · python-docx · openpyxl · python-pptx · langdetect ·
reportlab · pandas · boto3 · firebase-admin · slowapi · python-jose ·
opentelemetry (deferred). **Twenty-plus removed.**

### Two deliberate substitutions

| was | now | why |
|---|---|---|
| `python-jose` | `PyJWT` | jose has been effectively unmaintained since 2021 and has carried algorithm-confusion CVEs. This is a recreation, so it recreates onto the safer library. |
| `passlib[bcrypt]` | `bcrypt` directly | the reference already imported bcrypt itself; passlib added a dependency and no capability. |

### One correction during the phase

`av>=12.0,<15` forced a source build — no cp311 wheel exists in that range on
Windows, and the install failed on a missing C++ toolchain. Widened to
`av>=13,<19`; 18.1.0 ships `av-18.1.0-cp311-abi3-win_amd64.whl`.

---

## 7. Application foundation

### Authentication

bcrypt (per-password salt, ≤72-byte rejection rather than silent truncation) ·
JWT access 15 min / refresh 7 days · **token type checked on every verify** ·
API keys stored as SHA-256 only · uniform failure for unknown-email and
wrong-password, including a dummy hash verification so the timings match.

Two decisions worth naming:

- **A refresh token carries no roles.** Roles in a 7-day token would delay a
  revocation by up to a week.
- **The access decision is rebuilt from the database on every request**, not read
  from the token. A revoked role or a disabled account takes effect immediately.

### Authorization — and the empty-scope hazard

Phase 0 called this the single most dangerous line in the migration. Vision OS
documents:

> *"`cameras` empty means every camera in the tenant… a principal with no access
> is expressed by having no grant at all."*

The natural application-side value for "no camera access yet" is an empty list.
Those two facts meeting quietly grants site-wide CCTV access to an account
intended to have none.

**`ScopeBreadth` makes them unconfusable:**

```
NONE           → no grant is built at all; to_grant() and to_scope() RAISE
LISTED         → exactly the named cameras; empty list refused at construction
ALL_IN_TENANT  → passes () deliberately, the platform's documented wildcard
```

Nine tests cover it, including: an unreadable stored breadth denies; a missing
grant denies; a `LISTED` grant with nothing listed denies; a wildcard cannot also
carry camera ids.

**Deny by default is structural, not conventional.**

### Roles

`super_admin` · `org_admin` · `restaurant_manager` · `kitchen_supervisor` ·
`hygiene_officer` · `auditor` · `developer`

Two assignments are load-bearing:

- `kitchen_supervisor` holds **no** `VIEW_EVIDENCE` — it is the role most likely
  to be a shared screen on a kitchen wall.
- `auditor` holds **no** `VIEW_LIVE` — reviewing the record and watching people
  work are different acts.

`Action.READ_EVIDENCE` is never implied by `READ_OBSERVATIONS`, and
`REGISTER_DEMAND` is not granted to read-only roles. Both asserted.

### Database

Async SQLAlchemy 2.0 · session-per-request with commit/rollback boundary ·
`pool_pre_ping` · health check that never raises · **`create_all` is
test-only and named `create_all_for_tests`** so its appearance in application
code is obviously wrong.

Four tables — `organizations`, `users`, `role_assignments`, `access_grants`.
**No restaurant, camera, incident or notification table**, per §14. One Alembic
revision, applied and verified.

### Redis

Degrades, never fatal. Connect returns a bool; `require()` raises at *request*
time naming the dependency. Error detail is the exception **type** only, because
a Redis URL carries a password and connection errors quote the URL.

### Observability

Prometheus counters prefixed `uwv_` so application metrics never collide with the
platform's own `MetricsEngine` names. Request-id middleware. loguru with
`diagnose=False` — with it enabled, loguru renders local variables into
tracebacks, which puts password hashes and tokens into logs.

`uwv_authorization_denied_total{permission}` and `uwv_evidence_access_total` are
security-relevant by design.

### Errors

One envelope: `{code, message, retryable, details, request_id}` — matching the
platform's own (09_API §8), so a consumer parses one shape rather than two.

Typed platform errors keep their stable `code`. Unhandled exceptions become a
generic `INTERNAL` with a request id; the detail goes to the log. Asserted: no
traceback, no module path, no SQL, no credential in any response body.

---

## 8. The Vision OS integration boundary

```
app/vision/composition.py    policy loading · the canonical registry · the identity guard
app/vision/runtime.py        lifecycle, assembly, status
```

Both **call** the platform's bootstraps. Neither contains detection, tracking,
cropping, understanding or compliance logic, and
`test_the_platform_never_imports_the_application` asserts the dependency runs one
way.

### The shared AttributeRegistry — §19, mandatory

```python
attributes     = build_attribute_registry(policies)   # built ONCE
registry_layer = build_registry_layer(platform, attributes=attributes)
assert_shared_attribute_registry(registry_layer, understanding, attributes)
```

The guard checks **object identity**, not equality, and fails assembly rather
than degrading. Two registries built from the same documents compare equal and
behave differently — which is exactly what happened in Phases 6.1 through 6.8,
where M7 refused 308 of 308 attributes M9 produced and `FRESH_ENOUGH` had never
once fired.

**Verified live:**

```
attributes : ('face_covering', 'hand_covering', 'head_covering', 'visible_object_kind')
M7 registry IS the canonical registry: True
```

Five tests in `tests/app/test_shared_attribute_registry.py`, including one that
proves the guard *can* fail when handed two registries — a guard that cannot fail
is a comment.

### Why the platform does not autostart

`VISION_AUTOSTART=false`. The platform needs a `SourcePort`, and Phase 1 binds
none — streaming RTSP and file replay are Phase 3.

A platform booted with no source would report itself healthy and observe nothing
forever, which is precisely the state invariant V8 exists to make unmistakable.
So the runtime starts *not started*, says so, and observation routes return
`VISION_UNAVAILABLE` rather than an empty result.

`assemble()` is fully implemented and exercised. **What is missing is the source,
not the composition.**

---

## 9. Regression coverage

### Freshness — §20

`tests/app/test_freshness_regression.py`, 28 tests:

- all **10** `TriggerReason` members present, count asserted
- all **8** `SkipReason` members present, count asserted
- `FRESH_ENOUGH` named explicitly, with the Phase 6 history in the docstring
- `AttributeStatus` staleness arithmetic — fresh, stale, and **never-observed has
  no age** (absent is not old; `ATTRIBUTE_MISSING` ≠ `ATTRIBUTE_STALE`)
- **validity windows asserted unchanged**: `head_covering` 120 000 ms,
  `hand_covering` 60 000 ms
- every declared attribute still carries `not_visible`
- `ComplianceState` and `UnknownReason` intact

**No `validity_ms` was tuned. No performance was optimised.** The platform's own
29 trigger tests migrated and pass; this file guards the *vocabulary* those tests
would rename alongside themselves.

### Semantic Ceiling

Four parametrised tests confirm the neutrality gate refuses `is_compliant`,
`ppe_violation`, `raise_alert`, `violation_state`.

### Evidence and scope

`READ_EVIDENCE` separation · tenant-required `Scope` · the platform's own refusal
of an unscoped query, re-verified after migration.

### DevTools

A developer reaches `/api/v1/devtools/vision`; a manager gets **403** with
`OUT_OF_SCOPE` and the missing permission named; an anonymous caller gets 401;
with `FEATURE_DEVTOOLS=false` the route is **404 — not mounted at all**.

**Failure injection does not exist in this repository.** Not permission-gated —
absent. A test probes three plausible paths and asserts 404.

---

## 10. Security verification — §32

| requirement | status | evidence |
|---|---|---|
| no secrets committed | ✅ | scan finds only test fixtures (`nvapi-test`, `nvapi-from-file`) |
| no `.env` in the repo | ✅ | absent; gitignored |
| no passwords in logs | ✅ | `diagnose=False`; Redis errors report type only |
| no RTSP credentials in source | ✅ | none present; `CCTV_CREDENTIAL_REF` is the preferred form |
| no NVIDIA key in source | ✅ | env only |
| no unsafe production defaults | ✅ | `assert_production_safe()` refuses to boot; 4 tests |
| deny-by-default authorization | ✅ | `ScopeBreadth.NONE`; 9 tests |
| tenant-aware scope | ✅ | `Scope` cannot be constructed without a tenant |
| evidence access separate | ✅ | `READ_EVIDENCE` never implied |
| safe CORS | ✅ | `*` refused at construction; explicit methods and headers |
| secure error responses | ✅ | no traceback, path, SQL or credential; asserted |
| security defaults OFF | ✅ | `SERVE_FRAMES`, `ALLOW_EVIDENCE`, `FEATURE_DEVTOOLS`, `FEATURE_LIVE_CCTV` |

The production-safety error names the variable and **not its value** — asserted
by `test_the_refusal_names_the_variable_and_not_its_value`.

---

## 11. Clean-environment verification — §28

```
py -3.11 -m venv .venv
pip install -e ".[inference,video,eval,test,lint]"
```

61 packages, none from any Atlas repository.

```
vision_os       1.0.0-flow1 -> ...\unityworks-vision-ai-backend\vision_os\__init__.py
compliance                  -> ...\unityworks-vision-ai-backend\compliance\__init__.py
API_VERSION     1.0.0
Actions         7
numpy/onnx/av   2.4.6  1.29.0  18.1.0
```

Then, with cwd **outside** the repository, `import vision_os` succeeds and
`sys.path` is unchanged by the import.

Alembic verified end to end against a scratch SQLite database:

```
Running upgrade -> d1f84b293431, identity foundation
tables: ['access_grants', 'alembic_version', 'organizations', 'role_assignments', 'users']
```

---

## 12. Reference repository protection — §1

| repository | HEAD | dirty |
|---|---|---|
| `atlas/frontend` | `a6bafbc` | 0 |
| `atlas/vision_os_demo` | `57b9c92` | 0 |
| `atlas/vision_os_validation_console` | `a284db3` | 0 |
| `atlas/backend` | `345292c` | **2 — not from this phase** |

### The two `atlas/backend` edits

`.env.example` and `app/config.py`, modified at **16:11** and **16:06** today —
during this session, from the IDE. **Neither was written by this phase**: the
migration reads `atlas/backend` and never writes to it, `app/config.py` was
opened read-only for reference, and `.env.example` was never opened at all.

**One of them needs immediate attention.**

```diff
  NVIDIA_MODEL=nvidia/llama-3.1-nemotron-nano-vl-8b-v1
- 
+ VISION_NVIDIA_API_KEY=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
```

`.env.example` is **git-tracked** (`git ls-files --error-unmatch` confirms). A
live-looking NVIDIA API key is therefore staged to be committed the next time
that repository is committed, and a secret in git history is not removed by
deleting the line later.

Recommended, in order: **rotate the key at NVIDIA**, revert the line in
`.env.example`, and put the real value in `.env` (already gitignored) or in this
backend's `VISION_NVIDIA_API_KEY`.

Not corrected here: `atlas/backend` is read-only for this phase, the edits are
the user's, and rotating a credential is their decision.

The second edit — `nvidia_chat_model`, temperature, `top_p`, `max_tokens` in
`app/config.py` — is a model-tuning change to the reference backend and affects
nothing in this repository.

---

## 13. Phase 1 gate results — §35

| # | gate | result |
|---|---|---|
| 1 | new backend repository exists | ✅ |
| 2 | Vision OS under the new canonical package | ✅ `vision_os/`, 201 files, byte-identical |
| 3 | no duplicate Vision OS package | ✅ asserted |
| 4 | no sibling-repository runtime dependency | ✅ asserted, incl. out-of-tree subprocess |
| 5 | no `sys.path` injection | ✅ AST-asserted across all source |
| 6 | clean-environment installation works | ✅ fresh venv, 61 packages |
| 7 | Vision OS imports correctly | ✅ |
| 8 | **all 94 platform test files pass** | ✅ **2,809 tests, 0 failures** |
| 9 | compliance tests pass | ✅ 71 tests, 4 files |
| 10 | shared AttributeRegistry regression | ✅ 5 tests, identity-checked |
| 11 | FRESH_ENOUGH regression | ✅ 28 tests |
| 12 | UNKNOWN semantics preserved | ✅ every attribute keeps `not_visible` |
| 13 | evidence security boundary | ✅ `READ_EVIDENCE` never implied |
| 14 | authentication foundation works | ✅ 21 tests |
| 15 | authorization foundation works | ✅ 24 tests |
| 16 | tenant-aware scope enforced | ✅ 3 tests + platform's own |
| 17 | production configuration exists | ✅ + startup refusal |
| 18 | `.env.example` exists | ✅ no real values |
| 19 | no secrets committed | ✅ test fixtures only |
| 20 | reference repositories untouched | ✅ by this phase — see §12 |
| 21 | backend documentation exists | ✅ README + 3 doc sets |
| 22 | reproducible from a clean checkout | ✅ |

---

## 14. Known limitations

Stated plainly, because each is a deliberate Phase 1 boundary rather than an
oversight.

1. **No frame acquisition.** No `SourcePort` or `DecoderPort` is bound. Phase 3.
2. **Vision OS does not autostart.** See §8.
3. **Understanding is not bound.** No VLM adapter is wired, so the M9→M7
   write-back path is proven structurally (shared registry, declared attributes)
   rather than end to end. The platform's own suite covers it; the application
   path returns in Phase 3.
4. **Evidence is in-memory** and does not survive a restart. A durable, encrypted
   adapter with a retention sweeper is required before any real deployment.
   Phase 5.
5. **No durable audit sink.** `AuditSinkPort` exists with no implementation bound.
6. **`SecretProviderPort` is unimplemented.** Secrets come from environment
   variables. `CCTV_CREDENTIAL_REF` is documented as the preferred form.
7. **No product endpoints.** No restaurant, camera, incident, notification or
   report route — correct for this phase.
8. **No rate limiting at the HTTP edge.** Vision OS has `ApiLimits`; the
   transport enforces none yet.
9. **No user-administration endpoints.** Users must be seeded directly. Creating
   a bootstrap admin path without the authorization surface to manage it would be
   a privilege-escalation route with no supervision.
10. **`vision_os/__init__.py` still claims "Flow 1".** Left verbatim: correcting
    it would edit the package this phase exists to move unchanged. First item for
    Phase 2.
11. **`tests/vision_os/conftest.py` mentions "the Atlas root conftest"**, which no
    longer exists. Harmless and left unedited for the same reason.
12. **Lint is not enforced on migrated code.** `vision_os/`, `compliance/`,
    `tools/` and their tests are ruff-excluded. Findings there are pre-existing,
    and fixing them would change code this phase must not change.

---

## 15. Phase 2 prerequisites

Phase 2 is the frontend: the production shell, authentication, and the DevTools
port.

### Backend contracts that already exist

| need | endpoint |
|---|---|
| login | `POST /api/v1/auth/login` |
| refresh | `POST /api/v1/auth/refresh` |
| identity and permissions | `GET /api/v1/auth/me` |
| operator status | `GET /api/v1/status` |
| liveness / readiness | `GET /health`, `GET /health/ready` |
| DevTools probe | `GET /api/v1/devtools/vision` |
| error envelope | uniform on every route |

`/api/v1/auth/me` returns roles, permissions and camera scope, which is what
role-aware navigation needs.

### What Phase 2 must add on this side

1. **Move the refresh token to an httpOnly, Secure, SameSite=Strict cookie.** It
   is returned in the response body today because there is no frontend yet. A
   refresh token readable by page JavaScript is one XSS from being stolen.
2. **Publish OpenAPI for type generation**, so `contract/types.ts` stops being
   476 hand-maintained lines.
3. **Port the DevTools read routes** from the validation harness — observations,
   state, crops, metrics, architecture — each behind `ACCESS_DEVTOOLS`, with a
   real `Principal` and a real tenant.
4. **WebSocket authentication**, for the live stream.
5. **A user-administration surface**, so accounts stop being seeded by hand.

### The decision Phase 2 cannot start without

**Frontend framework.** Vite + React Router (preserving the console's 6,941 LOC
and 11 tests) versus Next.js (inheriting `atlas/frontend`'s auth and BFF
patterns). Phase 0 recommends **Vite + React Router, porting the auth patterns by
hand**. It remains unratified, and it is the largest irreversible frontend choice
in the programme.

### Not Phase 2

Live RTSP (Phase 3, still blocked on TCP 554 at the site) · durable evidence and
audit (Phase 5) · product features (Phase 4).

---

## 16. What was deliberately not done

No Vision OS contract, port, schema, prompt, detector, threshold, compliance rule
or configuration document was changed. No test was weakened, deleted or rewritten
to make the migration easier — the only test-side change is the addition of
`tests/__init__.py`, forced by the package rename and explained in the file
itself.

No performance was optimised. No freshness window was tuned. No product feature
was implemented. No frontend work was started.
