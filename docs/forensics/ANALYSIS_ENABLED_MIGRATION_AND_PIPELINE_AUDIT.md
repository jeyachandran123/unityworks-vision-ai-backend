# Stage 0 — `analysis_enabled` Migration & Pipeline Audit

**Scope:** restore schema consistency around `cameras.analysis_enabled` and determine
whether a database/configuration blocker stands in front of the perception work.
No perception architecture was changed.

| | |
|---|---|
| Repository | `unityworks-vision-ai-backend` |
| Branch | `feat/unityworks-vision-os-prod-hardining` |
| HEAD | `9bfbe8f` |
| Date | 2026-09-03 |
| Companion | [`VISION_OS_PERCEPTION_FORENSIC_AUDIT.md`](VISION_OS_PERCEPTION_FORENSIC_AUDIT.md) |

---

## 1. Executive conclusion

### **YES, AFTER FIXES** — for the migration and schema question.
### **NO** — a separate, unrelated blocker stops the alert pipeline today.

Two independent findings, and they must not be conflated:

**Finding 1 — `analysis_enabled` was real drift, now fixed.** Restoring the migration file
repaired Alembic completely, but it did **not** repair the application. The column existed in
Postgres and in the migration, and existed nowhere in `app/`, `vision_os/`, `tests/` or
`compliance/`. `alembic check` proved it as `remove_column` drift: autogenerate wanted to
*drop* the column to make the database match the models. The consequence was not a broken
pipeline — it was a silently ignored operator decision: every `enabled` camera was analysed
regardless of what its row said, and the wall/analysis split the column exists to express was
unavailable. **Fixed in this stage, in four files.**

**Finding 2 — the alert pipeline is stopped, and `analysis_enabled` is not why.** Alerts have
been dead for **~2 h 50 min** (last incident `05:06:41Z`, audit run at `07:58:30Z`). The
cause is the observation-log conformance gate, reproduced deterministically below. It is
**not** a database or configuration problem, it is **not** `analysis_enabled`, and per the
Stage 6 rules I have **escalated it rather than fixed it** — the fix lives in previously
reverted work I was instructed not to restore.

So: the schema foundation is now sound, but the pipeline cannot produce an alert until
Finding 2 is decided on. Both statements are in the go/no-go at §11.

---

## 2. Git and migration state before changes

```
branch : feat/unityworks-vision-os-prod-hardining
HEAD   : 9bfbe8f refactor(authorization): remove VIEW_MODEL_EVALUATION permission…

git status --short
  ?? docs/forensics/
  ?? migrations/versions/20260903_d4a1c8e37b52_camera_analysis_enabled.py
```

The migration file is present and **untracked** — restored into the working tree, not
committed. Nothing else was modified.

### Alembic state — proven, not assumed

```
$ alembic heads
d4a1c8e37b52 (head)                      ← exactly one head

$ alembic current
d4a1c8e37b52 (head)                      ← database matches head

$ alembic history
c9d5f21ab340 -> d4a1c8e37b52 (head), Separate "this camera streams" from "this camera is analysed".
b7c41e08d5aa -> c9d5f21ab340, product module scaffold
a3c7e1b40d92 -> b7c41e08d5aa, camera zone assignment history
91946b102297 -> a3c7e1b40d92, evidence subject geometry
d1f84b293431 -> 91946b102297, durable domain: cameras, evidence, incidents, frames, audit
<base>       -> d1f84b293431, identity foundation

$ alembic upgrade head
(clean, no-op — already at head)
```

The revision chain declared in the files is linear and complete:

```
d1f84b293431 → 91946b102297 → a3c7e1b40d92 → b7c41e08d5aa → c9d5f21ab340 → d4a1c8e37b52
```

**The historical failure is repaired.** The database recorded `d4a1c8e37b52` while the
migration file had been deleted, so Alembic could not locate the revision and
`alembic upgrade head` failed. Restoring the file closes that gap: `current` resolves,
`history` is continuous, there is a single head, and there is no recorded revision missing
from the repository.

### Schema drift — the finding restoring the file did *not* fix

`alembic check` reported drift. Classified by operation:

| Operation kind | Count | Assessment |
|---|---|---|
| `modify_default` | 276 | **Pre-existing noise.** Python-side vs server-side default representation on `board_usage_events`, `pos_connectors`, `pos_sync_runs`. Unrelated to this work and untouched. |
| `remove_column` on `cameras.analysis_enabled` | 1 | **Real drift.** ← the finding |
| `add_column` / `add_table` / `modify_nullable` / `modify_type` | **0** | No other structural drift |

```
('remove_column', None, 'cameras',
   Column('analysis_enabled', BOOLEAN(), table=<cameras>,
          nullable=False, server_default=DefaultClause(...)))
```

`remove_column` is autogenerate saying: *the database has this column, the models do not, so
to reconcile them I would drop it.* A subsequent `alembic revision --autogenerate` would have
generated a migration **deleting the column** — losing the operator's per-camera analysis
decisions permanently.

Confirmed independently:

```
$ grep -rn "analysis_enabled" app/ vision_os/ tests/ compliance/ --include=*.py
(no matches)

ORM Camera columns:
['id','organization_id','restaurant_id','zone_id','camera_key','name','purpose',
 'host','rtsp_port','channel','stream_type','username','credential_ref',
 'analysis_fps','enabled','created_at','updated_at']        ← 17 columns, no analysis_enabled
```

---

## 3. `analysis_enabled` schema truth

### Migration definition

```python
# migrations/versions/20260903_d4a1c8e37b52_camera_analysis_enabled.py
op.add_column(
    "cameras",
    sa.Column("analysis_enabled", sa.Boolean(),
              nullable=False, server_default=sa.true()),
)
```

### Live database

```
column_name       data_type   is_nullable   column_default
analysis_enabled  boolean     NO            true
enabled           boolean     NO            (none)

rows where analysis_enabled IS NULL : 0
```

### Layer-by-layer truth table

| Layer | Before this stage | After |
|---|---|---|
| Migration file | ✅ present, chain valid | ✅ unchanged |
| Postgres column | ✅ `boolean NOT NULL DEFAULT true` | ✅ unchanged |
| `alembic current` | ✅ `d4a1c8e37b52` | ✅ unchanged |
| **ORM model** | ❌ **absent** → drift | ✅ mapped, `default=True`, `server_default=true` |
| **API `to_wire`** | ❌ **absent** — clients could not see it | ✅ exposed |
| **`CameraService.update` allow-list** | ❌ **absent** — silently dropped on PATCH | ✅ accepted |
| **Value validation** | ❌ none | ✅ non-boolean rejected |
| **Runtime scheduling** | ❌ **not read** — every enabled camera analysed | ✅ governs perception sessions |

**Nullability:** `NOT NULL` in both migration and model; **zero** NULL rows. There is no third
state and no ambiguous runtime interpretation to resolve.

**Default agreement:** migration `server_default=sa.true()`, model `default=True` **and**
`server_default=sa_true()`. They agree, which is what makes a fresh
`create_all_for_tests()` database behave identically to a migrated production one.

---

## 4. Existing camera behaviour — Stage 5 answered explicitly

> **Were existing cameras automatically enabled or disabled by the migration?**
> **Enabled.** No camera was silently disabled by this migration.

Proven three ways, not inferred:

1. The migration specifies `server_default=sa.true()`. Postgres applies that value to every
   existing row when adding a `NOT NULL` column — `true` is the only value they can have
   received.
2. The live schema still reports `column_default = true`.
3. `select count(*) from cameras where analysis_enabled is null` → **0**. No row escaped the
   backfill, so no row can behave differently from a new one.

**Could production cameras have silently become `analysis_enabled = false` after migration?**
No. `false` cannot arise from this migration. Twelve rows *are* currently `false`, and those
values were written deliberately afterwards, not by the migration.

### Current estate

| Camera | `enabled` | `analysis_enabled` | `analysis_fps` |
|---|---|---|---|
| cam-01 … cam-10 | `false` | `false` | 4.0 |
| **cam-11** | **`true`** | **`true`** | 2.0 |
| **cam-12** | **`true`** | **`true`** | 2.0 |
| **cam-13** | **`true`** | **`true`** | 2.0 |
| **cam-14** | **`true`** | **`true`** | 2.0 |
| cam-15, cam-16 | `false` | `false` | 4.0 |

Note this changed during the audit window — at the time of the perception audit all sixteen
rows were `enabled = true`. Today only the four kitchens are.

**A consequence worth stating plainly:** because `analysis_enabled` was read by nothing, the
current runtime behaviour happens to be correct *by coincidence* — the four analysed cameras
are exactly the four enabled ones. The moment anyone enables a corridor for the wall, it
would also have been analysed, spending detection and the global model-call budget on it.
That is the failure this column exists to prevent, and it was one click away.

---

## 5. End-to-end pipeline trace — where `analysis_enabled` enters and exits

```
Camera row (Postgres)                         analysis_enabled = true|false
      │
      ▼
Camera ORM model               app/domain/models.py        ← ENTERS (was absent)
      │
      ├─► API GET/PATCH        app/domain/cameras.py:to_wire / update allow-list
      │                                                    ← ENTERS (was absent)
      ▼
CameraService.enabled_for_runtime()          filters on `enabled` only
      │
      ├─────────────────────────────┬──────────────────────────────────┐
      ▼                             ▼                                  │
_start_camera_wall              _start_cameras_from_database           │
app/main.py:450                 app/main.py:332                        │
  every enabled row               ← EXITS HERE: rows filtered on       │
  (viewing only)                    `row.host and row.analysis_enabled`│
      │                             │                                  │
      ▼                             ▼                                  │
  MJPEG wall                  live.start_from_records(configs)         │
  (no perception)                   │                                  │
                                    ▼                                  │
                          VisionSession._consume → ANALYSIS.run(...)   │
                                    ▼                                  │
                     detection → tracking → registry → cropping        │
                                    ▼                                  │
                              understanding (VLM)                      │
                                    ▼                                  │
                    ✗✗✗ synthesis — STOPS HERE TODAY ✗✗✗  ◄────────────┘
                                    ▼                        (§7)
                          exposure → ComplianceDriver
                                    ▼
                          IncidentService.open → alert
```

**`analysis_enabled` enters at the ORM and API, and exits at exactly one runtime decision** —
`app/main.py:_start_cameras_from_database`. That is the only place a perception session is
opened; it is called from the application lifespan and nowhere else.

### The runtime contract, stated honestly

| Question | Answer |
|---|---|
| Where is the decision made? | `app/main.py:356` — building `configs` for `live.start_from_records` |
| Before or after frame acquisition? | **Before.** A non-analysed camera never opens an analysis session, so no frame is acquired for perception. It still streams to the wall. |
| Does it affect all cameras? | Yes, uniformly. |
| Can a disabled camera still produce stale observations? | No new ones. Historical rows persist, correctly — they were true when written. |
| Does enabling take effect without a restart? | **No.** The row is read at start. This is the pre-existing contract and is unchanged by this stage. |
| Does the runtime cache camera configuration? | Effectively yes — the roster is read once at lifespan start. |
| Is a database update observed by the live loop? | **No.** It is observed at the next start. |

The restart requirement is a real limitation and it is deliberately left as-is: making it
live belongs in a supervisor that reconciles the running estate against camera rows on a
timer, off the request path. Test
`test_turning_analysis_off_removes_a_camera_from_the_next_start` pins the contract as it
actually is, rather than asserting a behaviour the system does not offer.

---

## 6. Real runtime evidence

Backend running on `:8010` throughout the audit.

| Probe | Result |
|---|---|
| `/health` | `{"status":"ok"}` |
| `/health/ready` | `{"ready":true,"database":true,"cache":true,"vision_os":true}` |
| Incidents in database | 1,990 |
| Latest incident | `2026-09-03 05:06:41Z` |
| Audit time | `2026-09-03 07:58:30Z` |
| **Alert silence** | **≈ 2 h 52 min** |
| Camera observation logs | `cam-11…14.jsonl`, frozen at `10:51` local |
| Conformance-kit artefacts in the production log | **7 files**, `10:52` local |

| Camera | `analysis_enabled` | Frames arriving | Analysis observed | Observations | Incidents |
|---|---|---|---|---|---|
| cam-11 | true | yes (wall) | started | **stopped** `10:51` | 392, none recent |
| cam-12 | true | yes (wall) | started | **stopped** `10:51` | 767, none recent |
| cam-13 | true | yes (wall) | started | **stopped** `10:51` | 703, none recent |
| cam-14 | true | yes (wall) | started | **stopped** `10:51` | 128, none recent |
| cam-01…10, 15, 16 | false | not enabled | n/a | n/a | 0 |

> `/health/ready` reports `vision_os: true` while no observation can be published. The
> readiness probe does not check that the synthesis/exposure layer bound. This is the
> "healthy dashboard over a dark pipeline" pattern this codebase warns about in several
> places, and it is why the outage was invisible. **Flagged, not fixed** — out of scope here.

---

## 7. Pipeline stop diagnosis

Against the required taxonomy, the answer is **not** "alerts are not working".

> **The pipeline stops at observation publication — the synthesis layer fails to bind at
> boot. This sits between (D) and (E): the VLM path runs, and no observation is ever
> published or persisted, so no finding, no incident and no alert can exist.**

### Root cause, reproduced deterministically

`vision_os/synthesis_bootstrap.py:_gate` runs each adapter's conformance kit **against the
live adapter**:

```python
report = effective.run(adapter, fast_only=True)      # writes real records
if not report.passed:
    raise ObservationError(...)
_purge_kit_traces(adapter)                           # calls adapter.reset() if it exists
```

`FileObservationLog` **has no `reset()`** — the `reset()` at `stores.py:342` belongs to a
different class (it clears `self._observations`, which is not in
`FileObservationLog.__slots__`). So the kit's records are written into the production
durable log and never removed. The module's own docstring concedes this:

> *"a durable adapter that keeps no per-partition deletion path retains a handful of fixture
> records under partitions no camera will ever use."*

Those retained records break the kit on the **next** boot. Reproduced on a scratch directory —
production untouched:

```
has reset(): False

boot 1: passed=True
  files now: [kit-log-idempotent, kit-log-monotonic, kit-log-order,
              kit-log-part-a, kit-log-part-b, kit-log-tail, kit-obs-cam].jsonl

boot 2: passed=False
  FAILURE: shape/interface:
  FAILURE: [L2] semantics/idempotent_by_id:
  FAILURE: [L3] semantics/positions_are_monotonic: positions repeated
  FAILURE: [L7] semantics/tail_follows_without_blocking: tail on an empty partition
           must return immediately with nothing, not block waiting for a first record
```

**Production `data/observations/` contains exactly those seven files.** The deployment is in
the boot-2 state: every restart from now on refuses the observation log.

The failure is caught and the process continues degraded — `app/vision/runtime.py:377`:

```python
except Exception as exc:
    logger.error(
        "synthesis not bound: {}: {}. Attributes will reach M7 and no "
        "observation will be published.", ...)
```

`synthesis = None`, so the exposure layer at line 425 is never built either. Detection,
tracking, registry, cropping and understanding all keep running and cost money; nothing they
produce can leave M7. The compliance driver then reads zero subjects every 5 seconds and
opens nothing.

### Why this is escalated, not fixed

Stage 6 permits fixes "directly related to restoring consistency around
`cameras.analysis_enabled`". This is not that. The repair — running the kit against a
disposable twin so it never touches the durable log — exists in the reverted work I was
explicitly instructed not to restore. **Escalating for your decision (§11).**

Two remediation options, neither taken:

| Option | Effect | Durability |
|---|---|---|
| **A.** Delete the 7 `kit-*.jsonl` files from `data/observations/` | Next boot passes (returns to boot-1 state) | **Temporary** — recurs on the boot after next |
| **B.** Gate the kit against a disposable twin | Kit never writes to the durable log | Permanent |

Option A is an operator action on data, not code, and is reversible. It is the fastest way to
get alerts back today. I have not run it because it modifies production data outside this
stage's mandate.

---

## 8. Fixes made

Four files. All confined to `analysis_enabled` consistency.

### `app/domain/models.py`
Added the mapped column, resolving the `remove_column` drift. `default=True` **and**
`server_default=sa_true()` so the model agrees with the migration and a test database matches
a migrated one. Added the `sqlalchemy.true` import.

### `app/domain/cameras.py`
* `to_wire` now returns `analysis_enabled` — a client showing only `enabled` cannot explain
  why a visibly live camera raises nothing.
* Added `analysis_enabled` to the `update` allow-list, so PATCH stops silently discarding it.
  Authorization is unchanged: the existing `MANAGE_CAMERAS` permission still gates the route.
* Rejects a non-boolean value. `"false"` is truthy in Python, so coercion could switch
  analysis **on** while the operator believed they had switched it off. Scoped to this flag
  only — `enabled` has the same untyped exposure and is deliberately left alone rather than
  change an existing API's behaviour in a stage scoped to this column.

### `app/main.py`
`_start_cameras_from_database` filters analysis sessions on
`row.host and row.analysis_enabled`, and logs the cameras that stream without being analysed
so a deliberate state does not read as a fault. This is the single point where the value
governs the runtime.

### `tests/app/test_analysis_enabled.py` *(new)*
13 tests — see §9.

**Not changed:** the migration file, any perception module, the detector, the VLM, tracking,
the registry, demand semantics, incident deduplication, compliance rules, policy documents,
or any frontend code.

---

## 9. Tests

| | Passed | Failed | Total |
|---|---|---|---|
| Before this stage | 3,915 | 1 | 3,916 |
| **After this stage** | **3,928** | **1** | **3,929** |
| Delta | **+13** | 0 | +13 |

The single failure is **pre-existing and unrelated**:
`tests/vision_os/understanding/test_ninety_b_configuration.py::test_no_production_module_names_the_model`
— `vision_os/adapters/understanding/nvidia_vl.py:73` hard-codes the 90B model name. It was
failing before this work and is recorded in the companion perception audit.

### What the 13 tests cover

**Schema (4)** — the column is mapped and not merely migrated (the exact defect found here);
`NOT NULL` so there is no third state; migration and model defaults agree; a new camera is
analysable but not yet streaming.

**API (5)** — the wire exposes it; an operator can change it without taking the camera off
the wall; `developer` (DevTools access, not operator authority) gets **403**; a non-boolean is
refused rather than coerced; omitting the field leaves it alone, so an unrelated PATCH cannot
silently stop analysis.

**Scheduling (4)** — driven through the real boundary, `app_main._start_cameras_from_database`,
with a recording `LiveRuntime` stand-in. Not a boolean getter: this is the only function that
opens a perception session.
* only analysed cameras get a session (`cam-61, cam-62` from four enabled rows);
* a non-analysed camera is suppressed from perception **and still returned by
  `enabled_for_runtime`**, so the wall will start it — the wall/analysis split proven in both
  directions;
* turning analysis off removes a camera from the next start — pinning the contract as it
  really is, not one the runtime does not offer;
* a site with nothing analysed reports `0`, not `None` — "narrowed to nothing" and "could not
  read the roster" stay distinguishable, so the bootstrap supervisor does not retry forever.

Regression suites re-run clean: `test_persistence`, `test_camera_bootstrap_recovery`,
`test_migration`, `test_camera_wall`, `test_administration` — 127 passed.

---

## 10. What remains intentionally unfixed

### The alert blocker (§7) — escalated, awaiting your decision
Conformance kit writes fixtures into the production `FileObservationLog`; the second and every
subsequent boot fails the kit; synthesis never binds; no observation is published. **This is
why there are no alerts right now.** Not fixed — the repair is in reverted work I was told not
to restore.

### The five perception defects from the companion audit — untouched
1. Policy `lifecycle` / `min_confidence` gates declared and never read.
2. `tracker.iou` — the documented fallback — hard-coded as the production tracker.
3. Track fragmentation from motion-free association arithmetic.
4. Transition IDs destroyed in `TrackingOutcome`, then wrongly reconstructed.
5. Registry re-entry binding structurally unreachable (loop ordering).

Plus: `face_covering` evaluated but `informational`, so it never raises an incident;
`hand_covering` working and honestly answering `not_visible`.

### Also observed, out of scope
* `/health/ready` reports `vision_os: true` while no observation can be published.
* `alembic check` reports 276 pre-existing `modify_default` differences on unrelated tables.

> **Per Stage 8:** the database and configuration blocker around `analysis_enabled` is
> resolved, but the perception architecture defects identified in the forensic audit remain
> and require the next stage. Nothing here should be read as Vision OS being fixed. No
> threshold was raised, no incident suppressed, no model changed.

---

## 11. Go / no-go recommendation

## **READY FOR PERCEPTION ARCHITECTURE REPAIR**

The `analysis_enabled` foundation is sound and proven:

* Alembic has one head, `current` matches it, the chain is continuous, `upgrade head` runs.
* The missing-revision failure is fully repaired.
* Zero structural schema drift; `analysis_enabled` no longer appears in `alembic check`.
* Migration, database, ORM, API and runtime now agree on type, nullability and default.
* No existing camera was silently disabled — the backfill was `true` and no row is NULL.
* The flag governs perception at one auditable point, covered by tests at the real boundary.
* 3,928 passing, one pre-existing unrelated failure.

**With one blocking caveat you must decide before any perception work can be validated:**

The pipeline cannot currently produce an alert, for a reason unrelated to this stage
(§7). Perception fixes can be *written* and *unit-tested* against this foundation, but they
**cannot be observed working end to end** until the observation-log gate is resolved.

**Recommended order:**
1. Decide on §7 — option A (delete the 7 `kit-*.jsonl` files) restores alerts today;
   option B fixes it permanently. Say which, and I will do only that.
2. Confirm alerts flow again.
3. Begin the perception repair, starting with R5 (forensic instrumentation) and R1
   (enforce the subject filter) from the companion audit.
