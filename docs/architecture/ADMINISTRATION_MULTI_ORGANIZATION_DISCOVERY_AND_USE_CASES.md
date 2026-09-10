# Multi-Organization Discovery and Use Cases

**Status:** Read-only architecture audit. No code, migrations, permissions, or UI were changed to produce this document. Every claim below is tagged **PROVEN CURRENT BEHAVIOUR**, **INFERENCE**, **PROPOSED FUTURE DESIGN**, **DECISION REQUIRED**, or **UNKNOWN**.

---

## 1. Executive summary

The backend (`unityworks-vision-ai-backend`) is **already built as a multi-tenant system** in its data model and query discipline, even though only one organization (`org-unityworks`) exists today. `organization_id` is a first-class, FK-enforced or join-enforced column on essentially every durable table (`app/domain/models.py`), and every query on every route I traced constructs its `WHERE` clause from the authenticated caller's `access.tenant_id` — never from client input. There is currently **no API to create an Organization**; the only way one is created is the operator CLI `scripts/manage.py create-user --org <id>` (`unityworks-vision-ai-backend/scripts/manage.py:104-112`), which silently creates the `Organization` row if it doesn't exist. There is no Organization CRUD UI or API endpoint at all — `app/api/administration.py` manages Restaurants (sites), Zones, and read-only Users, but never Organizations.

"DVR" is not a modeled entity anywhere in the schema or API; it exists only as a naming convention in comments and UI hint text (`Camera.channel`, described as "The DVR has 16 channels" — `app/domain/models.py:106`, `app/domain/cameras.py:3-6`; frontend hint "The DVR channel this camera is wired to" — `unityworks-vision-ai-frontend/src/features/persistence-routes.tsx:400`). Cameras sharing a DVR are only informally grouped by identical `host`/`rtsp_port` values.

The single biggest gap for multi-org readiness is not isolation (which is strong) but **administration surface area**: there is no way, through the product, to create a second organization, assign its first admin, or manage cross-organization visibility for a platform operator (`super_admin`). Today `super_admin` holds every permission except `MANAGE_PATRON_ID` (`app/authorization/model.py:213`) but that grant is still tenant-scoped by `AccessDecision.tenant_id` — a super_admin account belongs to exactly one organization and cannot see across organizations at all, by construction (`app/authorization/model.py:436-451`). This means today's "platform admin" role is not actually a cross-org role; it is just the most powerful role *within* a tenant.

## 2. Exact current architecture

- **Backend:** FastAPI + SQLAlchemy 2.0 async ORM + (presumably) Alembic/Postgres. No `alembic/` directory was found in this checkout (**UNKNOWN** — migrations may live outside this working copy or the project may create schema via `Base.metadata` directly; not confirmed either way).
- **Domain layer** (`app/domain/*.py`): `Restaurant` (= "site"), `Zone`, `Camera`, `CameraZoneAssignment` (append-only zone-history table), `EvidenceRecord`, `Incident`, `FrameRecord`, `AuditEvent` — all in `app/domain/models.py`.
- **Identity layer** (`app/users/models.py`): `Organization`, `User`, `RoleAssignment`, `AccessGrant` — deliberately kept separate from the domain tables ("Four tables, and no more... this is deliberately not the restaurant domain" — `app/users/models.py:1-10`).
- **Authorization** (`app/authorization/model.py`): a closed `Role` enum, a closed `Permission` enum, a static `ROLE_PERMISSIONS` map, and `AccessDecision` — the single object built once per request from DB state, never from client input.
- **API** (`app/api/*.py`): `administration.py` (sites/zones/users read), `product.py` (cameras/incidents/evidence/frames/audit), `reports.py`, `evaluation.py`, `wall.py` (live streaming tickets), `websocket.py`, `integrations.py`, `patron.py`, `analytics.py`, `devtools.py`.
- **Vision OS boundary:** a separate platform (`vision_os.*` imports) that the application treats as an external system producing observations; the application never stores perception results directly (`app/domain/models.py:1-20`).
- **Frontend** (`unityworks-vision-ai-frontend`): React + TypeScript, React Query, a typed `PERMISSIONS`/`ROLES` mirror of the backend enums (`src/app/permissions/permissions.ts`), route guards (`RequirePermission`), and a design system in `src/shared/ui/product.tsx` / `src/shared/ui/primitives.tsx` (Plane, Region, SectionRule, Figure, DataTable, PageIntro, etc.) that `AdministrationPage` and `CamerasPage` already use.

## 3. Current Organization/Site/Zone/Camera relationship map

| Relationship | Enforcement | Citation |
|---|---|---|
| `Restaurant.organization_id → organizations.id` | **FK-enforced**, `ondelete=CASCADE`, `NOT NULL` | `app/domain/models.py:64-67` |
| `Zone.restaurant_id → restaurants.id` | **FK-enforced**, `ondelete=CASCADE`, `NOT NULL`. Zone has **no `organization_id` column of its own** — tenancy is only reachable by joining through `Restaurant` | `app/domain/models.py:90-93`; confirmed app-level in `app/api/administration.py:282-291` ("`zones` carries no organization column of its own, so filtering on the parent is the only construction that cannot leak another organization's areas") |
| `Camera.organization_id → organizations.id` | **FK-enforced**, `ondelete=CASCADE`, `NOT NULL` | `app/domain/models.py:124-126` |
| `Camera.restaurant_id → restaurants.id` | **FK-enforced**, `ondelete=CASCADE`, `NOT NULL` | `app/domain/models.py:127-129` |
| `Camera.zone_id → zones.id` | **FK-enforced**, `ondelete=SET NULL`, **nullable** — a camera need not have a zone | `app/domain/models.py:130-132` |
| `Camera.camera_key` uniqueness | Unique **per organization**, `UniqueConstraint("organization_id", "camera_key")` | `app/domain/models.py:118` |
| `Restaurant.slug` uniqueness | Unique **per organization** | `app/domain/models.py:60` |
| Historical camera→zone attribution | `CameraZoneAssignment` — a separate, append-only interval table, keyed by `organization_id` + `camera_key`, deliberately **not** a foreign key to `Camera` or `Zone` (uses plain string ids) so that renaming/deleting a zone or camera never rewrites history | `app/domain/models.py:181-260` |

Note: `Camera.organization_id` is **redundant with** `Camera.restaurant_id → Restaurant.organization_id` (a camera's org is derivable by joining through its restaurant) but is stored directly rather than derived — this is a deliberate denormalization that lets every camera query filter directly on `organization_id` without a join, and also lets the schema catch a bug where a camera's `restaurant_id` and `organization_id` disagree (**INFERENCE** — no explicit CHECK constraint was found enforcing that `Camera.organization_id == Restaurant.organization_id` for its own `restaurant_id`; this is asserted only by application code paths, e.g. `app/api/administration.py:322-325` for zones and by `CameraService.create` always setting `organization_id=organization_id` from the caller's tenant, never from the restaurant row — `app/domain/cameras.py:86-90`).

## 4. Current database relationship map

```
Organization (users.models)
 ├── User (organization_id FK, CASCADE)
 │    ├── RoleAssignment (user_id FK, CASCADE)
 │    └── AccessGrant (user_id FK, CASCADE)  — camera_breadth/camera_ids/site_ids
 ├── Restaurant (organization_id FK, CASCADE)
 │    └── Zone (restaurant_id FK, CASCADE)   — NO organization_id column
 ├── Camera (organization_id FK CASCADE, restaurant_id FK CASCADE, zone_id FK SET NULL, nullable)
 ├── CameraZoneAssignment (organization_id column, NOT a FK; camera_key/zone_id/restaurant_id are plain strings)
 ├── EvidenceRecord (organization_id FK CASCADE; camera_key/frame_ref/object_id/observation_id are plain strings referencing Vision OS, not FKs)
 ├── Incident (organization_id FK CASCADE; restaurant_id FK SET NULL, nullable; zone_id plain string, nullable; camera_key plain string)
 ├── FrameRecord (organization_id FK CASCADE; camera_key plain string)
 └── AuditEvent (organization_id column, NOT declared as a FK — app/domain/models.py:534)
```

`AuditEvent.organization_id` (`app/domain/models.py:534`) has no `ForeignKey(...)` — every other table's org column is FK-enforced, audit's is not. **INFERENCE**: this is likely deliberate, so an audit row for a deleted organization can still be retained/exported, but it does mean the DB itself will not stop an audit row being written against a non-existent `organization_id`; that is left to `AuditTrail.record` always deriving it from `access.tenant_id`.

## 5. Current administration capabilities

| Entity | Create | Read | Update | Delete/Retire | Where |
|---|---|---|---|---|---|
| Organization | **No API route.** Only `scripts/manage.py create-user --org <id>` (CLI-only, creates the org implicitly if missing) | No API route | No API route | No API route | `scripts/manage.py:96-121` |
| Restaurant (site) | `POST /api/v1/restaurants`, gated `MANAGE_ORGANIZATION` | `GET /api/v1/restaurants`, gated `VIEW_USERS` | `PATCH /api/v1/restaurants/{id}` (name, timezone, is_active — **not** slug), gated `MANAGE_ORGANIZATION` | No delete route | `app/api/administration.py:126-260` |
| Zone | `POST /api/v1/zones`, gated `MANAGE_ORGANIZATION` | `GET /api/v1/zones?restaurant_id=`, gated `VIEW_USERS` | `PATCH /api/v1/zones/{id}` (name only), gated `MANAGE_ORGANIZATION` | No delete route | `app/api/administration.py:276-376` |
| Camera | `POST /api/v1/cameras`, gated `MANAGE_CAMERAS` | `GET /api/v1/cameras`, gated `VIEW_CAMERAS` | `PATCH /api/v1/cameras/{key}`, gated `MANAGE_CAMERAS` (name/purpose/host/port/channel/stream_type/username/credential_ref/analysis_fps/zone_id/enabled/analysis_enabled) | `DELETE /api/v1/cameras/{key}` — full retire (see §7/§10.G), gated `MANAGE_CAMERAS` | `app/api/product.py:62-230`, `app/domain/cameras.py:41-332` |
| User | Read only, no create/update/delete route | `GET /api/v1/users`, gated `VIEW_USERS`, explicitly `write_available: false` with a stated reason (no invitation/reset delivery channel yet) | — | — | `app/api/administration.py:382-441` |

Frontend mirrors this exactly: `AdministrationPage` (`unityworks-vision-ai-frontend/src/features/administration.tsx`) has working "Add a site" and "Add a zone" forms gated behind `PermissionGate permission={PERMISSIONS.manageOrganization}`, and an explicit non-editable users table with the server's own "cannot be created here yet" message rendered verbatim (`administration.tsx:438-447`). There is a separate `CamerasPage` (`unityworks-vision-ai-frontend/src/features/persistence-routes.tsx:158` onward) with its own camera create form (`channel`, `restaurant_id`, host, etc. — `persistence-routes.tsx:340-410`). **There is no Organization admin page anywhere in the frontend** — no route, no form (confirmed by reading `AppRouter.tsx` routes list, `unityworks-vision-ai-frontend/src/app/router/AppRouter.tsx:75-220`, which has `/admin` → `AdministrationPage` and `/cameras` → `CamerasPage` and nothing organization-shaped).

## 6. Current role/permission capabilities

Seven closed roles (`app/authorization/model.py:46-61`): `super_admin`, `org_admin`, `restaurant_manager`, `kitchen_supervisor`, `hygiene_officer`, `auditor`, `developer`. Permissions are a closed, flat enum with **no inheritance** (`ROLE_PERMISSIONS`, `app/authorization/model.py:200-366`).

Relevant to administration:
- `MANAGE_ORGANIZATION` — held only by `super_admin` (via `frozenset(Permission) - {MANAGE_PATRON_ID}`, `model.py:213`) and `org_admin` (`model.py:216`). Gates create/update of Restaurant and Zone.
- `MANAGE_CAMERAS` — held by `super_admin` and `org_admin` (`model.py:224`). Gates create/update/delete of Camera.
- `MANAGE_USERS` — held by `super_admin` and `org_admin` (`model.py:217`), but **there is no route that uses it** — no user-management API exists yet (`app/api/administration.py:13-26`), so this permission is currently declared but has nothing to protect (**PROVEN CURRENT BEHAVIOUR** that the permission exists; the absence of a consuming route is confirmed by grep across `app/api/*.py` finding no `MANAGE_USERS` dependency anywhere but the model file itself).
- No permission named anything like `MANAGE_PLATFORM` / `CREATE_ORGANIZATION` exists. **There is no permission concept for cross-organization administration at all** — the permission model assumes every principal belongs to exactly one tenant and every permission it grants is scoped by `AccessDecision.tenant_id`, which is mandatory and validated non-empty in `AccessDecision.__post_init__` (`model.py:444-451`).

Crucially: **`super_admin` is not a platform-wide role today.** It is the most-privileged role *within a single organization*. `AccessDecision.tenant_id` is derived once per request from the authenticated user's own `organization_id` (`app/auth/service.py:122-139`, `decision_for_claims`), and there is no code path anywhere that lets a `super_admin` (or any role) construct an `AccessDecision` for a tenant other than their own account's. This is the load-bearing fact for §9 below.

## 7. CCTV/DVR architecture findings

- **There is no DVR entity.** No table, no API resource, nothing in `app/domain/models.py` named `DVR`, `NVR`, or similar. Grep across the backend for `DVR|dvr` returns only comments and doc strings referencing "the DVR has 16 channels" (`app/domain/models.py:106`, `app/domain/cameras.py:3-6`) and a UI hint string (`persistence-routes.tsx:400`).
- **RTSP/channel/credentials live directly on `Camera`:** `host`, `rtsp_port` (default 554), `channel` (int), `stream_type` ("main"/"sub"), `username`, `credential_ref` (a pointer string like `env:CCTV_PASSWORD`, never a plaintext secret — `app/domain/models.py:143-149`).
- **Cameras can and do share a DVR informally**, purely by having identical `host`/`rtsp_port` values with different `channel` numbers. Nothing in the schema groups them — there is no `dvr_id` foreign key. A physical DVR is thus an emergent property of the data (rows with equal `host`), not a queryable entity.
- **What identifies a physical camera vs. a DVR channel**: `camera_key` is the stable pipeline identity ("`cam-01`", unique per organization — `app/domain/models.py:134-136`), and `channel` is the DVR-side channel number that `camera_key` is wired to. `camera_key` is chosen by whoever registers the camera (via the create API/UI) and is not derived from `host`+`channel` by the system.
- **`enabled` vs `analysis_enabled` are two independent booleans** (`app/domain/models.py:153,167`), explicitly split so a camera can stream to the live wall (`enabled=true`) without being analysed (`analysis_enabled=false`), or vice versa is not meaningful since analysis without a stream has no source — see the docstring at `app/domain/models.py:151-169`.

## 8. Multi-organization isolation audit

Classification key: (A) explicitly organization-scoped in the query, (B) indirectly scoped through a parent join, (C) currently global/unscoped, (D) ambiguous, (E) not applicable.

| Resource | Class | Evidence |
|---|---|---|
| Restaurant (site) | **A** | `Restaurant.organization_id == access.tenant_id` constructed directly, `app/api/administration.py:138-139`, `:252-254` |
| Zone | **B** | No own `organization_id`; scoped via `JOIN Restaurant ... WHERE Restaurant.organization_id == access.tenant_id`, `app/api/administration.py:288-291`, `:352-357` |
| Camera | **A** | `Camera.organization_id == organization_id` in every `CameraService` method, `app/domain/cameras.py:308`, `:325-328` |
| CameraZoneAssignment | **A** | `organization_id` filtered directly, `app/domain/zone_attribution.py` (record/lookup functions), and index `ix_cza_camera_time` keyed on `organization_id` first, `app/domain/models.py:225-226` |
| Live wall / streaming session tickets | **A** | Ticket payload and HMAC include `tenant_id` explicitly (`mint_ticket`/`verify_ticket`, `app/api/wall.py:62-90`); camera lookup passes `organization_id=access.tenant_id` (`app/api/wall.py:113,169,278`) |
| Observations | **A** (via Vision OS `Scope`) | `app/domain/observations.py:93-100` builds a Vision OS `Scope(tenant_id=..., camera_ids=...)` from `access.tenant_id` before any read reaches the platform |
| Incidents | **A** | Every `IncidentService` method takes and filters on `organization_id` (`app/domain/incidents.py:79,108,215,244,266`) |
| Alerts | **UNKNOWN** — no dedicated `alerts` table or service was located distinct from Incidents/Notifications; `app/domain/notifications.py` exists but was not read in full this pass. Treat as **D** pending direct confirmation. |
| Evidence | **A** | `EvidenceRecord.organization_id == organization_id` in every read/expire path (`app/domain/evidence.py:123,178,201-202`); storage path itself is partitioned by org: `self._root / organization_id / ...` (`app/domain/evidence.py:85`) |
| Reports | **A** | Every source function in `app/reporting/sources.py` filters on `request.organization_id` (lines 44, 108, 250, 281, 410, 423, 431, 471, 539, 596) |
| Frames | **A** | `FrameRecord.organization_id` FK-enforced and indexed (`app/domain/models.py:483-485,479`); `frame_ref` traceability route not individually re-checked this pass but `FrameService` in `app/api/product.py` follows the same `organization_id=access.tenant_id` pattern used throughout that file |
| Staff hygiene / compliance modules (cutting board, table occupancy, meal detection, people count, demography) | **D — not verified this pass.** These live in `app/domain/modules.py` (766 lines) and `app/vision/compliance_driver.py`, neither of which was read in full given the token budget. Grep confirms `organization_id`/`tenant_id` appear in `modules.py`, but the query-level enforcement per module was not individually traced. **Flag this as the top follow-up item before implementation.** |
| Model evaluation | **E — not applicable / deliberately global.** `app/api/evaluation.py:56-66`: evaluation artifacts are read from files on disk, not the database, and the docstring states explicitly: "the same answer for every tenant in this deployment — these are properties of the build, not of an organization's data." `tenant_id` is echoed in the response for display only. |
| Audit log | **A**, with one caveat | `AuditEvent.organization_id` is filtered directly everywhere it's read (e.g. `app/reporting/sources.py:539,596`), but the column itself is **not FK-enforced** (`app/domain/models.py:534` — no `ForeignKey(...)`, unlike every other table's org column). This is a schema-level looseness, not a query-level leak (**PROVEN CURRENT BEHAVIOUR** for both halves of that statement). |
| Users / RoleAssignment / AccessGrant | **A** | `User.organization_id == access.tenant_id` (`app/api/administration.py:398`), joined for `RoleAssignment` (`administration.py:409-411`) |

## 9. Cross-organization risk findings

1. **No query-level leak was found in any traced path.** Every read/write in `app/api/administration.py`, `app/api/product.py`, `app/reporting/sources.py`, and `app/domain/*.py` builds its `WHERE` from `access.tenant_id`, which itself is derived only from the authenticated user's own `organization_id` row (`app/auth/service.py:122-139`) — never from a request header, path param, or body field. This is the opposite of typical multi-tenant risk: the codebase already assumes multiple tenants exist and defends against a second one by construction.
2. **The concrete risk is administrative, not query-level: there is no product-level way to provision Organization #2 safely.** The only path is the CLI (`scripts/manage.py create-user --org <id>`, `scripts/manage.py:104-112`), which is un-audited (no `AuditTrail.record` call in that script — confirmed absent by reading the function), has no validation beyond the org id string, and is run by whoever has shell access to the deployment, not by any `Permission`-gated actor. **If a second organization is added today by running this script, its creation is invisible to `GET /api/v1/audit`** (`app/api/product.py:556` region) because no audit event is ever written for it. This is the single biggest gap for multi-org readiness — not isolation of data already in the system, but the absence of a governed, audited creation path.
3. **`AuditEvent.organization_id` has no FK constraint** (`app/domain/models.py:534`). If an organization is deleted (there is no delete API for it, but the CLI or a future migration could), audit rows referencing it are not cascade-deleted and not orphan-checked, unlike every other table. Low risk today (organizations are never deleted), but worth fixing before an "Add Organization" *and* "Delete/Deactivate Organization" flow both exist.
4. **`Camera.organization_id` is redundant with `Restaurant.organization_id`** (§3). No CHECK constraint was found enforcing agreement between the two. `CameraService.create` always sets `Camera.organization_id` from the caller's authenticated tenant (`app/domain/cameras.py:86-90`), and never accepts `restaurant_id` from another tenant because `restaurant_id` is not validated against `Restaurant.organization_id` inside `CameraService.create` itself — that check happens one layer up, in `app/api/administration.py:322-325` for **zones**, but I did not find an equivalent guard for **camera creation** confirming `restaurant_id` belongs to `organization_id` before insert. **This needs verification**: `app/domain/cameras.py:49-117` (`CameraService.create`) takes `restaurant_id` as a bare string and never queries `Restaurant` to confirm it belongs to `organization_id`. If the API route (`app/api/product.py:76-117`) also never checks this, **a caller could theoretically register a camera whose `restaurant_id` points at another organization's restaurant while `organization_id` is set to their own tenant**, producing a Camera row with disagreeing `organization_id`/`restaurant.organization_id`. I read `app/api/product.py:76-117` and confirmed it passes `restaurant_id=str(payload.get("restaurant_id", ""))` straight through with no ownership check before calling `service.create`. **This is a PROVEN CURRENT BEHAVIOUR gap**, not multi-org-specific (it exists today, single-tenant), but it becomes a real cross-tenant boundary violation the moment org #2 exists: an org_admin of Org A could name a `restaurant_id` belonging to Org B, and the resulting Camera row (with `organization_id=OrgA`) would then dangle against a `restaurant_id` FK that legitimately points into Org B's `restaurants` table — no FK violation occurs because the FK only checks that the row exists, not that it belongs to the same organization.

## 10. Real-world administration use cases

### A. Add a new Organization
- **Today:** no API/UI. Only `scripts/manage.py create-user --org <id> --email ... --role ...` (`scripts/manage.py:96-121`), which creates the Organization row inline if missing, with just `id`, `name` (defaults to the id string unless it's the default org), `slug` (= id). No validation of slug format, no dedupe beyond the primary key, no audit trail.
- **Should be:** a `POST /api/v1/organizations` route gated by a new permission (e.g. `MANAGE_PLATFORM` — **DECISION REQUIRED**, see §18) held only by a genuinely cross-tenant platform-operator role, since today's `super_admin` is tenant-scoped and structurally cannot see other tenants (§6). Required fields: `name`, `slug` (validated unique, immutable once set — following the existing `Restaurant.slug` precedent at `administration.py:219-220`). Optional: initial admin email to bootstrap the first `org_admin` user (since Users have no create API either — this use case is blocked on User creation, §5). Consequences for children: none at creation (no sites/zones/cameras exist yet).

### B. Add Sites to an Organization
- **Today:** fully implemented. `POST /api/v1/restaurants`, gated `MANAGE_ORGANIZATION`, `app/api/administration.py:160-193`. Required: `name`. Optional: `slug` (derived from name if absent), `timezone` (defaults UTC), `is_active` (defaults true). Validation: slug must contain ≥1 alphanumeric char (`administration.py:169-170`); uniqueness enforced by DB constraint `uq_restaurant_slug` (org+slug). Lifecycle: `is_active` boolean only — no soft-delete state machine. Consequences for children: none defined for deactivation — Zones/Cameras under an inactive Restaurant are not automatically disabled (**PROVEN CURRENT BEHAVIOUR**: `update_restaurant` at `administration.py:200-239` only sets `restaurant.is_active`, nothing cascades to `Zone` or `Camera` rows).

### C. Add Zones to a Site
- **Today:** fully implemented. `POST /api/v1/zones`, gated `MANAGE_ORGANIZATION`, `app/api/administration.py:315-341`. Required: `restaurant_id`, `name`. The route explicitly verifies the restaurant belongs to the caller's tenant before insert (`_restaurant_in_tenant`, `administration.py:325`) — this is the guard that §9 finding 4 shows is **missing** for camera creation. No `is_active` state, no delete route. Consequences: none codified for zone deactivation since there is none; a zone can only be renamed.

### D. Add CCTV/DVR infrastructure to a Site
- **Today: no such use case exists**, because there is no DVR entity (§7). "Adding DVR infrastructure" today means registering each camera channel individually via `POST /api/v1/cameras` and giving matching cameras the same `host`/`rtsp_port`. There is no bulk "register a 16-channel DVR" flow, no validation that a `host` is reachable, and no UI grouping of cameras by shared host (**PROVEN CURRENT BEHAVIOUR** — confirmed absent in both `app/api/product.py` and `persistence-routes.tsx`).

### E. Add Cameras (Org→Site→Zone→Camera, preserving DVR/channel linkage)
- **Today:** `POST /api/v1/cameras`, gated `MANAGE_CAMERAS`, `app/api/product.py:75-117` / `app/domain/cameras.py:49-117`. Required: `restaurant_id`, `camera_key`, `channel` (validated int, presumably >0 — see `_validate` at `cameras.py:334+`, not fully read). Optional: `name`, `host`, `rtsp_port` (554), `stream_type` ("sub"), `username`, `credential_ref`, `analysis_fps` (4.0), `purpose`, `zone_id`. Created **disabled** by default (`enabled=False` always, forced server-side at `product.py:100` — the payload's `enabled` field is ignored on create, a deliberate two-step: register, then separately enable). A `CameraZoneAssignment` interval row is opened at creation time even if `zone_id` is `None` (`cameras.py:104-116`) — "no zone" is itself a recorded assignment. **Gap found**: `restaurant_id` is not verified to belong to the caller's organization before insert (§9.4) — this should be closed regardless of multi-org work, and becomes urgent once org #2 exists.

### F. Move a Camera between zones/sites
- **Today:** `PATCH /api/v1/cameras/{key}` with a new `zone_id` closes the currently-open `CameraZoneAssignment` interval and opens a new one (`app/domain/cameras.py:154-171`). `camera_key`, `organization_id`, and `restaurant_id` are **not** in the `allowed` set of patchable fields on `update()` (`cameras.py:124-137`) — **so a camera cannot be moved between restaurants (sites) through this route today, only between zones within its existing restaurant.** Moving between organizations is impossible by construction (route always scopes to `access.tenant_id`). What must stay immutable per the existing design intent: `camera_key` (pipeline identity), and historical attribution — `CameraZoneAssignment` rows are never edited or deleted, only closed (`effective_to` set), which is exactly the mechanism that already answers "what zone was this camera in when incident X happened" without rewriting history (`app/domain/models.py:181-260`). Incidents and Evidence do **not** currently carry a frozen zone/site snapshot of their own — `Incident.zone_id` is a plain nullable string set once at incident creation (not re-derived), and `Incident.restaurant_id` is a nullable FK with `ondelete=SET NULL` (`app/domain/models.py:411-414`) — meaning if a Restaurant is ever deleted, existing Incidents lose their site attribution (set to NULL) even though the `CameraZoneAssignment` history table would still have the answer. **DECISION REQUIRED**: whether Restaurant deletion should ever be allowed given this SET NULL behavior, or whether Restaurants should only ever be deactivated, never deleted.

### G. Disable/retire a Camera — four distinct actions, not one
The codebase already distinguishes three of these four states; the fourth ("offline") is not a stored state at all:
1. **Temporary disable (stop streaming/recording):** `PATCH /cameras/{key} {"enabled": false}` → `CameraService.update`/`set_enabled` (`cameras.py:183-187`). Stops the RTSP session; "a row that is not enabled creates no Vision OS session at all" (`cameras.py:3-6`). Reversible.
2. **Disable-analysis-only (keep streaming, stop perception spend):** `PATCH /cameras/{key} {"analysis_enabled": false}` — independent boolean, explicitly guarded against untyped-truthy coercion bugs (`cameras.py:138-152`). Reversible. Does not stop the live-wall stream.
3. **Offline:** **not a stored state.** "Offline" would be a runtime/health signal (the camera isn't reachable), not a configuration row — this is presumably surfaced via `VIEW_CAMERA_HEALTH` / coverage data from Vision OS rather than a `Camera` column. **UNKNOWN** — not traced this pass (would require reading `app/vision/manager.py` / health/coverage reporting, out of budget).
4. **Permanent retirement (delete):** `DELETE /cameras/{key}` → `CameraService.retire` (`cameras.py:189-286`). This is destructive and irreversible: it purges the camera's entire observation-log partition (`observation_log.truncate(...)`, all records, `cameras.py:251-265`), closes (never deletes) the open `CameraZoneAssignment` interval, and only then deletes the `Camera` row. It **refuses to run at all** if a durable observation log is configured but unreachable from the current process (`cameras.py:242-249`) — "neither, rather than one" — specifically to avoid orphaning a partition that would then never be swept for retention. Two audit events are written: `camera.deleted` and (implicitly) the truncation record (`app/api/product.py:180-196` docstring). This is a fundamentally different action from #1/#2 and should not be exposed behind the same UI toggle as either.

### H. Second-organization isolation test
What Organization A must never be able to see or reach belonging to Organization B, verified against **traced** code (§8):
- Its Restaurants, Zones (via join), Cameras, Incidents, Evidence (including the evidence storage path itself, which is partitioned by `organization_id` on disk — `app/domain/evidence.py:85`), Frames, Reports, live-wall streaming tickets (HMAC-bound to `tenant_id`, `app/api/wall.py:62-90`), and observations (via Vision OS `Scope`).
- Its Users, RoleAssignments, AccessGrants (§8 table, row "Users").
- **Not directly tested this pass, flagged as follow-up**: staff-hygiene/compliance modules (`app/domain/modules.py`), notifications (`app/domain/notifications.py`), POS/patron integrations (`app/api/integrations.py`, `app/api/patron.py`) — grep confirms `organization_id`/`tenant_id` presence in all of these files but query-level enforcement was not individually verified in this pass.
- **UI test**: with no Organization switcher anywhere in the frontend and `AccessDecision.tenant_id` fixed per authenticated session (§6), an Org A user literally cannot address Org B's `restaurant_id`/`camera_key`/etc. even by guessing an id in a URL, because every backend route re-derives `organization_id` from the session and 404s (not 403s — deliberately, to avoid confirming existence, `app/api/administration.py:246-249`) on a mismatch.

## 11. Proposed domain model options — PROPOSED FUTURE DESIGN

Grounded in what exists today (§3–§4), not invented fresh:

**Option 1 — Org → Site → Zone → Camera (current shape, formalized).** Keep the existing schema exactly as-is; add Organization CRUD, and close the gap in §9.4 (verify `restaurant_id` ownership on camera create/update). DVR stays informal (shared `host`). Lowest migration risk since it changes nothing structurally — only adds routes and one validation check.

**Option 2 — Org → Site → DVR → Camera → Zone.** Introduce a first-class `DVR` table (`id`, `organization_id`, `restaurant_id`, `host`, `rtsp_port`, `label`) and give `Camera` a `dvr_id` FK, dropping `host`/`rtsp_port` off `Camera` in favor of inheriting them from its DVR (only `channel`, `stream_type`, `username`, `credential_ref` stay per-camera, since credentials can differ per channel even on one DVR). This makes "add CCTV infrastructure to a site" (use case D) a real, auditable, bulk-friendly action, and lets health/coverage be reported per-DVR ("3 of 16 channels configured"). Highest implementation cost; requires a data migration to backfill `DVR` rows from today's distinct `(organization_id, restaurant_id, host, rtsp_port)` tuples.

**Option 3 — Org → Site → Zone → Camera, with DVR as metadata-only grouping.** Add a nullable `Camera.dvr_label` (free-text) or a lightweight `DvrGroup` table with no FK relationships beyond a soft `dvr_group_id` on `Camera` — enough for the UI to group and count channels per physical unit, without a full entity model or migration risk to the `host`/`rtsp_port`/`channel` columns already on `Camera`. Middle ground: closer to Option 1's risk profile, delivers most of Option 2's UX value (§13).

## 12. Recommended future architecture — PROPOSED FUTURE DESIGN

**Recommend Option 1 for Stage 1–4, with Option 3's `dvr_group_id` added opportunistically in Stage 5** (§17), not Option 2. Justification against the evidence: the isolation and query discipline (§8) is already excellent and Option 1 preserves it untouched — the actual gap this audit found is administrative (no Organization CRUD, no camera-ownership check on create, no User provisioning), not structural. Introducing a full `DVR` entity (Option 2) before any multi-org administration exists would be solving a problem (grouping channels) that no current user has asked for, at real migration cost, while the proven gap (§9) — Organization creation with no audit trail, camera creation with no restaurant-ownership check — remains unaddressed. Option 3's lightweight grouping can be added later, additively, once real DVR-grouping UX needs are validated against actual multi-site deployments.

## 13. Administration UX information architecture — PROPOSED FUTURE DESIGN

Current state, read from `AdministrationPage` (`administration.tsx`) and `CamerasPage` (`persistence-routes.tsx`) plus the route table (`AppRouter.tsx:75-220`):
- **Dead ends found:** `AdministrationPage` shows sites and zones with counts (`zone_count`, `camera_count`), but the zone rows and site rows are plain `DataTable` rows with **no click-through** to `CamerasPage` filtered by that site/zone (confirmed: `siteColumns`/`zoneColumns` in `administration.tsx:111-150` render plain text/badges, no `<Link>`). A user who wants "show me the cameras in this zone" must navigate to `/cameras` and use whatever filtering exists there separately — no deep link carries `restaurant_id` or `zone_id` as a query param from Administration into Cameras.
- **Missing parent context:** `CamerasPage`'s "Add camera" form (`persistence-routes.tsx:340-410`) asks the operator to *type* a `restaurant_id` as a free-text `Input` (`persistence-routes.tsx:401`), not select it from a dropdown of that org's actual restaurants — despite `AdministrationPage` already fetching and rendering exactly that list. This is the same class of gap Zone-creation *doesn't* have (Zone creation does use a `<select>` populated from `sites`, `administration.tsx:353-372`).
- **No Organization surface at all** — consistent with §5/§10.A: there's nothing to link to yet.

**Proposed IA** (consistent with the existing "Vision OS" design system components already in `src/shared/ui/product.tsx` — `PageIntro`, `SectionRule`, `Region`, `Plane`, `Figure` — not a generic admin table):
- Promote `/admin` from a flat three-section page into a hierarchical drill-down using the same `SectionRule`/`Region`/`Plane` composition already established: an org-level `PageIntro`, a `Figure`-row summary (sites/zones/cameras/accounts, as today), then a **site detail view** (`/admin/sites/:id`) reusing `Plane`+`DataTable` to show that site's zones and cameras inline, each camera row linking to `/cameras/:cameraKey` (which already exists and already renders `Restaurant`/`Zone` as plain text at `persistence-routes.tsx:1623-1624` — those should become links back into `/admin/sites/:id`).
- Camera creation form's `restaurant_id` and `zone_id` fields become `<select>`s sourced from the same `organizationApi.restaurants`/`zones` queries `AdministrationPage` already uses, matching the pattern Zone-creation already sets (`administration.tsx:353-372`).
- If Organization CRUD is added (§10.A), it becomes a **new top-level page above** `/admin` (e.g. `/organizations`, visible only to a genuinely cross-tenant role — see §18 decision on whether such a role should exist at all), using the same `PageIntro`/`SectionRule` shell, with each organization row linking into that organization's own `/admin` — this only makes sense once a cross-tenant principal concept exists (§9, §18).

## 14. Backwards compatibility requirements

- `org-unityworks` must remain reachable by its exact existing `organization_id` string — nothing may require re-keying it, since `camera_key`, `restaurant.slug`, evidence storage paths (`app/domain/evidence.py:85`), and JWT-embedded `tenant_id` (`app/api/wall.py:62-90`) all key off `organization_id` directly.
- The `write_available: false` contract on `GET /api/v1/users` (`administration.py:435-440`) must not silently start returning `true` without an actual invitation/reset delivery mechanism being built — the frontend renders that exact server sentence (`administration.tsx:438-447`); changing the meaning of the field without shipping the mechanism would produce a UI lie.
- `Camera.enabled` default-false-on-create behavior (`cameras.py:68-73`, "creating a camera and switching it on are two acts") must be preserved — any bulk-import path added for multi-org onboarding must not silently start cameras.
- The `CameraZoneAssignment` history table's append-only, never-back-dated guarantee (`app/domain/models.py:212-220`) must not be violated by any new bulk-move or bulk-import tooling.
- `CameraService.retire`'s refuse-rather-than-orphan behavior (`cameras.py:242-249`) must be preserved by any new bulk camera-deletion path introduced for organization offboarding.

## 15. Migration risks

- **Backfilling `Organization` beyond `org-unityworks`** is low-risk given the schema already requires `organization_id` everywhere (no wide nullable-then-backfill migration needed, unlike a system retrofitting tenancy).
- **The camera-ownership gap (§9.4)** should be fixed *before* org #2 exists in production, not as part of the multi-org migration itself — fixing it later, after cross-tenant `restaurant_id` values may already have been accepted by a lenient create path, could require a data-integrity audit of existing `Camera` rows.
- **If Option 2 (§11) is ever chosen**, backfilling `DVR` rows from distinct `(organization_id, restaurant_id, host, rtsp_port)` tuples in existing `Camera` data is nontrivial where `host` is blank (`default=""`, `app/domain/models.py:143`) for cameras that were registered without transport details — those would need either a synthetic placeholder DVR per camera or manual reconciliation.
- **`AuditEvent.organization_id`'s missing FK** (§9.3) should be fixed (add the FK, or explicitly document why it's intentionally loose) before any Organization *deletion* capability is built, since deletion is exactly the scenario that constraint would protect against.

## 16. Explicit non-goals (what this audit is NOT deciding)

- Whether a cross-tenant "platform operator" role should exist, and what it can see (§18 — decision required).
- Whether Organizations can ever be deleted (vs. only deactivated) — and if deletable, what happens to Restaurants with `ondelete=CASCADE` beneath them (§3) versus Incidents with `ondelete=SET NULL` (§10.F).
- Whether DVR becomes a first-class entity (Option 2) — left as a later, additive decision (§12).
- Any pricing, billing, or organization-tier/plan model — not present anywhere in the current schema and out of scope for this audit.
- User invitation/provisioning delivery mechanism (email, SSO, etc.) — explicitly deferred by the existing codebase (`administration.py:13-26`) and not re-opened here.
- Any specific migration script, Alembic revision, or code change — this document proposes no code.

## 17. Staged development plan

**Stage 1 — Data/domain architecture.** Decide and formalize the domain model option (§11/§12): confirm Option 1 (no schema change) versus adding `dvr_group_id` (Option 3). Add the missing `restaurant_id`-ownership check to `CameraService.create`/`update` (§9.4) as a prerequisite bug fix, independent of multi-org. Decide the FK question for `AuditEvent.organization_id` (§9.3, §15). Depends on: none — this is the foundation stage.

**Stage 2 — Organization isolation.** Formal verification (not just this audit's sampling) of every table/module flagged **D** or unverified in §8 — staff-hygiene modules, notifications, POS/patron integrations — with the same query-level tracing rigor applied here. Depends on Stage 1's decisions being fixed so the verification target is stable.

**Stage 3 — Administration APIs.** Build `POST/GET/PATCH /api/v1/organizations` gated by whatever cross-tenant role §18 decides on; build the User-provisioning API this audit found entirely absent (§5) — a prerequisite for use case A ("add an org" is incomplete without a way to create that org's first admin). Depends on Stage 2's isolation guarantees holding, since these routes are the first ones a genuinely cross-tenant caller will ever exercise.

**Stage 4 — Administration UI.** Build the Organization admin page and the drill-down IA proposed in §13, wire the missing deep links (site→cameras, camera→site/zone) and convert the free-text `restaurant_id` camera-creation field to a select. Depends on Stage 3's APIs existing.

**Stage 5 — Camera/DVR operational controls.** Split the collapsed "disable" action in the UI into its real distinct operations (§10.G: temporary disable / analysis-only / retire), and — if chosen — add the lightweight DVR-grouping metadata from Option 3. Depends on Stage 1's domain-model decision and Stage 4's UI shell.

**Stage 6 — Migration and backwards compatibility.** Write and test the actual migration for whatever schema changes Stages 1/5 introduced, verify every backwards-compatibility requirement in §14 against `org-unityworks`'s real data, and script the Organization #2 onboarding path end-to-end using only the new APIs (retiring the CLI-only path from §5/§9.2). Depends on all prior stages being code-complete.

**Stage 7 — Verification.** A dedicated cross-organization isolation test suite implementing use case H (§10.H) exhaustively — every resource type, both API and UI, with two real organizations and no shared data — run before any production rollout of a second organization. Depends on Stage 6.

## 18. Decisions that require owner approval

1. **Should `super_admin` become a genuinely cross-tenant role**, or should cross-org administration be a separate, new role/permission entirely (e.g. a `platform_operator` outside the existing `Role` enum's per-tenant model)? Today's `super_admin` is tenant-scoped by construction (§6) — this is a real fork, not a naming detail, because `AccessDecision.tenant_id` is currently mandatory and singular (`model.py:436-451`).
2. **Should Organizations ever be deletable**, or only deactivatable? This determines whether the `Restaurant.organization_id` `ondelete=CASCADE` behavior (§3) is ever actually exercised, and whether the `Incident.restaurant_id` `ondelete=SET NULL` history-loss behavior (§10.F) needs to be fixed first.
3. **Is Option 2's first-class DVR entity ever wanted**, or is informal host-grouping (current state) plus optional lightweight metadata (Option 3) sufficient long-term? This is a real product question about whether "DVR" needs to be independently manageable (firmware, credentials rotation across all its channels at once, health per-unit) versus per-camera.
4. **What is the actual account-provisioning mechanism** for a new organization's first admin? This audit found user creation entirely unbuilt (§5) — multi-org cannot ship use case A end-to-end without deciding this (email invite? SSO? admin-set-password despite the documented objection at `administration.py:15-26`?).
5. **Should camera creation's missing restaurant-ownership check (§9.4) be treated as a pre-existing single-tenant bug to fix immediately**, independent of and before any multi-org work, given it is already exploitable in principle once org #2 exists via the CLI script?

---

**Files touched to produce this report:** only this file. No production code, migration, configuration, or UI file was modified.
