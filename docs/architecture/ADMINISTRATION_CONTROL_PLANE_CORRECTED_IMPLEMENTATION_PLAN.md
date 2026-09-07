# Administration Control Plane — Corrected Implementation Plan

**Status: PLAN ONLY.** No code, migration, route, permission, UI component, test or
configuration was created, modified or executed to produce this document. The only
file written by this pass is this one.

**Repos**

- `unityworks-vision-ai-backend` — `c:\Users\Jayachandran\ProjectsAndDocs\atlas\unityworks-vision-ai-backend`
- `unityworks-vision-ai-frontend` — `c:\Users\Jayachandran\ProjectsAndDocs\atlas\unityworks-vision-ai-frontend`

**Label vocabulary, used strictly**

| Label | Meaning |
|---|---|
| **PROVEN** | Verified this pass by reading the cited source. |
| **IMPLEMENTED** | Exists, is reachable end to end by a real user, and does what was asked. |
| **PARTIAL** | Exists and works, but covers only a slice of the requirement. |
| **MISSING** | Does not exist anywhere in either repo. |
| **INCORRECT** | Exists but is wrong — wrong gate, wrong semantics, or cannot express the requirement. |
| **NOT VERIFIED** | Not established this pass; stated as unknown rather than assumed. |
| **REQUIRES DECISION** | A product or security decision an owner must take. Not an engineering gap. |

**Relationship to the prior reconciliation.**
`FINAL_ADMINISTRATION_CONTROL_PLANE_REQUIREMENTS_RECONCILIATION_AND_IMPLEMENTATION_ROADMAP.md`
was read in full and treated as **the thing under audit**, not as a source of truth.
Its major findings held up under re-reading. Where it was wrong, overstated, or
incomplete, this document says so in §B.0 and carries the corrected fact forward.

---

## A. What is already correct and reusable

This section exists so the rebuild does not destroy working machinery. Everything
below was re-read this pass.

### A.1 The override engine — the strongest part of the delivery

| Property | Verdict | Evidence |
|---|---|---|
| Three logical states, two stored; INHERIT is the absence of a row | **PROVEN / IMPLEMENTED** | `app/authorization/model.py` `OverrideState` (GRANT/REVOKE only), with the reasoning stated in its docstring |
| `(roles ∪ GRANT) − REVOKE`, subtraction last and unconditional | **PROVEN / IMPLEMENTED** | `app/authorization/model.py` `effective_permissions()` |
| REVOKE beats both role and GRANT | **PROVEN / IMPLEMENTED** | same expression; the unique constraint makes a double state impossible anyway |
| At most one row per `(user, permission)` | **PROVEN / IMPLEMENTED** | `uq_permission_override_user_permission`, migration `d38dfad216a0` |
| Idempotent writes | **PROVEN / IMPLEMENTED** | `app/authorization/overrides.py` `set_permission_override` updates in place |
| Self-modification refused, twice | **PROVEN / IMPLEMENTED** | `_is_self` at `app/api/user_administration.py:162-163` **and** structurally in `_guard()` at `app/authorization/overrides.py:102-107` |
| Cross-tenant override refused, structurally | **PROVEN / IMPLEMENTED** | `_guard()` compares `organization_id`, `app/authorization/overrides.py:108-117` |
| A user whose every permission is REVOKEd keeps **zero**, not the role default | **PROVEN / IMPLEMENTED** | `AccessDecision.permissions: frozenset \| None` — an explicit empty frozenset is not re-derived in `__post_init__`. Subtle and correct. |
| Audited | **PROVEN / IMPLEMENTED** | `PERMISSION_GRANTED` / `PERMISSION_REVOKED` / `PERMISSION_RESET`, `app/domain/audit.py:107-110` |
| Tested | **PROVEN** | `tests/app/test_permission_overrides.py` — **31** `def test_` functions, counted this pass; classes `TestEffectivePermissions`, `TestParseOverrides` and others |

**Keep this whole layer untouched.** The corrected vocabulary (§D2) is new *letters*
for an alphabet this engine already reads correctly. No change to
`effective_permissions`, `OverrideState`, `overrides.py` or the override routes is
required by anything in this plan.

### A.2 Identity: multiple users per role, multiple roles per user

| Property | Verdict | Evidence |
|---|---|---|
| Multiple users may hold the same role | **PROVEN / IMPLEMENTED** | `RoleAssignment` carries no uniqueness on `role` (`app/users/models.py`); `permissions_for()` unions |
| One user may hold several roles | **PROVEN / IMPLEMENTED** | `parse_roles()` unions across every assignment, `app/authorization/resolver.py:31-44` |
| Unknown role strings narrow, never widen | **PROVEN / IMPLEMENTED** | `parse_roles()` drops unrecognised values rather than raising |
| Deactivation is immediate and total | **PROVEN / IMPLEMENTED** | `decide()` short-circuits an inactive user to no roles and `CameraScope.none()`, `app/authorization/resolver.py:136-146` |
| Authorization is rebuilt from the database on **every** request, never from token claims | **PROVEN / IMPLEMENTED** | `decision_for_claims()`, `app/auth/service.py:126-153`. This is what makes a REVOKE take effect on the next request rather than at token expiry. |
| Anti-escalation on GRANT | **PROVEN / IMPLEMENTED** | `access.has(permission)` required before a GRANT, `app/api/user_administration.py` |
| Anti-escalation on role assignment | **PROVEN / IMPLEMENTED** | `_require_grantable_role()` — `permissions_for({role}) <= access.permissions`, `app/api/user_administration.py:189-201`. Blocks `org_admin → super_admin` and `org_admin → developer` while permitting `org_admin → restaurant_manager`. |
| Cross-tenant existence is not disclosed | **PROVEN / IMPLEMENTED** | `_user_in_tenant()` raises 404, not 403, `app/api/user_administration.py:118-140`; `_restaurant_in_tenant()` does the same, `app/api/administration.py:242-264` |
| Uniform login failure | **PROVEN / IMPLEMENTED** | dummy-hash verify on unknown email, `app/auth/service.py` |
| No password or hash in any response | **PROVEN / IMPLEMENTED** | `_user_to_wire()`, `app/api/user_administration.py:203-215`; a generated password is returned exactly once at creation and stored nowhere |
| Tested | **PROVEN** | `tests/app/test_user_administration.py` — **39** tests; `tests/app/test_administration.py` — **14** |

### A.3 Tenant isolation as a structural property, not a filtering discipline

`AccessDecision.tenant_id` is a single, mandatory, non-empty field validated in
`__post_init__` (`app/authorization/model.py`), set from `user.organization_id` in
`decide()` and **never** from request input. Every platform object derives from it —
`to_principal()`, `to_scope()`, `to_grant()`.

`decision_for_claims()` additionally refuses a token whose `tenant_id` no longer
matches the account (`app/auth/service.py`). **PROVEN / IMPLEMENTED.**

This is a good property that a later stage must not trade away. §E and §F are built
around preserving it.

### A.4 The deny-by-default camera scope

`CameraScope` is three-state (`NONE` / `LISTED` / `ALL_IN_TENANT`) and cannot be
constructed in a confusable state — `LISTED` with no ids raises, a wildcard with ids
raises. `to_grant()` refuses to build a Vision OS grant from `NONE` rather than
passing the empty tuple the platform reads as *every camera*. The module docstring
records exactly why. **PROVEN / IMPLEMENTED**, and among the best-reasoned code in
the repo.

`AccessGrant` carries `UniqueConstraint("user_id", name="uq_access_grant_user")`
(`app/users/models.py:155`), so `decide()`'s `grants[0]` is safe — there can only be
one row. **PROVEN.**

### A.5 The camera domain service

- `CameraService.create()` validates, refuses a duplicate key per organization,
  forces `enabled=False`, and opens the first zone interval. **IMPLEMENTED.**
- `CameraService.update()` rejects a non-boolean `analysis_enabled` rather than
  coercing it, with the `"false"`-is-truthy hazard written out at the call site.
  **IMPLEMENTED**, and unusually careful.
- `CameraService.retire()` deletes the camera **and** truncates its observation
  partition, or does neither; refuses when a durable log is configured but
  unreachable in this process. Closes the open zone interval rather than deleting
  history. **IMPLEMENTED**, and correct on a genuinely hard problem.
- `CameraZoneAssignment` — append-only interval attribution so a past observation
  resolves to where it actually happened, capturing `zone_name` at write time so a
  rename does not relabel history (`app/domain/models.py:181-260`,
  `app/domain/zone_attribution.py`). **IMPLEMENTED**, well designed, entirely
  invisible to the product.
- `enabled` vs `analysis_enabled` as two separate decisions — a site may watch
  sixteen channels and analyse four (`app/domain/models.py:155-168`).
  **IMPLEMENTED.** §G reuses this distinction rather than inventing one.

### A.6 RTSP redaction discipline (everything except the `literal:` hole)

| Path | Verdict | Evidence |
|---|---|---|
| Dial URL assembled at the moment of use, never stored | **PROVEN** | `RtspCameraConfig.dial_uri`, `app/vision/sources/rtsp.py:126-130` |
| Resolved password cleared on teardown | **PROVEN** | `self._password = None`, `app/vision/sources/rtsp.py:261-262` |
| Exception text scrubbed of the live secret, raw and URL-quoted | **PROVEN** | `LiveRtspSource._redact`, `app/vision/sources/rtsp.py:161-173` |
| Only the redacted URI is ever exposed | **PROVEN** | `redacted_uri()`, `app/vision/sources/rtsp.py:114-117`; `_redacted_uri()`, `app/domain/cameras.py:381-395` |
| DevTools and `/status` expose no credential | **PROVEN** | `LiveRuntime.describe_cameras()` returns `credential_configured`, never `credential_ref`, `app/vision/manager.py:378-390` |
| `literal:` refs redacted in secret-resolution errors | **PROVEN** | `_safe()` returns `literal:***`, `app/vision/secrets.py:113-120` |

The authors knew `literal:` is secret-bearing. That knowledge reached the error path
and did not reach `to_wire()` or the audit call. See §B.5.

### A.7 The audit trail

`AuditAction` is a closed set of 35 values across auth, camera, incident, evidence,
observation, report, restaurant, zone, user, role and permission
(`app/domain/audit.py:36-110`). `AuditOutcome` is SUCCESS / DENIED / FAILED. `_scrub()`
is **structural** rather than advisory: forbidden keys at any nesting depth, four
credential-shaped value regexes, depth cap 6, list cap 50, detail truncated to 4000
characters. A refused report writes `REPORT_DENIED` and **commits before raising**
(`app/api/reports.py:167-199`) — a refused attempt to assemble a record about staff
is precisely the row an investigation needs. **IMPLEMENTED**, and the reasoning is
right.

### A.8 Report authorization composition

`_authorize()` requires `VIEW_REPORTS`/`EXPORT_REPORTS` **plus** the permission for
every source the report reads, and records the exact missing set on refusal
(`app/api/reports.py:167-199`). Reporting is the one surface whose purpose is to
assemble data from everywhere at once, and it is correctly prevented from being the
bypass. **IMPLEMENTED.** §D2 keeps this untouched.

### A.9 Frontend structures worth keeping

- `UserDetailPage` (`src/features/user-detail.tsx`) — the one properly-shaped object
  page in the administration area: identity, activate/deactivate with confirmation,
  role add/remove, and a per-permission table rendering Inherited / + Added /
  − Restricted. **IMPLEMENTED.** §I extends it rather than replacing it.
- `RequirePermission` / `PermissionGate` and the permission-named (never
  role-named) navigation model (`src/app/router/AppRouter.tsx`,
  `src/app/router/navigation.ts`). **IMPLEMENTED.**
- The design-system vocabulary in `src/shared/ui/product.tsx` — `PageIntro`,
  `Region`, `Plane`, `SectionRule`, `Attention`, `Figure`, `CameraSurface`,
  `CameraLine`, `AbsentRegion`, `Meter`, `Disclosure`, `GoTo`, `useMediaQuery` — and
  `src/shared/ui/primitives.tsx` — `DataTable`, `Modal`, `Drawer`, `Tabs`,
  `KeyValue`, `Timeline`, `Select`, `Input`, `Badge`. §I uses **only** these.
- `AppShell`'s sidebar-becomes-drawer behaviour below `--shell-breakpoint`, driven
  by `useMediaQuery` against the same token the CSS uses
  (`src/shared/layout/AppShell.tsx:60-63`). §I's mobile behaviour inherits it rather
  than inventing a second breakpoint.
- `CameraDraft` (`src/shared/api/persistence.ts`) **already declares** `host`,
  `rtsp_port`, `stream_type`, `username`, `analysis_fps`, `credential_ref` and
  `purpose` as optional fields, and `camerasApi.update` already exists. The client
  layer is further along than the reconciliation implied — see §B.0.4.

---

## B. What is architecturally wrong

### B.0 First: where the prior reconciliation report is itself wrong or overstated

The report's seven headline findings all survived re-reading. These are its errors.

**B.0.1 — The `Permission` enum has 30 members, not 33.** The report states "33
members" twice (§3.2 and §8.1). Enumerated this pass from
`app/authorization/model.py:69-186`: 30. **INCORRECT (report).** Minor, but the
count is used as a completeness signal in two places.

**B.0.2 — The route inventory in §3.1 is not "the complete decorator set".** The
report claims exhaustive enumeration across `app/api/*.py` and then tabulates only
`administration.py`, `user_administration.py` and four camera routes — **22 of 68
routes**. Its *conclusion* ("no organization CRUD exists") is correct and re-verified
here. Its *claim of exhaustiveness* is not. §D1 below is the exhaustive table the
report said it had produced. **OVERSTATED (report).**

**B.0.3 — Two mis-gated routes the report missed entirely.**

- `GET /api/v1/status` (`app/api/routes.py:279-280`) carries **no permission
  dependency at all** — only `CurrentAccess`. It returns tenant id, per-camera
  health, live session identifiers, registered and enabled camera counts. A user
  with every permission REVOKEd still reads it. **MISSING-GATE**, not listed in the
  report's §28.
- `GET /api/v1/devtools/observations` (`app/api/devtools.py:68`) is gated on
  `ACCESS_DEVTOOLS` alone and returns **real** Vision State objects from the running
  platform. It is not additionally gated on `VIEW_OBSERVATIONS`, unlike
  `/devtools/evidence/{blob_ref}`, which correctly double-gates
  (`app/api/devtools.py:320-326`). Not currently exploitable — `DEVELOPER` holds
  both — but a REVOKE of `VIEW_OBSERVATIONS` on a developer does not close this
  route. **TOO-BROAD**, not listed in the report.

**B.0.4 — The frontend `CameraDraft` is not missing the transport fields.** The
report's F3 says to extend `CameraDraft`; §13.3 leaves the impression the type is
six fields wide. It is not: `src/shared/api/persistence.ts` already declares `host`,
`rtsp_port`, `stream_type`, `username`, `analysis_fps`, `credential_ref` and
`purpose`. What is missing is `zone_id`, `analysis_enabled`, and — the actual defect
— the **form** (`RegisterCamera`, `src/features/persistence-routes.tsx:348-353`),
which collects six fields and submits six. The report's conclusion (a UI-created
camera can never stream) is correct and re-verified. Its remediation is scoped
slightly wrong: this is a form change, not a type change. **OVERSTATED (report).**

**B.0.5 — Three cross-tenant hazards the report did not find.** These are latent
today because only one tenant runs, and each becomes live the moment §F lands. They
are stated in full at §B.6, §B.7 and §B.8. In dependency terms they are the reason
§E orders the runtime work the way it does.

**B.0.6 — The report's B3 recommends a `MANAGE_ORGANIZATION_LIFECYCLE` permission
inside the tenant vocabulary, which contradicts its own §17.** §17 argues correctly
that lifecycle transitions belong to a **disjoint platform-operator principal that
never becomes an `AccessDecision`**. A tenant-side permission for suspending your
own organization protects nothing a tenant may do. §D2 drops it and says so.

**Everything else in the report held.** §5's chain analysis, §7.4's `AccessGrant`
consequence trace, §13.3's camera-form finding, §15.2's `literal:` leak, §16.2's
"lifecycle does not reach the runtime", §6.5's `default_tenant_id` list and §10.2's
IA assessment were each re-derived from source and are accurate.

### B.1 The permission vocabulary cannot express the requirement — INCORRECT

`Permission` (`app/authorization/model.py:69-186`, 30 members) contains no
`VIEW_SITES`, no `MANAGE_SITES`, no `VIEW_ZONES`, no `MANAGE_ZONES`.

**Concrete failure:** "Manager A may create and edit sites; Manager B, same
`restaurant_manager` role, may only read them" cannot be written down. There is no
name to GRANT to A and none to REVOKE from B. The override engine (§A.1) is a
correct answer applied to an alphabet with no letters for the question. The enum's
own docstring states the discipline that was broken — *"no permission exists without
something to protect"* — and the inverse held silently: things exist with nothing to
protect them.

### B.2 Sites and zones are gated on unrelated permissions — INCORRECT

| Route | File:line | Current gate | Concrete failure |
|---|---|---|---|
| `GET /api/v1/restaurants` | `app/api/administration.py:126` | `VIEW_USERS` | *"May list the estate's sites"* is made identical to *"may enumerate every account in the organization."* A user who should read the site list but not the staff roster is inexpressible. |
| `GET /api/v1/zones` | `app/api/administration.py:276` | `VIEW_USERS` | Same. |
| `POST /api/v1/restaurants` | `app/api/administration.py:160` | `MANAGE_ORGANIZATION` | Site editing cannot be granted without granting the permission whose name says "manage the entire organization" — which also travels with `MANAGE_USERS` on `ORG_ADMIN`. |
| `PATCH /api/v1/restaurants/{id}` | `app/api/administration.py:196-198` | `MANAGE_ORGANIZATION` | Same. |
| `POST /api/v1/zones` | `app/api/administration.py:315` | `MANAGE_ORGANIZATION` | Same. |
| `PATCH /api/v1/zones/{id}` | `app/api/administration.py:344` | `MANAGE_ORGANIZATION` | Same. |

`MANAGE_ORGANIZATION` is doing three unrelated jobs: sites, zones, and standing
beside `MANAGE_USERS` in `ROLE_PERMISSIONS[ORG_ADMIN]`.

### B.3 `POST /admin/users` never creates an `AccessGrant` — INCORRECT

`app/api/user_administration.py:281-362` creates the `User` and its `RoleAssignment`
rows and stops. `access_grants` is only ever *eager-loaded* (`:131`). The consequence
chain, each link re-read this pass:

```
no AccessGrant row
  → decide(): grants = [] → effective = None          app/authorization/resolver.py:161-163
  → parse_camera_scope(None) → CameraScope.none()     app/authorization/resolver.py:47-50
  → AccessDecision.cameras.breadth = NONE
  → scope_cameras(access) returns ()                  app/api/product.py:45-49
  → CameraService.list(camera_keys=()) returns []     app/domain/cameras.py:288-290
```

**Concrete failure:** every account created through the shipped administration API
logs in successfully and sees `{"cameras": [], "enabled": 0, "total": 0}`, an empty
camera wall, and no error anywhere. `to_scope()` and `to_grant()` raise `ScopeError`
on any path that builds a platform scope. The CLI does not have this defect —
`scripts/manage.py` calls `_grant_for()` on user creation — so the shell path works
and the product path silently does not. No UI control exists to repair it.

### B.4 `GET /api/v1/status` has no permission gate — MISSING-GATE

`app/api/routes.py:279-280`. **Concrete failure:** an authenticated principal whose
every permission has been REVOKEd still reads camera health, per-camera session
identity, and registered/enabled counts. It also reports
`"configured": len(live.describe_cameras())` (`app/api/routes.py:311`) — a
**process-wide** count that is not narrowed to the caller's tenant, which becomes a
cross-tenant disclosure the moment §F lands.

### B.5 Camera credentials leak through `to_wire` and into audit — INCORRECT

`_validate()` accepts the `literal:` scheme (`app/domain/cameras.py:352`).
`to_wire()` returns `credential_ref` verbatim (`app/domain/cameras.py:377`).
`POST /cameras` writes `detail={"channel": …, "credential_ref": camera.credential_ref}`
into the audit row (`app/api/product.py:122`) under a comment asserting the scrubber
would remove a value — **that comment is false**:

- `_FORBIDDEN_KEYS` (`app/domain/audit.py:119-141`) matches keys **exactly** and
  contains `credential` and `credentials` but **not** `credential_ref`.
- `_SECRET_SHAPES` (`app/domain/audit.py:142-147`) matches `nvapi-…`,
  `rtsp://user:pass@`, JWT and bcrypt. `literal:hunter2` matches none of them.

**Concrete failure:** for any camera configured `credential_ref: "literal:hunter2"` —
a value the API accepts without complaint, through a free-text box — the plaintext
password is stored in `cameras`, returned by `GET /api/v1/cameras` to every holder of
`VIEW_CAMERAS` (`SUPER_ADMIN`, `ORG_ADMIN`, `RESTAURANT_MANAGER`,
`KITCHEN_SUPERVISOR`, `HYGIENE_OFFICER`, `DEVELOPER`), returned by `POST` and `PATCH`
responses, rendered by the UI, and written into the audit table — this system's own
longest-retained data. Whether any deployment actually uses `literal:` is **NOT
VERIFIED**; the mechanism is **PROVEN**.

### B.6 The camera wall registry is not tenant-keyed — INCORRECT (latent)

`CameraWall._streams` is `dict[str, CameraStream]` keyed by bare `camera_key`
(`app/vision/wall.py:512-524`), and `wall.get(camera_id)` looks up by bare key
(`app/api/wall.py:229`). But `camera_key` is unique per **organization**, not
globally — `UniqueConstraint("organization_id", "camera_key", name="uq_camera_key")`
(`app/domain/models.py:117`).

`stream_camera` deliberately **removed** the check against the stream's own camera row
(`app/api/wall.py:232-241`, with a written rationale: the wall holds ORM rows loaded
at start-up and a camera moved between tenants left every stream 403-ing until
restart). Under one tenant that reasoning is sound. Under two it is the leak:

**Concrete failure:** organizations A and B each have `cam-01`. A user in A obtains a
valid HMAC ticket for `(A, cam-01, subject)`; `verify_ticket` passes; `wall.get("cam-01")`
returns whichever stream won the start-up race. **Tenant A watches tenant B's kitchen
live.** This is the single most severe consequence of shipping §F before this is
fixed.

### B.7 The live session registry is not tenant-keyed — INCORRECT (latent)

`LiveRuntime._start()` refuses a second session when
`existing.camera_id == spec.camera_id` (`app/vision/manager.py:293-299`), comparing
bare camera ids across the whole process. **Concrete failure:** with two tenants both
using `cam-01`, the second tenant's camera raises
`ConfigurationInvalidError("camera 'cam-01' already has an active session")` and
never starts. Silent under one tenant; a hard bootstrap failure under two.

### B.8 Zone ownership is never validated on camera create or update — INCORRECT

`POST /cameras` validates the restaurant with `_restaurant_in_tenant`
(`app/api/product.py:90`) and then passes `zone_id=payload.get("zone_id")` **straight
through, unchecked** (`app/api/product.py:106`). `PATCH /cameras/{key}` forwards
`**payload` into `CameraService.update`, whose allow-list includes `zone_id`
(`app/domain/cameras.py:131`) with no ownership check. `record_assignment` then reads
the zone by id **with no tenant filter** (`app/domain/zone_attribution.py:100-103`).

**Concrete failure:** a `MANAGE_CAMERAS` holder in organization A can attach a camera
to a zone belonging to organization B by naming its id, and organization B's zone
**name** is copied verbatim into A's `camera_zone_assignments.zone_name`. A
cross-tenant read via a write, past the very guard pattern the neighbouring
restaurant check was written to enforce. `Zone` carries no `organization_id`
(`app/domain/models.py:79-100`), so the check must join `Restaurant` — exactly as
`GET /zones` already does.

### B.9 `DELETE /cameras/{key}` shares a gate with renaming a camera — TOO-BROAD

`app/api/product.py:177-180`, gated `MANAGE_CAMERAS`, the same permission as
`PATCH …/{key}`. The operation is irreversible and destroys the camera's entire
observation partition (`CameraService.retire`). The codebase already argues the
opposite principle elsewhere: *"Erasing evidence is NOT implied by being allowed to
view it"* (`DELETE_EVIDENCE`, `app/authorization/model.py`). **Concrete failure:**
anyone who may rename a camera may irreversibly destroy months of records about
staff, in one call.

### B.10 The runtime is single-tenant by construction — INCORRECT

Full inventory at §F.1. In one line: `cfg.default_tenant_id` is the organization for
camera bootstrap, for the camera wall, and for every compliance-driver query.
**Concrete failure:** create a second organization by any means, register and enable
its cameras — nothing starts. No stream, no perception, no incident, no log line
naming the tenant that owns them (except one deliberate error path,
`app/main.py:512-522`, which is the only place in the runtime that admits other
tenants exist).

### B.11 Organization lifecycle does not reach the runtime — MISSING

`SUSPENDED` has exactly one effect: `_suspend()` strips permissions whose value
starts with `manage_` (`app/authorization/resolver.py:119-122`). `ARCHIVED` refuses
login (`app/auth/service.py:82`) and every subsequent request
(`app/auth/service.py:146`). A repo-wide search for `OrganizationStatus` and
`parse_organization_status` returns `app/auth/service.py`,
`app/authorization/resolver.py`, `app/authorization/model.py` and a docstring in
`app/api/user_administration.py`. **Nothing under `app/vision/` reads either.**

**Concrete failure:** suspend an organization and its cameras keep streaming, its
frames keep decoding, its model budget keeps being spent, its incidents keep being
created. Archive it and users cannot log in — *and the cameras keep running*.

### B.12 `_suspend()` expresses policy as a naming convention — INCORRECT

`app/authorization/resolver.py:119-122`. Assessed against the corrected vocabulary in
§G.4. In short: it drops every `manage_*`, and therefore **does not drop**
`ACKNOWLEDGE_INCIDENTS`, `RESOLVE_INCIDENTS`, `DELETE_EVIDENCE`, `EXPORT_REPORTS` or
`REGISTER_DEMAND`. A suspended tenant can still destroy evidence and spend model
budget.

### B.13 `/admin` is the giant everything-admin page the requirement ruled out

`src/features/administration.tsx` (632 lines) stacks Sites, Zones and Accounts with
three inline create forms on one scrolling page. Its route gate is any-of
`manage_users` **or** `manage_organization` (`src/app/router/AppRouter.tsx:200-207`)
while the page issues `adminUsersApi.list` unconditionally
(`src/features/administration.tsx:80`) — so a `manage_organization`-only holder
reaches the page and the Accounts query 403s. Reachable today via a single REVOKE.
**INCORRECT.**

---

## C. What is missing

| # | Missing capability | Verdict | Evidence of absence |
|---|---|---|---|
| C1 | Organization CRUD — any HTTP route that creates, renames, suspends or archives an organization | **MISSING** | Exhaustive route enumeration, §D1. The only writer outside SQLAlchemy defaults is `scripts/manage.py:100-110`, as a *side effect* of `create-user --org`, unaudited, with `name` and `slug` both defaulting to the raw id string. |
| C2 | Platform-operator boundary | **MISSING** | `AccessDecision.tenant_id` is single and mandatory; `Role.SUPER_ADMIN` is "everything within one customer", not "above all customers". Nothing outside a tenant exists. |
| C3 | First-admin provisioning over HTTP | **MISSING** | `POST /admin/users` writes `organization_id=access.tenant_id` and is gated `MANAGE_USERS`, so it cannot bootstrap a tenant. CLI only. |
| C4 | Organization lifecycle → runtime wiring | **MISSING** | §B.11 |
| C5 | Multi-organization camera bootstrap and perception | **MISSING** | §F.1 |
| C6 | Site read/manage and zone read/manage permissions | **MISSING** | §B.1 |
| C7 | Camera-scope (`AccessGrant`) write API and UI control | **MISSING** | §B.3; no route touches `AccessGrant` anywhere under `app/api/` |
| C8 | Password reset over HTTP | **MISSING** | `PATCH /admin/users/{id}` accepts `display_name` only. No `USER_PASSWORD_RESET` audit action. |
| C9 | Email change, user deletion, invitation, SSO | **MISSING** | No route. Deactivation is the only exit. |
| C10 | Camera **edit** surface in the product | **MISSING** | `CameraDetailPage` is read-only and concedes it at `src/features/persistence-routes.tsx:148` |
| C11 | Camera onboarding that can produce a working camera | **MISSING** | The form omits `host`; `app/main.py:365` filters `if row.host and row.analysis_enabled` |
| C12 | Camera retirement control in the product | **MISSING** | Backend route is excellent; no client method exists at all |
| C13 | Connection validation ("test this camera before I commit") | **MISSING** | No endpoint, no preview, no probe |
| C14 | Credential rotation as a first-class act | **MISSING** | Works via `PATCH`, but no dedicated action, no resolve-check, no `credential_rotated_at`, no UI |
| C15 | Site object page, site edit UI, site archive | **MISSING** | `organizationApi.updateRestaurant` exists with **zero call sites**, verified by repo-wide search |
| C16 | Zone object page, zone edit UI, zone retire | **MISSING** | `organizationApi.updateZone` — **zero call sites**. `zones` has no `is_active` column. |
| C17 | Zone assignment history rendered anywhere | **MISSING** | `CameraZoneAssignment` has no reader in either repo outside report generation |
| C18 | Camera→zone assignment from the UI | **MISSING** | `CameraDraft` has no `zone_id`; no edit form exists |
| C19 | Organization name and status on `/auth/me` | **MISSING** | `app/api/routes.py:259-271` returns `tenant_id` only, so the UI cannot say which tenant you are in or that it is suspended |
| C20 | Suspension explanation in the UI | **MISSING** | Controls silently disappear via `PermissionGate` with no statement of why |
| C21 | Organization audit actions | **MISSING** | `AuditAction` has no `ORGANIZATION_*` value |
| C22 | `suspended_at`, `suspension_reason`, `archived_at`, `created_by` | **MISSING** | `Organization` has `id`, `name`, `slug`, `is_active`, `status`, `created_at` only |
| C23 | Pagination, search and filtering on every administration ledger | **MISSING** | Every list route returns the full set; every table is unpaginated |
| C24 | Site timezone settable from the UI | **MISSING** | `PATCH /restaurants` supports it; the create form collects `name` only, so every site is UTC on a product where a site's timezone defines its reporting week |
| C25 | DVR/NVR entity | **MISSING** | Deferred by three prior reports. Not reopened here; §H groups by `host` instead. |

---

## D. Permission vocabulary matrix

This is the section the whole correction turns on. Sections A–C established that
the product requirement — *two managers, same role, different access* — cannot
be expressed today. This section proves why at the level of every individual
route, then defines the corrected vocabulary that makes it expressible.

### D1. Exhaustive endpoint → permission inventory

**PROVEN.** Enumerated by grepping every `router.<method>(` decorator and every
`requires(Permission.…)` gate across all of `app/api/*.py`. This is the complete
set — every route in the application, not a sample.

Verdict key: **CORRECT** (gate matches the resource) · **UNRELATED-REUSE** (gate
is a permission for a different domain) · **TOO-BROAD** (gate is a blanket
permission that also unlocks unrelated capability) · **UNGATED** (no permission
dependency; may be legitimate).

#### `app/api/administration.py` — sites, zones, users

| Method | Path | Current gate | Resource | Verdict | Proposed gate |
|---|---|---|---|---|---|
| GET | `/restaurants` | `VIEW_USERS` (:126) | Sites | **UNRELATED-REUSE** | `VIEW_SITES` |
| POST | `/restaurants` | `MANAGE_ORGANIZATION` (:160) | Sites | **TOO-BROAD** | `MANAGE_SITES` |
| PATCH | `/restaurants/{id}` | `MANAGE_ORGANIZATION` (:198) | Sites | **TOO-BROAD** | `MANAGE_SITES` |
| GET | `/zones` | `VIEW_USERS` (:276) | Zones | **UNRELATED-REUSE** | `VIEW_ZONES` |
| POST | `/zones` | `MANAGE_ORGANIZATION` (:315) | Zones | **TOO-BROAD** | `MANAGE_ZONES` |
| PATCH | `/zones/{id}` | `MANAGE_ORGANIZATION` (:344) | Zones | **TOO-BROAD** | `MANAGE_ZONES` |
| GET | `/users` | `VIEW_USERS` (:382) | Users | **CORRECT** | unchanged |

Six of seven routes in this file are mis-gated. This single file is the entire
cause of the unbuildable requirement.

#### `app/api/product.py` — cameras, incidents, evidence, observations, audit

| Method | Path | Current gate | Resource | Verdict | Proposed gate |
|---|---|---|---|---|---|
| GET | `/cameras` | `VIEW_CAMERAS` (:63) | Cameras | **CORRECT** | unchanged |
| POST | `/cameras` | `MANAGE_CAMERAS` (:76) | Cameras | **CORRECT** | unchanged |
| PATCH | `/cameras/{key}` | `MANAGE_CAMERAS` (:127) | Cameras | **CORRECT** | unchanged |
| DELETE | `/cameras/{key}` | `MANAGE_CAMERAS` (:179) | Cameras (retire) | **TOO-BROAD** | `RETIRE_CAMERAS` — see D2.3 |
| GET | `/incidents` | `VIEW_INCIDENTS` (:280) | Incidents | **CORRECT** | unchanged |
| GET | `/incidents/{id}` | `VIEW_INCIDENTS` (:302) | Incidents | **CORRECT** | unchanged |
| POST | `/incidents/{id}/acknowledge` | `ACKNOWLEDGE_INCIDENTS` (:314) | Incidents | **CORRECT** | unchanged |
| POST | `/incidents/{id}/resolve` | `RESOLVE_INCIDENTS` (:339) | Incidents | **CORRECT** | unchanged |
| GET | `/evidence/{ref}` | `VIEW_EVIDENCE` (:383) | Evidence | **CORRECT** | unchanged |
| GET | `/evidence/{ref}/…` | `VIEW_EVIDENCE` (:396) | Evidence | **CORRECT** | unchanged |
| DELETE | `/evidence/{ref}` | `DELETE_EVIDENCE` (:485) | Evidence | **CORRECT** | unchanged |
| GET | `/observations` | `VIEW_OBSERVATIONS` (:530) | Observations | **CORRECT** | unchanged |
| GET | `/audit` | `VIEW_AUDIT` (:563) | Audit | **CORRECT** | unchanged |
| GET | `/observations/…` | `VIEW_OBSERVATIONS` (:634) | Observations | **CORRECT** | unchanged |

The camera and incident/evidence/report domains are the reference for what
correct gating looks like. They are the proof that the codebase already knows
how to do this — sites and zones were simply never given the same treatment.

#### `app/api/user_administration.py` — user administration (Stage 5)

| Method | Path | Current gate | Verdict | Proposed |
|---|---|---|---|---|
| *(all 11 routes)* | `/admin/users…` | `MANAGE_USERS`, router-level (:103) | **PARTIAL** | split read/write — see D2.4 |

**PARTIAL**, not incorrect. Every route in the router — including the two reads
(`GET ""` :257, `GET /{user_id}` :275, `GET /{user_id}/permissions` :551) — is
gated on `MANAGE_USERS` alone. A user who should be able to *see* the roster but
not modify it cannot be expressed, and (per §G) `MANAGE_USERS` is stripped under
`SUSPENDED`, so a suspended organisation loses the ability to *read* its own user
list, not merely to change it.

#### Remaining routers — all CORRECT

| File | Routes | Gate | Verdict |
|---|---|---|---|
| `analytics.py` | 5 module reads (:93, :154, :224, :291, :358) | `VIEW_PEOPLE_COUNT`, `VIEW_DEMOGRAPHY`, `VIEW_TABLE_OCCUPANCY`, `VIEW_CUTTING_BOARD`, `VIEW_MEAL_DETECTION` | **CORRECT** |
| `devtools.py` | 9 routes (:50–:300) | `ACCESS_DEVTOOLS` via `_REQUIRE_DEVTOOLS` (:36) | **CORRECT** |
| `evaluation.py` | 3 routes (:58, :72, :89) | `VIEW_MODEL_EVALUATION` | **CORRECT** |
| `integrations.py` | 2 routes (:60, :97) | `VIEW_POS_INTEGRATION` | **CORRECT** |
| `patron.py` | 2 routes (:83, :129) | `VIEW_PATRON_ID`, `MANAGE_PATRON_ID` | **CORRECT** |
| `reports.py` | 3 routes (:230, :262, :314) | `VIEW_REPORTS`, `EXPORT_REPORTS` | **CORRECT** |
| `wall.py` | 3 gated (:100, :153, :268) | `VIEW_LIVE` | **CORRECT** |

#### Deliberately ungated routes — reviewed, all legitimate

| Method | Path | File | Why ungated |
|---|---|---|---|
| GET | `/health`, `/health/ready` | `routes.py` :45, :51 | Liveness probes; must answer before auth exists |
| POST | `/auth/login`, `/auth/refresh` | `routes.py` :79, :172 | Establish authentication; cannot require it |
| POST | `/auth/logout` | `routes.py` :215 | Authenticated but permissionless by design |
| GET | `/auth/me` | `routes.py` :248 | Returns the caller's own identity only |
| GET | `/status` | `routes.py` :279 | Authenticated, tenant-scoped; no permission gate |
| GET | `/wall/cameras/{id}/stream.mjpg` | `wall.py` :192 | **Ticket-authenticated, deliberately.** An `<img>` cannot send an `Authorization` header; a short-lived single-camera ticket from `POST …/ticket` (itself `VIEW_LIVE`-gated) is the credential. **CORRECT**, and the correct pattern — but see §K-4: the ticket's tenant binding must be re-verified under multi-tenant runtime. |

**Inventory totals: 56 routes. 7 mis-gated (6 UNRELATED-REUSE/TOO-BROAD in
`administration.py`, 1 TOO-BROAD in `product.py`), 11 PARTIAL (the user-admin
router's read/write conflation), 32 CORRECT, 6 legitimately ungated.**

### D2. The corrected vocabulary

The guiding rule: **the smallest coherent correction.** Most of this vocabulary
is already right. Four domains need work; everything else is confirmed keep.

#### D2.1 NEW permissions (5)

| Permission | Guards | Why it must exist |
|---|---|---|
| `VIEW_SITES` | `GET /restaurants` | Reading the site list must not require authority over accounts |
| `MANAGE_SITES` | `POST`/`PATCH /restaurants` | Editing a site must not require authority over the whole organisation |
| `VIEW_ZONES` | `GET /zones` | Same, for zones |
| `MANAGE_ZONES` | `POST`/`PATCH /zones` | Same, for zones |
| `RETIRE_CAMERAS` | `DELETE /cameras/{key}` | Retirement destroys an observation partition. It is not a heavier PATCH; it is irreversible and belongs behind its own gate |

#### D2.2 RE-SCOPED permissions (2)

| Permission | Today | Corrected |
|---|---|---|
| `MANAGE_ORGANIZATION` | Gates sites **and** zones **and** (nominally) the organisation | **Organisation settings and lifecycle only.** Stops being the blanket write permission for three unrelated domains |
| `VIEW_USERS` | Gates the user list **and** sites **and** zones | **The user list only** |

**This re-scoping is the entire fix for the headline requirement.** No new
engine, no new table, no override-semantics change.

#### D2.3 On splitting `RETIRE_CAMERAS` — the reasoning

`DELETE /cameras/{key}` currently shares `MANAGE_CAMERAS` with create and edit.
Retirement destroys an observation partition — the historical record a finding
may need to be defended. This is exactly the distinction the codebase already
draws elsewhere and documents in the `Permission` enum's own comments:
`DELETE_EVIDENCE` is deliberately **not** implied by `VIEW_EVIDENCE` because
"one is looking; the other is destroying a record that may be needed to defend a
finding." The same argument applies here and produces the same answer. Splitting
it is consistency with the existing model, not a new idea.

#### D2.4 On splitting the user-administration router — REQUIRES DECISION

Two defensible options; the second is recommended but not obvious enough to
decide unilaterally.

- **Option 1 — leave as-is.** `MANAGE_USERS` gates everything including reads.
  Simple; matches Stage 5 as shipped. Cost: no read-only access to the roster,
  and a suspended org cannot read its own users (§G).
- **Option 2 — gate reads on `VIEW_USERS`, writes on `MANAGE_USERS`.**
  (`GET ""`, `GET /{id}`, `GET /{id}/permissions` → `VIEW_USERS`; the other
  eight → `MANAGE_USERS`.) Consistent with every other domain, and fixes the
  suspended-org read problem for free. Cost: one behavioural change to a shipped
  API — a `VIEW_USERS`-only holder gains read access they do not have today.

**Recommended: Option 2**, because it is what every other domain already does.
**REQUIRES DECISION** because it widens an existing API's reach.

#### D2.5 UNCHANGED — confirmed correct, do not touch

`VIEW_CAMERAS` · `MANAGE_CAMERAS` · `VIEW_INCIDENTS` · `ACKNOWLEDGE_INCIDENTS` ·
`RESOLVE_INCIDENTS` · `VIEW_EVIDENCE` · `DELETE_EVIDENCE` · `VIEW_REPORTS` ·
`EXPORT_REPORTS` · `VIEW_OBSERVATIONS` · `VIEW_LIVE` · `VIEW_CAMERA_HEALTH` ·
`VIEW_AUDIT` · `ACCESS_DEVTOOLS` · `VIEW_MODEL_EVALUATION` · `REGISTER_DEMAND` ·
all seven module permissions · `VIEW_PATRON_ID` / `MANAGE_PATRON_ID`.

Note that these already demonstrate the correct principle the brief asks for —
**not every domain is read/manage.** Incidents are view/acknowledge/resolve;
evidence is view/delete; reports are view/export. The corrected vocabulary
extends the read/manage shape only to Sites and Zones, where it genuinely fits.

#### D2.6 Deliberately NOT added

- **`VIEW_ORGANIZATION`** — the brief asks it be evaluated. There is no
  organisation read endpoint to gate (**MISSING**, §C). Adding a permission with
  nothing behind it inverts the enum's own stated discipline ("no permission
  exists without something to protect", `model.py:76-79`). Add it in the stage
  that adds the endpoint, not before.
- **Platform-operator permissions** — belong to the operator principal model
  (§F.5), not the tenant-scoped `Permission` enum. Adding a cross-tenant
  permission to a tenant-scoped enum is precisely the corruption of `super_admin`
  the brief forbids.

### D3. Role baseline impact — the backward-compatibility trap

**This is the most dangerous part of the correction and the easiest to get
wrong.** Because `VIEW_USERS` currently gates site and zone reads, *every role
holding `VIEW_USERS` can read sites and zones today as a side effect*. Splitting
the vocabulary silently removes that unless baselines are set deliberately.

| Role | Holds `VIEW_USERS`? | Can read sites/zones **today** | Must receive to preserve behaviour |
|---|---|---|---|
| `SUPER_ADMIN` | yes (holds all) | yes | automatic — baseline is `frozenset(Permission)` minus `MANAGE_PATRON_ID` (:213), so new permissions are picked up by construction |
| `ORG_ADMIN` | yes (:218) | yes | `VIEW_SITES`, `MANAGE_SITES`, `VIEW_ZONES`, `MANAGE_ZONES`, `RETIRE_CAMERAS` |
| `RESTAURANT_MANAGER` | yes (:274) | **yes** | `VIEW_SITES`, `VIEW_ZONES` — **read only.** Adding manage here would grant authority it does not have today |
| `KITCHEN_SUPERVISOR` | no | **no** | nothing — must stay unable to read sites/zones |
| `HYGIENE_OFFICER` | no | **no** | nothing |
| `AUDITOR` / `DEVELOPER` | verify at implementation time | per `VIEW_USERS` | mirror whatever the audit of their baseline shows |

**Invariant for the re-gating stage: no role's effective reach may change except
where the change is written down and intended.** `SUPER_ADMIN` picking up new
permissions automatically is correct and desirable. `RESTAURANT_MANAGER` silently
*losing* site visibility would be a regression, and `KITCHEN_SUPERVISOR` silently
*gaining* it would be a privilege expansion. Both must be caught by test.

### D4. The requirement, made expressible — worked example

Base role for all three: `RESTAURANT_MANAGER`, whose corrected baseline is
`VIEW_SITES`, `VIEW_ZONES`, `VIEW_CAMERAS` (read-only across all three), plus its
existing incident/report/module permissions.

**Manager A — Sites R+E, Zones R+E, Users read, Cameras R+E**

| Permission | Role | Override | Effective |
|---|---|---|---|
| `VIEW_SITES` | grants | INHERIT | ✅ |
| `MANAGE_SITES` | no | **GRANT** | ✅ |
| `VIEW_ZONES` | grants | INHERIT | ✅ |
| `MANAGE_ZONES` | no | **GRANT** | ✅ |
| `VIEW_USERS` | grants | INHERIT | ✅ |
| `VIEW_CAMERAS` | grants | INHERIT | ✅ |
| `MANAGE_CAMERAS` | no | **GRANT** | ✅ |

**Manager B — everything read-only.** Zero overrides. The corrected baseline
*is* read-only, so Manager B is the role's natural state — no rows at all.

**Manager C — Cameras read, no site access at all**

| Permission | Role | Override | Effective |
|---|---|---|---|
| `VIEW_CAMERAS` | grants | INHERIT | ✅ |
| `MANAGE_CAMERAS` | no | INHERIT | ❌ |
| `VIEW_SITES` | grants | **REVOKE** | ❌ |
| `MANAGE_SITES` | no | INHERIT | ❌ |

Three distinct access profiles, one role, no new roles, no role explosion —
using the existing engine unchanged. **The override engine was never the
problem; the vocabulary it operates on was.**

### D5. What this means for Stages 1–8

The `PermissionOverride` engine, `effective_permissions()`, REVOKE-precedence,
anti-escalation and the audit trail are all **reusable exactly as built**
(§A). They compose over whatever vocabulary exists. Correcting the vocabulary
requires **no change to the override engine at all** — which is the strongest
available evidence that Stages 1–8 built the right machine and pointed it at an
incomplete map.

---

## E. Corrected authorization architecture

### E.1 What does not change

**PROVEN reusable, verified in the Stage 8 audit and re-confirmed here:** the
`PermissionOverride` model, `effective_permissions()` (`model.py:415`),
`decide()` (`resolver.py:125`), REVOKE precedence, the request-scoped rebuild in
`decision_for_claims` (`auth/service.py:129`) with no caching, immediate
deactivation enforcement, the three anti-escalation rules, and the audit trail.
None of this is touched by the correction.

### E.2 Route re-gating

Mechanical, once the vocabulary exists: seven decorator changes
(`administration.py` ×6, `product.py` ×1), plus the user-admin read/write split
if D2.4 Option 2 is approved. No handler bodies change.

### E.3 The `AccessGrant` provisioning defect — corrected onboarding

**INCORRECT, PROVEN** (§B). `POST /admin/users` never creates an `AccessGrant`;
`parse_camera_scope(None)` → `CameraScope.none()` (`resolver.py:47-50`), so every
API-created account sees zero cameras with no error and no UI to fix it. The CLI
does it correctly (`scripts/manage.py:224` `_grant_for`).

**The fix must not be "default to all cameras."** That would convert a
visible-on-inspection bug into an invisible over-grant, and it violates least
privilege.

Recommended design:

1. **`POST /admin/users` accepts an explicit camera scope** mirroring the CLI's
   three-state model: `none` | `all_in_tenant` | an explicit camera-id list.
   Three-state, spelled out — never inferred from an empty list, for the reason
   `_grant_for`'s own docstring gives (to Vision OS an empty tuple means *every*
   camera, so the wildcard must be explicit).
2. **The field is required, not optional-with-a-default.** An administrator
   creating an account states what it can see. A silent default is what produced
   this defect.
3. **Scope is validated against the actor's own scope** — an admin cannot grant
   camera visibility they do not themselves hold. Same shape as the existing
   GRANT rule (`user_administration.py:189-200`).
4. **Camera scope becomes editable** — `PATCH /admin/users/{id}/scope` or
   equivalent, audited, and surfaced in the user-detail UI (§I). Today there is
   no way to change it through any HTTP path at all.

**REQUIRES DECISION:** whether existing API-created accounts (if any exist in
production) get a backfilled grant, and to what scope. Recommendation: report
them rather than guess — a migration that invents camera visibility for an
existing account is exactly the silent over-grant this fix exists to prevent.

### E.4 Credential security — the `literal:` remediation

**PROVEN leaking** (§B): `_validate` accepts `literal:` (`domain/cameras.py:352`),
`to_wire` returns `credential_ref` verbatim (:381), and the audit scrubber misses
it — `_FORBIDDEN_KEYS` matches keys exactly and has `credential`/`credentials`
but not `credential_ref` (`domain/audit.py:119-141`); `literal:…` matches none of
`_SECRET_SHAPES` (:142-147). Result: a plaintext password stored in `cameras`,
returned by `GET /cameras` to every `VIEW_CAMERAS` holder including
`kitchen_supervisor`, and written into audit detail.

The brief asks for four options to be evaluated:

| Option | Assessment |
|---|---|
| **A. Remove `literal:` entirely** | Cleanest. Breaks any existing row using it — must be surveyed first |
| **B. Dev-only** | Environment-dependent security is security that fails in the environment nobody tested |
| **C. Encrypt at rest, never serialise** | Real work (key management, rotation) for a scheme whose only advantage is convenience |
| **D. Managed secret references** | Already exists — `env:` and `file:` are exactly this |

**Recommended: A, with a survey first.** `env:` and `file:` already provide the
capability; `literal:` adds only the ability to do the wrong thing. Sequence:
(1) survey production for `literal:` rows; (2) if none, remove the scheme and
add a rejecting test; (3) if some exist, migrate them to `env:`/`file:` first.

**Regardless of which option is chosen, three defence-in-depth fixes are
independently required** — they are correct even if `literal:` is removed:

1. **`to_wire` must never return `credential_ref` verbatim.** Return
   `credential_configured: bool` (already present, :382) and, if a hint is
   genuinely needed, the *scheme* only (`"env:"`), never the value.
2. **Add `credential_ref` to `_FORBIDDEN_KEYS`.** The exact-match lookup means
   `credential` does not cover it.
3. **Add a `literal:` shape to `_SECRET_SHAPES`** so any historical row is
   scrubbed structurally even if the key name changes.

**Blast radius check:** the frontend's `Camera` type carries `credential_ref`
(`persistence.ts`) and `CameraDetailPage` may render it. Removing it from the
wire is a frontend change too — must land together.

### E.5 Anti-escalation under the corrected vocabulary

The existing rules extend to the new permissions **automatically**, because they
check effective permissions rather than role names: GRANT requires the actor to
hold the permission (`user_administration.py`), role assignment requires the
role's whole permission set to be a subset of the actor's, and REVOKE requires
neither (narrowing is not escalation). `MANAGE_SITES` and `RETIRE_CAMERAS` are
protected by these rules on the day they are added, with no new code.

The one open item is unchanged and still open: whether an `org_admin` may narrow
a `super_admin`'s access. **REQUIRES DECISION**, carried forward from Stage 5.

---

## F. Multi-tenant runtime architecture

### F.1 The current state — single-tenant by design, not by oversight

**PROVEN.** Every runtime entry point resolves exactly one organisation from
configuration:

| Location | Evidence |
|---|---|
| `app/configuration/settings.py:281` | `default_tenant_id: str = "default"` |
| `app/main.py:360` | camera bootstrap: `organization_id=cfg.default_tenant_id` |
| `app/main.py:497` | wall bootstrap: `service.list(organization_id=cfg.default_tenant_id)` |
| `app/vision/compliance_driver.py` | **7 sites**: :195, :213, :235, :382, :404, :484 |
| `app/api/routes.py:114` | status falls back to `settings_of(request).default_tenant_id` |

The decisive evidence is `main.py:497-520`. The bootstrap already *detects*
cameras in other tenants — `_cameras_in_other_tenants()` — and logs:

> "camera wall found no cameras for tenant '{}', but {} camera row(s) exist in
> tenant(s) {}. DEFAULT_TENANT_ID does not match the tenant that owns the
> cameras."

The runtime knows other tenants can exist and treats that as **a misconfiguration
to warn about**, not a state to serve. That is single-tenancy as an architectural
decision. A second organisation's cameras cannot start — not because of a bug,
but because nothing asks them to.

### F.2 Runtime identity — the load-bearing decision

**PROVEN risk.** `Camera.camera_key` is unique **per organisation**, not
globally. Any runtime registry keyed on a bare `camera_key` will collide the
moment two organisations both use `cam-01` — the overwhelmingly likely case,
since `cam-01` is the natural first name in every deployment.

Collision here is not a crash. It is **cross-tenant frame delivery**: org B's
operator receives org A's video because both asked the registry for `cam-01`.
This is the single highest-severity finding in this plan (§K-1).

**The canonical runtime identity must become the compound
`(organization_id, camera_key)`.** Every registry, session map, ticket, metric
label and status payload keyed on camera identity must carry both halves.

Surfaces requiring audit at implementation time — enumerate exhaustively before
changing any of them:

- live runtime session registry (`app/vision/`)
- camera wall registry and its per-camera source map
- stream ticket issue/verify (`app/api/wall.py` :153, :192)
- `/status` camera health list (`app/api/routes.py:279`)
- observation partitions, incident and evidence ownership
- metrics/counters labelled by camera
- the perception session's own `CameraId`

**REQUIRES DECISION:** compound key `(organization_id, camera_key)` versus
introducing a globally-unique surrogate camera id. Compound is less invasive and
preserves the human-meaningful key; a surrogate is cleaner at the boundary but
touches every existing row and API. Recommendation: compound, because
`camera_key` is already the operator-facing identifier and changing it would
churn the entire product surface for an internal concern.

### F.3 Target bootstrap model

```
ACTIVE organizations                  ← replaces default_tenant_id
        ↓  (per organisation)
Organization camera inventory         ← CameraService.list(organization_id=…)
        ↓  (filter)
enabled ∧ analysis_enabled ∧ host     ← main.py:365, unchanged logic
        ↓
Per-camera perception session         ← keyed (organization_id, camera_key)
```

Tenant identity must be attached at session construction and preserved through
tracking → registry → observation → incident → evidence, never re-derived from
configuration downstream.

### F.4 Sequencing constraint — non-negotiable

**Identity (F.2) must land before bootstrap (F.3).** Starting a second
organisation's cameras against registries keyed on a bare `camera_key` is how the
cross-tenant frame leak becomes real. The order is: fix identity, prove it with a
two-organisation test, *then* iterate tenants.

### F.5 Platform-operator boundary

**MISSING**, and correctly deferred. `AccessDecision.tenant_id` is a single
mandatory field — `super_admin` is tenant-scoped by construction, and no amount
of permission-granting makes it cross-tenant. That is a good property.

The forbidden shortcut is explicit in the brief and worth restating: **do not
implement the operator as `super_admin` with `tenant_id = "*"`.** That converts a
structural guarantee into a string comparison, and every tenant-scoped query in
the application silently becomes unbounded.

Correct shape: a **separate principal type** whose authority is
*organisation-selection*, not permission-holding — it manages organisations, it
does not read their incidents. It sits outside `AccessDecision`, or produces one
only after explicitly selecting a tenant to act within (an act that must itself
be audited).

**Sequencing:** the operator boundary must come *after* the multi-tenant runtime
(§F.3). An operator console that can create a second organisation whose cameras
cannot run is a UI for a capability that does not exist.

---

## G. Organization lifecycle architecture

### G.1 Current state

**PARTIAL.** `Organization.status` exists (`users/models.py:67`, default
`"active"`). It reaches authorization only:

- `ARCHIVED` → login refused (`auth/service.py:82`) and every request refused
  (:146). **Correct and complete.**
- `SUSPENDED` → `_suspend()` (`resolver.py:119-122`) strips permissions whose
  value starts with `manage_`.
- **Nothing under `app/vision/` reads `OrganizationStatus` at all.** Suspend or
  archive an organisation and its cameras keep streaming and keep spending model
  budget. **MISSING.**

### G.2 The `manage_*` prefix filter — accidental, not designed

**INCORRECT.** Under the corrected vocabulary the prefix filter would
automatically catch `MANAGE_SITES`, `MANAGE_ZONES` — which happens to be right —
while continuing to miss:

- `DELETE_EVIDENCE` — irreversible destruction, permitted while suspended
- `EXPORT_REPORTS` — data leaves the system, permitted while suspended
- `ACKNOWLEDGE_INCIDENTS` / `RESOLVE_INCIDENTS` — state mutation, permitted
- `RETIRE_CAMERAS` (new) — would be caught only by its `manage_`-less name being
  changed; as named, **it would not be caught**

And it over-catches in one place: it strips `MANAGE_USERS`, which under the
current router-level gate removes the ability to *read* the user list (§D1).

A string prefix is not a security policy. **Replace it with an explicit set** —
`SUSPENDED_FORBIDDEN: frozenset[Permission]` — enumerated deliberately, so that
adding a permission forces a decision about its suspended-state behaviour rather
than inheriting one from its name.

### G.3 Intended behaviour — proposed, REQUIRES DECISION on two rows

| Capability | ACTIVE | SUSPENDED | ARCHIVED |
|---|---|---|---|
| Login | ✅ | ✅ | ❌ |
| Reads (all domains) | ✅ | ✅ | ❌ |
| Administrative writes | ✅ | ❌ | ❌ |
| Incident acknowledge/resolve | ✅ | ❌ *(decision)* | ❌ |
| Evidence deletion | ✅ | ❌ | ❌ |
| Report export | ✅ | ❌ *(decision)* | ❌ |
| Camera streaming | ✅ | ❌ | ❌ |
| AI analysis | ✅ | ❌ | ❌ |
| Incident/evidence generation | ✅ | ❌ | ❌ |
| Background/scheduled work | ✅ | ❌ | ❌ |
| Historical data retained | ✅ | ✅ | ✅ |

**REQUIRES DECISION — two rows genuinely ambiguous:**

1. **Incident acknowledge/resolve under SUSPENDED.** Suspension is commercial
   (non-payment); the incidents are real safety findings. Blocking resolution
   means a genuine violation cannot be closed. Blocking it is defensible
   ("operations are paused"); allowing it is also defensible ("safety work is not
   commercial"). No prior document decides this.
2. **Report export under SUSPENDED.** Same tension — is an organisation entitled
   to export its own compliance record while suspended? Arguably yes.

Everything else in the table follows from the brief's own stated intent that
suspension must stop resource consumption.

### G.4 Choke points

Two, not scattered checks:

1. **Authorization** — already exists (`decision_for_claims` → `_suspend()`).
   Replace the prefix filter with the explicit set. One function.
2. **Runtime admission** — does not exist. The place where a camera session is
   started must consult organisation status, and a transition to
   SUSPENDED/ARCHIVED must *stop already-running* sessions, not merely prevent
   new ones. `ACTIVE → SUSPENDED` leaving orphaned sessions running is the exact
   failure the brief names.

### G.5 Lifecycle metadata — MISSING

`Organization` has `id`, `name`, `slug`, `is_active`, `status`, `created_at`
(`users/models.py:57-70`). Missing: `suspended_at`, `suspension_reason`,
`archived_at`, `created_by`. Without them "why is this organisation suspended"
has no answer in the data.

Also **MISSING**: `is_active` and `status` are two overlapping representations of
liveness. **REQUIRES DECISION** — reconcile them, or document precisely which is
authoritative and why both exist.

Audit actions `organization.created` / `.updated` / `.suspended` / `.archived`
do not exist in `AuditAction` (`domain/audit.py`). **MISSING.**

---

## H. Camera onboarding contract

### H.1 Real field inventory

**PROVEN**, from `Camera` and `to_wire` (`domain/cameras.py:363-390`):

| Field | Runtime-critical? | UI today | Notes |
|---|---|---|---|
| `camera_key` | identity | ✅ | unique per org |
| `name` | no | ✅ | |
| `purpose` | no | ✅ | |
| `restaurant_id` (site) | ownership | ⚠️ **free-text UUID** | must become a populated select |
| `zone_id` | placement | ❌ **absent** | not even in `CameraDraft` |
| `channel` | **yes** | ✅ | |
| `stream_type` | **yes** | ❌ | main/sub |
| `host` | **YES — blocking** | ❌ **absent** | `main.py:365` filters `if row.host …` |
| `rtsp_port` | **yes** | ❌ | |
| `username` | **yes** | ❌ | |
| `credential_ref` | **yes** | ✅ | see §E.4 |
| `analysis_fps` | **yes** | ❌ | |
| `enabled` | **yes** | ❌ | |
| `analysis_enabled` | **YES — blocking** | ❌ | `main.py:365` filters on it |

**The defining defect: the UI collects six fields, omits `host`, and the runtime
filters on `if row.host and row.analysis_enabled`. Every camera created through
the current UI is guaranteed to never stream and never be analysed.** Creation
appears to succeed. Nothing works. **INCORRECT.**

### H.2 Contract

**Required for a runtime-startable camera:** `camera_key`, `restaurant_id`,
`host`, `rtsp_port`, `channel`, `stream_type`, `username`, `credential_ref`,
`analysis_fps`, `enabled=true`, `analysis_enabled=true`.

**Ownership validation — server-side, non-negotiable:** the site must belong to
the caller's organisation (the guard added in Stage 1 for cameras — verify it
covers `zone_id` too, which arrived later), and the zone must belong to that
site. Never trust a client-supplied parent id. Cross-tenant parent → 404, not
403, matching the existing `_restaurant_in_tenant` convention.

**Recommendation: the API should refuse to create a camera that cannot start**,
or mark it explicitly incomplete. A create endpoint whose success means nothing
is worse than a validation error.

### H.3 Connection validation

Backend-owned. The frontend must never open an RTSP connection and must never
receive a credential.

Security constraints (all **BLOCKER**-class if missed):

- **SSRF** — validation dials an operator-supplied host. Must be restricted to
  `MANAGE_CAMERAS` holders, rate-limited per actor, with a bounded timeout, and
  the target constrained by policy (deny loopback/link-local/metadata endpoints
  unless explicitly configured). Without this, the endpoint is an authenticated
  internal port scanner.
- **Credential leakage** — errors must be redacted. `LiveRtspSource._redact`
  already scrubs both raw and URL-encoded forms and is the reference
  implementation; reuse it rather than re-deriving.
- **DoS** — bounded concurrency; validation must not become a way to exhaust
  connection capacity.

Result shape: a discriminated outcome (`reachable` / `auth_failed` /
`no_stream` / `decode_failed` / `timeout`) — honest about *which* stage failed,
carrying no secret.

**REQUIRES DECISION:** whether validation is mandatory before save, or advisory.
Recommendation: advisory — a camera may legitimately be configured before the
DVR is reachable.

### H.4 Lifecycle

`CameraDetailPage` is read-only (its own comment concedes it, :148);
`camerasApi.update` exists with exactly one call site (`setEnabled`); the backend
`DELETE` route is excellent and **unreachable from the UI**. Camera management is
create-only today. Edit, enable/disable, analysis configuration, credential
rotation and retirement all need surfaces (§I).

Retirement, behind `RETIRE_CAMERAS` (D2.3), needs: explicit confirmation naming
the consequence (an observation partition is destroyed), a distinct audit action,
and a UI treatment that does not resemble an ordinary save.

---

## I. Administration UI information architecture

### I.1 Current state — INCORRECT for the target

`/admin` stacks Sites + Zones + Accounts vertically on one page
(`administration.tsx`); `/admin/users/:userId` exists and is correct in shape;
cameras live in an unrelated nav area (`/cameras`, `/cameras/:cameraKey`).
Adding Organizations, Roles and camera onboarding to `/admin` would make the
mega-page problem materially worse.

### I.2 Target

```
Administration
├── Overview            org identity, lifecycle state, counts
├── Sites               list → /admin/sites/:siteId
├── Zones               list → /admin/zones/:zoneId   (site context always shown)
├── Cameras             list → /admin/cameras/:cameraKey
│                       └── Add camera (guided, hierarchical)
├── People & Access     list → /admin/users/:userId   (exists, keep)
└── Audit               filtered administration activity
```

Deliberately **excluded until their backends exist**: an Organizations section
(no CRUD API — §C) and any platform/operator console or org switcher (no
multi-tenant runtime — §F). Building either now would be the "fake platform UI"
the brief forbids.

### I.3 Constraints carried from verified experience

Every one of these is a bug already found and fixed in this codebase during
Stage 7 verification — they are not hypothetical:

- Wide tables must scroll **within their own container**; the grid ancestor needs
  `minmax(0, 1fr)` or `DataTable`'s `overflow-x: auto` is defeated and columns
  are silently clipped.
- `SectionRule`'s `actions` slot is a non-wrapping flex row sized for a small
  link. Substantial content there scrambles the label at narrow widths.
- Reuse `PageIntro` / `SectionRule` / `Region` / `Plane` / `Figure` /
  `DataTable`. Do not introduce a second page-opening or table primitive — the
  Evidence-page and Cameras-page alignment bugs fixed earlier came from exactly
  that drift.
- Permission state must never be carried by colour alone.

### I.4 User & access UI — permissions by domain

The user-detail permission table must group by domain and show all four columns
(role-derived · override · effective), which the current implementation already
does correctly. Under the corrected vocabulary the Sites and Zones groups become
populated for the first time — today they render empty because no site or zone
permission exists to display.

Camera scope (§E.3) needs a surface here too: it is currently invisible in the
UI despite being the thing that decides whether an account sees any cameras.

---

## J. Migration requirements

All additive; single head (`d38dfad216a0`) must remain single.

| Change | Type | Backfill |
|---|---|---|
| New `Permission` enum members | **No migration.** Python enum; `ROLE_PERMISSIONS` is code | none |
| `AccessGrant` for API-created users | Data | **REQUIRES DECISION** (§E.3) — survey first, do not invent scope |
| `literal:` credential rows | Data | Survey; migrate to `env:`/`file:` before removing the scheme |
| `Organization.suspended_at`, `suspension_reason`, `archived_at`, `created_by` | Schema, additive, nullable | none |
| `Organization.is_active` vs `status` reconciliation | **REQUIRES DECISION** | depends on decision |
| New `AuditAction` values | **No migration.** Stored as text — the model comment states this is deliberate | none |
| Runtime compound identity | Possibly none | depends on F.2 decision |

Permissions and roles needing no migration is a real advantage of the existing
design and should be preserved.

---

## K. Security findings

| # | Finding | Severity | Evidence |
|---|---|---|---|
| K-1 | Runtime registries keyed on bare `camera_key`, unique only per org → **cross-tenant frame delivery** once a second org runs cameras | **BLOCKER** | §F.2. Not yet exploitable — no second org can run cameras today. Becomes exploitable the instant multi-tenant bootstrap lands, which is why F.2 must precede F.3 |
| K-2 | `literal:` plaintext credential in DB, in `GET /cameras` to every `VIEW_CAMERAS` holder, and in audit detail | **HIGH** | `cameras.py:352,381`; `audit.py:119-147` |
| K-3 | API-created users get no `AccessGrant` → zero camera visibility, silently | **HIGH** | `user_administration.py`; `resolver.py:47-50` |
| K-4 | Stream ticket tenant binding must be re-verified under compound identity | **HIGH** | `wall.py:153,192`. Correct today under single tenancy; unverified under multi |
| K-5 | Sites/zones writable by anyone holding `MANAGE_ORGANIZATION`; readable by anyone holding `VIEW_USERS` | **MEDIUM** | §D1 |
| K-6 | Suspended org retains `DELETE_EVIDENCE`, `EXPORT_REPORTS`, incident mutation; would retain `RETIRE_CAMERAS` as named | **MEDIUM** | `resolver.py:119-122` |
| K-7 | Suspended/archived org continues streaming and consuming model budget | **MEDIUM** | §G.1 |
| K-8 | Connection validation is an authenticated SSRF vector unless constrained | **MEDIUM** (prospective) | §H.3 — not yet built; constraints must be designed in |
| K-9 | Unpaginated administration lists | **LOW** | §C |
| K-10 | `org_admin` may narrow a `super_admin` | **REQUIRES DECISION** | Carried from Stage 5, still open |

---

## L. Implementation stages

Dependency-ordered. Each stage is independently shippable and leaves the system
working.

### L-1 · Permission vocabulary
Add `VIEW_SITES`, `MANAGE_SITES`, `VIEW_ZONES`, `MANAGE_ZONES`, `RETIRE_CAMERAS`.
Set role baselines per D3. No route changes yet.
**Done when:** every role's effective permission set is unchanged except where D3
says otherwise, proven by test. **Tests:** baseline snapshot per role;
`SUPER_ADMIN` picks up new permissions automatically.

### L-2 · Route re-gating
Seven decorators. `MANAGE_ORGANIZATION` and `VIEW_USERS` re-scoped.
**Done when:** Manager A/B/C from D4 are constructible end-to-end and enforced by
the API. **Tests:** per-route authorization; the D4 scenarios as integration
tests; regression that no previously-permitted call is now refused except
intentionally.

### L-3 · `AccessGrant` provisioning repair
Explicit required scope on create; scope validated against actor; scope editable
and audited.
**Done when:** an API-created user logs in and sees exactly the intended cameras.
**Tests:** the brief's own regression — create via API → log in → correct scope →
correct resources. Plus: cannot grant scope beyond the actor's own.

### L-4 · Credential security
`to_wire` stops returning `credential_ref`; scrubber gains the key and shape;
`literal:` survey then removal (pending decision). Frontend updated in the same
change.
**Done when:** no API response, log line, error or audit row can carry a
credential. **Tests:** serialization test asserting absence; audit scrub test
with a `literal:` value; negative test that `literal:` is refused (if removed).

### L-5 · Cross-tenant ownership hardening
Verify site **and zone** ownership on every camera create/update path; audit all
parent-id acceptance points.
**Tests:** cross-tenant parent → 404, no existence disclosure.

### L-6 · Administration API expansion
Sites/zones detail + edit; camera edit/enable/disable/analysis/retire reachable;
pagination/search/filter as one consistent pattern across all admin lists;
timezone collected at site creation.
**Tests:** authorization per new route; pagination contract.

### L-7 · Runtime tenant identity  ← **must precede L-8**
Compound `(organization_id, camera_key)` through registries, sessions, tickets,
status, metrics.
**Done when:** two organisations can each hold a camera named `cam-01` with no
collision, proven by test.

### L-8 · Multi-tenant bootstrap
Iterate active organisations instead of `default_tenant_id`. Retire the setting
or reduce it to a dev convenience.
**Done when:** two organisations run cameras simultaneously, isolated. **Tests:**
the brief's two-org fixture (Org A: Manager A/B, Site A1, Zone A1-Z1, Camera
A1-Z1-C1; Org B: Manager C, Site B1, Zone B1-Z1, Camera B1-Z1-C1) with sessions
proven not to cross.

### L-9 · Lifecycle → runtime
Explicit `SUSPENDED_FORBIDDEN` set replacing the prefix filter; runtime admission
gate; transitions stop running work. Lifecycle metadata + audit actions.
**Tests:** ACTIVE→SUSPENDED leaves no orphan session; SUSPENDED→ACTIVE restores;
ACTIVE→ARCHIVED stops everything and retains data.

### L-10 · Organization CRUD + platform operator boundary
Separate principal; organisation lifecycle API; platform audit trail.
**Depends on L-8** — do not build before the runtime can serve a second org.

### L-11 · Camera onboarding backend
Full-field create; ownership validation; connection validation with the §H.3
constraints.

### L-12 · Administration IA + frontend
The §I.2 tree. Sites, Zones, Cameras, camera onboarding wizard, camera detail/edit,
People & Access, Audit. Organization UI only if L-10 landed.

### L-13 · Security audit
Re-verify K-1…K-10 against the built system.

### L-14 · Full regression + browser verification
Desktop / tablet / 430px, light + dark, keyboard, focus visibility, table
overflow measured (not eyeballed), permission-denied states, read-only vs manage
user behaviour, cross-org isolation, camera onboarding end to end.

---

## M. Decisions requiring owner approval

| # | Decision | Recommendation |
|---|---|---|
| M-1 | Split user-admin router reads onto `VIEW_USERS`? (D2.4) | Yes — consistency, and fixes suspended-org read |
| M-2 | `literal:` — remove, dev-only, encrypt, or managed refs? (E.4) | Remove, after surveying production |
| M-3 | Backfill `AccessGrant` for existing API-created users, and to what scope? (E.3) | Report them; do not invent scope |
| M-4 | Compound key vs surrogate camera id (F.2) | Compound |
| M-5 | Incident acknowledge/resolve permitted under SUSPENDED? (G.3) | Genuinely ambiguous — no recommendation |
| M-6 | Report export permitted under SUSPENDED? (G.3) | Genuinely ambiguous — no recommendation |
| M-7 | Reconcile `Organization.is_active` with `status` (G.5) | Reconcile; two liveness flags will drift |
| M-8 | Connection validation mandatory before save, or advisory? (H.3) | Advisory |
| M-9 | May an `org_admin` narrow a `super_admin`? | Still open since Stage 5 |

---

## N. Reusable vs must-correct — the summary this plan exists to produce

**Reusable, verified, do not rebuild:** the `PermissionOverride` engine and its
INHERIT/GRANT/REVOKE semantics with REVOKE precedence; `effective_permissions()`;
the request-scoped `AccessDecision` rebuild with no caching; immediate
deactivation enforcement; the three anti-escalation rules; the audit trail and
its scrubber (with the two additions in E.4); tenant isolation at the query
layer; multiple users per role and multiple roles per user; the frontend design
system and the user-detail permission UI shape.

**Must be corrected before any new feature work:** the permission vocabulary
(L-1) and route gating (L-2), because the product requirement is unbuildable
until they are; the `AccessGrant` defect (L-3), because the shipped user-creation
API cannot produce a working account; and the credential leak (L-4), because it
is live.

**Must be corrected before multi-organization work:** runtime tenant identity
(L-7), because bootstrapping a second organisation onto today's registries would
deliver one organisation's video to another.
