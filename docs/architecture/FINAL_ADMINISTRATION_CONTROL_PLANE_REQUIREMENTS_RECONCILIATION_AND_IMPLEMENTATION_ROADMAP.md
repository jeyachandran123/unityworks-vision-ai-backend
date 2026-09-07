# Final Administration Control Plane — Requirements Reconciliation and Implementation Roadmap

**Status:** Discovery / gap analysis / roadmap. **No code, migration, route, permission, UI component or configuration was created, modified or executed to produce this document.** The only file written is this one.

**Repos audited**
- `unityworks-vision-ai-backend` — `c:\Users\Jayachandran\ProjectsAndDocs\atlas\unityworks-vision-ai-backend`
- `unityworks-vision-ai-frontend` — `c:\Users\Jayachandran\ProjectsAndDocs\atlas\unityworks-vision-ai-frontend`

**Label vocabulary — used strictly, never blurred**

| Label | Meaning |
|---|---|
| **PROVEN** | Verified this pass by direct reading of the cited code. |
| **IMPLEMENTED** | Exists, is reachable end-to-end by a real user, and does what was asked. |
| **PARTIAL** | Exists and works, but covers only a slice of the stated requirement. |
| **MISSING** | Does not exist anywhere in either repo. |
| **INCORRECT** | Exists but is wrong — wrong gate, wrong semantics, or unable to express the requirement. |
| **NOT VERIFIED** | Not established this pass; stated as unknown rather than assumed. |
| **REQUIRES DECISION** | A product/security decision the owner must make; not an engineering gap. |

**Method note.** Prior reports in `docs/architecture/` were skimmed for context and are **the thing under audit, not the source of truth**. Every claim below carrying PROVEN was re-read from source this pass. Where a prior report's claim was not re-verified, it is labelled NOT VERIFIED and said so.

---

## 1. Executive summary

Stages 1–8 built a **correct mechanism on an incomplete vocabulary, exposed through a thin UI, inside a single-tenant runtime**. That sentence is the whole finding.

What was actually delivered and works:

- A three-state INHERIT / GRANT / REVOKE permission-override model with REVOKE-wins composition, recomputed from the database on every request (`app/authorization/resolver.py:125-171`, `app/authorization/model.py:415-427`). **IMPLEMENTED and correct.**
- An eleven-route `MANAGE_USERS`-gated user-administration API with tenant narrowing, anti-escalation rules and audit rows (`app/api/user_administration.py`). **IMPLEMENTED**, with one defect (§7.4).
- A per-user detail page rendering Inherited / + Added / − Restricted (`src/features/user-detail.tsx`). **IMPLEMENTED.**

What was **not** delivered, and what the prior stages' completeness claims obscured:

1. **The headline requirement is not expressible.** "Manager A: Sites Read+Edit; Manager B, same role: Sites Read Only" cannot be written down, because `Permission` contains no `VIEW_SITES`, no `MANAGE_SITES`, no `VIEW_ZONES`, no `MANAGE_ZONES` (`app/authorization/model.py:69-186`). The override machinery is a correct answer applied to an alphabet with no letters for the question. **INCORRECT / MISSING** — not partial.
2. **Sites and zones are gated on unrelated permissions.** Listing restaurants requires `VIEW_USERS`; creating or editing one requires `MANAGE_ORGANIZATION` — the same blanket permission that governs users and everything else (`app/api/administration.py:126, 160, 196-198, 276, 315, 344`). **INCORRECT.**
3. **There is no Organization CRUD anywhere.** No route in either repo creates, renames, suspends or archives an organization. A second tenant can be created today **only** as a side effect of `python scripts/manage.py create-user --org <new-id>` (`scripts/manage.py:100-110`). **MISSING.**
4. **The perception and streaming runtime is hard-wired to one tenant.** `cfg.default_tenant_id` is the organization for camera bootstrap and for every compliance-driver query (`app/main.py:360`, `app/vision/compliance_driver.py:195, 213, 235, 382, 404, 484, 522`, `app/vision/manager.py:238`). A second organization's cameras would never start. **MISSING** — this makes "multi-organization" aspirational rather than real.
5. **Organization lifecycle does not reach cameras, analysis or streaming.** `SUSPENDED` only strips `manage_*` from a live request (`app/authorization/resolver.py:119-122`); `ARCHIVED` only refuses login and API calls (`app/auth/service.py:82, 146`). Neither is read by the camera bootstrap or the compliance driver. A suspended tenant keeps streaming, decoding and spending model budget. **MISSING.**
6. **Camera onboarding is a six-field modal that cannot configure a camera.** The registration form collects `camera_key`, `name`, `channel`, `restaurant_id`, `credential_ref`, `purpose` (`src/features/persistence-routes.tsx:347-410`). It omits `host` — **without which no stream can ever open** — plus `rtsp_port`, `stream_type`, `username`, `analysis_fps` and `zone_id`. There is no camera **edit** surface at all. **PARTIAL, bordering on non-functional.**
7. **A camera credential can be a plaintext password, and it is returned in API responses and written to the audit trail.** `_validate` accepts `literal:` (`app/domain/cameras.py:352`), and `to_wire` returns `credential_ref` verbatim (`app/domain/cameras.py:381`). §15 states the exact conditions. **INCORRECT.**
8. **A user created through the shipped admin API cannot see a single camera.** `POST /api/v1/admin/users` creates the `User` and its `RoleAssignment` rows and **never creates an `AccessGrant`** (`app/api/user_administration.py:326-345`). With no grant, `parse_camera_scope(None)` returns `CameraScope.none()` (`app/authorization/resolver.py:47-50`), `scope_cameras()` returns `()` (`app/api/product.py:45-49`), and every camera-backed surface returns an empty list — silently. **INCORRECT**, and in practical terms the single most damaging defect of Stage 5.

**Readiness verdict, stated up front:** the control plane is **NOT READY**. It is a defensible identity-and-permission slice with a working override engine, sitting on a permission vocabulary that cannot express the requirement that motivated it, inside a runtime that is single-tenant by construction, fronted by a camera onboarding form that cannot produce a working camera. See §30.

---

## 2. What the originally requested product actually requires

Reconstructed from the user's stated requirements, expressed as testable capabilities.

**R1 — Multi-organization.** More than one customer organization coexists. Each is created, named, and lifecycle-managed through the product, not the shell. Data, cameras, users and audit are isolated per organization.

**R2 — Organization lifecycle with a defined blast radius.** `ACTIVE` / `SUSPENDED` / `ARCHIVED`, with an explicit, documented answer to: *what happens to cameras, to analysis, to streaming, to background work* in each state.

**R3 — Platform operator boundary.** Somebody outside any one tenant can list organizations, create them, suspend them, and provision their first administrator — without becoming a member of every tenant.

**R4 — First-admin provisioning over HTTP.** A new organization gets a working administrator account through an authenticated, audited API call.

**R5 — Multiple users per role; multiple roles per user.** Both directions, without role explosion.

**R6 — Per-user feature access differing within the same role.** *The headline.* Two `restaurant_manager` accounts where one may create and edit sites and the other may only read them.

**R7 — Three-state override semantics.** INHERIT (no row) / GRANT / REVOKE, with REVOKE winning over both role and GRANT, applied on the very next request.

**R8 — A permission vocabulary with a read/manage pair per administered domain.** Sites, zones, cameras, users, organizations — each with its own view and its own manage permission, so R6 can be expressed for any of them.

**R9 — Site administration.** Create, read, edit, deactivate; timezone; per-site camera and zone counts.

**R10 — Zone administration.** Create, read, edit, move, retire; assignment of cameras to zones with history preserved.

**R11 — Camera / CCTV administration.** Full configuration lifecycle: transport (host, port, channel, stream type, credentials), placement (site, zone), analysis policy (fps, analysis on/off), operational state (enabled), retirement.

**R12 — CCTV onboarding a real operator can complete.** From "I have a DVR on the wall" to "the camera is streaming and analysed" without a shell, a raw UUID, or a database client.

**R13 — RTSP secret safety.** No password in a list response, a detail response, a log line, an error message, an audit row, or the UI. A stated update path for rotating a credential.

**R14 — An administration information architecture that scales.** No single everything-admin page. Deep-linkable object addresses. Permission-aware navigation. Scales to many organizations, many sites, many cameras.

**R15 — Audit.** Every administrative act recorded with actor, tenant, resource, request id, outcome.

**R16 — Verification.** Automated tests plus real browser verification of the reachable flows.

---

## 3. Current implementation

### 3.1 Backend — administration surface, complete route inventory

`app/api/administration.py` — prefix `/api/v1`:

| Route | Line | Gate | Verdict |
|---|---|---|---|
| `GET /restaurants` | 126 | `VIEW_USERS` | **INCORRECT** gate |
| `POST /restaurants` | 160 | `MANAGE_ORGANIZATION` | **INCORRECT** gate |
| `PATCH /restaurants/{restaurant_id}` | 196-198 | `MANAGE_ORGANIZATION` | **INCORRECT** gate |
| `GET /zones` | 276 | `VIEW_USERS` | **INCORRECT** gate |
| `POST /zones` | 315 | `MANAGE_ORGANIZATION` | **INCORRECT** gate |
| `PATCH /zones/{zone_id}` | 344 | `MANAGE_ORGANIZATION` | **INCORRECT** gate |
| `GET /users` | 382 | `VIEW_USERS` | **IMPLEMENTED**, correct |

There is **no** `DELETE` for a restaurant or a zone, and no site/zone deactivation route beyond `PATCH … is_active` on a restaurant (`app/api/administration.py:216-218`). Zones have no `is_active` column at all (`app/domain/models.py:79-100`).

`app/api/user_administration.py` — prefix `/api/v1/admin/users`, router-level `MANAGE_USERS` dependency (`:100-105`):

| Route | Line | Verdict |
|---|---|---|
| `GET ""` | 257 | IMPLEMENTED |
| `GET /{user_id}` | 275 | IMPLEMENTED |
| `POST ""` | 281 | **INCORRECT** — no `AccessGrant` created (§7.4) |
| `PATCH /{user_id}` | 365 | IMPLEMENTED (display name only) |
| `POST /{user_id}/activate` | 407 | IMPLEMENTED |
| `POST /{user_id}/deactivate` | 429 | IMPLEMENTED |
| `POST /{user_id}/roles` | 467 | IMPLEMENTED |
| `DELETE /{user_id}/roles/{role_value}` | 507 | IMPLEMENTED |
| `GET /{user_id}/permissions` | 551 | IMPLEMENTED |
| `PUT /{user_id}/permissions/{permission_value}` | 562 | IMPLEMENTED |
| `DELETE /{user_id}/permissions/{permission_value}` | 629 | IMPLEMENTED |

`app/api/product.py` — camera routes:

| Route | Line | Gate |
|---|---|---|
| `GET /cameras` | 63 | `VIEW_CAMERAS` |
| `POST /cameras` | 76 | `MANAGE_CAMERAS` |
| `PATCH /cameras/{camera_key}` | 127 | `MANAGE_CAMERAS` |
| `DELETE /cameras/{camera_key}` | 177 | `MANAGE_CAMERAS` |

Cameras are the **only** administered domain with a correct read/manage permission pair. **PROVEN** (`app/authorization/model.py:94-95`).

**No organization routes exist in either repo.** PROVEN by exhaustive route enumeration across `app/api/*.py` — the complete decorator set was read this pass and contains no organization or tenant CRUD.

### 3.2 Backend — authorization core

- `Permission` enum, 33 members, `app/authorization/model.py:69-186`. **No site permission. No zone permission.** PROVEN.
- `ROLE_PERMISSIONS`, `:200-366` — flat, explicit, no inheritance.
- `effective_permissions()`, `:415-427` — `(role ∪ GRANT) − REVOKE`, REVOKE last and unconditional.
- `OverrideState`, `:373-392` — GRANT / REVOKE only; INHERIT is the absence of a row.
- `OrganizationStatus`, `:397-409` — ACTIVE / SUSPENDED / ARCHIVED.
- `AccessDecision`, `:479-541` — `tenant_id: str`, single, mandatory, validated non-empty at `:507-511`, never read from request input.
- `decide()`, `app/authorization/resolver.py:125-171` — recomputes roles, overrides and lifecycle from freshly loaded rows on every call.
- `decision_for_claims()`, `app/auth/service.py:126-153` — rebuilds from the database on **every request**, explicitly not from token claims. This is what makes revocation immediate.
- `set_permission_override` / `clear_permission_override`, `app/authorization/overrides.py:46-97`, with structural `_guard()` at `:102-117` refusing self-modification and cross-tenant writes.

### 3.3 Frontend — routes and navigation

`src/app/router/AppRouter.tsx`:

| Path | Gate | Surface |
|---|---|---|
| `/admin` | any-of `manage_users`, `manage_organization` | `AdministrationPage` — Sites + Zones + Accounts stacked |
| `/admin/users/:userId` | `manage_users` | `UserDetailPage` |
| `/cameras` | any-of `view_cameras`, `view_camera_health` | `CamerasPage` |
| `/cameras/:cameraKey` | same | `CameraDetailPage` (read-only) |

`src/app/router/navigation.ts:102ff` — five sections: Operations, Compliance, Intelligence, **Platform** (Cameras, Integrations, Administration, Patron ID), Engineering. Cameras and Administration are **siblings under Platform**, and Administration's hint reads "Restaurants, zones, users and roles" — cameras are explicitly outside the administration idea, which is precisely the IA problem in §10.

There is **no organization surface anywhere in the frontend.** PROVEN — a repo-wide search for "organization" in `src/` returns only permission-string constants, comment text and API response types. `/auth/me` returns `tenant_id` but no organization name and no lifecycle status (`app/api/routes.py:259-271`), so the UI cannot display which tenant you are in or that it is suspended.

---

## 4. What the current migration actually provides

Migration `d38dfad216a0` — `migrations/versions/20260904_d38dfad216a0_permission_overrides_and_organization_.py`. Two additive changes:

1. `permission_overrides` table: `id`, `user_id` (FK → `users.id`, `ondelete=CASCADE`), `permission`, `state`, `granted_at`, `granted_by`, plus `uq_permission_override_user_permission` on `(user_id, permission)`.
2. `organizations.status` — `String(32)`, `nullable=False`, `server_default='active'`.

**What it provides:** exactly one durable statement per `(user, permission)` pair, and one durable lifecycle string per organization. Nothing more. **PROVEN.**

**What it does not provide — and each is load-bearing:**

- No site or zone permission rows, because permissions are a code enum, not data. Adding `VIEW_SITES` is a **code change**, not a migration.
- No `organizations` CRUD, no organization audit trail, no `created_by`, no `suspended_at`, no `suspension_reason`.
- No scope override. `permission_overrides` says *whether* a user may manage cameras; `AccessGrant` says *which* cameras. The two are unrelated tables and only the first got an API.
- No `zones.is_active`. A zone can be created and renamed, never retired.
- No camera credential versioning, no `credential_rotated_at`.

The migration is **correct for what it claims** and **materially smaller than the requested feature**.

---

## 5. Why `permission_overrides` alone is not the complete feature

The override table answers: *may this specific user do X, notwithstanding their role?* That is the right question. It is one of five the requested product asks, and it is the only one that got built.

**The chain that must hold for any administered capability**

```
1. permission vocabulary   → is there a name for this capability?
2. API gating              → does the route actually check that name?
3. tenant ownership check  → is the object provably in the caller's org?
4. frontend surface        → is there a control that exercises it?
5. reachable flow          → can a real user complete it end to end?
```

Applied honestly:

| Domain | 1. Vocabulary | 2. Gate | 3. Tenant check | 4. Surface | 5. Reachable |
|---|---|---|---|---|---|
| Users | ✅ `view_users`/`manage_users` | ✅ | ✅ `_user_in_tenant` (`user_administration.py:118-140`) | ✅ `/admin` + `/admin/users/:id` | ✅ |
| Sites | ❌ **none** | ❌ borrowed | ✅ `_restaurant_in_tenant` (`administration.py:242`) | ⚠️ create only | ⚠️ create only |
| Zones | ❌ **none** | ❌ borrowed | ✅ | ⚠️ create only | ⚠️ create only |
| Cameras | ✅ `view_cameras`/`manage_cameras` | ✅ | ✅ (`product.py:88`) | ⚠️ 6-field create + enable toggle | ❌ cannot produce a working camera |
| Organizations | ❌ none | ❌ no route | n/a | ❌ none | ❌ |

**Three rows out of five fail at step 1.** No amount of override machinery repairs that: an override is a statement *about a permission*, and there is no permission to make a statement about.

**The concrete failure of R6.** Manager B is `restaurant_manager`. To make them read-only over sites you would REVOKE the site-edit permission. There is no site-edit permission. The nearest available lever is `MANAGE_ORGANIZATION`, which Manager A does not hold either (`ROLE_PERMISSIONS[RESTAURANT_MANAGER]`, `model.py:270-296`) — so **neither** manager can edit a site today, and the difference the user asked for cannot be created in either direction. Classified **INCORRECT / MISSING**, and stated plainly: *the feature the eight stages existed to deliver cannot be demonstrated.*

---

## 6. Multi-organization gap analysis

### 6.1 Is there any Organization CRUD API?

**MISSING. PROVEN.** Every `@router.*` decorator across `app/api/` was enumerated this pass. There is no organization or tenant create, read, update, delete, suspend or archive route. The `Organization` model (`app/users/models.py:47-72`) is written to by exactly one place outside SQLAlchemy defaults: `scripts/manage.py`.

### 6.2 Can a second organization be created at all today?

**Yes — by one path, and it is a side effect.** `scripts/manage.py create-user --org <new-org-id>` (`scripts/manage.py:100-110`):

```python
org = (await session.execute(select(Organization).where(Organization.id == args.org))).scalar_one_or_none()
if org is None:
    org = Organization(id=args.org, name=DEFAULT_ORG_NAME if args.org == DEFAULT_ORG_ID else args.org, slug=args.org)
    session.add(org)
    print(f"created organization {args.org}")
```

Properties of this path, all PROVEN:
- Shell access to the server is required. There is no HTTP path.
- `name` and `slug` both default to the raw id string — a tenant literally named `acme-2`.
- The act is **unaudited**. No `AuditEvent` row is written; there is no `ORGANIZATION_CREATED` value in `AuditAction` (`app/domain/audit.py:36-110`).
- `status` takes the column default `'active'`.
- No `Restaurant` is created, so the new tenant has zero sites until its first admin calls `POST /restaurants`.
- The command's *primary* purpose is creating a user; organization creation is an implicit fallback. Typo the `--org` flag and you have silently created a tenant.

Classified **MISSING** for the product requirement, **PARTIAL** as an operational escape hatch.

### 6.3 Platform-operator boundary

**MISSING. PROVEN.** `AccessDecision.tenant_id` is a single, mandatory, non-empty field (`app/authorization/model.py:494`, validated `:507-511`). Every derived platform object carries exactly one tenant: `to_principal()` (`:527-536`), `to_scope()` (`:538-561`), `to_grant()` (`:563-598`). `decide()` sets it from `user.organization_id` (`resolver.py:164`).

**What that structurally forbids:** a principal cannot name two tenants, so no caller can be cross-tenant without either (a) changing `AccessDecision`'s shape — which touches every tenant-scoped query in the application — or (b) introducing a *separate* platform-operator concept that never becomes an `AccessDecision`. `Role.SUPER_ADMIN` is tenant-scoped like every other role; it is "everything within one customer", not "above all customers".

This is a **good property being mistaken for a gap**. The isolation guarantee is exactly what makes it safe. The correct fix is (b), not (a). See §17.

### 6.4 First-admin provisioning for a new organization

**MISSING over HTTP. PARTIAL via CLI. PROVEN.**

The only mechanism is `scripts/manage.py create-user` (`:88-124`), which:
- hashes the password through the same `hash_password` the login path verifies against (`:118`);
- assigns exactly one `RoleAssignment` (`:121`);
- creates an `AccessGrant` via `_grant_for()` (`:122`, `:205-232`) — **note: the CLI does create a grant; the HTTP API does not.** See §7.4.

`POST /api/v1/admin/users` cannot bootstrap a new organization: it is gated on `MANAGE_USERS` and writes `organization_id=access.tenant_id` (`user_administration.py:326`), so it can only create users **inside the caller's existing tenant**. There is no chicken-and-egg escape.

### 6.5 The runtime is single-tenant — the finding that makes this section decisive

**MISSING. PROVEN.**

```
app/main.py:360                    enabled_for_runtime(organization_id=cfg.default_tenant_id)
app/main.py:497, 499, 519          same
app/vision/compliance_driver.py:195, 213, 235, 382, 404, 484, 522   default_tenant_id
app/vision/manager.py:238          default_tenant_id
app/configuration/settings.py:281  default_tenant_id: str = "default"
```

`_start_cameras_from_database` (`app/main.py:338-397`) reads **one organization's** enabled cameras and starts them. The compliance driver writes incidents, reads cameras and constructs `TenantId` from the same single setting.

**Consequence:** create a second organization by any means, register its cameras, enable them — and nothing starts. No stream, no perception, no incident. The API surface is multi-tenant; the runtime is not. Any roadmap that adds organization CRUD without addressing this ships a tenant that can be administered and cannot be used.

---

## 7. Identity gap analysis

### 7.1 What is correctly built — cited, so the balance is visible

| Capability | Verdict | Evidence |
|---|---|---|
| Multiple users per role | **IMPLEMENTED** | `RoleAssignment` has no uniqueness on `role`; `permissions_for` unions across holders (`model.py:368-373`) |
| Multiple roles per user | **IMPLEMENTED** | `parse_roles` over all `role_assignments` (`resolver.py:31-44`); union semantics |
| Activate / deactivate | **IMPLEMENTED** | `user_administration.py:407-465`; `decide()` short-circuits an inactive user to no roles and no cameras (`resolver.py:136-146`) |
| Immediate session invalidation | **IMPLEMENTED** | `decision_for_claims` rebuilds from the DB every request and refuses an inactive account (`auth/service.py:126-153`). A revoked role or deactivated account takes effect on the next request, not at token expiry. |
| Self-modification refused | **IMPLEMENTED** | `_is_self` at the route (`user_administration.py:162-163`) **and** structurally in `_guard()` (`overrides.py:102-107`) — defence in depth, not one check |
| Cross-tenant refused | **IMPLEMENTED** | `_user_in_tenant` returns 404 not 403 (`user_administration.py:118-140`); `_guard()` compares `organization_id` (`overrides.py:108-117`) |
| Anti-escalation on GRANT | **IMPLEMENTED** | `access.has(permission)` required before GRANT |
| Anti-escalation on role assign | **IMPLEMENTED** | `_require_grantable_role` — `permissions_for({role}) <= access.permissions` (`user_administration.py:189-201`). Correctly blocks `org_admin` → `super_admin`/`developer` while permitting `org_admin` → `restaurant_manager`. |
| Uniform login failure | **IMPLEMENTED** | dummy-hash verify on unknown email (`auth/service.py:39, 67-69`) |
| Audit of user administration | **IMPLEMENTED** | 9 `AuditAction` values, `audit.py:101-110` |
| Password never returned | **IMPLEMENTED** | `_user_to_wire` (`user_administration.py:203-215`); generated password returned exactly once at creation (`:356-361`) |

This is real, careful work and should be recognised as such.

### 7.2 Password reset for an existing user — **MISSING**

`PATCH /admin/users/{user_id}` accepts `display_name` only (`user_administration.py:365-405`). An administrator whose user has forgotten their password has no HTTP path; `scripts/manage.py reset-password` is the only mechanism. There is no `USER_PASSWORD_RESET` audit action.

### 7.3 Email change, deletion, invitation, SSO — **MISSING**

No route changes a user's email. No route deletes a user (deactivate is the only exit). No invitation flow. No SSO. The freeze document §28.4 flags the provisioning mechanism as REQUIRES DECISION; that decision was never taken and Stage 5 shipped the admin-set-password half without it.

### 7.4 **The `AccessGrant` defect — INCORRECT, and the most damaging one found**

`POST /api/v1/admin/users` (`app/api/user_administration.py:281-362`) creates the `User` (`:326-333`) and its `RoleAssignment` rows (`:337-338`). **It never creates an `AccessGrant`.** A repo-wide search for `AccessGrant` in `app/` returns four files — `resolver.py`, `users/models.py` and nothing under `app/api/`. **PROVEN.**

The consequence chain, every link read this pass:

```
no AccessGrant row
  → decide(): grants = [] → effective = None            resolver.py:166-168
  → parse_camera_scope(None) → CameraScope.none()       resolver.py:47-50
  → AccessDecision.cameras.breadth = NONE
  → scope_cameras(access) returns ()                    product.py:45-49
  → CameraService.list(camera_keys=()) returns []       domain/cameras.py:288-290
```

So for **every account created through the shipped administration API**:
- `GET /api/v1/cameras` returns `{"cameras": [], "enabled": 0, "total": 0}`.
- `GET /api/v1/wall/cameras` returns an empty wall (`app/api/wall.py:100-149`) — **no error, just nothing**.
- `AccessDecision.to_scope()` and `.to_grant()` raise `ScopeError` (`model.py:548-553`, `:573-580`) on any path that builds a platform scope.

The user detail page exposes no camera-scope control either (`src/features/user-detail.tsx` — roles and overrides only), so an administrator cannot repair it from the UI. The **only** fix today is `python scripts/manage.py grant --email … --cameras all`.

Stage 5–8 reports describe user creation as working. It creates a user who can log in and see an empty product. That is the gap between "the route returns 200" and "the requirement is met", and it is exactly the over-claiming the user pushed back on.

---

## 8. Role and feature-access gap analysis

### 8.1 The vocabulary, audited domain by domain

`Permission`, `app/authorization/model.py:69-186` — 33 members:

| Domain | View | Manage | Verdict |
|---|---|---|---|
| Organization | — | `MANAGE_ORGANIZATION` (:82) | **INCORRECT** — a manage with no matching view, doing double duty as the sites/zones gate |
| Users | `VIEW_USERS` (:84) | `MANAGE_USERS` (:83) | ✅ correct pair |
| **Sites** | **—** | **—** | **MISSING** |
| **Zones** | **—** | **—** | **MISSING** |
| Cameras | `VIEW_CAMERAS` (:95) | `MANAGE_CAMERAS` (:94) | ✅ correct pair — the only administered domain done right |
| Camera health | `VIEW_CAMERA_HEALTH` (:88) | — | acceptable (read-only domain) |
| Live | `VIEW_LIVE` (:86) | — | acceptable |
| Observations / Evidence | `VIEW_OBSERVATIONS`, `VIEW_EVIDENCE`, `DELETE_EVIDENCE` | | correct |
| Incidents | `VIEW_INCIDENTS`, `ACKNOWLEDGE_INCIDENTS`, `RESOLVE_INCIDENTS` | | correct |
| Audit | `VIEW_AUDIT` | — | correct |
| Reports | `VIEW_REPORTS`, `EXPORT_REPORTS` | — | correct |
| Modules | `VIEW_*` × 6, `MANAGE_TABLE_OCCUPANCY`, `MANAGE_CUTTING_BOARD`, `MANAGE_PATRON_ID`, `MANAGE_POS_INTEGRATION` | | correct |
| Engineering | `ACCESS_DEVTOOLS`, `REGISTER_DEMAND`, `VIEW_MODEL_EVALUATION` | | correct |

The enum's own docstring states the discipline it broke: *"no permission exists without something to protect"* (`:76-80`). The inverse held silently — **things exist with no permission to protect them**, and they were gated on the nearest available name instead.

### 8.2 `MANAGE_ORGANIZATION` is doing three unrelated jobs

Today, holding `MANAGE_ORGANIZATION` means: create/edit sites, create/edit zones, and — by `ROLE_PERMISSIONS[ORG_ADMIN]` (`:214-217`) — sit alongside `MANAGE_USERS`. There is no way to grant site editing without granting the permission whose name says "manage the entire organization". **INCORRECT.**

### 8.3 The read gate is the more alarming half

`GET /restaurants` and `GET /zones` require `VIEW_USERS` (`administration.py:126, 276`). The docstring at `:128-134` defends this on the grounds that a restaurant manager needs to read the structure — true — but it reaches that outcome by making *"may list the estate's sites"* identical to *"may enumerate every account in the organization."* A user who should see the site list but not the staff roster cannot be expressed. **INCORRECT.**

### 8.4 Role coverage

`ROLE_PERMISSIONS` (`:200-366`) is well-reasoned and its comments are unusually honest (KITCHEN_SUPERVISOR without `VIEW_EVIDENCE`; AUDITOR without `VIEW_LIVE`; SUPER_ADMIN explicitly minus `MANAGE_PATRON_ID`). **IMPLEMENTED.** The gap is not the mapping; it is the alphabet being mapped.

---

## 9. INHERIT / GRANT / REVOKE status

**IMPLEMENTED and correct. PROVEN.** This is the strongest part of the delivery.

| Property | Verdict | Evidence |
|---|---|---|
| Three states, two stored | ✅ | `OverrideState` GRANT/REVOKE only; absence = INHERIT (`model.py:373-392`) |
| REVOKE wins over role | ✅ | `(roles ∪ granted) − revoked`, subtraction last (`model.py:415-427`) |
| REVOKE wins over GRANT | ✅ | same expression; the unique constraint makes both states on one pair impossible anyway |
| At most one row per (user, permission) | ✅ | `uq_permission_override_user_permission`, migration `d38dfad216a0` |
| Idempotent writes | ✅ | `set_permission_override` updates in place (`overrides.py:66-79`) |
| Applied on next request | ✅ | `decision_for_claims` → `decide()` from fresh rows (`auth/service.py:126-153`) |
| Unknown values deny | ✅ | `parse_overrides` drops unreadable permission/state rather than guessing (`resolver.py:70-99`) |
| Zero-permission user survives | ✅ | `AccessDecision.permissions: frozenset \| None` — an explicit empty set is *not* re-derived from roles (`model.py:496-500`, `:512-513`). Subtle and correct. |
| Self-override refused | ✅ | `overrides.py:102-107` |
| Cross-tenant override refused | ✅ | `overrides.py:108-117` |
| Audited | ✅ | `PERMISSION_GRANTED` / `PERMISSION_REVOKED` / `PERMISSION_RESET` (`audit.py:108-110`) |
| Surfaced in UI | ✅ | Inherited / + Added / − Restricted (`src/features/user-detail.tsx:77-79`) |
| Tested | ✅ | 31 tests in `tests/app/test_permission_overrides.py`, 39 in `tests/app/test_user_administration.py` |

**The one open item:** whether a `MANAGE_USERS` holder may REVOKE a permission from a target who holds it via a role the actor cannot reach. Documented as deliberately unimplemented at `user_administration.py:63-79`, and carried as REQUIRES DECISION in the freeze doc's Decision Table. **REQUIRES DECISION**, still open. See §29.1.

**Verdict:** the engine is right. §5 is why the engine alone is not the feature.

---

## 10. Administration UI IA gap analysis

### 10.1 Current state, PROVEN

`/admin` → `AdministrationPage` (`src/features/administration.tsx`, 632 lines) renders, on one scrolling page:

- `SectionRule order={2}` — an organization summary strip.
- `SectionRule order={3}` — **Sites**: a `DataTable`, plus an inline "Add a site" form taking **`name` only** (`:97-103`, `:305-341`).
- `SectionRule order={4}` — **Zones**: a `DataTable`, plus "Add a zone" taking site-select + name (`:105-112`, `:344-463`).
- `SectionRule order={5}` — **Accounts**: a `DataTable` linking to `/admin/users/:userId`, plus "Add account" with email, display name, password, role checkboxes (`:114-131`, `:466-629`).

`/admin/users/:userId` → `UserDetailPage` (618 lines) — the one properly-shaped object page in the administration area: identity, activate/deactivate with confirmation, role add/remove, per-permission override table.

Cameras live in a *different navigation section*: Platform → Cameras → `/cameras`, `/cameras/:cameraKey` (`navigation.ts:~253-270`).

### 10.2 Assessed against R14

| Requirement | Verdict | Why |
|---|---|---|
| No giant everything-admin page | **INCORRECT** | `/admin` is precisely that: three unrelated domains stacked with three inline create forms. |
| Scales to many organizations | **MISSING** | No organization list, no organization object page, no tenant switcher. `/auth/me` does not even return the org's name. |
| Scales to many sites | **PARTIAL** | One unpaginated `DataTable`, no search, no filter, no per-site object page. A site is a row you cannot open. |
| Scales to many cameras | **PARTIAL** | `/cameras` is a flat unpaginated list across the whole tenant; no grouping by site, zone or DVR host. |
| Deep-linkable | **PARTIAL** | Only users and cameras have addresses. `/admin/sites/:id` and `/admin/zones/:id` do not exist. A site cannot be linked from an incident, a report or an email. |
| Permission-aware | **PARTIAL** | `RequirePermission` and `PermissionGate` are used correctly, but `/admin`'s gate is any-of `manage_users` **or** `manage_organization` while the page unconditionally issues `adminUsersApi.list` — a `manage_organization`-only holder reaches the page and the Accounts query 403s. Reachable today via a REVOKE. |
| Cameras belong to administration | **INCORRECT** | Cameras are the most administrative object in the product and sit in a different nav section from Administration, whose own hint text ("Restaurants, zones, users and roles") excludes them. |

### 10.3 Target IA, grounded in the existing design system

The vocabulary already in `src/shared/ui/product.tsx` — `PageIntro` (:107), `SectionRule` (:279), `Region` (:223), `Plane` (:251), `Figure` (:524), `Meter` (:1186), `CameraSurface` (:688), `CameraLine` (:904), `Attention` (:376), `AbsentRegion` (:1121), `GoTo` (:1773), `Disclosure` (:1684) — plus `DataTable` (`primitives.tsx:708`), `Modal` (:852), `Drawer` (:924), `Tabs` (:787), `KeyValue` (:973). The proposal below uses **only** these. No new component family, no generic admin-dashboard chrome, no card grid, no sidebar-within-a-page.

**The structural rule:** the existing product already distinguishes a *ledger* from an *object page* — `/incidents` vs `/incidents/:id`, `/cameras` vs `/cameras/:cameraKey`. Administration currently has that shape for users alone. Extend the pattern it already has rather than inventing one.

```
Administration                                   (nav section, replaces "Platform" for admin objects)
├── /admin                     Overview     PageIntro + Figure row (sites, zones, cameras, accounts,
│                                           enabled cameras, analysed cameras) + Attention for anything
│                                           unhealthy. SectionRule per domain, each ending in a GoTo.
│                                           No create forms. This page answers "what is the estate",
│                                           and every row is a way in.
├── /admin/sites               Ledger       DataTable: name, timezone, zones, cameras, active.
│   └── /admin/sites/:siteId   Object       PageIntro (site) + Tabs: Details (KeyValue + edit Drawer)
│                                           · Zones (DataTable scoped to this site)
│                                           · Cameras (CameraLine list, scoped)
│                                           · Activity (audit rows for this resource)
├── /admin/zones/:zoneId       Object       Reached from a site. Rename, retire, and the camera
│                                           assignment history that CameraZoneAssignment already stores
│                                           — rendered as a Timeline, which is exactly what that table
│                                           was built to make renderable.
├── /admin/cameras             Ledger       Moves here from /cameras. Grouped by site, then by host
│                                           (the informal DVR grouping) using SectionRule per group.
│                                           Meter carries enabled/analysed proportion over the real
│                                           denominator, the pattern CamerasPage already uses well.
│   └── /admin/cameras/:key    Object       Tabs: Transport (host/port/channel/stream/username/
│                                           credential_ref) · Placement (site, zone, purpose)
│                                           · Analysis (analysis_fps, analysis_enabled)
│                                           · Live state (existing read-only content, kept)
│                                           · Activity. Each tab a Drawer to edit; each save one
│                                           PATCH and one audit row.
├── /admin/accounts            Ledger       The existing Accounts table, alone on its own page.
│   └── /admin/accounts/:id    Object       UserDetailPage as it stands today — unchanged — plus a
│                                           Camera scope section (§7.4's missing control).
└── /admin/organization        Object       This tenant: name, status, counts, lifecycle banner.
                                            Uses AbsentRegion when a capability is off rather than
                                            hiding it, which is this design system's existing and
                                            correct posture about absence.

Platform Operator                            (separate nav section, only for the platform-operator
├── /operator/organizations     Ledger        boundary of §17; invisible to every tenant user)
└── /operator/organizations/:id Object
```

**Onboarding, using the existing `Tabs` + `Drawer` rather than a new wizard component:** `/admin/cameras/new` as a `Drawer` with three `Tabs` — Transport, Placement, Analysis — each a small form, with the "Register, disabled" affordance and its existing copy preserved (`persistence-routes.tsx:387-390`), because "created disabled, enabling is a separate audited act" is a genuinely good property and its wording is already right.

---

## 11. Site administration gap analysis

**Real schema.** `Restaurant`, `app/domain/models.py:55-77`: `id`, `organization_id` (FK CASCADE), `name`, `slug`, `timezone` (default `"UTC"`), `is_active`, `created_at`. Unique `(organization_id, slug)`.

| Capability | Verdict | Evidence |
|---|---|---|
| Read permission | **MISSING** | no `VIEW_SITES` in `Permission` |
| Manage permission | **MISSING** | no `MANAGE_SITES` |
| List API | **PARTIAL/INCORRECT** | `GET /restaurants` exists, correctly tenant-scoped, gated on `VIEW_USERS` (`administration.py:126`) |
| Create API | **PARTIAL/INCORRECT** | `POST /restaurants` (:160) — `organization_id` from `access.tenant_id`, never the body (:172); gated `MANAGE_ORGANIZATION` |
| Update API | **PARTIAL/INCORRECT** | `PATCH /restaurants/{id}` (:196-198) — name, timezone, is_active |
| Delete / archive API | **MISSING** | none; `is_active=false` is the only exit |
| Tenant ownership check | **IMPLEMENTED** | `_restaurant_in_tenant` (:242-264), reused by camera create (`product.py:88`) |
| Audit | **IMPLEMENTED** | `RESTAURANT_CREATED`, `RESTAURANT_UPDATED` (`audit.py:90-91`) |
| Frontend create | **PARTIAL** | name only (`administration.tsx:97-103`) — timezone and slug not settable, so every site is UTC |
| Frontend edit | **MISSING** | `organizationApi.updateRestaurant` exists (`src/shared/api/observations.ts:192`) with **zero call sites**. PROVEN by repo-wide search. |
| Site object page | **MISSING** | no `/admin/sites/:id` |
| R6 expressible for sites | **INCORRECT** | see §5 |

**Plainly stated:** site administration is a create-only surface with a mis-gated read, an unreachable edit API, no deletion, and a timezone field no operator can set — on a compliance product where a site's timezone determines what "this week's violations" means.

---

## 12. Zone administration gap analysis

**Real schema.** `Zone`, `app/domain/models.py:79-100`: `id`, `restaurant_id` (FK CASCADE), `name`, `created_at`. **No `is_active`. No `organization_id`** — tenancy is inherited through the restaurant, which is why every zone query must join.

| Capability | Verdict | Evidence |
|---|---|---|
| Read / manage permission | **MISSING** | none in `Permission` |
| List API | **PARTIAL/INCORRECT** | `GET /zones` (:276), gated `VIEW_USERS` |
| Create API | **PARTIAL/INCORRECT** | `POST /zones` (:315), gated `MANAGE_ORGANIZATION`, restaurant ownership verified at :325 |
| Update API | **PARTIAL/INCORRECT** | `PATCH /zones/{id}` (:344) — name only |
| Retire / delete | **MISSING** | no column, no route |
| Move zone between sites | **MISSING** | `restaurant_id` not in the patch surface |
| Audit | **IMPLEMENTED** | `ZONE_CREATED`, `ZONE_UPDATED` (`audit.py:92-93`) |
| Frontend create | **IMPLEMENTED** (narrow) | site select + name (`administration.tsx:344-463`) |
| Frontend edit | **MISSING** | `organizationApi.updateZone` (`observations.ts:203`) — **zero call sites**. PROVEN. |
| Zone object page | **MISSING** | no `/admin/zones/:id` |
| Camera→zone assignment from UI | **MISSING** | see §13.3 |

**`CameraZoneAssignment` — IMPLEMENTED and unusually well designed, and entirely invisible.** `app/domain/models.py:181-260`: append-only interval table (`camera_key`, `zone_id`, `zone_name`, `restaurant_id`, `effective_from`, `effective_to`) written by `record_assignment` on camera create (`domain/cameras.py:98-106`) and on any zone change during update (`:155-163`), closing the open interval on retirement (`:277-289`). It exists so a past observation resolves to where it actually happened rather than where the camera is now — a genuinely hard problem, solved correctly.

**And no frontend renders it, and no frontend can write to it.** `CameraDraft` (`src/shared/api/persistence.ts:49-61`) has **no `zone_id` field at all**, so the registration form cannot set a zone, and there is no camera edit form to change one. The only writer is `PATCH /api/v1/cameras/{key}` with a `zone_id` body — reachable by curl, by nothing in the product. **PARTIAL** (backend correct, frontend absent).

---

## 13. Camera administration gap analysis

### 13.1 The real `Camera` schema, field by field

`app/domain/models.py:103-179`. This is the actual table, not an approximation:

| Field | Line | Type / default | Settable via API | Settable via frontend |
|---|---|---|---|---|
| `id` | 125 | `String(64)`, uuid4 hex | no (generated) | no |
| `organization_id` | 126 | FK `organizations.id` CASCADE | no — from `access.tenant_id` | n/a |
| `restaurant_id` | 127 | FK `restaurants.id` CASCADE | ✅ create + update | ⚠️ **raw id typed into a text box** |
| `zone_id` | 130 | FK `zones.id` SET NULL, nullable | ✅ create + update | ❌ **not in `CameraDraft`** |
| `camera_key` | 136 | `String(64)`, unique per org | ✅ create only | ✅ create |
| `name` | 137 | `String(255)` | ✅ | ✅ |
| `purpose` | 140 | `String(255)`, default `""` | ✅ | ✅ |
| `host` | 143 | `String(255)`, default `""` | ✅ | ❌ **absent — no stream possible without it** |
| `rtsp_port` | 144 | `Integer`, default 554 | ✅ | ❌ |
| `channel` | 145 | `Integer`, default 1 | ✅ | ✅ |
| `stream_type` | 146 | `String(16)`, default `"sub"`, ∈ {main, sub} | ✅ | ❌ |
| `username` | 147 | `String(128)`, default `""` | ✅ | ❌ |
| `credential_ref` | 149 | `String(512)`, default `""` | ✅ | ✅ |
| `analysis_fps` | 152 | float, default 4.0 | ✅ | ❌ |
| `enabled` | 153 | `Boolean`, **default False** | update only (create forces False, `product.py:107`) | ✅ toggle |
| `analysis_enabled` | 167 | `Boolean`, default True, `server_default=true` | **update only** — `CameraService.create` has no such parameter (`domain/cameras.py:49-71`) | ❌ |
| `created_at` / `updated_at` | 170-176 | timestamps | no | no |

Constraints: `uq_camera_key` on `(organization_id, camera_key)`; indexes on `(organization_id, enabled)` and `restaurant_id` (`:115-120`).

### 13.2 Backend camera administration

**IMPLEMENTED and, on the whole, well built.**

- `CameraService.create` (`domain/cameras.py:49-115`) — validates, refuses duplicate key, forces disabled-by-default, opens the first zone interval.
- `CameraService.update` (`:119-172`) — allow-list of 12 mutable fields (`:124-138`); rejects a non-boolean `analysis_enabled` rather than coercing it (`:145-151`), with an explicit note that a truthy `"false"` string could otherwise switch analysis **on** while the operator believed the opposite. Correct and unusually careful.
- `CameraService.retire` (`:177-271`) — deletes the camera **and** truncates its observation partition, or does neither; refuses when a durable log is configured but unreachable. Writes two audit rows. Excellent.
- Tenant ownership on create: `_restaurant_in_tenant` before insert (`product.py:88`).

**Gap:** `update`'s allow-list silently ignores unknown or `None` fields (`:165-168`) rather than rejecting them — a typo'd field name is a silent no-op returning 200. **PARTIAL.**

**Gap:** `analysis_enabled` cannot be set at creation. A camera is created analysed-by-default and must be patched to become watch-only. **PARTIAL.**

### 13.3 Frontend camera administration — the biggest single gap

`CamerasPage`, `src/features/persistence-routes.tsx:158-328`:
- Reads `GET /cameras`; renders `Figure` totals (registered / enabled / not-processed) and a `DataTable`.
- One write: an Enable/Disable button behind `PermissionGate manageCameras` (`:186-198`) → `camerasApi.setEnabled`.
- A "Register a camera" button behind the same gate (`:219-223`) opening `RegisterCamera`.

`RegisterCamera` modal, `:330-411` — **the entirety of camera onboarding in this product**:

```
camera_key      Input, free text
name            Input, free text
channel         Input, free text coerced with Number()
restaurant_id   Input, FREE TEXT — the operator types a 32-char uuid4 hex by hand
credential_ref  Input, free text
purpose         Input, free text
```

Submit calls `camerasApi.create` with exactly those six (`:356-367`). Validity requires only non-empty `camera_key`, non-empty `restaurant_id`, numeric `channel` (`:369-372`).

**Therefore a camera created through the product's own UI has `host = ""`.** And `_start_cameras_from_database` filters `if row.host and row.analysis_enabled` (`app/main.py:365`). **A camera onboarded through the frontend can never stream and can never be analysed, no matter how many times its Enable toggle is pressed.** PROVEN by reading both sides. The UI will happily show it as "Enabled".

`CameraDetailPage`, `:1416-1600+` — **read-only**. Live state, incidents raised here, retained frames, and a "Back to the estate" link. No edit control of any kind. The file's own comment at `:148` concedes it: *"Editing and retirement are deliberately not here."*

`camerasApi.update` exists (`src/shared/api/persistence.ts:67-68`) and is called from **exactly one place**, `setEnabled` (`:69-70`). PROVEN by repo-wide search.

**Classification: PARTIAL, and non-functional for its primary purpose.** Camera administration in the UI is: see a list, toggle enabled, and create a camera that cannot work.

---

## 14. CCTV onboarding gap analysis

The requested journey (R12) versus what exists:

| Step | Required | Today | Verdict |
|---|---|---|---|
| 1. Record the recorder | Register DVR/NVR host, port, credentials once | No DVR entity; `host`/`rtsp_port` repeated per camera, informally grouped | **MISSING** (deferred by prior decision) |
| 2. Discover channels | Probe or enumerate the DVR's channels | Nothing | **MISSING** |
| 3. Create the camera | Full transport config | 6 fields, no `host` | **PARTIAL / non-functional** |
| 4. Place it | Pick site from a list; pick zone from that site's zones | `restaurant_id` typed as a raw uuid; zone not offered at all | **INCORRECT** |
| 5. Set the credential | Choose or create a credential reference | Free-text box; no validation feedback before submit; `literal:` silently accepted (§15) | **PARTIAL** |
| 6. Test the connection | "Try this camera" before committing | Nothing — no test endpoint, no preview | **MISSING** |
| 7. Set analysis policy | `analysis_fps`, `analysis_enabled` | Neither in the form; `analysis_enabled` not even in the create API | **MISSING** |
| 8. Enable | Deliberate, audited second act | ✅ toggle, `CAMERA_ENABLED`/`CAMERA_DISABLED` audit (`product.py:146-155`) | **IMPLEMENTED** |
| 9. Confirm it works | See first frame / health | `/cameras/:key` shows live state read-only, and `/live` shows the wall | **PARTIAL** |
| 10. Edit later | Change host, port, zone, fps | No edit surface | **MISSING** |
| 11. Retire | Delete camera + purge observations | ✅ backend `DELETE /cameras/{key}` is excellent (`product.py:177-270`) — **no frontend control** | **PARTIAL** |

**Honest summary:** the backend can onboard a camera completely. The product cannot. An operator today must either curl the API or edit the database. Six of eleven steps are MISSING or INCORRECT, including step 3, without which the rest is moot.

---

## 15. RTSP security gap analysis

### 15.1 What is genuinely safe — verified end to end

| Path | Verdict | Evidence |
|---|---|---|
| Row stores a reference, not a value (intended design) | ✅ | `Camera.credential_ref` doc (`models.py:148-149`); `to_rtsp_config` passes it unresolved (`domain/cameras.py:497-517`) |
| Dial URL built at the moment of use, never stored | ✅ | `RtspCameraConfig.dial_uri` (`app/vision/sources/rtsp.py:126-130`) |
| Resolved password does not outlive the source | ✅ | `self._password = None` on teardown (`rtsp.py:261-262`) |
| Exception messages scrubbed of the live secret | ✅ | `LiveRtspSource._redact` replaces both raw and URL-quoted forms, and the username prefix (`rtsp.py:161-173`), applied at `:210` |
| Redacted URI is the only URI exposed | ✅ | `RtspCameraConfig.redacted_uri()` (`rtsp.py:114-117`) → `***:***@host:port/path` |
| `/devtools/live` and `/devtools/vision` expose no credential | ✅ | `LiveRuntime.describe_cameras()` returns `camera_id`, `uri` (redacted), `channel`, `stream_type`, `analysis_fps`, `credential_configured` — **no `credential_ref`** (`app/vision/manager.py:378-390`), consumed at `devtools.py:211, 228` and `routes.py:314` |
| `literal:` refs redacted in secret-resolution errors | ✅ | `_safe()` returns `literal:***` (`app/vision/secrets.py:113-120`) |
| Camera `to_wire` never emits a dialling URL | ✅ | `_redacted_uri` (`domain/cameras.py:396-405`) |
| Audit `_scrub` strips credential-shaped values | ✅ (partially) | `_FORBIDDEN_KEYS` and `_SECRET_SHAPES` including `rtsp://user:pass@` (`audit.py:118-148`) |
| Frontend has no password field | ✅ | `RegisterCamera` (`persistence-routes.tsx:330-411`); the comment at `:330-337` states the property explicitly and is accurate about the form |

**On the narrow question the prior reports answered — "does a password appear in a list response?" — the answer is no**, provided every deployment uses `env:` or `file:`.

### 15.2 **The leak — INCORRECT, and it is real**

`_validate` explicitly permits `literal:` (`app/domain/cameras.py:349-361`):

```python
if credential_ref and not any(
    credential_ref.startswith(scheme) for scheme in ("env:", "file:", "literal:")
):
    raise ValidationError("credential_ref must be a reference (env:, file: or literal:), never a password; …")
```

`EnvironmentSecretProvider.resolve` implements it as *the reference contains the secret* (`app/vision/secrets.py:76-81`), and `secrets.py:16` labels it "development only, and it says so".

`to_wire` then returns the field **verbatim** (`app/domain/cameras.py:363-395`):

```python
"credential_ref": camera.credential_ref,
"credential_configured": bool(camera.credential_ref),
```

**Therefore, for any camera configured with `literal:hunter2` — a value the API accepts without complaint, offered through a free-text box the frontend labels "A pointer such as env:CCTV_PASSWORD":**

1. **The plaintext password is stored in the `cameras` table.** The table's own docstring says *"a database dump must not be a credential dump"* (`domain/cameras.py:10-14`, `models.py:110-113`). With `literal:`, it is one.
2. **It is returned by `GET /api/v1/cameras`** to every holder of `VIEW_CAMERAS` — which is `SUPER_ADMIN`, `ORG_ADMIN`, `RESTAURANT_MANAGER`, `KITCHEN_SUPERVISOR`, `HYGIENE_OFFICER` and `DEVELOPER` (`model.py:200-366`). A kitchen supervisor on a shared wall screen holds it.
3. **It is returned by `POST /cameras` and `PATCH /cameras/{key}` responses** (`product.py:122, 175`).
4. **It is written into the audit trail.** `product.py:119-123` records `detail={"channel": …, "credential_ref": camera.credential_ref}` with the comment *"credential_ref is a pointer and safe to record; the scrubber would remove a value even if one were passed by mistake."* **That comment is wrong.** `credential_ref` is not in `_FORBIDDEN_KEYS` (`audit.py:118-138`), and `literal:hunter2` matches none of `_SECRET_SHAPES` (`audit.py:141-146` — nvapi, `rtsp://u:p@`, JWT, bcrypt). Audit rows are, by this system's own statement, its longest-retained data (`audit.py:154-157`).
5. **It reaches the UI.** `Camera.credential_ref` is in the frontend type and rendered wherever camera detail is shown.

**The redaction in `secrets.py:_safe` proves the authors knew `literal:` is a secret-bearing form.** That knowledge reached the error-message path and did not reach `to_wire` or the audit call.

**Verdict: LEAKING, conditionally.** Safe if and only if every deployment uses `env:` or `file:` exclusively — a discipline nothing in the code enforces, on a field whose input is a free-text box. Given the frontend hint text guides toward `env:`, a real leak in the current deployment is **NOT VERIFIED** (no production data was inspected); the *mechanism* is PROVEN.

### 15.3 Credential update semantics

**PARTIAL.** `credential_ref` is in `update`'s allow-list (`domain/cameras.py:132`), so `PATCH /cameras/{key}` with a new reference rotates it, revalidated at `:167-173`, audited as a generic `CAMERA_UPDATED` with `detail={"fields": [...]}` (`product.py:166-174`) — field names only, correctly. But:
- No `CREDENTIAL_ROTATED` audit action.
- No `credential_rotated_at` column.
- No bulk rotation across the cameras sharing a DVR host.
- No frontend path (no camera edit form).
- No verification that the new reference resolves before it is stored — a bad rotation is discovered when the stream drops.

### 15.4 Required changes

1. **Remove `literal:` from `_validate`'s accepted schemes** (`domain/cameras.py:352`), or gate it behind a settings flag that is off unless the deployment is development.
2. **Never return `credential_ref` from `to_wire`.** Return `credential_scheme` (the part before `:`) and `credential_configured`. The scheme is diagnostic; the reference body is not needed by any UI.
3. **Add `credential_ref` to `_FORBIDDEN_KEYS`** and stop passing it in the `CAMERA_CREATED` detail (`product.py:122`).
4. Add `literal:` to `_SECRET_SHAPES` as a defence-in-depth regex.
5. Add a `POST /cameras/{key}/credential` rotation route with its own audit action and a resolve-check before commit.

---

## 16. Organization lifecycle gap analysis

### 16.1 What each state actually does today — traced, PROVEN

**`ACTIVE`** — unchanged behaviour. `parse_organization_status(None)` also reads as ACTIVE (`resolver.py:111-112`).

**`SUSPENDED`** — **exactly one effect.** In `decide()` (`resolver.py:158-159`):

```python
if org_status is OrganizationStatus.SUSPENDED:
    permissions = _suspend(permissions)
```

and `_suspend` (`:119-122`) is:

```python
return frozenset(p for p in permissions if not p.value.startswith("manage_"))
```

A **string-prefix filter**. Its consequences, none of them documented at the call site:
- Drops `MANAGE_ORGANIZATION`, `MANAGE_USERS`, `MANAGE_CAMERAS`, `MANAGE_TABLE_OCCUPANCY`, `MANAGE_CUTTING_BOARD`, `MANAGE_PATRON_ID`, `MANAGE_POS_INTEGRATION`.
- **Does not drop** `ACKNOWLEDGE_INCIDENTS`, `RESOLVE_INCIDENTS`, `DELETE_EVIDENCE`, `EXPORT_REPORTS`, `REGISTER_DEMAND`. A suspended tenant can still **delete evidence**, **export reports**, and **spend model budget** via `REGISTER_DEMAND`. Whether that is intended is **REQUIRES DECISION**; that it is undocumented and follows from a naming convention rather than a policy statement is **INCORRECT**.
- Any future permission whose name happens to start with `manage_` is silently swept in; any restricting permission that does not is silently missed.

**`ARCHIVED`** — refuses login (`auth/service.py:82`) and refuses every authenticated request (`auth/service.py:146`), so a token minted before archival stops working immediately. Correct and well done.

**Unreadable status** — `parse_organization_status` returns `SUSPENDED` for an unparseable value (`resolver.py:113-116`), denying toward the narrower state. Correct.

### 16.2 Does SUSPENDED or ARCHIVED stop cameras, analysis or streaming?

**No. MISSING. PROVEN.**

- `_start_cameras_from_database` (`app/main.py:338-397`) queries `CameraService.enabled_for_runtime(organization_id=cfg.default_tenant_id)`, which filters on `Camera.enabled` alone (`domain/cameras.py:319-324`). **No organization is loaded, no status is read.**
- `app/vision/compliance_driver.py` constructs `TenantId(self._settings.default_tenant_id)` at `:195` and `:235` and queries cameras at `:213` without touching `Organization`.
- `app/vision/manager.py:238` likewise.
- A repo-wide search for `OrganizationStatus` and `parse_organization_status` returns **only** `auth/service.py`, `authorization/resolver.py`, `authorization/model.py` and `api/user_administration.py`'s docstring. **Nothing under `app/vision/` reads it.**

**Therefore:** suspend an organization and its cameras keep streaming, its frames keep decoding, its VLM calls keep being made, its incidents keep being created. Archive it and users cannot log in — *and the cameras keep running*. For a suspension that exists to stop a non-paying or non-compliant customer, that is the wrong behaviour, and it is silent.

The freeze document's §17 matrix marks streaming and background analysis under SUSPENDED as **REQUIRES OWNER DECISION** and calls it "the single most important open item". That decision was never taken; Stages 1–8 shipped the lifecycle column and the auth-chokepoint half without it. Recording the state honestly: the **decision** is REQUIRES DECISION; the **absence of any wiring at all, including for ARCHIVED, where the answer is not in doubt** is **MISSING**.

### 16.3 Everything else about lifecycle

| Capability | Verdict |
|---|---|
| Set an organization's status | **MISSING** — no route, no CLI; requires direct SQL |
| Audit a status change | **MISSING** — no `ORGANIZATION_SUSPENDED`/`ARCHIVED` action (`audit.py:36-110`) |
| Reason and timestamp for suspension | **MISSING** — no columns |
| Tell the user their org is suspended | **MISSING** — `/auth/me` returns no status (`routes.py:259-271`); the UI simply loses its buttons via `PermissionGate` with no explanation |
| Un-archive | **MISSING** |
| Retention / deletion on archive | **MISSING** — data retained indefinitely by design (`model.py:406-408`), with no sweep and no stated policy |

---

## 17. Platform operator boundary

**MISSING. PROVEN.** §6.3 establishes the structural facts. This section states the target.

**The property to preserve.** `AccessDecision` carries exactly one `tenant_id`, mandatory and non-empty, derived from the authenticated user's row and never from request input (`model.py:494, 507-511`; `resolver.py:164`; `auth/service.py:10-16`). Cross-tenant leakage is impossible *by construction*. Widening `Role.SUPER_ADMIN` to be cross-tenant would destroy that guarantee across every tenant-scoped query in the application.

**The recommended shape — a second, disjoint principal type.**

```
PlatformOperator                      never becomes an AccessDecision
├── authenticates through its own path (separate table or an is_platform_operator flag)
├── reaches ONLY /api/v1/operator/* — organization CRUD, lifecycle, first-admin provisioning
├── CANNOT read tenant data: no observations, no evidence, no incidents, no imagery, no audit detail
├── every act writes an audit row against the TARGET organization, attributed to the operator
└── the operator API constructs no Scope, no Grant, no Principal
```

**Why disjoint rather than a wider role:** a platform operator's job is *administering tenancy*, not *seeing inside tenants*. Making it a role would give it an `AccessDecision`, which would give it a `tenant_id`, which is the exact confusion to avoid. Keeping it disjoint also means the tenant-scoped code never learns that operators exist — no `if is_platform_operator` branches scattered through query builders, which is where this kind of design usually rots.

**REQUIRES DECISION:** whether a platform operator may ever *impersonate* a tenant admin for support. Recommendation: **no**, at least initially — impersonation is the feature that turns a narrow boundary into a wide one, and this product handles CCTV imagery of identifiable staff.

---

## 18. Required backend work

| # | Work | Files | Verdict addressed |
|---|---|---|---|
| B1 | Add `VIEW_SITES`, `MANAGE_SITES`, `VIEW_ZONES`, `MANAGE_ZONES` to `Permission`; assign in `ROLE_PERMISSIONS` (`RESTAURANT_MANAGER` gets `VIEW_SITES`/`VIEW_ZONES`; `ORG_ADMIN` gets all four) | `app/authorization/model.py:69-186, 200-366` | §5, §8, R6, R8 |
| B2 | Re-gate the six site/zone routes onto the new permissions | `app/api/administration.py:126, 160, 196-198, 276, 315, 344` | §11, §12 |
| B3 | Add `VIEW_ORGANIZATION`, `MANAGE_ORGANIZATION_LIFECYCLE`; narrow `MANAGE_ORGANIZATION`'s remit to organization-level settings only | `app/authorization/model.py` | §8.2 |
| B4 | **Fix `create_user` to create an `AccessGrant`.** Accept an explicit camera scope (`none` / `listed` / `all_in_tenant`) in the payload; refuse to default to `all_in_tenant` silently | `app/api/user_administration.py:281-362` | **§7.4 — highest priority** |
| B5 | Add `PUT /admin/users/{id}/access-grant` for camera/site scope, with the same anti-escalation discipline | `app/api/user_administration.py` | §7.4 |
| B6 | Add `POST /admin/users/{id}/reset-password` returning a generated password once; new `USER_PASSWORD_RESET` audit action | `app/api/user_administration.py`, `app/domain/audit.py` | §7.2 |
| B7 | Remove `literal:` from accepted credential schemes (or gate on development) | `app/domain/cameras.py:349-361` | §15.2 |
| B8 | Stop returning `credential_ref`; return `credential_scheme` + `credential_configured` | `app/domain/cameras.py:363-395` | §15.2 |
| B9 | Add `credential_ref` to `_FORBIDDEN_KEYS`; stop passing it in `CAMERA_CREATED` detail; add a `literal:` regex to `_SECRET_SHAPES` | `app/domain/audit.py:118-148`, `app/api/product.py:122` | §15.2 |
| B10 | `POST /cameras/{key}/credential` — rotate with resolve-check and its own audit action | `app/api/product.py`, `app/domain/cameras.py` | §15.3 |
| B11 | Accept `analysis_enabled` in `CameraService.create`; reject unknown fields in `update` rather than ignoring them | `app/domain/cameras.py:49-71, 165-168` | §13.2 |
| B12 | `GET/POST/PATCH /api/v1/operator/organizations` behind the platform-operator boundary; `POST …/{id}/admins` for first-admin provisioning | new module | §6, §17, R1, R3, R4 |
| B13 | Organization lifecycle routes with `ORGANIZATION_SUSPENDED` / `ORGANIZATION_ARCHIVED` / `ORGANIZATION_REACTIVATED` audit actions | new module, `app/domain/audit.py` | §16.3 |
| B14 | Replace `_suspend`'s prefix filter with an explicit `SUSPENDED_DENIES: frozenset[Permission]` set | `app/authorization/resolver.py:119-122` | §16.1 |
| B15 | **Make the runtime multi-tenant.** Iterate organizations for camera bootstrap; carry `organization_id` through the compliance driver instead of `default_tenant_id` | `app/main.py:338-397, 490-525`, `app/vision/compliance_driver.py`, `app/vision/manager.py:238` | §6.5 |
| B16 | Gate the runtime on lifecycle: ARCHIVED stops sessions; SUSPENDED per the §29.2 decision | `app/main.py`, `app/vision/manager.py` | §16.2 |
| B17 | `DELETE /zones/{id}` or `zones.is_active`; site archive route | `app/api/administration.py` | §11, §12 |
| B18 | Return organization name and status from `/auth/me` | `app/api/routes.py:259-271` | §16.3, §10 |
| B19 | Site/zone/camera list pagination and filtering | `administration.py`, `product.py` | §10.2 |

## 19. Required frontend work

| # | Work | Files |
|---|---|---|
| F1 | Add the four site/zone permission constants | `src/app/permissions/permissions.ts:15-84` |
| F2 | Split `/admin` into the IA of §10.3: overview, `/admin/sites`, `/admin/sites/:id`, `/admin/zones/:id`, `/admin/accounts`, `/admin/accounts/:id`, `/admin/organization` | `src/features/administration.tsx`, `src/app/router/AppRouter.tsx`, `src/app/router/navigation.ts` |
| F3 | **Rebuild camera registration**: `host`, `rtsp_port`, `stream_type`, `username`, `analysis_fps`, `analysis_enabled`; **site and zone as `Select`s populated from the API, never free text** | `src/features/persistence-routes.tsx:330-411`, `src/shared/api/persistence.ts:49-61` |
| F4 | **Add a camera edit surface** — Drawer with Transport / Placement / Analysis tabs on `/admin/cameras/:key`, wiring the already-existing `camerasApi.update` | `src/features/persistence-routes.tsx:1416+` |
| F5 | Camera retirement control behind `PermissionGate manageCameras`, with a confirmation naming the observation purge | same |
| F6 | Wire the unused `organizationApi.updateRestaurant` / `updateZone` to real edit surfaces | `src/features/administration.tsx`, `src/shared/api/observations.ts:192, 203` |
| F7 | Camera-scope control on the user object page (pairs with B5) | `src/features/user-detail.tsx` |
| F8 | Zone assignment history as a `Timeline` on the zone and camera object pages | new |
| F9 | Organization lifecycle banner — `Attention` when suspended, stating what is restricted, rather than silently hiding controls | `src/shared/layout/AppShell.tsx` |
| F10 | Move Cameras into the Administration nav section; add the Platform Operator section, visible only to operators | `src/app/router/navigation.ts:102ff` |
| F11 | Fix `/admin`'s gate so a `manage_organization`-only holder does not trigger a 403 accounts query | `src/features/administration.tsx:80`, `AppRouter.tsx` |
| F12 | Password-reset control on the user object page (pairs with B6) | `src/features/user-detail.tsx` |

## 20. Required database work

| # | Migration | Notes |
|---|---|---|
| D1 | `zones.is_active BOOLEAN NOT NULL SERVER_DEFAULT true` | additive |
| D2 | `organizations.suspended_at`, `suspension_reason`, `archived_at` (nullable) | additive |
| D3 | `organizations.created_by` (nullable) | additive |
| D4 | `cameras.credential_rotated_at` (nullable) | additive |
| D5 | Platform-operator identity — new table or `users.is_platform_operator` | **REQUIRES DECISION** (§29.3) |
| D6 | Index `zones(restaurant_id, name)` for the zone ledger | additive |
| D7 | Optional `cameras.dvr_group_id` (nullable) | deferred; prior decision, unchanged |

**No permission migration is needed.** `Permission` is a code enum; B1 is a code change and a deploy, not a data migration. Say so explicitly, because the shape of `permission_overrides` invites the opposite assumption.

---

## 21. Security requirements

| # | Requirement | Current | Verdict |
|---|---|---|---|
| S1 | Tenancy never from request input | `AccessDecision.tenant_id` from the user row only | **IMPLEMENTED** |
| S2 | Cross-tenant reads are 404, not 403 | `_user_in_tenant`, `_restaurant_in_tenant` | **IMPLEMENTED** |
| S3 | Authorization recomputed per request | `decision_for_claims` | **IMPLEMENTED** |
| S4 | Empty camera tuple never means "all" | `CameraScope` three-state, `to_grant` refuses NONE | **IMPLEMENTED** |
| S5 | Unknown stored values narrow access | `parse_roles`, `parse_camera_scope`, `parse_overrides`, `parse_organization_status` | **IMPLEMENTED** |
| S6 | No self-escalation | route check + structural `_guard` | **IMPLEMENTED** |
| S7 | Grantor may only grant what they hold | `access.has()`, `_require_grantable_role` | **IMPLEMENTED** |
| S8 | No password or hash in any response | `_user_to_wire`; generated password returned once | **IMPLEMENTED** |
| S9 | **No RTSP secret in any response, log, error or audit row** | `literal:` breaks it (§15.2) | **INCORRECT** |
| S10 | New permissions default to nobody | must hold for B1/B3 — grant deliberately, never `frozenset(Permission)` | **REQUIRED** |
| S11 | Platform operator cannot read tenant data | boundary does not exist | **MISSING** |
| S12 | Lifecycle enforced at the runtime, not only the API | not wired (§16.2) | **MISSING** |
| S13 | New API-created users default to **no** camera scope, explicitly, rather than by omission | currently by omission, and silently (§7.4) | **INCORRECT** |
| S14 | Every administrative write audited | true for users/sites/zones/cameras; **not** for organizations | **PARTIAL** |

---

## 22. Audit requirements

Existing `AuditAction` (`app/domain/audit.py:36-110`): auth ×3, camera ×5, incident ×3, evidence ×5, observation ×2, report ×3, `POLICY_CHANGED`, `ADMIN_CHANGED`, restaurant ×2, zone ×2, user ×4, role ×2, permission ×3. `AuditOutcome` = SUCCESS / DENIED / FAILED (`:113-116`). `_scrub` is structural (`:152-…`), depth-capped at 6, list-capped at 50, detail truncated to 4000 chars (`:230`).

**Required additions:**

| Action | Why |
|---|---|
| `ORGANIZATION_CREATED` / `_UPDATED` / `_SUSPENDED` / `_ARCHIVED` / `_REACTIVATED` | §6.2 — tenant creation is currently entirely unaudited |
| `USER_PASSWORD_RESET` | §7.2 |
| `ACCESS_GRANT_CHANGED` | §7.4 / B5 — changing which cameras a user reaches is at least as consequential as changing a role |
| `CAMERA_CREDENTIAL_ROTATED` | §15.3 |
| `SITE_ARCHIVED`, `ZONE_RETIRED` | §11, §12 |

**Required fixes:**

1. `credential_ref` into `_FORBIDDEN_KEYS`, and out of the `CAMERA_CREATED` detail (`product.py:122`). The comment there asserting the scrubber protects it is false.
2. `literal:[^\s]+` into `_SECRET_SHAPES`.
3. Every platform-operator act must write against the **target** organization, attributed to the operator, so a tenant's own audit trail shows what was done to it.

---

## 23. Testing requirements

**Current coverage.** Backend: `tests/app/test_permission_overrides.py` (31 tests), `test_user_administration.py` (39), `test_administration.py` (14), plus `test_persistence.py`, `test_camera_bootstrap_recovery.py`, `test_analysis_enabled.py`, `test_camera_wall.py`, `test_migration.py`. Frontend: 24 suites including `administration.test.tsx` (181 lines), `user-detail.test.tsx` (408), `persistence.test.tsx` (503), `information-architecture.test.tsx` (292), `shell.test.tsx` (355).

**Required new tests — each maps to a finding above:**

*Permission vocabulary (B1/B2)*
1. A `restaurant_manager` with `VIEW_SITES` and no `MANAGE_SITES` gets 200 on `GET /restaurants` and 403 on `POST /restaurants`.
2. **The headline test:** two users, same `restaurant_manager` role, one with a GRANT of `MANAGE_SITES`; assert one may create a site and the other may not. *This test cannot be written today, and its existence is the acceptance criterion for R6.*
3. `VIEW_USERS` alone no longer opens `GET /restaurants`.

*Access grant (B4/B5)*
4. A user created via `POST /admin/users` with `camera_scope: "all_in_tenant"` sees the tenant's cameras.
5. One created with no scope stated sees none — **and the API says so in the response**, rather than returning a user who silently sees nothing.
6. Changing a user's scope takes effect on their next request.

*RTSP security (B7/B8/B9)*
7. `POST /cameras` with `credential_ref: "literal:hunter2"` is refused in production configuration.
8. No `/cameras` response body contains the substring after a scheme prefix — assert on the raw JSON.
9. An audit row created from a camera create contains no credential material.

*Lifecycle (B13/B14/B16)*
10. Suspending an organization removes exactly the permissions in `SUSPENDED_DENIES` and no others — assert the set, not the prefix.
11. Archiving stops camera sessions for that organization and leaves other organizations' sessions running.
12. A suspended organization cannot `DELETE_EVIDENCE` (once §29.6 is decided).

*Multi-tenancy (B12/B15)*
13. Two organizations, each with enabled cameras; both start. Today this fails.
14. Organization A's operator cannot read B's cameras, incidents, evidence or audit.
15. A platform operator can create an organization and its first admin, and **cannot** read any tenant's observations.

*Frontend*
16. The camera registration form submits `host`; a camera created through the UI satisfies the runtime's `if row.host` filter.
17. Site and zone are `Select`s, not text inputs — assert no free-text uuid entry survives.
18. The camera edit drawer round-trips every mutable field.
19. `/admin/sites/:id` and `/admin/zones/:id` deep-link correctly and honour permission gates.
20. A `manage_organization`-only holder reaching `/admin` triggers no 403 query.

## 24. Browser verification requirements

Prior stages performed CDP-driven headless verification (`ADMINISTRATION_CONTROL_PLANE_END_TO_END_INTEGRATION_AUDIT.md` §20). **NOT VERIFIED** this pass — no browser was run. Required flows for the work below, each as a real interaction, not a screenshot:

1. Sign in as `org_admin` → `/admin` → create a site with a timezone → open `/admin/sites/:id` → rename it → confirm the audit row.
2. Create a zone from the site page → open the zone → rename → retire.
3. Register a camera **with host, port, stream type, credential, site and zone chosen from selects** → confirm it appears disabled → enable it → confirm a live session starts. *This is the flow that is impossible today.*
4. Edit that camera's `analysis_fps` and toggle `analysis_enabled`; confirm the runtime honours both.
5. Create a user with an explicit camera scope → sign in as them → confirm they see cameras. *Today this shows an empty product.*
6. GRANT `MANAGE_SITES` to Manager A, leave Manager B inheriting → sign in as each → confirm A sees the create control and B does not, and that B's API call is refused. **This is R6's browser acceptance test.**
7. REVOKE a permission → confirm the affected user's next navigation reflects it without re-login.
8. Suspend an organization → confirm the banner appears, writes are refused, reads continue, and (per §29.2) streaming behaves as decided.
9. As a platform operator, create an organization and its first admin → sign in as that admin → confirm an empty but working tenant.
10. Confirm at 1280px and 768px that no administration table overflows the page horizontally.
11. **Inspect the `/cameras` network response in DevTools and confirm no credential material is present.**

---

## 25. Full staged implementation roadmap

Stages A–Q. Ordering is dependency-driven; two deviations from a naive reading are called out and justified. Every capability in §2 appears somewhere.

**Stage A — Correct the permission vocabulary.** Add `VIEW_SITES`, `MANAGE_SITES`, `VIEW_ZONES`, `MANAGE_ZONES`, `VIEW_ORGANIZATION`, `MANAGE_ORGANIZATION_LIFECYCLE`. Assign in `ROLE_PERMISSIONS` deliberately. No route changes yet. *Everything downstream depends on this; it is first because R6 is unrepresentable until it lands.* → B1, B3

**Stage B — Re-gate sites and zones.** Point the six routes at the new permissions. Tests 1 and 3. *Deviation note: this precedes the AccessGrant fix even though that fix is more damaging, because A and B are one coherent change to the same enum-and-gate surface and splitting them leaves the codebase in a state where new permissions exist and nothing checks them.* → B2

**Stage C — Fix the `AccessGrant` defect.** Explicit camera scope on `create_user`; `PUT /admin/users/{id}/access-grant`; `ACCESS_GRANT_CHANGED` audit; frontend control. Tests 4–6, F7. *The most damaging single defect; it is third only because it should ship on a corrected vocabulary.* → B4, B5, F7

**Stage D — Close the RTSP credential leak.** Drop `literal:` in production, stop returning `credential_ref`, scrub the audit path. Tests 7–9. *Security fix; could be moved to first if the deployment is known to use `literal:`.* → B7, B8, B9

**Stage E — Camera credential rotation.** Rotation route, resolve-check, audit action, `credential_rotated_at`. → B10, D4

**Stage F — Complete the camera administration API.** `analysis_enabled` at creation; reject unknown update fields; pagination. → B11, B19

**Stage G — Rebuild camera onboarding in the frontend.** Full transport form; site and zone as selects; `CameraDraft` extended. Tests 16–17, browser flow 3. → F3

**Stage H — Camera edit and retirement surfaces.** `/admin/cameras/:key` with Transport / Placement / Analysis tabs; retirement with confirmation. Test 18, browser flow 4. → F4, F5

**Stage I — Site and zone object pages and edit surfaces.** `/admin/sites`, `/admin/sites/:id`, `/admin/zones/:id`; wire the two unused API clients; `zones.is_active`; site archive. Zone assignment history as a `Timeline`. Test 19, browser flows 1–2. → B17, D1, D6, F6, F8

**Stage J — Split the administration IA.** Implement §10.3: overview, ledgers, object pages; move Cameras into Administration; fix the `/admin` gate. Tests 19–20, browser flow 10. → F2, F10, F11

**Stage K — User lifecycle completion.** Password reset over HTTP with its audit action and UI control. → B6, F12

**Stage L — Organization lifecycle, correctly.** Replace `_suspend`'s prefix filter with an explicit deny set; add `suspended_at`, `suspension_reason`, `archived_at`; lifecycle routes and audit actions; surface status in `/auth/me` and as an `Attention` banner. Tests 10, 12; browser flow 8. → B13, B14, B18, D2, F9

**Stage M — Make the runtime multi-tenant.** Iterate organizations for camera bootstrap; carry `organization_id` through the compliance driver and manager. Test 13. *Deviation note: this must precede organization CRUD. Shipping the ability to create tenants whose cameras can never start would be the same class of over-claim this document exists to correct.* → B15

**Stage N — Wire lifecycle into the runtime.** ARCHIVED stops sessions; SUSPENDED per §29.2. Test 11. → B16

**Stage O — Platform operator boundary.** Operator identity (per D5's decision); `/api/v1/operator/*`; organization CRUD; first-admin provisioning; `ORGANIZATION_*` audit actions. Tests 14–15, browser flow 9. → B12, D3, D5

**Stage P — Platform operator frontend.** `/operator/organizations` ledger and object page; nav section; tenant context in the shell. → F10

**Stage Q — Verification and hardening.** Full test suite (§23), full browser matrix (§24), a written SUSPENDED/ARCHIVED behaviour matrix that matches the code line for line, and a documented CCTV onboarding runbook.

**Dependency summary:** A → B → {C, D} → {E, F} → G → H → I → J; L → N; M → N → O → P; Q last.

---

## 26. Explicit list of completed requirements

**IMPLEMENTED, verified this pass:**

1. Multiple users per role — `RoleAssignment` has no per-role uniqueness; union semantics in `permissions_for` (`model.py:368-373`).
2. Multiple roles per user — `parse_roles` unions across all assignments (`resolver.py:31-44`).
3. Three-state INHERIT / GRANT / REVOKE — `OverrideState` + absence-of-row (`model.py:373-392`).
4. REVOKE wins over role and over GRANT — `(roles ∪ granted) − revoked` (`model.py:415-427`).
5. At most one override per (user, permission) — `uq_permission_override_user_permission` (`d38dfad216a0`).
6. Immediate effect of any access change — `decision_for_claims` rebuilds per request (`auth/service.py:126-153`).
7. A fully-revoked user keeps zero permissions rather than re-deriving from roles (`model.py:496-500, 512-513`).
8. Self-modification refused, twice (`user_administration.py:162-163`; `overrides.py:102-107`).
9. Cross-tenant modification refused, structurally (`overrides.py:108-117`).
10. Anti-escalation on GRANT and on role assignment (`user_administration.py:189-201`).
11. Cross-tenant existence not disclosed — 404 not 403 (`user_administration.py:118-140`).
12. User activate / deactivate with immediate effect (`user_administration.py:407-465`; `resolver.py:136-146`).
13. Eleven-route user-administration API, uniformly `MANAGE_USERS`-gated (`user_administration.py:100-105`).
14. Per-user permission matrix API returning stored state, role default and effective result (`user_administration.py:217-254`).
15. User object page with Inherited / + Added / − Restricted (`src/features/user-detail.tsx`).
16. Nine user-administration audit actions with structural scrubbing (`audit.py:101-110, 152+`).
17. `ARCHIVED` refuses login **and** every subsequent request (`auth/service.py:82, 146`).
18. Unknown stored values narrow rather than widen access — four independent parsers (`resolver.py`).
19. `CameraScope` three-state; `to_grant()` refuses to convert NONE (`model.py:481-598`).
20. Camera CRUD API with tenant ownership check and observation-partition purge on retirement (`product.py:63-270`; `domain/cameras.py:177-271`).
21. `CameraZoneAssignment` append-only historical attribution (`models.py:181-260`; `domain/cameras.py:98-106, 155-163`).
22. `analysis_enabled` separated from `enabled`, with non-boolean input rejected rather than coerced (`domain/cameras.py:145-151`).
23. Camera create is disabled-by-default; enabling is separately audited (`product.py:107, 146-155`).
24. RTSP dial URL built at use, never stored; password cleared on teardown; exception messages redacted (`rtsp.py:126-130, 161-173, 261-262`).
25. DevTools and status endpoints expose no credential (`manager.py:378-390`).
26. Permission-gated routing and navigation on the frontend (`AppRouter.tsx`, `navigation.ts`, `PermissionGate`).
27. 84 backend tests across the three administration suites; 24 frontend suites.

## 27. Explicit list of missing requirements

1. `VIEW_SITES` / `MANAGE_SITES` permissions — **MISSING** (`model.py:69-186`).
2. `VIEW_ZONES` / `MANAGE_ZONES` permissions — **MISSING**.
3. `VIEW_ORGANIZATION` permission — **MISSING**.
4. Organization CRUD API — **MISSING** (no route in either repo).
5. Organization lifecycle transition API — **MISSING**.
6. Organization audit actions — **MISSING** (`audit.py:36-110`).
7. `suspended_at`, `suspension_reason`, `archived_at`, `created_by` columns — **MISSING**.
8. Platform-operator boundary — **MISSING** (§6.3, §17).
9. First-admin provisioning over HTTP — **MISSING** (CLI only).
10. Multi-tenant camera bootstrap — **MISSING** (`main.py:360`).
11. Multi-tenant perception / compliance driver — **MISSING** (`compliance_driver.py:195, 213, 235, 382, 404, 484, 522`).
12. Lifecycle gating of streaming — **MISSING** (§16.2).
13. Lifecycle gating of analysis / background work — **MISSING** (§16.2).
14. Password reset over HTTP — **MISSING** (§7.2).
15. Email change, user deletion, invitation, SSO — **MISSING** (§7.3).
16. Camera-scope (`AccessGrant`) write API — **MISSING** (§7.4).
17. Camera-scope UI control — **MISSING**.
18. Site object page `/admin/sites/:id` — **MISSING**.
19. Zone object page `/admin/zones/:id` — **MISSING**.
20. Site edit UI — **MISSING** (API client exists, zero call sites).
21. Zone edit UI — **MISSING** (same).
22. Site archive / zone retire — **MISSING** (`zones` has no `is_active`).
23. Camera edit UI — **MISSING** (`persistence-routes.tsx:148` concedes it).
24. Camera retirement UI — **MISSING** (backend route exists).
25. `host`, `rtsp_port`, `stream_type`, `username`, `analysis_fps` in the registration form — **MISSING** (`persistence-routes.tsx:347-354`).
26. `zone_id` anywhere in the frontend camera path — **MISSING** (`persistence.ts:49-61`).
27. Zone assignment history rendered anywhere — **MISSING**.
28. Camera connection test before commit — **MISSING**.
29. Credential rotation route, audit action and UI — **MISSING**.
30. DVR / NVR entity — **MISSING** (deferred by prior decision; unchanged).
31. Organization name and status on `/auth/me` — **MISSING**.
32. Suspension explanation in the UI — **MISSING**.
33. Pagination, search and filtering on every administration ledger — **MISSING**.
34. Tests 1–20 of §23 — **MISSING**.

## 28. Explicit list of incorrect / partial implementations

**INCORRECT**

1. `GET /restaurants` gated on `VIEW_USERS` (`administration.py:126`) — reading the estate requires the authority to enumerate every account.
2. `GET /zones` gated on `VIEW_USERS` (`administration.py:276`) — same.
3. `POST/PATCH /restaurants` gated on `MANAGE_ORGANIZATION` (`:160, 196-198`) — no way to grant site editing without granting whole-organization authority.
4. `POST/PATCH /zones` gated on `MANAGE_ORGANIZATION` (`:315, 344`) — same.
5. **R6 is unrepresentable** — no site or zone permission exists to GRANT or REVOKE (§5). The feature the eight stages existed to deliver cannot be demonstrated.
6. **`POST /admin/users` creates no `AccessGrant`** (`user_administration.py:281-362`) — every API-created account sees zero cameras, silently (§7.4).
7. **`literal:` credential refs are accepted and echoed** — plaintext password in the database, in list and detail responses to six roles, and in the longest-retained audit table (§15.2). The comment at `product.py:120-121` asserting the scrubber protects it is false.
8. `_suspend`'s string-prefix filter (`resolver.py:119-122`) — policy expressed as a naming convention; leaves `DELETE_EVIDENCE`, `EXPORT_REPORTS` and `REGISTER_DEMAND` in force for a suspended tenant.
9. `restaurant_id` as a free-text field in camera registration (`persistence-routes.tsx:400`) — an operator hand-types a uuid4 hex.
10. Cameras sit outside the Administration navigation section (`navigation.ts:~253-270`) while being the most administrative object in the product.
11. `/admin`'s any-of gate versus its unconditional `MANAGE_USERS` query (`administration.tsx:80`) — a reachable 403.

**PARTIAL**

12. `PATCH /restaurants/{id}` supports name, timezone and `is_active`; the UI exposes none of them.
13. `PATCH /zones/{id}` supports name only; no move between sites; UI exposes none.
14. Site creation collects `name` only — every site is UTC on a product where a site's timezone defines its reporting week.
15. `CameraService.update` silently ignores unknown fields (`domain/cameras.py:165-168`) — a typo returns 200 and changes nothing.
16. `analysis_enabled` cannot be set at creation (`domain/cameras.py:49-71`).
17. Camera onboarding covers 6 of 13 configurable fields and omits the one (`host`) without which the runtime skips the camera entirely (`main.py:365`).
18. `CameraDetailPage` is read-only.
19. Credential rotation works via `PATCH` but has no dedicated action, no resolve-check, no rotation timestamp, no UI.
20. Audit `_scrub` is structurally sound but its key list omits `credential_ref`.
21. Organization lifecycle exists as a column and an auth-chokepoint behaviour, with no way to set it and no reach into the runtime.
22. `/admin` is the "giant everything-admin page" the requirement explicitly ruled out.

---

## 29. Design decisions still required

**29.1 — REVOKE anti-escalation asymmetry.** May a `MANAGE_USERS` holder REVOKE a permission from a target who holds it via a role the actor cannot themselves reach? Concretely: may an `org_admin` strip a `super_admin`? Documented as deliberately unresolved at `user_administration.py:63-79`. **REQUIRES DECISION.** *Recommendation:* keep the asymmetry (revoking narrows and carries no escalation risk in the direction the GRANT rule blocks), but add a role-precedence floor so a strictly-higher role cannot be narrowed by a strictly-lower one.

**29.2 — SUSPENDED cascade to streaming and analysis.** The freeze document called this "the single most important open item"; it is still open, and it is now blocking Stage N. Options: (a) suspension is billing-only — streaming and analysis continue; (b) suspension stops analysis (the expensive half) and keeps streaming; (c) suspension stops both. **REQUIRES DECISION.** *Recommendation:* **(b)**. Analysis is where the money goes and `analysis_enabled` already exists as the per-camera expression of exactly that distinction (`models.py:155-168`), so (b) reuses a concept the system already has instead of inventing one. **ARCHIVED must stop both** — that half is not in doubt and should not wait on this decision.

**29.3 — Platform-operator identity representation.** A separate `platform_operators` table, or `users.is_platform_operator`? **REQUIRES DECISION.** *Recommendation:* a separate table. A flag on `users` means every `User` query must remember the distinction, and `User` already carries `organization_id NOT NULL` — which a platform operator has no honest value for.

**29.4 — Platform-operator impersonation.** May an operator act as a tenant admin for support? **REQUIRES DECISION.** *Recommendation:* no, initially.

**29.5 — Organization deletability.** Hard delete (exercising `ondelete=CASCADE` on restaurants, zones, cameras, users) or suspend/archive only? **REQUIRES DECISION.** *Recommendation:* archive only. `Incident.restaurant_id` is `ondelete=SET NULL` (`models.py:411-413`), so deleting a restaurant silently orphans incident history — the same class of error `CameraZoneAssignment` was built to prevent.

**29.6 — What SUSPENDED denies, explicitly.** Replacing the prefix filter requires naming the set. Is `DELETE_EVIDENCE` denied under suspension? `EXPORT_REPORTS`? `REGISTER_DEMAND`? **REQUIRES DECISION.** *Recommendation:* deny all `MANAGE_*` plus `DELETE_EVIDENCE` and `REGISTER_DEMAND` (both destroy or spend); allow `EXPORT_REPORTS` (a suspended customer must be able to take their compliance record with them).

**29.7 — Default camera scope for API-created users.** Once B4 lands, what does a payload with no scope mean? **REQUIRES DECISION.** *Recommendation:* refuse the request. Both silent defaults are wrong — `all_in_tenant` over-grants, `none` creates the invisible-broken-account this document reports as §7.4.

**29.8 — `literal:` credential refs.** Remove entirely, or gate on a development-only setting? **REQUIRES DECISION.** *Recommendation:* gate, and additionally never echo the reference body regardless of scheme (B8), so the two protections are independent.

**29.9 — Site-scoped permissions.** R6 was stated for sites as a class. Is per-site authority ("Manager A may edit Site 1 but not Site 2") also required? `AccessGrant.site_ids` exists (`resolver.py:167`) and is populated by nothing. **REQUIRES DECISION.** *Recommendation:* out of scope for A–Q; the permission-level distinction must land first.

**29.10 — DVR entity.** Reaffirmed as deferred by three prior reports. No new evidence found this pass to overturn it. **REQUIRES DECISION** for a future stage only.

---

## 30. Final readiness verdict

**NOT READY.**

**What is genuinely ready.** The identity and permission-override core is well built, carefully reasoned, tested, and correct on every property in §26. The camera domain service — validation, disabled-by-default, the retire-or-refuse discipline, the zone-assignment history — is among the better-designed code in either repo. The RTSP runtime's redaction discipline is real. That work should not be re-litigated.

**Why it is not ready.**

1. **The requirement that motivated eight stages cannot be demonstrated.** Two same-role managers with different site authority is not expressible, because the permission vocabulary has no site permissions (§5). This is INCORRECT, not partial.
2. **Every account created through the shipped administration API is silently broken.** No `AccessGrant`, therefore no cameras, therefore an empty product with no error (§7.4).
3. **Camera onboarding through the product cannot produce a working camera.** The form omits `host`, and the runtime skips any camera without one (§13.3, §14).
4. **A camera credential can be a plaintext password, returned in API responses to six roles and written to the longest-retained table in the system** (§15.2).
5. **"Multi-organization" is not real.** No CRUD, no operator boundary, and a perception and camera runtime hard-wired to `default_tenant_id` (§6).
6. **Organization lifecycle does not reach cameras, analysis or streaming** — the explicit question asked, answered "no" with citation (§16.2).
7. **`/admin` is the giant everything-admin page the requirement ruled out** (§10.2).

**On the prior stages' claims.** Stages 5–8 report their scope accurately in the small: the routes exist, the tests pass, the overrides compose correctly. The over-claim is in the framing — a working override engine over an incomplete vocabulary was presented as the identity-and-access feature, and a camera-registration modal missing the connection host was presented as camera administration. Both are slices. Saying so is the point of this document.

**Corrections that must land before any further feature work:**
- **Stage A + B** — the permission vocabulary and the site/zone gates. Nothing about R6 is real until these exist.
- **Stage C** — the `AccessGrant` defect. It makes the shipped user-administration API unable to produce a usable account.
- **Stage D** — the `literal:` credential leak. It is a security defect and independent of everything else.

Stages E–Q may then proceed in the order given, with §29.2 (SUSPENDED cascade) decided before Stage N and §29.3 (operator identity) before Stage O.

---

**Files touched to produce this document:** only this file — `unityworks-vision-ai-backend/docs/architecture/FINAL_ADMINISTRATION_CONTROL_PLANE_REQUIREMENTS_RECONCILIATION_AND_IMPLEMENTATION_ROADMAP.md`. No production code, migration, route, permission, role, UI component, test, configuration, or perception / detection / tracking / VLM / vision_os / compliance / observation / incident logic file was created, modified, or executed in either repository.
