# Multi-Organization Administration — Domain Architecture, Stage 1

**Status:** Read-only architecture design, except for one narrow, independently-confirmed code fix (Part C / §19). Every claim about existing behaviour is labelled **CONFIRMED** (re-verified by reading the cited lines myself in this pass), **INFERRED** (a reasonable conclusion not directly stated in code), **PROPOSED** (a recommendation, not existing behaviour), or **UNKNOWN** (not traced this pass). This document supersedes nothing in `ADMINISTRATION_MULTI_ORGANIZATION_DISCOVERY_AND_USE_CASES.md`; it re-verifies its load-bearing claims and extends them into a staged domain-architecture design.

---

## 1. Executive summary

The backend is already a multi-tenant system in its data model and query discipline: `organization_id` is present on every durable domain table and every traced query is constructed from the authenticated caller's `access.tenant_id`, never from client input (**CONFIRMED**, §3, §18). The one real gap this audit set out to check — camera creation accepting a `restaurant_id` without verifying it belongs to the caller's organization — is **CONFIRMED** exactly as the prior report described, and has been fixed as the one permitted code change (§19). Everything else in this document is a design proposal for Stage 2+: there is currently no Organization CRUD, no first-class DVR entity, no user-invitation mechanism, and no cross-tenant "platform operator" role — `super_admin` is tenant-scoped by construction (**CONFIRMED**, §3). The recommended future domain model keeps the existing `Organization → Site → Zone → Camera` shape, adds Organization lifecycle and provisioning as new administrative surface (not a schema change), and treats DVR as an additive, lightweight grouping concept rather than a new required entity (§5, §6).

## 2. Current architecture as actually implemented

- **Backend:** FastAPI + SQLAlchemy 2.0 async ORM. `app/infrastructure/database.py` provides `create_all_for_tests`, used by the test suite (`tests/app/conftest.py:16`, **CONFIRMED** by reading the fixture); production schema is expected to go through Alembic per that module's own docstring (**INFERRED** — an `alembic/` directory was not re-searched this pass; treat as **UNKNOWN**, consistent with the prior report).
- **Domain layer** (`app/domain/models.py`): `Restaurant` (site, lines 55–77), `Zone` (79–100), `Camera` (103–178), `CameraZoneAssignment` (181+), plus `EvidenceRecord`, `Incident`, `FrameRecord`, `AuditEvent` (not re-read line-by-line this pass; citations for these follow the prior report, re-used as **INFERRED** unless independently re-checked below).
- **Identity layer** (`app/users/models.py`): `Organization`, `User`, `RoleAssignment`, `AccessGrant` — kept structurally separate from the domain tables.
- **Authorization** (`app/authorization/model.py`): a closed `Role` enum (line 55 area), a closed `Permission` enum, a static `ROLE_PERMISSIONS` map (lines 200–366, **CONFIRMED** by direct read this pass), and `AccessDecision` (lines 428–476, **CONFIRMED**) — built once per request, `tenant_id` mandatory and validated non-empty in `__post_init__` (lines 444–451, **CONFIRMED**).
- **API** (`app/api/*.py`): `administration.py` (restaurants/zones/users), `product.py` (cameras/incidents/evidence/frames/audit), `reports.py`, `evaluation.py`, `wall.py` (live streaming tickets), `websocket.py`, `integrations.py`, `patron.py`, `analytics.py`, `devtools.py`. All confirmed present by directory listing this pass.
- **Frontend** (`unityworks-vision-ai-frontend`): React + TypeScript, a typed `PERMISSIONS`/`ROLES` mirror of the backend enums in `src/app/permissions/permissions.ts` (**CONFIRMED**, full file read this pass — it states explicitly, lines 9–12, that the frontend guard "is UX... not a security boundary"), route guards (`PermissionGate`, imported at `src/features/administration.tsx:34`, **CONFIRMED**), and a design system in `src/shared/ui/product.tsx` exporting `PageIntro`/`SectionRule`/`Region`/`Plane`/`Figure`/`DataTable`-shaped components (**CONFIRMED** present by grep this pass; not re-read line-by-line).

## 3. Current tenant model

- `AccessDecision.tenant_id: str` is a single mandatory field (`app/authorization/model.py:437`), raising in `__post_init__` if empty (lines 444–451) — **CONFIRMED** by direct read. There is no field or code path anywhere in this dataclass that represents "more than one tenant" or "no tenant / cross-tenant."
- `ROLE_PERMISSIONS[Role.SUPER_ADMIN] = frozenset(Permission) - {Permission.MANAGE_PATRON_ID}` (`app/authorization/model.py:213`, **CONFIRMED** by direct read). This grants every permission except patron re-identification — but every one of those permissions is checked against `AccessDecision.permissions`, which is itself scoped by the single `tenant_id` on the same object. **CONFIRMED**: `super_admin` today is the most powerful role *within one organization*, not a platform-wide role. There is no code path constructing an `AccessDecision` for a tenant other than the authenticated user's own `organization_id` row (not re-traced end-to-end into `app/auth/service.py` this pass, but the invariant is enforced structurally by `AccessDecision` accepting exactly one `tenant_id`, and `app/auth/service.py:135` was seen this pass comparing `decision.tenant_id != claims.tenant_id`, consistent with a single-tenant-per-session model — **CONFIRMED** for that comparison, **INFERRED** for the full claims-issuance path).
- Every API route inspected this pass (`app/api/administration.py`, `app/api/product.py`, `app/api/analytics.py`, `app/api/patron.py`, `app/api/wall.py`, `app/api/websocket.py`, `app/api/devtools.py`, `app/api/integrations.py`) constructs its query or scope object from `access.tenant_id`, never from a request body/path/query field named `organization_id` or similar (**CONFIRMED** by grep across `app/api/*.py` this pass — 56 occurrences of `organization_id == access.tenant_id` / `organization_id=access.tenant_id` patterns, sampled and spot-read in `administration.py`, `product.py`, `analytics.py`, `patron.py`, `wall.py`, `devtools.py`, `websocket.py`).
- Frontend: `PERMISSIONS`/`ROLES` are a typed mirror only; the module's own docstring states the frontend is "written on the assumption that anyone can bypass" the UI guard (`permissions.ts:9-12`, **CONFIRMED**). Enforcement is entirely server-side.

**Conclusion (CONFIRMED):** today's tenant model is single-tenant-per-session, with no cross-tenant read path anywhere in the traced code, and no permission or role concept for "sees more than one organization."

## 4. Existing organization→site→zone→camera relationships

Re-verified this pass:

- `Restaurant.organization_id → organizations.id`, FK, `ondelete=CASCADE`, `NOT NULL` (`app/domain/models.py:65-67`, **CONFIRMED** by direct read).
- `Zone.restaurant_id → restaurants.id`, FK, `ondelete=CASCADE`, `NOT NULL`; **Zone has no `organization_id` column of its own** (`app/domain/models.py:90-93`, **CONFIRMED**). Tenancy is reachable only by joining through `Restaurant`.
- `Camera.organization_id → organizations.id` and `Camera.restaurant_id → restaurants.id`, both FK `ondelete=CASCADE`, `NOT NULL` (`app/domain/models.py:124-129`, **CONFIRMED**). `Camera.zone_id → zones.id`, FK `ondelete=SET NULL`, nullable (`models.py:130-132`, **CONFIRMED**).
- `Camera.camera_key` unique per organization (`UniqueConstraint("organization_id", "camera_key")`, `models.py:118`, **CONFIRMED**). `Restaurant.slug` unique per organization (`models.py:60`, **CONFIRMED**).
- `CameraZoneAssignment` is an append-only interval table keyed by `organization_id` + `camera_key` as plain strings, deliberately not FK'd to `Camera`/`Zone` so history survives rename/delete (`models.py:181-260`, **CONFIRMED** by direct read of the docstring and table definition this pass).
- `Camera.organization_id` is redundant with `Restaurant.organization_id` (derivable via the FK join) but stored directly for query performance — no CHECK constraint enforces agreement between the two (**CONFIRMED absence** — no such constraint appears in the `Camera.__table_args__` read this pass, `models.py:117-121`). This is exactly the condition that made the camera-ownership gap possible (§19).
- Zone creation already guards this: `_restaurant_in_tenant(session, access.tenant_id, restaurant_id)` is called before insert (`app/api/administration.py:325`, **CONFIRMED** by direct read this pass — the comment at line 323-324 reads "Checked before insert so a zone can never be attached to another organization's restaurant by naming its id"). Camera creation, before this fix, did not call it (§19).

## 5. DVR/NVR architecture analysis

- **No DVR entity exists.** No table, column, or API resource named `DVR`/`NVR` was found in `app/domain/models.py` (**CONFIRMED** — the `Camera` class, fully read this pass, has no `dvr_id` or equivalent). "DVR" appears only as a naming convention: the `Camera` docstring says "the DVR has 16 channels" (`models.py:106`, **CONFIRMED**) and `app/domain/cameras.py`'s module docstring repeats it (lines 3-6, **CONFIRMED**).
- RTSP/credential fields live directly on `Camera`: `host` (default `""`), `rtsp_port` (default 554), `channel` (int), `stream_type` ("main"/"sub"), `username`, `credential_ref` (**CONFIRMED**, `models.py:143-149`, all fields read directly this pass).
- **Credentials are never stored as values.** `credential_ref` holds a pointer string such as `env:CCTV_PASSWORD` (**CONFIRMED** by the `Camera` class docstring, `models.py:111-113`, and `cameras.py:8-13`, which states explicitly: "A database dump must not be a credential dump"). The actual `SecretProvider`/resolution mechanism was not re-read this pass (**UNKNOWN** — carried over from the prior report, not independently re-verified; flagged as a follow-up before any DVR-level credential centralization is designed).
- Cameras sharing a DVR are only informally grouped by identical `host`/`rtsp_port` — there is no `dvr_id` or grouping key (**CONFIRMED absence**, same read as above).
- Answers to the specific questions posed:
  - **(a) Is a first-class DVR/NVR entity justified now?** **PROPOSED: No.** The isolation and query discipline gap this audit was asked to check (camera ownership) was administrative, not structural (§19), and no current use case in the traced code (bulk DVR registration, per-DVR health, centralized credential rotation) exists to justify the migration cost.
  - **(b) Can cameras keep standalone RTSP config during a migration window?** **PROPOSED: Yes** — nothing forces a schema change; `host`/`rtsp_port`/`channel`/`credential_ref` staying on `Camera` is the lowest-risk path and preserves every existing `org-unityworks` row unchanged.
  - **(c) Can one DVR safely serve multiple sites?** **PROPOSED:** architecturally yes if a `DVR` entity is ever introduced, since nothing in the current schema ties a physical unit to exactly one `Restaurant` — but this is speculative since no such entity exists today (**UNKNOWN** in current behaviour, since there is nothing to test).
  - **(d) Should DVR credentials be centralized or stay per-camera?** **PROPOSED:** stay per-camera for Stage 1–4. `credential_ref` already supports different channels on one physical DVR using different credentials (documented explicitly by the `stream_type`/per-channel design), so per-camera credentials are not a limitation being worked around today.
  - **(e) Are current secret-storage mechanisms sufficient for centralization?** **UNKNOWN** — the `SecretProvider` resolution path was not read this pass; this must be verified before any centralization design, not assumed.
  - **(f) How would migration happen without breaking existing `org-unityworks` cameras?** **PROPOSED:** additive only — a nullable `dvr_id`/`dvr_group_id` backfilled from distinct `(organization_id, restaurant_id, host, rtsp_port)` tuples, with `host`/`rtsp_port` remaining on `Camera` as the source of truth during a transition window, exactly mirroring the prior report's Option 3.
  - **(g) How should DVR connectivity health be represented without exposing credentials?** **PROPOSED:** a runtime/health signal per camera or per shared-host group (reachable/unreachable, last-seen), never carrying `credential_ref` or any resolved secret in its payload — consistent with the existing separation between `Camera.enabled`/`analysis_enabled` (configuration) and health (a query-time signal, not a stored column) already established for cameras (`models.py:151-169`, **CONFIRMED** as the existing precedent for keeping operational state out of the credential-bearing row).
- **No DVR table is created by this document or by the code change in §19** — consistent with the guardrail.

## 6. Recommended future domain model

**PROPOSED**, following the same reasoning as the prior discovery report and independently re-affirmed by this pass's re-reading of the schema: keep `Organization → Site (Restaurant) → Zone → Camera` exactly as it is structurally. Add three things, none of which are schema changes to the domain tables:

1. **Organization administration surface** (CRUD + lifecycle state) on the existing `Organization` table in `app/users/models.py` — this table already exists; today it has no API surface (§9).
2. **User provisioning** — currently entirely read-only (`GET /api/v1/users`, `write_available: false`, `app/api/administration.py` region cited in the prior report at lines 435-440, not re-read line-for-line this pass but consistent with the frontend's verbatim rendering of that message, `administration.tsx:438` region, **CONFIRMED** present by grep this pass).
3. **A lightweight, optional DVR grouping key** (§5), added opportunistically, not as a Stage 1–4 dependency.

This is Option 1 (formalized current shape) from the prior report, re-affirmed rather than re-derived from scratch, because independent re-reading of the schema this pass found nothing to contradict it: the isolation discipline is strong (§3, §18) and the actual gap was a single missing ownership check (§19), not a structural flaw.

## 7. Proposed entity relationship diagram (text form)

```
Organization  [lifecycle: active | suspended | archived — PROPOSED, §8]
 ├── User (organization_id FK CASCADE)
 │    ├── RoleAssignment (user_id FK CASCADE)
 │    └── AccessGrant (user_id FK CASCADE)
 ├── Restaurant "Site" (organization_id FK CASCADE)
 │    └── Zone (restaurant_id FK CASCADE — no organization_id column)
 ├── Camera (organization_id FK CASCADE, restaurant_id FK CASCADE, zone_id FK SET NULL)
 │    [PROPOSED, additive, optional: dvr_group_id — no FK, metadata only, §5/§6]
 ├── CameraZoneAssignment (organization_id column, plain-string camera_key/zone_id/restaurant_id — append-only)
 ├── EvidenceRecord / Incident / FrameRecord (organization_id FK CASCADE, various nullable child refs)
 └── AuditEvent (organization_id column — NOT FK-enforced, carried over from the prior report, not independently re-verified this pass — UNKNOWN/INFERRED)
```

No new entities are introduced in this diagram beyond what the prior report already established; the only addition is the bracketed lifecycle/DVR-grouping notes, both explicitly marked PROPOSED.

## 8. Organization lifecycle design (PROPOSED)

No lifecycle state exists on `Organization` today beyond its bare row (**INFERRED** — the `Organization` model was not re-read line-by-line this pass; carried from the prior report's finding that Organization has no CRUD API and only `id`/`name`/`slug` are set by the CLI). Proposed states, defined operationally:

- **Active** — full read/write, as today's single `org-unityworks` behaves.
- **Suspended** — PROPOSED meaning: existing users can still authenticate (so an admin can review data) but every `MANAGE_*`-gated write is refused, and no new perception/analysis work is scheduled for that tenant's cameras (existing evidence/reports remain readable). This is a reversible, non-destructive state for billing holds or incident response.
- **Archived** — PROPOSED meaning: no login permitted; data retained per retention policy but not reachable through any live API; a step below deletion, distinct from Suspended in that it is not expected to be reversed in the normal course of business.

**Explicitly separated from existing behaviour:** none of these three states exist in the schema or enforcement path today. Building them requires (a) an `Organization.status` column, (b) an authorization-layer check (most naturally in `AccessDecision` construction or the `requires()` dependency) that refuses non-read operations for a non-Active tenant, and (c) a decision on whether "suspended" should also gate live camera streaming (§13 of the prior report already established that Restaurant/Camera `is_active` today has **no cascading effect** on `Zone`/`Camera` rows — `app/api/administration.py:200-239`, not re-read this pass but consistent with this pass's reading of `update_restaurant`, lines 200-239, which sets only `restaurant.is_active` with no cascade). Organization-level suspension would need to decide explicitly whether it cascades, unlike today's Restaurant-level flag, which does not.

## 9. Organization provisioning use cases (PROPOSED)

- **Today:** no API exists. The only creation path is `scripts/manage.py create-user --org <id>` (**INFERRED**, carried from the prior report — not re-read this pass, but consistent with there being zero `POST /api/v1/organizations`-shaped route found in `app/api/administration.py`'s route list this pass, which covers only `/restaurants`, `/zones`, `/users` read).
- **PROPOSED minimal Stage 3 provisioning path:** `POST /api/v1/organizations` gated by a new permission (name TBD — a decision requiring product-owner approval, §23) held only by a genuinely cross-tenant role. Required: `name`, `slug` (validated unique, immutable, following the `Restaurant.slug` precedent at `models.py:60`). No email/invitation infrastructure exists in this backend (**CONFIRMED absence** — no mail/SMTP/notification-delivery module was found under `app/` in any file read or grepped this pass beyond the in-app `NotificationEvent` dataclass in `app/domain/notifications.py`, which builds a payload from incidents, not user invitations). **Do not assume an invite flow exists — it must be built or explicitly deferred.**
- **PROPOSED smallest viable admin-provisioning path**, given no invite infrastructure: an admin-set, one-time password issued out-of-band (e.g. shown once in the response to a platform operator, who relays it manually), with the account forced to change it on first login if such a mechanism exists (**UNKNOWN** — `force_password_change`-shaped logic was not found or searched for this pass; treat as a build item, not an assumption).
- **What happens if org creation succeeds but admin provisioning fails:** PROPOSED — the two must be one transaction (or the org creation route must itself require and validate the first admin's fields in the same request) so an Organization row is never left with zero users; today's schema does not prevent an orgless `Organization` row, but that is an operational dead-end an operator cannot recover from through the product (no create-user API today, §9).
- **Can an org exist with zero sites?** **PROPOSED: yes**, consistent with the fact that `Restaurant` creation is already a fully separate act from Organization creation (§4) and nothing in the schema requires a non-empty `restaurants` relationship.
- **Can cameras/DVRs exist before their parent (site) exists?** **CONFIRMED: no**, by the FK constraints already in place — `Camera.restaurant_id` is `NOT NULL` with an FK to `restaurants.id` (`models.py:127-129`), so a camera cannot be inserted without an existing Restaurant row, and this pass's fix in §19 makes that restaurant additionally have to belong to the caller's own organization.
- **Audit:** the prior report found `scripts/manage.py`'s CLI org-creation path writes no `AuditTrail` record (**INFERRED**, carried over, not re-read this pass). Any product-level `POST /api/v1/organizations` must call `AuditTrail.record(...)`, following the pattern already used by `create_zone`/`create_camera` (`administration.py:331-340`, `product.py:105-116`, both **CONFIRMED** by direct read this pass).

## 10. Site provisioning use cases

**Today, fully implemented (CONFIRMED):** `POST /api/v1/restaurants`, gated `MANAGE_ORGANIZATION`. Required: `name`. Optional: `slug` (derived from name), `timezone` (default UTC), `is_active` (default true). `organization_id` in the request body is ignored — tenancy comes from `access.tenant_id` (re-verified by this pass's read of `app/api/administration.py` and by the existing test `test_a_restaurant_cannot_be_created_into_another_organization`, `tests/app/test_administration.py:85-101`, **CONFIRMED** by direct read this pass). No delete route; `is_active` does not cascade to child `Zone`/`Camera` rows (§8). This needs no Stage 2+ change beyond whatever cross-tenant provisioning wraps around it.

## 11. DVR/NVR provisioning use cases

**Today: no such use case exists** (§5) — registering DVR infrastructure means registering each camera channel individually via `POST /api/v1/cameras`, giving matching cameras the same `host`/`rtsp_port`. **PROPOSED, deferred:** if a lightweight `dvr_group_id` is added (§5f), a provisioning use case would be "register N channels against one physical unit in one call" — explicitly not recommended for Stage 1–4 (§22).

## 12. Camera provisioning use cases

**Today (CONFIRMED, re-verified this pass):** `POST /api/v1/cameras`, gated `MANAGE_CAMERAS` (`app/api/product.py:75`). Required: `restaurant_id`, `camera_key`, `channel`. Optional: `name`, `host`, `rtsp_port` (554), `stream_type` ("sub"), `username`, `credential_ref`, `analysis_fps` (4.0), `purpose`, `zone_id`. Created **disabled** by default — the route forces `enabled=False` regardless of payload (`product.py:100`, **CONFIRMED**). A `CameraZoneAssignment` interval is opened at creation even when `zone_id` is `None` (`app/domain/cameras.py:104-116`, **CONFIRMED** by direct read this pass).

**As of this pass, camera creation now verifies `restaurant_id` belongs to the caller's organization before insert** (§19) — this closes the one confirmed gap and is the only behavioural change to this use case.

## 13. Zone assignment use cases

Unchanged by this audit. **CONFIRMED, re-verified this pass:** `POST /api/v1/zones` already checks `_restaurant_in_tenant` before insert (`administration.py:322-325`); `PATCH /cameras/{key}` with a new `zone_id` closes the current `CameraZoneAssignment` interval and opens a new one (**INFERRED**, carried from the prior report at `cameras.py:154-171`, not re-read line-by-line this pass but consistent with the docstring at `models.py:212-220` on append-only, never-back-dated intervals, which was read directly this pass).

## 14. Organization administrator lifecycle

**PROPOSED**, since no admin-lifecycle mechanism exists today beyond `RoleAssignment`/`AccessGrant` rows: an org admin is created (via whatever provisioning path §9 settles on), can be suspended (role/grant removed without deleting the `User` row, preserving audit attribution), and — if the organization itself is archived (§8) — loses login entirely without the underlying `User`/`AuditEvent` rows being deleted, consistent with `AuditEvent`'s apparent design intent to survive its parent (carried from the prior report's finding that `AuditEvent.organization_id` has no FK constraint, `models.py:534` per that report — **not re-verified this pass**, flagged **UNKNOWN** pending re-confirmation before this is relied upon for a deletion/archival design).

## 15. Role and permission model recommendations

**PROPOSED, smallest safe change (not implemented):** introduce a new permission (e.g. `MANAGE_PLATFORM` or `CREATE_ORGANIZATION` — exact name a product-owner decision, §23) held by a new role that sits *outside* the existing `Role` enum's per-tenant assumption, rather than redefining `super_admin`. Justification: `AccessDecision.tenant_id` is a single mandatory string (§3) — making `super_admin` cross-tenant would require either (a) a sentinel "all tenants" value threaded through every one of the ~56 query sites found in §3, which is a large blast radius for a single role change, or (b) a parallel code path for platform operators that never touches `AccessDecision.tenant_id` at all. **(b) is the safer shape**: a platform-operator concept that talks only to the `Organization` table (create, list, suspend) and never to any tenant-scoped resource (cameras, incidents, evidence), so it never needs to construct a `Scope`/`AccessDecision` with more than one tenant in it. This preserves every existing tenant-isolation guarantee found in §18 untouched. **No role or permission enum was changed to produce this document**, per the guardrails.

## 16. Full administration information architecture (map, not routes)

Re-verified this pass against `administration.tsx` and the design-system exports:

- **Current dead end (CONFIRMED, re-verified):** `PermissionGate` gates the "Add a site"/"Add a zone" forms behind `manageOrganization` (`administration.tsx:250`, `:317`), and the Users table renders the server's `write_available: false` message verbatim (`:438` region, **CONFIRMED** present by grep). No Organization-level page exists in the frontend (**INFERRED**, carried from the prior report's read of `AppRouter.tsx`; not re-read this pass).
- **PROPOSED information architecture**, composed from existing design-system primitives only (`PageIntro`, `SectionRule`, `Region`, `Plane`, `Figure`, `DataTable` — all confirmed present in `src/shared/ui/product.tsx` by this pass's grep):
  - **Organizations** (top-level, visible only to the platform-operator role of §15) — a `PageIntro` + `DataTable` of organizations, each row showing lifecycle state (§8) and linking into that organization's own Administration view. Actions: create, suspend/reactivate, archive. No cross-tenant camera/incident/evidence view — a platform operator manages organizations, not their contents (§15's isolation argument).
  - **Administration** (per-organization, today's `/admin`, `org_admin`-gated) — unchanged shell, extended with drill-down: a site detail region showing that site's zones and cameras inline (mirroring the prior report's proposal), each camera row linking to the existing camera detail page.
  - **Cameras** (today's `/cameras`) — unchanged in scope; its create form's `restaurant_id`/`zone_id` fields become selects sourced from the same data Administration already fetches, closing the free-text gap noted in the prior report (not independently re-verified this pass, carried as **INFERRED**).
  - **Users** — stays read-only until §9's provisioning path exists; the `write_available: false` contract must not change ahead of that mechanism shipping (§21).

Page-state table (permission / scope / actions / navigable relationships), per requirement:

| Page state | Intended user | Permission | Org scope | Actions | Navigates to |
|---|---|---|---|---|---|
| Organizations list | Platform operator | new platform permission (§15) | cross-org (list only) | create, suspend, archive | one org's Administration |
| Administration (site/zone) | org_admin | `manage_organization` (existing) | own org only | create/edit site, create/edit zone | Cameras (filtered), Users (read-only) |
| Cameras | org_admin, restaurant_manager (view) | `manage_cameras` / `view_cameras` (existing) | own org only | create/edit/enable/retire camera | Administration (site/zone) |
| Users | org_admin (read only today) | `view_users` (existing) | own org only | none (until §9 ships) | none |

## 17. Proposed page/route map (proposal only — no route added)

```
/organizations                     [NEW — platform operator only, §15]
  /organizations/:id                → that org's admin shell below

/admin                              [existing, org-scoped]
  /admin/sites/:id                  [NEW drill-down, org-scoped]

/cameras                            [existing — creation form gains selects, no new route]
/cameras/:cameraKey                 [existing — gains back-links to /admin/sites/:id]

/users                              [existing, read-only until §9 ships]
```

No route was added to the codebase; this is a design proposal only, per the guardrails.

## 18. Tenant-boundary security audit table

Re-verified this pass by tracing actual queries, not endpoint decorators. Classification: **SAFE** (query/scope is always constructed from `access.tenant_id`), **NEEDS FIX**, **NOT VERIFIED** (not read this pass).

| Module | Classification | Evidence |
|---|---|---|
| Organizations | SAFE (no write API exists to attack) | No `POST/PATCH /organizations` route found in `app/api/administration.py`'s route list this pass — nothing to leak because nothing is writable through the product. |
| Sites / restaurants | SAFE | `Restaurant.organization_id == access.tenant_id` constructed directly; `_restaurant_in_tenant` (`administration.py:242-259`) 404s on mismatch, re-read this pass. |
| Zones | SAFE | No own `organization_id`; scoped via join to `Restaurant.organization_id == access.tenant_id` (`administration.py:288-291`, re-read this pass); create path guarded by `_restaurant_in_tenant` (`:325`). |
| Cameras — read/update/list | SAFE | Every `CameraService` call in `product.py` passes `organization_id=access.tenant_id` (lines 65, 87, re-read this pass). |
| Cameras — create | **FIXED this pass** (was NEEDS FIX) | See §19. |
| DVR/RTSP config | SAFE (no independent entity/route to attack) | No DVR table/route exists (§5); RTSP fields inherit whatever isolation `Camera` already has. |
| Incidents | NOT VERIFIED this pass | Carried from prior report's citation (`app/domain/incidents.py:79,108,215,244,266`) — not re-read this pass; re-verify before Stage 2. |
| Evidence | NOT VERIFIED this pass | Carried from prior report (`app/domain/evidence.py:123,178,201-202`, storage path partitioned by org at `:85`) — not re-read this pass. |
| Frames | NOT VERIFIED this pass | Carried from prior report; `FrameRecord.organization_id` presence assumed FK-enforced, not re-read this pass. |
| Live wall / streaming tickets | SAFE | `mint_ticket`/`verify_ticket` bind `tenant_id` into the HMAC payload (`app/api/wall.py:62-90`, re-read this pass — signature confirmed at lines 62,70,75,84); camera lookups pass `organization_id=access.tenant_id` (`:113,169,278`, re-read this pass). |
| Reports | NOT VERIFIED this pass | Carried from prior report's citation of `app/reporting/sources.py` filtering on `request.organization_id`; not re-read this pass. |
| Observations | NOT VERIFIED this pass | Carried from prior report (`app/domain/observations.py:93-100` building a Vision OS `Scope` from `access.tenant_id`); not re-read this pass. |
| Staff hygiene / compliance modules | NOT VERIFIED — query-level enforcement not traced | `app/domain/modules.py` (re-grepped this pass): every table (`PeopleCountBucket`, `DemographyBucket`, `DiningTableEvent`, `BoardPolicy`, `BoardEvent`, `DishEvent`, `PatronToken`, `PosConnector`, `PosRun`) carries an `organization_id` column with an index, confirming schema-level tenancy, but the query-level `WHERE` construction per endpoint was not traced this pass (same gap the prior report flagged — still the top follow-up item). |
| Notifications | NOT VERIFIED — likely SAFE by construction, but not traced | `app/domain/notifications.py:59` — `NotificationEvent.organization_id` is populated from `incident.organization_id` at construction (line 228, re-read this pass), so it inherits Incident's isolation if Incident is SAFE, but Incident itself is NOT VERIFIED this pass (see above) — treat the whole chain as NOT VERIFIED. |
| POS / patron integrations | SAFE (route-level, re-verified this pass) | `app/api/integrations.py:70` passes `organization_id=access.tenant_id` on write; line 110 filters `PosConnector.organization_id == access.tenant_id` on read — both directly re-read this pass. `app/api/patron.py:99` similarly passes `organization_id=access.tenant_id`. |
| Users / RoleAssignment / AccessGrant | NOT VERIFIED this pass | Carried from prior report (`administration.py:398,409-411`); not re-read this pass. |
| Audit log | NOT VERIFIED this pass, one known looseness | Carried from prior report: `AuditEvent.organization_id` has no FK constraint (`models.py:534` per that report) — schema-level looseness, not a query-level leak; not independently re-verified this pass. |
| Analytics | SAFE | `app/api/analytics.py` re-grepped this pass: `organization_id=access.tenant_id` at lines 100, 160, 231, 298, 365 — all five write/read sites scoped from the session. |
| Devtools / live sessions | SAFE | `app/api/devtools.py` re-grepped this pass: every `TenantId(access.tenant_id)` / `tenant_id=access.tenant_id` construction (lines 47, 105-109, 206, 224, 349, 357) is session-derived. |
| Websocket | SAFE | `app/api/websocket.py:78,186` — `live.visible(tenant_id=access.tenant_id, ...)`, re-read this pass. |
| Search / export endpoints | NOT VERIFIED — no dedicated search endpoint found | Grepped this pass for `def.*search`/`def.*export` across `app/api` and `app/domain`; only `app/reporting/sources.py` (exports) was found, and its per-source `organization_id` filtering was not re-read this pass (carried from prior report). |
| Perception runtime / camera session construction | NOT VERIFIED this pass | Not read this pass; carried as an open item from the prior report's own admission that `app/vision/manager.py` was out of budget. |

## 19. Camera ownership-gap finding and repair status

**CONFIRMED, independently re-verified this pass**, exactly as the prior report described:

- `app/api/product.py` (`create_camera`, originally lines 75-117) passed `restaurant_id=str(payload.get("restaurant_id", ""))` straight to `CameraService.create` with no check that the restaurant belongs to `access.tenant_id`.
- `app/domain/cameras.py` (`CameraService.create`, lines 49-117, re-read this pass in full) never queries `Restaurant` at all — it only checks `camera_key` uniqueness within the given `organization_id` (line 82-84) and then inserts, trusting `restaurant_id` unconditionally.
- By contrast, zone creation already guards this: `app/api/administration.py:322-325` calls `_restaurant_in_tenant(session, access.tenant_id, restaurant_id)` before insert, with the comment "Checked before insert so a zone can never be attached to another organization's restaurant by naming its id."
- **Attack scenario confirmed:** an `org_admin` of Organization A, holding `MANAGE_CAMERAS`, could submit `POST /api/v1/cameras` with a `restaurant_id` belonging to Organization B. The resulting `Camera` row would have `organization_id = A` (from `access.tenant_id`, never trusted from the client) but `restaurant_id` pointing into B's `restaurants` table — no FK violation occurs because the FK only validates the row's existence, not its owning organization.

**Fix applied (the one permitted code change):**

- `unityworks-vision-ai-backend/app/api/product.py` — imported `_restaurant_in_tenant` from `app.api.administration` and call it in `create_camera` before `CameraService.create`, mirroring the zone-creation pattern exactly (same helper, same 404-not-403 semantics — "it exists but is not yours" is not disclosed, consistent with `_restaurant_in_tenant`'s own docstring at `administration.py:246-249`). Diff:

  ```diff
  + from app.api.administration import _restaurant_in_tenant
  ...
       service = camera_domain.CameraService(session)
       audit = AuditTrail(session)

  +    restaurant_id = str(payload.get("restaurant_id", ""))
  +    # Checked before insert so a camera can never be attached to another
  +    # organization's restaurant by naming its id — the same guard zone
  +    # creation already applies (`app/api/administration.py:325`).
  +    await _restaurant_in_tenant(session, access.tenant_id, restaurant_id)
  +
       camera = await service.create(
           organization_id=access.tenant_id,
  -        restaurant_id=str(payload.get("restaurant_id", "")),
  +        restaurant_id=restaurant_id,
  ```

- `unityworks-vision-ai-backend/tests/app/test_persistence.py` — updated the two existing camera-creation-through-the-API tests (`test_a_camera_created_through_the_api_starts_disabled`, `test_enabling_a_camera_gets_its_own_audit_action`) to create a real `Restaurant` via `POST /api/v1/restaurants` first, since they previously posted a `restaurant_id` (`"rest-01"`) that never corresponded to an actual row — a gap the fix now correctly rejects. Added two new tests:
  - `test_a_camera_cannot_be_attached_to_another_organizations_restaurant` — an `org-other` caller attempts to create a camera against `org-test`'s restaurant; asserts `404` and asserts no `camera.created` audit event was written for `org-other`.
  - `test_a_camera_created_into_a_nonexistent_restaurant_is_rejected` — asserts `404` for a `restaurant_id` that doesn't exist at all (this was previously silently accepted).
  - No privileged/platform-level bypass test was added, because none exists in the current codebase (§3, §15) — inventing one would violate the task's explicit instruction not to invent a bypass that isn't there.

**Test results:** `pytest tests/app/test_persistence.py tests/app/test_administration.py -v` → **59 passed, 0 failed** (full run, both files, including the two updated tests and the two new tests). Command and result reproduced below.

```
collected 59 items
tests\app\test_persistence.py .......................................... [ 71%]
...                                                                      [ 76%]
tests\app\test_administration.py ..............                          [100%]
============================= 59 passed in 37.74s =============================
```

**Scope check:** `git status`/`git diff --stat` after the fix show exactly two modified files — `app/api/product.py` (+9/-1 net lines) and `tests/app/test_persistence.py` (+56/-3 net lines) — no other file was touched.

## 20. Migration strategy from existing org-unityworks data

**PROPOSED, no migration required for the fix in §19** — the fix only adds a validation check on the create path; it does not change any column, and no existing `org-unityworks` `Camera` row is affected retroactively (the fix is not applied to historical rows, only to future create calls, per the task's guardrail against a broader rewrite). If a future audit finds that any existing `Camera` row already has a `restaurant_id` disagreeing with its `organization_id` (possible under the pre-fix code, though not confirmed to exist in `org-unityworks`'s actual data — **UNKNOWN**, not queried this pass), that would need a one-time data-integrity sweep before Stage 2, independent of this fix.

For the broader domain-model recommendations (§6): no schema change is proposed for Stage 1–4, so no migration is required. If Organization lifecycle (§8) or a DVR grouping key (§5) are later approved, both are additive nullable columns and do not require backfilling `org-unityworks`'s existing rows beyond a default value (`status = 'active'`, `dvr_group_id = NULL`).

## 21. Backward compatibility risks

- `org-unityworks` must remain reachable by its exact existing `organization_id` string — nothing in this document or the §19 fix changes any existing identifier.
- The §19 fix is strictly more restrictive than before: any code path (script, integration test, or manual API call) that was relying on the previously-unchecked behavior of creating a camera against a nonexistent or cross-org `restaurant_id` will now receive a `404` instead of a `200`. This was true only for a `restaurant_id` that either didn't exist or belonged to another organization — a legitimate same-organization `restaurant_id` continues to work exactly as before (verified by the retained, now-passing `test_a_camera_created_through_the_api_starts_disabled` test, §19).
- The `write_available: false` contract on `GET /api/v1/users` must not change ahead of an actual invitation/reset mechanism shipping (§9) — carried unchanged from the prior report, not touched by this pass.
- `Camera.enabled` default-false-on-create behavior is unchanged by the fix (still forced server-side at `product.py`, now a few lines further down due to the inserted check, but the same force-disable line is present and unmodified — re-verified this pass).

## 22. Things explicitly NOT recommended

- **Do not introduce a first-class `DVR` table now** (§5a) — no current use case in the traced code justifies the migration cost; revisit once real multi-site DVR-grouping UX needs are validated.
- **Do not make `super_admin` cross-tenant by redefining its existing role** (§15) — the blast radius across ~56 tenant-scoped query sites is too large for a role-semantics change; a separate platform-operator concept is the safer shape.
- **Do not build an Organization suspend/archive cascade to child resources** without first deciding whether it should mirror or diverge from today's non-cascading `Restaurant.is_active` behavior (§8) — building it silently one way forecloses the other without product-owner input.
- **Do not assume email/invitation infrastructure exists** (§9) — it does not, and any provisioning design that assumes it will produce a broken user-facing promise, mirroring the exact failure mode the existing `write_available: false` message was written to avoid.
- **Do not treat the "SAFE" rows in §18 as a substitute for tracing the "NOT VERIFIED" rows** — several modules (incidents, evidence, frames, reports, observations, staff-hygiene, users, audit) were not re-read this pass and are carried at the prior report's confidence level only; they must be independently re-verified before Stage 2 ships.

## 23. Decisions that require product-owner approval

1. **What is the actual account-provisioning mechanism** for a new organization's first admin, given no invite/email infrastructure exists today (§9)? Admin-set one-time password? SSO? Something else?
2. **Should a genuinely cross-tenant "platform operator" role be introduced as a new, separate concept** (§15), rather than ever making `super_admin` itself cross-tenant? This is a real fork in the authorization model, not a naming choice.
3. **Should Organizations ever be deletable, or only suspendable/archivable** (§8)? This determines whether `Restaurant.organization_id`'s `ondelete=CASCADE` is ever actually exercised in production.
4. **Does Organization suspension cascade to child resources** (block live streaming? block new analysis? block report generation?), or only block new writes while leaving existing reads/streams untouched (§8)? Today's precedent (`Restaurant.is_active`) does not cascade — the org-level state should not silently pick a different answer.
5. **Is a first-class DVR entity ever wanted for its own sake** (centralized credential rotation, per-unit health, firmware tracking), independent of the "no current use case" finding in §5 — or is per-camera configuration the permanent design?
6. **What permission name and scope should gate Organization creation/suspension** (§15) — a new enum value, and exactly which existing roles (if any) should be grandfathered into holding it?

## 24. Recommended Stage 2 implementation sequence

1. **Close the remaining query-level verification gap** — formally re-trace every row in §18 marked NOT VERIFIED (incidents, evidence, frames, reports, observations, staff-hygiene modules, users, audit) with the same line-level rigor applied to the SAFE rows in this pass. This is a prerequisite for any multi-org rollout, independent of new features.
2. **Decide §23's six product-owner questions** — Stage 3+ cannot be designed concretely until these are answered, particularly #1 (provisioning mechanism) and #2 (platform-operator role shape).
3. **Build Organization CRUD + lifecycle** (§8, §9) as new API surface only — no schema change to `Restaurant`/`Zone`/`Camera`.
4. **Build the platform-operator role and its isolated, organization-management-only permission set** (§15), verified to never construct a multi-tenant `AccessDecision`.
5. **Build User provisioning**, gated on the decision from §23.1.
6. **Frontend**: add the Organizations page and site drill-down (§16, §17) once the APIs from steps 3-5 exist; convert the camera-creation form's free-text `restaurant_id`/`zone_id` fields to selects.
7. **Re-evaluate DVR grouping** (§5, §22) only after real multi-site operational feedback exists — not before.

---

**Files touched to produce this document:**
- `unityworks-vision-ai-backend/app/api/product.py` — the one permitted code change (§19).
- `unityworks-vision-ai-backend/tests/app/test_persistence.py` — test updates/additions for that change (§19).
- `unityworks-vision-ai-backend/docs/architecture/MULTI_ORGANIZATION_ADMINISTRATION_DOMAIN_ARCHITECTURE_STAGE_1.md` — this file.

No migration, permission definition, role definition, route, UI component, or perception/detection/tracking/VLM file was created or modified.
