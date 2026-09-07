# Final Administration, Multi-Organization, Identity, Permission & CCTV Architecture Freeze

**Status:** Read-only, fifth-pass synthesis and freeze document. No production code, migration, route, permission, role, or UI file was modified to produce this document — the only file created is this one. Labels used throughout: **VERIFIED FROM CODE** (file:line, read directly in this pass), **INFERRED**, **RECOMMENDED**, **REQUIRES OWNER DECISION**.

Source documents read in full and treated as authoritative background (cited, not re-derived, except where this pass explicitly re-verifies a load-bearing claim):
- `ADMINISTRATION_MULTI_ORGANIZATION_DISCOVERY_AND_USE_CASES.md` ("Discovery")
- `MULTI_ORGANIZATION_ADMINISTRATION_DOMAIN_ARCHITECTURE_STAGE_1.md` ("Stage 1")
- `IDENTITY_USERS_ROLES_AND_PERMISSION_OVERRIDES_DISCOVERY.md` ("Identity")
- `FINAL_MULTI_ORGANIZATION_IDENTITY_ACCESS_CAMERA_ROADMAP.md` ("Roadmap")

`git status --porcelain` re-checked this pass in all three repositories: `unityworks-vision-ai-backend` shows only the pre-existing modified files from the Stage 1 fix (`app/api/product.py`, `tests/app/test_persistence.py`) plus the four untracked prior-report `.md` files and this new file; `unityworks-vision-ai-frontend` and `vision_os_validation_console` are clean.

---

## 1. Executive summary

The four prior passes established, and this pass re-confirms directly in code, that the backend is a well-isolated multi-tenant system with a strong `Organization → Site (Restaurant) → Zone → Camera` hierarchy (VERIFIED FROM CODE, §3), a sound identity model where `RoleAssignment` already supports many roles per user and many users per role (VERIFIED FROM CODE, re-opened directly this pass, §5), and an authorization chokepoint (`AccessDecision`, built fresh every request) that has never been found to leak across tenants in five passes of tracing (§21). Two items flagged NOT VERIFIED by the Roadmap report are resolved definitively this pass: **Observations is VERIFIED TENANT-SAFE** (the prior grep for the literal string `organization_id` missed that the module scopes by `tenant_id`, not `organization_id` — §22), and **Notifications is VERIFIED TENANT-SAFE** by construction, because there is no notification read/list API at all — only fire-and-forget dispatch derived from an already-tenant-scoped `Incident` row (§23).

This pass's hardest, genuinely new question is whether the previously-recommended `effective_permissions = role_permissions UNION explicit_grants` (allow-only) model is sufficient. It is **not**, in the one specific case (Case C: role grants edit, admin needs one specific user restricted to read-only) that a pure union structurally cannot express. Investigation of the actual `Permission` enum (VERIFIED FROM CODE, §6) shows that Sites, Zones, and Cameras already have **separate READ and WRITE permission constants** (`VIEW_USERS`/`MANAGE_ORGANIZATION` for Sites and Zones; `VIEW_CAMERAS`/`MANAGE_CAMERAS` for Cameras) — so granularity is not the missing piece. The missing piece is that permissions are unioned only from **roles**, and `Role` is a **closed, code-defined Python enum** (VERIFIED FROM CODE, `app/authorization/model.py:46-66`) — an admin cannot create a narrower role at runtime. This tips the final recommendation toward a three-state **INHERIT / GRANT / REVOKE** per-(user, permission) model rather than "just assign a narrower role" (§8–§10), while preserving the existing role-union mechanism unchanged for every case that doesn't need subtraction.

## 2. Existing architecture verified from code

Carried and re-affirmed, not re-derived, from Discovery §2, Stage 1 §2, Identity §2–§3, Roadmap §1 — all four already read code directly. Re-spot-checked this pass: FastAPI + SQLAlchemy 2.0 async backend; domain layer (`app/domain/models.py`): `Restaurant`, `Zone`, `Camera`, `CameraZoneAssignment`, `EvidenceRecord`, `Incident`, `FrameRecord`, `AuditEvent`; identity layer (`app/users/models.py`): `Organization`, `User`, `RoleAssignment`, `AccessGrant`; authorization (`app/authorization/model.py`): closed `Role` enum, closed `Permission` enum, static `ROLE_PERMISSIONS` map, `AccessDecision`. No new architectural facts contradict any prior pass.

## 3. Organization→Site→Zone→Camera domain validation

VERIFIED FROM CODE (carried, re-affirmed a fourth time by Roadmap §3, not re-derived — no contradicting evidence found this pass): the hierarchy is correct as-is. `Restaurant.organization_id`, `Camera.organization_id`, `Camera.restaurant_id` are FK-enforced `NOT NULL CASCADE`; `Zone` has no own `organization_id` and is reachable only by joining through `Restaurant`. Nothing in this pass's targeted re-reads (§20–§22) found any reason to revisit this shape.

## 4. User identity and account architecture

VERIFIED FROM CODE (carried from Identity §2–§3, unchanged): one `User` row = one login identity; `User.email` is unique **per organization**, not globally (`app/users/models.py`, `UniqueConstraint("organization_id", "email")`); no separate Person/Identity table exists by deliberate design ("Four tables, and no more"). Account creation remains CLI-only (`scripts/manage.py create-user`), ungated by any `Permission`, unaudited (MISSING PRODUCT CAPABILITY — no user-write API exists).

## 5. Multiple users per role verification

**RE-VERIFIED DIRECTLY THIS PASS, load-bearing.** `app/users/models.py:108-132`:

```
class RoleAssignment(Base):
    """One role held by one user. A user may hold several."""
    __tablename__ = "role_assignments"
    __table_args__ = (UniqueConstraint("user_id", "role", name="uq_role_assignment"),)
    id: Mapped[str] = ...
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(64), nullable=False)
    granted_at: Mapped[datetime] = ...
    granted_by: Mapped[str | None] = ...
```

VERIFIED FROM CODE: the uniqueness constraint is on the **pair** `(user_id, role)`, not on `role` alone and not on `user_id` alone. This means: (a) one user CAN hold multiple `RoleAssignment` rows (multiple roles), because only the `(user_id, role)` pair must be unique, not `user_id`; (b) many users CAN hold the same `role` string, because nothing constrains `role` to be unique. Both halves of "many-roles-per-user and many-users-per-role" are independently confirmed by this single constraint shape. `permissions_for(roles: frozenset[Role])` (`app/authorization/model.py:369`) unions every held role's permission set — re-confirmed present this pass by direct grep and read.

**Conclusion:** the prior reports' claim is CONFIRMED exactly as stated; no schema change is needed to support many-roles-per-user or many-users-per-role — it already works today.

## 6. Role permission architecture

VERIFIED FROM CODE, `app/authorization/model.py`: `Role` is a closed `enum.Enum` (seven values: `SUPER_ADMIN`, `ORG_ADMIN`, `RESTAURANT_MANAGER`, `KITCHEN_SUPERVISOR`, `HYGIENE_OFFICER`, `AUDITOR`, `DEVELOPER` — `model.py:46-59`, re-read this pass). `Permission` is a closed, flat `enum.Enum` with no inheritance. `ROLE_PERMISSIONS: dict[Role, frozenset[Permission]]` is the sole static source of what a role grants.

**Load-bearing new fact for §7–§10:** roles are **code-defined**, not database rows. Creating a new role (e.g. "Manager minus camera edit") requires a code change and a deploy — it is not something an admin can do through any API or UI, today or in any proposed design that keeps `Role` a Python enum. This is the single fact that decides the Case C analysis below.

Re-confirmed this pass (`model.py`, grepped and spot-read): Sites/Zones read is gated by `VIEW_USERS`, write by `MANAGE_ORGANIZATION`; Cameras read is gated by `VIEW_CAMERAS` (`model.py:95`), write by `MANAGE_CAMERAS` (`model.py:94`) — **these are already four distinct permission enum values, two read/write pairs**, confirmed by direct read of `model.py:82-95` and their appearance in `ROLE_PERMISSIONS` at lines 216-225 (super_admin/org_admin), 274, 279, 301, 321, 360 (other roles' partial grants).

## 7. Same-role/different-user access problem analysis

This is the task's Case A–D framing, evaluated against the verified facts above:

- **Case A** (role grants access, user inherits it) — works under pure UNION. Not in dispute.
- **Case B** (role doesn't grant access, admin adds it for one user) — works under UNION as `explicit_grants`, the mechanism Identity §9/Roadmap §9 already recommended (a new `PermissionGrant`-shaped table, additive).
- **Case C** (role grants EDIT, admin needs ONE specific user restricted to READ-ONLY) — **does not work under pure UNION.** A `frozenset` union has no subtraction operator; there is no way for an additional row to remove a permission the role already grants. Two candidate fixes exist and are evaluated in §8–§10.
- **Case D** (role grants nothing, user has nothing) — trivially works, not in dispute.

**Is Case C real, or resolved by the existing READ/WRITE split found in §6?** It is **real**, not resolved by §6's finding. §6 confirms `VIEW_CAMERAS`/`MANAGE_CAMERAS` (and the Sites/Zones equivalents) are already separate permission constants — this means the *target state* an admin wants ("this one Manager should hold `VIEW_CAMERAS` but not `MANAGE_CAMERAS`") is expressible as a permission set. But the *only* mechanism that composes a user's permission set today is role union (§6), and `Role` is a closed code enum (§6) — so the admin cannot express "this one Manager, minus `MANAGE_CAMERAS`" without either a code change (new role) or a subtraction mechanism. The READ/WRITE split answers "is there a permission granular enough to represent read-only camera access" (yes) — it does not answer "can an admin assign that granularity to one user without a deploy" (no, today).

## 8. Final recommendation for user-specific permission customization

**RECOMMENDED: three-state per-(user, permission) model — INHERIT (default) / GRANT (explicit add, Case B) / REVOKE (explicit subtract, Case C).** Justification, weighing all three options named in the task:

- **(a) Pure UNION continues.** Rejected as the sole mechanism: cannot express Case C at all, full stop, regardless of how granular the underlying `Permission` enum is (§7).
- **(b) UNION + explicit per-(user,permission) REVOKE (three-state).** RECOMMENDED. A `PermissionOverride`-shaped table (`user_id`, `permission`, `state ∈ {GRANT, REVOKE}`, `granted_by`, `granted_at`) composes as: `effective = (role_permissions ∪ {p : override(p)=GRANT}) − {p : override(p)=REVOKE}`. This is a strict, additive extension of the schema Identity §9 already proposed (a per-user override table) — it does not replace or weaken the existing role-union mechanism, it adds one more term to the same computation, evaluated at the same chokepoint (§9).
- **(c) Keep permissions role-only; model "restrict" as a narrower role assignment.** Evaluated seriously, not rubber-stamped away: this works cleanly in an org where the set of needed permission combinations is small and stable — an admin picks `RESTAURANT_MANAGER_READONLY` instead of `RESTAURANT_MANAGER` for the one user who needs it. But because `Role` is a closed Python enum (§6), **creating that narrower role is a code change and a deploy, not an admin action.** Every future one-off restriction (a different Manager needs to keep camera-edit but lose zone-edit; a third needs to keep both but lose evidence-delete) would require a new enum value and a new `ROLE_PERMISSIONS` entry, each shipped by engineering. This is the textbook role-explosion path the task asks to avoid, and it defeats the entire purpose of admin-self-service permission customization — it would functionally never be used because every request becomes a support ticket to engineering.

**Recommendation: (b).** It keeps `ROLE_PERMISSIONS`/`Role` completely untouched (roles stay the reviewed, bundled, code-reviewed baseline — the existing design's own stated virtue, "no inheritance, because an inherited permission is one nobody remembers granting," `model.py` docstring, carried from Identity §5), while giving an admin a runtime, no-deploy tool for the genuinely rare one-off restriction. It does not reopen the deny-override question Identity §11 correctly closed for a *different* shape of problem — Identity §11 evaluated and rejected a role-vs-role or blanket deny concept; this is a narrow, explicit, per-user, per-permission, individually-audited override, not a general precedence system.

## 9. Effective permission calculation model

**RE-LOCATED THIS PASS, load-bearing.** The exact composition function: `permissions_for(roles: frozenset[Role]) -> frozenset[Permission]` at `app/authorization/model.py:369`, called from `AccessDecision.__post_init__` at `app/authorization/model.py:453`:

```
object.__setattr__(self, "permissions", permissions_for(self.roles))
```

`AccessDecision` itself is constructed at `app/authorization/model.py:428` (class) / `:444` (`__post_init__`, the method containing the line above). `AccessDecision` is built by `decide(user, *, grant=None)` at `app/authorization/resolver.py:68`, called on every request via `current_access` → `decision_for_claims` (`app/api/dependencies.py`, `app/auth/service.py`) — re-confirmed present this pass by grep, unchanged from Roadmap §1/§12's citation.

**RECOMMENDED insertion point for the three-state model:** inside `AccessDecision.__post_init__` (`model.py:444` region, immediately after the `permissions_for(self.roles)` call at line 453), or equivalently inside `decide()` before constructing `AccessDecision` — the same open implementation-shape question Identity §18.1 already flagged, now answered as "either is fine, but `__post_init__` keeps the union+subtract logic co-located with the single line it extends," a minor implementation preference, not a new architectural decision.

`effective_permissions = (permissions_for(roles) ∪ grants) − revokes`, computed once per request at this exact point, inheriting the "no cache to invalidate, takes effect on the very next request" guarantee already proven for role changes (Roadmap §12).

## 10. Whether allow-only UNION is sufficient or insufficient (answer definitively)

**INSUFFICIENT. Definitive answer, re-derived from first principles in §7–§9, not inherited from any prior report's conclusion (the prior reports recommended UNION-only because they had not yet posed Case C against the actual closed-enum `Role` constraint).** A pure union of role permission sets has no subtraction operator; Case C is a real, reachable operational need (an admin managing several `restaurant_manager`s who wants exactly one of them read-only without touching the role for everyone else or waiting on a deploy); and the alternative that avoids adding a REVOKE concept (narrower roles) is foreclosed by `Role` being a closed, code-defined enum (§6, §8). **RECOMMENDED: extend to UNION + REVOKE (§8 option (b)).**

## 11. Action/feature permission inventory

Consolidated from Roadmap §10 (verified there by direct enum read, re-spot-checked this pass at `model.py:82-186`), not re-derived:

| Feature area | Read permission | Write permission | Tier shape |
|---|---|---|---|
| Sites, Zones | `VIEW_USERS` (VERIFIED, odd pairing but confirmed) | `MANAGE_ORGANIZATION` | No Access / Read / Read+Edit |
| Cameras | `VIEW_CAMERAS` | `MANAGE_CAMERAS` | No Access / Read / Read+Edit |
| Camera health | `VIEW_CAMERA_HEALTH` | none — runtime signal, no edit action exists | No Access / Read only |
| Live feed | `VIEW_LIVE` | none | No Access / Read only |
| Observations | `VIEW_OBSERVATIONS` | none | No Access / Read only |
| Evidence | `VIEW_EVIDENCE` | `DELETE_EVIDENCE` (delete, not edit) | No Access / Read / Read+Delete |
| Incidents | `VIEW_INCIDENTS` | `ACKNOWLEDGE_INCIDENTS`, `RESOLVE_INCIDENTS` (two separate write actions) | No Access / Read / Read+Acknowledge / Read+Acknowledge+Resolve |
| Reports | `VIEW_REPORTS` | `EXPORT_REPORTS` (export, not edit) | No Access / Read / Read+Export |
| Users | `VIEW_USERS` | `MANAGE_USERS` (declared, unused — no route consumes it) | No Access / Read only (write side dormant) |
| Organization | none dedicated — no Organization CRUD exists | n/a | Unbuilt |
| Model Evaluation | `VIEW_MODEL_EVALUATION` | none, engineering-axis permission | No Access / Read only, separate axis |
| DevTools | `ACCESS_DEVTOOLS`, `REGISTER_DEMAND` | n/a, not a read/write pair | Separate engineering axis, never folded into the business triad |

Sites/Zones/Cameras separate-read/write-constant fact (load-bearing for §7–§10) re-confirmed directly this pass, not merely inherited: `model.py:82` (`MANAGE_ORGANIZATION`), `:84` (`VIEW_USERS`), `:94` (`MANAGE_CAMERAS`), `:95` (`VIEW_CAMERAS`).

## 12. Read-only/Read+Edit/Custom UX mapping

RECOMMENDED, carried from Roadmap §10/§17 (a genuine finding of that pass, re-affirmed, not re-derived): the UI must render **per-feature variable tiers**, not a uniform three-column table — Evidence is Read+Delete not Read+Edit; Incidents is a three-step progression; Reports is Read+Export. A "Custom" tier, newly relevant given §8's REVOKE recommendation, should render as: base tier from the user's role(s), with each individually GRANTed permission shown as "+ added" and each REVOKEd permission shown as "− removed," never as an opaque flat permission-string list (detailed further in §25).

## 13. Platform Operator vs Organization Super Admin

RECOMMENDED, carried unchanged from Discovery §18.1, Stage 1 §15, Roadmap §5 (re-affirmed a fourth time, no new evidence found this pass to overturn it): `super_admin` stays tenant-scoped by construction (`AccessDecision.tenant_id` is a single mandatory field, `model.py`, re-confirmed present this pass). A genuinely cross-tenant "platform operator" must be a structurally separate concept that only ever talks to the `Organization` table and never constructs an `AccessDecision` naming more than one tenant — never a redefinition of `super_admin`.

## 14. Anti-escalation architecture

RECOMMENDED, carried from Identity §10/Roadmap §11, **evaluated against the Case-C/REVOKE recommendation as the task requires, not merely restated**: the existing rule — a grantor may only grant a permission they themselves currently hold, checked against their own freshly-rebuilt `AccessDecision` at the moment of the write — continues to apply to GRANT unchanged. **REVOKE needs its own, distinct rule, because revoking is not the mirror image of granting:** a grantor should be able to revoke a permission from a target user **only within a scope they administer** (i.e., the grantor must hold `MANAGE_USERS` in the target's own organization, exactly as for GRANT), but — unlike GRANT — **a grantor does not need to personally hold the permission being revoked** in order to revoke it from someone else (an `org_admin` who does not hold `MANAGE_PATRON_ID` should still be able to revoke `MANAGE_PATRON_ID` from a user whose role grants it, if that role happens to be one the org_admin can administer). RECOMMENDED concrete rule: REVOKE requires `MANAGE_USERS` in the target's tenant (same gate as GRANT/role-assignment) but has no "grantor must hold it" precondition, since revoking narrows access rather than expanding it and therefore carries no escalation risk in the direction GRANT's rule exists to block. A REVOKE of a permission the grantor doesn't hold cannot be used to *escalate* anyone, only to *restrict* — this asymmetry should be stated explicitly in the implementation, not left to be inferred from the GRANT rule.

## 15. User provisioning lifecycle

VERIFIED FROM CODE / RECOMMENDED, unchanged from Identity §12–§13, Roadmap §6 (re-affirmed, not re-derived): created only via `scripts/manage.py create-user`, no product-level event; activated implicitly at creation (`User.is_active` ORM default `True`); no CLI subcommand or route sets `is_active = False` — deactivation is not currently possible through any code path; no self-service reset; RECOMMENDED provisioning path remains admin-set one-time password mirroring the existing `reset_password --generate` pattern, gated on the already-declared, currently-unused `MANAGE_USERS` permission once a real user-write API exists.

## 16. Organization lifecycle

RECOMMENDED, carried from Roadmap §4 (re-affirmed, not re-derived): three states — ACTIVE (unchanged current behavior), SUSPENDED (login succeeds, reads continue, `MANAGE_*` writes refused — RECOMMENDED insertion point `decision_for_claims`, `app/auth/service.py`, the same chokepoint identified for §9's permission composition), ARCHIVED (login refused entirely, mirrors today's `Organization.is_active=False` behavior, data retained not destroyed).

## 17. ACTIVE/SUSPENDED/ARCHIVED behavior matrix

Carried from Roadmap §4, presented here as the still-open decision it is, per the task's explicit instruction not to resolve it in this pass:

| Capability | ACTIVE | SUSPENDED (proposed) | ARCHIVED (proposed) |
|---|---|---|---|
| Login | Allowed | Allowed (so an admin can review/appeal) | Refused |
| Read (VIEW_*) | Allowed | Allowed | Refused |
| Write (MANAGE_*) | Allowed | Refused | Refused |
| User management | Allowed | Refused | Refused |
| Site/Zone/Camera management | Allowed | Refused | Refused |
| **Live streaming** | Running | **REQUIRES OWNER DECISION — not resolved here (Roadmap §25.1, still the single most important open item, unchanged by this pass)** | Stopped |
| **Background analysis** | Running | **REQUIRES OWNER DECISION — same open item** | Stopped |
| Report generation (existing data) | Allowed | Allowed (read) | Refused |
| API access | Full | Read-only | Refused |

## 18. CCTV/camera administration architecture

VERIFIED FROM CODE (carried, consolidated from Discovery §7/§10, Stage 1 §5, Roadmap §15 — not re-investigated this pass per the guardrail against touching RTSP runtime code): RTSP/credential fields live directly on `Camera` (`host`, `rtsp_port`, `channel`, `stream_type`, `username`, `credential_ref` — a pointer, never a plaintext secret). The camera-ownership fix (`_restaurant_in_tenant` guard on camera create) is **re-confirmed still present this pass**: `app/api/product.py:31` imports `_restaurant_in_tenant` from `app.api.administration`, matching the Stage 1 §19 fix exactly.

## 19. DVR entity decision

RECOMMENDED (carried, reaffirmed a fourth time from Discovery §11–§12, Stage 1 §5–§6, Roadmap §16 — no new evidence found this pass to overturn it): do not introduce a first-class DVR/NVR table now. No traced use case requires bulk DVR registration, per-DVR health, or centralized credential rotation. A lightweight, additive `dvr_group_id` metadata field remains the recommended eventual addition, deferred until real multi-site feedback exists.

## 20. Shared connection configuration analysis

VERIFIED FROM CODE (carried, unchanged): cameras sharing a physical DVR are only informally grouped by identical `host`/`rtsp_port` values; there is no `dvr_id` FK. `credential_ref` supports different credentials per channel on the same physical unit, so centralizing credentials would be a net loss of flexibility, not a gain, without a demonstrated need. Not re-investigated further this pass, per the guardrail against touching RTSP/connection runtime code.

## 21. Tenant isolation verification table

Consolidated from Roadmap §2 (the most recently and thoroughly re-traced pass), with §22–§23 below resolving its two open items definitively:

| Module | Classification | Evidence |
|---|---|---|
| Restaurants / Zones / Cameras | VERIFIED TENANT-SAFE | `_restaurant_in_tenant` guard present and re-confirmed (`product.py:31`, §18) |
| Incidents | VERIFIED TENANT-SAFE | `app/domain/incidents.py` — every `IncidentService` method filters on `organization_id` (Roadmap §2) |
| Evidence | VERIFIED TENANT-SAFE | `EvidenceRecord.organization_id` filtered on every read path; storage path partitioned by org on disk (Roadmap §2) |
| Frames | VERIFIED TENANT-SAFE | Every frame call passes `organization_id=access.tenant_id` (Roadmap §2) |
| Reports | VERIFIED TENANT-SAFE | Every `app/reporting/sources.py` source function filters on `request.organization_id` (Roadmap §2) |
| Observations | **VERIFIED TENANT-SAFE — resolved this pass, see §22** | `app/domain/observations.py:93,100` |
| Notifications | **VERIFIED TENANT-SAFE — resolved this pass, see §23** | `app/domain/notifications.py:228`, no read API exists |
| Staff hygiene / compliance modules | VERIFIED TENANT-SAFE, currently moot (no live detection) | `app/api/analytics.py` / `app/api/capability.py` — `organization_id=access.tenant_id` on every call (Roadmap §2) |
| Patron tokens, POS integrations | VERIFIED TENANT-SAFE | `app/api/patron.py:99`, `app/api/integrations.py:70,110` (Roadmap §2) |
| Users / RoleAssignment / AccessGrant | VERIFIED TENANT-SAFE | `app/api/administration.py` route + `_restaurant_in_tenant` (Roadmap §2) |
| Audit log | VERIFIED TENANT-SAFE at query level, one schema looseness | `AuditEvent.organization_id` has no FK constraint (carried, not re-verified this pass) |
| Model evaluation | VERIFIED TENANT-SAFE / not applicable by design | Disk artifacts, identical for every tenant (carried) |
| Live wall, Analytics, DevTools, Websocket | VERIFIED TENANT-SAFE | Unchanged (Roadmap §2) |

## 22. Observations verification result (definitive, re-traced)

**VERIFIED TENANT-SAFE. Definitive.** This resolves the Roadmap report's NOT VERIFIED flag, which resulted from grepping for the literal string `organization_id` in `app/domain/observations.py` and finding zero matches — the module in fact scopes by `tenant_id`, not `organization_id`, which is why that grep produced a false negative.

Direct read of `app/domain/observations.py:72-118` (`query_observations`) this pass:

```
tenant = TenantId(access.tenant_id)
principal = Principal(subject=access.subject, tenant_id=tenant, scopes=..., display_name=...)
scope = Scope(tenant_id=tenant, camera_ids=tuple(CameraId(c) for c in cameras))
```

(`app/domain/observations.py:93,96,100`) — the Vision OS `Scope` and `Principal` passed to the platform's own `query_observations` call are built directly and exclusively from `access.tenant_id`, the same `AccessDecision` field verified in §9 to be derived only from the authenticated user's own `organization_id` row, never from client input.

Both call sites re-traced this pass:
- `app/api/product.py:706-718` — when the caller's grant is tenant-wide, the `cameras` tuple passed into `query_observations` is populated from `CameraService(session).list(organization_id=access.tenant_id, ...)` (`product.py:707-708`) — i.e. even the camera list argument is pre-filtered to the caller's own tenant before `query_observations` is ever called.
- `app/reporting/sources.py:255-274` — same pattern: `cameras` is populated from a tenant-scoped registered-camera query before being passed to `observation_fold.query_observations`.

**Conclusion:** no cross-tenant leak is possible in the observations read path; both the `Scope`/`Principal` construction and the camera list fed into it are tenant-derived at every step.

## 23. Notifications isolation result (definitive, re-traced)

**VERIFIED TENANT-SAFE. Definitive.** Direct read of `app/domain/notifications.py` (full file) this pass, plus the call-site trace:

- There is **no notification read, list, or query API anywhere in the backend.** `app/api/routes.py:10` states explicitly: "There is no restaurant, camera, incident, notification or report route" (re-confirmed by grep across `app/api/*` for `notification`/`Notifier`/`incident_opened` this pass — the only matches are the dispatch call site and the wiring in `app/main.py`). Notifications are **fire-and-forget dispatch only**, to a `LogChannel` (application log), `FileChannel` (append-only JSONL on disk), or `NullChannel` — there is no durable, queryable `NotificationEvent` table in the schema, so there is no read path to leak through, by construction.
- The one field that carries tenant identity, `IncidentNotice.organization_id`, is populated at dispatch time directly from `incident.organization_id` (`app/domain/notifications.py:228`, inside `_notice_from`) — it is never independently queried or filtered; it inherits whatever tenant scoping produced the `Incident` object in the first place, and `Incident` is itself VERIFIED TENANT-SAFE (§21, carried from Roadmap §2).
- Call site: `Notifier.incident_opened(incident, finding)` is invoked from `app/vision/compliance_driver.py:436` (cited for location only, per the guardrail against reading compliance/incident logic in depth — this citation was obtained by grep, not by opening or analyzing that file's logic).

**Conclusion:** no cross-tenant leak is possible, both because the one tenant-carrying field is derived from an already-safe source and because there is no read/query surface at all for a leak to manifest through.

## 24. Audit event architecture

Consolidated from Identity §15, Roadmap §13 (carried, not re-derived): `AuditTrail` is append-only, `_scrub()` structurally strips credential-shaped values from any `detail` payload. Status per event, extended for this pass's REVOKE recommendation:

| Event | Status |
|---|---|
| `LOGIN` / `LOGOUT` / `LOGIN_FAILED` | ALREADY IMPLEMENTED |
| `RESTAURANT_CREATED` / `ZONE_CREATED` / camera create/update/delete | ALREADY IMPLEMENTED |
| `OBSERVATIONS_READ` | ALREADY IMPLEMENTED (confirmed this pass, `app/api/product.py:722-733`) |
| Organization created/suspended/archived | MISSING — no Organization CRUD exists to audit yet |
| User created/deactivated | MISSING — no user-write API exists |
| Role assigned/revoked | MISSING — no route writes `RoleAssignment` beyond the CLI, which is itself unaudited |
| Permission GRANT issued | REQUIRES NEW EVENT (e.g. `permission.granted`) — net-new for §8's recommendation |
| Permission REVOKE issued | REQUIRES NEW EVENT (e.g. `permission.revoked`) — net-new for §8's recommendation, and per §14 must record whether the grantor held the permission being revoked (they need not, by the asymmetric rule) |
| CLI (`scripts/manage.py`) account operations | MISSING — confirmed absent, no `AuditTrail` call in that script |

## 25. Future Administration information architecture

RECOMMENDED, extending Identity §14/Roadmap §17 with the Case-C answer this pass required (not previously specified): a **People & Access** area sits inside `/admin`, showing per user: their role(s) (base tier), then a per-permission row distinguishing three visual states — **"Role Inherited"** (from `permissions_for(roles)`, no distinguishing marker needed beyond showing it came from a named role), **"Added"** (an explicit GRANT, shown with a "+" and who granted it/when, from `RoleAssignment.granted_by`-style attribution), and **"Restricted"** (an explicit REVOKE, shown with a "−" and who revoked it/when). This must never collapse into a flat permission-string list — each row's provenance (which role granted it, or that it's an override) must be visible, exactly as the task requires, so an admin auditing one user's access can answer "why does this person have this" without a second lookup.

## 26. Complete finalized end-to-end use cases

Consolidated, not re-derived, from all four prior reports' use-case sections (Discovery §10, Stage 1 §9-14, Identity §7/§13, Roadmap §6): Add Organization (blocked on Organization CRUD + user provisioning, both MISSING); Add Site/Zone (fully implemented today); Add Camera (fully implemented, ownership-checked, §18); Move Camera between zones (implemented) / between sites (not implemented — `restaurant_id` not patchable); Retire Camera (implemented, refuse-rather-than-orphan); Restrict one Manager to read-only cameras while keeping their other permissions (the Case C scenario this pass resolves — requires the §8 REVOKE mechanism, not buildable today).

## 27. Risks and unresolved decisions

Consolidated from all four prior reports' decision lists, deduplicated: (1) SUSPENDED cascade to streaming/analysis — unresolved, §17; (2) `super_admin` vs. platform-operator role shape — resolved as recommendation (§13), not as owner sign-off; (3) account-provisioning mechanism for new orgs — unresolved; (4) Organization deletability — unresolved; (5) DVR entity — resolved as recommendation against building it (§19); (6) `AuditEvent.organization_id` missing FK — low risk today, should be fixed before Organization deletion is ever built; (7) **new this pass:** REVOKE's anti-escalation asymmetry (§14) needs explicit owner sign-off before implementation, since it is a genuinely new rule with no existing precedent in the codebase to point to.

## 28. Explicit list of decisions that MUST be approved before implementation

1. Adopt the three-state INHERIT/GRANT/REVOKE permission model (§8, §10) — REQUIRES OWNER DECISION.
2. Confirm the asymmetric anti-escalation rule for REVOKE (§14) — grantor need not hold the permission being revoked, only `MANAGE_USERS` in the target's tenant — REQUIRES OWNER DECISION.
3. SUSPENDED cascade to streaming/analysis (§17) — REQUIRES OWNER DECISION, unresolved by design per task instructions.
4. Account-provisioning mechanism for a new organization's first admin (§15, carried) — REQUIRES OWNER DECISION.
5. Platform-operator role introduction, exact permission name (§13, carried) — REQUIRES OWNER DECISION.
6. Organization deletability vs. suspend/archive-only (carried) — REQUIRES OWNER DECISION.
7. New `AuditAction` enum values and exact string naming for permission GRANT/REVOKE events (§24) — REQUIRES OWNER DECISION.

## 29. Recommended implementation stages after approval

1. **Verification stage (already substantially complete as of this pass):** Observations and Notifications now VERIFIED TENANT-SAFE (§22-23) — no further tracing required before Stage 2 on these two items specifically.
2. **Schema stage:** add `PermissionOverride` table (`user_id`, `permission`, `state`, `granted_by`, `granted_at`) — additive only, no existing table changes.
3. **Authorization stage:** extend `AccessDecision.__post_init__` (§9) to compute `(permissions_for(roles) ∪ grants) − revokes`; implement the asymmetric anti-escalation rule (§14) at the same write chokepoint used for `requires()`.
4. **Audit stage:** add `permission.granted`/`permission.revoked` `AuditAction` values (§24, §28.7).
5. **API stage:** build the user-write API (`MANAGE_USERS`-gated) that the override mechanism depends on — Case B/C are meaningless without a way to write to `PermissionOverride` through a real, gated route, not the CLI.
6. **Frontend stage:** build the People & Access "Role Inherited / Added / Restricted" UI (§25).
7. **Organization lifecycle and platform-operator stages** proceed independently per the prior reports' staging (Roadmap §21), unaffected by this pass's permission-model work.

---

## Decision Table

| DECISION | CURRENT STATE | OPTIONS | RECOMMENDATION | WHY | IMPLEMENTATION IMPACT | USER APPROVAL REQUIRED |
|---|---|---|---|---|---|---|
| Effective permission composition model | Role-union only (`permissions_for(roles)`, `model.py:369`, `AccessDecision.__post_init__`, `model.py:444-453`); no per-user grant/revoke exists | (a) Pure UNION; (b) UNION + per-user REVOKE (three-state); (c) Model restriction as a narrower role | **(b) UNION + REVOKE** | Case C is unreachable under (a); (c) requires a code deploy per new role since `Role` is a closed enum (`model.py:46-59`) — defeats admin self-service, causes role explosion | New `PermissionOverride` table; extend `AccessDecision.__post_init__`; new audit events | YES |
| SUSPENDED organization cascade to streaming/analysis | No org-lifecycle states exist at all today; `Restaurant.is_active` precedent does NOT cascade to children | Cascade (pause streaming/analysis) vs. do not cascade (only block writes) | Not resolved by this report, per task instructions — present as open | Real product/cost trade-off, no code precedent settles it either way | Insertion point identified (`decision_for_claims`), not built | YES |
| REVOKE anti-escalation rule | No REVOKE mechanism exists; GRANT's rule (grantor must hold what they grant) is the only precedent | Symmetric (grantor must hold permission to revoke it) vs. asymmetric (any `MANAGE_USERS` holder in-scope can revoke, regardless of their own holding) | **Asymmetric** | Revoking narrows access, carries no escalation risk in the direction GRANT's rule blocks; a symmetric rule would block legitimate restriction by an admin who doesn't personally hold every permission their reports do | New rule, distinct code path from GRANT's check | YES |
| Platform Operator vs Organization Super Admin | `super_admin` is tenant-scoped by construction (`AccessDecision.tenant_id` single mandatory field) | Make `super_admin` cross-tenant vs. new separate platform-operator concept | Separate concept, never touching multi-tenant `AccessDecision` | Blast radius across ~56 tenant-scoped query sites too large for a role-semantics change | New role/permission concept, new `/organizations` API | YES |
| First-class DVR entity | No DVR table; cameras share host/rtsp_port informally | Introduce `DVR` table vs. keep informal vs. lightweight `dvr_group_id` metadata | Keep informal now; defer lightweight metadata | No traced use case justifies migration cost | None now; additive nullable column if ever approved | YES (for future stage only) |
| Organization deletability | No Organization CRUD exists at all | Deletable (exercises `ondelete=CASCADE`) vs. suspend/archive only | Not re-resolved this pass, carried open | `Incident.restaurant_id ondelete=SET NULL` loses history if Restaurant is ever deleted | Affects whether Restaurant/Incident FK behavior needs fixing first | YES |
| Account-provisioning mechanism | CLI-only, ungated, unaudited | Admin-set one-time password vs. email invite (missing infra) vs. SSO (missing infra) | Admin-set one-time password (lowest novelty) | Extends existing `reset_password --generate` pattern; no invite/SSO infra exists today | New user-write API needed regardless of which option is chosen | YES |

---

**Files touched to produce this document:** only this file — `unityworks-vision-ai-backend/docs/architecture/FINAL_ADMINISTRATION_MULTI_ORGANIZATION_IDENTITY_PERMISSION_CCTV_ARCHITECTURE_FREEZE.md`. No production code, migration, route, permission, role, UI component, or perception/detection/tracking/VLM/vision_os/compliance/observation/incident logic file was created, modified, or executed.
