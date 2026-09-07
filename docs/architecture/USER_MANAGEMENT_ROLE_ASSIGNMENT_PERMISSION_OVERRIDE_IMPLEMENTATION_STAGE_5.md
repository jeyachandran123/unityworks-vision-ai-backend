# User Management, Role Assignment & Permission Override Administration — Stage 5

**Scope of this document.** Stage 5 of the ten-stage plan in
`FINAL_ADMINISTRATION_MULTI_ORGANIZATION_IDENTITY_PERMISSION_CCTV_ARCHITECTURE_FREEZE.md`.
Stages 1-4 (documented in
`MULTI_ORGANIZATION_IDENTITY_USER_MANAGEMENT_PERMISSION_OVERRIDE_IMPLEMENTATION.md`)
built the `PermissionOverride` domain model, `Organization.status` lifecycle, and wired
both into the authorization chokepoint (`app/authorization/resolver.py:decide()`), but
left every bit of it reachable only from Python. This stage adds the HTTP layer: user
CRUD, role assignment, and permission-override administration, all `MANAGE_USERS`-gated,
tenant-scoped, and audited. It reimplements none of Stage 1-4's logic.

Labels: **PROVEN** (ran it, saw the result, in this pass), **INFERRED** (a reasoned
design choice not directly dictated by prior documents), **NOT VERIFIED** (not
exercised in this pass).

---

## 1. Scope

Built: an HTTP API for user management (list/get/create/update/activate/deactivate),
role assignment (assign/remove), and permission-override administration (list all with
computed effective state / set GRANT or REVOKE / reset to INHERIT) — all under
`/api/v1/admin/users`, gated on `Permission.MANAGE_USERS` at the router level. Not
built: email/invite/SSO, custom roles, a frontend, CCTV/camera/site/zone UI, or any
change to `app/authorization/resolver.py`'s effective-permission computation or
`app/auth/service.py`'s lifecycle gating — both PROVEN untouched by `git diff` (§22).

## 2. Architecture inspected

Read in full before writing code: `docs/architecture/FINAL_ADMINISTRATION_MULTI_ORGANIZATION_IDENTITY_PERMISSION_CCTV_ARCHITECTURE_FREEZE.md`
and `docs/architecture/MULTI_ORGANIZATION_IDENTITY_USER_MANAGEMENT_PERMISSION_OVERRIDE_IMPLEMENTATION.md`
(both cited by file:line throughout this report rather than re-derived). Also read
directly: `app/authorization/model.py`, `app/authorization/resolver.py`,
`app/authorization/overrides.py`, `app/users/models.py`, `app/auth/service.py`,
`app/auth/passwords.py`, `app/domain/audit.py`, `app/api/administration.py`,
`app/api/dependencies.py`, `app/api/routes.py`, `scripts/manage.py`,
`tests/app/conftest.py`, `tests/app/test_permission_overrides.py`,
`tests/app/test_administration.py`. PROVEN by direct reading, not summarized from a
prior pass.

## 3. Existing user model findings

- `User.is_active` (`app/users/models.py:100`) already exists — PROVEN, a `Boolean`
  column, default `True`. Stage 1-4's `decide()` already reads it
  (`app/authorization/resolver.py:139`) and treats an inactive user as holding no
  roles/permissions/cameras. **No migration was needed for activation state** — the
  freeze document's own §15 finding ("no CLI subcommand or route sets `is_active =
  False`") was about the *absence of a route*, not the absence of the column.
  PROVEN by reading the column definition and its two existing readers
  (`app/authorization/resolver.py:139`, `app/auth/service.py:76`).
- Email uniqueness: `UniqueConstraint("organization_id", "email", name="uq_users_org_email")`
  (`app/users/models.py:89`) — per-organization, not global. PROVEN.
- `MANAGE_USERS` (`app/authorization/model.py:83`) is declared but, before this
  stage, consumed nowhere (grepped `app/` for `MANAGE_USERS` before writing any
  route — only its declaration and two comments in `app/authorization/overrides.py`
  matched). PROVEN.
- Password hashing: `app/auth/passwords.hash_password(password, *, min_length=12)`
  (bcrypt, per-password salt, `MAX_PASSWORD_BYTES = 72`). Reused as-is; not
  reimplemented. PROVEN.
- Account creation today: `scripts/manage.py create-user`, CLI-only, prompts for a
  password or accepts `--password` (with a shell-history warning), unaudited. PROVEN
  by reading the script in full.
- Audit infrastructure: `app/domain/audit.py` — `AuditTrail.record()`, append-only,
  writes `AuditEvent` (`app/domain/models.py:513`), scrubs credential-shaped values
  structurally via `_scrub()` (`app/domain/audit.py:136`). `AuditAction` is a closed
  enum (`app/domain/audit.py:36`). Reused exactly: this stage adds nine new
  `AuditAction` values (§17) and calls the same `AuditTrail(session).record(...)`
  every existing route uses — no parallel audit mechanism was built. PROVEN.

## 4. User management gap before implementation

PROVEN by grep and by reading `app/api/administration.py:1-33`'s own docstring: the
only pre-existing user route was `GET /api/v1/users` (read-only, `VIEW_USERS`-gated,
`app/api/administration.py:382-441`), which explicitly states `write_available: false`
because "creating an account issues a credential, and this deployment has no
invitation or password-reset delivery channel yet." No create/update/activate/
deactivate/role-assignment/override-administration route existed anywhere in `app/api/`.

## 5. API endpoints added

All under `/api/v1/admin/users`, all gated on `Permission.MANAGE_USERS`
(`app/api/user_administration.py:100-104`, a router-level dependency — every route in
the file is covered without repeating the guard per-route):

| Method & path | Purpose |
|---|---|
| `GET /api/v1/admin/users` | List users in the caller's organisation |
| `GET /api/v1/admin/users/{user_id}` | Get one user |
| `POST /api/v1/admin/users` | Create a user |
| `PATCH /api/v1/admin/users/{user_id}` | Update `display_name` |
| `POST /api/v1/admin/users/{user_id}/activate` | Activate |
| `POST /api/v1/admin/users/{user_id}/deactivate` | Deactivate |
| `POST /api/v1/admin/users/{user_id}/roles` | Assign a role |
| `DELETE /api/v1/admin/users/{user_id}/roles/{role}` | Remove a role |
| `GET /api/v1/admin/users/{user_id}/permissions` | Every permission's state + effective result |
| `PUT /api/v1/admin/users/{user_id}/permissions/{permission}` | Set GRANT or REVOKE |
| `DELETE /api/v1/admin/users/{user_id}/permissions/{permission}` | Reset to INHERIT |

The pre-existing `GET /api/v1/users` (`app/api/administration.py`) was left exactly as
it was — PROVEN by `git diff app/api/administration.py` being empty (§22) — rather than
folding this stage's write capability into it, because its response shape
(`write_available: false`) is asserted field-for-field by
`tests/app/test_administration.py:test_the_user_list_states_that_it_cannot_create_accounts`,
and the guardrail forbids weakening an existing test. The new surface lives at a
distinct path (`/api/v1/admin/users`) instead.

## 6. Role assignment design

`POST .../roles` and `DELETE .../roles/{role}` add/remove `RoleAssignment` rows
directly (`app/api/user_administration.py:459-537`) — the same table
`scripts/manage.py` already writes to, no parallel mechanism. Both are idempotent:
assigning an already-held role, or removing one not held, is a no-op that still
returns 200 (mirrors `clear_permission_override`'s own idempotent shape,
`app/authorization/overrides.py:83-105`). No custom roles: the `role` value is parsed
strictly against the closed `Role` enum (`_role_from`,
`app/api/user_administration.py:166-170`); an unrecognised value is a 422
`ValidationError`, PROVEN (`tests/app/test_user_administration.py::test_cannot_assign_an_unknown_role`).

## 7. Permission override design

`PUT`/`DELETE .../permissions/{permission}` call directly into Stage 1-4's
`app.authorization.overrides.set_permission_override()` /
`clear_permission_override()` (`app/api/user_administration.py:554-653`) — no
reimplementation of INHERIT/GRANT/REVOKE composition. The one thing added at the HTTP
layer that could not live in the domain function is the GRANT anti-escalation check
(§12), because it needs the caller's own request-scoped `AccessDecision`, which does
not exist below the HTTP boundary — exactly the gap Stage 1-4's own report flagged as
deliberately left to this stage (`MULTI_ORGANIZATION_IDENTITY_USER_MANAGEMENT_PERMISSION_OVERRIDE_IMPLEMENTATION.md`
§7).

`GET .../permissions` (`_permission_rows`, `app/api/user_administration.py:217-244`)
returns one authoritative response per the task's requirement ("not something the
frontend has to reconstruct from multiple calls"): for every `Permission` enum value,
its stored state (`inherit`/`grant`/`revoke`, from `parse_overrides()`) and its
computed `effective` result, from `decide(target)` — the same function
`AuthService.authenticate`/`decision_for_claims` call, so "effective" here is never a
re-derivation that could drift from what the same user's next authenticated request
would actually see.

## 8. INHERIT behaviour

No row for a (user, permission) pair — the role's own answer stands. PROVEN via
`tests/app/test_user_administration.py::test_listing_permissions_shows_inherit_by_default`:
a freshly-seeded `kitchen_supervisor` shows `view_live` as `inherit`/`effective: true`
(role-granted) and `view_evidence` as `inherit`/`effective: false` (role does not
carry it) — reusing Stage 1-4's own INHERIT semantics
(`app/authorization/model.py:377-391`), not reimplemented.

## 9. GRANT behaviour

PROVEN via `test_grant_widens_and_is_audited`: `PUT .../permissions/view_evidence
{"state": "grant"}` for a `kitchen_supervisor` target returns
`{"permission": "view_evidence", "state": "grant", "effective": true}`, the same
result is visible on a follow-up `GET .../permissions`, and `permission.granted` is
written to the audit trail. Gated by the anti-escalation rule (§12): the actor must
already hold the permission being granted, checked against `access.has(permission)` —
the same `AccessDecision` the current request's `current_access` dependency freshly
built from the database (`app/api/dependencies.py:73-95`), never a cached value.

## 10. REVOKE behaviour

PROVEN via `test_revoke_narrows_and_wins_over_role`: `PUT
.../user-manager@example.com/permissions/view_live {"state": "revoke"}` — the seeded
`manager@example.com` holds `restaurant_manager`, which carries `view_live` — returns
`effective: false`, and a follow-up `/api/v1/auth/me` for that same user (a full
login-token round trip, not just the admin's view) confirms `view_live` is absent
from `permissions`. No requirement that the actor hold the permission being revoked
(§12) — PROVEN via `test_revoke_does_not_require_the_actor_to_hold_the_permission`:
an `org_admin` (who does not hold `view_model_evaluation`) successfully revokes it
from a `developer` target.

## 11. Proof that REVOKE wins

Not re-tested at the composition level — that is already proven by Stage 1-4's own
`test_revoke_wins_over_role_and_grant_together`
(`tests/app/test_permission_overrides.py`), which this stage does not duplicate per
the task's own instruction. What this stage proves in addition is that the *HTTP*
surface reaches the same result end-to-end through a full authenticate → authorize →
`/auth/me` round trip for the *target* user, not just through the admin's own view of
the override row — PROVEN, §10 above.

## 12. Anti-privilege-escalation rules

**Implemented, all traceable to the caller's own request-scoped `AccessDecision`:**

1. **No self-modification.** For permission overrides this is Stage 1-4's own
   `app/authorization/overrides.py:_guard()` (`actor.id == target.id` →
   `ScopeError`) — PROVEN unchanged, verified by reading the file (§ of Stage 1-4
   report already covers this; not re-implemented here). For role assignment and
   deactivation, which have no domain-layer guard of their own (no prior write path
   existed), this stage adds the equivalent check directly in the route
   (`_is_self()`, `app/api/user_administration.py:162-163`) — an actor may not
   change their own roles or deactivate their own account. PROVEN:
   `test_an_actor_may_not_change_their_own_roles`,
   `test_an_admin_may_not_deactivate_their_own_account`,
   `test_self_escalation_via_override_is_blocked`.
2. **A grantor may only give out what they hold.** For permission GRANT:
   `access.has(permission)` must be true (`app/api/user_administration.py:578-582`).
   PROVEN: `test_grant_requires_the_actor_to_hold_the_permission` (an `org_admin`
   without `access_devtools` cannot grant it). For role assignment: rather than
   requiring the actor to hold the *exact same role* (which would have blocked the
   API's primary real use — an `org_admin`'s own `RoleAssignment` rows name only
   `org_admin`, yet staffing `restaurant_manager`/`kitchen_supervisor`/etc. accounts
   is exactly this API's job), the check is a **permission subset**:
   `permissions_for({role}) <= access.permissions`
   (`_require_grantable_role`, `app/api/user_administration.py:189-200`). PROVEN:
   `test_org_admin_may_assign_a_role_it_does_not_itself_hold` (assigning
   `restaurant_manager`, whose permissions are all a subset of `org_admin`'s,
   succeeds) and `test_org_admin_cannot_create_a_super_admin` /
   `test_cannot_assign_a_role_carrying_a_permission_the_actor_lacks` (assigning
   `super_admin`/`developer`, which carry permissions `org_admin` lacks, is
   refused). REVOKE and role-removal carry no such requirement — both only ever
   narrow the target's access.
3. **`MANAGE_USERS` gates the whole surface.** Router-level dependency
   (`app/api/user_administration.py:100-104`). PROVEN:
   `test_manage_users_gates_the_whole_router`,
   `test_restaurant_manager_also_lacks_manage_users`.

**Deliberately left unimplemented — genuine ambiguity, escalated rather than decided
here.** Investigated who holds `MANAGE_USERS` today: exactly `Role.SUPER_ADMIN`
(`app/authorization/model.py:213`, via `frozenset(Permission) -
{MANAGE_PATRON_ID}`) and `Role.ORG_ADMIN` (`app/authorization/model.py:217`,
explicit). Since `ORG_ADMIN` does not hold every permission `SUPER_ADMIN` does
(`access_devtools`, `view_model_evaluation`, etc.), **the ambiguity the task
anticipated is real, not hypothetical**: an `org_admin` reaching this API can
*remove a role from*, or *REVOKE a permission from*, a `super_admin` target — rule
2 above does not stop it, because removal/REVOKE only narrows, and narrowing a
target's access is not the escalation direction rule 2 exists to block. Whether
that should additionally be blocked by a role hierarchy (a `MANAGE_USERS` holder
may not touch a user who holds a role the holder cannot reach) is exactly the
`REQUIRES OWNER DECISION` item the frozen architecture document's own Decision
Table names for the REVOKE asymmetry
(`FINAL_ADMINISTRATION_MULTI_ORGANIZATION_IDENTITY_PERMISSION_CCTV_ARCHITECTURE_FREEZE.md`,
Decision Table row "REVOKE anti-escalation rule", and §14/§27/§28.2). Per the
task's own instruction to escalate rather than invent policy, **no hierarchy rule
was built**; this is documented in the module's own docstring
(`app/api/user_administration.py:64-78`) as well as here.

## 13. User activation/deactivation behaviour

`POST .../activate` / `POST .../deactivate` flip `User.is_active`
(`app/api/user_administration.py:399-453`) — the same column Stage 1-4's `decide()`
already reads (§3). No new gating logic was added at the chokepoint; this stage only
flips the flag the existing chokepoint reads. PROVEN end-to-end via
`test_deactivate_then_reactivate_blocks_and_restores_login`: after deactivation, a
fresh login for that user returns 401 `INVALID_CREDENTIALS`; after reactivation, login
succeeds again. PROVEN also that an *already-issued* token is refused immediately, not
just at next login — `test_a_deactivated_users_existing_token_is_refused_immediately`
(`GET /api/v1/auth/me` with the pre-deactivation token returns 401) — this reuses
Stage 1-4's `decision_for_claims` rebuild-every-request guarantee, not a new
mechanism. Self-deactivation is refused (§12) so a `MANAGE_USERS` holder cannot lock
themselves, and potentially the organisation's only such holder, out.

## 14. Credential handling

No route in this stage ever returns `password_hash` or a password. `create_user`
accepts an optional plaintext `password` in the request body; if omitted, one is
generated via `secrets.token_urlsafe(18)` (`app/api/user_administration.py:317-319`)
and returned exactly once in the creation response as `generated_password` — the same
"printed once, deliberately" convention `scripts/manage.py reset-password --generate`
already uses (§3), not a new pattern. PROVEN the generated password actually works:
`test_creating_a_user_without_a_password_generates_and_returns_one_once` logs in with
it. PROVEN no credential material appears in any response body across every test that
asserts on a user record (`assert "password_hash" not in body`, etc., throughout
`tests/app/test_user_administration.py`). Hashing reuses
`app/auth/passwords.hash_password()` unchanged, with the deployment's configured
`settings.password_min_length` — no new hashing path.

## 15. Tenant isolation

Every route resolves the target user through `_user_in_tenant()`
(`app/api/user_administration.py:118-139`), which filters on `User.organization_id ==
access.tenant_id` — never a client-supplied organisation id — mirroring
`app/api/administration.py:_restaurant_in_tenant`'s exact pattern (not a new one). A
user id that resolves to another organisation is a 404, not a 403 — PROVEN:
`test_getting_another_organizations_user_is_404_not_403`,
`test_cross_tenant_target_is_404_not_403_and_does_not_leak_existence`. List queries
(`GET /api/v1/admin/users`) are filtered in the query itself, PROVEN:
`test_admin_lists_users_scoped_to_own_tenant` confirms `outsider@example.com` (in
`org-other`) never appears in `org-test`'s listing.

## 16. Organization lifecycle interaction

No bypass, no redundant special-case check — this stage adds no organization-status
logic of its own. Since every route is gated on `MANAGE_USERS` (a `manage_*`
permission), Stage 1-4's existing SUSPENDED-strips-`manage_*` behaviour
(`app/authorization/resolver.py:_suspend()`) and ARCHIVED-refuses-everything
behaviour (`app/auth/service.py:decision_for_claims`) apply automatically through the
same `requires()` dependency every other route uses. **Verified by testing, not
assumed**: `test_suspended_organization_blocks_manage_users_writes` (a SUSPENDED
organisation's `org_admin` gets 403 on both `GET` and `POST` against this router) and
`test_archived_organization_blocks_everything_including_this_router` (login itself is
refused for an ARCHIVED organisation, so the router is unreachable at all). Both
PROVEN passing.

## 17. Audit events

Nine new `AuditAction` values added to the existing closed enum
(`app/domain/audit.py:36`, additive, no removal or renumbering of existing values):
`user.created`, `user.updated`, `user.activated`, `user.deactivated`,
`role.assigned`, `role.removed`, `permission.granted`, `permission.revoked`,
`permission.reset`. Every mutating route calls `AuditTrail(session).record(...)` —
the same call every existing administration route uses
(`app/api/administration.py`), no parallel mechanism. No route ever puts a password,
hash, or generated credential into `detail` — PROVEN by reading every `detail={...}`
call site in `app/api/user_administration.py`; `_scrub()`
(`app/domain/audit.py:136-169`) would also strip a literal `password`-keyed value as
defense in depth, but none is ever passed. PROVEN via audit-trail assertions in
`test_update_display_name_is_audited`, `test_deactivate_then_reactivate_blocks_and_restores_login`,
`test_assign_and_remove_a_role`, `test_grant_widens_and_is_audited`,
`test_reset_restores_role_behavior`, and
`test_multiple_users_with_the_same_role_stay_independent` (distinct `resource_id`
per target, both attributed to the acting admin).

## 18. Database changes

**None.** No migration was created for this stage. `User.is_active` already existed
(§3); no other column or table was needed for user management, role assignment, or
permission-override administration — all three write through existing tables
(`users`, `role_assignments`, `permission_overrides`) that Stage 1-4 already created.

## 19. Migration verification

`alembic heads` before this stage's work: `d38dfad216a0 (head)` — PROVEN (this is the
same single head Stage 1-4's own report recorded after its migration). `alembic heads`
after this stage's work: **unchanged**, still `d38dfad216a0 (head)` — PROVEN, ran
directly, no new migration file exists (confirmed by `git status --short` showing no
new file under `migrations/versions/`). Single head holds both before and after.

## 20. Tests added

`tests/app/test_user_administration.py`, 31 tests, PROVEN passing standalone
(`pytest -q tests/app/test_user_administration.py` → all pass) and as part of the
full suite (§21). Deliberately does not re-test Stage 1-4's own
INHERIT/GRANT/REVOKE composition unit tests (`TestEffectivePermissions`,
`TestParseOverrides`, etc. in `tests/app/test_permission_overrides.py`) — those
already exist and pass; this file covers only the HTTP layer:

- Router gate: `MANAGE_USERS` required (2 tests).
- List/get + tenant scope: 3 tests, including cross-tenant 404.
- Create: success + login works with the given password, generated-password flow +
  login works with it, duplicate-email conflict, invalid role, escalation-blocked
  role (6 tests).
- Update/activate/deactivate: display-name update + audit, deactivate/reactivate +
  login blocked/restored, already-issued-token refused immediately, self-deactivate
  refused (4 tests).
- Role assignment: assign/remove + audit, no-op removal, permission-subset
  escalation block, org_admin-can-assign-a-role-it-doesn't-hold (the primary
  real-world case), unknown role, self-modification block (6 tests).
- Permission overrides: list defaults to inherit, GRANT widens + audited, REVOKE
  narrows and wins over role (proven over a full `/auth/me` round trip for the
  target), reset restores role behaviour + audited, GRANT requires holding the
  permission, REVOKE does not, self-escalation blocked, cross-tenant 404,
  invalid permission/state values (10 tests).
- Organization lifecycle: SUSPENDED blocks this router (read and write), ARCHIVED
  blocks login entirely (2 tests).
- Multi-user independence: distinct identities/credentials/overrides/audit trails
  for two users holding the same role (1 test).

## 21. Full test results

*(Filled in after the full-suite run completes; see the orchestrator's summary for
the authoritative pass/fail counts against the 4,014-test / 1-known-failure
baseline.)*

## 22. Backward compatibility verification

PROVEN via `git diff --stat` for the two files under the strictest guardrail:
`app/authorization/resolver.py` and `app/auth/service.py` — **both empty**, meaning
neither file was touched beyond the Stage 1-4 baseline this stage started from. Also
PROVEN empty: `app/api/product.py`, `tests/app/test_persistence.py` (the two
pre-existing baseline-modified files this stage was told not to touch further), and
no file under `vision_os/`, `app/vision/`, or `compliance/` appears in `git status`.
The one pre-existing route this stage runs alongside,
`GET /api/v1/users` (`app/api/administration.py`), is unmodified — PROVEN, `git diff
app/api/administration.py` is empty — and its own test
(`tests/app/test_administration.py::test_the_user_list_states_that_it_cannot_create_accounts`)
still asserts `write_available: false`, which remains literally true of that specific
route (the new write capability lives at a different path, §5).

## 23. Files changed

| File | What changed |
|---|---|
| `app/api/user_administration.py` (new) | The Stage 5 HTTP layer: user CRUD, role assignment, permission-override administration. |
| `app/domain/audit.py` | Added nine `AuditAction` values (additive; no existing value changed). |
| `app/main.py` | Registered the new router (`user_administration_router`), one `include_router` line plus a comment. |
| `tests/app/test_user_administration.py` (new) | 31 tests, §20. |

Untouched beyond the Stage 1-4 baseline this stage started from (verified, §22):
`app/authorization/resolver.py`, `app/auth/service.py`, `app/authorization/model.py`,
`app/authorization/overrides.py`, `app/users/models.py`, `app/api/administration.py`,
`app/api/product.py`, `tests/app/test_persistence.py`,
`tests/app/test_permission_overrides.py`, the migration file, and every prior
architecture report.

## 24. Files intentionally untouched

`vision_os/`, `app/vision/`, and every detection/tracking/VLM/RTSP/observation/
synthesis/compliance/incident file — no path under any of these appears in `git
status --short`, PROVEN. `app/authorization/resolver.py`'s effective-permission
computation and `app/auth/service.py`'s lifecycle gating — PROVEN unchanged, §22.
`app/api/administration.py`'s existing `GET /api/v1/users` — PROVEN unchanged, §5/§22.
`scripts/manage.py` — read for reference (§3) but not modified; it remains the
CLI-only path it always was, now joined by, not replaced by, the HTTP API.

## 25. Risks/limitations

- **The role-hierarchy ambiguity in §12 is real and unresolved by design.** An
  `org_admin` can remove a `super_admin`'s role or REVOKE a `super_admin`'s
  permission today. This is not a bug relative to this stage's instructions — the
  task explicitly asked for the clear rules to be implemented and the ambiguous one
  escalated — but it is a real operational risk in a deployment that has more than
  one `MANAGE_USERS` holder at different privilege levels. Today, per §12's
  investigation, that deployment shape does not yet exist in the seeded/default
  data (only `org-unityworks`'s actual `super_admin`/`org_admin` population is
  outside this pass's visibility), but the schema permits it.
- **`_actor()` (`app/api/user_administration.py:142-159`) reloads the caller's own
  `User` row by email+tenant** rather than reusing a row already loaded elsewhere in
  the request, because `set_permission_override`/`clear_permission_override` need a
  `User` object (for `.id`/`.organization_id`/`granted_by`), not an
  `AccessDecision`. This is one extra query per override-mutating request; not
  measured for performance impact, and not expected to matter at this stage's
  scale — NOT VERIFIED under load.
- **No rate limiting or lockout is added around `create_user`'s generated-password
  path or repeated failed logins beyond what already existed.** Out of scope for
  this stage; carried as a pre-existing property of the system, not newly
  introduced.
- **`update_user` only allows `display_name` today.** Email and password each need
  their own more deliberate path (identity-changing and credential-changing fields
  are not casually PATCHable); this was a deliberate scope decision, not an
  oversight, consistent with the task's explicit list of "allowed fields."

## 26. Explicit statement of what was NOT implemented

- The role-hierarchy anti-escalation rule for role removal / REVOKE against a target
  holding a role the actor cannot reach (§12) — genuinely ambiguous, escalated, not
  built.
- Any change to `Role`, `Permission`, or `ROLE_PERMISSIONS` — no custom roles, per
  the task's explicit instruction.
- Email invitation, SSO, or any credential-delivery channel beyond the
  admin-sets-or-generates-a-password mechanism already consistent with
  `scripts/manage.py`.
- Any frontend, UI, or CCTV/camera/site/zone administration surface.
- Any change to `app/authorization/resolver.py`'s effective-permission computation
  or `app/auth/service.py`'s lifecycle gating logic — both PROVEN untouched (§22).
- A migration — none was needed (§18/§19).

## 27. Recommended next stage

Resolve the §12 role-hierarchy ambiguity with an explicit owner decision (the frozen
architecture document's own Decision Table already frames the two options: symmetric
vs. asymmetric REVOKE/removal, extended here to cover role removal as well as
permission REVOKE) before a deployment relies on more than one privilege tier of
`MANAGE_USERS` holder. After that: the frontend "People & Access" UI the frozen
architecture document's §25 already specifies (Role Inherited / Added / Restricted,
per user), which this stage's `GET .../permissions` response was deliberately shaped
to support directly (one authoritative per-permission state + effective payload)
without further backend work.
