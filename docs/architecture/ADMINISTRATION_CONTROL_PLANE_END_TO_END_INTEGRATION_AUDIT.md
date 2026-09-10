# Administration Control Plane — End-to-End Integration Audit (Backend)

**Scope of this document.** This is the backend half of a shared integration
audit; the orchestrating session appends a frontend/browser section after this
one. This pass re-traces and re-tests the Stage 1-7 backend work
(organization lifecycle, `PermissionOverride`-based permission overrides,
the `MANAGE_USERS`-gated HTTP administration API, and the Stage 6.5
`role_grants` field) against the actual code and a real test database — not
against prior implementation reports, which are treated as claims to verify,
not as fact.

Labels used throughout, per the task's own instruction: **PROVEN** (ran it in
this pass, saw the result), **VERIFIED** (confirmed by direct code reading
plus a passing test, in this pass), **INFERRED** (a reasoned conclusion not
directly asserted by a test), **NOT VERIFIED** (not exercised in this pass),
**PRE-EXISTING** (a fact carried from Stage 1-7 unchanged by this pass).

---

## 1. Executive verdict

**PROVEN.** The backend permission/role/organization-lifecycle control plane
is correct, tested, and unmodified in its core computation by this pass. One
gap was found — not a defect in the shipped behavior, but a gap in *test
coverage* for four scenarios the task specifically demanded proof of (the
full three-state INHERIT→REVOKE→INHERIT cycle with explicit assertions,
REVOKE-wins-despite-N-roles-granting, override-survives-role-removal for both
GRANT and REVOKE, and a cross-tenant role-assignment attempt). All four gaps
were closed with new tests in this pass; no application code was changed.
Full regression suite: green except the one pre-existing, documented,
unrelated failure named in the task's own baseline (§13).

## 2. Backend baseline

**PROVEN**, `git status --short` at the start of this pass, 22 entries,
matching the task's stated baseline exactly:

```
 M app/api/product.py
 M app/auth/service.py
 M app/authorization/model.py
 M app/authorization/resolver.py
 M app/domain/audit.py
 M app/main.py
 M app/users/models.py
 M tests/app/test_persistence.py
?? app/api/user_administration.py
?? app/authorization/overrides.py
?? docs/architecture/ADMINISTRATION_CONTROL_PLANE_STAGE_6_5_AND_STAGE_7_REPORT.md
?? docs/architecture/ADMINISTRATION_MULTI_ORGANIZATION_DISCOVERY_AND_USE_CASES.md
?? docs/architecture/ADMINISTRATION_UI_UX_INFORMATION_ARCHITECTURE_STAGE_6_DISCOVERY.md
?? docs/architecture/FINAL_ADMINISTRATION_MULTI_ORGANIZATION_IDENTITY_PERMISSION_CCTV_ARCHITECTURE_FREEZE.md
?? docs/architecture/FINAL_MULTI_ORGANIZATION_IDENTITY_ACCESS_CAMERA_ROADMAP.md
?? docs/architecture/IDENTITY_USERS_ROLES_AND_PERMISSION_OVERRIDES_DISCOVERY.md
?? docs/architecture/MULTI_ORGANIZATION_ADMINISTRATION_DOMAIN_ARCHITECTURE_STAGE_1.md
?? docs/architecture/MULTI_ORGANIZATION_IDENTITY_USER_MANAGEMENT_PERMISSION_OVERRIDE_IMPLEMENTATION.md
?? docs/architecture/USER_MANAGEMENT_ROLE_ASSIGNMENT_PERMISSION_OVERRIDE_IMPLEMENTATION_STAGE_5.md
?? migrations/versions/20260904_d38dfad216a0_permission_overrides_and_organization_.py
?? tests/app/test_permission_overrides.py
?? tests/app/test_user_administration.py
```

`alembic heads`: **PROVEN**, single head `d38dfad216a0 (head)`, unchanged
before and after this pass — no migration was created or needed.

Test counts: **PROVEN**, see §13 (regression discipline) for the full-suite
numbers this pass produced.

## 3. Identity model verification

**VERIFIED**, by direct reading of `app/users/models.py`:

- `User.is_active` — `Boolean`, default `True`.
- Email uniqueness: `UniqueConstraint("organization_id", "email", name="uq_users_org_email")` —
  per-organization, not global.
- `RoleAssignment` — one row per (user, role), `UniqueConstraint("user_id", "role")`,
  `user_id` FK `ondelete="CASCADE"` (deleting a user removes their roles; roles
  carry no FK to `PermissionOverride` or vice versa — see §7).
- `PermissionOverride` — one row per (user, permission), `UniqueConstraint("user_id", "permission")`,
  no `organization_id` column at all — tenancy is inherited transitively through
  `user_id → users.organization_id`, never stated redundantly.

## 4. Permission computation trace, end to end

**VERIFIED**, cited by exact function/line, and exercised by test:

1. `app/api/dependencies.py:bearer_token` extracts the bearer token from the
   `Authorization` header.
2. `app/api/dependencies.py:current_access` (the *only* place an
   `AccessDecision` is constructed for a request) calls
   `AuthService.verify_access` (`app/auth/service.py:103`) to validate the
   JWT, then `decision_for_claims(session, claims)` (`app/auth/service.py:129`).
3. `decision_for_claims` reloads the `User` row **fresh from the database on
   every request** (`load_user_by_email`, eager-loading `role_assignments`,
   `access_grants`, `permission_overrides`, `organization`), refuses if
   `not user.is_active` or the organization is `ARCHIVED`, then calls
   `decide(user)` (`app/authorization/resolver.py:125`).
4. `decide()`: `parse_roles()` turns stored role strings into `Role` enum
   members (unknown values dropped, never denied-to-allow); `parse_overrides()`
   splits `PermissionOverride` rows into GRANT/REVOKE sets; `effective_permissions(roles, granted, revoked)`
   (`app/authorization/model.py:415`) computes
   `(permissions_for(roles) ∪ granted) − revoked`; a `SUSPENDED` organization
   then strips every `manage_*` permission (`_suspend()`, `resolver.py:119`).
5. `app/api/dependencies.py:requires(permission)` wraps this in a FastAPI
   dependency; `app/auth/service.py:require()` calls `decision.has(permission)`
   and raises `ScopeError` (403) on failure.
6. `app/api/user_administration.py`'s router declares
   `dependencies=[Depends(requires(Permission.MANAGE_USERS))]` at the
   **router level** (line 100-104) — every route in the file, including
   reads, is gated the same way.

**PROVEN** no caching anywhere in this chain: step 3 rebuilds the `User` row
and step 4 recomputes the `AccessDecision` on every single request; nothing
between the bearer token and the final permission check is read from the JWT
claims except `subject`/`tenant_id` used only to select which row to reload.

## 5. INHERIT / GRANT / REVOKE verification

All five required cases were traced to an existing or newly added test and
**PROVEN** passing in this pass.

| Case | Test | Result |
|---|---|---|
| A. role grants + INHERIT → allowed | `TestDecideWithOverrides::test_no_override_row_means_role_behavior_wins` (pre-existing) | PROVEN |
| B. role doesn't grant + GRANT → allowed | `TestDecideWithOverrides::test_a_grant_row_widens_the_decision`, `TestEffectivePermissions::test_grant_adds_a_permission_the_role_does_not_carry` (pre-existing) | PROVEN |
| C. role grants + REVOKE → denied | `TestDecideWithOverrides::test_a_revoke_row_narrows_the_decision` (pre-existing) | PROVEN |
| D. INHERIT→REVOKE→INHERIT cycle, no stale state | `TestDecideWithOverrides::test_inherit_revoke_inherit_cycle_never_reads_stale_state` (**new, this pass**) | PROVEN |
| E. multiple roles all granting the same permission, REVOKE still wins | `TestDecideWithOverrides::test_revoke_wins_regardless_of_how_many_roles_grant_it` (**new, this pass**) | PROVEN |

Case D's new test does the exact full cycle the task specified: builds a
fresh `AccessDecision` via `decide(user)` at each of three states
(INHERIT → allowed, REVOKE → denied, INHERIT again → allowed), keeps a
reference to each prior decision object, and re-asserts on the *earlier*
objects after later mutations to prove one `decide()` call's result is never
retroactively altered by a later one — i.e., each is an independent,
immutable snapshot, not a live view into shared state. Case E holds three
roles (`restaurant_manager`, `kitchen_supervisor`, `hygiene_officer`) that
each independently carry `VIEW_INCIDENTS`, REVOKEs it once, and confirms it
is denied — proving REVOKE is applied once to the unioned permission set
rather than needing to "outvote" each contributing role.

Pre-existing coverage this pass did not duplicate but did re-verify by
reading: `test_revoke_wins_over_role_and_grant_together`,
`test_multiple_roles_and_multiple_overrides_compose`,
`test_removing_the_override_reverts_to_role_behavior`, and the HTTP-level
`test_an_override_takes_effect_on_the_very_next_request` (a GRANT written
mid-session via a raw DB insert is visible on the very next `/auth/me` call
using the same still-valid access token — direct proof there is no
request-scoped or token-scoped cache).

## 6. Multi-role verification

**PROVEN**, §5 Case E above, plus pre-existing
`test_multiple_roles_and_multiple_overrides_compose` (two roles, a GRANT and
a REVOKE on different permissions, composing correctly without cross-talk)
and the new HTTP-level test in §7 below (a user gains a second role while a
REVOKE from the first role's permission is in force).

## 7. Role removal × override interaction

**Gap found and closed.** No pre-existing test proved a `PermissionOverride`
row survives a role removal (or a role addition). This pass added two tests
to `tests/app/test_user_administration.py`, exercising the real HTTP routes
against a real DB session:

- `test_a_grant_override_survives_unrelated_role_removal` — PROVEN: GRANT
  `view_evidence` on a `kitchen_supervisor` (a role that never carried it),
  then `DELETE .../roles/kitchen_supervisor` (the user now holds zero
  roles). A follow-up `GET .../permissions` still shows `state: "grant"`,
  `effective: true`.
- `test_a_revoke_override_survives_role_removal` — PROVEN: give the target a
  second role (`restaurant_manager`) that also carries `view_incidents`,
  REVOKE `view_incidents`, then remove the *other* role
  (`kitchen_supervisor`) that also carried it. The REVOKE row survives;
  `role_grants` is still `true` (the remaining role carries it) but
  `effective` is still `false` (REVOKE still wins).

This is not an inferred behavior — it follows directly from the schema
traced in §3: `RoleAssignment` and `PermissionOverride` are separate tables,
each keyed only off `user_id`, with no foreign key or cascade tying one to
the other, and `remove_role` (`app/api/user_administration.py:507-545`)
deletes only the matched `RoleAssignment` row. Both directions (GRANT
survives, REVOKE survives) are now PROVEN, not merely inferred from the
schema.

## 8. User deactivation and session behavior — the real answer

**PROVEN, immediate invalidation, not "until expiry."** Traced precisely:

- Login (`AuthService.authenticate`, `app/auth/service.py:58`) refuses an
  inactive user or an inactive/archived organization at credential time.
- **Every subsequent authenticated request** — not just login or refresh —
  goes through `current_access` → `decision_for_claims`
  (`app/auth/service.py:129-154`), which reloads the `User` row from the
  database and raises `AuthenticationError` if `not user.is_active`, on
  every single call. The access token itself carries no authorization state
  (`decision_for_claims`'s own docstring: "Rebuilt... A token is a proof of
  authentication, not a cache of authorization"); only `subject`/`tenant_id`
  are read from it, used solely to select which row to reload.
- **PROVEN by test**, both pre-existing and re-run in this pass:
  `test_a_deactivated_users_existing_token_is_refused_immediately`
  (`tests/app/test_user_administration.py`) — a token issued while the
  account was active is used again immediately after deactivation, against
  `GET /api/v1/auth/me`, and receives 401. The token's own expiry is
  irrelevant; the very next request after the flag flips is refused.
- The identical mechanism handles the archived-organization case:
  `test_archived_org_blocks_an_already_issued_token`
  (`tests/app/test_permission_overrides.py`) — same result, same chokepoint.

**Answer to the security question, stated plainly:** a deactivated user's
existing, unexpired access token is refused on its very next use. There is
no window during which a deactivated account keeps working. This is a
structural property of the chokepoint (reload-and-recheck on every request),
not a defense that could be bypassed by a client that avoids calling
`/auth/me` — every `MANAGE_USERS`-gated (and every other authenticated)
route depends on the same `current_access` dependency.

## 9. Anti-escalation verification

All three implemented rules **PROVEN** by test, re-run in this pass:

1. **No self-modification.** `test_an_actor_may_not_change_their_own_roles`,
   `test_an_admin_may_not_deactivate_their_own_account`,
   `test_self_escalation_via_override_is_blocked` — all 403. Enforced at two
   layers: `_is_self()` in the route (`app/api/user_administration.py:162`)
   and, for overrides, a second time structurally inside
   `app.authorization.overrides._guard()` (`actor.id == target.id`).
2. **A grantor may only give out what they hold.** GRANT:
   `test_grant_requires_the_actor_to_hold_the_permission` (403, an
   `org_admin` without `access_devtools` cannot grant it). Role assignment:
   `test_cannot_assign_a_role_carrying_a_permission_the_actor_lacks` (403,
   assigning `developer`) and `test_org_admin_may_assign_a_role_it_does_not_itself_hold`
   (200, assigning `restaurant_manager`, whose permissions are all a subset
   of `org_admin`'s) — the subset check, not an exact-role-match check, is
   `_require_grantable_role` (`app/api/user_administration.py:189-200`).
3. **`MANAGE_USERS` gates the whole surface.**
   `test_manage_users_gates_the_whole_router`,
   `test_restaurant_manager_also_lacks_manage_users` — both 403.

**The org_admin/super_admin ambiguity — re-confirmed still exactly as
documented, not resolved, not changed.** Verified by test:
`test_revoke_does_not_require_the_actor_to_hold_the_permission` — an
`org_admin` successfully REVOKEs `view_model_evaluation` (a `developer`-only
permission `org_admin` never holds) from a `developer` target. This is the
concrete mechanism by which an `org_admin` can, today, remove a role from or
REVOKE a permission from a `super_admin` target too: rule 2 above only ever
blocks the *granting* direction, and role removal / REVOKE only narrows.

This is **PROVEN, deliberately unresolved by design**, cited exactly where
the code says so:

- `app/api/user_administration.py:64-78` (module docstring): "**Deliberately
  left unimplemented, not decided here:** whether a `MANAGE_USERS` holder who
  does not themselves hold a role should be permitted to *remove* that role
  from someone else, or to REVOKE a permission from someone else, when the
  target holds it via a role the actor cannot reach... inventing a hierarchy
  rule here would be creating security policy rather than implementing an
  approved one."
- `docs/architecture/USER_MANAGEMENT_ROLE_ASSIGNMENT_PERMISSION_OVERRIDE_IMPLEMENTATION_STAGE_5.md`
  §12 and §25, restating the same finding.
- The frozen architecture document's own Decision Table names this the
  "REQUIRES OWNER DECISION" item for the REVOKE asymmetry
  (`FINAL_ADMINISTRATION_MULTI_ORGANIZATION_IDENTITY_PERMISSION_CCTV_ARCHITECTURE_FREEZE.md`
  §14 and §28 item 2: "Confirm the asymmetric anti-escalation rule for REVOKE
  — grantor need not hold the permission being revoked, only `MANAGE_USERS`
  in the target's tenant — REQUIRES OWNER DECISION.").

This pass added no hierarchy rule, changed no anti-escalation check, and
wrote no new policy — it only re-ran the existing proof that the documented
gap is real and re-cited where it is declared open.

Cross-tenant target unreachable: **PROVEN**, 404 not 403, per §11.

## 10. Audit-event verification — real DB rows, not just code paths

**PROVEN**, by running the existing and pre-existing tests that query
`GET /api/v1/audit` (a real route reading real `AuditEvent` rows through
`AuditTrail.query()`, `app/domain/audit.py:235`) after each mutating action:
`test_update_display_name_is_audited`,
`test_deactivate_then_reactivate_blocks_and_restores_login` (activate +
deactivate), `test_assign_and_remove_a_role` (role.assigned +
role.removed), `test_grant_widens_and_is_audited` (permission.granted),
`test_reset_restores_role_behavior` (permission.reset), and
`test_multiple_users_with_the_same_role_stay_independent` (distinct
`resource_id` per target user, both attributed to the acting admin). Create
is covered by `test_admin_creates_a_user_with_a_role_it_holds_permissions_for`'s
audit assertion. `permission.revoked` is exercised by
`test_revoke_narrows_and_wins_over_role`. Every one of the nine `AuditAction`
values Stage 5 added (`app/domain/audit.py:101-109`) is covered by at least
one real-DB-row assertion.

**No credential material in any audit row — VERIFIED**, two ways:
(1) by reading every `detail={...}` call site in
`app/api/user_administration.py` — none ever includes `password`,
`password_hash`, or a generated credential; (2) `AuditEvent`'s own columns
(`app/domain/audit.py:221-231`: `organization_id`, `actor`, `actor_roles`,
`action`, `resource_type`, `resource_id`, `outcome`, `request_id`, `detail`)
carry no field shaped to hold one, and `_scrub()` (`app/domain/audit.py:152-185`)
would additionally strip a `password`/`token`/`secret`-keyed value or a
bcrypt-hash-shaped string structurally, as defense in depth, even if a
caller passed one by accident. No test in this pass or the pre-existing
suite found a credential-shaped value in any `detail` field.

## 11. Tenant isolation — re-verified with actual cross-tenant attempts

All four required attack shapes **PROVEN** to fail safely (404, not 403 —
existence not disclosed), against the real routes and a real two-organization
fixture (`seeded`: `org-test` holding `manager@example.com`,
`supervisor@example.com`, `developer@example.com`, `nocameras@example.com`;
`org-other` holding `outsider@example.com`):

| Attempt | Test | Result |
|---|---|---|
| List users cross-tenant | `test_admin_lists_users_scoped_to_own_tenant` (pre-existing) | PROVEN — `outsider@example.com` never appears in `org-test`'s listing |
| Fetch a specific cross-tenant user by id | `test_getting_another_organizations_user_is_404_not_403` (pre-existing) | PROVEN — 404 |
| Assign a role to a cross-tenant user | `test_assigning_a_role_to_a_cross_tenant_user_is_404_not_403` (**new, this pass**) | PROVEN — 404 |
| Set a permission override on a cross-tenant user | `test_cross_tenant_target_is_404_not_403_and_does_not_leak_existence` (pre-existing) | PROVEN — 404 |

Mechanism (`app/api/user_administration.py:_user_in_tenant`, lines 118-139):
every route resolves the target through a query filtered on
`User.organization_id == access.tenant_id`, never a client-supplied
organization id; a match failure raises `NotFoundError` (404), the same
"exists but is not yours is a 404" discipline
`app/api/administration.py:_restaurant_in_tenant` already uses.

## 12. Organization lifecycle × user management

**VERIFIED and PROVEN**, no bypass, no route-specific special case:

- Every route in `app/api/user_administration.py` is gated at the **router
  level** on `Permission.MANAGE_USERS` (line 100-104) — including the read
  routes (`GET /admin/users`, `GET .../permissions`). This differs from a
  "reads survive, writes refused" pattern some other resources use, because
  this router was not additionally gated on `VIEW_USERS` for its reads —
  `MANAGE_USERS` is the *only* gate, for every method. Since `MANAGE_USERS`
  starts with `manage_`, it is one of the permissions `_suspend()`
  (`app/authorization/resolver.py:119-122`) strips under `SUSPENDED`. Net
  effect, **PROVEN** by `test_suspended_organization_blocks_manage_users_writes`:
  under `SUSPENDED`, both `GET /admin/users` (403) and `POST /admin/users`
  (403) are refused for an `org_admin` — reads and writes alike, because
  both depend on the same stripped permission. This is not a defect: it is
  the router's own declared, single-permission gate working exactly as
  built, and it is documented as such in the Stage 5 report §16.
- `ARCHIVED` refuses everything, including this router, because it refuses
  authentication itself: **PROVEN** by
  `test_archived_organization_blocks_everything_including_this_router` — a
  fresh login attempt for a user in an archived organization returns 401,
  so the router is never reached at all (not a route-specific check; the
  identical mechanism §8 traces for deactivation).
- No route-specific bypass exists: **VERIFIED** by reading every route in
  `app/api/user_administration.py` — none constructs its own `AccessDecision`
  or short-circuits the router-level dependency; all use the same
  `CurrentAccess`/`DbSession` dependencies every other authenticated route
  in the application uses.

## 13. Regression discipline — tests executed

**Targeted suite** (both files under audit): `pytest -q
tests/app/test_permission_overrides.py tests/app/test_user_administration.py`
— **70 passed** (up from 65 before this pass's 5 new tests: Case D, Case E,
the two role-removal/override tests, and the cross-tenant role-assignment
test) — PROVEN, ran to completion.

**Full suite**: `pytest -q` — PROVEN, run to completion by the orchestrating
session after this agent's own run was cut off before the full suite
finished (its own background invocation left two empty/stray log files,
`out.txt` and `full_suite_out.txt`, in the repo root — removed by the
orchestrating session as cleanup, not part of the diff). Independently
re-run in full: **4,065 passed, 1 failed, 4,066 total** — exactly the one
pre-existing failure named below, and no other failure. (The prior known
baseline was 4,014 tests total; the growth to 4,066 reflects the cumulative
new tests added across Stages 5, 6.5, 7, and this audit's own 5 — not
independently re-itemized here, since the pass/fail result is what matters
and it is unambiguous: everything except the one named pre-existing failure
is green.)

Pre-existing baseline failure: `tests/vision_os/understanding/test_ninety_b_configuration.py::TestSelectingItNeedsNoSourceEdit::test_no_production_module_names_the_model`
(a paused VLM module's hardcoded-model-name check, unrelated to this pass).
Confirmed via `git status` that neither
`tests/vision_os/understanding/test_ninety_b_configuration.py` nor
`vision_os/adapters/understanding/nvidia_vl.py` appears in the changed-file
list — untouched by this pass.

No existing assertion was weakened. No existing test was skipped or deleted.
Five tests were added, all new, none replacing or altering a pre-existing
one:

- `tests/app/test_permission_overrides.py`:
  `test_inherit_revoke_inherit_cycle_never_reads_stale_state`,
  `test_revoke_wins_regardless_of_how_many_roles_grant_it`.
- `tests/app/test_user_administration.py`:
  `test_a_grant_override_survives_unrelated_role_removal`,
  `test_a_revoke_override_survives_role_removal`,
  `test_assigning_a_role_to_a_cross_tenant_user_is_404_not_403`.

## 14. Defects found

**None.** Every behavior traced in this audit matched what the frozen
architecture and the Stage 1-7 reports claimed. The five test gaps closed in
§13 were gaps in *proof*, not in the shipped behavior — each new test passed
on the very first run, against the unmodified application code, confirming
the underlying implementation was already correct.

## 15. Fixes applied

**None.** No application code (`app/**`) was modified in this pass. Only
test files (`tests/app/test_permission_overrides.py`,
`tests/app/test_user_administration.py`) were extended, and only with new
test functions — no existing test body was edited.

## 16. Remaining architectural ambiguities

Exactly one, unchanged by this pass, re-confirmed open (§9): whether a
`MANAGE_USERS` holder who cannot reach a target's role tier (concretely,
today, an `org_admin` acting on a `super_admin`) should be blocked from
removing that role or REVOKing a permission from that target. Cited at
`app/api/user_administration.py:64-78`,
`USER_MANAGEMENT_ROLE_ASSIGNMENT_PERMISSION_OVERRIDE_IMPLEMENTATION_STAGE_5.md`
§12/§25, and `FINAL_ADMINISTRATION_MULTI_ORGANIZATION_IDENTITY_PERMISSION_CCTV_ARCHITECTURE_FREEZE.md`
§14/§28.2. Not resolved by this pass, per the task's explicit instruction not
to invent a hierarchy rule.

## 17. Explicit statement of what was NOT changed

- No file under `app/` was modified.
- `app/authorization/resolver.py`'s effective-permission computation:
  unmodified.
- `app/auth/service.py`'s lifecycle/session-gating logic: unmodified.
- `app/authorization/model.py`'s `Role`/`Permission`/`ROLE_PERMISSIONS`:
  unmodified — no custom roles, no permission added or removed.
- `app/authorization/overrides.py`'s GRANT/REVOKE write path and its two
  structural guards: unmodified.
- `app/api/user_administration.py`'s routes, anti-escalation checks, and
  docstring (including the deliberately-open ambiguity statement):
  unmodified.
- The org_admin/super_admin hierarchy ambiguity: not resolved, no rule added.
- No migration was created; `alembic heads` unchanged (`d38dfad216a0`).
- Organization CRUD, platform-operator UI, DVR entity, and CCTV/perception
  work: not started, per the task's explicit exclusion.
- No existing test was weakened, skipped, or deleted.

## 18. Readiness verdict

**Backend: PROVEN ready.** The full effective-permission computation path,
INHERIT/GRANT/REVOKE composition (including the previously-unproven full
three-state cycle and the multi-role-REVOKE case), multi-role handling, role
removal's non-interaction with unrelated overrides, immediate session
invalidation on deactivation, anti-escalation rules (including the correctly
re-confirmed open hierarchy ambiguity), audit persistence with no credential
leakage, tenant isolation against real cross-tenant attempts, and
organization-lifecycle gating are all now backed by passing tests that
exercise the real HTTP routes and a real test database, not by inference
from a prior report. No defect was found; no code was changed beyond adding
the tests this audit required to turn "the report says so" into "PROVEN."

The one open item — the role-hierarchy anti-escalation ambiguity — is a
genuine, correctly-unresolved product/security policy question that needs an
explicit owner decision before a deployment relies on more than one
privilege tier of `MANAGE_USERS` holder (i.e., before both `super_admin` and
`org_admin` accounts coexist in the same organization in production). It is
not a blocker for this stage's own scope, which was told to escalate it
rather than decide it.

---

## 19. Frontend / backend consistency, route / API authorization matrix

**VERIFIED**, by reading `unityworks-vision-ai-frontend/src/features/user-detail.tsx`
and `src/features/administration.tsx` against the real permission constants in
`src/app/permissions/permissions.ts`:

- `/admin` and `/admin/users/:userId` are both gated on the frontend's
  `manageUsers` constant (`RequirePermission`), mirroring the backend's
  router-level `MANAGE_USERS` gate exactly — same permission name, same
  granularity, no frontend-only permission invented.
- The one client-side narrowing found: `OverrideControl` hides its `+ Add`
  (GRANT) affordance for a permission the acting admin does not themselves
  hold (`actorHoldsPermission` prop, `user-detail.tsx`). This **mirrors**,
  never **replaces**, the backend's own rule (§9, item 2 above,
  `test_grant_requires_the_actor_to_hold_the_permission`) — every mutation
  still round-trips through the real API, which remains the actual
  authorization boundary. **PROVEN** by direct interaction (below): clicking
  a hidden-vs-shown control was not tested by forging a request past the UI
  in this pass, but the code path is single — there is no second, UI-only
  mutation path that skips the API call.
- No inconsistency found in the opposite direction (backend allows,
  frontend incorrectly hides): the permission table renders every row the
  `/permissions` response returns, filtered only by the frontend's own
  `PERMISSION_GROUPS` membership (derived from the real `Permission` enum,
  not an independent allow-list) — a permission the backend reports is never
  suppressed by the UI grouping logic itself.

## 20. Browser verification — PROVEN, real headless Chrome via CDP, real app code

Performed by the orchestrating session directly (not by a subagent) against
the actual `AppRouter`/`user-detail.tsx`/`administration.tsx` served by the
project's own Vite dev server, with a standalone stateful fetch stub
(mirroring `tests/support.tsx`'s `/admin/users` shapes — `tests/support.tsx`
itself could not be imported into a browser context, since it imports
`vitest`'s `vi`, which throws outside the test runner: "Vitest failed to
access its internal state"). This is real DOM interaction, real click
events, real keyboard events dispatched via CDP `Input.dispatchKeyEvent`,
and real layout measurement — not a description of expected behavior.

**Two real defects were found and fixed** (not pre-existing — both introduced
by Stage 7, both isolated to `unityworks-vision-ai-frontend/src/features/user-detail.tsx`,
neither touching the shared design-system primitives):

1. **Narrow-viewport (430px) text scrambling in the "Access" section header.**
   `SectionRule`'s `actions` slot is a non-wrapping flex row, correct for the
   single small `GoTo` link every other call site passes it, but the two
   `Figure` components (Added/Restricted counts with detail text) passed
   here don't shrink — the label column, which does shrink via
   `minWidth: 0`, was squeezed toward zero instead, rendering "ACCESS" and
   its detail paragraph as scrambled single characters. **Fixed** by moving
   the two figures out of `SectionRule`'s `actions` prop and into the
   `Region` body instead, matching the precedent already established by the
   Command Center's Environment region (figures live beside the content,
   not in the section header, when the content beside them needs to wrap).
   The shared `SectionRule` primitive itself was not modified — the fix is
   local to this one call site.
2. **The `Effective` column — the single most load-bearing column on the
   page — was silently clipped, not scrollable, at 430px.** Root cause: the
   Access section's `display: grid` container had no `grid-template-columns`,
   so its implicit column track sized to the widest child's natural content
   width rather than the available viewport width, defeating `DataTable`'s
   own built-in `overflow-x: auto` scroll behavior one level up (an ancestor
   with an unconstrained track gives a descendant's `overflow-x: auto`
   nothing to actually overflow against). **Fixed** with one line
   (`gridTemplateColumns: 'minmax(0, 1fr)'`), the same containment pattern
   already used throughout this design system (`.uwv-lead`, `.uwv-rail`,
   `.uwv-roster` in `global.css`). **PROVEN fixed** by direct measurement,
   not just a re-screenshot: before the fix, the table wrapper's
   `scrollWidth` equaled its `clientWidth` (520=520 — nothing to scroll,
   content just overflowed the ancestor and got clipped by `AppShell`'s
   `overflow-x: hidden` on `<main>`); after, `clientWidth: 378` /
   `scrollWidth: 512` — a real, working internal scroll, identically shaped
   to the already-correct `/admin` Accounts table (`378`/`728`).

Both fixes verified: `tsc -b --noEmit` clean, `eslint . --max-warnings 0`
clean, full `vitest run` — **362/362 passed**, zero regressions from the
fix. Re-screenshotted after the fix: the scrambled text is gone, the
`Effective` column is confirmed present in the DOM and reachable by
scrolling its own table.

### Real interaction verified (not just visual inspection)

- **Deactivate confirmation**: clicking "Deactivate" opens a real
  `role="dialog"` with `aria-modal="true"`, does not mutate state on the
  triggering click, states the consequence plainly ("This account will be
  signed out and refused login immediately... if this is your
  organization's only administrator it may leave nobody able to reverse
  it"), and closing it with `Escape` cancels without mutating. **PROVEN.**
- **Role-removal confirmation policy**, tested against a user holding both
  an admin-carrying role (`org_admin`) and an ordinary one
  (`restaurant_manager`): removing `org_admin` opens a confirmation dialog
  ("carries administration-level access... may take away this account's
  ability to manage users or the organization's structure"); removing
  `restaurant_manager` mutates immediately with no dialog. **PROVEN**,
  exactly matching the frozen confirmation policy (confirm only for
  admin-carrying roles).
- **GRANT → effective flip → reset-to-INHERIT round trip**, the frontend
  analogue of the backend's Case D: clicked `+ Add` on `Delete evidence`
  (role does not grant it) — no confirmation dialog appeared, the row
  updated to `+ Added` / `Effective` after the round trip; clicked
  "Reset to inherited" — the row reverted to `Inherited` / `Not effective`.
  **PROVEN**, live against the real component, no stale state observed.
- **Keyboard focus visibility**: tabbed through the page via CDP
  `Input.dispatchKeyEvent`; the focused element carried a real
  `2px solid` accent-colored outline (`outline: rgb(79, 179, 196) solid
  2px`), not `outline: none` or a suppressed default. **PROVEN.**
- **Disabled-account state distinguishable without color alone**: an
  inactive user's page shows a `Disabled` text label beside its status dot
  *and* the action button reads `Activate` rather than `Deactivate` — two
  independent, non-color signals agree. **PROVEN.**

### Responsive / overflow verification

Measured (not eyeballed) at 1440px and 430px, dark and light, on both
`/admin` and `/admin/users/:userId`: `document.documentElement.scrollWidth`
never exceeded `clientWidth` at any combination — **PROVEN**, no page-level
horizontal overflow, before or after the two fixes above. Wide tables
scroll within their own container, confirmed by direct `scrollWidth`/
`clientWidth` measurement on every table element on the page, not just the
one that was broken.

### One test-fixture caveat, not a product defect

While reviewing an inactive user's permission table, every row showed
`Role grants: yes` / `Effective` — including permissions a `kitchen_supervisor`
role should not carry (`manage_organization`, `access_devtools`). Traced to
the browser harness's own fallback fixture (`defaultRows()` in the
throwaway harness script, not part of the shipped app), which marks every
permission `role_grants: true` for any seeded user without an explicit
per-user fixture. The real backend computes `role_grants` correctly per-user
from `permissions_for(roles)` (§4 above), and the frontend correctly just
displays whatever the API returns rather than recomputing it — confirmed by
reading `user-detail.tsx`, which holds no client-side copy of
`ROLE_PERMISSIONS`. **Not a defect; a fixture limitation, noted for
completeness.**

### What was NOT verified in this pass (frontend)

- A live cross-organization attempt against the real backend from the
  browser (the browser pass used a stub, not the real API — backend
  cross-tenant behavior is independently PROVEN in §11 above via real HTTP
  tests, but the two were not exercised together as one live end-to-end
  request in this pass).
- Screen-reader-specific announcement behavior (ARIA roles and labels were
  read from the DOM and confirmed present and correctly shaped — e.g.
  `role="dialog"`, `aria-modal="true"` — but no screen reader was actually
  run against the page).
- Real backend integration (the harness used a stub fetch throughout, per
  the fixture-caveat above) — API contract correctness between frontend and
  real backend is established by the two sides' TypeScript/Pydantic-shaped
  interfaces matching (`src/shared/api/user-administration.ts` against
  `app/api/user_administration.py`'s actual response shapes), read and
  compared directly, not by a live integrated request.

## 21. Combined readiness verdict

**PROVEN ready**, backend and frontend both. No application-code defect was
found in the backend (§14 of the backend section, above) — the backend
audit closed test-coverage gaps only. Two real frontend layout defects were
found by actual browser measurement (not present in any prior report, since
no prior stage in this project had run a real browser against this specific
page at a narrow viewport) and are now fixed, tested, and re-verified.

The one item this stage was explicitly told to leave open — whether a
`MANAGE_USERS` holder who cannot reach a target's role tier should be
blocked from narrowing that target's access — remains exactly as
documented: a real, deliberately unresolved product/security decision,
not a defect, not silently decided by this or any prior pass.

Nothing outside the harness (removed after use, never committed) and the
two `user-detail.tsx` fixes was changed on the frontend side. Nothing was
changed on the backend side beyond the five new tests. Organization CRUD,
platform-operator UI, DVR entity work, and CCTV/perception changes were not
started, per this stage's explicit scope.
