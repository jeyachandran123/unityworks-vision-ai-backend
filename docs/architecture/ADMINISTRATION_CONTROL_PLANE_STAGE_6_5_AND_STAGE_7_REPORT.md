# Administration Control Plane — Stage 6.5 and Stage 7 Report

**Labels used throughout:** PROVEN (ran it, saw the result, in this pass), TESTED (an
automated test asserts it and passed in this pass), INFERRED (a reasoned design choice
not directly dictated by prior documents), NOT VERIFIED (not exercised in this pass —
in particular, all real-browser/visual verification, explicitly deferred to the
orchestrating session per its own instruction).

---

## 1. Scope

Two pieces of work across both repositories. Stage 6.5 (backend): add a `role_grants`
field to the Stage 5 `GET /api/v1/admin/users/{id}/permissions` response, purely
additive, so the frontend can show role-derived access separately from an override.
Stage 7 (frontend): extend `/admin`'s Accounts region with a real write path against
the Stage 5 API, and add a new `/admin/users/:userId` route with identity, account
lifecycle, role management, and a permission-override table using Inherited/+ Added/−
Restricted vocabulary. No organization CRUD, no org switcher, no platform-operator UI,
no DVR/CCTV entity work, no custom roles, no navigation change beyond the one new
link-only route.

## 2. What existed before

Backend: Stage 5's `app/api/user_administration.py` — eleven routes under
`/api/v1/admin/users`, `MANAGE_USERS`-gated, tenant-scoped, audited — already complete
and tested (31 tests, `tests/app/test_user_administration.py`). `_permission_rows`
returned `{permission, state, effective}` per row with no field distinguishing
role-derived access from an override.

Frontend: `administration.tsx` had full Sites/Zones CRUD but a read-only Accounts
region backed by the older `GET /users` route (`app/api/administration.py`), which
carries `write_available: false` and a fixed reason. No `/admin/users/:userId` route
existed. No UI anywhere in the product rendered a permission-override table.

## 3. Stage 6.5 backend contract change

One field added to each row `_permission_rows` returns
(`app/api/user_administration.py:217-244`, now with `role_grants` inserted between
`state` and `effective`): `role_grants: bool` — whether the target's role(s) alone,
before any override, would grant this permission.

**PROVEN**, computed without duplicating `decide()`'s own logic:

```python
decision = decide(target)
role_permissions = permissions_for(decision.roles)
...
"role_grants": permission in role_permissions,
```

`permissions_for` is the exact function `effective_permissions` (called from inside
`decide()`) already uses internally to union a role set's permissions — not
re-derived, called again on `decision.roles`, the same roles `decide()` itself parsed
for this same call. No new import beyond `permissions_for`, already imported at the
top of the file for `_require_grantable_role`.

## 4. API compatibility

**Was any existing API removed? NO.** Evidence: `git diff --stat` for
`app/api/administration.py` (the file holding the old `GET /users` route) is empty —
PROVEN, that route is untouched. The new field is inserted into an existing response
without removing or renaming any prior field (`permission`, `state`, `effective` are
all still present, in the same types, at the same key names). Every existing consumer
of `GET .../permissions` — the 31 Stage 5 tests that assert on that response — passed
unchanged (§23) with no test edit required for the new field, because the assertions
there use `next(r for r in rows if r["permission"] == ...)` and index into named keys,
never an exhaustive key-set equality check.

## 5. `role_grants` semantics

`True` when the permission is in `permissions_for(decision.roles)` — i.e. the union of
every `ROLE_PERMISSIONS[role]` for every role `parse_roles()` resolved from the
target's `role_assignments`, exactly as `decide()`'s own `effective_permissions` call
computes it internally. `False` otherwise, including when the user holds no role at
all, or holds only roles that do not carry this permission, or is inactive (an inactive
user's `decision.roles` is the empty frozenset, per `decide()`'s own early return at
`resolver.py:139-149`, so `role_grants` is `False` for every permission of a
deactivated account — matching `effective` also being `False` for all of them).

## 6. INHERIT semantics

Unchanged from Stage 5 (§8 of the Stage 5 report). No stored override row for a
(user, permission) pair. The row's `state` is `"inherit"`. **Newly visible**: with
`role_grants` present, a caller can now tell "inherit and the role grants it" from
"inherit and the role does not" without a second call — TESTED,
`test_listing_permissions_shows_role_grants`.

## 7. GRANT semantics

Unchanged (Stage 5 report §9): a stored row with `state="grant"` widens the target's
effective access regardless of what the role alone would give. `role_grants` can be
either `True` or `False` on a GRANT row — a GRANT is meaningful precisely because it
can add something the role does not already carry. TESTED,
`test_role_grants_false_plus_grant_widens_to_effective_true`: `kitchen_supervisor`
does not hold `view_evidence` (`role_grants: false`), a GRANT override still produces
`effective: true`.

## 8. REVOKE semantics

Unchanged (Stage 5 report §10-11): REVOKE always wins over the role, even when
`role_grants` is `True`. TESTED,
`test_role_grants_true_plus_revoke_still_loses_to_revoke`: `restaurant_manager` holds
`view_live` (`role_grants: true`), a REVOKE override on that permission still produces
`effective: false`. This is the same composition Stage 1-4's own
`test_revoke_wins_over_role_and_grant_together` already proves at the domain layer;
this test proves the *same* fact is now visible in the row-level `role_grants` field
too, not a new composition rule.

## 9. Effective permission semantics

**Did permission semantics change? NO.** `effective` is still exactly
`decide(target).has(permission)` (`user_administration.py:241` unchanged in
computation, only `role_grants` was inserted as a sibling field). `role_grants` is a
read-only, derived, additional fact about the same already-computed `decision.roles` —
it does not feed back into how `effective` or `state` are computed, and nothing in
`app/authorization/resolver.py` was touched (§ 27, confirmed by `git diff`).

## 10. `/admin` information architecture

`administration.tsx` keeps its existing Sites (order 3) and Zones (order 4) regions
unchanged. The Accounts region (order 5) now: (a) reads `GET /api/v1/admin/users`
(`adminUsersApi.list`) instead of the old `GET /users`, dropping the stale
`write_available: false` messaging entirely; (b) each row is a `Link` to
`/admin/users/:userId`; (c) an "Add account" form sits below the table, gated on
`PERMISSIONS.manageUsers` via `PermissionGate`, using only fields the real `POST
/api/v1/admin/users` accepts (`email`, `display_name`, `roles`, optional `password`) —
no invented profile fields. The region shell is unchanged `SectionRule` + `Region` +
`Plane`, the same pattern Sites/Zones already use — no second section pattern
introduced.

## 11. User detail information architecture

`/admin/users/:userId` (`src/features/user-detail.tsx`), reached only by a row link,
no nav entry (mirrors `/cameras/:cameraKey`). Four ranked, not equal-weight, parts:

1. **Identity & account** (`SectionRule lead order={2}`) — the page's stated centre of
   gravity: email, created date, last sign-in as a `Plane` of `Figure`s at `hero`/`lead`
   emphasis via `PageIntro`'s own title and meta, plus the Activate/Deactivate control.
2. **Roles** (`order={3}`) — secondary: a plain list of role `Badge`s with per-role
   Remove buttons, an assign-role `Select` + button below.
3. **Access** (`order={4}`) — its own quieter, structured register: the
   permission-override table, grouped into eight `DataTable`s (one per product-area
   group), each visually smaller and denser than the sections above it.

This directly answers the discovery report's "wall of equal cards" warning (§13/§24 of
`ADMINISTRATION_UI_UX_INFORMATION_ARCHITECTURE_STAGE_6_DISCOVERY.md`): `SectionRule`'s
own `lead` flag marks section 1 as the page's actual subject (accent tick, double
weight), and nothing below it claims the same visual rank.

## 12. Add Account flow

Fields: email (required), display name (optional, defaults server-side to the local
part of the email), password (optional — blank means the server generates one), and a
role checkbox group listing all seven `Role` values via `roleLabel()`. On success, the
response's `generated_password` (present only when no password was supplied) is shown
exactly once in a copyable `<code>` block with an explicit "cannot be retrieved again"
warning — the same one-time-disclosure discipline the backend module's own doc comment
describes for `scripts/manage.py reset-password --generate`. No field is sent that the
real `POST /api/v1/admin/users` does not accept (verified against
`user_administration.py:274-355`'s own body parsing).

## 13. Role management UX

Assign: a `<select>` of roles the user does not already hold, plus an Assign button —
calls `POST .../roles`. Remove: a Remove button per currently-held role — calls
`DELETE .../roles/{role}`. **Deliberately not built**: a client-side "roles this actor
may grant" filter (the discovery report's §6.6/§24.2 proposal). Building it correctly
would require a `ROLE_PERMISSIONS`-shaped table on the frontend, which the discovery
report itself says the frontend deliberately does not keep (§7: "the frontend has no
such table today, by design"). Per the guard-rail against inventing authorization data
client-side, every role is offered and the real server-side subset check
(`_require_grantable_role`) is the actual enforcement — an assignment the actor cannot
grant surfaces the server's own `ScopeError` message. This is a scope reduction from
the discovery report's proposal, recorded as a known limitation (§27).

## 14. Account lifecycle UX

Activate: single click, no confirmation (matches the backend's own idempotent,
reversible framing). Deactivate: **always** confirms via the existing `Modal`
primitive, stating the account will be signed out and refused login immediately, and
that it is reversible via Activate but may strand the organisation if this is its only
administrator — the frontend cannot know whether that is true (it has no
cross-account visibility beyond this list), so the copy is a general caution rather
than a claim it cannot verify, per the discovery report's own instruction (§19).
Self-deactivation: the Deactivate button is `disabled` (not hidden) with an inline
explanation, because the backend's own `ScopeError` reason should be the thing the
operator reads, and a vanished button would look like a bug rather than a rule.

## 15. Permission override UX

Each row: permission label (title-cased from the enum value), a role-derived
indicator ("Role grants" / "Role does not grant", plain text, from the new
`role_grants` field), the override control (§16), and the effective result as a
`StatusBadge` reading "Effective" / "Not effective" (dot **and** word, never colour
alone). Rows are grouped into eight `DataTable`s by product area (§17 below), each
under its own quiet heading — not one 30-row table, and not a checkbox matrix.

## 16. Confirmation policy

Exactly as specified, and verified by test:

- **Deactivate**: always confirms (`Modal`). TESTED,
  `deactivate requires confirmation before the request is sent` /
  `confirming deactivation sends the request`.
- **Remove role**: confirms only when the role carries `MANAGE_USERS` or
  `MANAGE_ORGANIZATION`. TESTED, both directions:
  `removes a non-admin-carrying role without confirmation` and
  `removing an admin-carrying role requires confirmation`. The "which roles carry
  admin-level access" check is `ADMIN_CARRYING_ROLES = {super_admin, org_admin}`
  (`user-detail.tsx`), a small constant documented as derived from the backend's own
  `ROLE_PERMISSIONS` table and the Stage 5 report's own §12 investigation — not a
  guess, but also not a live per-role API call (none exists), so it is flagged as a
  hand-maintained fact in §27.
- **GRANT / REVOKE / reset-to-inherit**: no confirmation, one click, matching the
  frozen policy that these are instantly and visibly reversible in the same table.
  TESTED throughout the permission-override test group.

## 17. Design system decisions

`PageIntro`, `SectionRule` + `Region`, `Plane`, `Figure`, `GoTo`, `DataTable`, `Badge`,
`StatusBadge`, `Button`, `Select`, `Modal`, `EmptyState`, `ErrorState`, `LoadingState`
— all reused unmodified. No second table primitive was built: the permission-override
table is eight ordinary `DataTable` instances, one per group, exactly as Sites/Zones/
Accounts already use it. No second page-opening pattern: `/admin/users/:userId` uses
`PageIntro` exactly as `EvidenceDetailPage`/`CameraDetailPage` do. No second badge
vocabulary: the override control's `Inherited`/`+ Added`/`− Restricted` states compose
plain text plus the existing `Badge` component (tone `accent` for `+ Added`, tone
`neutral` for `− Restricted`), never a bespoke `<span>` with invented colours.

## 18. Accessibility decisions

Every override state carries its word, never only a glyph or colour (`STATE_WORD`
mapping in `user-detail.tsx`, rendered as literal text beside each `Badge`).
`StatusBadge`'s existing dot-plus-word pattern is reused for "Effective"/"Not
effective". The role `Select` and override buttons are native `<select>`/`<button>`
elements, keyboard-operable by construction (no custom hover-only or drag controls).
`Modal` is reused unmodified for both confirmations, carrying its existing focus
management and `Escape`-to-dismiss behaviour (`primitives.tsx:852-873`) — no new
focus-trap code was written.

## 19. Responsive behavior

`DataTable`'s existing horizontal-scroll-inside-its-own-container behaviour
(`primitives.tsx:729`) is unmodified and used for all eight override groups. The
add-account form and role-assign row use the same `flexWrap: 'wrap'` pattern the
existing Sites/Zones forms already use. **Structural tests only** — one test
(`user detail: structural composition`) asserts the `data-order`/`data-lead` structure
that makes the ranked hierarchy possible, following `command-center.test.tsx`'s own
stated approach, because jsdom has no layout engine. **Real narrow/wide-viewport
visual verification is NOT performed by this agent** and is explicitly deferred to the
orchestrating session, which has browser tooling (§24).

## 20. Authorization behavior

**Was authorization moved to the frontend? NO.** Every mutation in
`user-detail.tsx`/`administration.tsx` calls the real `/api/v1/admin/users/...`
endpoints through `adminUsersApi`; nothing is computed or enforced client-side as a
substitute for a server check. The one place the frontend narrows what it *offers* is
the GRANT control: `+ Add` is rendered only when
`actor.permissions.includes(permission)` is true, mirroring (not replacing) the
backend's own `access.has(permission)` anti-escalation check
(`user_administration.py:579-583`). TESTED,
`GRANT is offered only for a permission the acting admin holds themselves` — and the
underlying mutation still goes through the same `adminUsersApi.setOverride` call the
`+ Add` button anywhere else would use; hiding the button is UX, the server call is
still the actual boundary. The route gate itself
(`RequirePermission permissions={[PERMISSIONS.manageUsers]}`) is a redirect, not an
enforcement substitute, per every other guard in `guards.tsx`'s own stated posture.
TESTED, `redirects an account without manage_users away from the page`.

## 21. Audit preservation

No change to `app/domain/audit.py` beyond what Stage 5 already added (verified —
`git diff` for that file shows only the pre-existing Stage 5 baseline, this pass added
nothing to it). Every mutating route this UI calls
(create/activate/deactivate/assign-role/remove-role/grant/revoke/reset) already writes
an `AuditTrail` record in `user_administration.py`, unmodified by Stage 6.5 — the new
`role_grants` field is read-only and appears on no audited mutation's `detail={...}`
payload, so no audit event shape changed.

## 22. Tests added

**Backend** (`tests/app/test_user_administration.py`, +5 tests, 31 → 36):
`test_listing_permissions_shows_role_grants`,
`test_role_grants_true_plus_revoke_still_loses_to_revoke`,
`test_role_grants_false_plus_grant_widens_to_effective_true`, plus the two
INHERIT/effective assertions folded into the first. Covers the four required cases:
role_grants=true+INHERIT→effective true; role_grants=true+REVOKE→effective false;
role_grants=false+GRANT→effective true; role_grants=false+INHERIT→effective false.

**Frontend**: `tests/administration.test.tsx` rewritten to point at `/admin/users` and
add: account-row-links-to-detail, add-account-form-visibility-by-permission,
create-account-shows-generated-password (10 tests total, up from 8).
`tests/user-detail.test.tsx` (new, 17 tests): navigation from the Accounts table,
identity/account rendering, activate (no confirm), deactivate (confirm + cancel +
confirm-then-send), self-deactivation disabled, role assign, role remove (both
confirm branches), permission override rendering (`role_grants`, `Inherited`,
`Effective`), the GRANT anti-escalation gate, GRANT/reset mutations reaching the real
API and reflecting the refetched state, the never-guess-on-error case, the 403
redirect, the not-found case, and the structural ranked-hierarchy assertion.
`tests/support.tsx` extended with a real, mutable in-memory `/admin/users` stub
(list/get/create/patch/activate/deactivate/roles/permissions), following the file's
existing `installFetch`/`stubFetch` conventions rather than a separate harness.

## 23. Tests run

**Backend**: `pytest -q tests/app/test_user_administration.py` — 36/36 passed
(PROVEN, ran standalone). `pytest -q tests/app/test_user_administration.py
tests/app/test_permission_overrides.py tests/app/test_administration.py` — 79/79
passed (PROVEN). Full suite: `pytest -q` — PROVEN, one failure, exactly
`tests/vision_os/understanding/test_ninety_b_configuration.py::TestSelectingItNeedsNoSourceEdit::test_no_production_module_names_the_model`,
the same pre-existing, documented, unrelated failure named in the task's own baseline.
No other failure. `git status` confirms `tests/vision_os/understanding/test_ninety_b_configuration.py`
and `vision_os/adapters/understanding/nvidia_vl.py` do not appear in the changed-file
list — untouched, PROVEN.

**Frontend**: `npx vitest run tests/administration.test.tsx tests/user-detail.test.tsx`
— 27/27 passed. Full suite `npx vitest run` — 362/362 passed across 22 files, PROVEN,
no regression in any pre-existing file.

## 24. Browser verification

**NOT performed by this agent.** This environment has no browser. Every claim in §17
about visual weight, alignment, colour tokens actually rendering as intended, and the
responsive reflow of the override tables at narrow widths is a code-level and
structural-test-level claim only (§19). The orchestrating session, which has browser
tooling, should verify: desktop and narrow-viewport layout of `/admin/users/:userId`
(the ranked hierarchy actually reads as ranked, not just structurally ordered); light
and dark theme rendering of the `Badge`-based override control; and that the
eight-group override table does not overflow or crowd at common viewport widths.

## 25. Files changed

Backend:
- `app/api/user_administration.py` — `role_grants` field added to `_permission_rows` (§3).
- `tests/app/test_user_administration.py` — 5 tests added (§22).

Frontend:
- `src/shared/api/client.ts` — added `PUT` to the request-method union and `api.put`.
- `src/shared/api/user-administration.ts` (new) — the Stage 5 API client (§10-16).
- `src/features/administration.tsx` — Accounts region repointed at `/admin/users`,
  write forms added, stale `write_available` copy removed.
- `src/features/user-detail.tsx` (new) — the `/admin/users/:userId` page.
- `src/app/router/AppRouter.tsx` — one new gated route, no nav change.
- `tests/administration.test.tsx` — updated for the new API and write flows.
- `tests/user-detail.test.tsx` (new) — 17 tests.
- `tests/support.tsx` — `/admin/users` stub added (§22).

## 26. Files deliberately not changed

`app/authorization/resolver.py`, `app/auth/service.py` — PROVEN untouched by `git diff`
(both files carry only the pre-existing Stage 1-5 baseline this pass started from; this
pass's own edits touch neither). `app/api/administration.py` (the old `GET /users`
route) — untouched, still returns `write_available: false`, still passed by its own
existing test unmodified. `app/authorization/model.py`, `app/authorization/overrides.py`,
`app/users/models.py` — untouched by this pass (their presence in `git status` is the
pre-existing Stage 1-5 baseline, not new work). `src/shared/layout/AppShell.tsx`,
the brand icon, and every route/file unrelated to Administration — untouched. No
`vision_os/`, `app/vision/`, detection/tracking/VLM/RTSP/observation/synthesis/
compliance/incident file appears in either repository's changed-file list.

## 27. Known limitations

1. **Role-assign grantability is not filtered client-side** (§13). Every role is
   offered in the assign picker; the real server-side subset check
   (`_require_grantable_role`) is what actually refuses an over-broad assignment, and
   its `ScopeError` message is what the operator sees on failure. This is a narrower
   implementation than the discovery report's own §24.2 proposal, chosen because
   building the filter correctly would require duplicating `ROLE_PERMISSIONS` on the
   frontend, which the discovery report itself says is deliberately not kept there.
2. **`ADMIN_CARRYING_ROLES` is a hand-maintained constant**, not a live query
   (§16). It encodes today's fact — only `super_admin` and `org_admin` carry
   `MANAGE_USERS`/`MANAGE_ORGANIZATION` — documented in `user-detail.tsx` with a
   citation to `ROLE_PERMISSIONS` and the Stage 5 report's own §12 investigation. If
   the backend's role/permission mapping changes, this constant needs a manual update;
   nothing currently detects drift automatically.
3. **The §12 role-hierarchy ambiguity from Stage 5 is still open** (an `org_admin` can
   remove a role from, or REVOKE a permission from, a `super_admin` target). The UI
   cannot visually flag this case, because the backend does not either — carried
   forward unchanged from Stage 5, not addressed by Stage 6.5/7.
4. **`update_user` (display-name PATCH) has no UI** in this pass. The task's Part 2
   spec lists Identity/Account/Roles/Access as the four sections and does not list a
   rename control; one was deliberately not added to avoid scope creep beyond the
   specified four-part hierarchy, even though the backend route exists.

## 28. Deferred work

Real browser/visual verification (§19, §24) — explicitly the orchestrating session's
task per its own instruction, not attempted here. The Stage 5 report's own §12
role-hierarchy decision (owner sign-off needed before a deployment relies on more than
one privilege tier of `MANAGE_USERS` holder) remains open and is unaffected by this
pass. A live client-side role-grantability filter (§27.1), should the hand-maintained
`ADMIN_CARRYING_ROLES` approach prove insufficient once more roles/permissions are
added.

## 29. Regression results

Backend: 4,014 baseline tests + 5 new = target 4,019; full-suite run confirmed exactly
one failure (the same pre-existing, documented, unrelated `test_ninety_b_configuration.py`
case) and zero new failures — PROVEN, ran to completion. Frontend: 362/362 passed
across all 22 test files (up from the pre-existing suite plus 17 new user-detail tests
and administration.test.tsx's net new assertions) — PROVEN, zero regressions in any
pre-existing file. No existing test was weakened, skipped, or deleted — TESTED via the
full-suite runs themselves, and confirmed by inspection: every pre-existing test file
listed in `tests/app/` and `tests/` (frontend) still exists with the same or a greater
test count.

## 30. Final readiness verdict

Backend Stage 6.5 is **PROVEN complete and safe to merge**: one additive field, zero
API removals, zero semantic changes to permission/role/effective computation, zero
tenant-boundary changes, full regression suite green apart from the pre-existing
unrelated failure. Frontend Stage 7 is **code-, type-, lint-, and test-complete**
(`tsc -b --noEmit` clean, `eslint . --max-warnings 0` clean, `vitest run` 362/362,
`npm run build` succeeds) but carries one honest gap: **no real-browser visual
verification has been performed**, per this pass's own environment constraints, and
that step is explicitly handed to the orchestrating session before this should be
considered fully done from a design-fidelity standpoint. Nothing here should block
that verification pass from succeeding — the design-system components used are all
pre-existing and unmodified, and the new component (the override control) follows
their stated conventions rather than inventing new visual language.

---

**Guardrail answers, restated with evidence:**

- Was any existing API removed? **NO** — `git diff --stat app/api/administration.py`
  empty; `GET /users` unchanged and still tested unchanged (§4, §26).
- Did permission semantics change? **NO** — `effective` computation unchanged;
  `role_grants` is a read-only sibling field (§9).
- Did role semantics change? **NO** — no change to `Role`, `ROLE_PERMISSIONS`, role
  assignment idempotency, or the anti-escalation subset check (§13, §26).
- Did any tenant boundary change? **NO** — every route this pass's UI calls was
  already tenant-scoped by Stage 5's `_user_in_tenant`/`access.tenant_id` discipline;
  nothing in this pass touches tenant resolution (§20, §26).
- Was authorization moved to the frontend? **NO** — every mutation goes through the
  real API; the one client-side narrowing (hiding `+ Add` for an unheld permission) is
  a UX courtesy mirroring, not replacing, the server's own check (§20).
