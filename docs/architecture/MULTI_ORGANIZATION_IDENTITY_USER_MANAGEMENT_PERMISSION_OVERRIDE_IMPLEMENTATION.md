# Multi-Organization Identity: Permission Override & Organization Lifecycle — Implementation Report

**Scope of this document.** This covers only Stages 1–4 of the ten-stage plan described in
`FINAL_ADMINISTRATION_MULTI_ORGANIZATION_IDENTITY_PERMISSION_CCTV_ARCHITECTURE_FREEZE.md`: the
`PermissionOverride` domain model, its migration, its integration into the authorization
chokepoint, and `Organization.status` lifecycle gating for login/read/write. It does **not**
claim the user-management APIs, role/override admin APIs, audit wiring, frontend UI, or
CCTV/streaming lifecycle gating (Stages 5–10) are done. Those are explicitly deferred; see
§8.

Labels used below: **PROVEN** (ran it, saw the result myself, in this pass), **TESTED**
(covered by an automated test that passed), **INFERRED** (a reasoned design choice not
directly dictated by the frozen architecture docs), **NOT VERIFIED** (not exercised in this
pass).

---

## 1. Executive summary

Added a three-state (INHERIT / GRANT / REVOKE) per-(user, permission) override table and
wired it into the single existing authorization chokepoint (`app/authorization/resolver.py:decide`),
so every existing route guard sees an already-adjusted effective permission set with zero
route-file changes. Added `Organization.status` (`active` / `suspended` / `archived`) as an
additive column with a safe default, and gated login/read/write (not streaming/perception) at
the same chokepoint plus the two places identity is established
(`AuthService.authenticate`, `decision_for_claims`). Added a domain-layer write path
(`app/authorization/overrides.py`) with two structural guards: no self-modification, no
cross-tenant reach. Added 29 new tests; the full suite (4,014 tests) shows 4,013 passing and
exactly one pre-existing, unrelated failure in `vision_os` — PROVEN via JUnit XML, see §7.

---

## 2. Files changed

| File | What changed |
|---|---|
| `app/authorization/model.py` | Added `OverrideState`, `OrganizationStatus` enums; `effective_permissions()`; changed `AccessDecision.permissions` default from `frozenset` (via `default_factory`) to `None` sentinel so an explicitly empty (fully-revoked) permission set is not silently re-derived from roles. |
| `app/authorization/resolver.py` | Added `parse_overrides()`, `parse_organization_status()`, `_suspend()`; `decide()` now composes `(role permissions ∪ GRANTs) − REVOKEs` and applies SUSPENDED filtering, both read from freshly loaded rows on every call. |
| `app/authorization/overrides.py` (new) | Domain-layer write path: `set_permission_override()`, `clear_permission_override()`, with `_guard()` enforcing no self-modification and no cross-tenant reach. Not an API route. |
| `app/users/models.py` | Added `Organization.status` column (`String(32)`, default `"active"`); added `PermissionOverride` model and `User.permission_overrides` relationship. |
| `app/auth/service.py` | `load_user_by_email` now eager-loads `permission_overrides`; `authenticate()` and `decision_for_claims()` both refuse when the organization is `ARCHIVED`. |
| `migrations/versions/20260904_d38dfad216a0_permission_overrides_and_organization_.py` (new) | The migration — see §3. |
| `tests/app/test_permission_overrides.py` (new) | 29 tests — see §6. |

Untouched, as instructed: `app/api/product.py`, `tests/app/test_persistence.py` (pre-existing
modified files), the five pre-existing untracked architecture reports, and everything under
`vision_os/`, `app/vision/`, `compliance/`. PROVEN via `git status --short` before and after —
only the seven files above (plus this report) are new/modified beyond the two pre-existing
ones, and no `vision_os`/`app/vision`/`compliance` path appears in `git status`.

No new API route was added. `app/authorization/overrides.py` is reachable only from Python
(tests, or future internal service code) — confirmed by grep: it is imported nowhere under
`app/api/`.

---

## 3. Migration

**File:** `migrations/versions/20260904_d38dfad216a0_permission_overrides_and_organization_.py`
**Revision:** `d38dfad216a0`, **down_revision:** `d4a1c8e37b52` (the prior head).

- **Single-head check before creating it:** PROVEN — `alembic heads` printed exactly
  `d4a1c8e37b52 (head)` before this migration was written.
- **Single-head check after creating it:** PROVEN — `alembic heads` prints exactly
  `d38dfad216a0 (head)`.
- **Upgrade:** PROVEN — `alembic upgrade head` against the project's configured Postgres
  database ran cleanly (`Running upgrade d4a1c8e37b52 -> d38dfad216a0`).
- **Downgrade/re-upgrade round trip:** PROVEN — `alembic downgrade -1` then
  `alembic upgrade head` both ran cleanly and left `alembic current` at `d38dfad216a0 (head)`.
- **Existing data defaults correctly:** PROVEN by direct query after upgrade —
  `select id, status from organizations` returned `('org-unityworks', 'active')`, with no
  manual backfill step. The column is `NOT NULL` with `server_default='active'`.
- **Purely additive, no existing table's data touched:** the migration creates one new table
  (`permission_overrides`) and adds one new column (`organizations.status`). It performs no
  `UPDATE`/`DELETE` against any existing table. `downgrade()` drops the column and the table,
  restoring the prior schema exactly.

`permission_overrides` columns: `id`, `user_id` (FK → `users.id`, `ON DELETE CASCADE`),
`permission`, `state`, `granted_at`, `granted_by` (nullable), with
`uq_permission_override_user_permission` unique on `(user_id, permission)` — mirroring
`uq_role_assignment`'s shape exactly.

---

## 4. `PermissionOverride` model and INHERIT/GRANT/REVOKE semantics

Model: `app/users/models.py`, class `PermissionOverride` (added after `AccessGrant`).
No row for a (user, permission) pair means **INHERIT** — the role's own answer stands; INHERIT
is never written as a row, only represented by absence. A stored row is always `GRANT` or
`REVOKE` (`app/authorization/model.py`, class `OverrideState`).

**Composition** — `app/authorization/model.py:effective_permissions()`:

```python
def effective_permissions(
    roles: frozenset[Role], *, granted: frozenset[Permission] = frozenset(),
    revoked: frozenset[Permission] = frozenset(),
) -> frozenset[Permission]:
    return (permissions_for(roles) | granted) - revoked
```

REVOKE is subtracted last, so it always wins over both the role and a GRANT for the same
permission (tested: `test_revoke_wins_over_role_and_grant_together`).

**Tenant scope.** `PermissionOverride` carries no `organization_id` column. Its only foreign
key is `user_id`, which already resolves to exactly one organization through `users`. This
means an override cannot exist for a user without that user already existing in one
organization — there is no path to construct a cross-tenant row. Verified no such path opens:
`app/authorization/overrides.py:_guard()` additionally checks `actor.organization_id !=
target.organization_id` as belt-and-suspenders for callers that load `actor`/`target` from
separate queries (TESTED: `test_cross_tenant_modification_is_refused`).

**A previously-existing bug this pass had to fix to make REVOKE meaningful:**
`AccessDecision.permissions` used to default via `field(default_factory=frozenset)`, and
`__post_init__` re-derived permissions from roles whenever `self.permissions` was falsy —
including an explicitly passed empty `frozenset()`. That meant a user whose role permissions
were *fully* revoked would have had their REVOKE silently undone by the fallback. Fixed by
changing the field to `frozenset[Permission] | None = None` and checking `is None` instead of
falsiness (`app/authorization/model.py:444-460`). TESTED:
`test_revoking_every_permission_a_role_holds_leaves_none`. No existing caller in the codebase
passed `permissions=` explicitly (grepped before changing), so this is backward compatible —
PROVEN by the full suite showing zero regressions.

---

## 5. Chokepoint integration

**The chokepoint, verified before writing anything:** `app/authorization/resolver.py:decide()`
is the only place a stored `User` becomes an `AccessDecision`; every route guard
(`app/api/dependencies.py:requires()` → `app/auth/service.py:require()`) calls
`decision.has(permission)`, reading `AccessDecision.permissions`. `decide()` is called from
exactly two places: `AuthService.authenticate()` (login) and `decision_for_claims()` (every
authenticated request, rebuilt from the database, never from the token) — both in
`app/auth/service.py`. This matches what the architecture-freeze document names as the
insertion point (§9/§17 of the freeze doc) — VERIFIED, not re-derived.

**What changed in `decide()`** (`app/authorization/resolver.py:170-206`, roughly): after
parsing roles, it now also parses the user's `permission_overrides` (`parse_overrides()`) and
computes `effective_permissions(roles, granted=granted, revoked=revoked)`. It then reads
`user.organization.status` via `parse_organization_status()`, and if `SUSPENDED`, strips every
`MANAGE_*` permission (`_suspend()`) from the result before constructing `AccessDecision`.

**No caching:** `decide()` is a pure function of the `User` object passed to it, and both call
sites load that `User` fresh from the database on every call (`load_user_by_email`, eagerly
loading `role_assignments`, `access_grants`, `permission_overrides`, and `organization`). There
is no cache in this path — confirmed by reading both call sites; nothing was found to
memoize or cache the decision. TESTED end-to-end over HTTP:
`test_an_override_takes_effect_on_the_very_next_request` writes a `GRANT` row mid-session and
confirms the very next `/api/v1/auth/me` call (same still-valid access token) reflects it.

**Zero route changes needed — verified, not assumed:** grepped every `requires(Permission.X)`
call site across `app/api/*.py` (administration, devtools, patron, reports, evaluation, …).
All of them depend on `require(decision, permission)` → `decision.has(permission)` →
membership in `decision.permissions`. None of them were touched in this pass, and the full
existing test suite (including `test_administration.py`, `test_foundation.py`'s
`TestDevToolsAuthorization`, `test_reports.py`, etc.) passes unchanged — PROVEN.

---

## 6. `Organization.status` and lifecycle gating

**Field:** `app/users/models.py`, `Organization.status: Mapped[str]`, default `"active"`.
**Enum:** `app/authorization/model.py`, `OrganizationStatus` — `ACTIVE` / `SUSPENDED` /
`ARCHIVED`.

**What is gated (confirmed by file:line):**

- `app/auth/service.py` (`AuthService.authenticate`, ~line 76-84): after the existing
  `is_active` checks, an `ARCHIVED` organization raises `InvalidCredentialsError` — login is
  refused, indistinguishable from a bad password (same enumeration-safety property the
  existing check already has).
- `app/auth/service.py` (`decision_for_claims`, ~line 130-149): an `ARCHIVED` organization
  raises `AuthenticationError` on every subsequent request using an already-issued token —
  so an archived org's users are locked out immediately, not just at their next login.
- `app/authorization/resolver.py` (`decide()`, ~line 188-192): a `SUSPENDED` organization
  strips every permission whose value starts with `manage_` from the effective set. This is
  the exact behavior the freeze document's §17 matrix specifies ("Write (`MANAGE_*`) | Allowed
  | Refused | Refused"). Reads (`VIEW_*`), and non-`MANAGE_*` write-adjacent actions such as
  `ACKNOWLEDGE_INCIDENTS`/`RESOLVE_INCIDENTS`/`EXPORT_REPORTS`/`REGISTER_DEMAND`, are **not**
  narrowed by this pass — INFERRED, following the freeze doc's matrix literally rather than
  inventing a broader "everything non-VIEW is a write" rule it did not specify. This is a
  narrower gate than "all writes"; noted as a limitation in §9.

**What is explicitly NOT gated, and confirmed untouched:** streaming, RTSP, perception,
camera-session lifecycle, and background scheduling. `git status`/`git diff` show no file
under `vision_os/`, `app/vision/`, or `compliance/` was modified in this pass. Camera sessions
continue to run for a SUSPENDED or even ARCHIVED organization's existing state exactly as
before — this pass only affects the HTTP/auth layer covered by `AccessDecision`.

**Existing data defaults correctly:** PROVEN, see §3 — `org-unityworks` reads `status='active'`
after migration with no manual step.

---

## 7. Anti-escalation

Implemented at `app/authorization/overrides.py:_guard()`, called by both
`set_permission_override()` and `clear_permission_override()`:

- **No self-modification:** `actor.id == target.id` raises `ScopeError` unconditionally.
  TESTED: `test_self_modification_is_refused`.
- **Cross-tenant impossibility:** structural (no `organization_id` on the override row; it can
  only ever apply to a `user_id` that already names one organization) plus an explicit
  `actor.organization_id != target.organization_id` check as defense in depth. TESTED:
  `test_cross_tenant_modification_is_refused`.

**Deliberately not built:** the asymmetric "grantor can only grant what they hold / need only
`MANAGE_USERS` to revoke" rule the freeze document calls out (§14) as a
`REQUIRES OWNER DECISION` item. That rule needs a request-scoped `AccessDecision` for the actor,
which only exists inside an HTTP request handled by an API route — and no such route exists
yet (Stage 5/6). Half-building it against a bare `User` object here would be a check that looks
complete but isn't, which is worse than the honest gap. Noted as a Stage 5/6 dependency, not
built.

---

## 8. Tests added

All in `tests/app/test_permission_overrides.py` (29 tests, PROVEN passing standalone and as
part of the full suite):

- `TestEffectivePermissions` (6): inherit-with-no-override, grant-adds, revoke-removes,
  revoke-wins-over-grant, multiple-roles-and-multiple-overrides-compose,
  revoking-every-permission-a-role-holds-leaves-none (the empty-permissions edge case).
- `TestParseOverrides` (3): unknown permission dropped, unknown state dropped, grant/revoke
  sorted correctly.
- `TestParseOrganizationStatus` (3): missing → ACTIVE, known values round-trip, unreadable
  value denies toward SUSPENDED rather than defaulting to ACTIVE.
- `TestDecideWithOverrides` (5): no-override-row, grant-row-widens, revoke-row-narrows,
  removing-the-override-reverts-to-role-behavior, inactive-user-reaches-nothing-regardless-of-overrides.
- `TestOrganizationLifecycleAtDecide` (3): active-unaffected, suspended-strips-manage,
  missing-status-column-value-reads-as-active.
- `TestOrganizationLifecycleOverHttp` (5): suspended-allows-login-and-reads-blocks-writes,
  archived-blocks-login, archived-blocks-an-already-issued-token, active-unchanged,
  override-takes-effect-on-the-very-next-request (immediate effect, no caching).
- `TestSetPermissionOverride` (4): grant-visible-through-decide, clearing-removes-it,
  self-modification-refused, cross-tenant-modification-refused.

---

## 9. Full test suite results

PROVEN via `alembic upgrade head`-applied Postgres DB and `python -m pytest -q
--junitxml=...`:

```
tests="4014" errors="0" failures="1"
```

**4,013 passed, 1 failed, 0 errors.**

The one failure — `tests/vision_os/understanding/test_ninety_b_configuration.py::TestSelectingItNeedsNoSourceEdit::test_no_production_module_names_the_model`
— asserts that a specific model name string is not hard-coded in
`vision_os/adapters/understanding/nvidia_vl.py:73`. PROVEN pre-existing and unrelated to this
pass: `git status --short` shows neither that source file nor that test file as modified by
this work, and this pass never touched anything under `vision_os/`. **Zero regressions from
this pass's changes** — the only failure present is one this pass did not create and could not
have caused, since it lives entirely inside code this pass's guardrails forbid touching.

Note on tooling: the default `python -m pytest -q` process on this Windows environment does
not reliably flush its final "`N passed, M failed in Xs`" summary line to a redirected file
(reproduced identically with `-u` and with `--tb=no`) — PROVEN not to affect correctness by
cross-checking against `--junitxml` output, which is written by a different code path
(`pytest_sessionfinish`) and gave the authoritative counts above.

---

## 10. Typecheck / lint results

- **Ruff** (the configured linter — `[tool.ruff]` in `pyproject.toml`): PROVEN clean on every
  file this pass touched (`app/authorization/model.py`, `app/authorization/resolver.py`,
  `app/authorization/overrides.py`, `app/users/models.py`, `app/auth/service.py`,
  `tests/app/test_permission_overrides.py`) — `ruff check` reports "All checks passed!".
  The new migration file fails Ruff's `I001` (import-sort) rule — PROVEN this is not a
  regression: every existing migration file (checked `20260902_c9d5f21ab340...py`) fails the
  identical rule for the identical reason (the module docstring precedes
  `from __future__ import annotations`), so the new migration matches the established, if
  imperfect, convention rather than introducing a new lint failure class.
- **Black:** PROVEN — `black --check` initially flagged `app/authorization/overrides.py` and
  `tests/app/test_permission_overrides.py`; both were reformatted with `black` and the test
  suite re-run afterward to confirm no behavior change (still 29/29 passing).
- **Mypy or any other typechecker:** NOT VERIFIED — no typechecker is configured in this
  project (no `mypy` section in `pyproject.toml`, no `mypy` installed in the virtualenv). No
  typecheck command exists to run.

---

## 11. Deferred to Stages 5–10, and why

- **User-management and role/override admin HTTP APIs (Stage 5/6).** This pass is explicitly
  scoped to domain model + migration + chokepoint + tests, reachable only from Python. The
  `set_permission_override`/`clear_permission_override` functions exist and are tested, but
  nothing calls them from an HTTP route.
- **The full asymmetric anti-escalation rule** (grantor needs `MANAGE_USERS` in the target's
  tenant, and for GRANT must hold the permission being granted) — needs the admin API's own
  request-scoped `AccessDecision` to mean anything; see §7.
- **Audit wiring for `permission.granted`/`permission.revoked` events** — the freeze document
  (§20-ish, audit trail section) marks new `AuditAction` values as
  `REQUIRES OWNER DECISION`/net-new; not built here, and no audit call exists in
  `app/authorization/overrides.py`.
- **Frontend UI** — out of scope by explicit instruction; the frontend repo was not touched.
- **CCTV/streaming lifecycle gating for SUSPENDED/ARCHIVED** — explicitly deferred by the
  freeze document itself (§17, "REQUIRES OWNER DECISION, unresolved by design per task
  instructions") and by this pass's own guardrails. Nothing under `vision_os/`, `app/vision/`,
  or camera-session/runtime code was touched; existing camera sessions for a suspended or
  archived organization are unaffected by this pass.
- **Full cross-repo regression** (frontend, validation console) — out of scope; only this
  backend repo was touched.

---

## 12. Known limitations

- **SUSPENDED gating is `MANAGE_*`-prefix-based, not a general read/write classification.**
  Permissions like `ACKNOWLEDGE_INCIDENTS`, `RESOLVE_INCIDENTS`, `DELETE_EVIDENCE`,
  `EXPORT_REPORTS`, `REGISTER_DEMAND`, and `ACCESS_DEVTOOLS` are write-adjacent or
  engineering-surface permissions that are **not** stripped by a SUSPENDED organization in this
  pass, because the frozen architecture document's own behavior matrix (§17) names only
  `MANAGE_*` as the gated category. INFERRED that this narrower reading is intentional rather
  than an oversight in the freeze document; NOT VERIFIED against an owner decision, since none
  is recorded for this specific boundary.
- **An unreadable `Organization.status` value denies toward `SUSPENDED`, not `ACTIVE` or
  `ARCHIVED`.** INFERRED as the middle-ground "unknown values deny, never widen" reading
  consistent with `parse_camera_scope`/`parse_roles`'s existing discipline, but this specific
  choice (rather than `ARCHIVED`, which would be the maximally restrictive reading) is not
  dictated by any of the five prior architecture reports.
- **An unreadable `PermissionOverride.state` or unknown `.permission` value is dropped
  entirely** (treated as neither GRANT nor REVOKE) rather than defaulting to REVOKE. INFERRED;
  this means a corrupted REVOKE row would silently under-restrict rather than over-restrict.
  Flagged here rather than silently decided.
- **No audit trail entry is written when an override is set or cleared** in this pass — see
  §11.
- The Windows test environment's stdout-flush quirk noted in §9 means a bare
  `python -m pytest -q > file` redirect should not be trusted for final counts on this machine;
  `--junitxml` is the reliable source.
