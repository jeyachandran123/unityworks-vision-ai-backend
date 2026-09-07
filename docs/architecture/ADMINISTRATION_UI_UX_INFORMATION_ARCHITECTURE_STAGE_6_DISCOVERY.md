# Administration UI/UX Information Architecture — Stage 6 Discovery

**Status:** Read-only discovery and architecture-design pass. No frontend or backend
code was modified to produce this document — the only file created is this one.
Labels used throughout: **CONFIRMED** (read directly in this pass, file:line cited),
**INFERRED** (a reasoned conclusion from confirmed facts), **PROPOSED** (a design
recommendation, not yet approved), **DECISION REQUIRED** (a genuine fork needing
owner sign-off before Stage 7 implementation).

Backend Stages 1-5 are treated as complete and verified — not re-audited. Tenant
isolation is not re-checked in this pass; it is carried from
`FINAL_ADMINISTRATION_MULTI_ORGANIZATION_IDENTITY_PERMISSION_CCTV_ARCHITECTURE_FREEZE.md`
(backend repo) as CONFIRMED background.

---

## 1. Executive summary

Stage 5 built a complete, `MANAGE_USERS`-gated, tenant-scoped HTTP surface for user
management, role assignment, and INHERIT/GRANT/REVOKE permission overrides — eleven
routes under `/api/v1/admin/users`
(`unityworks-vision-ai-backend/app/api/user_administration.py:100-104,250-656`). The
frontend's `AdministrationPage`
(`unityworks-vision-ai-frontend/src/features/administration.tsx`) does not call any of
them. It still calls the older, separate, read-only `GET /users` route
(`organizationApi.users`, `unityworks-vision-ai-frontend/src/shared/api/observations.ts:206`,
hitting `app/api/administration.py`'s pre-existing `list_users`, confirmed unchanged by
Stage 5 §5), and renders the server's own `write_available: false` sentence
(`administration.tsx:438-447`) — which is now **factually stale**: account creation is
available, at a different path the frontend has never been pointed at.

This report's core recommendation is **not** a new admin section. It is: extend the
existing `Accounts` region of `administration.tsx` with real write actions (create,
role assign/remove, activate/deactivate) driven by the new `/api/v1/admin/users`
client, and add exactly one new route — `/admin/users/:userId` — for the
per-user permission-override view, following the same `/resource/:id` pattern
`CameraDetailPage` and `EvidenceDetailPage` already establish
(`unityworks-vision-ai-frontend/src/app/router/AppRouter.tsx:189-190,179`). No
Level A/platform-operator UI, no organization switcher, and no new top-level nav
section are proposed — Stage 5 built user management within the single existing
organization, not Organization CRUD, and inventing chrome for a concept
(multiple organizations, a platform operator) that has no backend surface yet would
be exactly the kind of UI-races-ahead-of-API error the freeze document's §13
warns against for `super_admin`.

## 2. Current frontend administration architecture

`AdministrationPage` (`unityworks-vision-ai-frontend/src/features/administration.tsx`)
is a single-route page with three `Region`s: Sites (`order={3}`), Zones (`order={4}`),
Accounts (`order={5}`) — `administration.tsx:202-451`. Sites and Zones are full CRUD
(list + create) gated on `PERMISSIONS.manageOrganization`
(`administration.tsx:250,317`). Accounts is list-only: `userColumns` renders email,
name, roles-as-badges, active/disabled `StatusBadge`, last sign-in
(`administration.tsx:152-182`), with no create form and no per-row action. The page's
own doc comment states the asymmetry is deliberate:
*"Restaurants and zones are created and renamed here... Users are **listed only**"*
(`administration.tsx:6-7`) — CONFIRMED written before Stage 5 existed, since it cites
the same `write_available: false` reasoning Stage 5 §5 says the backend still returns
unchanged from the pre-Stage-5 route.

## 3. Current backend administration capabilities from Stage 5

CONFIRMED, `unityworks-vision-ai-backend/app/api/user_administration.py`, router
prefix `/api/v1/admin/users`, single router-level dependency
`Depends(requires(Permission.MANAGE_USERS))` (`:100-104`) covering all eleven routes:

| Method & path | Line | Purpose |
|---|---|---|
| `GET /api/v1/admin/users` | `:250-265` | List users in caller's org |
| `GET /api/v1/admin/users/{user_id}` | `:268-271` | Get one user |
| `POST /api/v1/admin/users` | `:274-355` | Create a user (optional password, else generated once) |
| `PATCH /api/v1/admin/users/{user_id}` | `:358-397` | Update `display_name` only |
| `POST /api/v1/admin/users/{user_id}/activate` | `:400-419` | Activate |
| `POST /api/v1/admin/users/{user_id}/deactivate` | `:422-454` | Deactivate (refuses self) |
| `POST /api/v1/admin/users/{user_id}/roles` | `:460-497` | Assign a role (idempotent) |
| `DELETE /api/v1/admin/users/{user_id}/roles/{role_value}` | `:500-538` | Remove a role (idempotent) |
| `GET /api/v1/admin/users/{user_id}/permissions` | `:544-552` | Every permission: state + effective |
| `PUT /api/v1/admin/users/{user_id}/permissions/{permission_value}` | `:555-619` | Set GRANT or REVOKE |
| `DELETE /api/v1/admin/users/{user_id}/permissions/{permission_value}` | `:622-656` | Reset to INHERIT |

Wire shapes CONFIRMED from `_user_to_wire` (`:203-214`): `{id, email, display_name,
is_active, roles: string[], created_at, last_login_at}` — no `password_hash` ever.
`create_user` additionally returns `generated_password` exactly once when no password
was supplied (`:349-354`). `_permission_rows` (`:217-244`) returns, per permission:
`{permission, state: "grant"|"revoke"|"inherit", effective: boolean}` — CONFIRMED this
is the single authoritative source for INHERIT/GRANT/REVOKE + effective, already
composed server-side via `decide()`, never reconstructed client-side (`:220-226`).

Anti-escalation rules CONFIRMED from the module docstring (`:32-78`) and enforced in
code: no self-modification of roles/overrides/deactivation (`:438-439,477-478,519-520`,
and `overrides.py`'s own `_guard`); a grantor may only assign a role or GRANT a
permission they themselves hold (`_require_grantable_role`, `:189-200`, and the
`access.has(permission)` check at `:579-583`); removal/REVOKE carry no such
precondition (`:510-514`, module docstring rule 2). One item is explicitly left open
by the backend itself, not resolved here: whether a `MANAGE_USERS` holder may
remove a role or REVOKE a permission from a target who holds it via a role the actor
cannot themselves reach (docstring `:64-78`) — flagged **DECISION REQUIRED** upstream,
carried into §19 below as a UX consideration (the UI cannot visually distinguish this
case from an ordinary removal, since the backend does not either).

## 4. Existing routes and navigation audit

CONFIRMED, `unityworks-vision-ai-frontend/src/app/router/AppRouter.tsx:199-207`: one
route, `/admin` → `AdministrationPage`, gated `RequirePermission
permissions=[PERMISSIONS.manageUsers, PERMISSIONS.manageOrganization]` (mode `any`,
the default). CONFIRMED, `unityworks-vision-ai-frontend/src/app/router/navigation.ts:247-276`:
one nav section, `id: 'platform'`, `label: 'Platform'`, containing one item, `id:
'admin'`, `path: '/admin'`, `permissions: [manageUsers, manageOrganization]`, hint
*"Restaurants, zones, users and roles"* (`:270-276`). Object-route precedent CONFIRMED
elsewhere: `/cameras/:cameraKey` (`AppRouter.tsx:190`), `/evidence/:evidenceRef`
(`:179`), `/incidents/:incidentId` (`:168`) — every one same-permission-as-its-list,
stated explicitly in that file's own comments (`:161-167,171-178`). No `/admin/*`
sub-route exists today.

## 5. Existing permission and role UI audit

CONFIRMED: the only permission-facing UI today is `PermissionGate`
(`unityworks-vision-ai-frontend/src/app/permissions/guards.tsx:74-85`), used in
`administration.tsx` twice to hide the Sites/Zones create forms
(`:250,317`), and the read-only roles-as-`Badge` list in the Accounts table
(`:158-168`). There is no UI anywhere in the frontend today that shows a
permission-by-permission breakdown, an override, or an effective-vs-inherited
distinction — the entire INHERIT/GRANT/REVOKE surface (§8 below) is net-new UI,
not an extension of an existing pattern.

## 6. User-management UX requirements

From the eleven routes (§3), the concrete actions Stage 6 must expose:

1. **List** users — already rendered (`administration.tsx:426-437`), needs re-pointing
   to the new list endpoint (§27) so the write actions below can act on the same rows.
2. **Create** a user — email, display name, initial role(s), optional password (else
   server-generated and returned once). The one-time-password contract
   (`user_administration.py:349-354`) is a UX requirement, not an implementation
   detail: the UI must display it once, in a copyable form, with a visible warning
   that it will not be retrievable again — the same "printed once" discipline the
   backend module doc already names for `scripts/manage.py reset-password --generate`.
3. **View** one user — a detail surface, not just a list row, because roles,
   overrides, and effective permissions do not fit in a table cell (§13).
4. **Update** display name — the only PATCHable field (`:366-372`); email and
   credentials are deliberately not editable here.
5. **Activate / Deactivate** — a toggle-shaped action, but deactivation refuses on
   self and needs sensitive-action treatment (§19).
6. **Assign / Remove role** — a picker constrained to the seven closed `Role` values
   `unityworks-vision-ai-backend/app/authorization/model.py:55-61`, filtered client-side
   by what a role would carry versus what the acting admin currently holds (mirrors
   `_require_grantable_role`, `user_administration.py:189-200` — a UX courtesy, not a
   substitute for the server check, matching the whole file's stated posture:
   *"guards are UX, not security"*, `guards.tsx:4-7`).
7. **Set / Reset permission override** — GRANT, REVOKE, or back to INHERIT, per
   permission, per user (§7-§8).

## 7. Permission override UX requirements

The admin must, for a given user and a given permission, see three distinct facts at
once without arithmetic:

- **Role-derived state** — would this permission be held from the user's role(s)
  alone, before any override? (Derivable client-side from `ROLE_PERMISSIONS`-shaped
  data, but the wire format does not currently expose "role-only" as a separate
  field — see §DECISION REQUIRED note below.)
- **Override state** — `inherit` (no row), `grant`, or `revoke`
  (`_permission_rows`, `user_administration.py:230-236`).
- **Effective result** — `decision.has(permission)`, the actual answer the platform
  gives this user right now (`:241`, `:614-619`, `:651-656`).

**Gap CONFIRMED in the wire contract**: `list_permission_overrides`
(`user_administration.py:544-552`) returns `{permission, state, effective}` per row —
it does not separately return "would this be true from roles alone", so a UI cannot
render "role says X, override changes it to Y" without either (a) requesting the same
user's role list and re-deriving `permissions_for(roles)` client-side against a
frontend copy of `ROLE_PERMISSIONS` (fragile — the frontend has no such table today,
by design; permission facts come from `/auth/me`, `permissions.ts:1-13`), or (b) the
backend adding a `role_grants: boolean` field to each row. **PROPOSED**: (b) is the
correct fix — it is one boolean computed from data `_permission_rows` already holds
(`permission in permissions_for(target's roles)`), and it keeps the frontend's stated
discipline of never re-deriving authorization facts client-side. This is a small,
additive Stage 7 backend change, not a Stage 6 architecture problem, and is listed
in §26/§29.

## 8. INHERIT/GRANT/REVOKE UX model

**PROPOSED.** A per-permission row, not a checkbox matrix, grouped by the categories
in §17. Each row shows, left to right: the permission's product-facing name; a
compact three-state control (`Inherited` / `+ Added` / `− Restricted`, matching the
freeze document's own recommended vocabulary, `FINAL_..._FREEZE.md:237`); and the
effective result as an existing `StatusBadge`-shaped grant/no-grant marker.

**Does the four-observation-state grammar extend, or does this need its own?**
PROPOSED: **its own, deliberately not reused.** `STATES`/`Meter` in
`unityworks-vision-ai-frontend/src/shared/semantics/observation.ts:49-90` solve "what
did the camera see" — a fact about the physical world with a `decided` axis (did
anyone even look) crossed with a `countsAsViolation` axis (is this a problem). A
permission override is a fact about a *decision an admin made*, with no "not visible"
or "unknown" case — every permission is always in exactly one of GRANT/REVOKE/INHERIT,
fully known, never partially observed. Reusing `present`/`absent`/`not_visible` colors
for a policy state would borrow meaning that does not apply (an admin reading amber
`not_visible` next to a permission would reasonably ask "not visible to whom?",
a question this domain cannot answer) — exactly the trap `Readiness`'s own doc comment
already names for a different pair of states: *"the two obvious choices are both
wrong... red says the module is broken, and it is not; grey is the UNKNOWN token...
a claim about data rather than about configuration"* (`product.tsx:1060-1064`).

What **does** transfer is the *principle*, not the palette: **form over color**,
stated once and followed twice. `Readiness` tells `awaiting` from `blocked` by
dashed-vs-solid outline and hollow-vs-barred glyph, never by hue
(`product.tsx:1075-1104`). PROPOSED: the override control follows the same rule —
`Inherited` renders as plain text with no border (the default, nothing to announce);
`+ Added` renders as a solid-outline badge with a `+` glyph in the existing `accent`
token; `− Restricted` renders as a solid-outline badge with a `−` glyph in the
existing `severity-medium`/`degraded` token (not `severity-critical` — a REVOKE is a
deliberate admin restriction, not an alarm). Every state carries its word
(`Inherited`/`Added`/`Restricted`) so colour is never the only signal, matching the
rule stated at the top of `primitives.tsx:9-11`. This is a genuinely new component
(§24), built from existing tokens, not a second design system.

**Interaction**: clicking `Inherited` opens a small inline choice (GRANT / REVOKE),
not a modal — the action is reversible (DELETE resets to INHERIT,
`user_administration.py:622-656`) and low-ceremony; clicking an existing `+ Added` or
`− Restricted` badge offers "Reset to inherited" plus a description of what that will
change the effective result to, computed from the same `effective` field the row
already carries, before the click, not after — no invented data, matching `Figure`'s
"never fabricate" rule (`product.tsx:32-35`) applied to a preview rather than a number.

## 9. Organization administration UX

**Out of scope for Stage 6.** CONFIRMED, freeze document §26: *"Add Organization
(blocked on Organization CRUD + user provisioning, both MISSING)"*. No
`/organizations` route exists in the backend today (freeze §11: *"Organization | none
dedicated — no Organization CRUD exists"*). PROPOSED: no organization-lifecycle UI
(create/suspend/archive an org) is designed in this report, because there is nothing
to point it at. This is the single clearest instance of the guardrail against
inventing UI ahead of an API.

## 10. Organization→Site→Zone→Camera UX flow

CONFIRMED unchanged by Stage 5: Sites and Zones already have full create+list UI in
`administration.tsx` (§2), Cameras are managed on their own persisted routes
(`CamerasPage`/`CameraDetailPage`, `persistence-routes.tsx:1416+`), not inside
`/admin`. PROPOSED: this flow is unaffected by Stage 6 and needs no new work — Users
is the only administrable resource without a write UI today, which is exactly the gap
Stage 5's backend fills and Stage 6 must close on the frontend.

## 11. Multi-organization navigation strategy

**DECISION REQUIRED, but not urgently — PROPOSED default: do nothing now.** CONFIRMED
facts driving this: exactly one organization exists in this deployment today (freeze
§13, INFERRED from no Organization CRUD existing to create a second one), and no
platform-operator role/permission exists (`Role` enum, `model.py:46-61`, has no
`platform_operator` member; `super_admin` is explicitly tenant-scoped by construction,
freeze §13). Building an organization switcher now would render a control with
exactly one always-selected option forever, which is worse than no control — it
teaches the operator that a choice exists where none does. PROPOSED: revisit only when
Organization CRUD ships (a future stage per freeze §29.7, explicitly independent of
Stage 5/6's work).

## 12. Platform operator versus tenant administration boundary

CONFIRMED, carried unchanged from freeze §13: `AccessDecision.tenant_id` is a single
mandatory field (`model.py:490-510`); every route Stage 5 built, including this
report's entire subject, operates inside that one tenant, narrowed via
`access.tenant_id`, never a caller-supplied id (`user_administration.py:17-20`, module
docstring). PROPOSED: the Stage 6 UI needs **no** Level A/Level B visual separation
today, because there is no Level A screen to separate from — `/admin` already *is*
Level B (org-scoped operations), and it stays that way. The distinction the user's own
framing asks about becomes real only once a platform-operator concept and an
`/organizations` API exist; until then a second visual register would be decorating an
absence.

## 13. Screen inventory

PROPOSED, minimum coherent set:

1. **Administration — Accounts region (extended, not new)**: the existing
   `administration.tsx` Accounts `Region` (`:412-451`) gains a "Add account" form
   (mirrors the existing Sites/Zones add-forms visually, `:250-287,317-408`) and each
   table row gains a link to (2).
2. **User detail** (`/admin/users/:userId`, new route): identity fields (email,
   display name, created/last sign-in), role list with assign/remove controls,
   activate/deactivate control, and the permission-override table (§8). This is where
   the eleven routes actually live in the UI — one screen, not eleven.

No third screen is proposed. A separate "Roles" or "Permissions" top-level page was
considered and rejected: `Role` is a closed, seven-member, code-defined enum
(`model.py:46-61`) with no admin-editable role *definitions* — there is nothing to
list or edit about a role in the abstract, only which roles a given user holds, which
belongs on that user's own screen.

## 14. Proposed routes

| Route | Purpose | Permission gate | Nav path | Back-navigation | Why a route, not a panel |
|---|---|---|---|---|---|
| `/admin` | Unchanged — Sites, Zones, Accounts | `manageUsers` OR `manageOrganization` (existing, `AppRouter.tsx:199-204`) | Platform → Administration (existing) | n/a, top-level | Unchanged |
| `/admin/users/:userId` | User detail: identity, roles, overrides | `manageUsers` (PROPOSED — narrower than `/admin`'s `any`-of-two, because writing to a user always needs `MANAGE_USERS` specifically, `user_administration.py:100-104`; a `manageOrganization`-only holder who lacks `manageUsers` should see `/admin`'s Sites/Zones but get redirected off a user's detail page, matching `RequirePermission`'s existing redirect-not-403 posture, `guards.tsx:56-59`) | Reached only via a link from the Accounts table row (no direct nav entry — same pattern as `/cameras/:cameraKey`, which has no nav item either) | `GoTo`-styled link back to `/admin#accounts` (or plain `/admin`) | A user's role list + eleven permission rows does not fit in a table row or a `Modal` without either scrolling a nested region inside a nested overlay (the exact defect class `EngineeringSurface`'s own doc comment names for a 6,776px page with "no way to move inside it", `product.tsx:1598-1602`) or truncating the override table — and it needs its own bookmarkable, linkable address for the same reason `AppRouter.tsx:161-167` gives incidents one: "no view has a URL, so no view can be linked... from an alert" applies just as much to "review this one user's access" from an audit finding. |

No other new routes. `PermissionGate`-hidden inline forms (create-user) stay on
`/admin` itself, matching the existing Sites/Zones pattern exactly — a form is not a
screen.

## 15. Proposed navigation structure

PROPOSED: **no navigation change.** The existing single `Platform → Administration`
entry (`navigation.ts:270-276`) continues to point at `/admin`; its hint text
(*"Restaurants, zones, users and roles"*, `:275`) already promises exactly what Stage 6
delivers and needs no edit. `/admin/users/:userId` gets no nav entry, matching every
other `:id` detail route in the app (`navigation.ts` has no entry for
`/cameras/:cameraKey`, `/evidence/:evidenceRef`, or `/incidents/:incidentId` either —
CONFIRMED by their absence from the grep in §4).

## 16. User journeys

1. **Create an account.** Admin opens `/admin` → Accounts → "Add account" → fills
   email, display name, picks role(s) from a list filtered to what they themselves
   hold → submits → sees the one-time password (or the account with none, if they set
   one explicitly) → row appears in the table.
2. **Investigate one person's access** (e.g. from an audit finding referencing an
   email). Admin opens `/admin` → Accounts → clicks the row → lands on
   `/admin/users/:userId` → sees roles, then the permission table with
   Inherited/Added/Restricted per row and the effective result beside each.
3. **Restrict one permission** (Case C from the freeze document, §7-§10 there). Admin
   is already on a user's detail page → finds the permission row (already showing
   `Inherited`, effective `true`, because a role grants it) → sets REVOKE → sees the
   row flip to `− Restricted`, effective `false`, immediately (optimistic or
   post-response render, implementation detail for Stage 7).
4. **Grant a one-off extra**. Same page, a permission currently `Inherited`/effective
   `false` → admin sets GRANT (only enabled if the admin holds it themselves, per
   `_require_grantable_role`/`access.has` parity, §3) → row flips to `+ Added`.
5. **Remove a role**. On the same page, admin removes `restaurant_manager` from a
   user who also holds `kitchen_supervisor` → role list updates → every permission row
   that was `Inherited` purely from the removed role, and not also granted by the
   remaining role or an override, flips its effective column to `false` — visible
   without navigating away, because the override table's `effective` field is
   authoritative and reloaded after the mutation (§3, `:614-619` pattern for GRANT/
   REVOKE; the roles route itself returns the updated user, `:497,538`, so a follow-up
   `GET .../permissions` refetch is the correct Stage 7 implementation, not a
   client-side recompute).
6. **Deactivate an account.** On the user detail page, admin clicks Deactivate →
   confirmation (§19) → account flips to disabled; if the admin is viewing their own
   account, the control is absent or disabled with the server's own reason surfaced
   (`ScopeError`, `:439`), matching the whole app's convention of showing the server's
   sentence rather than inventing a UI-side rule (`administration.tsx:438-447`'s own
   pattern, generalized).

## 17. Backend API→frontend action mapping

| Route | Method | UI action | Screen |
|---|---|---|---|
| `/api/v1/admin/users` | GET | List accounts | `/admin` Accounts region |
| `/api/v1/admin/users/{id}` | GET | Load user detail | `/admin/users/:userId` |
| `/api/v1/admin/users` | POST | "Add account" form submit | `/admin` Accounts region |
| `/api/v1/admin/users/{id}` | PATCH | Edit display name | `/admin/users/:userId` |
| `/api/v1/admin/users/{id}/activate` | POST | Activate control | `/admin/users/:userId` (and a row action on `/admin` if space allows) |
| `/api/v1/admin/users/{id}/deactivate` | POST | Deactivate control (confirmed) | `/admin/users/:userId` |
| `/api/v1/admin/users/{id}/roles` | POST | Assign-role picker | `/admin/users/:userId` |
| `/api/v1/admin/users/{id}/roles/{role}` | DELETE | Remove-role action (confirmed) | `/admin/users/:userId` |
| `/api/v1/admin/users/{id}/permissions` | GET | Render override table | `/admin/users/:userId` |
| `/api/v1/admin/users/{id}/permissions/{perm}` | PUT | Set GRANT/REVOKE | `/admin/users/:userId` override row |
| `/api/v1/admin/users/{id}/permissions/{perm}` | DELETE | Reset to inherited | `/admin/users/:userId` override row |

## 18. Permission→UI action mapping

| Permission (`Permission` enum member) | UI surface it gates | Category (§ below) |
|---|---|---|
| `MANAGE_USERS` (`model.py:83`) | Every write action in §17 | Identity/Admin |
| `VIEW_USERS` (`model.py:84`) | Reading the Accounts table (existing, unchanged, `GET /users`) | Identity/Admin |
| `MANAGE_ORGANIZATION` (`model.py:82`) | Sites/Zones create forms (existing, unchanged) | Identity/Admin |
| `VIEW_CAMERAS`/`MANAGE_CAMERAS` | Cameras pages (unaffected by Stage 6) | Operations |
| `VIEW_INCIDENTS`/`ACKNOWLEDGE_INCIDENTS`/`RESOLVE_INCIDENTS` | Incidents (unaffected) | Compliance |
| `VIEW_EVIDENCE`/`DELETE_EVIDENCE` | Evidence (unaffected) | Compliance |
| `VIEW_AUDIT` | Audit (unaffected) | Compliance |
| `VIEW_REPORTS`/`EXPORT_REPORTS` | Reports (unaffected) | Analytics |
| `ACCESS_DEVTOOLS`/`REGISTER_DEMAND`/`VIEW_MODEL_EVALUATION` | DevTools/Model Evaluation (unaffected) | Engineering |

Groupings derived directly from the enum's own section comments
(`model.py:81,86,93,97,111,136,183`), not invented: *"identity and administration"*,
*"observation surfaces"*, *"sites and cameras"*, *"incidents"*, *"reporting"*,
*"product modules with no data source yet"*, *"engineering"*. §7 of the freeze
document's own table (§11 there) groups the same way. This confirms the user's
suggested categories (Operations, Compliance, Analytics, Engineering) map onto real
code, with one correction: the freeze document's own table already separates
Sites/Zones (gated oddly on `VIEW_USERS`/`MANAGE_ORGANIZATION`, CONFIRMED §6 there,
re-confirmed `model.py:216-217,84,82`) from Cameras (`VIEW_CAMERAS`/`MANAGE_CAMERAS`)
— both are "Operations" in product terms but are two different permission pairs, which
the override table (§8) must show as two separate rows, never merged.

## 19. Sensitive privilege-change UX requirements

PROPOSED, using the existing `Modal` primitive
(`unityworks-vision-ai-frontend/src/shared/ui/primitives.tsx:852-864`, `open`/`onClose`/
`title`/`children`/`footer`), not a new component:

- **Deactivate**: confirm via `Modal`, stating the account will be signed out and
  refused login immediately (`decide()` already treats an inactive user as holding no
  roles, backend docstring `:426-430`) — the UI must not imply this is reversible-free;
  it is reversible (Activate undoes it) but the modal should say so, since the backend
  doc explicitly worries about an admin locking out their organisation's only
  `MANAGE_USERS` holder (`:432-435`) — the frontend cannot detect that case (it doesn't
  know if the target is the *only* holder), so the copy should be a general caution,
  not a specific claim the UI can't verify.
- **Remove role**: confirm via `Modal` only when the role being removed carries
  `MANAGE_USERS` or `MANAGE_ORGANIZATION` (i.e., removing it could itself remove the
  target's own admin capability) — PROPOSED as a judgment call to avoid confirmation
  fatigue on routine narrowing (removing `kitchen_supervisor` from someone who also
  holds `restaurant_manager` is low-stakes); removing an admin-carrying role is not.
- **REVOKE a permission**: no modal — PROPOSED, because it is instantly, visibly
  reversible in the same table (reset-to-inherited is one click, §8), unlike
  deactivation or role removal which change what a user can *reach*, not just one
  row's display.
- **GRANT a permission**: no modal, same reasoning, plus it is already
  self-limited to what the actor holds (§3).

This is a **new judgment call**, not dictated by any prior document — flagged
**DECISION REQUIRED** in §19's own recommendation only insofar as the confirm/no-confirm
split above is a UX opinion Stage 7 should get explicit sign-off on before building,
not a technical fork.

## 20. Responsive UX requirements

PROPOSED, consistent with existing patterns: the permission-override table must
reflow at the same breakpoint `DataTable` already handles for every other table in the
product (`administration.tsx`'s Sites/Zones/Accounts tables use the same `DataTable`,
`primitives.tsx:708+`, unmodified) — no new responsive mechanism. The role-assign
picker and add-account form follow the existing Sites/Zones add-form pattern's own
`flexWrap: 'wrap'` behavior (`administration.tsx:332-339`), already proven at narrow
widths. `PageIntro`'s own `uwv-lead` breakpoint handling (`product.tsx:177-178`)
requires no change since `/admin/users/:userId` reuses `PageIntro` unmodified.

## 21. Accessibility requirements

PROPOSED, extending existing conventions rather than inventing new ones: the
override control's state must be conveyed by the visible word
(Inherited/Added/Restricted, §8), not only the glyph or border style, matching the
rule stated in `primitives.tsx:9-11`. The `Modal` primitive already manages focus on
open (`primitives.tsx:869-871`) and dismiss (`useDismiss`) — reused as-is for §19's
confirmations, no new focus-trap code needed. The role picker and permission-state
control should be `<select>`/button-based rather than drag or hover-only interactions,
consistent with `Select`'s existing native-element-with-custom-chrome approach
(`primitives.tsx:194-238`, kept "still keyboard operable" per its own comment
`:212-214`).

## 22. Design-system reuse strategy

**Reused as-is (no modification needed):** `PageIntro`, `SectionRule`/`Region`,
`Plane`, `DataTable`, `Input`, `Select`, `Button`, `Badge`, `StatusBadge`, `EmptyState`,
`ErrorState`, `LoadingState`, `Modal`, `GoTo`, `Figure`, `PermissionGate`,
`RequirePermission`.

**Genuinely new:** a permission-override row control (§8, §24); a role-assignment
picker (constrained-select + grantability filter, §6.6); an effective-access summary
strip on the user detail page (a `Plane` of `Figure`s — role count, added count,
restricted count — composed from existing primitives, not a new component in itself).

**Must NOT be duplicated** (§25): a second table component for the override list —
`DataTable` already handles sortable, empty-stated, captioned tabular data
(`administration.tsx`'s three existing tables all prove it handles this shape); a
second page-opening pattern for `/admin/users/:userId` — `PageIntro` already composes
eyebrow/title/standfirst/meta/actions and every rebuilt page in the product uses it
(`persistence-routes.tsx:1384-1394` for `EvidenceDetailPage`, cited as the closest
existing analog: an object detail page reached only via a link from its list, with a
`GoTo`-styled back link in `actions`).

## 23. Components that can be reused

Listed with citations: `PageIntro` (`product.tsx:107-212`), `SectionRule`/`Region`
(`:279-344`,`:223-237`), `Plane` (`:251-277`), `DataTable` (`primitives.tsx:708+`),
`Input`/`Select`/`Button` (`:159-238`,`:65-105`), `Badge`/`StatusBadge`
(`:244+`,`:361+`), `EmptyState`/`ErrorState`/`LoadingState` (used throughout
`administration.tsx`), `Modal` (`:852-864`), `GoTo`/`GoToButton`
(`product.tsx:1773-1799`), `Figure` (`:524-604`), `PermissionGate`/`RequirePermission`
(`guards.tsx:44-85`).

## 24. Components that should be created

1. **`PermissionOverrideRow`** (name PROPOSED) — one row of the override table:
   permission label, category (§18), the Inherited/Added/Restricted control (§8), the
   effective badge. Built from `Badge`-shaped primitives and existing tokens; no new
   colors.
2. **`RolePicker`** (name PROPOSED) — a `Select`-based control listing the seven
   `Role` values with `roleLabel()` (already exists,
   `unityworks-vision-ai-frontend/src/app/permissions/permissions.ts:112-114`) for
   display, disabling any role the acting admin cannot grant (client-side courtesy
   mirroring `_require_grantable_role`, §6.6) — composed from `Select`, not a new
   input primitive.
3. **`AccessSummary`** (name PROPOSED, optional) — a `Plane` of `Figure`s at the top of
   `/admin/users/:userId` (role count / added count / restricted count) — composable
   entirely from existing `Figure`, no new primitive strictly required, listed here
   only because it does not exist as an assembled unit today.

None of these are a second design system; all three compose `product.tsx`/
`primitives.tsx` exports.

## 25. Components that must NOT be duplicated

Explicit, per the task's framing: **do not build a second table primitive** —
`DataTable` already renders the Sites, Zones, and Accounts tables in this exact file
and handles column width, numeric alignment, empty state, and caption; a bespoke
override-table grid would be the same defect class as the Evidence-page and
Cameras-page alignment bugs referenced in `PageIntro`'s own doc comment
(`product.tsx:130-138`, "Cameras' standfirst is five lines... opening a gap above the
eyebrow that had nothing to do with spacing") — drift between two implementations of
"a labeled table" is exactly how that class of bug is introduced. **Do not build a
second page-opening pattern** for `/admin/users/:userId` — `PageIntro` is mandatory.
**Do not build a second badge/status vocabulary** for GRANT/REVOKE/INHERIT — extend
`Badge`'s existing tone system or compose a thin wrapper around it (§24.1), never a
parallel `<span>`-with-inline-styles implementation, which is how `Readiness` and
`StateTally` ended up needing their own doc-comment justification for *not* reusing
an existing pattern (`product.tsx:1056-1073,974-990`) — a new component here needs the
same explicit justification this report gives in §8, not a silent copy-paste.

## 26. Risks

1. **Wire-contract gap** (§7): the override list does not expose "role-derived"
   separately from "effective" — without a `role_grants` field or equivalent, the
   INHERIT/GRANT/REVOKE UI cannot show the three-fact view the task requires without
   a fragile client-side re-derivation. PROPOSED as a small additive Stage 7 backend
   change, listed again in §29.
2. **Stale-copy risk**: `write_available`/`write_unavailable_reason`
   (`administration.tsx:438-447`) reads from the *old* `GET /users` route, which
   Stage 5 confirms is untouched (§3) and will keep returning `write_available: false`
   forever unless the frontend stops reading it once it switches to the new list
   endpoint. If Stage 7 keeps calling the old route for the list (e.g. out of inertia)
   while adding writes against the new one, the page will show working create/edit
   controls directly above a sentence claiming they don't exist — a direct
   self-contradiction. Must switch list-reads to `GET /api/v1/admin/users` as part of
   the same change, not phase them.
3. **REVOKE-of-role-the-actor-can't-reach** (§3, backend docstring `:64-78`): the UI
   has no way to warn an admin they are about to remove/revoke something outside their
   own reach, because the backend does not flag this case either. Low risk given it
   requires an `org_admin` acting on a `super_admin`/`developer` target, both rare in
   a single-org deployment, but should be named to Stage 7, not silently inherited.
4. **Confirmation-fatigue vs. under-warning** (§19): the confirm/no-confirm split
   proposed there is a judgment call; getting it wrong in either direction is a real
   but low-severity UX risk, not a security one (server-side checks are unaffected
   either way).

## 27. Backward compatibility

The existing `/admin` route, its Sites/Zones behavior, and its permission gate
(`AppRouter.tsx:199-207`) are unaffected — Stage 6 only replaces the Accounts region's
read source and adds write affordances plus one new route. `GET /users` (the old
route) can remain in place for any other caller (none currently known) since Stage 5
left it untouched by design (§3) — no backend deprecation is proposed or required by
this report. The `OrgUser`/`UserList` TypeScript types
(`unityworks-vision-ai-frontend/src/shared/api/observations.ts:165-186`) will need a
parallel or replacing type for the new wire shape (§3's `_user_to_wire`, which is a
superset: same fields, no `write_available` wrapper) — an additive type change, not a
breaking one, since nothing else in the frontend imports `OrgUser`/`UserList` outside
`administration.tsx` (not verified exhaustively this pass; a Stage 7 grep should
confirm before deleting the old types).

## 28. Explicitly out of scope

Per the task's guardrails and §9/§11 above: Organization CRUD/lifecycle UI (create,
suspend, archive an org); a platform-operator role or its UI; an organization
switcher; custom/editable roles (the `Role` enum is closed by design, §13); any
change to CCTV/camera/site/zone screens beyond what already exists; any change to
perception, ML, or Vision OS surfaces; implementing any of the components in §24 (this
report proposes their shape, not their code).

## 29. Proposed Stage 7 implementation plan

1. Point `AdministrationPage`'s Accounts region at `GET /api/v1/admin/users`
   (replacing `organizationApi.users`), and drop the `write_available` messaging
   (§26.2) — do this as the first change, so the stale copy cannot coexist with new
   write controls even transiently.
2. Backend: add a `role_grants: boolean` field to each `_permission_rows` entry
   (§7, §26.1) — the smallest change that unblocks a truthful three-fact override UI;
   a few-line addition to `user_administration.py:217-244`, no schema change.
3. Build `PermissionOverrideRow`, `RolePicker` (§24), and the `/admin/users/:userId`
   route (§14), wired to the eleven routes per §17.
4. Add the "Add account" form to `/admin`'s Accounts region, mirroring the existing
   Sites/Zones add-form layout exactly (§13.1).
5. Add row-level links from the Accounts table to `/admin/users/:userId`.
6. Wire the confirmation pattern (§19) using the existing `Modal`.
7. Update `OrgUser`/`UserList` types or add parallel ones for the new wire shape
   (§27).

## 30. Final route/navigation/permission matrix

| Route | Nav entry | Permission gate | New/existing |
|---|---|---|---|
| `/admin` | Platform → Administration | `manageUsers` OR `manageOrganization` | Existing, content extended |
| `/admin/users/:userId` | None (row-link only, matches `/cameras/:cameraKey` precedent) | `manageUsers` | New |

No other route or nav change is proposed by this report.

---

**Files read to produce this document** (backend):
`app/api/user_administration.py` (full), `app/authorization/model.py` (full),
`docs/architecture/FINAL_ADMINISTRATION_MULTI_ORGANIZATION_IDENTITY_PERMISSION_CCTV_ARCHITECTURE_FREEZE.md`
(full), `docs/architecture/USER_MANAGEMENT_ROLE_ASSIGNMENT_PERMISSION_OVERRIDE_IMPLEMENTATION_STAGE_5.md`
(§1-7). **Files read** (frontend): `src/shared/ui/product.tsx` (full),
`src/features/administration.tsx` (full), `src/app/router/AppRouter.tsx` (full),
`src/app/permissions/permissions.ts` (full), `src/app/permissions/guards.tsx` (full),
`src/shared/semantics/observation.ts` (full), `src/app/router/navigation.ts` (grepped,
relevant sections), `src/shared/ui/primitives.tsx` (partial — Button/Input/Select/
Badge signatures, `Modal` signature), `src/shared/api/observations.ts` (grepped —
`OrgUser`/`UserList`/`organizationApi`), `src/features/persistence-routes.tsx`
(grepped — `CameraDetailPage`/`EvidenceDetailPage` route bodies). No file in either
repository was modified. `git status` confirmed clean in
`unityworks-vision-ai-frontend`; `unityworks-vision-ai-backend` shows only
pre-existing Stage 1-5 uncommitted work, untouched by this pass.
