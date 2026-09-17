---
name: api-route-change
description: Use when adding, changing or removing an HTTP or WebSocket route, request/response model, route permission or router in app/api/ of the Vision AI backend — including new endpoints for the frontend, admin, platform-operator or report surfaces.
---

# API Route Change

## Overview

A route change is one ritual with six parts. Skipping any one produces a failure no single test
catches: an unguarded route, a frontend typed against an API that is gone, or an existence leak
across tenants.

```
router → permission → narrowed query → audit (if sensitive) → test (allowed + refused + cross-tenant)
       → export_openapi → frontend types:generate
```

## 1. Choose the principal — this decides everything else

| Surface | Principal | Dependency |
|---|---|---|
| Anything inside one organization | `AccessDecision` | `dependencies=[Depends(requires(Permission.X))]` + `access: CurrentAccess` |
| Cross-organization control plane | `PlatformOperator` | `current_operator` — only in `app/api/platform.py` / `platform_administration.py` |

Never widen `super_admin` to reach across tenants: it is a tenant role, and every customer holds one.

## 2. The route

- Put it in the router that owns the area (`product.py`, `administration.py`, `reports.py`,
  `user_administration.py`, …). A new router needs `app.include_router(...)` in `app/main.py`
  `create_app`, with a comment saying what gates it.
- Name the **exact** permission. Permissions never imply one another (`VIEW_EVIDENCE` ⊄
  `VIEW_OBSERVATIONS`). A new permission is added to `app/authorization/model.py::Permission`, granted
  to roles deliberately, and mirrored in the frontend's `src/app/permissions/permissions.ts`.
- `REGISTER_DEMAND` stays unwired, and there is no "run evaluation" endpoint — both would let a user
  spend the model budget.

## 3. The query

- **Construct it narrowed**, never filter after loading: `tenant_id` from `access`, camera scope
  from `access.cameras` (`ALL_IN_TENANT` → `None` = tenant-wide; `LISTED` → those ids; an **empty
  tuple matches nothing**).
- A resource in another tenant returns **404, not 403** — existence is itself a disclosure.
- Runtime identity is `organization_id:camera_key` (`app/domain/runtime_identity.py`), never a bare key.
- Observation values go through `app/domain/observations.py` only; `not_visible` survives to the wire.
- Something that cannot be computed returns `available: false` with a reason (see `app/api/capability.py`),
  never an empty list or `0` that reads as a clean result.

## 4. Audit

Evidence, reports, exports, user/permission administration and platform actions write an audit row
via `app/domain/audit.py`. **A refusal is audited with the same weight as a success**, committed
before the error propagates; sensitive responses are `Cache-Control: no-store`. Copy the shape of
`app/api/reports.py`.

## 5. Tests — three cases minimum

In `tests/app/`, using the conftest fixtures (`client`, `seeded`, `bearer`, `make_user`, `admit`):

```python
async def test_<route>_is_refused_without_<permission>(client, seeded):
    headers = await bearer(client, "supervisor@example.com")     # a role lacking the permission
    assert (await client.get("/api/v1/<path>", headers=headers)).status_code == 403

async def test_<route>_does_not_reveal_another_organizations_<thing>(client, seeded):
    headers = await bearer(client, "outsider@example.com")       # org-other
    assert (await client.get("/api/v1/<path>/<id-in-org-test>", headers=headers)).status_code == 404
```

plus the allowed case asserting the real body. Seeded users: `manager@`, `supervisor@`
(`kitchen_supervisor`), `developer@`, `nocameras@` (camera breadth `none`), `outsider@` (org-other).
Fixture roles must hold exactly the role's permissions — over-granting makes the refusal test a
tautology.

## 6. Contract

```bash
.venv/Scripts/python.exe scripts/export_openapi.py
```

Then **REQUIRED:** `contract-sync` — it locates the frontend by its `team.conf` kind (never by folder
name), regenerates its types, runs its verify and handles the CI pin. `/check contract` confirms the
whole seam. Finish with `backend-verify`.

## Common mistakes

- Returning 403 for another tenant's id.
- `camera_ids == ()` treated as "all cameras".
- Auditing the success path only.
- Committing backend without regenerating frontend types (the team.conf `remind` for `app/api/` prints the commands).
