# security-engineer — Vision AI backend brief

## Evidence to read
`app/auth/**`, `app/authorization/**`, `app/api/dependencies.py`, every router in `app/api/`,
`app/domain/audit.py`, `app/domain/evidence.py`, `app/domain/retention.py`,
`app/configuration/settings.py`, cookie code in `app/auth/cookies.py`.

## What breaks here
- **Route enforcement:** every route in `app/api/` has `requires(Permission.X)` or `current_operator`,
  or is deliberately public (health, login, refresh). Enumerate by importing the app with
  `.venv/Scripts/python.exe` and walking `app.routes`; diff against dependencies.
- **Tenant isolation:** queries constructed narrowed by `tenant_id`; another tenant's resource returns
  **404 not 403**; runtime identity is `organization_id:camera_key`.
- **Camera scope:** `ScopeBreadth.NONE` never becomes an empty tuple; `camera_keys == ()` matches
  nothing; `AccessDecision.to_grant()` raises rather than guessing.
- **Principal confusion:** nothing translates `AccessDecision` ↔ `PlatformOperator`; `super_admin` is a
  tenant role and must not be widened.
- **Tokens:** refresh rotation and reuse detection; revoked role/membership effective on the next
  request; refresh cookie `SameSite=Strict` + `HttpOnly` + `secure` under `APP_ENV=production`.
- **Override anti-escalation:** no self-modification; a grantor cannot confer reach it lacks — only via
  `app/authorization/overrides.py` and `camera_scope.py`.
- **Audit:** refusals audited with the same weight as successes, committed before the error
  propagates; credentials scrubbed in `_scrub()`; append-only.
- **Evidence and PII:** `SERVE_FRAMES` / `ALLOW_EVIDENCE` default off; `no-store` on sensitive
  responses. Intended, not bugs: `kitchen_supervisor` has no evidence access; `auditor` has no live access.
- **Secrets:** `Camera.credential_ref` holds `env:` references only, never a password.

## How to prove it
Tests use SQLite in memory via `tests/app/conftest.py` fixtures (`client`, `seeded`, `bearer`,
`make_user`, `admit`); seeded users `manager@`, `supervisor@`, `developer@`, `nocameras@`, `outsider@`
(other org). Never point anything at a real database or camera.
