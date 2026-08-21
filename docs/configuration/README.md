# Configuration

## Precedence

```
1. defaults in code        safe, minimal, never security-relevant
2. .env file               local development
3. environment variables   deployment specifics
4. SecretProviderPort      secrets, resolved by reference      (Phase 2)
5. database                per-organization settings           (Phase 4)
```

Later layers win, **except** that layer 4 is not an override of layer 3. It is a
different kind of value: a secret enters the earlier layers as a *reference*
(`CCTV_CREDENTIAL_REF`) and is resolved at layer 4, so the value itself never
appears in a config file, an environment dump, a log line or a `repr()`.

Layers 4 and 5 are not implemented. Their position is fixed now so that adding
them later is an addition rather than a re-ordering.

## Two configuration systems, deliberately

| system | owns | read by |
|---|---|---|
| `app.configuration.settings.Settings` | application concerns | the application |
| Vision OS `ConfigSourcePort` + provider modules | platform concerns | the platform, at its composition roots |

The platform's adapters read their own environment at their composition
root — `VISION_DETECTOR_PROVIDER`, `VISION_UNDERSTANDER_PROVIDER` and the rest.
That is the platform's design, and duplicating those reads into `Settings` would
create two sources of truth for one value.

What `Settings` holds is the subset the *application* decides about: which policy
documents to hand the platform, and whether imagery may leave the process.

## Production hardening

`Settings.assert_production_safe()` runs in `create_app()`, before anything opens
a socket. In `APP_ENV=production` it refuses to start when:

- `SECRET_KEY` is unset or still the development default
- `DB_PASSWORD` is unset or still `postgres`
- `APP_DEBUG` is true
- `JWT_ALGORITHM=RS256` without both key paths

The error names the variable and **never its value**, so it is actionable in a
log aggregator without becoming a disclosure.

## Settings that change security posture

| setting | default | meaning |
|---|---|---|
| `SERVE_FRAMES` | `false` | decoded frames may leave the process |
| `ALLOW_EVIDENCE` | `false` | crop imagery may be retrieved |
| `FEATURE_DEVTOOLS` | `false` | `/api/v1/devtools/*` is mounted at all |
| `FEATURE_LIVE_CCTV` | `false` | live ingestion (Phase 3) |
| `APP_DEBUG` | `false` | also gates `/docs`, `/redoc`, `/openapi.json` |
| `CORS_ORIGINS` | localhost only | `*` is refused at construction |
| `DB_ECHO` | `false` | SQL logging; query logs contain personal data |

All four flags are **deployment decisions**, not user settings. The reference
validation console's launcher turned the first two on because a local
engineering tool that hides the pictures is useless — that reasoning is sound for
a laptop and is a privacy breach on a network, so this backend has no equivalent
of that launcher.

Startup logs a warning whenever either imagery flag is on.

## Vision OS policy documents

```
VISION_SEMANTIC_POLICY=./config/policies/kitchen-safety.example.json,./config/policies/object-identity.example.json
VISION_VERIFICATION_RULES=./config/policies/verification.example.json
COMPLIANCE_RULES=./config/rules/site-safety.example.json
```

Comma-separated, because a policy carries one subject-class scope and the two
active use cases concern different subjects.

**An empty setting is valid**: the platform declares no attributes, demands
nothing, spends no model calls, and says so through its capability summary.
Detection, tracking, the registry and the Observation API all still run.

**A named-but-missing document raises at startup.** Skipping it silently would
leave a deployment believing a policy is in force.

Adding a use case is a JSON file, not a release. Not one attribute name, value or
end-user sentence appears in Python anywhere in this repository.

## Per-restaurant configuration (Phase 4)

Some settings cannot stay process-wide once there is more than one customer:
semantic policy, compliance rules, camera list, evidence retention, notification
routing. These move into the database as document references and versions. The
document formats do not change; only where the pointer lives.

Environment variables remain the *deployment* layer — which database, which VLM,
which weights.
