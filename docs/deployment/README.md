# Deployment

## What runs

One process: FastAPI under uvicorn, owning the Vision OS platform in-process.
PostgreSQL and Redis are external.

```
uvicorn ──► app  ──► compliance ──► vision_os
             │
             ├──► PostgreSQL   required
             └──► Redis        optional; degrades
```

## Development

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"
cp .env.example .env          # then set SECRET_KEY and DB_PASSWORD
alembic upgrade head
uvicorn app.main:app --reload
```

## Production

```bash
pip install ".[inference]"
APP_ENV=production alembic upgrade head
APP_ENV=production uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Checklist before the first production boot:

- [ ] `SECRET_KEY` generated per environment, never shared or reused
- [ ] `DB_PASSWORD` set
- [ ] `APP_DEBUG=false` — this also closes `/docs`, `/redoc`, `/openapi.json`
- [ ] `CORS_ORIGINS` set to the real frontend origin
- [ ] `SERVE_FRAMES=false`, `ALLOW_EVIDENCE=false` unless deliberately decided
- [ ] `FEATURE_DEVTOOLS=false` unless deliberately decided
- [ ] TLS terminated in front of the process; `/metrics` on an internal network only
- [ ] `.env` not committed, not in the image, mounted as a secret

The application enforces the first four itself and refuses to start otherwise.

## Startup behaviour

```
settings → production safety check → logging → database engine → Redis → Vision OS
```

**Configuration is fatal. Everything else degrades.**

| dependency | unavailable at boot |
|---|---|
| configuration | **refuses to start** — a misconfigured process cannot be trusted to fail in a bounded way |
| database | boots; reported unready; verified on first use |
| Redis | boots with a warning; features that need it fail at request time, naming it |
| Vision OS | boots with a warning; observation routes return `VISION_UNAVAILABLE`, never an empty result |

The database is not verified at boot on purpose: a database briefly unreachable
while an orchestrator brings the stack up is not a reason to crash-loop.

## Health

| endpoint | audience | auth |
|---|---|---|
| `GET /health` | load balancer | none — `{"status":"ok"}` and nothing else |
| `GET /health/ready` | orchestrator | none — booleans only, 503 when not ready |
| `GET /api/v1/status` | operator | bearer, tenant-scoped |
| `GET /metrics` | Prometheus | none — **internal network only** |

`/health` and `/health/ready` deliberately disclose no versions, component names
or counts. An unauthenticated health endpoint is reconnaissance for anyone who
can reach the port.

## Vision OS is not started in this phase

`VISION_AUTOSTART` defaults to `false`. The platform needs a `SourcePort` to read
from and Phase 1 binds none — the streaming RTSP and file adapters arrive in
Phase 3.

A platform booted with no source would report itself healthy and observe nothing
forever, which is precisely the state invariant V8 exists to make impossible to
misread. So the runtime starts *not started*, says so in `/api/v1/status`, and
observation routes return `VISION_UNAVAILABLE` rather than an empty list.

`VisionRuntime.assemble()` is fully implemented and exercised by tests. What is
missing is the source, not the composition.

## Evidence storage

`EVIDENCE_STORE=memory` is the default and **does not survive a restart**. That
is accidentally private and entirely unusable.

A durable, encrypted adapter with a working retention sweeper is required before
any real deployment. `EvidenceStorePort` already specifies retention, quota,
tombstones and `erase(scope)`; what is missing is an implementation, not a
design. Phase 5.

## Container

```dockerfile
FROM python:3.11-slim
WORKDIR /srv
COPY pyproject.toml README.md ./
COPY vision_os/ vision_os/
COPY compliance/ compliance/
COPY app/ app/
COPY config/ config/
COPY models/ models/
COPY migrations/ migrations/ alembic.ini ./
RUN pip install --no-cache-dir ".[inference]"
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`tests/` and `tools/` are excluded from the image deliberately: neither serves
traffic, and `tools/` pulls in PyAV and Pillow.

Run migrations as a separate step, not in `CMD`. Two replicas starting at once
would otherwise race the same migration.

## Backup

| asset | note |
|---|---|
| PostgreSQL | identity and (from Phase 4) incidents |
| evidence store | CCTV imagery of identifiable people — **encrypt at rest, honour retention** |
| `config/` | policies and rules; versioned in git |
| `models/yolov8n.onnx` | committed; the deployment is reproducible from a checkout |

Evidence backups inherit the retention obligation of the originals. A backup that
outlives its retention window has not been retained — it has been kept.
