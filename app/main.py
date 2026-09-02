"""The FastAPI application.

### Startup ordering, and what may fail

    settings → production safety check → logging → database → cache → Vision OS

The production safety check runs **before** anything opens a socket, so a
deployment with a default secret fails at boot rather than serving traffic it
should not.

After that the rule is: **degrade, do not refuse**. Redis down warns; Vision OS
failing to assemble warns; both are reported through `/health/ready` and
`/api/v1/status`. Only configuration is fatal, because a misconfigured process
cannot be trusted to be wrong in a bounded way.

The database is the one dependency this application does not verify at boot. It
is verified at readiness and on first use, because a database that is briefly
unreachable while an orchestrator brings the stack up is not a reason to
crash-loop.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from loguru import logger

from app.api.administration import router as administration_router
from app.api.analytics import router as analytics_router
from app.api.evaluation import router as evaluation_router
from app.api.integrations import router as integrations_router
from app.api.patron import router as patron_router
from app.api.product import router as product_router
from app.api.reports import router as reports_router
from app.api.routes import build_router, devtools_router
from app.api.wall import router as wall_router
from app.api.websocket import router as websocket_router
from app.auth.service import AuthService
from app.auth.tokens import TokenService
from app.configuration.settings import Settings, get_settings
from app.errors import AppError
from app.infrastructure.cache import Cache
from app.infrastructure.database import Database
from app.infrastructure.observability import (
    VISION_READY,
    configure_logging,
    metrics_payload,
    request_context_middleware,
)
from app.vision.analysis_loop import ANALYSIS
from app.vision.demands import register_policy_demands
from app.vision.ingest import FrameIngest
from app.vision.manager import LiveRuntime
from app.vision.runtime import VisionRuntime
from app.vision.taps import TapBus
from app.vision.wall import CameraWall


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory.

    Takes settings so tests can construct a differently-configured app without
    mutating process environment.
    """
    cfg = settings or get_settings()
    cfg.assert_production_safe()
    configure_logging(cfg)

    app = FastAPI(
        title=cfg.app_name,
        version="1.0.0",
        description=(
            "Production backend for UnityWorks Vision AI. Owns the Vision OS "
            "perception platform and exposes it through an authorized API."
        ),
        # Schema documentation is a map of the attack surface. Debug builds only.
        docs_url="/docs" if cfg.app_debug else None,
        redoc_url="/redoc" if cfg.app_debug else None,
        openapi_url="/openapi.json" if cfg.app_debug else None,
        lifespan=_lifespan,
    )

    app.state.settings = cfg
    app.state.database = Database(cfg)
    app.state.cache = Cache(cfg)
    app.state.auth = AuthService(TokenService(cfg))
    app.state.vision = VisionRuntime(cfg)
    app.state.live = LiveRuntime(cfg)
    # Live viewing. Independent of Vision OS by design: the wall owns its own
    # sessions and never calls the perception path.
    app.state.wall = CameraWall(cfg)
    # Bounded rings over the platform's own bus. Attached at start-up when a
    # platform exists; harmless and empty when one does not.
    app.state.taps = TapBus()
    app.state.ingest = None
    app.state.demands = None
    app.state.notifier = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins_list,
        allow_credentials=True,
        # Explicit, not "*". A credentialed API that reflects arbitrary methods
        # and headers has given up the preflight as a control.
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-Id"],
    )
    app.middleware("http")(request_context_middleware)

    _install_error_handlers(app)

    app.include_router(build_router())
    app.include_router(product_router)
    app.include_router(administration_router)
    # The seven modules that have a schema and a permission but no data source.
    # Registered unconditionally: a route that answers "not configured, and here
    # is what is missing" is more useful than one that 404s, and hiding it
    # behind a flag would mean the honest answer needs configuring too.
    app.include_router(analytics_router)
    app.include_router(integrations_router)
    app.include_router(patron_router)
    # Reporting reads what every other surface writes, so it is registered
    # last — and every route in it is gated on its sources' own permissions.
    app.include_router(reports_router)
    # Model evaluation reads committed artifacts from disk. Registered like
    # any other product surface: its own permission, no write path.
    app.include_router(evaluation_router)
    app.include_router(wall_router)
    app.include_router(websocket_router)
    if cfg.feature_devtools:
        # The flag decides whether the routes exist at all; ACCESS_DEVTOOLS on
        # each route decides who may reach them. Two gates, independently set.
        app.include_router(devtools_router)
        logger.warning("DevTools routes are mounted (FEATURE_DEVTOOLS=true)")

    if cfg.metrics_enabled:

        @app.get("/metrics", include_in_schema=False)
        async def metrics() -> Response:
            body, content_type = metrics_payload()
            return Response(content=body, media_type=content_type)

    return app


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg: Settings = app.state.settings
    database: Database = app.state.database
    cache: Cache = app.state.cache
    vision: VisionRuntime = app.state.vision

    logger.info("Starting {} [{}]", cfg.app_name, cfg.app_env)

    database.connect()
    logger.info("Database engine ready")

    if not await cache.connect():
        # Degraded, not fatal. Refusing to boot without Redis takes down every
        # route, including the ones that never touch it.
        logger.warning(
            "Redis unavailable — features that need it will fail at request "
            "time with the dependency named; other routes are unaffected"
        )

    # Started before any session can submit to it. CPU-bound perception runs
    # there instead of on this loop; see `app/vision/analysis_loop.py`.
    ANALYSIS.start()

    assembled = await vision.start()
    VISION_READY.set(1 if assembled else 0)

    # The seam. Frames reach the platform only because this line runs, and it
    # runs before any camera starts so every session goes through it.
    if assembled and vision.composition is not None:
        live_runtime: LiveRuntime = app.state.live
        ingest = FrameIngest(vision.composition, ledger=live_runtime.ledger)
        live_runtime.bind_ingest(ingest)
        app.state.ingest = ingest
        app.state.taps.attach(getattr(vision.composition.platform, "bus", None))

        # Without this the platform correctly analyses nothing: M8 skips every
        # candidate with `no_demand`, and zero model calls is the right answer
        # to a question nobody asked. This is the application stating what it is
        # willing to pay to look at.
        app.state.demands = register_policy_demands(
            vision.composition,
            freshness_ms=cfg.vision_demand_freshness_ms,
        )
        logger.info("Vision OS ingest bound — frames will reach the perception path")

        # The application's own decision layer. Vision OS reports what it sees
        # and knows nothing about rules, incidents or users; this reads the
        # Observation API like any other consumer and writes to the Incident
        # domain. Nothing in `vision_os` imports it or is aware it runs.
        app.state.compliance = _build_compliance_driver(app, cfg, vision)
    else:
        # Said plainly. A deployment whose cameras run but whose frames reach no
        # model is a deployment that looks healthy and observes nothing, and
        # that must never be discovered from a dashboard reading zero.
        app.state.ingest = None
        app.state.demands = None
        app.state.compliance = None
        logger.warning(
            "Vision OS is not assembled — camera frames will be acquired and "
            "released without reaching detection"
        )

    # Retention first, before anything is served. A record past its retention
    # date must not be servable for the window between boot and the first sweep.
    # Marking always happens; erasure only when the deployment enabled it.
    await _sweep_retention(app)

    # The ONLY place a camera session starts. Not on import, not on a DevTools
    # request, not as a side effect of anything else.
    #
    # The camera set now comes from the database, so restarting the process
    # restores exactly the cameras that were running before it stopped — which
    # is what "recovery" means here. A row that is not `enabled` opens no socket.
    # The camera wall is started from the same durable camera rows, and started
    # whether or not Vision OS assembled — viewing must not depend on analysis.
    #
    # Both halves run through one bootstrap so that a database which is not yet
    # accepting connections is a *delay* rather than a permanent failure. If the
    # roster cannot be read, a supervisor retries in the background until it can;
    # see `_camera_bootstrap_supervisor` for why the previous one-shot bind left
    # cameras dark behind a healthy-looking dashboard.
    live: LiveRuntime = app.state.live
    app.state.camera_bootstrap = None
    wall_done, live_done = await _bootstrap_cameras_once(app)
    if not (wall_done and live_done):
        logger.warning(
            "camera roster unreadable at start-up (wall_read={}, live_read={}) "
            "— retrying in the background; the process stays up and every route "
            "that does not need the database keeps serving",
            wall_done,
            live_done,
        )
        app.state.camera_bootstrap = asyncio.create_task(
            _camera_bootstrap_supervisor(app, wall_done=wall_done, live_done=live_done),
            name="camera-bootstrap",
        )

    if cfg.serve_frames or cfg.allow_evidence:
        # Loud on purpose. These are the two settings that decide whether CCTV
        # imagery of identifiable people can leave the process.
        logger.warning(
            "Imagery egress is enabled — serve_frames={}, allow_evidence={}",
            cfg.serve_frames,
            cfg.allow_evidence,
        )

    yield

    logger.info("Shutting down")
    # Before the runtimes it drives, so a retry cannot start a camera session
    # while the process is tearing them down.
    bootstrap = getattr(app.state, "camera_bootstrap", None)
    if bootstrap is not None and not bootstrap.done():
        bootstrap.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await bootstrap
    await app.state.wall.stop_all()
    if vision.composition is not None:
        app.state.taps.detach(getattr(vision.composition.platform, "bus", None))
    await live.stop_all()
    await vision.stop()
    # After the sessions that submit to it, before the process exits.
    ANALYSIS.stop()
    await cache.disconnect()
    await database.disconnect()
    logger.info("Shutdown complete")


def _observation_log_of(app: FastAPI) -> object | None:
    """The bound ObservationLogPort, or `None` if this process has no synthesis.

    Reached through the composition rather than rebuilt, because a second
    `FileObservationLog` over the same directory would be a second writer to an
    append-only store — and the sweep must truncate the log the pipeline is
    actually appending to, not a lookalike.
    """
    vision = getattr(app.state, "vision", None)
    composition = getattr(vision, "composition", None)
    synthesis = getattr(composition, "synthesis", None) if composition else None
    return getattr(synthesis, "log", None) if synthesis else None


async def _sweep_retention(app: FastAPI) -> None:
    """Apply retention at boot. Never fatal.

    A database that is briefly unreachable while the stack comes up must not
    crash-loop the process, so a failed sweep is reported and start-up
    continues — the next sweep will catch what this one missed. What it must
    never do is fail *silently*: unswept data is a retention promise not kept.
    """
    from app.domain.retention import RetentionService

    cfg: Settings = app.state.settings
    database: Database = app.state.database
    try:
        async with database.session_scope() as session:
            service = RetentionService(
                session,
                root=cfg.evidence_path,
                evidence_days=cfg.evidence_retention_days,
                incident_days=cfg.incident_retention_days,
                audit_days=cfg.audit_retention_days,
                observation_days=cfg.observation_retention_days,
                # `None` when synthesis is not assembled — there is genuinely no
                # log to sweep then, and the service reports that rather than a
                # zero that would read as "swept and found nothing".
                observation_log=_observation_log_of(app),
            )
            report = await service.sweep(erase=cfg.retention_sweep_enabled)
        logger.info("retention at boot: {}", report.as_dict())
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        logger.error(
            "retention sweep failed at boot: {}: {}. Data past its retention "
            "date may still be present.",
            type(exc).__name__,
            exc,
        )


async def _start_cameras_from_database(app: FastAPI) -> int | None:
    """Read the enabled cameras and start them. **This is the recovery path.**

    Returns the number of sessions started, or **`None` when the roster could
    not be read at all**. That distinction is the whole point: "the database
    said zero cameras" and "the database did not answer" are different facts,
    and collapsing both to `0` is what let a deployment boot ahead of Postgres,
    bind nothing, and then sit there looking healthy forever. `None` is the
    signal the bootstrap supervisor retries on.

    Never raises: no camera is better than no application, and `/health/ready`
    reports the database anyway.
    """
    from app.domain.cameras import CameraService, to_rtsp_config

    cfg: Settings = app.state.settings
    database: Database = app.state.database
    live: LiveRuntime = app.state.live

    try:
        async with database.session_scope() as session:
            rows = await CameraService(session).enabled_for_runtime(
                organization_id=cfg.default_tenant_id
            )
            configs = [to_rtsp_config(row) for row in rows if row.host]
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        logger.error(
            "could not read camera configuration: {}: {}. No camera session "
            "started yet; the bootstrap supervisor will retry.",
            type(exc).__name__,
            exc,
        )
        return None

    if not configs:
        # Not an error. A deployment with no enabled camera is a valid
        # deployment, and it says so rather than failing to boot.
        logger.info(
            "no enabled camera rows; nothing to start. Cameras are managed at "
            "/api/v1/cameras, not by CCTV_CHANNELS."
        )
        return 0

    return await live.start_from_records(configs)


def _build_compliance_driver(app: FastAPI, cfg, vision):
    """Load the deployment's rules and start the evaluation pass.

    A rule document that will not load is reported and the driver is left off,
    rather than starting one that silently decides nothing: a compliance system
    that is quietly not running is worse than one that is visibly absent.
    """
    from app.vision.compliance_driver import ComplianceDriver
    from compliance import RuleDocumentError, load_rules

    if not cfg.compliance_rules:
        logger.warning(
            "COMPLIANCE_RULES is not set — observations will be produced and "
            "no compliance decision will be made"
        )
        return None

    try:
        rules = load_rules(cfg.compliance_rules)
    except RuleDocumentError as exc:
        logger.error("compliance rules not loaded: {}. No verdicts will be produced.", exc)
        return None
    if rules is None:
        return None

    producible = frozenset(vision.status().to_wire().get("attributes", ()))
    unproducible = frozenset(rules.required_attributes) - producible
    if unproducible:
        # Named at boot rather than left to look like caution. A rule waiting on
        # an attribute nothing observes sits at UNKNOWN forever, which reads as
        # a careful system and is actually a silent misconfiguration.
        logger.warning(
            "compliance rules depend on attribute(s) nothing observes: {}. "
            "Those conditions can only ever be UNKNOWN.",
            sorted(unproducible),
        )

    from app.domain.notifications import build_notifier

    try:
        notifier = build_notifier(cfg)
    except ValueError as exc:
        # A typo in a channel name must not look like a working deployment that
        # happens to tell nobody.
        logger.error("notifications not configured: {}. No one will be told.", exc)
        notifier = None
    app.state.notifier = notifier

    driver = ComplianceDriver(
        settings=cfg,
        vision=vision,
        database=app.state.database,
        rules=rules,
        interval_s=cfg.compliance_interval_s,
        # Evidence comes from the wall's already-encoded latest frame, so
        # capturing costs nothing on the analysis path.
        wall=app.state.wall,
        notifier=notifier,
    )
    driver.start()
    logger.info(
        "compliance engine started — ruleset {} with {} rule(s), every {}s; "
        "evidence capture {}, notifications {}",
        rules.version,
        len(rules.rules),
        cfg.compliance_interval_s,
        "on" if cfg.evidence_capture else "off",
        notifier.channel_id if notifier else "off",
    )
    return driver


async def _start_camera_wall(app: FastAPI) -> int | None:
    """Open a viewing stream for every enabled camera row.

    Separate from `_start_cameras_from_database`, which starts *analysis*
    sessions. A deployment may reasonably watch sixteen cameras and analyse
    none, and this is the line that keeps those two decisions apart.

    Returns `None` when the camera list could not be read, for the same reason
    `_start_cameras_from_database` does — see its docstring.
    """
    from app.domain.cameras import CameraService

    cfg: Settings = app.state.settings
    database: Database = app.state.database
    wall: CameraWall = app.state.wall
    cameras: list = []
    elsewhere: dict[str, int] = {}

    if not cfg.feature_camera_wall:
        logger.info("camera wall disabled (FEATURE_CAMERA_WALL=false)")
        return 0

    try:
        async with database.session_scope() as session:
            service = CameraService(session)
            cameras = await service.list(organization_id=cfg.default_tenant_id)
            if not cameras:
                elsewhere = await _cameras_in_other_tenants(
                    session, cfg.default_tenant_id
                )
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        logger.error(
            "camera wall could not read its camera list: {}: {}. "
            "The bootstrap supervisor will retry.",
            type(exc).__name__,
            exc,
        )
        return None

    if not cameras:
        if elsewhere:
            # The failure that produced "0 channels, 16 connecting": cameras
            # exist, but in a tenant this deployment does not serve. Silence
            # here reads as "no cameras configured", which sends somebody to
            # look at the DVR instead of at DEFAULT_TENANT_ID.
            logger.error(
                "camera wall found no cameras for tenant '{}', but {} camera "
                "row(s) exist in tenant(s) {}. DEFAULT_TENANT_ID does not match "
                "the tenant that owns the cameras.",
                cfg.default_tenant_id,
                sum(elsewhere.values()),
                sorted(elsewhere),
            )
        else:
            logger.info("camera wall: no camera rows configured")
        return 0
    return await wall.start_cameras(cameras)


#: Backoff for the camera bootstrap supervisor, in seconds.
#:
#: Starts fast because the common case is a stack coming up together and the
#: database being seconds behind, and settles at 30 s because a database that
#: has been down for a minute is an outage, not a race, and polling it harder
#: helps nobody.
_BOOTSTRAP_BACKOFF = (1.0, 2.0, 5.0, 10.0, 15.0, 30.0)


async def _bootstrap_cameras_once(
    app: FastAPI, *, need_wall: bool = True, need_live: bool = True
) -> tuple[bool, bool]:
    """One attempt at binding the camera roster. Returns `(wall_read, live_read)`.

    Both halves are attempted, because the wall and the analysis sessions are
    independent decisions — a deployment may legitimately view sixteen cameras
    and analyse none.

    They are also reported **separately**, and that is not fussiness. The two
    reads happen seconds apart, so a database finishing its start-up between
    them leaves one succeeding and one failing. Treating that as "bootstrapped"
    strands whichever half lost the race: observed on 2026-08-31, where the wall
    read failed at 08:21:54, the live read succeeded at 08:21:57, and the wall —
    the half `/api/v1/wall/cameras` reports and the UI renders — stayed empty
    behind a log line announcing success.

    `need_wall` / `need_live` let the supervisor retry only what is outstanding,
    so a half that already bound is not asked to bind twice. That matters for
    the analysis sessions, which refuse a duplicate camera outright.
    """
    wall_read = not need_wall
    live_read = not need_live

    if need_wall:
        wall_read = await _start_camera_wall(app) is not None

    if need_live:
        live_result = await _start_cameras_from_database(app)
        live_read = live_result is not None
        if live_result:
            logger.warning(
                "live CCTV runtime started {} camera session(s)", live_result
            )

    # `0` is a real answer ("no enabled cameras"); only `None` means nobody
    # answered, and only a read that happened counts as done.
    return wall_read, live_read


async def _camera_bootstrap_supervisor(
    app: FastAPI, *, wall_done: bool = False, live_done: bool = False
) -> None:
    """Keep trying to bind cameras until the database answers. Then stop.

    ### Why the one-shot bind was unsafe

    Camera binding used to happen exactly once, inline in the lifespan. If the
    process started before Postgres accepted connections — which is ordinary
    when a stack comes up together, and was reproduced on 2026-08-31 with the
    backend booting ten minutes ahead of its database — the roster read failed,
    the handler logged an error, and **nothing ever tried again**.

    What made that failure mode expensive is that everything else recovered on
    its own. SQLAlchemy reconnects lazily, so the API started serving the moment
    the database appeared, and `/health/ready` went green. Only the cameras
    stayed dark, with `frames_decoded=0`, `reconnects=0` and an empty
    `last_error` — the signature of something that never started, sitting behind
    a dashboard that otherwise looked healthy. The remedy was a manual restart
    that nothing in the system asked for.

    ### Why a supervisor rather than a readiness gate

    Blocking start-up until the database answers would trade a silent failure
    for a crash loop, and would take down every route that does not need the
    database — including `/health`, which is what an operator reads to find out
    what is wrong. Retrying in the background keeps the process serving while
    the dependency arrives.

    ### Why it stops

    It retries the **bootstrap**, not the cameras. Once the roster is read the
    task exits and ordinary mechanisms own the rest: RTSP reconnection is the
    session's own `ReconnectPolicy`, and enabling a camera later goes through
    `PATCH /api/v1/cameras/{key}`. A supervisor that kept polling would be a
    second, competing source of truth for which cameras should be running.
    """
    need_wall = not wall_done
    need_live = not live_done
    attempt = 0

    while need_wall or need_live:
        attempt += 1
        # Walk the backoff, then hold at its last value. A database that returns
        # after ten minutes should still get its cameras, and this task costs
        # one query per interval.
        delay = _BOOTSTRAP_BACKOFF[min(attempt - 1, len(_BOOTSTRAP_BACKOFF) - 1)]
        await asyncio.sleep(delay)
        try:
            wall_ok, live_ok = await _bootstrap_cameras_once(
                app, need_wall=need_wall, need_live=need_live
            )
            # Only retry what is still outstanding. Re-running a half that
            # already bound would ask the analysis runtime to start a camera it
            # is already running, which it refuses — a real error logged for a
            # non-problem.
            if wall_ok:
                need_wall = False
            if live_ok:
                need_live = False
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a retry must not kill the task
            logger.error(
                "camera bootstrap retry {} failed: {}: {}",
                attempt,
                type(exc).__name__,
                exc,
            )

    logger.warning(
        "camera bootstrap completed on retry {} — the database became "
        "available after start-up",
        attempt,
    )


async def _cameras_in_other_tenants(session, tenant_id: str) -> dict[str, int]:
    """Camera counts per tenant, excluding this one. For diagnosing a mismatch."""
    from sqlalchemy import func, select

    from app.domain.models import Camera

    result = await session.execute(
        select(Camera.organization_id, func.count())
        .where(Camera.organization_id != tenant_id)
        .group_by(Camera.organization_id)
    )
    return {row[0]: int(row[1]) for row in result.all()}


def _install_error_handlers(app: FastAPI) -> None:
    def _request_id(request: Request) -> str:
        return getattr(request.state, "request_id", "")

    # Codes that are part of a normal flow rather than a problem. They still
    # return their status; they just do not deserve a line in the log every time
    # somebody opens the page.
    _routine = {"NO_SESSION"}

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        if exc.code in _routine:
            logger.debug("{}: {}", exc.code, exc.message)
        else:
            logger.info("{}: {}", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.http_status, content=exc.to_envelope(_request_id(request))
        )

    # Vision OS renders its own typed errors with their stable `code`. Registered
    # on the base class so every current and future platform error keeps it,
    # without this file enumerating them — and so a documented policy bound like
    # `WindowTooLargeError` reaches the client as guidance rather than as a 500.
    try:
        from vision_os.core.errors import VisionOSError

        @app.exception_handler(VisionOSError)
        async def _vision_error(request: Request, exc: VisionOSError) -> JSONResponse:
            code = getattr(exc, "code", "VISION_ERROR")
            logger.warning("Vision OS error {}: {}", code, exc)
            return JSONResponse(
                status_code=400,
                content={
                    "code": str(code),
                    "message": str(exc),
                    "retryable": False,
                    "details": {},
                    "request_id": _request_id(request),
                },
            )
    except Exception:  # noqa: BLE001 - platform absent; the generic handler covers it
        logger.warning("Vision OS error types unavailable; generic handling applies")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # The detail goes to the log, indexed by request id. The client gets the
        # id and nothing else: a traceback in a response body is a file-system
        # map, a dependency inventory and often a fragment of a query.
        logger.exception("Unhandled {}", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "code": "INTERNAL",
                "message": "an unexpected error occurred",
                "retryable": False,
                "details": {},
                "request_id": _request_id(request),
            },
        )


def run() -> None:
    """Console-script entry point."""
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        "app.main:app",
        host=cfg.app_host,
        port=cfg.app_port,
        reload=cfg.app_debug,
        log_config=None,
    )


_app: FastAPI | None = None


def __getattr__(name: str) -> Any:
    """Build ``app`` on first access, so importing this module builds nothing.

    ``uvicorn app.main:app`` still resolves, because that is an attribute lookup.
    A module-level ``app = create_app()`` would instead construct a
    default-configured application every time anything imported this file —
    including a test that only wanted ``create_app`` with its own settings, and
    including tooling that imports for introspection.
    """
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app", "create_app", "run"]
