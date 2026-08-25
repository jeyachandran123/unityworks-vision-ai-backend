"""DevTools read routes.

Every route here requires, in order: authentication, `ACCESS_DEVTOOLS`, a valid
tenant, and — for imagery — the separate evidence privilege. The
`FEATURE_DEVTOOLS` flag decides whether the router is mounted at all, so a
deployment that has not enabled DevTools returns 404 rather than 403: the routes
do not exist to be forbidden.

**Read-only, structurally.** There is no write path to Vision State here, and
none to find: `ObservationApi` exposes none to call.

### Why these routes serve a fixture in this phase

Phase 1 binds no `SourcePort`, so nothing acquires frames. Rather than return
empty results — which would be indistinguishable from "the platform observed
nothing", the exact confusion invariant V8 exists to prevent — DevTools serves a
**labelled fixture** through the real Observation API, with real authorization
and real scoping. Every response says `"kind": "fixture"`.

Phase 3 replaces the fixture with a file-replay source and these routes do not
change.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import CurrentAccess, live_of, requires, settings_of, vision_of
from app.authorization.model import Permission
from app.errors import EvidenceForbiddenError

router = APIRouter(prefix="/api/v1/devtools", tags=["devtools"])

_REQUIRE_DEVTOOLS = Depends(requires(Permission.ACCESS_DEVTOOLS))


def _fixture(access) -> Any:
    """Build the fixture for the caller's tenant.

    Per-tenant rather than process-wide, so a DevTools user in one organization
    cannot see another's fixture — the same scoping rule the real path obeys.
    """
    from app.vision.fixture import build_fixture_session

    return build_fixture_session(access.tenant_id)


@router.get("/vision", dependencies=[_REQUIRE_DEVTOOLS])
async def vision_diagnostics(request: Request) -> dict[str, Any]:
    """What the platform is configured to observe.

    No imagery, no credentials, no principal — this answers "what is this
    platform configured to see", which is the question an engineer debugging a
    silent camera actually has.
    """
    vision = vision_of(request)
    settings = settings_of(request)
    payload = vision.status().to_wire()
    payload["imagery"] = {
        "serve_frames": settings.serve_frames,
        "allow_evidence": settings.allow_evidence,
    }
    return payload


@router.get("/observations", dependencies=[_REQUIRE_DEVTOOLS])
async def real_observations(request: Request, access: CurrentAccess) -> dict[str, Any]:
    """Vision State for the **real** running platform, not the fixture.

    `/state` answers from `build_fixture_session` and always has — it says
    `"kind": "fixture"` and is honest about it. But the exposure layer built
    over the live synthesis state has had no route at all, so the observations
    the platform actually produced from real cameras could not be read from
    outside the process.

    Scope is the caller's own cameras, exactly as `/state` does it, and
    `_render` is shared with it so `not_visible` survives to the wire here for
    the same reason it does there.

    **`StateFilter` is left at its default.** Phase 6A.4 traced an empty result
    to that default excluding `PROVISIONAL` objects and widening it was
    explicitly rejected then: an object the registry has not confirmed is not
    something a compliance decision may rest on, and showing it here to make a
    screen look populated would be the same mistake in a different place.
    """
    vision = vision_of(request)
    exposure = getattr(getattr(vision, "composition", None), "exposure", None)
    if exposure is None:
        return {
            "kind": "real",
            "available": False,
            "reason": "no exposure layer; Vision OS is not assembled",
            "objects": [],
        }

    from vision_os.core.model.api import Scope
    from vision_os.core.model.ids import CameraId, TenantId

    cameras = _visible_cameras(access)
    if cameras is None:
        live = live_of(request)
        cameras = tuple(
            s.camera_id for s in live.visible(tenant_id=access.tenant_id, camera_ids=None)
        )

    scope = Scope(
        tenant_id=TenantId(access.tenant_id),
        camera_ids=tuple(CameraId(c) for c in cameras),
    )
    result = exposure.api.query_state(_principal(access), scope)
    objects = [_render(obj) for obj in result.objects]
    return {
        "kind": "real",
        "available": True,
        "cameras_queried": list(cameras),
        "complete": bool(getattr(result, "complete", True)),
        "object_count": len(objects),
        "objects": objects,
    }


@router.get("/compliance", dependencies=[_REQUIRE_DEVTOOLS])
async def compliance_state(request: Request) -> dict[str, Any]:
    """The live compliance verdicts, and the rules that produced them.

    Verdicts are recomputed from current Vision State on each call rather than
    read from a cache, so what this returns is what the rules say *now*. It does
    not write: incidents are opened by the driver's own pass, and a read that
    raised incidents would make every refresh of a dashboard a business event.
    """
    driver = getattr(request.app.state, "compliance", None)
    if driver is None:
        return {
            "available": False,
            "reason": "no compliance engine; rules unloaded or Vision OS unassembled",
        }

    from app.vision.compliance_driver import _finding_wire

    cameras = await driver.cameras()
    run, findings = driver.evaluate(driver.snapshot(tuple(cameras)))

    rules = driver.evaluator.rules
    return {
        "available": True,
        "ruleset_version": rules.version,
        "rules": [
            {
                "rule_id": r.rule_id,
                "version": r.version,
                "severity": r.severity,
                "attributes": sorted(r.attributes),
            }
            for r in rules.rules
        ],
        "summary": run.to_wire(),
        "last_applied_pass": driver.last_pass.to_wire(),
        "findings": [_finding_wire(f) for f in findings],
    }


@router.get("/metrics", dependencies=[_REQUIRE_DEVTOOLS])
async def platform_metrics(request: Request) -> dict[str, Any]:
    """The **real** platform's own counters, gauges and histograms.

    Every other diagnostic here is either a fixture (`/state`,
    `/capabilities`) or describes the acquisition side (`/live`). None of them
    can answer the question an engineer actually has about a running kitchen:
    *did a person get detected, was a model asked about them, did the answer
    come back, and was it written to the object?*

    `metrics_view` and `health_view` were built in Phase 6 to answer exactly
    that and were never given a route, so the numbers existed in the process
    and could not be read from outside it. This is that route, and it declares
    no metric names of its own — the vocabulary is the platform's.
    """
    from app.vision.taps import health_view, metrics_view

    vision = vision_of(request)
    platform = getattr(getattr(vision, "composition", None), "platform", None)
    if platform is None:
        return {"assembled": False, "reason": vision.status().to_wire().get("reason", "")}

    return {
        "assembled": True,
        "metrics": metrics_view(platform),
        "health": health_view(platform),
    }


@router.get("/sessions", dependencies=[_REQUIRE_DEVTOOLS])
async def sessions(request: Request, access: CurrentAccess) -> dict[str, Any]:
    """Sessions available to inspect — **real ones first**.

    Live and replay sessions come from the runtime and are scoped to the
    caller's tenant and cameras. The fixture is appended and labelled, so a real
    session is never confused with it and DevTools keeps working before any
    source is configured.
    """
    live = live_of(request)
    cameras = _visible_cameras(access)
    real = [
        session.to_wire()
        for session in live.visible(tenant_id=access.tenant_id, camera_ids=cameras)
    ]
    return {
        "runtime": live.summary().to_wire(),
        "sessions": real + [_fixture(access).to_wire()],
        "cameras_configured": live.describe_cameras(),
    }


@router.get("/live", dependencies=[_REQUIRE_DEVTOOLS])
async def live_runtime(request: Request, access: CurrentAccess) -> dict[str, Any]:
    """The live runtime in full: sessions, queues, sources, drop counters.

    Everything a source knows except the credential. The URI is the redacted
    form and there is no code path here that can produce the real one.
    """
    live = live_of(request)
    cameras = _visible_cameras(access)
    sessions = live.visible(tenant_id=access.tenant_id, camera_ids=cameras)
    return {
        "runtime": live.summary().to_wire(),
        "sessions": [session.to_wire() for session in sessions],
        "cameras_configured": live.describe_cameras(),
        "backpressure": {
            "policy": "drop-oldest",
            "rationale": (
                "For live monitoring the newest frame answers the question being "
                "asked. Blocking the producer to keep old frames stalls the "
                "decoder and turns a processing problem into a camera outage."
            ),
        },
    }


def _visible_cameras(access) -> tuple[str, ...] | None:
    """The caller's camera scope. `None` is tenant-wide, `()` is none."""
    from app.authorization.model import ScopeBreadth

    if access.cameras.breadth is ScopeBreadth.ALL_IN_TENANT:
        return None
    return access.cameras.camera_ids


@router.get("/capabilities", dependencies=[_REQUIRE_DEVTOOLS])
async def capabilities(access: CurrentAccess) -> dict[str, Any]:
    """Live capability, from the platform.

    *"Capability is live state, not documentation"* (09_API §5.2). What a bound
    model can actually produce is the difference between a rule that can reach a
    verdict and one that will sit at UNKNOWN forever.
    """
    session = _fixture(access)
    summary = session.api.capabilities(_principal(access), _scope(access, session))
    return {
        "kind": "fixture",
        "taxonomy_version": getattr(summary, "taxonomy_version", ""),
        "producible_classes": [str(c) for c in getattr(summary, "producible_classes", ())],
        "producible_attributes": [str(a) for a in getattr(summary, "producible_attributes", ())],
    }


@router.get("/state", dependencies=[_REQUIRE_DEVTOOLS])
async def vision_state(access: CurrentAccess) -> dict[str, Any]:
    """Current Vision State for the caller's scope.

    The scope is the one the authorizer returned, never the one the client
    asked for. 12_SECURITY §4.2 designs the leak out by constructing every query
    already scoped, and post-filtering here would quietly reintroduce it.
    """
    session = _fixture(access)
    result = session.api.query_state(_principal(access), _scope(access, session))

    # `result.objects` is the API's answer; `result.snapshot.partitions` names
    # which cameras it covers. Both are reported, because "three objects" and
    # "three objects across one camera, and a second camera returned nothing"
    # are different facts and an engineer needs the second one.
    objects = [_render(obj) for obj in result.objects]

    return {
        "kind": "fixture",
        "session_id": session.to_wire()["session_id"],
        "observation_count": session.observation_count,
        "complete": bool(getattr(result, "complete", True)),
        "partitions": [
            {
                "camera_id": str(camera_id),
                "object_count": sum(1 for o in objects if o["camera_id"] == str(camera_id)),
            }
            for camera_id in result.snapshot.partitions
        ],
        "objects": objects,
    }


@router.get("/evidence/{blob_ref}", dependencies=[_REQUIRE_DEVTOOLS])
async def evidence(blob_ref: str, request: Request, access: CurrentAccess) -> dict[str, Any]:
    """Retrieve evidence imagery.

    Two gates beyond DevTools, and neither is implied by the other:

    1. `ALLOW_EVIDENCE` — a deployment decision. Off by default.
    2. `VIEW_EVIDENCE` — the caller's own privilege. Never implied by
       `VIEW_OBSERVATIONS`: *"Reading 'a person was here' and viewing their image
       are categorically different acts."*
    """
    from app.infrastructure.observability import EVIDENCE_ACCESS

    if not settings_of(request).allow_evidence:
        EVIDENCE_ACCESS.labels("deployment_disabled").inc()
        raise EvidenceForbiddenError(
            "evidence retrieval is disabled for this deployment",
            details={"setting": "ALLOW_EVIDENCE"},
        )

    if not access.has(Permission.VIEW_EVIDENCE):
        EVIDENCE_ACCESS.labels("forbidden").inc()
        raise EvidenceForbiddenError(
            "this account may read observations but not the imagery behind them",
            details={"required": Permission.VIEW_EVIDENCE.value},
        )

    EVIDENCE_ACCESS.labels("unavailable").inc()
    # The fixture stores no blobs. Reported as unavailable rather than as an
    # empty image, because "there is no evidence here" and "you may not see it"
    # must never be the same answer.
    return {
        "kind": "fixture",
        "blob_ref": blob_ref,
        "available": False,
        "reason": "the fixture session stores no evidence blobs",
    }


def _principal(access):
    """The Vision OS principal for a DevTools caller.

    Fixed subject `devtools`, matching the fixture's grant, and the caller's real
    tenant. External identity exists only at this boundary and travels no
    further down.
    """
    from vision_os.core.model.api import Principal
    from vision_os.core.model.ids import TenantId

    return Principal(subject="devtools", tenant_id=TenantId(access.tenant_id))


def _scope(access, session):
    from vision_os.core.model.api import Scope
    from vision_os.core.model.ids import CameraId, TenantId

    return Scope(
        tenant_id=TenantId(access.tenant_id),
        camera_ids=(CameraId(session.camera_id),),
    )


def _render(obj) -> dict[str, Any]:
    """One `ObjectView` for the wire, attributes included.

    **Attribute values pass through exactly as the platform reported them** —
    `not_visible` stays `not_visible`. Collapsing it to a boolean anywhere
    between here and the screen would destroy the distinction between "observed
    absent" and "could not see", and one of those is a violation while the other
    never is. It is the whole reason the domain carries five values, not two.
    """
    attributes = getattr(obj, "attributes", {}) or {}
    items = attributes.items() if hasattr(attributes, "items") else ()

    return {
        "object_id": str(getattr(obj, "object_id", "")),
        "camera_id": str(getattr(obj, "camera_id", "")),
        "class_id": str(getattr(obj, "class_id", "")),
        "lifecycle": getattr(getattr(obj, "lifecycle", None), "value", ""),
        "first_seen": _instant(getattr(obj, "first_seen", None)),
        "last_seen": _instant(getattr(obj, "last_seen", None)),
        "observation_count": int(getattr(obj, "observation_count", 0) or 0),
        "attributes": [
            {
                "key": str(key),
                "value": str(getattr(view, "value", "")),
                "observed_at": _instant(getattr(view, "observed_at", None)),
                "valid_until": _instant(getattr(view, "valid_until", None)),
                # SELF_REPORTED confidence is a model's opinion about itself and
                # 02_VOM §7.2 says it "is not a probability". Reported with its
                # semantics attached so a UI cannot present it as one.
                "confidence": _confidence(getattr(view, "confidence", None)),
            }
            for key, view in items
        ],
    }


def _instant(value) -> int | None:
    ns = getattr(value, "ns", None)
    return int(ns) if ns is not None else None


def _confidence(value) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "value": float(getattr(value, "value", 0.0)),
        "semantics": getattr(getattr(value, "semantics", None), "value", ""),
        "calibrated": bool(getattr(value, "calibrated", False)),
    }


__all__ = ["router"]
