"""The observation fold: platform observations → product subjects. **One copy.**

### Why this module exists at all

The fold used to live in `app/api/product.py`, and `app/reporting/sources.py`
imported it from there — reporting reaching up into the API layer, which is
backwards. Phase 3 disclosed that rather than hiding it, and this is the fix.

The alternative was a second implementation in the reporting layer, and that is
precisely the wrong trade. This function is the one place where an attribute
value from Vision OS becomes a record a product surface renders, and therefore
the one place where `not_visible` could be quietly collapsed into `none`. Two
copies means two places to get that wrong, and the second one drifts — the same
argument `app/domain/models.py` makes about a second source of truth for
perception.

So: one implementation, in the domain layer, imported by both consumers. A test
calls it through both call sites and asserts byte-identical output.

### The application still stores none of this

Nothing here writes a row. It is a **projection** of Vision OS's own observation
log, read through its Observation API and shaped for a screen — which is exactly
what `models.py` permits and what an application-side perception table would not
be.

### The rule the fold must never break

Values are passed through as the platform reported them. No normalisation, no
mapping to a boolean, no collapsing of the four states into two. Resolution into
PRESENT / ABSENT / NOT_VISIBLE / UNKNOWN happens once, on the frontend, in
`shared/semantics/observation.ts`. Anything this module "helpfully" tidied would
be a verdict invented in transport, about a person nobody could see.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.authorization.model import AccessDecision


def ppe_state(value: object) -> str:
    """The attribute value, passed through **exactly** as the platform reported.

    No normalisation, no mapping to a boolean, no collapsing of `not_visible`
    into `none`. The four states are the product's core safety property and the
    frontend resolves them through its own `observation.ts`; anything this
    function "helpfully" tidied would be a verdict invented in transport.
    """
    return "" if value is None else str(value)


def observation_confidence(value: Any) -> dict[str, Any] | None:
    """Confidence with its semantics attached.

    `SELF_REPORTED` is a model's opinion about itself and 02_VOM §7.2 states it
    *"is not a probability"*. Carrying the semantics means a UI cannot render it
    as one without saying so.
    """
    if value is None:
        return None
    return {
        "value": float(getattr(value, "value", 0.0)),
        "semantics": getattr(getattr(value, "semantics", None), "value", ""),
        "calibrated": bool(getattr(value, "calibrated", False)),
    }


def query_observations(
    api: Any,
    access: AccessDecision,
    cameras: tuple[str, ...],
    start: datetime,
    end: datetime,
    limit: int,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Read the platform's log and fold it into subjects.

    Returns `(subjects, observation_count, window_fully_observable)`.

    Kept free of FastAPI and of the reporting engine so it is testable without a
    request and importable by both without either depending on the other. The
    platform imports stay inside the function rather than at module import time,
    matching how every other boundary in this application reaches Vision OS.
    """
    from vision_os.core.model.api import Principal, Scope, TimeWindow
    from vision_os.core.model.ids import CameraId, TenantId
    from vision_os.core.model.timebase import Instant

    tenant = TenantId(access.tenant_id)
    principal = Principal(
        subject=access.subject,
        tenant_id=tenant,
        scopes=tuple(sorted(p.value for p in access.permissions)),
        display_name=access.display_name,
    )
    scope = Scope(tenant_id=tenant, camera_ids=tuple(CameraId(c) for c in cameras))
    window = TimeWindow(
        start=Instant(int(start.timestamp() * 1_000_000_000)),
        end=Instant(int(end.timestamp() * 1_000_000_000)),
    )

    page = api.query_observations(principal, scope, window, limit=limit)

    # object_id → subject. Insertion order is the platform's ordering, which is
    # (t_capture, observation_id) and therefore oldest first; the newest value
    # for an attribute is the last one seen, which is why this overwrites.
    subjects: dict[str, dict[str, Any]] = {}
    observations = tuple(getattr(page, "observations", ()) or ())

    for observation in observations:
        object_id = str(getattr(observation, "object_id", "") or "")
        if not object_id:
            # `coverage` observations carry no subject: they are statements
            # about the platform, not about anyone seen. Not a hygiene row.
            continue

        captured = getattr(getattr(observation, "t_capture", None), "ns", None)
        subject = subjects.setdefault(
            object_id,
            {
                "object_id": object_id,
                "camera_key": str(getattr(observation, "camera_id", "") or ""),
                "class_id": str(getattr(observation, "class_id", "") or ""),
                "first_seen": captured,
                "last_seen": captured,
                "attributes": {},
            },
        )
        if captured is not None:
            if subject["first_seen"] is None or captured < subject["first_seen"]:
                subject["first_seen"] = captured
            if subject["last_seen"] is None or captured > subject["last_seen"]:
                subject["last_seen"] = captured

        for attribute in getattr(observation, "attributes", ()) or ():
            key = str(getattr(attribute, "key", "") or "")
            if not key:
                continue
            observed_at = getattr(getattr(attribute, "observed_at", None), "ns", None)
            held = subject["attributes"].get(key)
            # Keep the freshest reading. Two observations can share a capture
            # instant, so ties keep the later one in log order rather than
            # flipping unpredictably between equal timestamps.
            if held is not None and observed_at is not None and held["observed_at"] is not None:
                if observed_at < held["observed_at"]:
                    continue
            subject["attributes"][key] = {
                "key": key,
                "value": ppe_state(getattr(attribute, "value", None)),
                "observed_at": observed_at,
                "valid_until": getattr(getattr(attribute, "valid_until", None), "ns", None),
                "confidence": observation_confidence(getattr(attribute, "confidence", None)),
            }

    rendered = []
    for subject in subjects.values():
        subject["attributes"] = sorted(subject["attributes"].values(), key=lambda a: a["key"])
        rendered.append(subject)
    rendered.sort(key=lambda s: (s["last_seen"] or 0), reverse=True)

    return rendered, len(observations), bool(getattr(page, "window_fully_observable", True))


__all__ = ["observation_confidence", "ppe_state", "query_observations"]
