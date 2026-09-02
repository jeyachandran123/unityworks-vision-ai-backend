"""The POS/ERP integration surface.

Two routes: the module's capability, and the connector list. Both read; nothing
writes, because writing a connector means accepting a credential reference and a
base URL for a system that reaches sales and often payment data, and the vendor
whose shape that configuration must take has not been chosen.

### The connector list is real and read-only

`pos_connectors` is a real table and this returns its real (empty) contents,
scoped to the caller's tenant. `credential_ref` is **not** in the response
shape — the row holds a reference rather than a secret, and there is still no
reason for an administration screen to see even the reference: knowing that a
credential lives at `env:POS_TOKEN` is a small disclosure, and small disclosures
are how the large ones are assembled.

`write_available: false` with a reason follows the pattern `/users` established:
the server decides whether a write path exists and carries the reason in the
payload, so one place owns it and the frontend renders the sentence rather than
inventing one.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import select

from app.api.capability import ModuleCapability, Requirement, render
from app.api.dependencies import CurrentAccess, DbSession, requires
from app.authorization.model import Permission
from app.domain import modules as module_models
from app.integrations.pos import POS_REQUIREMENTS, gateway_for

router = APIRouter(prefix="/api/v1", tags=["integrations"])


POS_INTEGRATION = ModuleCapability(
    module="pos_integration",
    title="POS / ERP Integration",
    purpose=(
        "The seam between this system and a point-of-sale or ERP. Underlies "
        "meal-detection reconciliation and any future order or table sync."
    ),
    reason=(
        "No POS adapter is bound. The port exists and the only adapter behind "
        "it is `pos.not_configured`, which refuses every call by name rather "
        "than returning an empty result — because 'no tickets' and 'not "
        "connected' are different answers and only one of them is a zero."
    ),
    requirements=tuple(Requirement(name, detail) for name, detail in POS_REQUIREMENTS),
    tables=("pos_connectors", "pos_sync_runs"),
    documentation="docs/architecture/NOT_YET_CONNECTED.md#pos--erp-integration",
)


@router.get(
    "/modules/pos-integration",
    dependencies=[Depends(requires(Permission.VIEW_POS_INTEGRATION))],
)
async def pos_integration(access: CurrentAccess, session: DbSession) -> dict[str, Any]:
    """POS capability, including what the bound adapter says about itself."""
    gateway = gateway_for()
    description = gateway.describe()

    return await render(
        POS_INTEGRATION,
        session,
        organization_id=access.tenant_id,
        models=(module_models.PosConnector, module_models.PosSyncRun),
        extra={
            # The adapter is asked rather than assumed, so this line stays true
            # on the day a real one is bound without this route changing.
            "adapter": {
                "bound": True,
                "id": "pos.not_configured",
                "vendor": description.vendor,
                "display_name": description.display_name,
                "available": description.available,
                "reason": description.reason,
                "capabilities": list(description.capabilities),
            },
            "write_available": False,
            "write_unavailable_reason": (
                "Registering a connector accepts a base URL and a credential "
                "reference for a system that reaches sales and often payment "
                "data. The shape that configuration must take depends on the "
                "vendor, and no vendor has been chosen."
            ),
        },
    )


@router.get(
    "/pos-connectors",
    dependencies=[Depends(requires(Permission.VIEW_POS_INTEGRATION))],
)
async def list_pos_connectors(access: CurrentAccess, session: DbSession) -> dict[str, Any]:
    """Configured connectors for this organisation. Real, and currently none.

    `credential_ref` is deliberately absent from every row. The database holds a
    reference rather than a secret, and an administration screen has no use even
    for the reference.
    """
    rows = (
        (
            await session.execute(
                select(module_models.PosConnector)
                .where(module_models.PosConnector.organization_id == access.tenant_id)
                .order_by(module_models.PosConnector.connector_key)
            )
        )
        .scalars()
        .all()
    )

    return {
        "connectors": [
            {
                "id": row.id,
                "connector_key": row.connector_key,
                "vendor": row.vendor,
                "display_name": row.display_name,
                "restaurant_id": row.restaurant_id,
                "is_active": bool(row.is_active),
                "capabilities": [c for c in row.capabilities.split(",") if c],
                "last_success_at": row.last_success_at.isoformat()
                if row.last_success_at
                else None,
                "last_error_at": row.last_error_at.isoformat() if row.last_error_at else None,
                "last_error": row.last_error,
            }
            for row in rows
        ],
        "count": len(rows),
        "write_available": False,
    }


__all__ = ["router"]
