"""Unique Patron ID — a surface whose only function is to report that it is shut.

In its own module rather than beside the other five, because the difference is
not one of degree. The others are waiting for *work*: a dataset, a floor plan, a
vendor. This one is waiting for *permission*, and a reader skimming a file of
five similar capability routes would not see that.

### The state is `blocked`, not `not_configured`

Every other module in this phase reports `not_configured` — nobody has connected
it yet. This reports `blocked`, and the distinction is the whole point: a
deployment could satisfy every configuration input here and still must not turn
it on, because what is missing is a completed DPIA and a named sign-off rather
than an engineering task.

### There is no write route, and the read route is gated twice

`VIEW_PATRON_ID` admits the status read. `MANAGE_PATRON_ID` — held by
`SUPER_ADMIN` alone — admits the detail of the gate itself. No route accepts a
token, and `app/domain/patron.require_writable` refuses unconditionally, so the
first real write is not one line away from existing.

### What the platform already says about this

Vision OS declares `EmbeddingPort` (P10) as **C2 · Biometric** and leaves it
*"declared, unbound, and unimplemented … deliberately"*; `IdentityResolverPort`
(P11) is *"Phase 2 and unimplemented"*; and 07_STATE §8.2 states the platform
*"holds no persistent biometric identity, which is a deliberate privacy posture,
not a limitation."* Both ports this module would need therefore already exist —
which is precisely why **no new adapter was written for it in this phase**.
Binding one is the act that requires the legal artifact, and it is exactly the
act that has not been authorised.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.api.capability import ModuleCapability, Requirement, render
from app.api.dependencies import CurrentAccess, DbSession, requires, settings_of
from app.authorization.model import Permission
from app.domain import modules as module_models
from app.domain import patron as patron_domain

router = APIRouter(prefix="/api/v1/modules", tags=["modules"])


PATRON_ID = ModuleCapability(
    module="patron_id",
    title="Unique Patron ID",
    purpose=(
        "A pseudonymous, site-scoped handle for a returning visitor — a salted "
        "hash and a visit count, never a face and never a name."
    ),
    reason=(
        "Blocked. This is the only module in the product that would identify a "
        "person across visits, and it cannot be enabled by configuration: it "
        "needs a completed Data Protection Impact Assessment, a working consent "
        "mechanism, and a named DPO sign-off. The perception platform "
        "deliberately holds no persistent biometric identity, and turning this "
        "on means contradicting that posture on purpose."
    ),
    requirements=tuple(
        Requirement(name, detail) for name, detail in patron_domain.REQUIREMENTS
    )
    + (
        Requirement(
            "biometric_source",
            "A bound biometric source. Vision OS's EmbeddingPort is classified "
            "C2 · Biometric and is left declared-but-unbound deliberately, so "
            "there is currently nothing that could produce a digest to hash. "
            "This is last because it is the smallest of the five problems.",
        ),
    ),
    tables=("patron_tokens",),
    documentation="docs/architecture/NOT_YET_CONNECTED.md#unique-patron-id",
    state="blocked",
)


@router.get("/patron-id", dependencies=[Depends(requires(Permission.VIEW_PATRON_ID))])
async def patron_id(
    request: Request, access: CurrentAccess, session: DbSession
) -> dict[str, Any]:
    """Whether patron identification is enabled. It is not, and it says why.

    The row count is real and is zero. It is reported for one reason: an
    operator asking "are we doing this" deserves an answer read from the table
    rather than from a claim in a document.
    """
    settings = settings_of(request)
    gate = patron_domain.gate_status(settings)

    return await render(
        PATRON_ID,
        session,
        organization_id=access.tenant_id,
        models=(module_models.PatronToken,),
        extra={
            "gate": gate.as_dict(),
            # Stated by the server so the page renders the guarantee from the
            # schema rather than asserting it on the frontend's own authority.
            "schema_guarantees": [
                "patron_tokens.token_hash is String(64) — a hex SHA-256 digest "
                "fits, a biometric template does not.",
                "patron_tokens has no binary column, and no image, template or "
                "embedding reference of any kind.",
                "consent_ref and legal_gate_ref are NOT NULL with no default: "
                "the database refuses a token that names neither the consent "
                "permitting it nor the approval authorising the capability.",
                "Erasure is a tombstone, matching evidence: the hash is "
                "cleared and the row survives, so a deletion stays provable.",
            ],
            # False, and there is no route that could make it true.
            "write_available": False,
            "write_unavailable_reason": (
                "No route accepts a patron token, and the domain write path "
                "refuses unconditionally. The refusal is not a permission "
                "failure and cannot be resolved by granting access."
            ),
        },
    )


@router.get(
    "/patron-id/gate",
    dependencies=[Depends(requires(Permission.MANAGE_PATRON_ID))],
)
async def patron_id_gate(request: Request, access: CurrentAccess) -> dict[str, Any]:
    """The gate in detail, for whoever would be responsible for opening it.

    Gated on `MANAGE_PATRON_ID`, which `SUPER_ADMIN` alone holds. Reading the
    detail of what would unlock biometric re-identification is itself a
    privilege, and an organisation administrator is the wrong altitude for it.

    Read-only and side-effect free. Asking changes nothing, and there is no
    corresponding write route — this reports the state of a decision, it does
    not make one.
    """
    settings = settings_of(request)
    gate = patron_domain.gate_status(settings)

    satisfied = {name for name, _ in patron_domain.REQUIREMENTS} - set(gate.missing)
    return {
        "module": "patron_id",
        "available": gate.available,
        "reason": gate.reason,
        "missing": list(gate.missing),
        "satisfied": sorted(satisfied),
        "requirements": [
            {"id": name, "detail": detail, "satisfied": name not in gate.missing}
            for name, detail in patron_domain.REQUIREMENTS
        ],
        # Named, never valued. Whether a legal gate reference is *set* is the
        # useful fact; what it points at is not this endpoint's business.
        "legal_gate_recorded": bool(settings.patron_id_legal_gate_ref.strip()),
        "pepper_reference_recorded": bool(settings.patron_id_pepper_ref.strip()),
        "enabled_flag": bool(settings.patron_id_enabled),
        "tenant_id": access.tenant_id,
    }


__all__ = ["router"]
