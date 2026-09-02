"""The patron-identification gate. Its job is to say no.

This module is the whole of Unique Patron ID's write path, and it contains no
matching, no hashing of anything real and no way to produce a token. It exists
so that the refusal is a single, testable, named object rather than an absence
somebody later fills in without noticing what they are turning on.

### Why this is a gate and not a feature flag

Vision OS declares the two ports re-identification would need and leaves both
unbound on purpose. ``EmbeddingPort`` (P10) is classified **C2 · Biometric** and
its own docstring reads *"declared, unbound, and unimplemented in this flow,
deliberately"*; ``IdentityResolverPort`` (P11) is *"Phase 2 and unimplemented"*;
and 07_STATE §8.2 states the platform *"holds no persistent biometric identity,
which is a deliberate privacy posture, not a limitation."*

The architecture therefore already encodes the legal position: turning this on
is not a matter of writing an adapter, it is a matter of contradicting a stated
posture, and that needs a document rather than a commit.

### Three inputs, not one boolean

``patron_id_enabled`` alone changes nothing. The gate also requires a
``legal_gate_ref`` — the DPIA or DPO sign-off, recorded on every row so no token
exists without naming its authority — and a ``pepper_ref``, without which tokens
would be portable between sites and the module would be a cross-site tracking
gallery rather than a returning-visitor count.

A caller that satisfies all three still gets a refusal from :func:`require_writable`,
because there is no biometric source bound to produce a digest from. That is the
correct state of this module today, and the refusal names it as its own item.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.configuration.settings import Settings
from app.errors import PatronIdentificationBlockedError


@dataclass(frozen=True, slots=True)
class GateStatus:
    """Whether the module may accept a write, and precisely what is missing."""

    available: bool
    reason: str
    #: Machine-readable names of the unmet requirements, in the order a
    #: deployment would satisfy them. Rendered by the frontend as a checklist.
    missing: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {"available": self.available, "reason": self.reason, "missing": list(self.missing)}


#: What the deployment must supply before the *configuration* gate opens. Even
#: with all three the module cannot write, because no biometric source is bound
#: — which is the fourth item, and the one that is not a setting.
REQUIREMENTS: tuple[tuple[str, str], ...] = (
    (
        "legal_review",
        "A completed Data Protection Impact Assessment and a named DPO "
        "sign-off, recorded as PATRON_ID_LEGAL_GATE_REF. Every token written "
        "carries this reference, so no row can exist without naming the "
        "authority that permitted it.",
    ),
    (
        "consent_mechanism",
        "A working way for a patron to give and withdraw consent, and a "
        "reference this backend can store per token. Consent that cannot be "
        "withdrawn is not consent, and a token whose consent cannot be named "
        "has no lawful basis.",
    ),
    (
        "site_pepper",
        "A site-scoped pepper, supplied as PATRON_ID_PEPPER_REF — a reference "
        "the SecretProvider resolves, never the value. Without it tokens are "
        "portable between sites and the module becomes a cross-site tracking "
        "gallery rather than a returning-visitor count.",
    ),
    (
        "deliberate_enablement",
        "PATRON_ID_ENABLED set true by a deployment that has done the three "
        "above. It is last on purpose: it is the switch, not the decision.",
    ),
)


def gate_status(settings: Settings) -> GateStatus:
    """What is standing between this deployment and a first patron token.

    Read-only and side-effect free, so a status route can call it on every
    request without the act of asking changing anything.
    """
    missing: list[str] = []
    if not settings.patron_id_legal_gate_ref.strip():
        missing.append("legal_review")
    # There is no consent-mechanism setting, because there is no consent
    # mechanism. Naming it as permanently missing is more honest than inventing
    # a flag a deployment could set without building anything.
    missing.append("consent_mechanism")
    if not settings.patron_id_pepper_ref.strip():
        missing.append("site_pepper")
    if not settings.patron_id_enabled:
        missing.append("deliberate_enablement")

    # Even a fully configured deployment is refused: nothing produces a
    # biometric digest to hash. Stated as its own requirement so the checklist
    # never reads as "one setting away".
    missing.append("biometric_source")

    order = {name: index for index, (name, _) in enumerate(REQUIREMENTS)}
    missing.sort(key=lambda name: order.get(name, len(order)))

    return GateStatus(
        available=False,
        reason=(
            "Unique Patron ID is blocked. It re-identifies people across "
            "visits, which the perception platform deliberately does not do — "
            "its biometric port is declared and left unbound — and it cannot "
            "be switched on by configuration alone."
        ),
        missing=tuple(missing),
    )


def require_writable(settings: Settings) -> None:
    """Raise unless a patron token may be written. **Always raises today.**

    Every future write path must call this first. It is deliberately not a
    boolean a caller can forget to check: the only shape is "call it and be
    refused", and a caller that skips it is visible in review as a missing line
    rather than as an inverted condition.
    """
    status = gate_status(settings)
    if status.available:  # pragma: no cover - unreachable while the gate is shut
        return
    raise PatronIdentificationBlockedError(
        status.reason, details={"missing": list(status.missing)}
    )


__all__ = [
    "REQUIREMENTS",
    "GateStatus",
    "PatronIdentificationBlockedError",
    "gate_status",
    "require_writable",
]
