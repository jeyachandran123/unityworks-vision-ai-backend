"""The POS/ERP port, and the only adapter that exists: the one that refuses.

### Why this port lives in `app/` and not in `vision_os/`

The architecture's rule for a new capability is *"a sibling adapter behind the
same port; no platform module changes"* — and it is worth being precise about
which port. Vision OS's ports describe **perception**: acquiring frames,
detecting, tracking, understanding, publishing observations. A till is not
perception. It produces no frame, has no capture instant, cannot be scoped by
camera, and reasoning about it in the platform's vocabulary would mean inventing
a `FrameRef` for a receipt.

So the seam is here, at the application layer, where the one-way dependency
`app → compliance → vision_os` puts external systems on the correct side. Adding
Square, Toast, Oracle Micros or an SAP endpoint later is a sibling adapter behind
:class:`PosGatewayPort`, selected by `PosConnector.vendor`, with no change to
this port and none to any caller.

### The refusing adapter is the product, for now

:class:`NotConfiguredPosGateway` is not a stub and not a mock. It never returns
plausible data; it raises a typed error that names every input still required.
That is the difference between a scaffold and a lie, and it is why a caller
wired against it today behaves correctly the moment a real adapter replaces it:
the failure path is already exercised, and nothing downstream has been written
against fabricated shapes.

### What a real adapter must never do

Store the payload. A POS response carries ticket lines, staff identifiers,
discounts and sometimes partial card data. `PosSyncRun` records a *digest* — see
its docstring — and an adapter that persisted the body would quietly turn a
compliance product into a store of retail and payment records governed by a
retention policy written for camera observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from app.errors import CapabilityNotConfiguredError


@dataclass(frozen=True, slots=True)
class TicketLine:
    """One sold item, in the shape this application needs and no wider.

    Deliberately narrow. A POS ticket has thirty fields and this carries five,
    because the only question the product asks of a till is *"was this dish
    actually sold"* — and a field that is never read is a field that leaks.

    No customer name, no payment instrument, no loyalty identifier: none of them
    is needed to reconcile a dish detection, and each is a category of data this
    application has no lawful basis to hold.
    """

    ticket_ref: str
    line_ref: str
    menu_item_ref: str
    quantity: int
    sold_at: datetime
    table_code: str = ""


@dataclass(frozen=True, slots=True)
class GatewayDescription:
    """What an adapter can do, asked rather than assumed.

    A caller branches on `capabilities` instead of on `vendor`, so support for a
    till that cannot report table numbers is a capability the adapter declines
    to declare rather than a special case in the reconciliation code.
    """

    vendor: str
    display_name: str
    available: bool
    reason: str = ""
    capabilities: tuple[str, ...] = ()
    #: Inputs still required before this adapter can be used at all. Empty for a
    #: working one; the whole point of the type for the one that is not.
    missing: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class PosGatewayPort(Protocol):
    """The application's one way to reach a point-of-sale system.

    ### Semantic contract

    | # | Obligation |
    |---|---|
    | **G1** | ``describe`` never raises and never blocks on the network. A status screen must be able to ask "is this connected" without connecting. |
    | **G2** | An adapter that cannot serve a call raises :class:`CapabilityNotConfiguredError` naming what is missing. It never returns an empty result, because "no tickets" and "not connected" are different answers and only one of them is a zero. |
    | **G3** | Returned lines carry no customer identity, no payment instrument and no loyalty reference — the shape of :class:`TicketLine` is the boundary, and an adapter must not widen it. |
    | **G4** | ``fetch_tickets`` is bounded by an explicit window. There is no "everything" call: an unbounded pull from a busy till is a denial of service against somebody else's system. |
    | **G5** | The adapter persists nothing. Writing the run record is the caller's job, so exactly one place decides what is kept. |
    """

    @property
    def vendor(self) -> str:
        """Which vendor this adapter serves. Matches `PosConnector.vendor`."""
        ...

    def describe(self) -> GatewayDescription:
        """Capability and readiness. Never raises, never touches the network (G1)."""
        ...

    def fetch_tickets(self, *, since: datetime, until: datetime) -> tuple[TicketLine, ...]:
        """Sold lines in a bounded window (G4).

        Raises:
            CapabilityNotConfiguredError: nothing is connected (G2).
        """
        ...

    def push_events(self, events: tuple[dict[str, object], ...]) -> int:
        """Send events to the POS/ERP. Returns how many it accepted.

        Raises:
            CapabilityNotConfiguredError: nothing is connected (G2).
        """
        ...


#: The exact inputs a real adapter needs, in the order a deployment gets them.
#: Rendered verbatim by the API and by the frontend — one list, one place.
POS_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    (
        "vendor_selection",
        "Which POS or ERP each site actually runs, per site. Chains commonly "
        "run more than one, and an adapter chosen for the group is an adapter "
        "wrong for half the estate.",
    ),
    (
        "api_documentation",
        "The vendor's API documentation and, specifically, its ticket-line "
        "schema and pagination model. Reconciliation is a join on the vendor's "
        "own item identifiers; without their shape there is nothing to join on.",
    ),
    (
        "credentials",
        "Sandbox and production credentials, supplied as a SecretProvider "
        "reference for PosConnector.credential_ref — never as a value in the "
        "database, exactly as camera credentials are handled.",
    ),
    (
        "menu_mapping",
        "A mapping from detector dish classes to the vendor's menu item "
        "identifiers. This does not exist and cannot be inferred: 'chicken "
        "rice' in a model's vocabulary is not the same string as a menu code.",
    ),
    (
        "rate_and_egress_agreement",
        "The vendor's rate limits, and written agreement that this system may "
        "read sales data at all. A till belongs to the operator, and pulling "
        "from it is a commercial decision before it is a technical one.",
    ),
)


class NotConfiguredPosGateway:
    """``pos.not_configured`` — the honest adapter. Bound by default.

    Named as an adapter id rather than as an absence, following the platform's
    own convention (``log.memory`` says what it is), so an operator reading a
    status page sees *which* adapter is bound rather than a blank.
    """

    __slots__ = ("_reason", "_vendor")

    def __init__(self, vendor: str = "") -> None:
        self._vendor = vendor
        self._reason = (
            "No point-of-sale adapter is bound. Meal detection can record what "
            "a camera saw, but it cannot be reconciled against what was sold "
            "until a vendor, its API documentation and its credentials exist."
        )

    @property
    def vendor(self) -> str:
        return self._vendor

    def describe(self) -> GatewayDescription:
        return GatewayDescription(
            vendor=self._vendor,
            display_name="No POS adapter bound",
            available=False,
            reason=self._reason,
            capabilities=(),
            missing=tuple(name for name, _ in POS_REQUIREMENTS),
        )

    def fetch_tickets(self, *, since: datetime, until: datetime) -> tuple[TicketLine, ...]:
        # Not an empty tuple. G2: a caller that received `()` would record a
        # successful sync that found no sales, and every dish detected in the
        # window would reconcile as UNMATCHED — manufacturing a discrepancy
        # report out of the fact that nothing was connected.
        raise CapabilityNotConfiguredError(
            self._reason, details={"missing": [name for name, _ in POS_REQUIREMENTS]}
        )

    def push_events(self, events: tuple[dict[str, object], ...]) -> int:
        raise CapabilityNotConfiguredError(
            self._reason, details={"missing": [name for name, _ in POS_REQUIREMENTS]}
        )


def gateway_for(vendor: str = "") -> PosGatewayPort:
    """Select the adapter for a vendor. Today there is exactly one.

    The selection point exists now so that adding a vendor is a line here plus a
    new module, rather than a caller learning to construct adapters — which is
    how a port turns back into a set of if-statements.
    """
    return NotConfiguredPosGateway(vendor)


__all__ = [
    "POS_REQUIREMENTS",
    "GatewayDescription",
    "NotConfiguredPosGateway",
    "PosGatewayPort",
    "TicketLine",
    "gateway_for",
]
