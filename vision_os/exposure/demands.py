"""Demand intake — the only inbound path (09_API §4).

> §1.1: *"Consumers must be able to influence what the platform spends money
> computing without telling it why."*

A demand is **not a state write**. It changes what work the platform chooses to
do; it cannot change an observation, an object, a track, or a projection. That is
why it coexists with §1.1's *"~~Mutate~~ — does not exist"*.

**This module never calls the Crop Manager.** `01_LAYERED` §3.2 breaks the only
possible cycle in the graph by making the path declarative:

> *"the API writes a demand record; the Crop Manager reads demand state at trigger
> time. **No call ever returns through the pipeline it entered.**"*

So `DemandIntake` authenticates, authorizes, validates and records. What M8 does
with the record afterwards is M8's business, and M14 never learns it.

**Durability.** §M14: *"The demand registry is durable — demands must survive
restart, or every consumer would have to re-register after every deployment and
attribute coverage would silently lapse in the interval."* §M13's ConfigStore row
holds *"Config, calibration, taxonomy, **registry**"*, which is where that
durability belongs.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..core.errors import DemandRejectedError, ForbiddenError
from ..core.model.api import Action, DemandView, Principal, Scope
from ..core.model.demand import (
    Demand,
    DemandAcknowledgement,
    DemandScope,
    DemandState,
    SubjectFilter,
)
from ..core.model.ids import (
    AttributeKey,
    CameraId,
    ClassId,
    DemandId,
    SiteId,
    TenantId,
)
from ..core.model.timebase import Duration, Instant
from ..core.ports.clock import Clock
from ..core.ports.exposure import AuthorizationPort
from ..kernel.metrics import MetricName, MetricsEngine
from ..perception.cropping.demands import DemandRegistry
from .audit import AuditTrail


@dataclass(frozen=True, slots=True)
class DemandRecord:
    """The durable form of a demand.

    Only what a restart needs to reconstruct the registry. Deliberately not the
    full `DemandState`: lifecycle status, throttling and last-served times are
    *derived* from what the platform can currently sustain, and restoring a
    stale "throttled" would tell a consumer about budget pressure that ended
    while the process was down.
    """

    demand_id: str
    subscriber: str
    tenant_id: str
    site_ids: tuple[str, ...]
    camera_ids: tuple[str, ...]
    required_attributes: tuple[str, ...]
    class_ids: tuple[str, ...]
    freshness_ms: int
    registered_at_ns: int
    expires_at_ns: int | None = None

    def to_json(self) -> dict:
        return {
            "demand_id": self.demand_id,
            "subscriber": self.subscriber,
            "tenant_id": self.tenant_id,
            "site_ids": list(self.site_ids),
            "camera_ids": list(self.camera_ids),
            "required_attributes": list(self.required_attributes),
            "class_ids": list(self.class_ids),
            "freshness_ms": self.freshness_ms,
            "registered_at_ns": self.registered_at_ns,
            "expires_at_ns": self.expires_at_ns,
        }

    @classmethod
    def from_json(cls, record: dict) -> DemandRecord:
        return cls(
            demand_id=record["demand_id"],
            subscriber=record["subscriber"],
            tenant_id=record["tenant_id"],
            site_ids=tuple(record.get("site_ids", ())),
            camera_ids=tuple(record.get("camera_ids", ())),
            required_attributes=tuple(record.get("required_attributes", ())),
            class_ids=tuple(record.get("class_ids", ())),
            freshness_ms=record["freshness_ms"],
            registered_at_ns=record["registered_at_ns"],
            expires_at_ns=record.get("expires_at_ns"),
        )


class DemandStore:
    """Durable storage for demand records.

    A file, because §M13's ConfigStore contract describes *"read-heavy,
    versioned, audited"* storage and a demand registry on a single node is a
    small versioned document. A clustered deployment binds a config service
    instead; nothing above this class knows which.
    """

    __slots__ = ("_lock", "_path")

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._lock = threading.Lock()

    def save(self, records: Sequence[DemandRecord]) -> None:
        if self._path is None:
            return
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps([r.to_json() for r in records], indent=2)
            temporary = self._path.with_suffix(".tmp")
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(self._path)

    def load(self) -> tuple[DemandRecord, ...]:
        """Read what survived the restart.

        A corrupt file returns nothing rather than raising: a platform that
        refused to boot because a demand file was truncated would turn a
        recoverable degradation — consumers re-register — into an outage. The
        loss is visible because the registry comes up empty, which §M14's
        lifecycle notifications surface to every affected subscriber.
        """
        if self._path is None or not self._path.exists():
            return ()
        with self._lock:
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                return ()
        out: list[DemandRecord] = []
        for entry in raw if isinstance(raw, list) else ():
            try:
                out.append(DemandRecord.from_json(entry))
            except (KeyError, TypeError):
                continue
        return tuple(out)

    @property
    def durable(self) -> bool:
        return self._path is not None


class DemandIntake:
    """M14's demand contract surface.

    Owns *intake and lifecycle*: authorization, tenant scoping, validation,
    acknowledgement, durability. The registry it writes to is the one M8 reads,
    and this class never reaches toward M8.
    """

    __slots__ = ("_audit", "_authz", "_clock", "_metrics", "_owners", "_registry", "_store")

    def __init__(
        self,
        *,
        clock: Clock,
        metrics: MetricsEngine,
        registry: DemandRegistry,
        authorizer: AuthorizationPort,
        audit: AuditTrail,
        store: DemandStore | None = None,
    ) -> None:
        self._clock = clock
        self._metrics = metrics
        self._registry = registry
        self._authz = authorizer
        self._audit = audit
        self._store = store or DemandStore()

        # Which tenant each demand belongs to. Held **here** rather than on the
        # Demand, because 09_API §4.1's contract declares no tenant field and
        # 12_SECURITY §5.1 puts external identity at this layer and nowhere else.
        # M8 reads demands to decide what to compute; it has no business knowing
        # which customer asked, and adding a tenant to the Demand would push that
        # knowledge down into L3.
        self._owners: dict[DemandId, TenantId] = {}

    # --- the contract ------------------------------------------------------- #

    def register(
        self,
        principal: Principal,
        demand: Demand,
        *,
        sustainable_freshness: Duration | None = None,
    ) -> DemandAcknowledgement:
        """Validate and admit a demand (§4.1).

        Raises:
            ForbiddenError: the principal may not register demands, or the demand
                reaches outside their tenant. 12_SECURITY §5.3: *"`register_demand`
                is privileged... Demands spend money and cause computation; they
                are not a read."*
            DemandRejectedError: no requested attribute is registered. §4.2 makes
                this *"the fourth and outermost ring of Semantic Ceiling
                enforcement"* — a consumer learns at registration rather than
                discovering a permanent absence of data weeks later.
        """
        if demand.subscriber != principal.subject:
            # A principal may only register demands in its own name. Without
            # this, `list_demands(subscriber)` would show a consumer demands it
            # never made, and `_require_owner` below would guard nothing —
            # anyone could revoke anyone's demand by claiming their subscriber id.
            raise ForbiddenError(
                f"principal '{principal.subject}' may not register a demand for "
                f"subscriber '{demand.subscriber}'",
                subject=principal.subject,
                subscriber=str(demand.subscriber),
            )

        scope = _scope_of(principal, demand)
        decision = self._authz.authorize(principal, Action.REGISTER_DEMAND, scope)
        if decision.denied:
            self._audit.denied(
                principal, Action.REGISTER_DEMAND, scope, detail=decision.reason
            )
            self._metrics.counter(
                MetricName.API_DENIALS, action=Action.REGISTER_DEMAND.value
            ).increment()
            raise ForbiddenError(
                f"principal '{principal.subject}' may not register demands: "
                f"{decision.reason}",
                subject=principal.subject,
            )

        try:
            acknowledgement = self._registry.register(
                demand,
                now=self._clock.now(),
                sustainable_freshness=sustainable_freshness,
            )
        except DemandRejectedError:
            self._metrics.counter(MetricName.API_DEMANDS_REJECTED).increment()
            self._audit.granted(
                principal,
                Action.REGISTER_DEMAND,
                scope,
                resource=str(demand.demand_id),
                detail="rejected: no requested attribute is registered",
            )
            raise

        self._owners[demand.demand_id] = principal.tenant_id
        self._metrics.counter(MetricName.API_DEMANDS_REGISTERED).increment()
        self._audit.granted(
            principal,
            Action.REGISTER_DEMAND,
            scope,
            resource=str(demand.demand_id),
            detail=f"status={acknowledgement.status.value}",
        )
        self._persist()
        return acknowledgement

    def update(
        self,
        principal: Principal,
        demand_id: DemandId,
        demand: Demand,
        *,
        sustainable_freshness: Duration | None = None,
    ) -> DemandAcknowledgement:
        """Replace a demand (§M14 `update_demand`).

        Implemented as revoke-then-register rather than as an in-place edit: a
        demand's acknowledgement reports what the platform can *currently*
        sustain, and mutating one in place would leave a stale
        ``effective_freshness`` attached to new terms.
        """
        self._require_owner(principal, demand_id)
        self._registry.revoke(demand_id)
        return self.register(principal, demand, sustainable_freshness=sustainable_freshness)

    def revoke(self, principal: Principal, demand_id: DemandId) -> None:
        self._require_owner(principal, demand_id)
        self._registry.revoke(demand_id)
        self._owners.pop(demand_id, None)
        self._audit.granted(
            principal,
            Action.REGISTER_DEMAND,
            _principal_scope(principal),
            resource=str(demand_id),
            detail="revoked",
        )
        self._persist()

    def list_for(self, principal: Principal) -> tuple[DemandView, ...]:
        """A subscriber's own demands (§M14 `list_demands`).

        Scoped to the principal at *construction* of the result, not filtered
        afterwards — the same rule 12_SECURITY §4.2 applies to queries.
        """
        return tuple(
            _view_of(state)
            for state in self._registry.all()
            if self._owners.get(state.demand.demand_id) == principal.tenant_id
            and state.demand.subscriber == principal.subject
        )

    # --- durability --------------------------------------------------------- #

    def restore(self) -> int:
        """Re-register what survived a restart. Returns how many came back.

        Demands whose attributes are no longer registered — a taxonomy changed
        while the platform was down — are dropped rather than restored broken,
        and the consumer learns through the lifecycle notification §4.4 requires.
        """
        restored = 0
        for record in self._store.load():
            demand = _demand_of(record)
            if demand is None:
                continue
            try:
                self._registry.register(demand, now=self._clock.now())
            except DemandRejectedError:
                continue
            self._owners[demand.demand_id] = TenantId(record.tenant_id)
            restored += 1
        return restored

    def _persist(self) -> None:
        if not self._store.durable:
            return
        self._store.save([
            _record_of(state, self._owners.get(state.demand.demand_id, TenantId("")))
            for state in self._registry.all()
        ])

    def _require_owner(self, principal: Principal, demand_id: DemandId) -> None:
        state = self._registry.get(demand_id)
        if state is None:
            raise ForbiddenError(
                f"no demand '{demand_id}' for this principal",
                demand_id=str(demand_id),
            )
        if (
            state.demand.subscriber != principal.subject
            or self._owners.get(demand_id) != principal.tenant_id
        ):
            # Deliberately the same message as "not found": telling a caller a
            # demand exists but belongs to someone else confirms its existence,
            # which is itself a small cross-tenant leak.
            raise ForbiddenError(
                f"no demand '{demand_id}' for this principal",
                demand_id=str(demand_id),
            )

    @property
    def registry(self) -> DemandRegistry:
        return self._registry


def _scope_of(principal: Principal, demand: Demand) -> Scope:
    return Scope(
        tenant_id=principal.tenant_id,
        site_ids=tuple(SiteId(s) for s in demand.scope.site_ids),
        camera_ids=tuple(CameraId(c) for c in demand.scope.camera_ids),
    )


def _principal_scope(principal: Principal) -> Scope:
    return Scope(tenant_id=principal.tenant_id)


def _view_of(state: DemandState) -> DemandView:
    return DemandView(
        demand_id=state.demand.demand_id,
        subscriber=state.demand.subscriber,
        status=state.status.value,
        required_attributes=tuple(state.demand.required_attributes),
        effective_freshness=state.effective_freshness,
        unsatisfiable=tuple(
            (key, reason.value) for key, reason in state.acknowledgement.unsatisfiable
        ),
    )


def _record_of(state: DemandState, tenant_id: TenantId) -> DemandRecord:
    demand = state.demand
    return DemandRecord(
        demand_id=str(demand.demand_id),
        subscriber=str(demand.subscriber),
        tenant_id=str(tenant_id),
        site_ids=tuple(str(s) for s in demand.scope.site_ids),
        camera_ids=tuple(str(c) for c in demand.scope.camera_ids),
        required_attributes=tuple(str(a) for a in demand.required_attributes),
        class_ids=tuple(str(c) for c in demand.subject_filter.class_ids),
        freshness_ms=int(demand.freshness.millis),
        registered_at_ns=state.activated_at.ns,
        expires_at_ns=demand.expires_at.ns if demand.expires_at else None,
    )


def _demand_of(record: DemandRecord) -> Demand | None:
    """Rebuild a demand from its durable record.

    Returns ``None`` for a record this build cannot reconstruct — a field added
    in a later version, say. Skipping beats raising: one unreadable demand costs
    a consumer a re-registration, while a raise costs every consumer their
    demands.
    """
    try:
        return Demand(
            demand_id=DemandId(record.demand_id),
            subscriber=record.subscriber,
            scope=DemandScope(
                site_ids=tuple(SiteId(s) for s in record.site_ids),
                camera_ids=tuple(CameraId(c) for c in record.camera_ids),
            ),
            subject_filter=SubjectFilter(
                class_ids=tuple(ClassId(c) for c in record.class_ids)
            ),
            required_attributes=tuple(
                AttributeKey(a) for a in record.required_attributes
            ),
            freshness=Duration.from_millis(record.freshness_ms),
            expires_at=(
                Instant(record.expires_at_ns)
                if record.expires_at_ns is not None
                else None
            ),
        )
    except (TypeError, ValueError):
        # Skipping beats raising: one unreadable demand costs its consumer a
        # re-registration, while a raise costs every consumer theirs. The broad
        # catch is deliberate but it is also a hiding place — a missing required
        # field here once made every restore silently return nothing, and only a
        # test that asserted the *count* found it.
        return None
