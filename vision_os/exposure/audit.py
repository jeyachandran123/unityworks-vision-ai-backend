"""The audit trail — who read what (§M14 responsibility 8, 12_SECURITY §8).

Audit exists here and nowhere else because this is the only layer where an actor
exists. 12_SECURITY §5.1: *"External identity exists only at the Observation
API."* A pipeline module could not write a meaningful audit record if it wanted
to — it has no idea who caused its work.

**Evidence access is audited differently from fact access**, and the difference is
the declared purpose. 12_SECURITY §5.4:

> *"Evidence access requires a declared purpose, recorded in the audit trail with
> the actor and the observation. This does not technically prevent misuse —
> nothing at this layer can — but it converts imagery access from an invisible act
> into an attributable one, which is the control that actually changes behaviour
> and the one regulators ask for."*
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

from ..core.model.api import Action, AuditRecord, Principal, Scope
from ..core.model.ids import TenantId
from ..core.model.timebase import Instant
from ..core.ports.clock import Clock
from ..core.ports.exposure import AuditSinkPort
from ..kernel.metrics import MetricName, MetricsEngine


class AuditTrail:
    """Records every access decision and forwards it to a sink.

    **Never raises into the request path.** An audit sink that failed a
    consumer's query would make observability a source of outages, which is the
    opposite of what it is for. Failures are counted and surface through metrics,
    where an operator can see that auditing has stopped working — itself an
    important thing to know.
    """

    __slots__ = ("_clock", "_failures", "_metrics", "_sinks")

    def __init__(
        self,
        *,
        clock: Clock,
        metrics: MetricsEngine,
        sinks: Sequence[AuditSinkPort] = (),
    ) -> None:
        self._clock = clock
        self._metrics = metrics
        self._sinks = tuple(sinks)
        self._failures = 0

    def granted(
        self,
        principal: Principal,
        action: Action,
        scope: Scope,
        *,
        resource: str = "",
        purpose: str = "",
        detail: str = "",
    ) -> None:
        self._write(
            principal, action, scope, granted=True, resource=resource,
            purpose=purpose, detail=detail,
        )

    def denied(
        self,
        principal: Principal,
        action: Action,
        scope: Scope,
        *,
        resource: str = "",
        detail: str = "",
    ) -> None:
        """A denial is audited as carefully as a grant.

        More carefully, in one respect: §M14's failure table requires a
        cross-tenant attempt be *"deny, audit, alarm"*. An unaudited denial leaves
        no trace that someone probed a boundary.
        """
        self._write(
            principal, action, scope, granted=False, resource=resource, detail=detail
        )

    def _write(
        self,
        principal: Principal,
        action: Action,
        scope: Scope,
        *,
        granted: bool,
        resource: str,
        purpose: str = "",
        detail: str = "",
    ) -> None:
        record = AuditRecord(
            at=self._clock.now(),
            principal=principal.subject,
            tenant_id=principal.tenant_id,
            action=action,
            resource=resource,
            granted=granted,
            purpose=purpose,
            detail=detail,
        )
        self._metrics.counter(
            MetricName.API_AUDIT_RECORDS,
            action=action.value,
            granted=str(granted).lower(),
        ).increment()

        for sink in self._sinks:
            try:
                sink.record(record)
            except Exception:  # noqa: BLE001 - auditing must not break serving
                self._failures += 1
                self._metrics.counter(MetricName.API_AUDIT_FAILURES).increment()

    @property
    def failures(self) -> int:
        """Sink failures since start.

        Non-zero means the platform is serving requests it cannot prove it
        served — worth an operator's attention even though nothing is broken from
        a consumer's point of view.
        """
        return self._failures


class CountingAuditSink:
    """An in-memory audit sink for embedded deployments and tests.

    Bounded, because an unbounded audit buffer in a long-running process is a
    memory leak that grows fastest exactly when the platform is busiest. A
    deployment with real audit obligations binds an append-only external sink;
    this one is honest about being a counter with a short tail.
    """

    __slots__ = ("_capacity", "_lock", "_records", "_total")

    def __init__(self, *, capacity: int = 1_000) -> None:
        if capacity < 1:
            raise ValueError("audit capacity must be positive")
        self._capacity = capacity
        self._records: list[AuditRecord] = []
        self._total = 0
        self._lock = threading.Lock()

    def record(self, entry: AuditRecord) -> None:
        with self._lock:
            self._records.append(entry)
            self._total += 1
            if len(self._records) > self._capacity:
                del self._records[: -self._capacity]

    @property
    def records(self) -> tuple[AuditRecord, ...]:
        with self._lock:
            return tuple(self._records)

    @property
    def total(self) -> int:
        """Every record ever written, including those the ring has dropped.

        Reported separately from ``len(records)`` so a reader can tell a quiet
        platform from a full buffer.
        """
        return self._total

    def for_principal(self, subject: str) -> tuple[AuditRecord, ...]:
        return tuple(r for r in self.records if r.principal == subject)

    def denials(self) -> tuple[AuditRecord, ...]:
        return tuple(r for r in self.records if not r.granted)

    def __len__(self) -> int:
        return len(self._records)


class NullAuditSink:
    """Discards audit records.

    Exists so that *"no audit configured"* is a stated choice rather than an
    empty tuple nobody noticed. A deployment binding this has decided it does not
    need an audit trail; a deployment binding nothing may simply have forgotten.
    """

    __slots__ = ()

    def record(self, entry: AuditRecord) -> None:
        return None


def evidence_audit_detail(
    observation_id: str, tenant_id: TenantId, at: Instant
) -> str:
    """The detail line for an evidence access.

    Names the observation rather than the blob: an auditor asks *"who looked at
    the picture behind this claim"*, and a content hash does not answer that.
    """
    return f"evidence for observation {observation_id} in tenant {tenant_id} at {at.ns}"
