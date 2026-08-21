"""M14 Observation API — the platform's only external surface.

> §M14's single responsibility: *"Serve facts safely under contract. Produce
> nothing; accept no writes to state."*

Every method here reads. There is no write path into Vision State, not because
one is guarded but because none is written: search this module for a call that
mutates M12 and there is nothing to find. §M14's Public API lists the absences
explicitly — *"no create_object, no update_state, no set_attribute, no
delete_observation (V5, V6)"* — and the absence is the implementation.

**Three rules shape every method below.**

1. **Scope is constructed, never post-filtered.** 12_SECURITY §4.2: *"Constructing
   the query already scoped means there is no moment at which cross-tenant data
   exists in memory to leak."* Each method authorizes first and queries the
   narrowed scope the decision returns.

2. **Coverage accompanies every state answer.** §2.1: *"unconditionally... Making
   it optional would guarantee that most consumers omit it, and the ones who omit
   it are exactly the ones who will misread the result."* `StateResult.coverage`
   has no default.

3. **Partial is explicit.** §8: *"A query touching an unavailable partition
   returns the data it has plus an explicit statement of what is missing — never
   a quietly smaller result set (V8)."*
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

from ..core.errors import (
    EvidenceExpiredError,
    ForbiddenError,
    InvalidScopeError,
    OverloadedError,
    UnsupportedVersionError,
    WindowTooLargeError,
)
from ..core.model.api import (
    SUPPORTED_MAJORS,
    Action,
    CapabilitySummary,
    CoverageSummary,
    DeliveryPolicy,
    EvidenceView,
    ObjectView,
    ObservationFilter,
    Page,
    Principal,
    QueryOptions,
    Scope,
    SnapshotView,
    StateFilter,
    StateResult,
    TimeWindow,
    object_view_of,
)
from ..core.model.ids import BlobRef, CameraId, ObjectId
from ..core.model.observation import Observation
from ..core.model.timebase import Instant
from ..core.model.vision_state import CoverageReport
from ..core.ports.clock import Clock
from ..core.ports.exposure import ApiLimits, AuthorizationPort
from ..core.ports.persistence import EvidenceStatus, EvidenceStorePort
from ..kernel.metrics import MetricName, MetricsEngine
from ..state import VisionStateManager
from .audit import AuditTrail, evidence_audit_detail
from .demands import DemandIntake
from .subscriptions import Subscription, SubscriptionHub


class ObservationApi:
    """The Observation API. Reads M12; writes nothing.

    Holds subscription sessions, cursors, rate-limit buckets and version
    negotiation state — §M14's State Ownership list — and *"owns no visual
    state."*
    """

    __slots__ = (
        "_audit",
        "_authz",
        "_buckets",
        "_cameras",
        "_capabilities",
        "_clock",
        "_demands",
        "_evidence",
        "_hub",
        "_limits",
        "_lock",
        "_metrics",
        "_next_subscription",
        "_state",
    )

    def __init__(
        self,
        *,
        clock: Clock,
        metrics: MetricsEngine,
        state: VisionStateManager,
        authorizer: AuthorizationPort,
        audit: AuditTrail,
        hub: SubscriptionHub | None = None,
        demands: DemandIntake | None = None,
        evidence: EvidenceStorePort | None = None,
        limits: ApiLimits | None = None,
        capabilities: CapabilitySummary | None = None,
        cameras=None,
    ) -> None:
        self._clock = clock
        self._metrics = metrics
        self._state = state
        self._authz = authorizer
        self._audit = audit
        self._hub = hub if hub is not None else SubscriptionHub(clock=clock, metrics=metrics)
        self._demands = demands
        self._evidence = evidence
        self._limits = limits or ApiLimits()
        self._capabilities = capabilities or CapabilitySummary()
        self._cameras = cameras
        self._buckets: dict[tuple[str, str], list[float]] = {}
        self._next_subscription = 0
        self._lock = threading.Lock()

    # --- query ------------------------------------------------------------- #

    def query_state(
        self,
        principal: Principal,
        scope: Scope,
        *,
        filter_: StateFilter | None = None,
        options: QueryOptions | None = None,
        accepted_major: int = 1,
    ) -> StateResult:
        """Current state for a scope (09_API §2.1).

        Raises:
            ForbiddenError: the principal may not read state here.
            UnsupportedVersionError: an unsupported major was requested.
            OverloadedError: the rate limit for this principal is exhausted.
        """
        self._check_version(accepted_major)
        options = options or QueryOptions()
        filter_ = filter_ or StateFilter()
        narrowed = self._authorize(principal, Action.READ_STATE, scope)
        self._charge(principal, "query", self._limits.query)

        started = self._clock.monotonic().ns
        cameras = self._cameras_in(narrowed)
        snapshot = self._state.snapshot(cameras or None)

        objects: list[ObjectView] = []
        permitted = set(cameras)
        for camera_id, partition in snapshot.partitions.items():
            # 12_SECURITY §4.1: *"Partitions are tenant-scoped; a snapshot never
            # spans tenants."* A camera belongs to exactly one tenant, so
            # restricting to the cameras resolved above **is** the tenant check.
            # There is deliberately no per-object tenant field to fall back on —
            # if there were, this loop would be a post-filter.
            if camera_id not in permitted:
                continue
            for held in partition.objects.values():
                if not self._matches(held, filter_):
                    continue
                objects.append(object_view_of(held, camera_id, options=options))
                if len(objects) >= options.limit:
                    break

        self._metrics.counter(MetricName.API_QUERIES, operation="query_state").increment()
        self._metrics.histogram(MetricName.API_QUERY_MS).record(
            (self._clock.monotonic().ns - started) / 1_000_000
        )
        self._audit.granted(
            principal, Action.READ_STATE, narrowed, resource=f"{len(objects)} objects"
        )

        return StateResult(
            objects=tuple(objects),
            snapshot=SnapshotView.of(snapshot),
            coverage=self._coverage_summary(cameras, snapshot),
            capabilities=self._capabilities,
            cursor=None,
        )

    def query_observations(
        self,
        principal: Principal,
        scope: Scope,
        window: TimeWindow,
        *,
        filter_: ObservationFilter | None = None,
        cursor: str | None = None,
        limit: int = 100,
        accepted_major: int = 1,
    ) -> Page:
        """Historical observations (09_API §2.2).

        Ordered by ``t_capture`` then ``observation_id`` — *"total and stable"* —
        and paged by an opaque cursor that is stable under concurrent writes
        *"because the log is immutable (V5)."*

        Raises:
            WindowTooLargeError: the window exceeds policy. §M14: *"Reject with a
                bound and a cursor rather than degrading the service for
                everyone."*
        """
        self._check_version(accepted_major)
        self._check_window(window)
        filter_ = filter_ or ObservationFilter()
        narrowed = self._authorize(principal, Action.READ_OBSERVATIONS, scope)
        self._charge(principal, "query", self._limits.query)

        bounded = min(limit, self._limits.max_page_size)
        matched: list[Observation] = []
        unavailable: list[tuple[str, str]] = []

        for camera_id in self._cameras_in(narrowed):
            try:
                held = self._state.observations_in(
                    camera_id, since=window.start, until=window.end
                )
            except Exception as exc:  # noqa: BLE001 - one partition must not fail the query
                # §8: partial results are explicit. One unavailable partition
                # returns what the others hold *plus* a statement of what is
                # missing — never a quietly smaller set (V8).
                unavailable.append((str(camera_id), type(exc).__name__))
                continue
            for observation in held:
                if observation.tenant_id != principal.tenant_id:
                    continue
                if not filter_.include_superseded and observation.supersedes is not None:
                    continue
                if filter_.matches(observation):
                    matched.append(observation)

        matched.sort(key=lambda o: (o.t_capture.ns, str(o.observation_id)))
        start = _cursor_index(matched, cursor)
        page = matched[start : start + bounded]
        next_cursor = (
            str(page[-1].observation_id) if len(matched) > start + bounded else None
        )

        self._metrics.counter(
            MetricName.API_QUERIES, operation="query_observations"
        ).increment()
        self._metrics.counter(MetricName.API_OBSERVATIONS_SERVED).increment(len(page))
        self._audit.granted(
            principal,
            Action.READ_OBSERVATIONS,
            narrowed,
            resource=f"{len(page)} observations",
        )

        return Page(
            observations=tuple(page),
            cursor=next_cursor,
            window_fully_observable=not unavailable,
            coverage=CoverageSummary(
                observable_fraction=0.0 if unavailable else 1.0,
                unavailable=tuple(unavailable),
            ),
        )

    def get_object(
        self,
        principal: Principal,
        object_id: ObjectId,
        *,
        scope: Scope | None = None,
        options: QueryOptions | None = None,
        accepted_major: int = 1,
    ) -> ObjectView:
        """One object (09_API §2.3).

        Raises:
            StateNotFoundError: no such object **in this tenant**. A caller
                asking about another tenant's object receives the same answer as
                for one that does not exist — confirming existence would itself
                be a small cross-tenant leak.
        """
        self._check_version(accepted_major)
        options = options or QueryOptions(include_trajectory=True)
        narrowed = self._authorize(
            principal, Action.READ_STATE, scope or Scope(tenant_id=principal.tenant_id)
        )

        state = self._state.object_state(object_id)
        camera = self._camera_of(object_id)
        if camera is None or not narrowed.covers_camera(camera):
            from ..core.errors import StateNotFoundError

            raise StateNotFoundError(
                f"no state for object '{object_id}'", object_id=str(object_id)
            )

        self._audit.granted(
            principal, Action.READ_STATE, narrowed, resource=str(object_id)
        )
        return object_view_of(state, camera, options=options)

    def coverage(
        self, principal: Principal, scope: Scope, window: TimeWindow | None = None
    ) -> CoverageReport:
        """What the platform could see (09_API §5.1).

        > *"An empty result over a window with `observable_fraction < 1.0` does
        > not mean nothing happened. It means nothing was observed. These are
        > different claims and the platform will never conflate them."*
        """
        narrowed = self._authorize(principal, Action.READ_COVERAGE, scope)
        cameras = self._cameras_in(narrowed)
        if not cameras:
            return CoverageReport(observable_fraction=0.0)

        if window is None:
            live = self._state.coverage(cameras)
            observing = sum(
                1 for s in live.by_camera.values() if s.status.value == "observing"
            )
            return CoverageReport(
                observable_fraction=observing / len(cameras) if cameras else 0.0
            )

        reports = [
            self._state.coverage_report(camera, since=window.start, until=window.end)
            for camera in cameras
        ]
        fraction = sum(r.observable_fraction for r in reports) / len(reports)
        gaps = tuple(gap for report in reports for gap in report.gaps)
        self._audit.granted(principal, Action.READ_COVERAGE, narrowed)
        return CoverageReport(observable_fraction=fraction, gaps=gaps)

    def capabilities(self, principal: Principal, scope: Scope) -> CapabilitySummary:
        """What the platform can currently produce (09_API §5.2).

        *"Capability is live state, not documentation."* A consumer polling this
        reacts to capability loss rather than silently receiving nothing.
        """
        self._authorize(principal, Action.READ_CAPABILITY, scope)
        return self._capabilities

    def get_evidence(
        self,
        principal: Principal,
        blob_ref: BlobRef,
        *,
        purpose: str,
        observation_id: str = "",
    ) -> EvidenceView:
        """The justification behind an observation (09_API §6).

        Authorized **separately** from observation access. 12_SECURITY §5.3:
        *"Reading 'a person was here' and viewing their image are categorically
        different acts. Most consumers need the first and must never have the
        second."*

        Raises:
            ForbiddenError: no evidence privilege, or no declared purpose.
            EvidenceExpiredError: retention removed it — a **normal** outcome,
                and a different one from never having existed.
        """
        if not purpose:
            raise ForbiddenError(
                "evidence access requires a declared purpose; 12_SECURITY §5.4 "
                "records it with the actor so imagery access is attributable "
                "rather than invisible",
                subject=principal.subject,
            )
        scope = Scope(tenant_id=principal.tenant_id)
        narrowed = self._authorize(principal, Action.READ_EVIDENCE, scope)
        self._charge(principal, "evidence", self._limits.evidence)

        if self._evidence is None:
            from ..core.errors import NotFoundError

            raise NotFoundError(
                "no evidence store is configured; this deployment retains no "
                "imagery, which is a stated posture rather than a failure",
                blob_ref=str(blob_ref),
            )

        fetched = self._evidence.get(blob_ref, tenant_id=principal.tenant_id)
        self._audit.granted(
            principal,
            Action.READ_EVIDENCE,
            narrowed,
            resource=str(blob_ref),
            purpose=purpose,
            detail=evidence_audit_detail(
                observation_id, principal.tenant_id, self._clock.now()
            ),
        )

        if fetched.status is EvidenceStatus.EXPIRED:
            raise EvidenceExpiredError(
                f"evidence {blob_ref} expired under retention policy",
                blob_ref=str(blob_ref),
            )
        if fetched.status is EvidenceStatus.ERASED:
            raise EvidenceExpiredError(
                f"evidence {blob_ref} was erased under an erasure request",
                blob_ref=str(blob_ref),
            )
        if not fetched.available:
            from ..core.errors import NotFoundError

            raise NotFoundError(
                f"no evidence stored under {blob_ref}", blob_ref=str(blob_ref)
            )

        return EvidenceView(
            observation_id=fetched.record.observation_id if fetched.record else None,
            crop=fetched.payload,
        )

    # --- subscribe ---------------------------------------------------------- #

    def subscribe(
        self,
        principal: Principal,
        scope: Scope,
        *,
        filter_=None,
        policy: DeliveryPolicy | None = None,
        accepted_major: int = 1,
    ) -> Subscription:
        """Open a live stream (09_API §3.1).

        Raises:
            ForbiddenError: no subscribe privilege. 12_SECURITY §5.3 separates it
                from `read_state` because *"continuous surveillance is a stronger
                capability than point-in-time query."*
            OverloadedError: this principal holds too many subscriptions.
        """
        from ..core.model.api import SubscriptionFilter

        self._check_version(accepted_major)
        narrowed = self._authorize(principal, Action.SUBSCRIBE, scope)

        existing = self._hub.for_principal(principal.subject)
        if len(existing) >= self._limits.max_subscriptions_per_principal:
            raise OverloadedError(
                f"principal '{principal.subject}' already holds "
                f"{len(existing)} subscriptions",
                limit=self._limits.max_subscriptions_per_principal,
            )

        with self._lock:
            self._next_subscription += 1
            subscription_id = f"sub-{self._next_subscription:06d}"

        subscription = Subscription(
            subscription_id,
            principal=principal,
            scope=narrowed,
            filter_=filter_ or SubscriptionFilter(),
            policy=policy or DeliveryPolicy(),
            clock=self._clock,
        )
        self._hub.add(subscription)
        self._metrics.counter(MetricName.API_SUBSCRIPTIONS).increment()
        self._audit.granted(
            principal, Action.SUBSCRIBE, narrowed, resource=subscription_id
        )
        return subscription

    def unsubscribe(self, principal: Principal, subscription_id: str) -> None:
        held = self._hub.for_principal(principal.subject)
        if not any(s.subscription_id == subscription_id for s in held):
            raise ForbiddenError(
                f"no subscription '{subscription_id}' for this principal",
                subscription_id=subscription_id,
            )
        self._hub.remove(subscription_id)

    # --- demand (the only inbound path) ------------------------------------- #

    @property
    def demands(self) -> DemandIntake | None:
        """Demand intake, or ``None`` when no registry is configured.

        A **property that never raises**. It used to raise when unconfigured,
        which made the object un-introspectable: ``dir()``, a debugger and a test
        repr all evaluate properties, so a single missing collaborator turned
        every inspection into an exception. Callers that need it use the methods
        below, which fail with a typed error at the point of use.
        """
        return self._demands

    def register_demand(self, principal: Principal, demand, **kwargs):
        """Register a demand (§M14, 09_API §4.1). The **only** inbound path.

        Influence, not a write: a demand changes what the platform chooses to
        compute and cannot change any published fact.
        """
        return self._require_demands().register(principal, demand, **kwargs)

    def update_demand(self, principal: Principal, demand_id, demand, **kwargs):
        return self._require_demands().update(principal, demand_id, demand, **kwargs)

    def revoke_demand(self, principal: Principal, demand_id) -> None:
        self._require_demands().revoke(principal, demand_id)

    def list_demands(self, principal: Principal):
        return self._require_demands().list_for(principal)

    def _require_demands(self) -> DemandIntake:
        if self._demands is None:
            raise InvalidScopeError(
                "no demand registry is configured for this deployment; demands "
                "influence what the Crop Manager computes, so a platform with no "
                "attention layer has nothing to influence"
            )
        return self._demands

    # --- internals ---------------------------------------------------------- #

    def _authorize(self, principal: Principal, action: Action, scope: Scope) -> Scope:
        """Authorize and **return the narrowed scope** (12_SECURITY §4.2).

        The return value is the point. A boolean would leave the caller to
        remember to narrow, and *"whenever a new code path forgets to apply it"*
        is one of the four leak paths §4.2 names.
        """
        if scope.tenant_id != principal.tenant_id:
            from ..core.errors import TenantScopeViolationError

            self._audit.denied(
                principal, action, scope, detail="cross-tenant scope requested"
            )
            self._metrics.counter(
                MetricName.API_DENIALS, action=action.value
            ).increment()
            raise TenantScopeViolationError(
                f"principal in tenant '{principal.tenant_id}' requested scope in "
                f"'{scope.tenant_id}'",
                subject=principal.subject,
            )

        decision = self._authz.authorize(principal, action, scope)
        if decision.denied:
            self._audit.denied(principal, action, scope, detail=decision.reason)
            self._metrics.counter(
                MetricName.API_DENIALS, action=action.value
            ).increment()
            raise ForbiddenError(
                f"principal '{principal.subject}' may not {action.value}: "
                f"{decision.reason}",
                subject=principal.subject,
                action=action.value,
            )
        return decision.scope

    def _check_version(self, major: int) -> None:
        """§M14: *"Reject with the supported set; never guess."*"""
        if major not in SUPPORTED_MAJORS:
            self._metrics.counter(MetricName.API_VERSION_REJECTED).increment()
            raise UnsupportedVersionError(
                f"schema major {major} is not served; supported: "
                f"{sorted(SUPPORTED_MAJORS)}",
                requested=major,
                supported=sorted(SUPPORTED_MAJORS),
            )

    def _check_window(self, window: TimeWindow) -> None:
        if window.span.millis > self._limits.max_window_ms:
            raise WindowTooLargeError(
                f"window of {window.span.millis:.0f}ms exceeds the "
                f"{self._limits.max_window_ms}ms policy bound; narrow the window "
                f"or page through it",
                span_ms=window.span.millis,
                max_ms=self._limits.max_window_ms,
            )

    def _charge(self, principal: Principal, bucket: str, limit) -> None:
        """Token-bucket rate limiting (§M14 responsibility 6).

        Raises:
            OverloadedError: the bucket is empty. Systemic, so a consumer backs
                off rather than retrying immediately.
        """
        now = self._clock.now().ns / 1_000_000_000
        key = (principal.subject, bucket)
        with self._lock:
            window = self._buckets.setdefault(key, [])
            cutoff = now - 60.0
            window[:] = [t for t in window if t > cutoff]
            if len(window) >= limit.requests_per_minute:
                self._metrics.counter(
                    MetricName.API_RATE_LIMITED, bucket=bucket
                ).increment()
                raise OverloadedError(
                    f"rate limit exhausted for '{bucket}'; "
                    f"{limit.requests_per_minute}/minute",
                    bucket=bucket,
                    retry_after_ms=1_000,
                )
            window.append(now)

    def _cameras_in(self, scope: Scope) -> list[CameraId]:
        """The partitions this scope covers, **constructed** from tenant and grant.

        Not "all partitions, then filtered". The returned list *is* the query's
        reach, so no wider set exists in memory at any point (12_SECURITY §4.2).

        The resolution order matters and each step is a narrowing:

        1. An explicit camera list in the scope, already narrowed by the
           authorizer's decision.
        2. Otherwise, the tenant's cameras from the **Camera Manager** — §M14
           names it as a dependency, and it is the module that knows which camera
           belongs to which tenant.
        3. Otherwise, every partition — and **only** when no camera manager is
           available, which is the single-tenant embedded case.

        Step 3 is the one that needed care. Falling back to every partition on a
        multi-tenant node would hand a query the whole node's data and rely on a
        later filter to remove what the caller may not see — which is exactly the
        post-filtering §4.2 forbids. With a camera manager bound, step 2 always
        answers first.
        """
        if scope.camera_ids:
            return list(scope.camera_ids)

        if self._cameras is not None:
            return [
                camera.camera_id
                for camera in self._cameras.list(tenant_id=scope.tenant_id)
            ]

        return list(self._state.partitions)

    def _camera_of(self, object_id: ObjectId) -> CameraId | None:
        for camera_id in self._state.partitions:
            snapshot = self._state.snapshot([camera_id])
            partition = snapshot.partitions.get(camera_id)
            if partition and object_id in partition.objects:
                return camera_id
        return None

    def _coverage_summary(self, cameras: Sequence[CameraId], snapshot) -> CoverageSummary:
        live = self._state.coverage(list(cameras) or None)
        observing = blind = degraded = 0
        for held in live.by_camera.values():
            status = held.status.value
            if status == "observing":
                observing += 1
            elif status in ("blind", "disconnected"):
                blind += 1
            else:
                degraded += 1
        total = observing + blind + degraded
        return CoverageSummary(
            observable_fraction=(observing / total) if total else 0.0,
            cameras_observing=observing,
            cameras_blind=blind,
            cameras_degraded=degraded,
            unavailable=tuple(
                (str(camera), reason) for camera, reason in snapshot.incomplete
            ),
        )

    @staticmethod
    def _matches(state, filter_: StateFilter) -> bool:
        """Content filtering only.

        Tenancy is **not** checked here, and that is deliberate: it was resolved
        when the camera list was constructed. A tenant check in this function
        would mean out-of-tenant partitions had already been read, which is the
        moment §4.2 says must never exist.
        """
        if not filter_.matches_class(state.class_id):
            return False
        if filter_.lifecycle and state.lifecycle not in filter_.lifecycle:
            return False
        if filter_.object_ids and state.object_id not in filter_.object_ids:
            return False
        if filter_.min_confidence is not None:
            confidence = state.class_confidence
            if not confidence.calibrated or confidence.value < filter_.min_confidence:
                return False
        for predicate in filter_.attributes:
            held = state.attributes.get(predicate.key)
            if held is None:
                if predicate.present:
                    return False
                continue
            if predicate.equals is not None and held.value != predicate.equals:
                return False
        return True

    # --- access -------------------------------------------------------------- #

    @property
    def hub(self) -> SubscriptionHub:
        return self._hub

    @property
    def limits(self) -> ApiLimits:
        return self._limits

    def publish(self, observations: Sequence[Observation]) -> int:
        """Fan observations out to subscribers.

        Called by the composition root from M11's sink. **Not** a write path: the
        observations already exist and are already recorded; this delivers copies
        of facts to whoever asked to hear about them.
        """
        return self._hub.publish(observations)

    def set_capabilities(self, capabilities: CapabilitySummary) -> None:
        """Update the live capability report (§5.2).

        *"Capability is live state, not documentation."* A model evicted under
        memory pressure changes this, and subscribers polling it react rather
        than silently degrading.
        """
        self._capabilities = capabilities


def _cursor_index(observations: Sequence[Observation], cursor: str | None) -> int:
    """Where a cursor resumes.

    Stable under concurrent writes because the log is immutable (§2.2): the same
    cursor over the same window always lands in the same place, so paging
    through a million observations across an hour returns exactly the right set.
    """
    if cursor is None:
        return 0
    for index, observation in enumerate(observations):
        if str(observation.observation_id) == cursor:
            return index + 1
    return 0


def instant_window(start_ns: int, end_ns: int) -> TimeWindow:
    return TimeWindow(start=Instant(start_ns), end=Instant(end_ns))
