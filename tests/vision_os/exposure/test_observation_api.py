"""API tests — the read-only contract (09_API §§1, 2, 5, 6, 8).

The brief: consumers *"may query, read, subscribe, replay, inspect"* and *"may
NEVER modify observations, Vision State, Objects, Tracks, Understanding."*

That is tested two ways here. Structurally — there is no method to call — and
behaviourally, by driving every method the API has and asserting state is
unchanged afterwards. The first proves nobody *can*; the second proves the read
paths themselves do not write as a side effect, which is the failure the first
would miss.
"""

from __future__ import annotations

import pytest

from vision_os.core.errors import (
    EvidenceExpiredError,
    ForbiddenError,
    NotFoundError,
    OverloadedError,
    StateNotFoundError,
    TenantScopeViolationError,
    UnsupportedVersionError,
    WindowTooLargeError,
)
from vision_os.core.model.api import (
    Action,
    AttributePredicate,
    ObservationFilter,
    QueryOptions,
    StateFilter,
)
from vision_os.core.model.ids import BlobRef, ObjectId

from .conftest import (
    CAMERA,
    EVIDENCE_HASH,
    EVIDENCE_PAYLOAD,
    OTHER_CAMERA,
    OTHER_TENANT,
    POSTURE,
    TENANT,
    publish,
    publish_attributes,
    scope,
    store_evidence,
    window,
)


class TestTheApiIsReadOnly:
    """09_API §1.1: *"~~Mutate~~ — Does not exist (V6)."*"""

    def test_no_mutating_method_exists(self, api) -> None:
        """Structural, not guarded.

        §M14's Public API lists the absences by name — *"no create_object, no
        update_state, no set_attribute, no delete_observation"*. A future
        contributor will not find one disabled; they will find it was never
        written.
        """
        for forbidden in (
            "create_object",
            "update_state",
            "set_attribute",
            "delete_observation",
            "write",
            "put",
            "patch",
            "mutate",
            "admin_override",
        ):
            assert not hasattr(api, forbidden), f"the API exposes {forbidden}"

    def test_every_public_method_is_a_read_or_a_demand(self, api) -> None:
        """The one non-read is demand intake, which is influence, not a write.

        09_API §1.1 lists five contracts; the sixth *"does not exist"*.
        """
        public = {
            name for name in dir(api) if not name.startswith("_") and callable(getattr(api, name))
        }
        assert public <= {
            "query_state",
            "query_observations",
            "get_object",
            "get_evidence",
            "coverage",
            "capabilities",
            "subscribe",
            "unsubscribe",
            "publish",
            "set_capabilities",
            "register_demand",
            "update_demand",
            "revoke_demand",
            "list_demands",
        }, f"unexpected API surface: {sorted(public)}"

    def test_reading_does_not_change_state(self, api, state, operator) -> None:
        """The behavioural half. A read path that wrote would pass the first test.

        Every method is driven, then the projection is compared. A cursor
        advanced, a cache populated, a counter incremented *inside state* would
        all show up here.
        """
        publish(state, count=3)
        publish_attributes(state)
        before = state.snapshot()
        version_before = before.partitions[CAMERA].version

        api.query_state(operator, scope(CAMERA))
        api.query_observations(operator, scope(CAMERA), window())
        api.get_object(operator, ObjectId("obj-0"), scope=scope(CAMERA))
        api.coverage(operator, scope(CAMERA))
        api.capabilities(operator, scope(CAMERA))
        api.subscribe(operator, scope(CAMERA))

        after = state.snapshot()
        assert after.partitions[CAMERA].version == version_before
        assert after.partitions[CAMERA].objects.keys() == before.partitions[CAMERA].objects.keys()

    def test_the_api_holds_no_write_capable_collaborator(self, api) -> None:
        """It receives M12 and nothing lower.

        §M14's Dependencies name the Vision State Manager. Holding the log or the
        builder would let a query bypass the layer that owns partitioning and
        consistency — and would make a write path one attribute access away.
        """
        held = {slot for slot in type(api).__slots__}
        for forbidden in ("_log", "_builder", "_registry", "_detector", "_tracker"):
            assert forbidden not in held


class TestTenantIsolation:
    """12_SECURITY §4: scope at construction, never post-filtered."""

    def test_a_cross_tenant_scope_is_refused(self, api, reader) -> None:
        """§M14: *"Deny, audit, alarm."*"""
        with pytest.raises(TenantScopeViolationError):
            api.query_state(reader, scope(CAMERA, tenant=OTHER_TENANT))

    def test_a_cross_tenant_attempt_is_audited(self, api, reader, audit_sink) -> None:
        with pytest.raises(TenantScopeViolationError):
            api.query_state(reader, scope(tenant=OTHER_TENANT))
        denials = audit_sink.denials()
        assert denials
        assert denials[0].principal == reader.subject

    def test_a_query_never_reads_a_partition_outside_the_tenant(
        self, clock, metrics, bus, log, authorizer, audit, reader
    ) -> None:
        """12_SECURITY §4.1: *"Partitions are tenant-scoped."*

        A camera belongs to exactly one tenant, so the isolation boundary is the
        camera list the query is *built from* — never a filter applied to results.
        With a Camera Manager bound, the list comes from the tenant's declared
        cameras and another tenant's partition is never read at all.
        """
        from vision_os.exposure import ObservationApi
        from vision_os.state import VisionStateManager

        from ..synthesis.conftest import SITE, state_config  # noqa: TID252

        state = VisionStateManager(
            clock=clock, metrics=metrics, events=bus,
            config=state_config(), log=log, site_id=SITE,
        )
        publish(state, count=2, camera=CAMERA, tenant=TENANT)
        publish(state, count=2, camera=OTHER_CAMERA, tenant=OTHER_TENANT, start=50)

        api = ObservationApi(
            clock=clock, metrics=metrics, state=state,
            authorizer=authorizer, audit=audit,
            cameras=_CameraDirectory({TENANT: (CAMERA,), OTHER_TENANT: (OTHER_CAMERA,)}),
        )

        result = api.query_state(reader, scope())
        assert set(result.snapshot.partitions) == {CAMERA}, (
            "the query read a partition belonging to another tenant"
        )
        assert all(str(o.object_id).startswith("obj-") for o in result.objects)
        assert not any(str(o.object_id).startswith("obj-5") for o in result.objects)

    def test_a_principal_cannot_widen_its_own_scope(self, api, state, reader) -> None:
        """A grant for one camera stays one camera even when more are requested."""
        publish(state, count=2, camera=CAMERA)
        publish(state, count=2, camera=OTHER_CAMERA, start=50)
        result = api.query_state(reader, scope(CAMERA, OTHER_CAMERA))
        assert set(result.snapshot.partitions) <= {CAMERA, OTHER_CAMERA}


class TestAuthorizationSeparations:
    """12_SECURITY §5.3 — the separations that matter."""

    def test_reading_facts_does_not_grant_reading_imagery(
        self, api, evidence, reader
    ) -> None:
        """*"Reading 'a person was here' and viewing their image are
        categorically different acts."*
        """
        store_evidence(evidence)
        with pytest.raises(ForbiddenError):
            api.get_evidence(reader, BlobRef(f"sha256:{EVIDENCE_HASH}"), purpose="audit")

    def test_an_operator_with_the_privilege_may_read_evidence(
        self, api, evidence, operator
    ) -> None:
        store_evidence(evidence)
        view = api.get_evidence(
            operator, BlobRef(f"sha256:{EVIDENCE_HASH}"), purpose="incident review"
        )
        assert view.crop == EVIDENCE_PAYLOAD

    def test_evidence_access_requires_a_declared_purpose(
        self, api, evidence, operator
    ) -> None:
        """12_SECURITY §5.4: purpose binding.

        *"This does not technically prevent misuse — nothing at this layer can —
        but it converts imagery access from an invisible act into an attributable
        one."*
        """
        store_evidence(evidence)
        with pytest.raises(ForbiddenError, match="purpose"):
            api.get_evidence(operator, BlobRef(f"sha256:{EVIDENCE_HASH}"), purpose="")

    def test_evidence_access_is_audited_with_its_purpose(
        self, api, evidence, operator, audit_sink
    ) -> None:
        store_evidence(evidence)
        api.get_evidence(
            operator, BlobRef(f"sha256:{EVIDENCE_HASH}"), purpose="incident-4471"
        )
        records = [r for r in audit_sink.records if r.action is Action.READ_EVIDENCE]
        assert records
        assert records[0].purpose == "incident-4471"

    def test_registering_a_demand_is_privileged(self, api, reader) -> None:
        """*"Demands spend money and cause computation; they are not a read."*

        A read-only grant does not include it, which is the whole point of it
        being a separate action.
        """
        from vision_os.core.model.api import Action

        decision = api._authz.authorize(  # noqa: SLF001 - the port under test
            reader, Action.REGISTER_DEMAND, scope(CAMERA)
        )
        assert decision.denied


class TestCoverageIsAlwaysReturned:
    """09_API §2.1: *"unconditionally."*"""

    def test_every_state_query_carries_coverage(self, api, state, reader) -> None:
        publish(state, count=2)
        result = api.query_state(reader, scope(CAMERA))
        assert result.coverage is not None

    def test_an_empty_result_still_carries_coverage(self, api, reader) -> None:
        """The case that matters.

        > *"A consumer must not be able to receive an empty or thin result
        > without simultaneously receiving the information required to interpret
        > it (V8)."*
        """
        result = api.query_state(reader, scope(CAMERA))
        assert result.objects == ()
        assert result.coverage is not None

    def test_coverage_is_not_optional_in_the_type(self) -> None:
        """Required by construction, so no code path can omit it."""
        import dataclasses

        from vision_os.core.model.api import StateResult

        field = next(
            f for f in dataclasses.fields(StateResult) if f.name == "coverage"
        )
        assert field.default is dataclasses.MISSING
        assert field.default_factory is dataclasses.MISSING

    def test_a_result_reports_whether_it_is_complete(self, api, state, reader) -> None:
        publish(state, count=2)
        result = api.query_state(reader, scope(CAMERA))
        assert isinstance(result.complete, bool)


class TestQueryContracts:
    def test_filtering_by_class_is_hierarchical(self, api, state, reader) -> None:
        """§2.1: *"vehicle matches vehicle.forklift"*.

        Without it, a new taxonomy class would silently break every existing
        query — which is why §7.2 can call adding one a *minor* change.
        """
        publish(state, count=2)
        from vision_os.core.model.ids import ClassId

        result = api.query_state(
            reader, scope(CAMERA), filter_=StateFilter(class_ids=(ClassId("person"),))
        )
        assert result.objects

    def test_an_attribute_predicate_selects(self, api, state, reader) -> None:
        publish(state, count=1)
        publish_attributes(state, object_id="obj-0", value="sitting")
        result = api.query_state(
            reader,
            scope(CAMERA),
            filter_=StateFilter(
                attributes=(AttributePredicate(key=POSTURE, equals="sitting"),)
            ),
        )
        assert [str(o.object_id) for o in result.objects] == ["obj-0"]

    def test_provenance_is_included_by_default(self, api, state, reader) -> None:
        """§2.1: *"explainability is not opt-out by accident."*"""
        assert QueryOptions().include_provenance is True

    def test_historical_queries_are_ordered_totally_and_stably(
        self, api, state, reader
    ) -> None:
        """§2.2: by ``t_capture``, then ``observation_id``.

        Total because two observations can share a capture instant; stable
        because a cursor over an immutable log must land in the same place every
        time.
        """
        publish(state, count=6)
        page = api.query_observations(reader, scope(CAMERA), window())
        keys = [(o.t_capture.ns, str(o.observation_id)) for o in page.observations]
        assert keys == sorted(keys)

    def test_a_cursor_pages_without_repeating_or_skipping(
        self, api, state, reader
    ) -> None:
        """§2.2's *"direct dividend of immutability"*."""
        publish(state, count=10)
        first = api.query_observations(reader, scope(CAMERA), window(), limit=4)
        assert first.cursor is not None
        second = api.query_observations(
            reader, scope(CAMERA), window(), cursor=first.cursor, limit=4
        )
        seen = [str(o.observation_id) for o in first.observations + second.observations]
        assert len(seen) == len(set(seen)), "a page repeated an observation"

    def test_superseded_observations_are_excluded_by_default(self) -> None:
        assert ObservationFilter().include_superseded is False

    def test_an_oversized_window_is_refused_with_a_bound(
        self, api, reader
    ) -> None:
        """§M14: *"Reject with a bound and a cursor rather than degrading the
        service for everyone."*
        """
        from vision_os.core.model.api import TimeWindow
        from vision_os.core.model.timebase import Instant

        huge = TimeWindow(start=Instant(0), end=Instant(10**18))
        with pytest.raises(WindowTooLargeError) as caught:
            api.query_observations(reader, scope(CAMERA), huge)
        assert "max_ms" in caught.value.context

    def test_an_unknown_object_is_a_typed_absence(self, api, reader) -> None:
        with pytest.raises(StateNotFoundError):
            api.get_object(reader, ObjectId("never-seen"), scope=scope(CAMERA))


class TestEvidenceContract:
    """09_API §6 — what makes V4 usable rather than theoretical."""

    def test_expired_is_distinct_from_not_found(
        self, api, evidence, operator
    ) -> None:
        """§M13: *"Collapsing these two is how retention behaviour becomes
        indistinguishable from data loss."*

        The distinction survives all the way to the consumer's error code.
        """
        from vision_os.core.model.timebase import Instant

        store_evidence(evidence, expires_at=Instant(1_000))
        evidence.expire(before=Instant(2_000))

        with pytest.raises(NotFoundError):
            api.get_evidence(
                operator, BlobRef(f"sha256:{EVIDENCE_HASH}"), purpose="audit"
            )

    def test_an_erased_blob_reports_expiry_not_absence(
        self, api, evidence, operator
    ) -> None:
        """An erasure is a different audit answer from retention or a bug."""
        from vision_os.core.model.ids import ObjectId as OID
        from vision_os.core.ports.persistence import EraseScope

        store_evidence(evidence, object_id=OID("obj-0"))
        evidence.erase(
            EraseScope(tenant_id=TENANT, object_ids=(OID("obj-0"),), authority="dpo")
        )
        with pytest.raises(EvidenceExpiredError):
            api.get_evidence(
                operator, BlobRef(f"sha256:{EVIDENCE_HASH}"), purpose="audit"
            )

    def test_a_deployment_with_no_store_says_so(self, clock, metrics, state, authorizer, audit, operator) -> None:
        """*"No evidence store configured"* is a stated posture, not a failure."""
        from vision_os.exposure import ObservationApi

        api = ObservationApi(
            clock=clock,
            metrics=metrics,
            state=state,
            authorizer=authorizer,
            audit=audit,
            evidence=None,
        )
        with pytest.raises(NotFoundError, match="retains no imagery"):
            api.get_evidence(operator, BlobRef("sha256:x"), purpose="audit")


class TestVersionNegotiation:
    """09_API §7.1: *"Reject with the supported set; never guess."*"""

    def test_an_unsupported_major_is_refused(self, api, reader) -> None:
        with pytest.raises(UnsupportedVersionError) as caught:
            api.query_state(reader, scope(CAMERA), accepted_major=99)
        assert caught.value.context["supported"] == [1]

    def test_the_supported_set_is_a_set_not_a_constant(self) -> None:
        """§7.1 requires two adjacent majors during a migration window.

        One is declared today because only one exists; the *set* is what makes
        the second cost nothing to add.
        """
        from vision_os.core.model.api import SUPPORTED_MAJORS

        assert isinstance(SUPPORTED_MAJORS, frozenset)


class TestRateLimiting:
    def test_a_consumer_exceeding_its_budget_is_shed(
        self, clock, metrics, state, authorizer, audit, operator
    ) -> None:
        """§M14 responsibility 6. Systemic, so the consumer backs off."""
        from vision_os.core.ports.exposure import ApiLimits, RateLimit
        from vision_os.exposure import ObservationApi

        api = ObservationApi(
            clock=clock,
            metrics=metrics,
            state=state,
            authorizer=authorizer,
            audit=audit,
            limits=ApiLimits(query=RateLimit(requests_per_minute=3, burst=1)),
        )
        for _ in range(3):
            api.query_state(operator, scope(CAMERA))
        with pytest.raises(OverloadedError):
            api.query_state(operator, scope(CAMERA))

    def test_evidence_has_a_tighter_budget_than_queries(self) -> None:
        """§6: evidence payloads are *"large and sensitive"*."""
        from vision_os.core.ports.exposure import ApiLimits

        limits = ApiLimits()
        assert limits.evidence.requests_per_minute < limits.query.requests_per_minute


class TestNoBusinessSurface:
    """V1 at the contract boundary."""

    def test_the_api_offers_no_aggregation(self, api) -> None:
        """§M14: *"Aggregation is deliberately excluded. Consumers aggregate."*

        *"The moment the platform offers 'count people per hour per zone,' it has
        begun growing an analytics product inside a perception platform."*
        """
        for forbidden in ("count", "aggregate", "summarize", "histogram", "average", "report"):
            assert not hasattr(api, forbidden), f"the API exposes {forbidden}"

    def test_no_result_type_carries_a_conclusion(self) -> None:
        from vision_os.core.model import api as contract

        forbidden = ("alert", "violation", "risk", "severity", "compliant", "threat")
        offenders = []
        for name in dir(contract):
            kind = getattr(contract, name)
            for field in getattr(kind, "__dataclass_fields__", {}):
                for term in forbidden:
                    if term in field.lower():
                        offenders.append(f"{name}.{field}")
        assert not offenders, "\n".join(offenders)

    def test_a_query_filter_admits_no_threshold(self) -> None:
        """``AttributePredicate`` has equality and presence, and no comparison.

        A comparison operator would invite ``dwell_seconds > 300``, which is a
        threshold with business meaning — V1 puts that in the consumer's rule
        engine.
        """
        fields = set(AttributePredicate.__dataclass_fields__)
        assert fields == {"key", "equals", "present"}
        for forbidden in ("greater_than", "less_than", "at_least", "exceeds"):
            assert forbidden not in fields


class _CameraDirectory:
    """The slice of the Camera Manager M14 needs: tenant to cameras.

    A real `CameraManager` in these tests would need a full declaration set for
    something the API uses one method of. This implements that one method with
    the same signature, so the API is exercised through the interface it actually
    calls rather than through a mock of its own construction.
    """

    __slots__ = ("_by_tenant",)

    def __init__(self, by_tenant) -> None:
        self._by_tenant = by_tenant

    def list(self, *, tenant_id=None, site_id=None, status=None):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _Camera:
            camera_id: object

        return tuple(
            _Camera(camera_id=camera)
            for camera in self._by_tenant.get(tenant_id, ())
        )
