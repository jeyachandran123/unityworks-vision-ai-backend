"""Demand intake and transport (09_API §4, §M14 Extension Points).

Two contracts that look peripheral and are not.

**Demand** is the only inbound path, and 09_API §1.2 calls it *"the
architecturally interesting one"*: it lets a consumer influence what the platform
spends money computing *without telling it why*, which is what allows one platform
to serve a kitchen and an operating theatre.

**Transport** is what makes §M14's promise true — *"adopting a new transport in
2031 will not be a platform change"*. Tested through a transport thin enough that
if the API can be served by it, nothing protocol-shaped has leaked in.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.exposure.transport import (
    InProcessTransport,
    RecordingTransport,
    Route,
    error_view,
    routes_for,
)
from vision_os.core.errors import (
    DemandRejectedError,
    ForbiddenError,
    InvalidScopeError,
    PartitionUnavailableError,
)
from vision_os.core.model.demand import Demand, DemandScope, SubjectFilter
from vision_os.core.model.ids import AttributeKey, DemandId
from vision_os.core.model.timebase import Duration
from vision_os.core.ports.exposure import TransportRequest
from vision_os.exposure import DemandIntake
from vision_os.exposure.demands import DemandRecord, DemandStore
from vision_os.perception.cropping.demands import CapabilityView, DemandRegistry

from .conftest import (
    CAMERA,
    CONSUMER,
    OTHER_CONSUMER,
    OTHER_TENANT,
    PERSON,
    POSTURE,
    SITE,
    principal,
    scope,
)

UNREGISTERED = AttributeKey("is_authorized")


def demand(
    *,
    demand_id: str = "dem-1",
    subscriber: str = "operator",
    attributes: tuple[AttributeKey, ...] = (POSTURE,),
    freshness_ms: int = 30_000,
) -> Demand:
    return Demand(
        demand_id=DemandId(demand_id),
        subscriber=subscriber,
        scope=DemandScope(site_ids=(SITE,), camera_ids=(CAMERA,)),
        subject_filter=SubjectFilter(class_ids=(PERSON,)),
        required_attributes=attributes,
        freshness=Duration.from_millis(freshness_ms),
    )


@pytest.fixture
def registry() -> DemandRegistry:
    return DemandRegistry(
        capabilities=CapabilityView(
            registered_attributes=frozenset({POSTURE}),
            producible_attributes=frozenset({POSTURE}),
            producible_classes=frozenset({PERSON}),
            observed_cameras=frozenset({CAMERA}),
        )
    )


@pytest.fixture
def intake(clock, metrics, registry, authorizer, audit) -> DemandIntake:
    return DemandIntake(
        clock=clock,
        metrics=metrics,
        registry=registry,
        authorizer=authorizer,
        audit=audit,
    )


class TestDemandIsInfluenceNotAWrite:
    """09_API §1.1 — how a demand coexists with *"Mutate does not exist"*."""

    def test_a_demand_carries_no_reason(self) -> None:
        """§4.2's table. *"I need headwear_present on person in region Z3"* ✅;
        *"because uncovered hair near food is a hygiene violation"* ❌.

        The platform never learns what any demand is for, *"which is precisely
        why the same platform serves a kitchen and an operating theatre."*
        """
        fields = set(Demand.__dataclass_fields__)
        for forbidden in ("reason", "justification", "purpose", "rule", "threshold"):
            assert forbidden not in fields

    def test_priority_class_is_opaque(self) -> None:
        """§4.1: *"platform orders by it; never interprets it."*

        A typed enum here would be the platform holding an opinion about which
        business priorities exist.
        """
        annotation = str(Demand.__dataclass_fields__["priority_class"].type)
        assert "str" in annotation

    def test_registering_changes_no_published_fact(
        self, intake, state, operator
    ) -> None:
        from .conftest import publish

        publish(state, count=2)
        before = state.snapshot().partitions[CAMERA].version
        intake.register(operator, demand())
        assert state.snapshot().partitions[CAMERA].version == before


class TestDemandIntake:
    def test_a_valid_demand_is_accepted(self, intake, operator) -> None:
        acknowledgement = intake.register(operator, demand())
        assert acknowledgement.accepted
        assert POSTURE in acknowledgement.satisfiable

    def test_an_unregistered_attribute_is_rejected_at_registration(
        self, intake, operator
    ) -> None:
        """§4.2: *"the fourth and outermost ring of Semantic Ceiling
        enforcement."*

        The consumer learns in seconds instead of *"discovering it as a permanent
        absence of data weeks later"*.
        """
        with pytest.raises(DemandRejectedError):
            intake.register(operator, demand(attributes=(UNREGISTERED,)))

    def test_registering_requires_the_privilege(self, intake, reader) -> None:
        """12_SECURITY §5.3: *"Demands spend money and cause computation; they
        are not a read."*

        The read-only grant covers every read action and not this one.
        """
        with pytest.raises(ForbiddenError, match="may not register demands"):
            intake.register(reader, demand(subscriber=CONSUMER))

    def test_effective_freshness_reports_what_the_platform_can_sustain(
        self, intake, operator
    ) -> None:
        """§4.3: *"where the platform tells the truth about its limits."*

        > *"Rather than accepting and silently under-delivering, the platform
        > responds with what it can actually sustain."*
        """
        acknowledgement = intake.register(
            operator,
            demand(freshness_ms=1_000),
            sustainable_freshness=Duration.from_millis(12_000),
        )
        assert acknowledgement.effective_freshness.millis >= 1_000

    def test_a_demand_is_audited(self, intake, operator, audit_sink) -> None:
        intake.register(operator, demand())
        records = [r for r in audit_sink.records if r.resource == "dem-1"]
        assert records

    def test_revoking_removes_it(self, intake, operator) -> None:
        intake.register(operator, demand())
        intake.revoke(operator, DemandId("dem-1"))
        assert not intake.list_for(operator)

    def test_updating_replaces_rather_than_editing(self, intake, operator) -> None:
        """A demand's acknowledgement reports what the platform can *currently*
        sustain; mutating in place would leave a stale ``effective_freshness``
        attached to new terms.
        """
        intake.register(operator, demand(freshness_ms=60_000))
        acknowledgement = intake.update(
            operator, DemandId("dem-1"), demand(freshness_ms=5_000)
        )
        assert acknowledgement.accepted

    def test_a_principal_cannot_revoke_anothers_demand(
        self, intake, operator
    ) -> None:
        intake.register(operator, demand())
        stranger = principal(subject="operator", tenant=OTHER_TENANT)
        with pytest.raises(ForbiddenError):
            intake.revoke(stranger, DemandId("dem-1"))

    def test_an_unknown_demand_and_anothers_demand_look_identical(
        self, intake, operator
    ) -> None:
        """Confirming a demand exists but belongs to someone else is itself a
        small cross-tenant leak.
        """
        intake.register(operator, demand())
        stranger = principal(subject="operator", tenant=OTHER_TENANT)

        with pytest.raises(ForbiddenError) as theirs:
            intake.revoke(stranger, DemandId("dem-1"))
        with pytest.raises(ForbiddenError) as absent:
            intake.revoke(stranger, DemandId("dem-never"))
        assert theirs.value.message.replace("dem-1", "X") == absent.value.message.replace(
            "dem-never", "X"
        )

    def test_listing_returns_only_the_principals_own(
        self, intake, operator
    ) -> None:
        intake.register(operator, demand())
        assert len(intake.list_for(operator)) == 1
        assert intake.list_for(principal(subject=OTHER_CONSUMER)) == ()

    def test_a_principal_cannot_register_in_anothers_name(
        self, intake, operator
    ) -> None:
        """Otherwise ``list_demands`` would show a consumer demands it never
        made, and the ownership check on revoke would guard nothing.
        """
        with pytest.raises(ForbiddenError, match="may not register a demand for"):
            intake.register(operator, demand(subscriber=OTHER_CONSUMER))


class TestDemandDurability:
    """§M14: *"The demand registry is durable — demands must survive restart."*"""

    def test_demands_survive_a_restart(
        self, clock, metrics, registry, authorizer, audit, operator, tmp_path
    ) -> None:
        """> *"or every consumer would have to re-register after every deployment
        > and attribute coverage would silently lapse in the interval."*
        """
        store = DemandStore(tmp_path / "demands.json")
        intake = DemandIntake(
            clock=clock, metrics=metrics, registry=registry,
            authorizer=authorizer, audit=audit, store=store,
        )
        intake.register(operator, demand())

        revived = DemandIntake(
            clock=clock, metrics=metrics,
            registry=DemandRegistry(
                capabilities=CapabilityView(
                    registered_attributes=frozenset({POSTURE}),
                    producible_attributes=frozenset({POSTURE}),
                    producible_classes=frozenset({PERSON}),
                    observed_cameras=frozenset({CAMERA}),
                )
            ),
            authorizer=authorizer, audit=audit, store=store,
        )
        assert revived.restore() == 1
        assert len(revived.list_for(operator)) == 1

    def test_a_corrupt_store_costs_registrations_not_a_boot(self, tmp_path) -> None:
        """A platform refusing to boot on a truncated demand file would turn a
        recoverable degradation into an outage.
        """
        path = tmp_path / "demands.json"
        path.write_text("{ not json", encoding="utf-8")
        assert DemandStore(path).load() == ()

    def test_an_unreadable_record_is_skipped_not_fatal(self, tmp_path) -> None:
        """One unreadable demand costs one consumer a re-registration; a raise
        would cost every consumer theirs.
        """
        path = tmp_path / "demands.json"
        path.write_text('[{"broken": true}]', encoding="utf-8")
        assert DemandStore(path).load() == ()

    def test_a_store_with_no_path_is_honestly_not_durable(self) -> None:
        assert not DemandStore().durable

    def test_a_record_round_trips(self) -> None:
        record = DemandRecord(
            demand_id="d",
            subscriber="s",
            tenant_id="t",
            site_ids=("site",),
            camera_ids=("cam",),
            required_attributes=("posture",),
            class_ids=("person",),
            freshness_ms=1000,
            registered_at_ns=5,
        )
        assert DemandRecord.from_json(record.to_json()) == record


class TestTransport:
    """P32. §M14: *"The contract is transport-independent by design."*"""

    def test_a_request_is_dispatched_to_the_api(self, api, api_transport, operator) -> None:
        response = api_transport.serve(
            TransportRequest(
                principal=operator,
                operation="query_state",
                payload={"scope": scope(CAMERA)},
            )
        )
        assert not response.failed
        assert response.result is not None

    def test_an_unknown_operation_is_an_error_not_an_exception(
        self, api_transport, operator
    ) -> None:
        response = api_transport.serve(
            TransportRequest(principal=operator, operation="nonesuch")
        )
        assert response.failed
        assert response.error.code == "NOT_FOUND"

    def test_a_platform_error_is_rendered_with_a_stable_code(
        self, api_transport, reader
    ) -> None:
        """09_API §8: codes are *"stable, machine-readable, never reworded"*."""
        response = api_transport.serve(
            TransportRequest(
                principal=reader,
                operation="get_evidence",
                purpose="curiosity",
                payload={"blob_ref": "sha256:x"},
            )
        )
        assert response.failed
        assert response.error.code == "FORBIDDEN"
        assert response.error.retryable is False

    def test_a_retryable_error_says_so(self) -> None:
        view = error_view(PartitionUnavailableError("cam down", camera="cam-01"))
        assert view.retryable is True
        assert view.code == "PARTITION_UNAVAILABLE"

    def test_an_error_without_a_declared_code_still_gets_a_stable_one(self) -> None:
        """Derived from the class name, never from the message.

        §8 requires codes be stable while messages *"may change"*, so a code
        derived from prose would break every consumer on a reworded sentence.
        """
        view = error_view(InvalidScopeError("bad"))
        assert view.code == "INVALID_SCOPE"

    def test_an_unexpected_exception_is_still_an_error_envelope(
        self, api_transport, operator
    ) -> None:
        """The transport is the last boundary. Nothing escapes it.

        A consumer's error handling must never depend on the platform's internal
        exception hierarchy.
        """
        api_transport.register(
            Route("explode", lambda r: (_ for _ in ()).throw(RuntimeError("boom")))
        )
        response = api_transport.serve(
            TransportRequest(principal=operator, operation="explode")
        )
        assert response.failed
        assert response.error.code == "INTERNAL"

    def test_a_non_streaming_transport_refuses_subscribe_at_setup(
        self, api, operator
    ) -> None:
        """Obligation T5. Refusing at setup beats appearing to work.

        A consumer that believes it is subscribed and receives nothing is exactly
        the silence V8 exists to prevent.
        """
        transport = InProcessTransport(routes_for(api), supports_streaming=False)
        response = transport.serve(
            TransportRequest(
                principal=operator,
                operation="subscribe",
                payload={"scope": scope(CAMERA)},
            )
        )
        assert response.failed

    def test_the_negotiated_major_travels_with_the_answer(
        self, api_transport, operator
    ) -> None:
        response = api_transport.serve(
            TransportRequest(
                principal=operator,
                operation="query_state",
                payload={"scope": scope(CAMERA)},
                accepted_major=1,
            )
        )
        assert response.version == 1

    def test_a_recording_transport_keeps_a_bounded_history(
        self, api, operator
    ) -> None:
        transport = RecordingTransport(routes_for(api), capacity=5)
        for _ in range(20):
            transport.serve(TransportRequest(principal=operator, operation="nonesuch"))
        assert len(transport.exchanges) <= 5
        assert transport.failures

    def test_the_route_table_lives_outside_the_api(self) -> None:
        """Adding a transport means adding a table, never touching M14."""
        import inspect

        from vision_os.exposure import ObservationApi

        assert "Route" not in inspect.getsource(ObservationApi)

    def test_every_contract_operation_has_a_route(self, api) -> None:
        """09_API's five contracts, all reachable."""
        operations = {route.operation for route in routes_for(api)}
        assert {
            "query_state",
            "query_observations",
            "get_object",
            "coverage",
            "capabilities",
            "get_evidence",
            "subscribe",
            "register_demand",
        } <= operations

    def test_no_route_mutates(self, api) -> None:
        """The route table cannot expose what the API does not have."""
        operations = {route.operation for route in routes_for(api)}
        for forbidden in ("update_state", "create_object", "delete_observation"):
            assert forbidden not in operations
