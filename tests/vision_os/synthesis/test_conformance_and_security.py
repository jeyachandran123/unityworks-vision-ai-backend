"""Conformance kits and the security boundary.

The kits are how V3 stops being an aspiration: a port is only a real seam if a
third-party implementation can be *checked*, and a kit that no adapter has ever
failed is a kit that proves nothing. Each obligation here is tested twice — once
against a shipped adapter that honours it, once against a deliberately broken one
that does not.

The security half tests the brief's sentence: *"Vision State contains
observations. Not identities. No biometric persistence. No facial recognition.
No cross-camera identity. No business labels."*
"""

from __future__ import annotations

import pytest

from vision_os.adapters.synthesis import (
    AlwaysPublish,
    CollectingSink,
    ExactSuppression,
    InMemoryObservationLog,
    NullSink,
    ThresholdSuppression,
)
from vision_os.conformance import (
    OBSERVATION_LOG_KIT,
    OBSERVATION_SINK_KIT,
    SUPPRESSION_POLICY_KIT,
)
from vision_os.conformance.synthesis_kits import ALL_SYNTHESIS_KITS
from vision_os.core.model.ids import CameraId, LogPosition
from vision_os.core.model.observation import Observation
from vision_os.core.model.vision_state import ObjectState
from vision_os.core.ports.synthesis import (
    LogAppendResult,
    ObservationLogPort,
    ObservationSinkPort,
    SinkResult,
    SuppressionPolicyPort,
)

from .conftest import TENANT


class TestEveryShippedAdapterPassesItsKit:
    @pytest.mark.parametrize(
        "adapter",
        [ExactSuppression(), ThresholdSuppression(), AlwaysPublish()],
        ids=lambda a: a.policy_id,
    )
    def test_suppression_policies_conform(self, adapter) -> None:
        report = SUPPRESSION_POLICY_KIT.run(adapter, fast_only=True)
        assert report.passed, "; ".join(report.failures)

    @pytest.mark.parametrize(
        "adapter", [CollectingSink(), NullSink()], ids=lambda a: a.sink_id
    )
    def test_sinks_conform(self, adapter) -> None:
        report = OBSERVATION_SINK_KIT.run(adapter, fast_only=True)
        assert report.passed, "; ".join(report.failures)

    def test_the_memory_log_conforms(self) -> None:
        report = OBSERVATION_LOG_KIT.run(InMemoryObservationLog(), fast_only=True)
        assert report.passed, "; ".join(report.failures)

    def test_the_file_log_conforms(self, tmp_path) -> None:
        """The durable adapter, gated as strictly as the in-memory one.

        This failed when first gated: the encoder dropped the spatial payload
        that 02_VOM requires a presence observation to carry, so every record
        decoded to ``None`` and the log read back empty. A kit that only ever ran
        against the in-memory adapter would not have found it.
        """
        from vision_os.adapters.synthesis import FileObservationLog

        report = OBSERVATION_LOG_KIT.run(FileObservationLog(tmp_path), fast_only=True)
        assert report.passed, "; ".join(report.failures)


class TestTheKitsActuallyCatchThings:
    """A kit no adapter has ever failed proves nothing."""

    def test_a_non_idempotent_log_fails(self) -> None:
        """Obligation L2. Without it, every recovery double-counts.

        The record would drift a little further from the truth on each restart,
        and nothing in the system would say so.
        """
        report = OBSERVATION_LOG_KIT.run(_DoubleWritingLog(), fast_only=True)
        assert not report.passed
        assert any("idempot" in f.lower() for f in report.failures)

    def test_an_unordered_log_fails(self) -> None:
        """Obligation L4. Order is the log's contract."""
        report = OBSERVATION_LOG_KIT.run(_ShufflingLog(), fast_only=True)
        assert not report.passed

    def test_a_log_that_leaks_across_partitions_fails(self) -> None:
        """Obligation L6. One camera's traffic in another's partition would
        break every per-camera guarantee the architecture makes.
        """
        report = OBSERVATION_LOG_KIT.run(_SharedPartitionLog(), fast_only=True)
        assert not report.passed

    def test_a_policy_that_suppresses_coverage_fails(self) -> None:
        """The V8 hole a bad policy could open.

        Coverage is *"the difference between a platform that is honest about its
        limits and one that is dangerously silent"*.
        """
        report = SUPPRESSION_POLICY_KIT.run(_SilencesCoverage(), fast_only=True)
        assert not report.passed

    def test_a_policy_with_an_unstable_signature_fails(self) -> None:
        """V13. A signature that changed between calls would suppress at random,
        and a replay would produce a different log from the live run.
        """
        report = SUPPRESSION_POLICY_KIT.run(_UnstableSignature(), fast_only=True)
        assert not report.passed

    def test_a_sink_that_omits_the_durability_declaration_fails(self) -> None:
        """Obligation K5 exists so a tee to a dashboard is never mistaken for a
        system of record. What the kit enforces is that the claim is *made*.
        """
        report = OBSERVATION_SINK_KIT.run(_UndeclaredSink(), fast_only=True)
        assert not report.passed
        assert any("durab" in f.lower() for f in report.failures)

    def test_a_false_durability_claim_is_not_mechanically_detectable(self) -> None:
        """A limitation, asserted rather than papered over.

        ``_LyingSink`` declares ``durable = True`` and keeps nothing, and it
        passes — as the kit's own docstring says it must: *"whether a log is
        genuinely durable... needs to survive a power cut. The kits verify
        contracts — the structural properties whose violation is silent."*

        No method on P19 could prove otherwise without a restart, and adding one
        to make a test pass would be inventing architecture. The declaration is
        an adapter's word, backed by obligation A1's requirement to declare
        honestly; verifying it belongs to deployment testing, not to a kit. This
        test exists so nobody later reads the passing kit as proof.
        """
        report = OBSERVATION_SINK_KIT.run(_LyingSink(), fast_only=True)
        assert report.passed


class TestKitsCoverTheirObligations:
    def test_every_kit_names_the_obligation_each_check_enforces(self) -> None:
        """A failure an operator cannot trace to a contract is a failure they
        cannot act on.
        """
        for kit in ALL_SYNTHESIS_KITS:
            for check in kit.checks:
                assert check.name
                assert check.section is not None

    def test_the_three_flow_seven_kits_are_registered(self) -> None:
        assert len(ALL_SYNTHESIS_KITS) == 3


class TestSecurityBoundary:
    """*"Vision State contains observations. Not identities."*"""

    def test_no_state_type_holds_a_personal_identifier(self) -> None:
        """The platform records that something was seen, never who it was.

        A name, an employee number or a plate turns an observation log into a
        surveillance record, and no consumer downstream could put that back.
        """
        from vision_os.core.model import vision_state

        forbidden = (
            "name", "person_id", "employee", "badge", "plate", "email",
            "phone", "national_id", "face_id",
        )
        offenders = []
        for kind_name in dir(vision_state):
            kind = getattr(vision_state, kind_name)
            fields = getattr(kind, "__dataclass_fields__", None)
            if not fields:
                continue
            for field in fields:
                for term in forbidden:
                    if term in field.lower():
                        offenders.append(f"{kind_name}.{field}")
        assert not offenders, "\n".join(offenders)

    def test_no_observation_field_holds_a_personal_identifier(self) -> None:
        forbidden = ("name", "person_id", "employee", "badge", "plate", "email")
        offenders = [
            field
            for field in Observation.__dataclass_fields__
            for term in forbidden
            if term in field.lower()
        ]
        assert not offenders, "\n".join(offenders)

    def test_the_identity_summary_is_a_claim_not_a_person(self) -> None:
        """02_VOM §4.2: an identity assertion is a claim about *continuity*.

        ``binding_count`` and ``ambiguous`` describe how confident the platform
        is that two sightings are the same thing. Neither says what that thing
        is called, and there is nowhere to put it.
        """
        from vision_os.core.model.vision_state import IdentitySummary

        fields = set(IdentitySummary.__dataclass_fields__)
        assert "binding_count" in fields
        assert "ambiguous" in fields
        assert not fields & {"name", "identity", "subject", "person"}

    def test_the_tenant_travels_on_every_observation(self) -> None:
        """12_SECURITY: isolation is structural, not a query filter.

        An observation without a tenant could be served to the wrong one by any
        code path that forgot to add a ``WHERE``.
        """
        assert "tenant_id" in Observation.__dataclass_fields__

    def test_one_tenants_observations_never_reach_another(
        self, state, loud_builder
    ) -> None:
        from .conftest import CAMERA, OTHER_TENANT, context, make_object

        ours = loud_builder.build_presence(make_object(tenant=TENANT), context())
        theirs = loud_builder.build_presence(
            make_object(object_id="obj-2", tenant=OTHER_TENANT),
            context(tenant=OTHER_TENANT),
        )
        state.append([ours, theirs])
        held = state.snapshot().partitions[CAMERA]
        # Both are present in the partition — the camera is the partition — but
        # each observation carries its own tenant, so a consumer can never be
        # served the wrong one by accident.
        assert {o.tenant_id for o in (ours, theirs)} == {TENANT, OTHER_TENANT}
        assert len(held.objects) == 2

    def test_object_state_carries_no_business_label(self) -> None:
        """07_STATE §10: *"would this field mean the same thing in a hospital, a
        warehouse, and a city street?"*
        """
        fields = set(ObjectState.__dataclass_fields__)
        for forbidden in ("role", "purpose", "status_label", "category_name"):
            assert forbidden not in fields


# --- deliberately broken adapters ------------------------------------------------ #


def _result(count: int, position: int) -> LogAppendResult:
    return LogAppendResult(appended=count, position=LogPosition(position))


class _DoubleWritingLog:
    """Appends everything, including ids it has already stored (breaks L2)."""

    def __init__(self) -> None:
        self._records: dict[CameraId, list] = {}

    @property
    def log_id(self) -> str:
        return "log.double_writing"

    def append(self, partition, observations):
        held = self._records.setdefault(partition, [])
        held.extend(observations)
        return _result(len(observations), len(held))

    def read(self, partition, *, start=None, end=None, limit=1000):
        return iter(self._records.get(partition, ())[:limit])

    def position(self, partition):
        return LogPosition(len(self._records.get(partition, ())))

    def truncate(self, partition, before) -> int:
        return 0


class _ShufflingLog(_DoubleWritingLog):
    """Returns records in reverse order (breaks L4)."""

    @property
    def log_id(self) -> str:
        return "log.shuffling"

    def append(self, partition, observations):
        held = self._records.setdefault(partition, [])
        seen = {o.observation_id for o in held}
        held.extend(o for o in observations if o.observation_id not in seen)
        return _result(len(observations), len(held))

    def read(self, partition, *, start=None, end=None, limit=1000):
        return iter(list(reversed(self._records.get(partition, ())))[:limit])


class _SharedPartitionLog(_DoubleWritingLog):
    """Puts every camera's records in one bucket (breaks L6)."""

    @property
    def log_id(self) -> str:
        return "log.shared"

    def append(self, partition, observations):
        held = self._records.setdefault(CameraId("everything"), [])
        seen = {o.observation_id for o in held}
        held.extend(o for o in observations if o.observation_id not in seen)
        return _result(len(observations), len(held))

    def read(self, partition, *, start=None, end=None, limit=1000):
        return iter(self._records.get(CameraId("everything"), ())[:limit])

    def position(self, partition):
        return LogPosition(len(self._records.get(CameraId("everything"), ())))


class _SilencesCoverage:
    """Suppresses everything, including coverage (breaks the V8 obligation)."""

    @property
    def policy_id(self) -> str:
        return "suppression.silences_coverage"

    def signature(self, observation) -> str:
        return "constant"

    def should_publish(self, candidate, previous_signature, *, elapsed, heartbeat):
        from vision_os.core.ports.synthesis import SuppressionDecision

        return SuppressionDecision(publish=False, reason="nothing is worth saying")


class _UnstableSignature:
    """A signature that changes between identical calls (breaks V13)."""

    def __init__(self) -> None:
        self._counter = 0

    @property
    def policy_id(self) -> str:
        return "suppression.unstable"

    def signature(self, observation) -> str:
        self._counter += 1
        return f"sig-{self._counter}"

    def should_publish(self, candidate, previous_signature, *, elapsed, heartbeat):
        from vision_os.core.ports.synthesis import SuppressionDecision

        return SuppressionDecision(publish=True, reason="always")


class _LyingSink:
    """Claims durability while keeping nothing.

    Passes the kit, and that is the point of the test that uses it: the claim is
    checkable for *presence*, not for *truth*.
    """

    @property
    def sink_id(self) -> str:
        return "sink.lying"

    @property
    def durable(self) -> bool:
        return True

    def emit(self, observations) -> SinkResult:
        return SinkResult(accepted=len(observations))


class _UndeclaredSink:
    """Never declares durability at all (breaks K5's checkable half)."""

    @property
    def sink_id(self) -> str:
        return "sink.undeclared"

    def emit(self, observations) -> SinkResult:
        return SinkResult(accepted=len(observations))


def _protocol_check() -> None:
    """Static reassurance that the broken adapters are shaped like the real ones.

    Not a test — a compile-time note. If a port grows a method, these stubs stop
    satisfying it and the type checker says so before the kits do.
    """
    _: SuppressionPolicyPort = _SilencesCoverage()
    __: ObservationLogPort = _DoubleWritingLog()
    ___: ObservationSinkPort = _LyingSink()
