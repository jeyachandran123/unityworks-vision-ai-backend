"""Every shipped adapter must pass its port's conformance kit.

This is the executable form of invariant V3. If an adapter in the repository
cannot pass its own kit, "every model is replaceable" is already false.

The suite also proves the kits have *teeth* by running deliberately broken
adapters through them — a kit that passes everything proves nothing.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.acquisition import (
    ArrivalTimeClockSync,
    FailingMask,
    InMemoryRawSource,
    NoMaskPolicy,
    PassthroughDecoder,
    PtsClockSync,
    StaticZoneMask,
    UnknownClockSync,
    WallclockHintClockSync,
)
from vision_os.adapters.configuration import (
    InMemoryConfigSource,
    InMemorySecretProvider,
)
from vision_os.adapters.memory import HostMemoryPool
from vision_os.adapters.observability import (
    InMemoryMetricsExporter,
    NullEventTransport,
    OpenMetricsTextExporter,
    RecordingEventTransport,
)
from vision_os.adapters.scheduling import (
    AdmitAllPolicy,
    CadenceAdmissionPolicy,
    NullChangeDetector,
    ResolutionLadderPolicy,
    SampledDigestChangeDetector,
)
from vision_os.conformance import KitSection, flow1_registry
from vision_os.conformance.flow1_kits import (
    ADMISSION_KIT,
    ALLOCATOR_KIT,
    CLOCK_SYNC_KIT,
    DECODER_KIT,
    PRIVACY_KIT,
)
from vision_os.core.errors import PoolExhaustedError
from vision_os.core.model.frame import DecodeQuality, FrameDimensions, PrivacyState
from vision_os.core.model.ids import PrivacyPolicyId
from vision_os.core.model.space import Point, Polygon
from vision_os.core.model.timebase import ClockQuality, Duration
from vision_os.core.ports.acquisition import DecodeOutcome, MaskOutcome
from vision_os.core.ports.scheduling import AdmissionVerdict, Fidelity
from vision_os.kernel.clock import VirtualClock
from vision_os.kernel.plugins import PortCatalogue

DIMENSIONS = FrameDimensions(width=8, height=4, colour_space="bgr24")
FRAME_BYTES = 8 * 4 * 3


def _adapters():
    clock = VirtualClock()
    zone = Polygon((Point(0, 0), Point(0.5, 0), Point(0.5, 0.5), Point(0, 0.5)))
    return [
        (PortCatalogue.ALLOCATOR, HostMemoryPool(slots=4, bytes_per_slot=FRAME_BYTES)),
        (PortCatalogue.DECODER, PassthroughDecoder(dimensions=DIMENSIONS)),
        (PortCatalogue.PRIVACY_MASK, NoMaskPolicy()),
        (
            PortCatalogue.PRIVACY_MASK,
            StaticZoneMask(policy_id=PrivacyPolicyId("p"), zones=(zone,)),
        ),
        (PortCatalogue.PRIVACY_MASK, FailingMask()),
        (PortCatalogue.CLOCK_SYNC, ArrivalTimeClockSync()),
        (PortCatalogue.CLOCK_SYNC, WallclockHintClockSync()),
        (PortCatalogue.CLOCK_SYNC, PtsClockSync()),
        (PortCatalogue.CLOCK_SYNC, UnknownClockSync()),
        (PortCatalogue.SOURCE, InMemoryRawSource([], clock=clock)),
        (PortCatalogue.ADMISSION_POLICY, CadenceAdmissionPolicy()),
        (PortCatalogue.ADMISSION_POLICY, AdmitAllPolicy()),
        (PortCatalogue.ADMISSION_POLICY, ResolutionLadderPolicy()),
        (PortCatalogue.CHANGE_DETECTOR, NullChangeDetector()),
        (PortCatalogue.CHANGE_DETECTOR, SampledDigestChangeDetector()),
        (PortCatalogue.CONFIG_SOURCE, InMemoryConfigSource({})),
        (PortCatalogue.SECRET_PROVIDER, InMemorySecretProvider({"a": "b"})),
        (PortCatalogue.EVENT_TRANSPORT, NullEventTransport()),
        (PortCatalogue.EVENT_TRANSPORT, RecordingEventTransport()),
        (PortCatalogue.METRICS_EXPORT, InMemoryMetricsExporter()),
        (PortCatalogue.METRICS_EXPORT, OpenMetricsTextExporter()),
    ]


@pytest.mark.parametrize(
    ("port_id", "adapter"),
    _adapters(),
    ids=lambda value: type(value).__name__ if not isinstance(value, str) else value.split(".")[0],
)
def test_every_shipped_adapter_passes_its_kit(port_id, adapter) -> None:
    kit = flow1_registry().get(port_id)
    assert kit is not None, f"no kit registered for {port_id}"
    report = kit.run(adapter, fast_only=False)
    assert report.passed, f"{type(adapter).__name__}: {report.failures}"


def test_every_flow1_port_has_a_kit() -> None:
    """An adapter cannot be activated for a port with no kit (V3)."""
    from vision_os.kernel.plugins.manifest import FLOW1_PORTS

    registry = flow1_registry()
    missing = [port for port in FLOW1_PORTS if registry.get(port) is None]
    assert not missing, f"ports without a conformance kit: {missing}"


def test_critical_kits_cover_the_failure_section() -> None:
    """Failure behaviour is where adapters diverge most dangerously."""
    for kit in (ALLOCATOR_KIT, DECODER_KIT, PRIVACY_KIT, CLOCK_SYNC_KIT):
        assert KitSection.FAILURE in kit.sections_covered(), kit.port_id


# --- proof that the kits have teeth ------------------------------------------- #


class _LeakyPool(HostMemoryPool):
    """Never returns slots — the slow soak failure that looks fine on day one."""

    def release(self, allocation) -> None:
        return None


class _SilentlyTruncatingDecoder(PassthroughDecoder):
    """Truncates oversized input instead of raising (adapter obligation A4)."""

    def decode_into(self, packet, slot):
        payload = packet.payload[: slot.capacity]
        slot.memory()[: len(payload)] = payload
        return DecodeOutcome(
            dimensions=DIMENSIONS,
            bytes_written=len(payload),
            decode_quality=DecodeQuality.KEYFRAME,
        )


class _FabricatingMask:
    """Reports success after an internal failure.

    The most dangerous privacy adapter possible: a compliance incident that
    looks exactly like success.
    """

    @property
    def policy_id(self):
        return PrivacyPolicyId("fabricating")

    def apply(self, slot, dimensions) -> MaskOutcome:
        return MaskOutcome(state=PrivacyState.MASK_FAILED)


class _OverconfidentClockSync:
    """Claims PTP accuracy while reporting a 900ms uncertainty."""

    def estimate(self, packet, ingest):
        from vision_os.core.ports.acquisition import CaptureEstimate

        return CaptureEstimate(
            t_capture=ingest,
            uncertainty=Duration.from_millis(900),
            quality=ClockQuality.PTP_LOCKED,
        )

    def reset(self) -> None:
        return None


class _UnattributedDropPolicy:
    """Drops without a reason — an invariant V8 violation."""

    def evaluate(self, context):
        if context.queue_full:
            return AdmissionVerdict(admit=False, reason=None)  # constructor will reject
        return AdmissionVerdict(
            admit=True,
            fidelity=Fidelity(
                inference_width=context.profile.inference_width,
                inference_height=context.profile.inference_height,
            ),
        )


class TestKitsRejectBrokenAdapters:
    def test_leaky_allocator_fails_the_resource_section(self) -> None:
        report = ALLOCATOR_KIT.run(
            _LeakyPool(slots=4, bytes_per_slot=FRAME_BYTES), fast_only=False
        )
        assert not report.passed
        assert any("no_steady_state_growth" in failure for failure in report.failures)

    def test_silently_truncating_decoder_fails(self) -> None:
        report = DECODER_KIT.run(_SilentlyTruncatingDecoder(dimensions=DIMENSIONS))
        assert not report.passed
        assert any("oversized_payload_is_typed" in f for f in report.failures)

    def test_fabricating_privacy_adapter_fails(self) -> None:
        """The fail-closed obligation, proven enforceable."""
        report = PRIVACY_KIT.run(_FabricatingMask())
        assert not report.passed
        assert any("fails_closed" in f for f in report.failures)

    def test_overconfident_clock_sync_fails(self) -> None:
        report = CLOCK_SYNC_KIT.run(_OverconfidentClockSync())
        assert not report.passed
        assert any("honest_without_hints" in f for f in report.failures)

    def test_unattributed_drop_policy_fails(self) -> None:
        report = ADMISSION_KIT.run(_UnattributedDropPolicy())
        assert not report.passed


class TestFastSubset:
    def test_fast_subset_runs_shape_semantics_and_failure_only(self) -> None:
        report = ALLOCATOR_KIT.run(
            HostMemoryPool(slots=4, bytes_per_slot=FRAME_BYTES), fast_only=True
        )
        assert report.passed
        assert report.fast_subset_only
        assert any("resource/" in skipped for skipped in report.skipped)

    def test_fast_subset_still_catches_the_catastrophic_class(self) -> None:
        """Seconds at load, before a single real frame is processed."""
        report = PRIVACY_KIT.run(_FabricatingMask(), fast_only=True)
        assert not report.passed


def test_report_summary_is_human_readable() -> None:
    report = ALLOCATOR_KIT.run(HostMemoryPool(slots=2, bytes_per_slot=16), fast_only=True)
    summary = report.summary()
    assert "PASS" in summary or "FAIL" in summary
    assert "fast" in summary


def test_pool_exhaustion_remains_typed_under_the_kit() -> None:
    pool = HostMemoryPool(slots=1, bytes_per_slot=16)
    pool.allocate(16)
    with pytest.raises(PoolExhaustedError):
        pool.allocate(16)
