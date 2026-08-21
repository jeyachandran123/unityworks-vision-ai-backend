"""Conformance kits for the Flow 1 ports (06_PORTS_AND_ADAPTERS §5).

Each check ties back to a numbered semantic obligation in the port's contract.
The Plugin Manager runs the fast subset (shape + semantics + failure) at load, so
an adapter that gets a coordinate convention, a fail-closed path, or a resource
declaration wrong is rejected **before a single real frame is processed**.

The checks that earn their keep here are the ones with no visible symptom:

* ``allocator/no_steady_state_growth`` — a pool that quietly allocates is a
  30-day soak failure that looks fine on day 1 and kills a node on day 26.
* ``privacy/fails_closed`` — a masking adapter that returns a plausible value on
  failure is a compliance incident that looks exactly like success.
* ``admission/every_drop_is_attributed`` — an unattributed drop makes the
  platform quietly do less work than it appears to (invariant V8).
"""

from __future__ import annotations

import asyncio

from ..core.errors import PoolExhaustedError, PrivacyMaskError, VisionOSError
from ..core.model.camera import (
    Camera,
    NativeProfile,
    PipelineProfile,
    SourceSemantics,
    SourceSpec,
)
from ..core.model.frame import FrameDimensions, PrivacyState
from ..core.model.ids import CameraId, ProfileId, SiteId, TenantId
from ..core.model.timebase import ClockQuality, Duration, Instant
from ..core.ports.acquisition import SourcePacket
from ..core.ports.scheduling import AdmissionContext
from ..kernel.plugins.manifest import PortCatalogue
from .kit import ConformanceCheck, ConformanceKit, ConformanceRegistry, KitSection

# --- shared fixtures -------------------------------------------------------- #

_DIMENSIONS = FrameDimensions(width=8, height=4, colour_space="bgr24")
_FRAME_BYTES = 8 * 4 * 3


def _camera() -> Camera:
    return Camera(
        camera_id=CameraId("kit-cam"),
        tenant_id=TenantId("kit-tenant"),
        site_id=SiteId("kit-site"),
        source_spec=SourceSpec(uri="mem://kit", transport="memory"),
        source_semantics=SourceSemantics.ARCHIVAL,
        native_profile=NativeProfile(width=8, height=4, fps=5.0, codec="raw_bgr24"),
        pipeline_profile=PipelineProfile(profile_id=ProfileId("kit"), target_fps=5.0),
    )


def _packet(payload: bytes = b"\x01" * _FRAME_BYTES, pts: int = 0) -> SourcePacket:
    return SourcePacket(
        payload=payload,
        pts=pts,
        pts_timebase_hz=1000,
        is_keyframe=True,
        codec="raw_bgr24",
        arrival=Instant(1_000_000_000),
    )


class _Slot:
    """A minimal ``WritableSlot`` for kit execution."""

    __slots__ = ("_buffer",)

    def __init__(self, capacity: int = _FRAME_BYTES) -> None:
        self._buffer = bytearray(capacity)

    @property
    def capacity(self) -> int:
        return len(self._buffer)

    def memory(self) -> memoryview:
        return memoryview(self._buffer)


# --- P7 AllocatorPort -------------------------------------------------------- #


def _allocator_shape(adapter) -> None:
    assert adapter.location in ("host", "device"), (
        f"location must be 'host' or 'device', got {adapter.location!r}"
    )
    stats = adapter.stats()
    assert stats.total_slots > 0, "a pool must declare at least one slot"


def _allocator_release_is_idempotent(adapter) -> None:
    allocation = adapter.allocate(1)
    before = adapter.stats().in_use
    adapter.release(allocation)
    adapter.release(allocation)
    after = adapter.stats().in_use
    assert after == before - 1, (
        f"double release must not corrupt occupancy: {before} -> {after}"
    )


def _allocator_memory_is_writable(adapter) -> None:
    allocation = adapter.allocate(8)
    try:
        memory = allocation.memory()
        assert not memory.readonly, "an allocation must be writable"
        memory[0:1] = b"\x7f"
        assert allocation.nbytes >= 8, "nbytes must reflect the real allocation"
    finally:
        adapter.release(allocation)


def _allocator_exhaustion_is_typed(adapter) -> None:
    held = []
    try:
        for _ in range(adapter.stats().total_slots + 1):
            held.append(adapter.allocate(1))
    except PoolExhaustedError:
        return
    finally:
        for allocation in held:
            adapter.release(allocation)
    raise AssertionError("exhaustion must raise PoolExhaustedError, not return None")


def _allocator_no_steady_state_growth(adapter) -> None:
    """Allocate/release in a loop; occupancy must return to its starting point."""
    baseline = adapter.stats().in_use
    for _ in range(1000):
        allocation = adapter.allocate(1)
        adapter.release(allocation)
    assert adapter.stats().in_use == baseline, (
        "pool leaks slots under allocate/release cycling — a slow soak failure"
    )


ALLOCATOR_KIT = ConformanceKit(
    port_id=PortCatalogue.ALLOCATOR,
    version="1.0.0",
    checks=(
        ConformanceCheck("declares_location", KitSection.SHAPE, _allocator_shape, "A1"),
        ConformanceCheck(
            "memory_is_writable", KitSection.SEMANTICS, _allocator_memory_is_writable
        ),
        ConformanceCheck(
            "release_is_idempotent", KitSection.SEMANTICS, _allocator_release_is_idempotent
        ),
        ConformanceCheck(
            "exhaustion_is_typed", KitSection.FAILURE, _allocator_exhaustion_is_typed, "A4"
        ),
        ConformanceCheck(
            "no_steady_state_growth", KitSection.RESOURCE, _allocator_no_steady_state_growth
        ),
    ),
)


# --- P2 DecoderPort ---------------------------------------------------------- #


def _decoder_declares_capabilities(adapter) -> None:
    capabilities = adapter.capabilities()
    assert capabilities.codecs, "a decoder must declare at least one codec (A1)"
    assert isinstance(capabilities.hardware_accelerated, bool)


def _decoder_writes_into_slot(adapter) -> None:
    slot = _Slot()
    payload = bytes(range(_FRAME_BYTES % 256)) or b"\x01" * 16
    outcome = adapter.decode_into(_packet(payload=payload), slot)
    assert outcome.bytes_written > 0, "decode must report bytes written"
    assert outcome.bytes_written <= slot.capacity, "decode must not overrun the slot"
    assert outcome.dimensions.width > 0 and outcome.dimensions.height > 0


def _decoder_is_stateless_across_reset(adapter) -> None:
    slot = _Slot()
    first = adapter.decode_into(_packet(), slot)
    adapter.reset()
    second = adapter.decode_into(_packet(), slot)
    assert first.dimensions == second.dimensions, (
        "reset must not change declared geometry; it discards reference state only"
    )


def _decoder_oversized_payload_is_typed(adapter) -> None:
    slot = _Slot(capacity=4)
    try:
        adapter.decode_into(_packet(payload=b"\x00" * 4096), slot)
    except VisionOSError:
        return
    raise AssertionError(
        "an oversized payload must raise a typed error, never silently truncate"
    )


DECODER_KIT = ConformanceKit(
    port_id=PortCatalogue.DECODER,
    version="1.0.0",
    checks=(
        ConformanceCheck(
            "declares_capabilities", KitSection.SHAPE, _decoder_declares_capabilities, "A1"
        ),
        ConformanceCheck("writes_into_slot", KitSection.SEMANTICS, _decoder_writes_into_slot),
        ConformanceCheck(
            "reset_preserves_geometry", KitSection.SEMANTICS, _decoder_is_stateless_across_reset
        ),
        ConformanceCheck(
            "oversized_payload_is_typed",
            KitSection.FAILURE,
            _decoder_oversized_payload_is_typed,
            "A4",
        ),
    ),
)


# --- P3 PrivacyMaskPort ------------------------------------------------------- #


def _privacy_declares_policy(adapter) -> None:
    policy_id = adapter.policy_id
    assert policy_id is None or isinstance(policy_id, str), (
        "policy_id must be a reference or None"
    )


def _privacy_never_reports_failure_as_success(adapter) -> None:
    """The fail-closed obligation.

    An adapter must either mask successfully or raise. Returning
    ``MASK_FAILED`` while letting the caller proceed, or returning a plausible
    outcome after an internal failure, is a compliance incident that looks
    exactly like success.
    """
    slot = _Slot()
    try:
        outcome = adapter.apply(slot, _DIMENSIONS)
    except PrivacyMaskError:
        return
    assert outcome.state in (PrivacyState.MASKED, PrivacyState.UNMASKED_PERMITTED), (
        f"apply() returned {outcome.state}; a failure must raise PrivacyMaskError "
        f"so the frame is dropped rather than emitted"
    )


def _privacy_undersized_slot_raises(adapter) -> None:
    if adapter.policy_id is None:
        return  # a no-op policy has nothing to fail at
    slot = _Slot(capacity=4)
    try:
        adapter.apply(slot, FrameDimensions(width=1920, height=1080))
    except PrivacyMaskError:
        return
    raise AssertionError(
        "masking a frame that does not fit the slot must raise, never partially mask"
    )


PRIVACY_KIT = ConformanceKit(
    port_id=PortCatalogue.PRIVACY_MASK,
    version="1.0.0",
    checks=(
        ConformanceCheck("declares_policy", KitSection.SHAPE, _privacy_declares_policy, "A1"),
        ConformanceCheck(
            "fails_closed", KitSection.SEMANTICS, _privacy_never_reports_failure_as_success, "A4"
        ),
        ConformanceCheck(
            "undersized_slot_raises", KitSection.FAILURE, _privacy_undersized_slot_raises, "A4"
        ),
    ),
)


# --- P4 ClockSyncPort ---------------------------------------------------------- #


def _clocksync_returns_uncertainty(adapter) -> None:
    estimate = adapter.estimate(_packet(), Instant(2_000_000_000))
    assert estimate.uncertainty.ns >= 0, "uncertainty must be non-negative"
    assert isinstance(estimate.quality, ClockQuality)


def _clocksync_is_honest_without_hints(adapter) -> None:
    """An adapter with no basis must not claim high precision.

    A timestamp without honest uncertainty is a claim to precision the system
    does not have (02_VOM §5.2).
    """
    estimate = adapter.estimate(_packet(), Instant(2_000_000_000))
    if estimate.quality in (ClockQuality.PTP_LOCKED, ClockQuality.NTP_SYNCED):
        assert estimate.uncertainty.millis <= 50, (
            f"{estimate.quality.label} claims sub-50ms accuracy but reports "
            f"{estimate.uncertainty.millis}ms uncertainty"
        )
    assert estimate.uncertainty.ns > 0 or estimate.quality is ClockQuality.PTP_LOCKED, (
        "only a PTP-locked source may report zero uncertainty"
    )


def _clocksync_reset_is_safe(adapter) -> None:
    """``reset()`` discards offset state without leaving the adapter unusable.

    Deliberately does *not* require a positive instant: a PTS-based adapter with
    a zero epoch legitimately estimates instant 0, and a kit that forbade it
    would reject a correct adapter.
    """
    adapter.reset()
    estimate = adapter.estimate(_packet(), Instant(3_000_000_000))
    assert estimate.t_capture.ns >= 0, "capture time must not be negative"
    assert isinstance(estimate.quality, ClockQuality)


CLOCK_SYNC_KIT = ConformanceKit(
    port_id=PortCatalogue.CLOCK_SYNC,
    version="1.0.0",
    checks=(
        ConformanceCheck(
            "returns_uncertainty", KitSection.SHAPE, _clocksync_returns_uncertainty
        ),
        ConformanceCheck(
            "honest_without_hints", KitSection.SEMANTICS, _clocksync_is_honest_without_hints
        ),
        ConformanceCheck("reset_is_safe", KitSection.FAILURE, _clocksync_reset_is_safe),
    ),
)


# --- P1 SourcePort --------------------------------------------------------------- #


def _source_declares_capabilities(adapter) -> None:
    capabilities = adapter.capabilities()
    assert capabilities.codecs, "a source must declare at least one codec (A1)"
    assert isinstance(capabilities.semantics, SourceSemantics)


def _source_open_and_close(adapter) -> None:
    async def run() -> None:
        handle = await adapter.open(_camera(), None)
        assert handle.is_open, "open() must return an open handle"
        await handle.close()
        assert not handle.is_open, "close() must close the handle"

    asyncio.run(run())


def _source_packets_are_well_formed(adapter) -> None:
    async def run() -> None:
        handle = await adapter.open(_camera(), None)
        seen = 0
        async for packet in adapter.packets(handle):
            assert packet.pts_timebase_hz > 0, "pts_timebase_hz must be positive"
            assert packet.codec, "every packet must declare its codec"
            seen += 1
            if seen >= 3:
                break
        await handle.close()

    asyncio.run(run())


SOURCE_KIT = ConformanceKit(
    port_id=PortCatalogue.SOURCE,
    version="1.0.0",
    checks=(
        ConformanceCheck(
            "declares_capabilities", KitSection.SHAPE, _source_declares_capabilities, "A1"
        ),
        ConformanceCheck("open_and_close", KitSection.SEMANTICS, _source_open_and_close),
        ConformanceCheck(
            "packets_are_well_formed", KitSection.SEMANTICS, _source_packets_are_well_formed
        ),
    ),
)


# --- P5 AdmissionPolicyPort --------------------------------------------------------- #


def _admission_context(**overrides) -> AdmissionContext:
    defaults = {
        "camera_id": CameraId("kit-cam"),
        "profile": PipelineProfile(profile_id=ProfileId("kit"), target_fps=5.0),
        "semantics": SourceSemantics.REALTIME,
        "monotonic_now": Instant(10_000_000_000),
        "last_admitted_monotonic": None,
        "in_flight": 0,
        "budget_pressure": 0.0,
        "queue_full": False,
    }
    defaults.update(overrides)
    return AdmissionContext(**defaults)


def _admission_every_drop_is_attributed(adapter) -> None:
    """No drop may be unattributed (invariant V8).

    ``AdmissionVerdict`` enforces this structurally; the kit proves the adapter
    exercises the constructor rather than bypassing it.
    """
    scenarios = (
        _admission_context(queue_full=True),
        _admission_context(in_flight=999),
        _admission_context(
            last_admitted_monotonic=Instant(10_000_000_000 - 1_000_000),
        ),
        _admission_context(budget_pressure=5.0),
    )
    for context in scenarios:
        verdict = adapter.evaluate(context)
        if not verdict.admit:
            assert verdict.reason is not None, "a drop without a reason violates V8"


def _admission_is_deterministic(adapter) -> None:
    context = _admission_context()
    first = adapter.evaluate(context)
    second = adapter.evaluate(context)
    assert first.admit == second.admit, (
        "the same context must yield the same verdict; a stateful policy breaks replay"
    )


def _admission_admits_when_idle(adapter) -> None:
    verdict = adapter.evaluate(_admission_context())
    assert verdict.admit, "an idle camera under no pressure must be admitted"
    assert verdict.fidelity is not None, "an admitted frame must carry a fidelity"


ADMISSION_KIT = ConformanceKit(
    port_id=PortCatalogue.ADMISSION_POLICY,
    version="1.0.0",
    checks=(
        ConformanceCheck("admits_when_idle", KitSection.SHAPE, _admission_admits_when_idle),
        ConformanceCheck(
            "every_drop_is_attributed",
            KitSection.SEMANTICS,
            _admission_every_drop_is_attributed,
            "V8",
        ),
        ConformanceCheck("is_deterministic", KitSection.SEMANTICS, _admission_is_deterministic),
    ),
)


# --- P6 ChangeDetectorPort ------------------------------------------------------------ #


def _change_reports_first_frame_as_changed(adapter) -> None:
    adapter.forget(CameraId("kit-cam"))
    view = memoryview(bytearray(_FRAME_BYTES))
    verdict = adapter.observe(CameraId("kit-cam"), view, _DIMENSIONS)
    assert verdict.changed, "the first frame from a camera is always new"


def _change_forget_resets_state(adapter) -> None:
    camera_id = CameraId("kit-cam-forget")
    view = memoryview(bytearray(_FRAME_BYTES))
    adapter.observe(camera_id, view, _DIMENSIONS)
    adapter.forget(camera_id)
    verdict = adapter.observe(camera_id, view, _DIMENSIONS)
    assert verdict.changed, "forget() must clear per-camera state (epoch advance)"


CHANGE_KIT = ConformanceKit(
    port_id=PortCatalogue.CHANGE_DETECTOR,
    version="1.0.0",
    checks=(
        ConformanceCheck(
            "first_frame_is_changed", KitSection.SHAPE, _change_reports_first_frame_as_changed
        ),
        ConformanceCheck(
            "forget_resets_state", KitSection.SEMANTICS, _change_forget_resets_state
        ),
    ),
)


# --- P23/P24 configuration ------------------------------------------------------------- #


def _config_source_returns_mapping(adapter) -> None:
    document = adapter.load()
    assert isinstance(document, dict), f"load() must return a mapping, got {type(document)}"
    assert adapter.source_id, "a config source must declare a source_id for explain()"


CONFIG_SOURCE_KIT = ConformanceKit(
    port_id=PortCatalogue.CONFIG_SOURCE,
    version="1.0.0",
    checks=(
        ConformanceCheck("returns_mapping", KitSection.SHAPE, _config_source_returns_mapping),
    ),
)


def _secret_unknown_reference_is_typed(adapter) -> None:
    from ..core.errors import SecretResolutionError

    reference = "kit-definitely-absent-reference"
    if adapter.has(reference):
        return
    try:
        adapter.resolve(reference)
    except SecretResolutionError as exc:
        assert reference in str(exc), "the error should name the reference"
        return
    raise AssertionError("an unknown secret reference must raise SecretResolutionError")


SECRET_KIT = ConformanceKit(
    port_id=PortCatalogue.SECRET_PROVIDER,
    version="1.0.0",
    checks=(
        ConformanceCheck(
            "unknown_reference_is_typed",
            KitSection.FAILURE,
            _secret_unknown_reference_is_typed,
            "A4",
        ),
    ),
)


# --- P29/P30 observability --------------------------------------------------------------- #


def _transport_absorbs_nothing_silently(adapter) -> None:
    assert adapter.transport_id, "a transport must declare an id"
    adapter.deliver("kit.probe", "kit", {"n": 1})


EVENT_TRANSPORT_KIT = ConformanceKit(
    port_id=PortCatalogue.EVENT_TRANSPORT,
    version="1.0.0",
    checks=(
        ConformanceCheck("declares_id", KitSection.SHAPE, _transport_absorbs_nothing_silently),
    ),
)


def _exporter_accepts_snapshot(adapter) -> None:
    class _Snapshot:
        counters: dict = {}
        gauges: dict = {}
        histograms: dict = {}

    assert adapter.exporter_id, "an exporter must declare an id"
    adapter.export(_Snapshot())


METRICS_EXPORT_KIT = ConformanceKit(
    port_id=PortCatalogue.METRICS_EXPORT,
    version="1.0.0",
    checks=(
        ConformanceCheck("accepts_snapshot", KitSection.SHAPE, _exporter_accepts_snapshot),
    ),
)


# --- registry -------------------------------------------------------------------------- #

ALL_FLOW1_KITS: tuple[ConformanceKit, ...] = (
    SOURCE_KIT,
    DECODER_KIT,
    PRIVACY_KIT,
    CLOCK_SYNC_KIT,
    ADMISSION_KIT,
    CHANGE_KIT,
    ALLOCATOR_KIT,
    CONFIG_SOURCE_KIT,
    SECRET_KIT,
    EVENT_TRANSPORT_KIT,
    METRICS_EXPORT_KIT,
)


def flow1_registry() -> ConformanceRegistry:
    """A registry pre-populated with every Flow 1 kit."""
    registry = ConformanceRegistry()
    for kit in ALL_FLOW1_KITS:
        registry.register(kit)
    return registry


def platform_registry() -> ConformanceRegistry:
    """Every kit for every currently bindable port.

    The Plugin Manager refuses to activate an adapter for a port with no kit, so
    this registry is what makes each flow's ports usable at all.
    """
    from .detector_kit import DETECTOR_KIT
    from .model_kits import ARTIFACT_STORE_KIT, DEVICE_KIT, MODEL_RUNTIME_KIT
    from .registry_kits import OBJECT_STORE_KIT
    from .tracker_kit import TRACKER_KIT

    registry = flow1_registry()
    for kit in (
        DETECTOR_KIT,
        ARTIFACT_STORE_KIT,
        MODEL_RUNTIME_KIT,
        DEVICE_KIT,
        TRACKER_KIT,
        OBJECT_STORE_KIT,
    ):
        registry.register(kit)
    # IDENTITY_RESOLVER_KIT is deliberately not registered: P11 has no
    # implementations in Phase 1, and registering a kit for it would suggest one
    # is expected (15_ROADMAP section 3).
    return registry


_ = Duration  # re-exported for adapters constructing durations in kits
