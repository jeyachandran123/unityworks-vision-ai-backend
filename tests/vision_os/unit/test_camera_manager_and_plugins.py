"""M1 Camera Manager and M17 Plugin Manager.

Camera Manager: identity stability, calibration versioning, and degradation to
normalized space when uncalibrated (invariant V9).

Plugin Manager: the conformance gate that makes invariant V3 enforceable rather
than aspirational.
"""

from __future__ import annotations

import pytest

from vision_os.acquisition import CameraManager
from vision_os.conformance import ConformanceRegistry, flow1_registry
from vision_os.conformance.kit import (
    ConformanceCheck,
    ConformanceKit,
    KitSection,
)
from vision_os.core.errors import (
    ConformanceFailedError,
    NotFoundError,
    PortIncompatibleError,
    SignatureInvalidError,
    UncalibratedError,
    ValidationError,
)
from vision_os.core.model.camera import CameraStatus
from vision_os.core.model.ids import CalibrationId, CameraId, PluginId, RegionId
from vision_os.core.model.space import Calibration, Homography, Point
from vision_os.kernel.config.schema import (
    CalibrationDeclaration,
    CameraDeclaration,
    ProfileDeclaration,
    RegionDeclaration,
)
from vision_os.kernel.plugins import (
    PluginDescriptor,
    PluginManager,
    PluginManifest,
    PortCatalogue,
    SignatureVerifier,
    VersionRange,
)

from ..conftest import SITE, TENANT

IDENTITY = Homography(((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 1.0)))


def _declarations(*, with_calibration: bool = False):
    profiles = (ProfileDeclaration(profile_id="standard", target_fps=5.0),)
    regions = (
        RegionDeclaration(
            region_id="Z3",
            label="Z3",
            vertices=((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
        ),
    )
    calibration = (
        CalibrationDeclaration(
            calibration_id="cal-v1",
            homography=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 1.0)),
            ground_uncertainty_at_unit_distance=0.05,
        )
        if with_calibration
        else None
    )
    cameras = (
        CameraDeclaration(
            camera_id="cam-01",
            tenant_id=str(TENANT),
            site_id=str(SITE),
            uri="mem://cam-01",
            transport="memory",
            source_semantics="archival",
            profile_id="standard",
            region_ids=("Z3",),
            calibration=calibration,
        ),
    )
    return cameras, profiles, regions


class TestCameraRegistry:
    def test_loads_declarations(self, camera_manager: CameraManager) -> None:
        cameras, profiles, regions = _declarations()
        camera_manager.load_declarations(
            cameras=cameras, profiles=profiles, regions=regions
        )
        camera = camera_manager.get(CameraId("cam-01"))
        assert camera.pipeline_profile.target_fps == 5.0
        assert camera.region_ids == (RegionId("Z3"),)

    def test_undeclared_profile_fails_at_startup(
        self, camera_manager: CameraManager
    ) -> None:
        """Provisioning fails fast, never at first frame."""
        cameras, _, regions = _declarations()
        with pytest.raises(ValidationError, match="undeclared profile"):
            camera_manager.load_declarations(cameras=cameras, profiles=(), regions=regions)

    def test_undeclared_region_fails_at_startup(
        self, camera_manager: CameraManager
    ) -> None:
        cameras, profiles, _ = _declarations()
        with pytest.raises(ValidationError, match="undeclared region"):
            camera_manager.load_declarations(cameras=cameras, profiles=profiles, regions=())

    def test_unknown_camera_is_typed(self, camera_manager: CameraManager) -> None:
        with pytest.raises(NotFoundError):
            camera_manager.get(CameraId("ghost"))

    def test_list_filters_by_scope(self, camera_manager: CameraManager) -> None:
        cameras, profiles, regions = _declarations()
        camera_manager.load_declarations(
            cameras=cameras, profiles=profiles, regions=regions
        )
        assert len(camera_manager.list(tenant_id=TENANT)) == 1
        assert len(camera_manager.list(site_id=SITE)) == 1
        assert len(camera_manager.list(status=CameraStatus.RETIRED)) == 0

    def test_reads_are_snapshot_isolated(self, camera_manager: CameraManager) -> None:
        """Records are immutable; a writer swaps a new version atomically."""
        cameras, profiles, regions = _declarations()
        camera_manager.load_declarations(
            cameras=cameras, profiles=profiles, regions=regions
        )
        before = camera_manager.get(CameraId("cam-01"))
        camera_manager.set_status(CameraId("cam-01"), CameraStatus.STREAMING)
        after = camera_manager.get(CameraId("cam-01"))

        assert before.status is CameraStatus.PROVISIONED
        assert after.status is CameraStatus.STREAMING

    def test_retire_marks_rather_than_deletes(self, camera_manager: CameraManager) -> None:
        cameras, profiles, regions = _declarations()
        camera_manager.load_declarations(
            cameras=cameras, profiles=profiles, regions=regions
        )
        camera_manager.retire(CameraId("cam-01"), reason="decommissioned")
        assert camera_manager.get(CameraId("cam-01")).status is CameraStatus.RETIRED

    def test_status_change_publishes_an_event(
        self, camera_manager: CameraManager, bus
    ) -> None:
        cameras, profiles, regions = _declarations()
        camera_manager.load_declarations(
            cameras=cameras, profiles=profiles, regions=regions
        )
        subscription = bus.subscribe(["camera.changed"])
        camera_manager.set_status(CameraId("cam-01"), CameraStatus.STREAMING)
        assert subscription.drain()


class TestCalibration:
    def test_uncalibrated_camera_degrades_rather_than_failing(
        self, camera_manager: CameraManager
    ) -> None:
        """Callers fall back to normalized space and omit ground fields (V9)."""
        cameras, profiles, regions = _declarations(with_calibration=False)
        camera_manager.load_declarations(
            cameras=cameras, profiles=profiles, regions=regions
        )
        with pytest.raises(UncalibratedError):
            camera_manager.project_to_ground(CameraId("cam-01"), Point(0.5, 0.5))

    def test_projection_returns_point_and_uncertainty(
        self, camera_manager: CameraManager
    ) -> None:
        cameras, profiles, regions = _declarations(with_calibration=True)
        camera_manager.load_declarations(
            cameras=cameras, profiles=profiles, regions=regions
        )
        point, ellipse = camera_manager.project_to_ground(CameraId("cam-01"), Point(0.5, 0.5))
        assert point.x == pytest.approx(5.0)
        assert ellipse.semi_major > 0

    def test_recalibration_mints_a_version_and_keeps_history(
        self, camera_manager: CameraManager
    ) -> None:
        """Historical observations stay interpretable under their own version."""
        cameras, profiles, regions = _declarations(with_calibration=True)
        camera_manager.load_declarations(
            cameras=cameras, profiles=profiles, regions=regions
        )
        camera_manager.recalibrate(
            CameraId("cam-01"),
            Calibration(CalibrationId("cal-v2"), IDENTITY, 0.02),
        )
        assert camera_manager.get_calibration(CameraId("cam-01")).calibration_id == "cal-v2"
        history = camera_manager.calibration_history(CameraId("cam-01"))
        assert [c.calibration_id for c in history] == ["cal-v1", "cal-v2"]

    def test_degenerate_calibration_is_rejected_and_previous_stays(
        self, camera_manager: CameraManager
    ) -> None:
        """A bad calibration must not blind a working camera."""
        cameras, profiles, regions = _declarations(with_calibration=True)
        camera_manager.load_declarations(
            cameras=cameras, profiles=profiles, regions=regions
        )
        singular = Homography(((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (0.0, 0.0, 1.0)))
        with pytest.raises(ValidationError, match="degenerate"):
            camera_manager.recalibrate(
                CameraId("cam-01"), Calibration(CalibrationId("bad"), singular, 0.05)
            )
        assert camera_manager.get_calibration(CameraId("cam-01")).calibration_id == "cal-v1"

    def test_viewpoint_drift_inflates_uncertainty_without_blinding(
        self, camera_manager: CameraManager, bus
    ) -> None:
        cameras, profiles, regions = _declarations(with_calibration=True)
        camera_manager.load_declarations(
            cameras=cameras, profiles=profiles, regions=regions
        )
        subscription = bus.subscribe(["camera.viewpoint_drift_suspected"])
        _, before = camera_manager.project_to_ground(CameraId("cam-01"), Point(0.5, 0.5))

        camera_manager.report_viewpoint_drift(CameraId("cam-01"), "scene registration shifted")

        assert subscription.drain()
        _, after = camera_manager.project_to_ground(CameraId("cam-01"), Point(0.5, 0.5))
        assert after.semi_major > before.semi_major
        assert camera_manager.get_calibration(CameraId("cam-01")).suspect


class TestRegionsAreOpaque:
    def test_regions_of_returns_geometry_with_uninterpreted_labels(
        self, camera_manager: CameraManager
    ) -> None:
        cameras, profiles, regions = _declarations()
        camera_manager.load_declarations(
            cameras=cameras, profiles=profiles, regions=regions
        )
        loaded = camera_manager.regions_of(CameraId("cam-01"))
        assert len(loaded) == 1
        assert loaded[0].label == "Z3"
        assert loaded[0].contains(Point(0.5, 0.5))


# --- Plugin Manager ---------------------------------------------------------- #


class _GoodAllocator:
    """Minimal conforming allocator."""

    def __init__(self) -> None:
        self._buffers = [bytearray(64) for _ in range(4)]
        self._free = list(range(4))
        self._in_use: set[int] = set()

    @property
    def location(self) -> str:
        return "host"

    def allocate(self, nbytes: int):
        from vision_os.core.errors import PoolExhaustedError

        if nbytes > 64:
            raise PoolExhaustedError("too large")
        if not self._free:
            raise PoolExhaustedError("exhausted")
        index = self._free.pop()
        self._in_use.add(index)
        return _Alloc(self._buffers[index], index)

    def release(self, allocation) -> None:
        if allocation.index in self._in_use:
            self._in_use.discard(allocation.index)
            self._free.append(allocation.index)

    def stats(self):
        return _Stats(4, len(self._in_use), 64)


class _Alloc:
    def __init__(self, buffer: bytearray, index: int) -> None:
        self._buffer = buffer
        self.index = index

    @property
    def nbytes(self) -> int:
        return len(self._buffer)

    def memory(self) -> memoryview:
        return memoryview(self._buffer)


class _Stats:
    def __init__(self, total: int, in_use: int, per_slot: int) -> None:
        self.total_slots = total
        self.in_use = in_use
        self.bytes_per_slot = per_slot


class _LeakyAllocator(_GoodAllocator):
    """Never returns slots to the pool — the slow soak failure."""

    def release(self, allocation) -> None:
        return None


def _manifest(
    plugin_id: str = "allocator.host",
    *,
    port=PortCatalogue.ALLOCATOR,
    platform_range: str = ">=1.0 <2.0",
    port_range: str = ">=1.0 <2.0",
    signature: str | None = None,
) -> PluginManifest:
    return PluginManifest(
        plugin_id=PluginId(plugin_id),
        version="1.0.0",
        port_id=port,
        port_version_range=VersionRange.parse(port_range),
        platform_range=VersionRange.parse(platform_range),
        signature=signature,
    )


class TestPluginConformanceGate:
    def test_conforming_plugin_loads_and_binds(self, plugins: PluginManager) -> None:
        plugins.register(PluginDescriptor(_manifest(), _GoodAllocator))
        loaded = plugins.load(PluginId("allocator.host"))
        assert loaded.conformance.passed
        assert plugins.activate(PluginId("allocator.host")) is not None
        assert plugins.resolve(PortCatalogue.ALLOCATOR) is not None

    def test_non_conforming_plugin_is_refused(self, plugins: PluginManager) -> None:
        """Invariant V3 as a gate in the loader, not a claim in a document."""
        plugins.register(
            PluginDescriptor(_manifest(plugin_id="allocator.leaky"), _LeakyAllocator)
        )
        with pytest.raises(ConformanceFailedError) as exc:
            plugins.load(PluginId("allocator.leaky"))
        assert exc.value.failures

    def test_conformance_failure_is_published(
        self, plugins: PluginManager, bus
    ) -> None:
        subscription = bus.subscribe(["plugin.rejected"])
        plugins.register(
            PluginDescriptor(_manifest(plugin_id="allocator.leaky"), _LeakyAllocator)
        )
        with pytest.raises(ConformanceFailedError):
            plugins.load(PluginId("allocator.leaky"))
        assert subscription.drain()

    def test_full_kit_run_covers_resource_section(self, plugins: PluginManager) -> None:
        plugins.register(PluginDescriptor(_manifest(), _GoodAllocator))
        plugins.load(PluginId("allocator.host"))
        report = plugins.run_conformance(PluginId("allocator.host"), fast_only=False)
        assert report.passed
        assert not report.skipped

    def test_fast_subset_skips_resource_checks(self, plugins: PluginManager) -> None:
        """Seconds at load, catching the catastrophic class before any real frame."""
        plugins.register(PluginDescriptor(_manifest(), _GoodAllocator))
        loaded = plugins.load(PluginId("allocator.host"))
        assert loaded.conformance.fast_subset_only
        assert loaded.conformance.skipped


class TestPluginCompatibility:
    def test_incompatible_platform_range_is_rejected(self, plugins: PluginManager) -> None:
        plugins.register(
            PluginDescriptor(_manifest(platform_range=">=9.0 <10.0"), _GoodAllocator)
        )
        with pytest.raises(PortIncompatibleError, match="outside"):
            plugins.load(PluginId("allocator.host"))

    def test_an_unimplemented_port_is_not_bindable(self, plugins: PluginManager) -> None:
        """A plugin for a port whose owning module does not exist cannot bind.

        ``PromptSourcePort`` belongs to M10, which no flow implemented — M9
        consumes prompts through a module seam instead. ``ApiTransportPort``
        became bindable in Flow 8, so this guard now tracks a port that is
        genuinely unowned rather than one merely waiting its turn.
        """
        plugins.register(
            PluginDescriptor(
                _manifest(plugin_id="prompts.git", port=PortCatalogue.PROMPT_SOURCE),
                _GoodAllocator,
            )
        )
        with pytest.raises(PortIncompatibleError, match="not bindable"):
            plugins.load(PluginId("prompts.git"))

    def test_the_embedding_port_is_never_bindable(self, plugins: PluginManager) -> None:
        """Not a frontier guard — a standing one.

        Appearance embeddings are C2 biometric data, disabled by default
        (12_SECURITY section 4.3). This must not become bindable when Flow 4
        ships, or any later flow.
        """
        plugins.register(
            PluginDescriptor(
                _manifest(plugin_id="embedding.osnet", port=PortCatalogue.EMBEDDING),
                _GoodAllocator,
            )
        )
        with pytest.raises(PortIncompatibleError, match="not bindable"):
            plugins.load(PluginId("embedding.osnet"))

    def test_unregistered_plugin_is_typed(self, plugins: PluginManager) -> None:
        from vision_os.core.errors import PluginError

        with pytest.raises(PluginError, match="not been registered"):
            plugins.load(PluginId("nope"))


class TestPluginSignatures:
    def test_unsigned_plugin_is_rejected_when_required(
        self, clock, bus, metrics
    ) -> None:
        """Unsigned code never loads (12_SECURITY §6)."""
        manager = PluginManager(
            clock=clock,
            bus=bus,
            metrics=metrics,
            conformance=flow1_registry(),
            require_signatures=True,
        )
        manager.register(PluginDescriptor(_manifest(), _GoodAllocator))
        with pytest.raises(SignatureInvalidError):
            manager.load(PluginId("allocator.host"))

    def test_untrusted_signature_is_rejected(self, clock, bus, metrics) -> None:
        manager = PluginManager(
            clock=clock,
            bus=bus,
            metrics=metrics,
            conformance=flow1_registry(),
            verifier=SignatureVerifier(trusted=frozenset({"trusted-key"})),
            require_signatures=True,
        )
        manager.register(
            PluginDescriptor(_manifest(signature="forged-key"), _GoodAllocator)
        )
        with pytest.raises(SignatureInvalidError):
            manager.load(PluginId("allocator.host"))

    def test_trusted_signature_is_accepted(self, clock, bus, metrics) -> None:
        manager = PluginManager(
            clock=clock,
            bus=bus,
            metrics=metrics,
            conformance=flow1_registry(),
            verifier=SignatureVerifier(trusted=frozenset({"trusted-key"})),
            require_signatures=True,
        )
        manager.register(
            PluginDescriptor(_manifest(signature="trusted-key"), _GoodAllocator)
        )
        assert manager.load(PluginId("allocator.host")).conformance.passed


class TestPluginWithoutKit:
    def test_port_without_a_registered_kit_cannot_activate(
        self, clock, bus, metrics
    ) -> None:
        manager = PluginManager(
            clock=clock, bus=bus, metrics=metrics, conformance=ConformanceRegistry()
        )
        manager.register(PluginDescriptor(_manifest(), _GoodAllocator))
        with pytest.raises(Exception, match="conformance kit"):
            manager.load(PluginId("allocator.host"))


class TestPluginBindingLifecycle:
    def test_swap_replaces_the_bound_adapter(self, plugins: PluginManager) -> None:
        plugins.register(PluginDescriptor(_manifest("allocator.a"), _GoodAllocator))
        plugins.register(PluginDescriptor(_manifest("allocator.b"), _GoodAllocator))
        plugins.load(PluginId("allocator.a"))
        first = plugins.activate(PluginId("allocator.a"))

        second = plugins.swap(PortCatalogue.ALLOCATOR, PluginId("allocator.b"))
        assert second is not first
        assert plugins.resolve(PortCatalogue.ALLOCATOR) is second

    def test_failed_swap_rolls_back_to_the_previous_binding(
        self, plugins: PluginManager
    ) -> None:
        """A half-applied swap is worse than an outdated adapter."""
        plugins.register(PluginDescriptor(_manifest("allocator.good"), _GoodAllocator))
        plugins.register(PluginDescriptor(_manifest("allocator.bad"), _LeakyAllocator))
        plugins.load(PluginId("allocator.good"))
        incumbent = plugins.activate(PluginId("allocator.good"))

        with pytest.raises(ConformanceFailedError):
            plugins.swap(PortCatalogue.ALLOCATOR, PluginId("allocator.bad"))

        assert plugins.resolve(PortCatalogue.ALLOCATOR) is incumbent

    def test_unload_removes_the_binding(self, plugins: PluginManager) -> None:
        from vision_os.core.errors import PluginError

        plugins.register(PluginDescriptor(_manifest(), _GoodAllocator))
        plugins.load(PluginId("allocator.host"))
        plugins.activate(PluginId("allocator.host"))
        plugins.unload(PluginId("allocator.host"))

        with pytest.raises(PluginError, match="no adapter is bound"):
            plugins.resolve(PortCatalogue.ALLOCATOR)

    def test_deactivate_clears_only_the_binding(self, plugins: PluginManager) -> None:
        plugins.register(PluginDescriptor(_manifest(), _GoodAllocator))
        plugins.load(PluginId("allocator.host"))
        plugins.activate(PluginId("allocator.host"))
        plugins.deactivate(PortCatalogue.ALLOCATOR)

        assert plugins.try_resolve(PortCatalogue.ALLOCATOR) is None
        assert len(plugins.catalogue()) == 1

    def test_try_resolve_returns_none_when_unbound(self, plugins: PluginManager) -> None:
        assert plugins.try_resolve(PortCatalogue.ALLOCATOR) is None

    def test_capabilities_are_published_for_gap_reporting(
        self, plugins: PluginManager
    ) -> None:
        """Declared capabilities are what make a capability gap detectable (V8)."""
        manifest = PluginManifest(
            plugin_id=PluginId("allocator.declared"),
            version="1.0.0",
            port_id=PortCatalogue.ALLOCATOR,
            port_version_range=VersionRange.parse(">=1.0 <2.0"),
            platform_range=VersionRange.parse(">=1.0 <2.0"),
            capabilities={"location": "host", "max_bytes": "64"},
        )
        plugins.register(PluginDescriptor(manifest, _GoodAllocator))
        plugins.load(PluginId("allocator.declared"))
        plugins.activate(PluginId("allocator.declared"))

        published = plugins.capabilities()
        assert published[PortCatalogue.ALLOCATOR]["location"] == "host"

    def test_bindings_are_reported(self, plugins: PluginManager) -> None:
        plugins.register(PluginDescriptor(_manifest(), _GoodAllocator))
        plugins.load(PluginId("allocator.host"))
        plugins.activate(PluginId("allocator.host"))
        assert plugins.bindings()[PortCatalogue.ALLOCATOR] == PluginId("allocator.host")

    def test_activate_before_load_is_typed(self, plugins: PluginManager) -> None:
        from vision_os.core.errors import PluginError

        with pytest.raises(PluginError, match="not loaded"):
            plugins.activate(PluginId("allocator.host"))

    def test_run_conformance_on_unloaded_plugin_is_typed(
        self, plugins: PluginManager
    ) -> None:
        from vision_os.core.errors import PluginError

        with pytest.raises(PluginError, match="not loaded"):
            plugins.run_conformance(PluginId("allocator.host"))

    def test_construction_failure_is_typed_and_published(
        self, plugins: PluginManager, bus
    ) -> None:
        from vision_os.core.errors import PluginError

        def explodes():
            raise RuntimeError("bad wiring")

        subscription = bus.subscribe(["plugin.rejected"])
        plugins.register(PluginDescriptor(_manifest("allocator.boom"), explodes))
        with pytest.raises(PluginError, match="failed to construct"):
            plugins.load(PluginId("allocator.boom"))
        assert subscription.drain()


class TestVersionRange:
    def test_parses_and_matches(self) -> None:
        rng = VersionRange.parse(">=1.2 <2.0")
        assert rng.contains("1.2.0")
        assert rng.contains("1.9.9")
        assert not rng.contains("2.0.0")
        assert not rng.contains("1.1.9")

    def test_malformed_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="malformed"):
            VersionRange.parse("~1.2")


class TestKitFramework:
    def test_failing_check_is_reported_with_its_obligation(self) -> None:
        def always_fails(_adapter) -> None:
            raise AssertionError("boom")

        kit = ConformanceKit(
            port_id=PortCatalogue.ALLOCATOR,
            version="1.0.0",
            checks=(
                ConformanceCheck("bad", KitSection.SEMANTICS, always_fails, obligation="D1"),
            ),
        )
        report = kit.run(object())
        assert not report.passed
        assert "[D1] semantics/bad" in report.failures[0]

    def test_unexpected_exception_is_a_failure_not_a_crash(self) -> None:
        def explodes(_adapter) -> None:
            raise RuntimeError("kaboom")

        kit = ConformanceKit(
            port_id=PortCatalogue.ALLOCATOR,
            version="1.0.0",
            checks=(ConformanceCheck("x", KitSection.SHAPE, explodes),),
        )
        report = kit.run(object())
        assert not report.passed
        assert "RuntimeError" in report.failures[0]
