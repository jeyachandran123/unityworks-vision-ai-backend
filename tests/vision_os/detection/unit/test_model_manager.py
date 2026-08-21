"""M18 Model Manager — artifacts, devices, residency, rollout, calibration.

The properties defended here fail silently otherwise: an unverified artifact that
loads anyway, a pinned model evicted under pressure, a fallback that becomes
permanent because nobody noticed, and a GPU that disappears and takes the
platform with it.
"""

from __future__ import annotations

import hashlib

import pytest

from vision_os.adapters.models import (
    CpuDeviceProvider,
    InMemoryArtifactStore,
    LocalArtifactStore,
    ScriptedRuntime,
    StaticDeviceProvider,
    compute_hash,
)
from vision_os.core.errors import (
    ArtifactIntegrityError,
    ArtifactUnavailableError,
    DeviceOutOfMemoryError,
    DeviceUnavailableError,
    LicenceViolationError,
    ModelLoadError,
    ModelUnavailableError,
    ValidationError,
)
from vision_os.core.model.ids import CalibrationId, ModelId
from vision_os.core.ports.models import ArtifactRef, DeviceInfo, DeviceKind
from vision_os.kernel.models import (
    CalibrationMethod,
    CalibrationProfile,
    DeviceBroker,
    ModelManager,
    ModelSpec,
    RolloutMode,
)

from ..conftest import MODEL_ID, register_reference_model

GIGABYTE = 1024**3


class TestArtifactVerification:
    def test_hash_mismatch_fails_closed(self, tmp_path) -> None:
        """A supply-chain event, not a network glitch. Never retried."""
        store = LocalArtifactStore(tmp_path / "cache")
        artifact = tmp_path / "weights.bin"
        artifact.write_bytes(b"real-weights")
        wrong = "blake2b:" + hashlib.blake2b(b"other", digest_size=32).hexdigest()

        with pytest.raises(ArtifactIntegrityError, match="supply-chain"):
            store.fetch(ArtifactRef(uri=f"file://{artifact}", expected_hash=wrong))

    def test_verified_artifact_is_cached_by_content(self, tmp_path) -> None:
        """Cached by hash, so an artifact is fetched once per node ever."""
        store = LocalArtifactStore(tmp_path / "cache")
        artifact = tmp_path / "weights.bin"
        artifact.write_bytes(b"real-weights")
        ref = ArtifactRef(
            uri=f"file://{artifact}", expected_hash=compute_hash(artifact)
        )

        first = store.fetch(ref)
        assert store.has(ref)
        artifact.unlink()
        assert store.fetch(ref) == first, "a cached artifact survives the source"

    def test_missing_artifact_is_transient(self, tmp_path) -> None:
        store = LocalArtifactStore(tmp_path / "cache")
        with pytest.raises(ArtifactUnavailableError) as exc:
            store.fetch(ArtifactRef(uri="file:///absent.bin", expected_hash="x"))
        assert exc.value.retryable

    def test_artifact_ref_requires_a_hash(self) -> None:
        """Loading unverified weights is a supply-chain vulnerability."""
        with pytest.raises(ValueError, match="expected hash"):
            ArtifactRef(uri="file:///x.bin", expected_hash="")


class TestDeviceBroker:
    def test_prefers_the_least_utilized_accelerator(self, broker: DeviceBroker) -> None:
        """Keeps a multi-GPU node balanced without a scheduler."""
        first = broker.reserve(owner="a", bytes_required=6 * GIGABYTE)
        second = broker.reserve(owner="b", bytes_required=1 * GIGABYTE)
        assert first.device_id != second.device_id

    def test_refuses_overcommit_rather_than_discovering_it(
        self, broker: DeviceBroker
    ) -> None:
        """Discovering an OOM mid-inference looks like a random error."""
        broker.reserve(owner="a", bytes_required=7 * GIGABYTE, pinned=True)
        broker.reserve(owner="b", bytes_required=7 * GIGABYTE, pinned=True)
        reservation = broker.reserve(owner="c", bytes_required=7 * GIGABYTE)
        assert reservation.device_id == "cpu", "CPU fallback absorbs the overflow"

    def test_denies_when_cpu_fallback_is_disabled(
        self, cpu_devices, gpu_devices
    ) -> None:
        strict = DeviceBroker((gpu_devices,), allow_cpu_fallback=False)
        strict.reserve(owner="a", bytes_required=7 * GIGABYTE, pinned=True)
        strict.reserve(owner="b", bytes_required=7 * GIGABYTE, pinned=True)
        with pytest.raises(DeviceOutOfMemoryError, match="no device can satisfy"):
            strict.reserve(owner="c", bytes_required=7 * GIGABYTE)

    def test_pinned_reservations_are_never_evicted(self, gpu_devices) -> None:
        """An operator who pinned a model meant it."""
        strict = DeviceBroker((gpu_devices,), allow_cpu_fallback=False)
        pinned = strict.reserve(
            owner="pinned", bytes_required=7 * GIGABYTE, pinned=True
        )
        strict.reserve(owner="other", bytes_required=7 * GIGABYTE, pinned=True)
        with pytest.raises(DeviceOutOfMemoryError):
            strict.reserve(owner="greedy", bytes_required=7 * GIGABYTE)
        assert pinned in strict.reservations_on(pinned.device_id)

    def test_non_pinned_reservations_are_evicted_to_make_room(
        self, gpu_devices
    ) -> None:
        strict = DeviceBroker((gpu_devices,), allow_cpu_fallback=False)
        strict.reserve(owner="evictable", bytes_required=7 * GIGABYTE, pinned=False)
        strict.reserve(owner="evictable2", bytes_required=7 * GIGABYTE, pinned=False)
        granted = strict.reserve(owner="urgent", bytes_required=7 * GIGABYTE)
        assert granted.owner == "urgent"

    def test_headroom_is_respected(self) -> None:
        """Filling a card to the last byte is how inference OOMs.

        A single-card broker, so the assertion is about headroom rather than
        about a second GPU absorbing the request.
        """
        single = StaticDeviceProvider(
            (DeviceInfo("cuda:0", DeviceKind.CUDA, 0, 8 * GIGABYTE, "solo"),)
        )
        strict = DeviceBroker(
            (single,), allow_cpu_fallback=False, headroom_fraction=0.5
        )
        strict.reserve(owner="a", bytes_required=4 * GIGABYTE, pinned=True)
        with pytest.raises(DeviceOutOfMemoryError):
            strict.reserve(owner="b", bytes_required=1 * GIGABYTE)

    def test_release_is_idempotent(self, broker: DeviceBroker) -> None:
        reservation = broker.reserve(owner="a", bytes_required=GIGABYTE)
        broker.release(reservation)
        broker.release(reservation)
        assert broker.report().total_reserved_bytes == 0

    def test_invalid_headroom_is_rejected(self, cpu_devices) -> None:
        with pytest.raises(ValueError, match="headroom_fraction"):
            DeviceBroker((cpu_devices,), headroom_fraction=1.0)


class TestGpuLossAndFallback:
    def test_a_disappeared_device_becomes_unselectable(
        self, cpu_devices, gpu_devices
    ) -> None:
        """Migration is the Model Manager's decision; the broker just stops
        offering the device (invariant V9)."""
        broker = DeviceBroker((cpu_devices, gpu_devices), allow_cpu_fallback=True)
        assert broker.is_available("cuda:0")

        gpu_devices.remove("cuda:0")
        gpu_devices.remove("cuda:1")
        broker.refresh()

        assert not broker.is_available("cuda:0")
        reservation = broker.reserve(owner="detector", bytes_required=GIGABYTE)
        assert reservation.device_id == "cpu", "the platform degrades rather than fails"

    def test_explicit_hint_for_a_dead_device_falls_back(
        self, cpu_devices, gpu_devices
    ) -> None:
        broker = DeviceBroker((cpu_devices, gpu_devices), allow_cpu_fallback=True)
        gpu_devices.remove("cuda:0")
        gpu_devices.remove("cuda:1")
        broker.refresh()
        reservation = broker.reserve(
            owner="detector", bytes_required=GIGABYTE, device_hint="cuda:0"
        )
        assert reservation.device_id == "cpu"

    def test_dead_hint_without_fallback_is_explicit(
        self, cpu_devices, gpu_devices
    ) -> None:
        """A site that forbade CPU fallback learns the capability is gone."""
        broker = DeviceBroker((cpu_devices, gpu_devices), allow_cpu_fallback=False)
        gpu_devices.remove("cuda:0")
        broker.refresh()
        with pytest.raises(DeviceUnavailableError, match="CPU fallback is disabled"):
            broker.reserve(
                owner="detector", bytes_required=GIGABYTE, device_hint="cuda:0"
            )

    def test_cpu_only_node_starts_normally(self, cpu_only_broker: DeviceBroker) -> None:
        """An edge box with no accelerator is the common case, not an error."""
        reservation = cpu_only_broker.reserve(owner="detector", bytes_required=GIGABYTE)
        assert reservation.device_id == "cpu"
        assert cpu_only_broker.report().accelerators_available == 0

    def test_a_broken_provider_does_not_blind_the_broker(self, cpu_devices) -> None:
        class ExplodingProvider:
            provider_id = "exploding"

            def enumerate(self):
                raise RuntimeError("driver fault")

            def is_available(self, device_id: str) -> bool:
                return False

            def utilization(self, device_id: str) -> float:
                return 0.0

        broker = DeviceBroker((cpu_devices, ExplodingProvider()))
        assert broker.reserve(owner="a", bytes_required=1).device_id == "cpu"

    def test_cuda_provider_is_absent_not_fatal(self) -> None:
        """A node without torch enumerates nothing and starts normally."""
        from vision_os.adapters.models import CudaDeviceProvider

        provider = CudaDeviceProvider()
        devices = provider.enumerate()
        assert devices is not None
        assert not provider.is_available("cuda:99")


class TestResidency:
    def test_acquire_loads_warms_and_refcounts(
        self, models: ModelManager, artifacts: InMemoryArtifactStore
    ) -> None:
        register_reference_model(models, artifacts)
        first = models.acquire(MODEL_ID, "1.0.0", owner="a")
        second = models.acquire(MODEL_ID, "1.0.0", owner="b")

        assert first.artifact_hash == second.artifact_hash
        assert first.load_state == "warm"
        assert models.is_resident(MODEL_ID, "1.0.0")

        models.release(first)
        assert not models.evict(MODEL_ID, "1.0.0"), "refcount still held"
        models.release(second)
        assert models.evict(MODEL_ID, "1.0.0")

    def test_second_acquire_reuses_the_resident_model(
        self, models: ModelManager, artifacts: InMemoryArtifactStore, runtime_adapter
    ) -> None:
        """Duplicate weights per consumer is how a GPU runs out at ten cameras."""
        register_reference_model(models, artifacts)
        models.acquire(MODEL_ID, "1.0.0", owner="a")
        models.acquire(MODEL_ID, "1.0.0", owner="b")
        assert len(runtime_adapter.loaded) == 1

    def test_unregistered_model_is_typed(self, models: ModelManager) -> None:
        with pytest.raises(ModelUnavailableError, match="not registered"):
            models.acquire(ModelId("ghost"), "1.0.0")

    def test_load_failure_marks_the_version_bad(
        self, clock, bus, metrics, broker, artifacts
    ) -> None:
        """The next resolve falls back to last known-good rather than retrying."""

        class FailingRuntime(ScriptedRuntime):
            def load(self, **kwargs):
                raise RuntimeError("corrupt weights")

        manager = ModelManager(
            clock=clock,
            bus=bus,
            metrics=metrics,
            broker=broker,
            artifacts=artifacts,
            runtimes=(FailingRuntime(),),
        )
        register_reference_model(manager, artifacts)
        with pytest.raises(ModelLoadError, match="failed to load"):
            manager.acquire(MODEL_ID, "1.0.0")
        with pytest.raises(ModelUnavailableError, match="marked bad"):
            manager.acquire(MODEL_ID, "1.0.0")

    def test_no_runtime_supports_the_artifact(
        self, clock, bus, metrics, broker, artifacts
    ) -> None:
        class PickyRuntime(ScriptedRuntime):
            def supports(self, artifact_path: str, precision: str) -> bool:
                return False

        manager = ModelManager(
            clock=clock,
            bus=bus,
            metrics=metrics,
            broker=broker,
            artifacts=artifacts,
            runtimes=(PickyRuntime(),),
        )
        register_reference_model(manager, artifacts)
        with pytest.raises(ModelLoadError, match="no registered runtime"):
            manager.acquire(MODEL_ID, "1.0.0")

    def test_integrity_failure_marks_the_version_bad(
        self, clock, bus, metrics, broker, runtime_adapter
    ) -> None:
        store = InMemoryArtifactStore()
        store.put("mem://bad.bin", b"real")
        manager = ModelManager(
            clock=clock,
            bus=bus,
            metrics=metrics,
            broker=broker,
            artifacts=store,
            runtimes=(runtime_adapter,),
        )
        manager.register(
            ModelSpec(
                model_id=MODEL_ID,
                version="1.0.0",
                artifact=ArtifactRef(uri="mem://bad.bin", expected_hash="blake2b:wrong"),
            )
        )
        with pytest.raises(ArtifactIntegrityError):
            manager.acquire(MODEL_ID, "1.0.0")

    def test_close_evicts_everything(
        self, models: ModelManager, artifacts: InMemoryArtifactStore
    ) -> None:
        register_reference_model(models, artifacts)
        models.acquire(MODEL_ID, "1.0.0", owner="a")
        models.close()
        assert not models.is_resident(MODEL_ID, "1.0.0")


class TestLicensing:
    def test_forbidden_context_is_refused_at_registration(
        self, models: ModelManager, artifacts: InMemoryArtifactStore
    ) -> None:
        """Checked here, never discovered in production."""
        digest = artifacts.put("mem://restricted.bin", b"w")
        with pytest.raises(LicenceViolationError, match="does not permit"):
            models.register(
                ModelSpec(
                    model_id=ModelId("restricted"),
                    version="1.0.0",
                    artifact=ArtifactRef(
                        uri="mem://restricted.bin", expected_hash=digest
                    ),
                    licence="research-only",
                    permitted_contexts=("research",),
                )
            )

    def test_permitted_context_is_accepted(
        self, models: ModelManager, artifacts: InMemoryArtifactStore
    ) -> None:
        register_reference_model(models, artifacts, permitted_contexts=("on_premise",))
        assert models.spec(MODEL_ID, "1.0.0")


class TestRollout:
    def test_pin_binds_a_role(
        self, models: ModelManager, artifacts: InMemoryArtifactStore
    ) -> None:
        register_reference_model(models, artifacts)
        models.pin("primary_detector", MODEL_ID, "1.0.0")
        assert models.resolve("primary_detector") == (MODEL_ID, "1.0.0")

    def test_swap_publishes_so_a_fallback_is_never_silent(
        self, models: ModelManager, artifacts: InMemoryArtifactStore, bus
    ) -> None:
        """A fallback nobody notices becomes permanent."""
        register_reference_model(models, artifacts, version="1.0.0")
        register_reference_model(models, artifacts, version="2.0.0")
        subscription = bus.subscribe(["model.swapped"])

        models.pin("primary_detector", MODEL_ID, "1.0.0")
        models.pin("primary_detector", MODEL_ID, "2.0.0")

        events = subscription.drain()
        assert events
        assert events[0].previous.endswith("1.0.0")

    def test_canary_routing_is_deterministic(
        self, models: ModelManager, artifacts: InMemoryArtifactStore
    ) -> None:
        """A random draw would make a replay produce a different split (V13)."""
        register_reference_model(models, artifacts, version="1.0.0")
        register_reference_model(models, artifacts, version="2.0.0")
        models.pin("primary_detector", MODEL_ID, "1.0.0")
        models.canary("primary_detector", MODEL_ID, "2.0.0", 0.25)

        resolved = [models.resolve("primary_detector")[1] for _ in range(12)]
        assert resolved.count("2.0.0") == 3
        assert resolved.count("1.0.0") == 9

    def test_shadow_candidate_never_becomes_the_resolved_model(
        self, models: ModelManager, artifacts: InMemoryArtifactStore
    ) -> None:
        """Shadow results must never reach platform state."""
        register_reference_model(models, artifacts, version="1.0.0")
        register_reference_model(models, artifacts, version="2.0.0")
        models.pin("primary_detector", MODEL_ID, "1.0.0")
        models.shadow("primary_detector", MODEL_ID, "2.0.0")

        for _ in range(10):
            assert models.resolve("primary_detector") == (MODEL_ID, "1.0.0")
        assert models.shadow_candidate("primary_detector") == (MODEL_ID, "2.0.0")
        assert models.role_binding("primary_detector").mode is RolloutMode.SHADOW

    def test_rollback_returns_to_the_baseline(
        self, models: ModelManager, artifacts: InMemoryArtifactStore
    ) -> None:
        """Mean time to recovery is a config revert, not a redeploy."""
        register_reference_model(models, artifacts, version="1.0.0")
        register_reference_model(models, artifacts, version="2.0.0")
        models.pin("primary_detector", MODEL_ID, "1.0.0")
        models.canary("primary_detector", MODEL_ID, "2.0.0", 1.0)
        models.rollback("primary_detector")

        for _ in range(5):
            assert models.resolve("primary_detector") == (MODEL_ID, "1.0.0")

    def test_canary_without_a_baseline_is_rejected(
        self, models: ModelManager, artifacts: InMemoryArtifactStore
    ) -> None:
        register_reference_model(models, artifacts)
        with pytest.raises(ValidationError, match="no baseline"):
            models.canary("unbound_role", MODEL_ID, "1.0.0", 0.5)

    def test_unbound_role_is_typed(self, models: ModelManager) -> None:
        with pytest.raises(ModelUnavailableError, match="no model bound"):
            models.resolve("nothing_here")


class TestCalibration:
    def test_absent_profile_is_none_not_identity(self, models: ModelManager) -> None:
        """Uncalibrated is stated, never papered over with an identity transform."""
        assert models.calibration(MODEL_ID, "1.0.0") is None

    def test_temperature_scaling_is_monotone(self) -> None:
        profile = CalibrationProfile(
            calibration_id=CalibrationId("cal"),
            model_id=MODEL_ID,
            model_version="1.0.0",
            method=CalibrationMethod.TEMPERATURE,
            temperature=2.0,
        )
        values = [profile.apply(score / 10) for score in range(11)]
        assert values == sorted(values)
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_temperature_handles_the_boundaries(self) -> None:
        """A raw score of exactly 0 or 1 is the input a detector most often emits."""
        profile = CalibrationProfile(
            calibration_id=CalibrationId("cal"),
            model_id=MODEL_ID,
            model_version="1.0.0",
            method=CalibrationMethod.TEMPERATURE,
            temperature=1.5,
        )
        assert 0.0 <= profile.apply(0.0) <= 1.0
        assert 0.0 <= profile.apply(1.0) <= 1.0

    def test_piecewise_interpolates(self) -> None:
        profile = CalibrationProfile(
            calibration_id=CalibrationId("cal"),
            model_id=MODEL_ID,
            model_version="1.0.0",
            method=CalibrationMethod.PIECEWISE,
            knots=((0.0, 0.0), (0.5, 0.2), (1.0, 1.0)),
        )
        assert profile.apply(0.25) == pytest.approx(0.1)
        assert profile.apply(0.75) == pytest.approx(0.6)

    def test_piecewise_requires_ascending_knots(self) -> None:
        with pytest.raises(ValueError, match="ascend"):
            CalibrationProfile(
                calibration_id=CalibrationId("cal"),
                model_id=MODEL_ID,
                model_version="1.0.0",
                method=CalibrationMethod.PIECEWISE,
                knots=((0.5, 0.2), (0.1, 0.0)),
            )

    def test_registered_profile_is_returned(self, models: ModelManager) -> None:
        profile = CalibrationProfile(
            calibration_id=CalibrationId("cal-1"),
            model_id=MODEL_ID,
            model_version="1.0.0",
            fitted_on="site-sg-01 validation set",
        )
        models.register_calibration(profile)
        assert models.calibration(MODEL_ID, "1.0.0") is profile


class TestDeviceReporting:
    def test_report_exposes_utilization(
        self, models: ModelManager, artifacts: InMemoryArtifactStore
    ) -> None:
        register_reference_model(models, artifacts, vram_bytes=2 * GIGABYTE)
        models.acquire(MODEL_ID, "1.0.0", owner="a")
        report = models.device_report()
        assert report.total_reserved_bytes >= 2 * GIGABYTE

    def test_residency_report_lists_resident_models(
        self, models: ModelManager, artifacts: InMemoryArtifactStore
    ) -> None:
        register_reference_model(models, artifacts)
        models.acquire(MODEL_ID, "1.0.0", owner="a")
        report = models.residency_report()
        assert any(entry[0] == str(MODEL_ID) for entry in report.resident)


def test_cpu_provider_always_offers_one_device() -> None:
    provider = CpuDeviceProvider()
    devices = provider.enumerate()
    assert len(devices) == 1
    assert devices[0].kind is DeviceKind.CPU
    assert provider.is_available("cpu")


def test_static_provider_supports_restore(gpu_devices: StaticDeviceProvider) -> None:
    gpu_devices.remove("cuda:0")
    assert not gpu_devices.is_available("cuda:0")
    gpu_devices.restore("cuda:0")
    assert gpu_devices.is_available("cuda:0")


def test_device_info_rejects_negative_memory() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DeviceInfo("cuda:0", DeviceKind.CUDA, 0, -1)
