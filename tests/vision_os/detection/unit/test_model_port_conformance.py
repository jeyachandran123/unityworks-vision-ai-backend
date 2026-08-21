"""Every shipped P25/P26/P27 adapter is run through its kit.

A conformance kit that is registered but never executed proves nothing — it is a
document that looks like a gate. These tests run ``ARTIFACT_STORE_KIT``,
``MODEL_RUNTIME_KIT`` and ``DEVICE_KIT`` against every adapter the platform
ships, and then run each kit against a deliberately broken adapter to prove the
kit *fails* when it should.

The second half matters more than the first. A kit that passes everything it is
shown is indistinguishable from no kit at all, so each obligation here is paired
with a fixture that violates exactly that obligation and nothing else.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest

from vision_os.adapters.models import (
    CpuDeviceProvider,
    CudaDeviceProvider,
    InMemoryArtifactStore,
    LocalArtifactStore,
    ScriptedRuntime,
    StaticDeviceProvider,
    UltralyticsRuntime,
)
from vision_os.conformance import (
    ARTIFACT_STORE_KIT,
    DEVICE_KIT,
    MODEL_RUNTIME_KIT,
)
from vision_os.core.errors import ArtifactIntegrityError, ArtifactUnavailableError
from vision_os.core.ports.models import ArtifactRef, DeviceInfo, DeviceKind, LoadedModel


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "artifacts"
    directory.mkdir()
    return directory


# --- the shipped adapters all conform ------------------------------------------ #


class TestShippedArtifactStores:
    def test_local_store_passes_its_kit(self, cache_dir: Path) -> None:
        report = ARTIFACT_STORE_KIT.run(LocalArtifactStore(cache_dir))
        assert report.passed, report.failures

    def test_in_memory_store_passes_its_kit(self) -> None:
        report = ARTIFACT_STORE_KIT.run(InMemoryArtifactStore())
        assert report.passed, report.failures

    def test_kit_executes_every_check(self, cache_dir: Path) -> None:
        report = ARTIFACT_STORE_KIT.run(LocalArtifactStore(cache_dir))
        assert len(report.executed) == len(ARTIFACT_STORE_KIT.checks)
        assert not report.skipped


class TestShippedRuntimes:
    def test_scripted_runtime_passes_its_kit(self) -> None:
        report = MODEL_RUNTIME_KIT.run(ScriptedRuntime())
        assert report.passed, report.failures

    def test_ultralytics_runtime_passes_its_kit(self) -> None:
        """Passes without ultralytics installed.

        ``supports`` answers False for the kit's scratch artifact, so the load
        checks return early. That is the intended shape: the kit verifies the
        *contract*, and a runtime that cannot serve an artifact says so rather
        than failing to be asked.
        """
        report = MODEL_RUNTIME_KIT.run(UltralyticsRuntime())
        assert report.passed, report.failures


class TestShippedDeviceProviders:
    def test_cpu_provider_passes_its_kit(self) -> None:
        report = DEVICE_KIT.run(CpuDeviceProvider())
        assert report.passed, report.failures

    def test_cuda_provider_passes_its_kit(self) -> None:
        """Passes on a machine with no CUDA — that is the point.

        The provider must degrade to an empty inventory (V9), not raise, so the
        kit is meaningful precisely on the CPU-only CI box.
        """
        report = DEVICE_KIT.run(CudaDeviceProvider())
        assert report.passed, report.failures

    def test_static_provider_passes_its_kit(self) -> None:
        devices = (
            DeviceInfo(
                device_id="cuda:0",
                kind=DeviceKind.CUDA,
                index=0,
                total_memory_bytes=8 * 1024**3,
                name="scripted",
                available=True,
            ),
        )
        report = DEVICE_KIT.run(StaticDeviceProvider(devices))
        assert report.passed, report.failures

    def test_static_provider_still_conforms_after_a_device_disappears(self) -> None:
        provider = StaticDeviceProvider(
            (
                DeviceInfo(
                    device_id="cuda:0",
                    kind=DeviceKind.CUDA,
                    index=0,
                    total_memory_bytes=8 * 1024**3,
                    name="scripted",
                    available=True,
                ),
            )
        )
        provider.remove("cuda:0")
        report = DEVICE_KIT.run(provider)
        assert report.passed, report.failures
        assert provider.is_available("cuda:0") is False


# --- the kits reject adapters that break their obligations ---------------------- #


class _UnverifyingStore:
    """Returns bytes without checking the declared hash — the supply-chain hole."""

    def __init__(self, payload: bytes = b"whatever") -> None:
        self._payload = payload

    @property
    def store_id(self) -> str:
        return "unverifying"

    def has(self, ref: ArtifactRef) -> bool:
        return True

    def fetch(self, ref: ArtifactRef) -> str:
        return "/some/path/that/was/never/checked"


class _AnonymousStore(_UnverifyingStore):
    @property
    def store_id(self) -> str:
        return ""


class _RaisingRuntime:
    """``supports`` raises instead of answering."""

    @property
    def runtime_id(self) -> str:
        return "raising"

    def supports(self, artifact_path: str, precision: str) -> bool:
        if not artifact_path:
            raise ValueError("no artifact path")
        return True

    def load(self, **kwargs) -> LoadedModel:
        raise NotImplementedError

    def unload(self, loaded: LoadedModel) -> None:
        return None


class _AmnesiacRuntime:
    """Loads, but forgets which weights it loaded (obligation A3)."""

    @property
    def runtime_id(self) -> str:
        return "amnesiac"

    def supports(self, artifact_path: str, precision: str) -> bool:
        return True

    def load(
        self,
        *,
        model_id: str,
        version: str,
        artifact_path: str,
        artifact_hash: str,
        device_id: str,
        precision: str,
        options=None,
    ) -> LoadedModel:
        return LoadedModel(
            model_id=model_id,
            version=version,
            artifact_hash="",  # the defect under test
            device_id=device_id,
            precision=precision,
            session=object(),
            runtime_id=self.runtime_id,
            load_ms=1.0,
        )

    def unload(self, loaded: LoadedModel) -> None:
        return None


class _FaultingUnloadRuntime(_AmnesiacRuntime):
    """Correct provenance, but a second unload faults — breaks rollback."""

    def __init__(self) -> None:
        self._unloaded: set[int] = set()

    @property
    def runtime_id(self) -> str:
        return "faulting-unload"

    def load(self, **kwargs) -> LoadedModel:
        return LoadedModel(
            model_id=kwargs["model_id"],
            version=kwargs["version"],
            artifact_hash=kwargs["artifact_hash"],
            device_id=kwargs["device_id"],
            precision=kwargs["precision"],
            session=object(),
            runtime_id=self.runtime_id,
            load_ms=1.0,
        )

    def unload(self, loaded: LoadedModel) -> None:
        key = id(loaded.session)
        if key in self._unloaded:
            raise RuntimeError("already unloaded")
        self._unloaded.add(key)


class _ThrowingDeviceProvider:
    """Raises when asked about a device that is gone — takes the platform with it."""

    @property
    def provider_id(self) -> str:
        return "throwing"

    def enumerate(self) -> Sequence[DeviceInfo]:
        return ()

    def is_available(self, device_id: str) -> bool:
        raise RuntimeError("device query failed")

    def utilization(self, device_id: str) -> float:
        return 0.0


class _OverUtilizedDeviceProvider:
    """Reports utilization outside [0,1], which would corrupt broker arithmetic."""

    @property
    def provider_id(self) -> str:
        return "over-utilized"

    def enumerate(self) -> Sequence[DeviceInfo]:
        return (
            DeviceInfo(
                device_id="cuda:0",
                kind=DeviceKind.CUDA,
                index=0,
                total_memory_bytes=1024,
                name="hot",
                available=True,
            ),
        )

    def is_available(self, device_id: str) -> bool:
        return True

    def utilization(self, device_id: str) -> float:
        return 3.7


def _failed_checks(report) -> set[str]:
    """The check names that failed, stripped of obligation prefix and message."""
    names = set()
    for failure in report.failures:
        head = failure.split(":", 1)[0]
        names.add(head.split("] ")[-1])
    return names


class TestKitsRejectBrokenAdapters:
    """A kit that passes everything is indistinguishable from no kit."""

    def test_unverified_artifact_is_caught(self) -> None:
        report = ARTIFACT_STORE_KIT.run(_UnverifyingStore())
        assert not report.passed
        assert "semantics/hash_mismatch_fails_closed" in _failed_checks(report)

    def test_missing_store_id_is_caught(self) -> None:
        report = ARTIFACT_STORE_KIT.run(_AnonymousStore())
        assert not report.passed
        assert "shape/declares_id" in _failed_checks(report)

    def test_partial_failure_still_runs_remaining_checks(self) -> None:
        """One failure must not abort the kit — the full picture is the point."""
        report = ARTIFACT_STORE_KIT.run(_AnonymousStore())
        assert len(report.executed) == len(ARTIFACT_STORE_KIT.checks)

    def test_raising_supports_is_caught(self) -> None:
        report = MODEL_RUNTIME_KIT.run(_RaisingRuntime())
        assert not report.passed
        assert "semantics/supports_is_total" in _failed_checks(report)

    def test_lost_provenance_is_caught(self) -> None:
        report = MODEL_RUNTIME_KIT.run(_AmnesiacRuntime())
        assert not report.passed
        assert "semantics/load_reports_provenance" in _failed_checks(report)

    def test_non_idempotent_unload_is_caught(self) -> None:
        report = MODEL_RUNTIME_KIT.run(_FaultingUnloadRuntime())
        assert not report.passed
        assert "failure/unload_is_idempotent" in _failed_checks(report)

    def test_raising_device_query_is_caught(self) -> None:
        report = DEVICE_KIT.run(_ThrowingDeviceProvider())
        assert not report.passed
        assert "failure/disappearance_is_reported_not_raised" in _failed_checks(report)

    def test_unbounded_utilization_is_caught(self) -> None:
        report = DEVICE_KIT.run(_OverUtilizedDeviceProvider())
        assert not report.passed
        assert "semantics/utilization_is_bounded" in _failed_checks(report)


class TestKitReportShape:
    def test_summary_names_the_port_and_outcome(self, cache_dir: Path) -> None:
        report = ARTIFACT_STORE_KIT.run(LocalArtifactStore(cache_dir))
        summary = report.summary()
        assert "PASS" in summary
        assert str(ARTIFACT_STORE_KIT.port_id) in summary

    def test_failure_message_identifies_the_obligation(self) -> None:
        report = ARTIFACT_STORE_KIT.run(_UnverifyingStore())
        joined = " ".join(report.failures)
        assert "A4" in joined, "a failure must name the obligation it breaks"

    def test_fast_subset_skips_nothing_these_kits_defer(self, cache_dir: Path) -> None:
        """All model-kit checks are in the fast subset, so load-time gating is total."""
        report = ARTIFACT_STORE_KIT.run(LocalArtifactStore(cache_dir), fast_only=True)
        assert not report.skipped
        assert report.fast_subset_only is True


class TestArtifactVerificationIsReal:
    """The kit checks the contract; these check the shipped implementation."""

    def test_local_store_accepts_a_correct_hash(self, cache_dir: Path, tmp_path: Path) -> None:
        source = tmp_path / "weights.bin"
        source.write_bytes(b"real-weights")
        digest = hashlib.blake2b(b"real-weights", digest_size=32).hexdigest()
        store = LocalArtifactStore(cache_dir)

        path = store.fetch(ArtifactRef(uri=f"file://{source}", expected_hash=f"blake2b:{digest}"))

        assert Path(path).read_bytes() == b"real-weights"

    def test_local_store_caches_by_content_not_name(
        self, cache_dir: Path, tmp_path: Path
    ) -> None:
        digest = "blake2b:" + hashlib.blake2b(b"shared", digest_size=32).hexdigest()
        first = tmp_path / "a.bin"
        second = tmp_path / "b.bin"
        first.write_bytes(b"shared")
        second.write_bytes(b"shared")
        store = LocalArtifactStore(cache_dir)

        path_a = store.fetch(ArtifactRef(uri=f"file://{first}", expected_hash=digest))
        path_b = store.fetch(ArtifactRef(uri=f"file://{second}", expected_hash=digest))

        assert path_a == path_b, "identical weights must occupy one cache slot"
        assert len(list(cache_dir.iterdir())) == 1

    def test_missing_source_is_unavailable_not_integrity(self, cache_dir: Path) -> None:
        """The distinction matters: unavailable is transient, integrity is terminal."""
        store = LocalArtifactStore(cache_dir)
        with pytest.raises(ArtifactUnavailableError):
            store.fetch(
                ArtifactRef(uri="file:///no/such/artifact.bin", expected_hash="blake2b:00")
            )

    def test_in_memory_store_round_trips_its_own_hash(self) -> None:
        store = InMemoryArtifactStore()
        digest = store.put("mem://weights", b"payload")
        assert store.fetch(ArtifactRef(uri="mem://weights", expected_hash=digest))

    def test_verification_can_be_disabled_only_explicitly(
        self, cache_dir: Path, tmp_path: Path
    ) -> None:
        """An escape hatch that must be asked for by name, never a default.

        The paired assertion is the important one: the *same* fetch that the lax
        store allows is refused by a default-constructed store, so the safety
        comes from the default rather than from the caller remembering.
        """
        source = tmp_path / "unverified.bin"
        source.write_bytes(b"content")
        bad_ref = ArtifactRef(uri=f"file://{source}", expected_hash="blake2b:wrong")

        lax = LocalArtifactStore(cache_dir, verify=False)
        assert Path(lax.fetch(bad_ref)).exists()

        strict = LocalArtifactStore(cache_dir / "strict")
        with pytest.raises(ArtifactIntegrityError) as caught:
            strict.fetch(bad_ref)
        assert "supply-chain" in str(caught.value), (
            "the refusal must say why, so an operator does not 'fix' it by disabling checks"
        )
