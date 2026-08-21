"""Conformance kits for the model ports — P25, P26, P27.

Smaller than ``kit.detector`` because these ports carry less semantic weight, but
each guards one thing that fails silently otherwise:

``artifact/hash_mismatch_fails_closed``
    An artifact store that returns unverified bytes is a supply-chain hole that
    behaves perfectly right up until it does not.

``device/disappearance_is_reported_not_raised``
    A provider that throws when a card is pulled takes the platform down with it,
    turning a degradation into an outage (invariant V9).

``runtime/unload_is_idempotent``
    Double-unload during a rollback is routine; a runtime that faults on it turns
    a safe rollback into a crash.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from ..core.errors import ArtifactIntegrityError, ArtifactUnavailableError
from ..core.ports.models import ArtifactRef
from ..kernel.plugins.manifest import PortCatalogue
from .kit import ConformanceCheck, ConformanceKit, KitSection

# --- P25 ArtifactStorePort ---------------------------------------------------- #


def _artifact_declares_id(adapter) -> None:
    assert adapter.store_id, "an artifact store must declare a store_id"


def _artifact_missing_is_typed(adapter) -> None:
    ref = ArtifactRef(
        uri="file:///definitely/absent/kit-artifact.bin",
        expected_hash="blake2b:0" * 1,
    )
    try:
        adapter.fetch(ref)
    except ArtifactUnavailableError:
        return
    except ArtifactIntegrityError:
        return
    raise AssertionError(
        "a missing artifact must raise ArtifactUnavailableError, never return a path"
    )


def _artifact_hash_mismatch_fails_closed(adapter) -> None:
    """A mismatch is a security event, never retried into success."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "kit-artifact.bin"
        path.write_bytes(b"kit-payload")
        wrong = "blake2b:" + hashlib.blake2b(b"different", digest_size=32).hexdigest()
        ref = ArtifactRef(uri=f"file://{path}", expected_hash=wrong)
        try:
            adapter.fetch(ref)
        except ArtifactIntegrityError:
            return
        except ArtifactUnavailableError:
            return
    raise AssertionError(
        "an artifact whose content does not match its declared hash must be "
        "refused; loading unverified weights is a supply-chain vulnerability"
    )


ARTIFACT_STORE_KIT = ConformanceKit(
    port_id=PortCatalogue.ARTIFACT_STORE,
    version="1.0.0",
    checks=(
        ConformanceCheck("declares_id", KitSection.SHAPE, _artifact_declares_id, "A1"),
        ConformanceCheck(
            "missing_is_typed", KitSection.FAILURE, _artifact_missing_is_typed, "A4"
        ),
        ConformanceCheck(
            "hash_mismatch_fails_closed",
            KitSection.SEMANTICS,
            _artifact_hash_mismatch_fails_closed,
            "A4",
        ),
    ),
)


# --- P26 ModelRuntimePort ------------------------------------------------------ #


def _runtime_declares_id(adapter) -> None:
    assert adapter.runtime_id, "a model runtime must declare a runtime_id"


def _runtime_supports_is_total(adapter) -> None:
    """``supports`` must answer for any input rather than raising.

    The Model Manager calls it while choosing a runtime; a raise there would make
    an unsupported artifact indistinguishable from a broken runtime.
    """
    for path, precision in (
        ("model.pt", "fp32"),
        ("model.onnx", "fp16"),
        ("model.unknown", "int8"),
        ("", "fp32"),
    ):
        result = adapter.supports(path, precision)
        assert isinstance(result, bool), (
            f"supports({path!r}, {precision!r}) returned {type(result).__name__}, "
            f"not a bool"
        )


def _runtime_unload_is_idempotent(adapter) -> None:
    """Double-unload happens during rollback and must not fault."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "kit-model.bin"
        path.write_bytes(b"kit")
        if not adapter.supports(str(path), "fp32"):
            return
        loaded = adapter.load(
            model_id="kit-model",
            version="1.0.0",
            artifact_path=str(path),
            artifact_hash="blake2b:kit",
            device_id="cpu",
            precision="fp32",
            options=None,
        )
        adapter.unload(loaded)
        adapter.unload(loaded)


def _runtime_load_reports_provenance(adapter) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "kit-model.bin"
        path.write_bytes(b"kit")
        if not adapter.supports(str(path), "fp32"):
            return
        loaded = adapter.load(
            model_id="kit-model",
            version="1.0.0",
            artifact_path=str(path),
            artifact_hash="blake2b:kit",
            device_id="cpu",
            precision="fp32",
            options=None,
        )
        try:
            assert loaded.model_id == "kit-model"
            assert loaded.artifact_hash == "blake2b:kit", (
                "a loaded model must carry the hash of the weights it loaded"
            )
            assert loaded.session is not None, "a loaded model must expose a session"
        finally:
            adapter.unload(loaded)


MODEL_RUNTIME_KIT = ConformanceKit(
    port_id=PortCatalogue.MODEL_RUNTIME,
    version="1.0.0",
    checks=(
        ConformanceCheck("declares_id", KitSection.SHAPE, _runtime_declares_id, "A1"),
        ConformanceCheck(
            "supports_is_total", KitSection.SEMANTICS, _runtime_supports_is_total
        ),
        ConformanceCheck(
            "load_reports_provenance",
            KitSection.SEMANTICS,
            _runtime_load_reports_provenance,
            "A3",
        ),
        ConformanceCheck(
            "unload_is_idempotent", KitSection.FAILURE, _runtime_unload_is_idempotent
        ),
    ),
)


# --- P27 DevicePort ------------------------------------------------------------- #


def _device_declares_id(adapter) -> None:
    assert adapter.provider_id, "a device provider must declare a provider_id"


def _device_enumerate_is_total(adapter) -> None:
    """An empty inventory is a valid answer, not an error.

    A node with no accelerator is the common case at the edge; a provider that
    raises there would make CPU-only deployment impossible.
    """
    devices = adapter.enumerate()
    assert devices is not None, "enumerate() must return a sequence, never None"
    for device in devices:
        assert device.device_id, "every device must have an id"
        assert device.total_memory_bytes >= 0


def _device_disappearance_is_reported_not_raised(adapter) -> None:
    """Liveness checks never raise (invariant V9)."""
    for device_id in ("cuda:99", "nonexistent", ""):
        result = adapter.is_available(device_id)
        assert isinstance(result, bool), (
            f"is_available({device_id!r}) returned {type(result).__name__}; a "
            f"disappeared device must be reported unavailable, never raised"
        )


def _device_utilization_is_bounded(adapter) -> None:
    for device in adapter.enumerate():
        value = adapter.utilization(device.device_id)
        assert 0.0 <= value <= 1.0, (
            f"utilization {value} for '{device.device_id}' escapes [0,1]"
        )


DEVICE_KIT = ConformanceKit(
    port_id=PortCatalogue.DEVICE,
    version="1.0.0",
    checks=(
        ConformanceCheck("declares_id", KitSection.SHAPE, _device_declares_id, "A1"),
        ConformanceCheck(
            "enumerate_is_total", KitSection.SEMANTICS, _device_enumerate_is_total
        ),
        ConformanceCheck(
            "utilization_is_bounded", KitSection.SEMANTICS, _device_utilization_is_bounded
        ),
        ConformanceCheck(
            "disappearance_is_reported_not_raised",
            KitSection.FAILURE,
            _device_disappearance_is_reported_not_raised,
            "A4",
        ),
    ),
)

ALL_MODEL_KITS: tuple[ConformanceKit, ...] = (
    ARTIFACT_STORE_KIT,
    MODEL_RUNTIME_KIT,
    DEVICE_KIT,
)
