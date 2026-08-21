"""Model ports — P25 ArtifactStore, P26 ModelRuntime, P27 Device.

Owner: M18 Model Manager.

The Model Manager knows about **weights, memory, devices, and versions** — never
about detectors, trackers, or attributes. That ignorance is what lets it serve
model kinds that do not exist yet, and it is why these three ports carry no
vision vocabulary at all.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class DeviceKind(enum.Enum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"
    EDGE = "edge"

    @property
    def is_accelerator(self) -> bool:
        return self is not DeviceKind.CPU


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """One compute device the platform may place a model on."""

    device_id: str
    kind: DeviceKind
    index: int = 0
    total_memory_bytes: int = 0
    name: str = ""
    available: bool = True

    def __post_init__(self) -> None:
        if not self.device_id:
            raise ValueError("device_id is required")
        if self.total_memory_bytes < 0:
            raise ValueError("total_memory_bytes must be non-negative")


@runtime_checkable
class DevicePort(Protocol):
    """P27 — device inventory and liveness.

    Implementations: CPU (always present), CUDA, ROCm, Metal, edge accelerators.

    A device that **disappears** must be reported as unavailable rather than
    raising: the Model Manager migrates handles to what remains and the platform
    degrades (invariant V9).
    """

    @property
    def provider_id(self) -> str: ...

    def enumerate(self) -> Sequence[DeviceInfo]:
        """Devices this provider can offer. May be empty; that is not an error."""
        ...

    def is_available(self, device_id: str) -> bool:
        """Liveness check. Must never raise."""
        ...

    def utilization(self, device_id: str) -> float:
        """Best-effort utilization in [0,1]. Return 0.0 when unknown."""
        ...


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Where a model artifact lives and what it must hash to."""

    uri: str
    expected_hash: str
    size_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("artifact uri is required")
        if not self.expected_hash:
            raise ValueError(
                "an artifact must declare its expected hash; loading unverified "
                "weights is a supply-chain vulnerability (12_SECURITY section 6)"
            )


@runtime_checkable
class ArtifactStorePort(Protocol):
    """P25 — fetch and verify model artifacts.

    Implementations: local filesystem, object storage, OCI registry.

    ``fetch`` **verifies the content hash and fails closed**. A hash mismatch is
    a supply-chain event, not a network glitch, and is never retried into
    success.
    """

    @property
    def store_id(self) -> str: ...

    def fetch(self, ref: ArtifactRef) -> str:
        """Return a local path to the verified artifact.

        Raises:
            ArtifactIntegrityError: on hash mismatch. Fails closed.
            ArtifactUnavailableError: transient; the caller may retry or fall
                back to a cached known-good version.
        """
        ...

    def has(self, ref: ArtifactRef) -> bool:
        """Whether a verified copy is already cached locally."""
        ...


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A framework-loaded model, ready to execute.

    Opaque to the platform: ``session`` is whatever the runtime adapter needs and
    the platform never inspects it.
    """

    model_id: str
    version: str
    artifact_hash: str
    device_id: str
    precision: str
    session: object
    vram_bytes: int = 0
    load_ms: float = 0.0
    warmup_ms: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class ModelRuntimePort(Protocol):
    """P26 — turn a verified artifact into an executable session.

    Implementations: native framework, ONNX Runtime, TensorRT, OpenVINO, Triton,
    cloud endpoints.

    This is what makes an adapter *family* portable: the same YOLO adapter runs on
    ultralytics locally and on ONNX at the edge, because the letterboxing,
    coordinate inversion and taxonomy mapping live in the adapter while only the
    tensor call lives here.
    """

    @property
    def runtime_id(self) -> str: ...

    def supports(self, artifact_path: str, precision: str) -> bool:
        """Whether this runtime can load that artifact at that precision."""
        ...

    def load(
        self,
        *,
        model_id: str,
        version: str,
        artifact_path: str,
        artifact_hash: str,
        device_id: str,
        precision: str,
        options: dict[str, str] | None = None,
    ) -> LoadedModel:
        """Load and place the model.

        Raises:
            ModelLoadError: corrupt, incompatible, or unsupported. The Model
                Manager marks the version bad and falls back to last known-good.
            DeviceOutOfMemoryError: the broker evicts and retries once.
        """
        ...

    def unload(self, loaded: LoadedModel) -> None:
        """Release device memory. Idempotent."""
        ...
