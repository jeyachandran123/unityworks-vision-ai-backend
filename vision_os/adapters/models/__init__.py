"""P25/P26/P27 model adapters — artifacts, runtimes, devices."""

from __future__ import annotations

from .artifacts import InMemoryArtifactStore, LocalArtifactStore, compute_hash
from .devices import (
    CPU_DEVICE_ID,
    CpuDeviceProvider,
    CudaDeviceProvider,
    StaticDeviceProvider,
)
from .runtimes import (
    DetectorSession,
    LetterboxedImage,
    LetterboxTransform,
    RawBox,
    ScriptedRuntime,
    ScriptedSession,
    UltralyticsRuntime,
    UltralyticsSession,
)

__all__ = [
    "CPU_DEVICE_ID",
    "CpuDeviceProvider",
    "CudaDeviceProvider",
    "DetectorSession",
    "InMemoryArtifactStore",
    "LetterboxTransform",
    "LetterboxedImage",
    "LocalArtifactStore",
    "RawBox",
    "ScriptedRuntime",
    "ScriptedSession",
    "StaticDeviceProvider",
    "UltralyticsRuntime",
    "UltralyticsSession",
    "compute_hash",
]
