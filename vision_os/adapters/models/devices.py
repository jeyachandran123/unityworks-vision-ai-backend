"""P27 ``DevicePort`` — device inventory adapters.

CPU is always present. CUDA is discovered through an optional import, so a node
without torch, without a driver, or without a card degrades to CPU rather than
failing to start (invariant V9).

**A disappeared device is reported unavailable, never raised.** Migration is the
Model Manager's decision; a provider that throws when a card is pulled would take
the platform down with it.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence

from ...core.ports.models import DeviceInfo, DeviceKind

CPU_DEVICE_ID = "cpu"


class CpuDeviceProvider:
    """The device that is always there.

    ``total_memory_bytes`` is zero by design: host memory is not brokered by
    capacity here, so the broker treats CPU as an unconditional fallback rather
    than competing for a number it does not own.
    """

    __slots__ = ()

    @property
    def provider_id(self) -> str:
        return "cpu"

    def enumerate(self) -> Sequence[DeviceInfo]:
        return (
            DeviceInfo(
                device_id=CPU_DEVICE_ID,
                kind=DeviceKind.CPU,
                index=0,
                total_memory_bytes=0,
                name="host cpu",
                available=True,
            ),
        )

    def is_available(self, device_id: str) -> bool:
        return device_id == CPU_DEVICE_ID

    def utilization(self, device_id: str) -> float:
        return 0.0


class CudaDeviceProvider:
    """CUDA devices, discovered lazily through torch if it is installed.

    The import is deliberately deferred and guarded: a platform that cannot start
    because an optional accelerator library is missing has made GPU support a
    hard dependency, which is exactly what the port exists to prevent.
    """

    def __init__(self, *, visible_devices: str | None = None) -> None:
        self._visible = visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES")
        self._lock = threading.Lock()
        self._cached: tuple[DeviceInfo, ...] | None = None

    @property
    def provider_id(self) -> str:
        return "cuda"

    def _torch(self):
        try:
            import torch  # noqa: PLC0415 - optional dependency, imported on demand
        except Exception:  # noqa: BLE001 - absent or broken torch is simply "no CUDA"
            return None
        return torch

    def enumerate(self) -> Sequence[DeviceInfo]:
        with self._lock:
            if self._cached is not None:
                return self._cached
            torch = self._torch()
            devices: list[DeviceInfo] = []
            if torch is not None:
                try:
                    if torch.cuda.is_available():
                        for index in range(torch.cuda.device_count()):
                            properties = torch.cuda.get_device_properties(index)
                            devices.append(
                                DeviceInfo(
                                    device_id=f"cuda:{index}",
                                    kind=DeviceKind.CUDA,
                                    index=index,
                                    total_memory_bytes=int(properties.total_memory),
                                    name=str(properties.name),
                                    available=True,
                                )
                            )
                except Exception:  # noqa: BLE001 - a broken driver is "no CUDA", not a crash
                    devices = []
            self._cached = tuple(devices)
            return self._cached

    def invalidate(self) -> None:
        """Force re-enumeration, e.g. after a device is hot-removed."""
        with self._lock:
            self._cached = None

    def is_available(self, device_id: str) -> bool:
        torch = self._torch()
        if torch is None:
            return False
        try:
            if not torch.cuda.is_available():
                return False
            index = int(device_id.split(":", 1)[1])
            return index < torch.cuda.device_count()
        except Exception:  # noqa: BLE001 - liveness checks never raise upward
            return False

    def utilization(self, device_id: str) -> float:
        torch = self._torch()
        if torch is None:
            return 0.0
        try:
            index = int(device_id.split(":", 1)[1])
            free, total = torch.cuda.mem_get_info(index)
            return (total - free) / total if total else 0.0
        except Exception:  # noqa: BLE001
            return 0.0


class StaticDeviceProvider:
    """A scripted device inventory.

    Ships in the adapter set rather than only in tests because GPU-loss
    behaviour is a compliance-grade guarantee, and a guarantee with no way to
    rehearse it is one nobody has verified.
    """

    def __init__(self, devices: Sequence[DeviceInfo]) -> None:
        self._devices = list(devices)
        self._lock = threading.Lock()

    @property
    def provider_id(self) -> str:
        return "static"

    def enumerate(self) -> Sequence[DeviceInfo]:
        with self._lock:
            return tuple(self._devices)

    def is_available(self, device_id: str) -> bool:
        with self._lock:
            return any(
                d.device_id == device_id and d.available for d in self._devices
            )

    def utilization(self, device_id: str) -> float:
        return 0.0

    def remove(self, device_id: str) -> None:
        """Simulate a device disappearing mid-operation."""
        with self._lock:
            self._devices = [
                DeviceInfo(
                    device_id=d.device_id,
                    kind=d.kind,
                    index=d.index,
                    total_memory_bytes=d.total_memory_bytes,
                    name=d.name,
                    available=False if d.device_id == device_id else d.available,
                )
                for d in self._devices
            ]

    def restore(self, device_id: str) -> None:
        with self._lock:
            self._devices = [
                DeviceInfo(
                    device_id=d.device_id,
                    kind=d.kind,
                    index=d.index,
                    total_memory_bytes=d.total_memory_bytes,
                    name=d.name,
                    available=True if d.device_id == device_id else d.available,
                )
                for d in self._devices
            ]
