"""P7 AllocatorPort — host memory pool adapter.

**No steady-state allocation.** Slots are carved at construction and recycled
thereafter; the running system allocates nothing, which is what makes 30-day soak
stability achievable rather than aspirational (03_MODULES M4).

A device-resident allocator (CUDA, unified memory, shared memory for
cross-process pipelines, RDMA-registered for a future distributed data plane) is
a sibling adapter behind the same port.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ...core.errors import PoolExhaustedError


class PooledAllocation:
    """One recycled buffer. Implements ``Allocation``."""

    __slots__ = ("_buffer", "_nbytes", "_index")

    def __init__(self, buffer: bytearray, index: int) -> None:
        self._buffer = buffer
        self._nbytes = len(buffer)
        self._index = index

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def index(self) -> int:
        return self._index

    def memory(self) -> memoryview:
        return memoryview(self._buffer)


@dataclass(frozen=True, slots=True)
class PoolStats:
    total_slots: int
    in_use: int
    bytes_per_slot: int


class HostMemoryPool:
    """A fixed-size pool of pre-allocated host buffers."""

    def __init__(self, *, slots: int, bytes_per_slot: int) -> None:
        if slots < 1:
            raise ValueError(f"slots must be >= 1, got {slots}")
        if bytes_per_slot < 1:
            raise ValueError(f"bytes_per_slot must be >= 1, got {bytes_per_slot}")
        self._bytes_per_slot = bytes_per_slot
        self._lock = threading.Lock()
        self._allocations = [PooledAllocation(bytearray(bytes_per_slot), i) for i in range(slots)]
        self._free: list[int] = list(range(slots))
        self._in_use: set[int] = set()

    @property
    def location(self) -> str:
        return "host"

    def allocate(self, nbytes: int) -> PooledAllocation:
        if nbytes > self._bytes_per_slot:
            raise PoolExhaustedError(
                f"requested {nbytes} bytes exceeds slot size {self._bytes_per_slot}",
                requested=nbytes,
            )
        with self._lock:
            if not self._free:
                raise PoolExhaustedError(
                    f"host pool exhausted ({len(self._in_use)}/{len(self._allocations)} in use)",
                    in_use=len(self._in_use),
                )
            index = self._free.pop()
            self._in_use.add(index)
            return self._allocations[index]

    def release(self, allocation: PooledAllocation) -> None:
        """Idempotent: releasing twice is safe and is not an error."""
        with self._lock:
            index = allocation.index
            if index in self._in_use:
                self._in_use.discard(index)
                self._free.append(index)

    def stats(self) -> PoolStats:
        with self._lock:
            return PoolStats(
                total_slots=len(self._allocations),
                in_use=len(self._in_use),
                bytes_per_slot=self._bytes_per_slot,
            )
