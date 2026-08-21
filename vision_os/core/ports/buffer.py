"""Buffer ports — P7 ``AllocatorPort`` (owner: M4 Frame Buffer).

The Frame Buffer owns pixel memory and its lifetime; it knows nothing about
pixels. Allocation strategy — host pinned, CUDA unified, shared memory for
cross-process pipelines, RDMA-registered for a future distributed data plane — is
an adapter concern (03_MODULES M4 extension points).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class WritableSlot(Protocol):
    """A writable region of pooled memory handed to a decoder.

    Exists so that decoding is zero-copy: the decoder writes directly into
    buffer-pool memory rather than allocating its own image and having the
    platform copy it. At 100 cameras a copy per frame makes memory bandwidth,
    not compute, the wall (11_PERFORMANCE §8).
    """

    @property
    def capacity(self) -> int:
        """Maximum bytes writable into this slot."""
        ...

    def memory(self) -> memoryview:
        """A writable view of the slot's backing store."""
        ...


@runtime_checkable
class Allocation(Protocol):
    """A pooled allocation owned by the allocator that produced it."""

    @property
    def nbytes(self) -> int: ...

    def memory(self) -> memoryview:
        """Writable view. Becomes read-only to consumers once published."""
        ...


@runtime_checkable
class AllocatorPort(Protocol):
    """P7 — pooled memory acquisition.

    Implementations must perform **no steady-state allocation**: pools are sized
    at startup and the running system allocates nothing, which is what makes
    30-day soak stability achievable (03_MODULES M4).
    """

    @property
    def location(self) -> str:
        """``host`` or ``device``. Used for placement decisions and telemetry."""
        ...

    def allocate(self, nbytes: int) -> Allocation:
        """Take an allocation of at least ``nbytes``.

        Raises:
            PoolExhaustedError: when no capacity is available.
        """
        ...

    def release(self, allocation: Allocation) -> None:
        """Return an allocation to the pool. Idempotent."""
        ...

    def stats(self) -> AllocatorStats:
        """Current pool occupancy, for capacity telemetry."""
        ...


class AllocatorStats(Protocol):
    @property
    def total_slots(self) -> int: ...

    @property
    def in_use(self) -> int: ...

    @property
    def bytes_per_slot(self) -> int: ...
