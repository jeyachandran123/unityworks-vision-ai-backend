"""The Clock port (08_RUNTIME_AND_THREADING §6.1).

**No module reads the system clock. Every module receives a ``Clock``.**

This single injection is the prerequisite for invariant V13. A module that calls
the system clock directly can never be replayed deterministically, and there is
no test infrastructure that repairs it afterward — which is why this is an
architectural rule rather than a testing convenience.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..model.timebase import Duration, Instant


@runtime_checkable
class Clock(Protocol):
    """Time, injected.

    Implementations: ``SystemClock`` (production), ``VirtualClock`` (replay and
    testing, advanced explicitly), ``ScaledClock`` (soak testing at N x real
    time). See ``kernel.clock``.
    """

    def now(self) -> Instant:
        """Current UTC instant."""
        ...

    def monotonic(self) -> Instant:
        """A monotonic reading immune to wall-clock steps.

        Cadence and timeouts use this; an NTP step must never corrupt scheduling
        phase or make a duration negative.
        """
        ...

    async def sleep(self, duration: Duration) -> None:
        """Suspend for ``duration`` on this clock's timeline."""
        ...

    @property
    def is_virtual(self) -> bool:
        """True when time advances only by explicit control (deterministic mode)."""
        ...
