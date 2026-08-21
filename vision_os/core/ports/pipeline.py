"""The pipeline continuation seam.

Flow 1's Runtime ends at admission: an admitted frame is counted and released.
This protocol is the **single, documented extension point** at which a later flow
resumes the admitted-frame path (Flow 1 report section 11).

It carries a ``FrameRef``, not a ``Frame``. The consumer takes its own lease from
the Frame Buffer, which means:

* the lease protocol is exercised rather than bypassed, so a frame evicted
  between admission and consumption produces a clean ``FrameUnavailableError``
  degradation instead of a dangling reference;
* the payload stays control-plane sized, so the same seam works when the consumer
  runs in another process or on another node (invariant V12).

Flow 1 remains unaware of any consumer: the Runtime holds this protocol and never
learns what implements it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..model.detection import DetectionOutcome
from ..model.ids import FrameRef
from .scheduling import Fidelity


@runtime_checkable
class AdmittedFrameConsumer(Protocol):
    """Resumes the pipeline after admission.

    Implementations **must not raise**. A failure in a later stage may never
    terminate the Vision Runtime or a source actor (invariant V9); it degrades,
    counts, and publishes.
    """

    async def on_admitted(self, frame_ref: FrameRef, fidelity: Fidelity) -> None:
        """Consume one admitted frame.

        Args:
            frame_ref: The admitted frame. Take a lease from the Frame Buffer to
                read its pixels; treat ``FrameUnavailableError`` as normal.
            fidelity: The resolution tier and model tier the scheduler selected.
        """
        ...


@runtime_checkable
class DetectionConsumer(Protocol):
    """Resumes the pipeline after detection — the L2 Detection-to-Tracking handoff.

    ``01_LAYERED`` section 2.1 classifies this as a **sideways within-layer**
    dependency: a direct call along the declared intra-layer order, not an Event
    Bus notification. ``08_RUNTIME`` section 5.2 specifies the connection as a
    bounded queue with a **block** policy, because *"ordering matters; dropping
    here corrupts tracks"*. The Event Bus is lossy by design and is therefore the
    wrong transport for this edge, however right it is for observability.

    **Every outcome is delivered, including empty and failed ones.** This is not
    an optimization detail — it is a correctness requirement:

    * an empty frame is exactly when a track coasts, ages, and eventually
      terminates. A consumer that only sees non-empty frames never ages anything;
    * ``failed`` and empty are different facts. "The detector broke" must never
      be indistinguishable from "nothing was there" (invariant V8).

    Implementations **must not raise**, for the same reason
    ``AdmittedFrameConsumer`` must not.
    """

    async def on_detected(self, outcome: DetectionOutcome) -> None:
        """Consume one frame's detection outcome, in per-camera frame order."""
        ...


@runtime_checkable
class RegistryConsumer(Protocol):
    """Resumes the pipeline after the Object Registry — the L2-to-L3 handoff.

    The Flow 5 seam, declared in the Flow 4 report's extension-point table. The
    Crop Manager attaches here exactly as the registry attached to tracking.

    It carries objects rather than pixels, for the same reason
    ``AdmittedFrameConsumer`` carries a ``FrameRef``: attention is a
    **control-plane** decision about metadata. Only the small subset that
    survives trigger evaluation ever leases pixels, which is what lets one node
    evaluate thousands of candidates a second (invariant V12, §M8 Performance).

    **Every update is delivered, including empty and failed ones**, for the same
    reason detection outcomes are: an empty population is when trigger state
    ages and demands go unserved, and ``failed`` must never be
    indistinguishable from "nothing was there" (invariant V8).

    Implementations **must not raise**.
    """

    async def on_registered(self, update: object) -> None:
        """Consume one camera's registry update, in per-camera frame order.

        The parameter is typed ``object`` because ``RegistryUpdate`` lives in
        ``perception.registry`` and ``core`` may not import a perception module —
        the dependency would point the wrong way through the layers (01_LAYERED
        section 2). Implementations narrow it.
        """
        ...
