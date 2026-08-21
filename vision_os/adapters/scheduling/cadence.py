"""P5 AdmissionPolicyPort — cadence and budget policy adapters.

``CadenceAdmissionPolicy`` is the platform default: an integer phase accumulator
on monotonic time, immune to wall-clock steps, allocation-free, with no locking.

Note what these policies deliberately cannot see: any notion of which camera
*matters*. ``priority_class`` is an opaque label a policy may order by and may
not interpret — "process the kitchen more often because it matters more" is a
business priority and belongs to the consumer (invariant V1/V2).
"""

from __future__ import annotations

from ...core.ports.scheduling import (
    AdmissionContext,
    AdmissionVerdict,
    DropReason,
    Fidelity,
)

_NANOS_PER_SECOND = 1_000_000_000


class CadenceAdmissionPolicy:
    """Admit at the profile's target rate, shedding under pressure.

    Evaluation order matters and is deliberate:

    1. **Queue full** — never pile work onto a stage that is already behind.
    2. **Cadence** — the by-design case, which dominates in a healthy system.
    3. **Budget** — node-level saturation, which is a signal, not a design.

    Checking cadence before budget means a healthy low-rate camera is never
    charged for a busy neighbour's saturation.
    """

    __slots__ = ("_pressure_ceiling",)

    def __init__(self, *, pressure_ceiling: float = 1.0) -> None:
        self._pressure_ceiling = pressure_ceiling

    def evaluate(self, context: AdmissionContext) -> AdmissionVerdict:
        if context.queue_full or context.in_flight >= context.profile.max_in_flight:
            return AdmissionVerdict(admit=False, reason=DropReason.QUEUE_FULL)

        interval_ns = int(_NANOS_PER_SECOND / context.profile.target_fps)
        last = context.last_admitted_monotonic
        if last is not None and (context.monotonic_now.ns - last.ns) < interval_ns:
            return AdmissionVerdict(admit=False, reason=DropReason.CADENCE)

        if context.budget_pressure >= self._pressure_ceiling:
            return AdmissionVerdict(admit=False, reason=DropReason.BUDGET_EXHAUSTED)

        return AdmissionVerdict(
            admit=True,
            fidelity=Fidelity(
                inference_width=context.profile.inference_width,
                inference_height=context.profile.inference_height,
            ),
        )


class AdmitAllPolicy:
    """Admit everything. For archival replay and deterministic tests.

    Archival sources protect completeness over latency, so shedding would defeat
    the purpose (01_LAYERED §5.3).
    """

    __slots__ = ()

    def evaluate(self, context: AdmissionContext) -> AdmissionVerdict:
        return AdmissionVerdict(
            admit=True,
            fidelity=Fidelity(
                inference_width=context.profile.inference_width,
                inference_height=context.profile.inference_height,
            ),
        )


class ResolutionLadderPolicy:
    """Cadence admission that reduces inference resolution under pressure.

    Step 3 of the acquisition degradation ladder. Lowering resolution silently
    changes *what the platform can see* — small and distant objects disappear —
    so the fidelity tier is recorded on the verdict and travels downstream rather
    than being applied invisibly (10_RELIABILITY §4.2).
    """

    __slots__ = ("_inner", "_degraded_scale", "_pressure_threshold")

    def __init__(
        self,
        *,
        degraded_scale: float = 0.5,
        pressure_threshold: float = 0.8,
    ) -> None:
        self._inner = CadenceAdmissionPolicy(pressure_ceiling=1.5)
        self._degraded_scale = degraded_scale
        self._pressure_threshold = pressure_threshold

    def evaluate(self, context: AdmissionContext) -> AdmissionVerdict:
        verdict = self._inner.evaluate(context)
        if not verdict.admit or context.budget_pressure < self._pressure_threshold:
            return verdict
        return AdmissionVerdict(
            admit=True,
            fidelity=Fidelity(
                inference_width=max(
                    32, int(context.profile.inference_width * self._degraded_scale)
                ),
                inference_height=max(
                    32, int(context.profile.inference_height * self._degraded_scale)
                ),
                tier="degraded",
            ),
        )
