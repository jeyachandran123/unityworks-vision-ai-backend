"""Suppression policies — P18.

> 06_PORTS: *exact, threshold, semantic.*

Three ship, and the differences between them are entirely about **what counts as
a change** — which is a deployment decision, never a platform one. A forensic
deployment wants every frame; a busy retail floor wants movement beyond a
threshold; a constrained edge link wants only material change.

Every policy obeys the same three non-negotiables, because they are what make
suppression safe rather than merely cheap:

**The first observation always publishes** (S1). There is nothing to compare
against, and suppressing it would mean an object could exist in the log only
implicitly.

**A heartbeat always publishes** (S2). §M11: *"a consumer must be able to
distinguish 'unchanged' from 'stopped observing.'"* Without it, a working camera
and a dead one produce identical silence.

**Coverage, lifecycle and identity never suppress.** Each is a transition by
definition, and 02_VOM §11.2 calls coverage *"not optional"*.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ...core.model.observation import Observation, ObservationType
from ...core.model.timebase import Duration
from ...core.ports.synthesis import SuppressionDecision

#: Types that are never suppressed, whatever a policy would prefer.
#:
#: ``coverage`` because a suppressed blindness report is the platform deciding
#: its own outage was not worth mentioning. ``lifecycle`` and ``identity``
#: because each *is* a transition: an unchanged one cannot occur, so suppressing
#: one can only ever drop a real event.
ALWAYS_PUBLISH: frozenset[ObservationType] = frozenset(
    {
        ObservationType.COVERAGE,
        ObservationType.LIFECYCLE,
        ObservationType.IDENTITY,
    }
)

#: Quantization for the exact policy's position digest, in normalized units.
#:
#: Sub-millimetre jitter in a normalized coordinate is not a change anybody can
#: observe, and treating it as one would defeat suppression entirely.
EXACT_POSITION_PRECISION = 3


def _mandatory(candidate: Observation, previous: str | None) -> str | None:
    """The reason this must publish regardless of content, or ``None``.

    Shared by every policy so the non-negotiables cannot be honoured in one
    implementation and forgotten in another.
    """
    if previous is None:
        return "first observation for this subject"
    if candidate.observation_type in ALWAYS_PUBLISH:
        return f"{candidate.observation_type.value} is never suppressed"
    if candidate.is_correction:
        return "correction; a correction nobody receives is not a correction"
    return None


def _digest(observation: Observation, *, precision: int) -> str:
    """Content digest at a chosen positional precision.

    Excludes ``observation_id``, ``t_published`` and timing deliberately (S7):
    those differ on every build, and including them would make every observation
    look changed.
    """
    hasher = hashlib.sha256()
    hasher.update(observation.observation_type.value.encode())
    hasher.update(str(observation.object_id or "").encode())
    hasher.update(str(observation.class_id or "").encode())
    hasher.update(
        (observation.lifecycle_state.value if observation.lifecycle_state else "").encode()
    )
    hasher.update(observation.measurement_basis.value.encode())

    if observation.spatial is not None and observation.spatial.bbox is not None:
        box = observation.spatial.bbox
        hasher.update(
            (
                f"{box.x1:.{precision}f},{box.y1:.{precision}f},"
                f"{box.x2:.{precision}f},{box.y2:.{precision}f}"
            ).encode()
        )
    for attribute in observation.attributes:
        hasher.update(f"{attribute.key}={attribute.value!r}".encode())
    if observation.coverage is not None:
        hasher.update(observation.coverage.status.value.encode())
        hasher.update(observation.coverage.reason.value.encode())
    if observation.lifecycle_transition is not None:
        hasher.update(observation.lifecycle_transition.previous.value.encode())
        hasher.update(observation.lifecycle_transition.current.value.encode())
    if observation.quality is not None and observation.quality.overall is not None:
        hasher.update(observation.quality.overall.value.encode())
    return hasher.hexdigest()


class ExactSuppression:
    """Publish when the content digest differs. ``suppression.exact``.

    The conservative default: any change at all publishes. Cheapest to reason
    about, and the right choice when a deployment does not yet know which changes
    matter to its consumers.
    """

    __slots__ = ()

    @property
    def policy_id(self) -> str:
        return "suppression.exact"

    def signature(self, observation: Observation) -> str:
        return _digest(observation, precision=EXACT_POSITION_PRECISION)

    def should_publish(
        self,
        candidate: Observation,
        previous_signature: str | None,
        *,
        elapsed: Duration,
        heartbeat: Duration,
    ) -> SuppressionDecision:
        mandatory = _mandatory(candidate, previous_signature)
        if mandatory:
            return SuppressionDecision(publish=True, reason=mandatory)

        if elapsed.ns >= heartbeat.ns:
            return SuppressionDecision(publish=True, reason="heartbeat cadence reached")

        if previous_signature != self.signature(candidate):
            return SuppressionDecision(publish=True, reason="content changed")

        return SuppressionDecision(
            publish=False, reason="identical to the last published observation"
        )


@dataclass(frozen=True, slots=True)
class ThresholdSuppression:
    """Coarser positional quantization. ``suppression.threshold``.

    For deployments where a stationary object's box jitter is not information.
    Implemented as *precision*, not as a distance comparison, so the policy stays
    a pure digest comparison and M11's state stays one opaque string per subject.

    Everything except position is still compared exactly: an attribute that
    changed is a change however still the object was.
    """

    position_threshold: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 < self.position_threshold <= 1.0:
            raise ValueError("position_threshold must be in (0,1]")

    @property
    def policy_id(self) -> str:
        return "suppression.threshold"

    @property
    def precision(self) -> int:
        """Decimal places implied by the threshold.

        A 0.01 threshold quantizes to two places, so movement under a hundredth
        of the frame produces the same digest and is suppressed.
        """
        import math

        return max(0, math.ceil(-math.log10(self.position_threshold)))

    def signature(self, observation: Observation) -> str:
        return _digest(observation, precision=self.precision)

    def should_publish(
        self,
        candidate: Observation,
        previous_signature: str | None,
        *,
        elapsed: Duration,
        heartbeat: Duration,
    ) -> SuppressionDecision:
        mandatory = _mandatory(candidate, previous_signature)
        if mandatory:
            return SuppressionDecision(publish=True, reason=mandatory)

        if elapsed.ns >= heartbeat.ns:
            return SuppressionDecision(publish=True, reason="heartbeat cadence reached")

        if previous_signature != self.signature(candidate):
            return SuppressionDecision(
                publish=True, reason="content changed beyond threshold"
            )

        return SuppressionDecision(
            publish=False, reason="unchanged within the configured threshold"
        )


class AlwaysPublish:
    """Never suppresses. ``suppression.always``.

    The forensic mode 06_PORTS lists. Also the honest baseline a conformance kit
    checks the others against: whatever a policy suppresses, this one publishes,
    so a test can compare volumes and confirm the reduction is real.
    """

    __slots__ = ()

    @property
    def policy_id(self) -> str:
        return "suppression.always"

    def signature(self, observation: Observation) -> str:
        return _digest(observation, precision=EXACT_POSITION_PRECISION)

    def should_publish(
        self,
        candidate: Observation,
        previous_signature: str | None,
        *,
        elapsed: Duration,
        heartbeat: Duration,
    ) -> SuppressionDecision:
        return SuppressionDecision(publish=True, reason="forensic mode: never suppress")
