"""P3 PrivacyMaskPort — masking adapters.

Applied in-place on the pooled slot, immediately post-decode and before the frame
is published, so that **no component ever sees unmasked pixels** (12_SECURITY
§2.1).

This is the platform's only fail-closed path: a masking failure drops the frame
rather than degrading it. A masking failure that proceeds is a compliance
incident regardless of intent.
"""

from __future__ import annotations

from ...core.errors import PrivacyMaskError
from ...core.model.frame import FrameDimensions, PrivacyState
from ...core.model.ids import PrivacyPolicyId
from ...core.model.space import Polygon
from ...core.ports.acquisition import MaskOutcome
from ...core.ports.buffer import WritableSlot

_FAILING_POLICY_ID = PrivacyPolicyId("failing")


class NoMaskPolicy:
    """No masking configured for this camera.

    Reports ``UNMASKED_PERMITTED`` rather than pretending to have masked — the
    distinction is auditable and travels on the frame.
    """

    __slots__ = ()

    @property
    def policy_id(self) -> PrivacyPolicyId | None:
        return None

    def apply(self, slot: WritableSlot, dimensions: FrameDimensions) -> MaskOutcome:
        return MaskOutcome(state=PrivacyState.UNMASKED_PERMITTED)


class StaticZoneMask:
    """Black out fixed polygonal regions (a neighbour's window, a public street).

    Geometry is in normalized image coordinates so a resolution change does not
    silently move the mask — a class of privacy regression that is otherwise
    invisible until someone reviews footage.
    """

    def __init__(
        self,
        *,
        policy_id: PrivacyPolicyId,
        zones: tuple[Polygon, ...],
        bytes_per_pixel: int = 3,
    ) -> None:
        if not zones:
            raise ValueError("StaticZoneMask requires at least one zone")
        self._policy_id = policy_id
        self._zones = zones
        self._bytes_per_pixel = bytes_per_pixel

    @property
    def policy_id(self) -> PrivacyPolicyId | None:
        return self._policy_id

    def apply(self, slot: WritableSlot, dimensions: FrameDimensions) -> MaskOutcome:
        expected = dimensions.width * dimensions.height * self._bytes_per_pixel
        memory = slot.memory()
        if expected > len(memory):
            raise PrivacyMaskError(
                f"cannot mask: frame needs {expected}B but slot holds {len(memory)}B",
                policy_id=str(self._policy_id),
            )

        masked = 0
        for zone in self._zones:
            bounds = zone.bounds
            x1 = max(0, int(bounds.x1 * dimensions.width))
            x2 = min(dimensions.width, int(bounds.x2 * dimensions.width) + 1)
            y1 = max(0, int(bounds.y1 * dimensions.height))
            y2 = min(dimensions.height, int(bounds.y2 * dimensions.height) + 1)
            if x2 <= x1 or y2 <= y1:
                continue
            stride = dimensions.width * self._bytes_per_pixel
            span = (x2 - x1) * self._bytes_per_pixel
            blank = bytes(span)
            for row in range(y1, y2):
                start = row * stride + x1 * self._bytes_per_pixel
                memory[start : start + span] = blank
            masked += 1

        return MaskOutcome(state=PrivacyState.MASKED, regions_masked=masked)


class FailingMask:
    """Always fails. Exercises the fail-closed path in tests and drills.

    Kept in the shipped adapter set deliberately: the fail-closed behaviour is a
    compliance guarantee, and a guarantee with no way to rehearse it is a
    guarantee nobody has verified.
    """

    __slots__ = ("_policy_id", "_reason")

    def __init__(
        self,
        *,
        policy_id: PrivacyPolicyId = _FAILING_POLICY_ID,
        reason: str = "synthetic masking failure",
    ) -> None:
        self._policy_id = policy_id
        self._reason = reason

    @property
    def policy_id(self) -> PrivacyPolicyId | None:
        return self._policy_id

    def apply(self, slot: WritableSlot, dimensions: FrameDimensions) -> MaskOutcome:
        raise PrivacyMaskError(self._reason, policy_id=str(self._policy_id))
