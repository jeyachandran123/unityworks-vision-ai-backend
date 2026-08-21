"""P6 ChangeDetectorPort — suppress frames that carry no new information.

The highest-value scheduler extension: in most real deployments the majority of
frames contain nothing new, and suppressing them here is the cheapest possible
saving (invariant V7).

Sampling rather than full-frame comparison is deliberate. At ~3000 frames a
second a full hash costs more than the saving is worth, and a stride-sampled
digest detects the cases that matter — a static scene and a frozen stream.
"""

from __future__ import annotations

import threading

from ...core.model.frame import FrameDimensions
from ...core.model.ids import CameraId
from ...core.ports.scheduling import ChangeVerdict


class NullChangeDetector:
    """Every frame is new. The default: suppression is opt-in per deployment."""

    __slots__ = ()

    def observe(
        self, camera_id: CameraId, view: memoryview, dimensions: FrameDimensions
    ) -> ChangeVerdict:
        return ChangeVerdict(changed=True, score=1.0)

    def forget(self, camera_id: CameraId) -> None:
        return None


class SampledDigestChangeDetector:
    """Detect change by comparing a stride-sampled digest of consecutive frames.

    Exact-match only: it reports "identical" or "different", never a similarity
    score it cannot honestly compute. A perceptual-difference detector is a
    sibling adapter behind the same port.
    """

    def __init__(self, *, samples: int = 64) -> None:
        if samples < 1:
            raise ValueError(f"samples must be >= 1, got {samples}")
        self._samples = samples
        self._lock = threading.Lock()
        self._last: dict[CameraId, int] = {}

    def observe(
        self, camera_id: CameraId, view: memoryview, dimensions: FrameDimensions
    ) -> ChangeVerdict:
        digest = self._digest(view)
        with self._lock:
            previous = self._last.get(camera_id)
            self._last[camera_id] = digest
        if previous is None or previous != digest:
            return ChangeVerdict(changed=True, score=1.0)
        return ChangeVerdict(changed=False, score=0.0)

    def forget(self, camera_id: CameraId) -> None:
        with self._lock:
            self._last.pop(camera_id, None)

    def _digest(self, view: memoryview) -> int:
        length = len(view)
        if length == 0:
            return 0
        step = max(1, length // self._samples)
        return hash(bytes(view[::step]))
