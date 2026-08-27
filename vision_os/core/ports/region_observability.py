"""P34 — ``RegionObservabilityPort``. Owner: M8 Crop Manager.

*"Is the body region this attribute asks about actually present in the
evidence?"* — asked **before** an expensive model is asked what is on it.

The port exists so that answering it is a swappable adapter decision. Pose is the
first producer; a segmentation mask, a face detector or a learned observability
head are siblings behind the same three members, and none of them changes a line
of the crop path.

### Obligations

An implementation must:

* **O1** — return exactly one verdict per requested attribute, no more and no
  fewer. A caller must never have to guess whether silence meant *observable* or
  *unassessed*.
* **O2** — declare its coverage in ``capabilities()``, and answer
  ``UNSUPPORTED`` for anything outside it rather than guessing. An adapter that
  quietly refuses attributes it does not model blinds a deployment on binding.
* **O3** — never express *absence of a covering*. This port reports where a body
  part is, never what is on it. Emitting a judgment here would route an attribute
  around the registry's neutrality gate.
* **O4** — be deterministic for a given frame and box, so a refusal can be
  reproduced from a replay six months later (V13).
* **O5** — never raise on legal-but-extreme geometry. A degenerate box is a
  ``NOT_LOCATED`` with a reason, not an exception; the crop path must degrade,
  never die (V9).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..model.ids import AttributeKey, CameraId
from ..model.region_observability import RegionVerdict
from ..model.space import Box


@dataclass(frozen=True, slots=True)
class RegionObservabilityRequest:
    """One subject, one frame, and the questions about to be asked of it.

    Carries the **source frame** rather than the crop. A producer that needs to
    locate a body part relative to the whole person cannot do it from a band
    already cut out of them — and a producer given the crop would be answering
    "is the head in the head crop", which is circular.
    """

    camera_id: CameraId
    box: Box
    """The subject box, in normalized frame coordinates."""

    attributes: tuple[AttributeKey, ...]
    """What is about to be asked. One verdict comes back per entry (O1)."""

    source_width: int
    source_height: int
    pixels: memoryview | None = None
    """Full-frame pixels. ``None`` means the caller could not supply them, and a
    producer must answer ``UNSUPPORTED`` rather than inventing a verdict."""

    colour_space: str = "bgr24"
    frame_key: str = ""
    """Opaque per-frame identity, so a producer may cache one inference across
    the several subjects in a frame. Never interpreted by the platform."""

    def __post_init__(self) -> None:
        if not self.attributes:
            raise ValueError("a request must name at least one attribute")


@dataclass(frozen=True, slots=True)
class RegionObservabilityCapabilities:
    """What this producer can speak to (O2)."""

    producer_id: str
    assessable_attributes: frozenset[AttributeKey] = field(default_factory=frozenset)
    requires_pixels: bool = True
    deterministic: bool = True


@runtime_checkable
class RegionObservabilityPort(Protocol):
    """P34 — locate the region a question is about, before paying to ask it."""

    def capabilities(self) -> RegionObservabilityCapabilities:
        """Declared honestly, and captured at binding rather than per request."""
        ...

    def assess(self, request: RegionObservabilityRequest) -> Sequence[RegionVerdict]:
        """One verdict per requested attribute, in request order (O1)."""
        ...
