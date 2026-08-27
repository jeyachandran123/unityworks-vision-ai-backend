"""Whether the body region a question asks about is actually in the evidence.

### Why this is not the quality gate

The quality gate answers *"is this crop physically usable?"* — scale, blur,
occlusion, exposure. This answers a different question: *"is the thing the
question is about present in the picture at all?"* A crop can be sharp, large,
well-exposed and contain no head.

The distinction is not theoretical. `config/policies/kitchen-safety.example.json`
records the measurement that forced it: of 13 heads a human could not read in
`datasets/kitchen-01`, 2 were blurred past the blur floor and **11 were perfectly
sharp** — unreadable because the head was turned away, bent down or outside the
box, *"which no quality axis can detect"*.

Those 11 are the ones that become confident violations against people nobody
could see. A quality axis cannot reach them, because nothing is wrong with the
pixels; something is missing from them.

### Three states, and deliberately no fourth

``LOCATED`` · ``LOW_CONFIDENCE`` · ``NOT_LOCATED``, plus ``UNSUPPORTED`` for a
producer that cannot speak to the region at all.

**There is no state meaning "absent".** A region observability producer sees
where a body part is, never what is on it. Letting it express absence would make
it a second, unaccountable attribute producer sitting outside the registry's
neutrality gate — which is exactly the Semantic Ceiling this platform enforces
everywhere else.

``LOW_CONFIDENCE`` exists because it is the case where the producer knows it is
guessing. Folding it into ``LOCATED`` hands the model the crop most likely to be
misread; folding it into ``NOT_LOCATED`` erases the distinction. Phase 4.4
measured both sides of that trade rather than assuming one.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .ids import AttributeKey


class RegionState(enum.Enum):
    """What a producer could establish about the region."""

    LOCATED = "located"
    """The region was found. A claim about it may be attempted."""

    LOW_CONFIDENCE = "low_confidence"
    """Signal present but below the floor — the producer knows it is guessing."""

    NOT_LOCATED = "not_located"
    """No signal at all. A confident absence, and the safest thing to act on."""

    UNSUPPORTED = "unsupported"
    """This producer cannot speak to this region.

    Declared rather than inferred, and it is **never** grounds for refusal: an
    attribute nobody can assess must behave exactly as it did before this port
    existed (V8 — a capability gap must be visible, not silently restrictive).
    """

    @property
    def is_observable(self) -> bool:
        """Whether a claim about this region may be attempted.

        ``UNSUPPORTED`` is observable **on purpose**. The producer said it had no
        opinion; treating no-opinion as refusal would let binding a partial
        adapter silently blind every attribute it does not cover.
        """
        return self in (RegionState.LOCATED, RegionState.UNSUPPORTED)

    @property
    def is_refusal(self) -> bool:
        return self in (RegionState.LOW_CONFIDENCE, RegionState.NOT_LOCATED)


@dataclass(frozen=True, slots=True)
class RegionVerdict:
    """One producer's answer about one region of one subject.

    ``detail`` is required on a refusal for the reason ``GateResult`` requires
    one: *"the VLM never answers for far-away people"* must be a statistic with a
    name rather than a mystery (02_VOM §10.7).
    """

    attribute: AttributeKey
    state: RegionState
    confidence: float = 0.0
    """The producer's own strength of signal, in [0,1]. Not a probability that
    the region is observable — producers differ in what they measure, and
    pretending otherwise would invent a calibration nobody performed."""

    signals_seen: int = 0
    """How many independent signals supported the verdict — keypoints, in the
    pose producer's case. Zero is normal for a refusal and is not an error."""

    producer_id: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        if self.signals_seen < 0:
            raise ValueError("signals_seen must be non-negative")
        if self.state.is_refusal and not self.detail:
            raise ValueError(
                "a refused region must name why; an unattributed refusal is a "
                "statistic nobody can act on"
            )

    @property
    def observable(self) -> bool:
        return self.state.is_observable
