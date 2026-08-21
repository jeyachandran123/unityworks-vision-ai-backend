"""The verification trigger policy — P12, composed rather than replacing.

### Why this is a P12 adapter and not a new stage

The question a verification policy answers is *"is the visual evidence we already
hold sufficient, or must we look again?"* — which is, word for word, the question
``TriggerPolicyPort`` exists to answer. Its obligations are already the ones a
verification policy needs: every candidate produces exactly one decision (G1),
carrying a reason or a skip and never both (G2), deterministically (G3),
reasoning about measurements and ids rather than meanings (G4).

So this wraps a trigger policy rather than becoming a second one, exactly as
``ExplicitRequestPolicy`` does. Everything not governed by a verification rule
falls through untouched, which is what keeps ordinary understanding unchanged.

### What it actually does, which is the opposite of what it sounds like

A demand for a corroborating attribute would, on its own, fire
``ATTRIBUTE_MISSING`` for **every** object in its scope — one model call per
object per freshness window. That is the brute-force pipeline this seam exists to
prevent.

So the policy's job is subtractive: the inner policy proposes, and this policy
**withdraws the proposal for every candidate whose detector claim already stands
on its own**. What survives is the narrow set where the detector genuinely cannot
answer — a closed-set model naming something outside its vocabulary, an
ambiguous score, an object whose class has been flapping.

### The ceiling holds here

This policy reads a confidence, a label-space kind, a set of class ids, an
evidence share and a quality grade. It never learns what a class *means*, and the
rule document that configures it contains thresholds and class ids — never a
domain, a role, or a verdict. A rule may say *"require corroboration when the
detector is closed-set and scores below 0.65"*. It cannot say *"require
corroboration in the kitchen"*.

### Why the document is loaded elsewhere

``from_document`` lives here and ``from_file`` does not. M8 holds no durable
store, and its architecture test enforces that structurally by refusing any
import of ``pathlib`` or ``os`` anywhere in the crop path — *"a vocabulary guard
can be worked around by naming a method something else; an import guard cannot"*.

Reading a file is a composition-time act, so it belongs with the other
composition-time loaders in ``adapters.configuration``, exactly where the
semantic policy's loader already sits. What remains here is a pure function from
a parsed document to rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ...core.errors import ConfigurationError
from ...core.model.crop import SkipReason, TriggerReason
from ...core.model.detection import QualityGrades
from ...core.model.ids import AttributeKey, ClassId
from ...core.ports.cropping import (
    CLOSED_SET,
    OPEN_VOCABULARY,
    TriggerCandidate,
    TriggerDecision,
)

#: Label-space kinds a rule may name. Closed by design: an unknown kind in a
#: document is a typo, and silently ignoring it would leave a deployment
#: believing it had configured scrutiny it had not.
_KNOWN_LABEL_SPACES = frozenset({CLOSED_SET, OPEN_VOCABULARY})


@dataclass(frozen=True, slots=True)
class CropRequirements:
    """The minimum input quality a corroborating look is worth paying for.

    A pre-extraction check on grades the platform already estimated, so a
    hopeless candidate is refused before it costs a crop *and* a model call. The
    real gate still runs after extraction; this only avoids paying to reach it.

    Every bound is ``None`` by default — *unconstrained*, not *zero*. A rule that
    declares no floor gets no floor, rather than one it never asked for.
    """

    min_scale_pixels: float | None = None
    max_truncation: float | None = None
    max_occlusion: float | None = None

    def unmet(self, grades: QualityGrades) -> str:
        """Which requirement this input fails, or ``""`` if none.

        An **unmeasured** grade never fails a requirement. "Not measured" and
        "measured as bad" are different claims (obligation Q2), and refusing on
        the first would make an unwired quality estimator look like a site full
        of unusable cameras.
        """
        if (
            self.min_scale_pixels is not None
            and grades.scale_pixels is not None
            and grades.scale_pixels < self.min_scale_pixels
        ):
            return (
                f"scale {grades.scale_pixels:.0f}px below the "
                f"{self.min_scale_pixels:.0f}px floor for corroboration"
            )
        if (
            self.max_truncation is not None
            and grades.truncation is not None
            and grades.truncation > self.max_truncation
        ):
            return f"truncation {grades.truncation:.2f} above {self.max_truncation:.2f}"
        if (
            self.max_occlusion is not None
            and grades.occlusion is not None
            and grades.occlusion > self.max_occlusion
        ):
            return f"occlusion {grades.occlusion:.2f} above {self.max_occlusion:.2f}"
        return ""


@dataclass(frozen=True, slots=True)
class TrustConditions:
    """When the detector's own claim is *not* enough.

    Conditions combine with **or**: any one firing means corroboration is
    warranted. That is the correct combination for this question — a claim is
    doubtful if it is ambiguous, *or* outside the model's vocabulary, *or*
    unstable over time, and requiring all three at once would mean the pen at
    0.454 sails through because it happened to be stable.

    A ``TrustConditions`` with nothing set fires for nothing, so a rule that
    declares no condition costs no model calls rather than every one.
    """

    label_spaces: frozenset[str] = frozenset()
    """Detector label-space kinds this rule scrutinises. Empty means *any*.

    A candidate whose label space is **undeclared** never matches a non-empty
    set. Undeclared means the deployment has not told the platform what its
    detector can name, and treating that as "outside the vocabulary" would send
    every object to a model on a configuration mistake."""

    confidence_below: float | None = None
    """Fires when the detector's class score is under this. An **absent** score
    does not fire: no class evidence retained is not the same as weak evidence."""

    outside_vocabulary: bool = False
    """Fires when the class is explicitly outside the detector's declared
    vocabulary. ``class_in_native_vocabulary is None`` — undeclared — never
    fires, for the reason ``label_spaces`` gives."""

    instability_above: float | None = None
    """Fires when the share of retained class evidence assigned to classes
    *other* than the published one exceeds this. A tracked object called three
    different things is making a weaker claim than its top score suggests, and
    that is invisible in the score alone."""

    classes: frozenset[ClassId] = frozenset()
    """Classes that always warrant corroboration, whatever the score.

    Opaque ids supplied by the document. The policy orders and matches by them
    and never asks what any of them means (V1/V2)."""

    def triggered(self, candidate: TriggerCandidate) -> str:
        """Why this candidate's claim needs corroborating, or ``""`` if it does not.

        A string rather than a boolean because the answer lands on
        ``TriggerDecision.detail`` and from there on the evidence: *"we spent a
        model call"* is only defensible if the reason travels with the result.
        """
        if self.label_spaces and candidate.label_space_kind not in self.label_spaces:
            return ""

        if self.classes and _matches_class(candidate.class_id, self.classes):
            return f"class '{candidate.class_id}' is declared identity-sensitive"

        if (
            self.confidence_below is not None
            and candidate.class_confidence is not None
            and candidate.class_confidence.value < self.confidence_below
        ):
            return (
                f"class score {candidate.class_confidence.value:.3f} is below the "
                f"{self.confidence_below:.3f} floor for a standalone claim"
            )

        if self.outside_vocabulary and candidate.class_in_native_vocabulary is False:
            return (
                f"class '{candidate.class_id}' is outside the detector's declared "
                f"vocabulary; a closed-set model returns the nearest word it knows"
            )

        if self.instability_above is not None:
            share = sum(weight for _, weight in candidate.class_alternatives)
            if share > self.instability_above:
                return (
                    f"{share:.2f} of retained class evidence names "
                    f"{len(candidate.class_alternatives)} other class(es)"
                )

        return ""

    @property
    def is_empty(self) -> bool:
        return not (
            self.confidence_below is not None
            or self.outside_vocabulary
            or self.instability_above is not None
            or self.classes
        )


@dataclass(frozen=True, slots=True)
class VerificationRule:
    """One rule: which attributes it governs, and when they are worth computing."""

    rule_id: str
    attributes: frozenset[AttributeKey]
    """The corroborating attributes this rule governs. A demand for one of these
    is subject to the trust conditions below; every other demand falls through
    untouched."""

    require_when: TrustConditions = TrustConditions()
    crop_must: CropRequirements = CropRequirements()
    subject_classes: frozenset[ClassId] = frozenset()
    """Narrows the rule to particular classes. Empty means every class."""

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ConfigurationError("a verification rule requires a rule_id")
        if not self.attributes:
            raise ConfigurationError(
                f"verification rule '{self.rule_id}' governs no attribute; a rule "
                f"that governs nothing would never be consulted and would sit in "
                f"the document looking as though it did something"
            )
        if self.require_when.is_empty:
            raise ConfigurationError(
                f"verification rule '{self.rule_id}' declares no condition under "
                f"which corroboration is required. Such a rule suppresses every "
                f"call to the attributes it governs, which is almost certainly not "
                f"what was meant; remove the rule instead"
            )

    def covers(self, candidate: TriggerCandidate) -> bool:
        if not self.subject_classes:
            return True
        return _matches_class(candidate.class_id, self.subject_classes)


@dataclass(frozen=True, slots=True)
class VerificationRules:
    """A deployment's verification document, as data.

    Carries no behaviour beyond answering "which rule governs this attribute for
    this candidate". Two deployments differ only in contents, which is the
    property that makes a new use case a file rather than a release.
    """

    version: str = "1.0.0"
    rules: tuple[VerificationRule, ...] = ()

    @property
    def governed_attributes(self) -> frozenset[AttributeKey]:
        keys: set[AttributeKey] = set()
        for rule in self.rules:
            keys |= rule.attributes
        return frozenset(keys)

    def governing(
        self, candidate: TriggerCandidate, key: AttributeKey
    ) -> VerificationRule | None:
        """The first rule governing this attribute for this candidate.

        First rather than best: document order is the tie-break, so the answer is
        stable and a deployment can reason about precedence by reading top to
        bottom (V13).
        """
        for rule in self.rules:
            if key in rule.attributes and rule.covers(candidate):
                return rule
        return None

    # --- construction ------------------------------------------------------- #

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> VerificationRules:
        entries = document.get("rules", ())
        if not isinstance(entries, list | tuple):
            raise ConfigurationError("'rules' must be a list")
        return cls(
            version=str(document.get("version", "1.0.0")),
            rules=tuple(_rule_from(entry) for entry in entries),
        )


class VerificationPolicy:
    """P12 — withdraw corroboration requests the detector's own claim can cover.

    Wraps an inner policy. Obligations G1–G3 are preserved structurally: exactly
    one decision comes back per candidate, in input order, and every path either
    returns the inner decision or replaces it with one carrying exactly one of
    reason or skip.
    """

    __slots__ = ("_inner", "_rules")

    def __init__(self, inner, rules: VerificationRules) -> None:
        self._inner = inner
        self._rules = rules

    @property
    def policy_id(self) -> str:
        return f"trigger.verification({self._inner.policy_id})"

    @property
    def rules(self) -> VerificationRules:
        return self._rules

    def evaluate(
        self,
        candidates: Sequence[TriggerCandidate],
        *,
        now,
        demands: Sequence[object],
    ) -> Sequence[TriggerDecision]:
        inner = list(self._inner.evaluate(candidates, now=now, demands=demands))
        if len(inner) != len(candidates):
            # The inner policy broke G1. Surfacing it here rather than papering
            # over it keeps the guarantee the engine relies on.
            raise ValueError(
                f"inner policy '{self._inner.policy_id}' returned {len(inner)} "
                f"decisions for {len(candidates)} candidates"
            )
        return [
            self._refine(candidate, decision)
            for candidate, decision in zip(candidates, inner, strict=True)
        ]

    # --- internals ----------------------------------------------------------- #

    def _refine(
        self, candidate: TriggerCandidate, decision: TriggerDecision
    ) -> TriggerDecision:
        """Withdraw, narrow, relabel or pass through — exactly one of the four."""
        if not decision.fires:
            # A skip stands. The inner policy already decided nothing is wanted,
            # is fresh enough, or cannot be afforded, and a verification rule is
            # not a licence to overrule a budget or a freshness window.
            return decision

        governed = tuple(
            key
            for key in decision.attributes
            if self._rules.governing(candidate, AttributeKey(key)) is not None
        )
        if not governed:
            return decision

        warranted, detail, unmet = self._assess(candidate, governed)

        if warranted:
            # Relabel rather than keep the inner reason: for a governed
            # attribute, *why* the platform is about to spend a model call is
            # that the identity does not stand on its own, and that is the reason
            # which lands on the evidence.
            return replace(
                decision,
                reason=TriggerReason.IDENTITY_UNVERIFIED,
                skip=None,
                detail=detail,
            )

        remainder = tuple(key for key in decision.attributes if key not in set(governed))
        if remainder:
            # Some attributes are governed and some are not. Narrow rather than
            # withdraw: suppressing the whole decision would silently stop
            # serving demands this policy has no opinion about.
            return replace(
                decision,
                attributes=remainder,
                detail=(
                    f"{decision.detail}; corroboration withdrawn for "
                    f"{len(governed)} attribute(s): {detail}"
                ).lstrip("; "),
            )

        if unmet:
            return TriggerDecision(
                object_id=candidate.object_id,
                skip=SkipReason.QUALITY_INSUFFICIENT,
                attributes=governed,
                priority_class=decision.priority_class,
                demand_ids=decision.demand_ids,
                detail=unmet,
            )

        return TriggerDecision(
            object_id=candidate.object_id,
            skip=SkipReason.EVIDENCE_SUFFICIENT,
            attributes=governed,
            priority_class=decision.priority_class,
            demand_ids=decision.demand_ids,
            detail=detail,
        )

    def _assess(
        self, candidate: TriggerCandidate, governed: Sequence[str]
    ) -> tuple[bool, str, str]:
        """Is corroboration warranted, why, and what quality bar blocks it.

        Trust is evaluated **before** quality deliberately. A candidate whose
        detector claim already stands is skipped as ``EVIDENCE_SUFFICIENT`` even
        on a poor crop — reporting ``QUALITY_INSUFFICIENT`` there would blame the
        camera for a call the policy was never going to make, and would make the
        gate-rejection statistic unreadable.
        """
        reasons: list[str] = []
        unmet: list[str] = []

        for key in governed:
            rule = self._rules.governing(candidate, AttributeKey(key))
            if rule is None:  # pragma: no cover - governed implies a rule
                continue
            why = rule.require_when.triggered(candidate)
            if not why:
                continue
            blocked = rule.crop_must.unmet(candidate.estimated_quality)
            if blocked:
                unmet.append(blocked)
                continue
            reasons.append(why)

        if reasons:
            return True, "; ".join(dict.fromkeys(reasons)), ""
        if unmet:
            return False, "", "; ".join(dict.fromkeys(unmet))
        return False, "detector claim is sufficient for the demanded evidence", ""


# --- document parsing --------------------------------------------------------- #


def _rule_from(entry: Mapping[str, Any]) -> VerificationRule:
    if not isinstance(entry, Mapping):
        raise ConfigurationError("each verification rule must be an object")

    require = entry.get("require_when", {}) or {}
    crop = entry.get("crop_must", {}) or {}

    spaces = frozenset(str(k) for k in require.get("detector_label_space", ()))
    unknown = spaces - _KNOWN_LABEL_SPACES
    if unknown:
        raise ConfigurationError(
            f"unknown detector label space(s) {sorted(unknown)}; supported: "
            f"{sorted(_KNOWN_LABEL_SPACES)}"
        )

    return VerificationRule(
        rule_id=str(entry.get("rule_id", "")).strip(),
        attributes=frozenset(
            AttributeKey(str(k)) for k in entry.get("attributes", ())
        ),
        subject_classes=frozenset(
            ClassId(str(c)) for c in entry.get("subject_classes", ())
        ),
        require_when=TrustConditions(
            label_spaces=spaces,
            confidence_below=_ratio(require.get("class_confidence_below"), "class_confidence_below"),
            outside_vocabulary=bool(require.get("class_outside_native_vocabulary", False)),
            instability_above=_ratio(require.get("class_instability_above"), "class_instability_above"),
            classes=frozenset(ClassId(str(c)) for c in require.get("classes", ())),
        ),
        crop_must=CropRequirements(
            min_scale_pixels=_positive(crop.get("min_scale_pixels"), "min_scale_pixels"),
            max_truncation=_ratio(crop.get("max_truncation"), "max_truncation"),
            max_occlusion=_ratio(crop.get("max_occlusion"), "max_occlusion"),
        ),
    )


def _matches_class(class_id: ClassId, allowed: frozenset[ClassId]) -> bool:
    """Hierarchical, like every other class filter in the platform."""
    return any(
        class_id == candidate or class_id.startswith(f"{candidate}.")
        for candidate in allowed
    )


def _ratio(value: Any, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ConfigurationError(f"'{name}' must be in [0,1], got {number}")
    return number


def _positive(value: Any, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if number < 0:
        raise ConfigurationError(f"'{name}' must be non-negative, got {number}")
    return number


__all__ = [
    "CropRequirements",
    "TrustConditions",
    "VerificationPolicy",
    "VerificationRule",
    "VerificationRules",
]
