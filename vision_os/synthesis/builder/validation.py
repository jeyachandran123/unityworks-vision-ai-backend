"""The final semantic gate — the third and last of the ceiling's three.

> `00_CHARTER` §4.3: *"**The Observation Builder.** It refuses to emit an
> observation containing an attribute absent from the registry."*

*Final* is literal. `07_STATE` §1.1 says state is derived from observations and
has no other write path; M14 serves state read-only. Nothing downstream
re-checks, so an attribute that passes here is a platform fact forever.

**The two responses, kept apart deliberately.** 04_MODULES §M11's failure table
prescribes opposite treatments, and conflating them fails in both directions:

| Failure | Response | What conflating it would cost |
|---|---|---|
| Attribute not in the registry | **Drop the attribute**, keep the observation | Rejecting the whole envelope loses a valid presence fact because one enrichment was bad |
| Envelope incomplete | **Reject the observation entirely**, alarm | Publishing it puts an unauditable fact in the permanent record — *"worse than no observation"* |

That asymmetry is the module. Everything else here is bookkeeping around it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from ...core.model.confidence import ConfidenceSemantics
from ...core.model.ids import AttributeKey, ClassId
from ...core.model.observation import (
    Observation,
    ObservationType,
    ValidationResult,
    Violation,
    ViolationKind,
)
from ...core.model.visual_object import Attribute
from ...perception.registry.attributes import AttributeRegistry, SchemaStatus


@dataclass(frozen=True, slots=True)
class TaxonomyView:
    """What the builder needs to know about the taxonomy, injected.

    A *view* rather than the registry itself: M11 needs to answer two questions —
    "is this class registered" and "are we all on the same version" — and handing
    it the full taxonomy would let it reach for class *meaning*, which it must
    never have.
    """

    version: str = ""
    classes: frozenset[ClassId] = frozenset()

    def knows(self, class_id: ClassId) -> bool:
        """Whether this class is registered.

        An empty class set means "not enforcing", which is the honest degenerate
        case for a deployment that has not declared a taxonomy — not a silent
        pass for one that has.
        """
        if not self.classes:
            return True
        return class_id in self.classes or any(
            class_id.startswith(f"{known}.") for known in self.classes
        )


class CeilingGate:
    """Applies the Attribute Schema Registry and envelope completeness.

    Pure and stateless: the same candidate always produces the same result, which
    is what lets a rejection be reproduced from a replay six months later (V13).
    Counting is the engine's job, not the gate's.
    """

    __slots__ = ("_registry", "_taxonomy")

    def __init__(
        self, registry: AttributeRegistry, taxonomy: TaxonomyView | None = None
    ) -> None:
        self._registry = registry
        self._taxonomy = taxonomy or TaxonomyView()

    @property
    def registry(self) -> AttributeRegistry:
        return self._registry

    @property
    def taxonomy(self) -> TaxonomyView:
        return self._taxonomy

    def validate(self, candidate: Observation) -> ValidationResult:
        """Put a candidate through the final gate.

        Envelope checks run **first**: if the observation cannot be audited there
        is no point examining its content, and reporting attribute problems on an
        envelope that will be rejected anyway sends an operator to the wrong fix.
        """
        envelope = self._check_envelope(candidate)
        if envelope:
            return ValidationResult(observation=None, violations=tuple(envelope))

        kept, dropped, attribute_violations = self._filter_attributes(candidate)

        if candidate.observation_type is ObservationType.ATTRIBUTE and not kept:
            # Every attribute was refused. An attribute observation with no
            # attributes says nothing, so there is nothing left to publish — but
            # the violations survive so the rejection is countable.
            return ValidationResult(
                observation=None,
                violations=tuple(attribute_violations),
                dropped_attributes=tuple(dropped),
            )

        narrowed = (
            candidate if len(kept) == len(candidate.attributes)
            else replace(candidate, attributes=tuple(kept))
        )
        return ValidationResult(
            observation=narrowed,
            violations=tuple(attribute_violations),
            dropped_attributes=tuple(dropped),
        )

    # --- envelope completeness (V4) ------------------------------------------ #

    def _check_envelope(self, candidate: Observation) -> list[Violation]:
        """§M11 responsibility 3: no observation without provenance, timing and
        evidence.

        ``Observation.__post_init__`` already refuses to construct an envelope
        with no producer or no config revision, so those cannot reach here. What
        this adds is the checks that need *context* the type cannot see: whether
        the taxonomy agrees, whether the class is registered, and whether an
        evidence-bearing type actually carries evidence.
        """
        violations: list[Violation] = []

        if candidate.observation_type is ObservationType.ATTRIBUTE:
            if candidate.evidence_ref is None:
                violations.append(
                    Violation(
                        kind=ViolationKind.MISSING_EVIDENCE,
                        detail=(
                            "an attribute observation must reference the evidence "
                            "behind its claims; a claim that cannot name the "
                            "pixels it rests on is not auditable (V4)"
                        ),
                    )
                )
            if candidate.timing.total_ms <= 0.0:
                violations.append(
                    Violation(
                        kind=ViolationKind.MISSING_TIMING,
                        detail=(
                            "an attribute observation must carry timing; without "
                            "it a cost or latency question about this fact is "
                            "unanswerable"
                        ),
                    )
                )

        if candidate.class_id is not None and not self._taxonomy.knows(
            candidate.class_id
        ):
            violations.append(
                Violation(
                    kind=ViolationKind.UNREGISTERED_CLASS,
                    detail=(
                        f"class '{candidate.class_id}' is not in the declared "
                        f"taxonomy; publishing it would put a class in the record "
                        f"that no consumer can resolve"
                    ),
                    field_name=str(candidate.class_id),
                )
            )

        if (
            self._taxonomy.version
            and candidate.taxonomy_version
            and candidate.taxonomy_version != self._taxonomy.version
        ):
            violations.append(
                Violation(
                    kind=ViolationKind.TAXONOMY_VERSION_MISMATCH,
                    detail=(
                        f"observation declares taxonomy {candidate.taxonomy_version} "
                        f"but this site runs {self._taxonomy.version}; this "
                        f"indicates a partial deployment and must be loud "
                        f"(04_MODULES section M11)"
                    ),
                )
            )

        return violations

    # --- the ceiling ---------------------------------------------------------- #

    def _filter_attributes(
        self, candidate: Observation
    ) -> tuple[list[Attribute], list[AttributeKey], list[Violation]]:
        """Keep what the registry recognises; drop and record the rest.

        Iteration is over the observation's own order, which is already stable
        because M9 sorted it — so two runs over the same inputs drop the same
        attributes in the same order (V13).
        """
        kept: list[Attribute] = []
        dropped: list[AttributeKey] = []
        violations: list[Violation] = []

        for attribute in candidate.attributes:
            problem = self._reject_reason(attribute, candidate.class_id)
            if problem is None:
                kept.append(attribute)
                continue
            dropped.append(attribute.key)
            violations.append(problem)

        return kept, dropped, violations

    def _reject_reason(
        self, attribute: Attribute, class_id: ClassId | None
    ) -> Violation | None:
        key = attribute.key
        schema = self._registry.get(key)

        if schema is None:
            return Violation(
                kind=ViolationKind.UNREGISTERED_ATTRIBUTE,
                detail=(
                    f"attribute '{key}' is not in the Attribute Schema Registry; "
                    f"a key the registry does not hold has not passed the "
                    f"neutrality gate (00_CHARTER section 4.3)"
                ),
                field_name=str(key),
            )

        if schema.status is not SchemaStatus.ACTIVE:
            return Violation(
                kind=ViolationKind.DEPRECATED_ATTRIBUTE,
                detail=f"attribute '{key}' is deprecated and no longer published",
                field_name=str(key),
            )

        if class_id is not None and not self._registry.applies(key, class_id):
            return Violation(
                kind=ViolationKind.ATTRIBUTE_NOT_APPLICABLE,
                detail=(
                    f"attribute '{key}' does not apply to class '{class_id}'"
                ),
                field_name=str(key),
            )

        problem = schema.accepts(attribute.value)
        if problem is not None:
            return Violation(
                kind=ViolationKind.ATTRIBUTE_VALUE_INVALID,
                detail=f"attribute '{key}': {problem}",
                field_name=str(key),
            )

        if attribute.confidence.semantics not in (
            ConfidenceSemantics.ATTRIBUTE,
            ConfidenceSemantics.SELF_REPORTED,
        ):
            return Violation(
                kind=ViolationKind.WRONG_CONFIDENCE_SEMANTICS,
                detail=(
                    f"attribute '{key}' carries "
                    f"{attribute.confidence.semantics.value} confidence; a "
                    f"detector's objectness score answers a different question "
                    f"from an attribute claim, and the two are not comparable "
                    f"(02_VOM section 7.2)"
                ),
                field_name=str(key),
            )

        return None


def ceiling_violations(violations: Sequence[Violation]) -> tuple[Violation, ...]:
    """The subset that is the Semantic Ceiling doing its job.

    Counted apart from ordinary schema mismatches because a sustained rate means
    a **producer has drifted** — a prompt change, a new model, a partial
    deployment — which is a different operator response from a model that formats
    badly.
    """
    return tuple(
        v for v in violations if v.kind is ViolationKind.UNREGISTERED_ATTRIBUTE
    )


def summarize(violations: Sequence[Violation]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for violation in violations:
        counts[violation.kind.value] = counts.get(violation.kind.value, 0) + 1
    return counts
