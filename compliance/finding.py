"""What a rule evaluation produced, and everything needed to defend it.

A ``Finding`` is the compliance layer's equivalent of Vision OS's ``Observation``:
immutable, fully attributed, and carrying its own receipt. The parallel is
deliberate — the platform's rule that *"an unexplainable observation is worse than
no observation"* applies with more force here, because a finding is what someone
acts on.

Three properties are enforced at construction rather than by convention:

``state`` is three-valued, never boolean
    ``UNKNOWN`` is a first-class answer. A rule engine that could only say yes or
    no would have to turn missing evidence into one of them, and both choices are
    wrong: absent evidence read as compliant hides real violations, and absent
    evidence read as a violation manufactures them.

a violation must name what failed
    A ``VIOLATION`` with no failed condition is a verdict nobody can act on or
    appeal.

an unknown must name why
    Same reason, and it is the field that tells an operator whether the fix is a
    camera, a budget, a model or a rule.

**Nothing here holds imagery.** Evidence travels as an ``EvidenceId`` handle that
the reader resolves under its own authorization, because reading *"a person was
here"* and viewing their image are categorically different acts (12_SECURITY §5.3).
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field

from vision_os.core.model.ids import (
    CameraId,
    ClassId,
    ObjectId,
    SiteId,
    TenantId,
)
from vision_os.core.model.timebase import Duration, Instant

FINDING_SCHEMA_VERSION = "1.0.0"


class ComplianceState(enum.Enum):
    """The three answers a rule may reach. Closed set."""

    COMPLIANT = "compliant"
    """Every required condition was established and held."""

    VIOLATION = "violation"
    """At least one required condition was established and did not hold."""

    UNKNOWN = "unknown"
    """No condition failed, and at least one could not be established. The
    platform could not see, could not afford to look, or has not been asked to
    produce the evidence this rule depends on."""

    NOT_APPLICABLE = "not_applicable"
    """The rule's guard did not match this subject. A conditional rule — *"if the
    food is X then the board must be X"* — says nothing at all about a subject
    whose food is Y, and reporting that as compliant would inflate a pass rate
    with subjects the rule never examined."""

    @property
    def is_decided(self) -> bool:
        """Whether evidence was sufficient to reach a verdict."""
        return self in (ComplianceState.COMPLIANT, ComplianceState.VIOLATION)

    @property
    def needs_attention(self) -> bool:
        return self is ComplianceState.VIOLATION


class UnknownReason(enum.Enum):
    """Why a condition could not be established.

    Every value names a *mechanism the platform reported*, translated into what
    it means for a rule. The translation lives here rather than in Vision OS
    because the platform reports that it could not see and why; deciding what
    that means for a business rule is this layer's judgment (invariant V1).
    """

    ATTRIBUTE_ABSENT = "attribute_absent"
    """The observation carries no value for this key. Never read as ``false``."""

    ATTRIBUTE_STALE = "attribute_stale"
    """Present, but older than the rule is willing to rely on."""

    EVIDENCE_UNVERIFIED = "evidence_unverified"
    """The rule requires corroborating evidence and none is present or fresh.
    The observation may well be correct; the rule declines to rely on it."""

    NOT_OBSERVABLE = "not_observable"
    """The platform **did** look and reported that it could not tell.

    Distinct from ``ATTRIBUTE_ABSENT``, and the distinction is the whole reason
    this value exists. An absent attribute means nothing was ever asked or the
    answer never arrived. This means a model examined the evidence and returned
    a value whose meaning is *"the thing you asked about is not visible here"* —
    a legitimate, registered domain value that the rule must treat as evidence
    of nothing rather than as evidence of absence.

    Without it, an enum carrying ``not_visible`` fails an equality test exactly
    as a bare hand would, and a person whose hands were inside a pot is reported
    as not wearing gloves. That is the failure this whole layer exists to
    prevent, arriving through the front door."""

    COVERAGE_GAP = "coverage_gap"
    """The platform was not fully observing the subject's scope. An empty or thin
    result under a coverage gap is not evidence of absence."""

    CAPABILITY_GAP = "capability_gap"
    """No bound model can produce the evidence here. Waiting will not help; this
    is a deployment fact, and the operator response is different from every other
    reason in this enum."""

    SUBJECT_NOT_OBSERVED = "subject_not_observed"
    """No object matched the rule's subject filter."""

    VALUE_UNPARSEABLE = "value_unparseable"
    """The held value could not be compared against the condition — a type
    mismatch between the rule and the registered attribute schema. A rule
    authoring error, surfaced rather than silently failing the condition."""


@dataclass(frozen=True, slots=True)
class SubjectRef:
    """Who the finding is about. Ids only; this layer adds no role or name."""

    object_id: ObjectId
    class_id: ClassId
    camera_id: CameraId
    tenant_id: TenantId = TenantId("")
    site_id: SiteId = SiteId("")
    label: str = ""
    """A display handle assigned by the caller — *"Employee #2"* in a frame
    story. Presentation only: no evaluation reads it, and it is never derived
    from anything the platform observed."""


@dataclass(frozen=True, slots=True)
class ConditionOutcome:
    """One condition's verdict, with what it compared and what it saw.

    ``observed`` is retained even when the condition passed, because a reviewer
    asking *"what did it actually see?"* should not have to re-query the platform
    to find out — and because a passing condition on a surprising value is how
    rule bugs are found.
    """

    attribute_key: str
    operator: str
    expected: object
    observed: object = None
    satisfied: bool | None = None
    """``True`` held · ``False`` failed · ``None`` could not be established."""

    unknown_reason: UnknownReason | None = None
    observed_at: Instant | None = None
    evidence_ref: str | None = None
    """Handle for the platform's evidence contract. Resolving it requires the
    separate evidence privilege, so the finding carries the reference and never
    the imagery."""

    message: str = ""
    """The rule document's own wording for this condition failing. Rendered from
    data, never generated by a model."""

    def __post_init__(self) -> None:
        if self.satisfied is None and self.unknown_reason is None:
            raise ValueError(
                f"condition on '{self.attribute_key}' is neither satisfied, "
                f"failed, nor attributed to a reason; an outcome nobody can "
                f"explain is one nobody can act on"
            )
        if self.satisfied is not None and self.unknown_reason is not None:
            raise ValueError(
                f"condition on '{self.attribute_key}' reports both a verdict and "
                f"an unknown reason; exactly one of them is true"
            )

    @property
    def failed(self) -> bool:
        return self.satisfied is False

    @property
    def unresolved(self) -> bool:
        return self.satisfied is None


@dataclass(frozen=True, slots=True)
class Finding:
    """One rule's verdict about one subject at one moment. Immutable.

    A re-evaluation produces a **new** finding; this one survives unchanged,
    because a consumer that acted on it acted correctly on the information then
    available and must be able to prove it.
    """

    finding_id: str
    rule_id: str
    rule_version: str
    ruleset_version: str
    state: ComplianceState
    subject: SubjectRef
    evaluated_at: Instant

    conditions: tuple[ConditionOutcome, ...] = ()
    severity: str = ""
    """An opaque label from the rule document. This layer orders by it and never
    interprets it — the same discipline the platform applies to priority."""

    coverage_fraction: float = 1.0
    """How much of the subject's scope the platform could observe when this was
    evaluated. Carried on every finding, not only uncertain ones: a compliant
    verdict reached under 40% coverage is a different claim from one reached
    under full coverage, and a reviewer cannot tell without this."""

    schema_version: str = FINDING_SCHEMA_VERSION
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rule_id or not self.rule_version:
            raise ValueError(
                "a finding must name the rule and version that produced it; a "
                "verdict traceable to no rule cannot be reviewed or appealed"
            )
        if self.state is ComplianceState.VIOLATION and not self.failed_conditions:
            raise ValueError(
                f"rule '{self.rule_id}' reported a violation with no failed "
                f"condition; a verdict that cannot name what failed is one "
                f"nobody can act on"
            )
        if self.state is ComplianceState.UNKNOWN and not self.unresolved_conditions:
            raise ValueError(
                f"rule '{self.rule_id}' reported unknown with no unresolved "
                f"condition; the reason is what tells an operator whether to fix "
                f"a camera, a budget, a model or the rule"
            )
        if not 0.0 <= self.coverage_fraction <= 1.0:
            raise ValueError("coverage_fraction must be in [0,1]")

    # --- derived, all pure --------------------------------------------------- #

    @property
    def failed_conditions(self) -> tuple[ConditionOutcome, ...]:
        return tuple(c for c in self.conditions if c.failed)

    @property
    def unresolved_conditions(self) -> tuple[ConditionOutcome, ...]:
        return tuple(c for c in self.conditions if c.unresolved)

    @property
    def satisfied_conditions(self) -> tuple[ConditionOutcome, ...]:
        return tuple(c for c in self.conditions if c.satisfied is True)

    @property
    def unknown_reasons(self) -> tuple[UnknownReason, ...]:
        seen: list[UnknownReason] = []
        for condition in self.unresolved_conditions:
            if condition.unknown_reason and condition.unknown_reason not in seen:
                seen.append(condition.unknown_reason)
        return tuple(seen)

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(c.evidence_ref for c in self.conditions if c.evidence_ref)
        )

    @property
    def pinned_rule(self) -> str:
        return f"{self.rule_id}@{self.rule_version}"

    def age(self, now: Instant) -> Duration:
        return Duration(max(0, now.ns - self.evaluated_at.ns))

    def describe(self) -> str:
        """The end-user sentence, assembled from structured data.

        Every word comes from the rule document or from an id. Nothing here is
        generated by a model, and nothing is inferred — this is *presentation of
        a structured violation*, which is why it can be regenerated identically
        from a stored finding six months later.
        """
        subject = self.subject.label or str(self.subject.object_id)
        if self.state is ComplianceState.VIOLATION:
            reasons = [c.message for c in self.failed_conditions if c.message]
            if not reasons:
                reasons = [
                    f"{c.attribute_key} {c.operator} {c.expected!r} "
                    f"(observed {c.observed!r})"
                    for c in self.failed_conditions
                ]
            return f"{subject}: " + "; ".join(reasons)
        if self.state is ComplianceState.UNKNOWN:
            named = ", ".join(r.value for r in self.unknown_reasons)
            return f"{subject}: cannot be assessed ({named})"
        if self.state is ComplianceState.NOT_APPLICABLE:
            return f"{subject}: rule '{self.rule_id}' does not apply"
        return f"{subject}: meets '{self.rule_id}'"


__all__ = [
    "FINDING_SCHEMA_VERSION",
    "ComplianceState",
    "ConditionOutcome",
    "Finding",
    "SubjectRef",
    "UnknownReason",
]
