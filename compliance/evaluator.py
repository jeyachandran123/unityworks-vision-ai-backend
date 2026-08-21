"""The rule evaluator — a pure function from facts to findings.

> **Single responsibility:** *Decide whether observed facts satisfy a stated
> requirement. Observe nothing, request nothing, infer nothing.*

**Determinism is the whole contract.** The same rule set, the same observation and
the same ``now`` produce the same finding, byte for byte. Nothing here reads a
clock, opens a socket, or calls a model — ``now`` is a parameter, and the only
collaborators are value types. An import test asserts the package's dependency
closure contains no model runtime and no HTTP client, so §13 is a build failure
rather than a review comment.

### Three-valued logic, and why a failure outranks an unknown

Conditions are evaluated with Kleene semantics over ``true``, ``false`` and
``unknown``, with **no short-circuit** — every condition is evaluated so that
every failure is recorded, not just the first.

The precedence is deliberate: any ``false`` makes the finding a ``VIOLATION``
even when other conditions are unknown. An observed *"headwear absent"* is a
violation whether or not gloves could be established, and suppressing it because
something else was unmeasurable would discard a fact the platform actually has.
The reverse ordering — unknown wins — would let one unobservable attribute mask
every real failure beside it.

### What absence means

An absent attribute is ``unknown``, never ``false``. The platform's own query
filter treats an absent value as a legitimate thing to match on, which is correct
for a *filter* and catastrophic for a *rule*: it would turn "we could not see"
into "the requirement was not met" and manufacture violations out of blind spots.
The one exception is the ``present`` operator, where absence is the answer being
asked for rather than a gap in the evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from vision_os.core.model.api import AttributeView, CoverageSummary, ObjectView
from vision_os.core.model.ids import AttributeKey
from vision_os.core.model.timebase import Instant

from .document import Condition, EvidenceRequirements, Operator, Rule, RuleSet
from .finding import (
    ComplianceState,
    ConditionOutcome,
    Finding,
    SubjectRef,
    UnknownReason,
)


class ComplianceEvaluator:
    """Evaluates rules against published observations. Holds no state.

    ``id_factory`` is injected rather than generated so a replay produces
    identical finding ids. A ``uuid4`` here would make every finding unequal to
    the one a re-run produced and would quietly break determinism testing — the
    property this class exists to guarantee.
    """

    __slots__ = ("_id_factory", "_rules")

    def __init__(
        self,
        rules: RuleSet,
        *,
        id_factory=None,
    ) -> None:
        self._rules = rules
        self._id_factory = id_factory or _deterministic_id

    @property
    def rules(self) -> RuleSet:
        return self._rules

    # --- public API ---------------------------------------------------------- #

    def evaluate_object(
        self,
        view: ObjectView,
        *,
        now: Instant,
        coverage: CoverageSummary | None = None,
        capability_gaps: Sequence[str] = (),
        subject_label: str = "",
    ) -> tuple[Finding, ...]:
        """Every rule covering this object's class, in document order.

        Order is document order rather than severity order: a caller that wants
        the worst first can sort, and a caller comparing two runs needs the order
        to be a property of the rules rather than of the data.
        """
        return tuple(
            self.evaluate_rule(
                rule,
                view,
                now=now,
                coverage=coverage,
                capability_gaps=capability_gaps,
                subject_label=subject_label,
            )
            for rule in self._rules.for_class(view.class_id)
            if rule.scope.covers_camera(view.camera_id)
        )

    def evaluate(
        self,
        views: Sequence[ObjectView],
        *,
        now: Instant,
        coverage: CoverageSummary | None = None,
        capability_gaps: Sequence[str] = (),
        labels: Mapping[str, str] | None = None,
    ) -> tuple[Finding, ...]:
        """Every rule against every subject, each evaluated independently.

        Independently is load-bearing: one subject's missing evidence must not
        change another's verdict, and two subjects in the same frame reach their
        own conclusions from their own attributes.
        """
        found: list[Finding] = []
        for view in views:
            found.extend(
                self.evaluate_object(
                    view,
                    now=now,
                    coverage=coverage,
                    capability_gaps=capability_gaps,
                    subject_label=(labels or {}).get(str(view.object_id), ""),
                )
            )
        return tuple(found)

    def evaluate_rule(
        self,
        rule: Rule,
        view: ObjectView,
        *,
        now: Instant,
        coverage: CoverageSummary | None = None,
        capability_gaps: Sequence[str] = (),
        subject_label: str = "",
    ) -> Finding:
        """One rule, one subject, one verdict."""
        subject = SubjectRef(
            object_id=view.object_id,
            class_id=view.class_id,
            camera_id=view.camera_id,
            label=subject_label,
        )
        observable = coverage.observable_fraction if coverage else 1.0
        gaps = frozenset(capability_gaps)

        # 1. The guard. A rule whose antecedent does not hold says nothing about
        #    this subject, and saying "compliant" would inflate a pass rate with
        #    subjects the rule never examined.
        guard = [
            self._assess(c, view, rule.evidence, now, gaps, coverage) for c in rule.when
        ]
        if any(outcome.failed for outcome in guard):
            return self._finding(
                rule, subject, ComplianceState.NOT_APPLICABLE, tuple(guard), now, observable
            )
        if any(outcome.unresolved for outcome in guard):
            return self._finding(
                rule, subject, ComplianceState.UNKNOWN, tuple(guard), now, observable
            )

        # 2. The requirement. Every condition is evaluated — no short-circuit —
        #    so a subject failing three requirements reports three failures.
        required = [
            self._assess(c, view, rule.evidence, now, gaps, coverage)
            for c in rule.require
        ]
        outcomes = (*guard, *required)

        if any(outcome.failed for outcome in required):
            state = ComplianceState.VIOLATION
        elif any(outcome.unresolved for outcome in required):
            state = ComplianceState.UNKNOWN
        else:
            state = ComplianceState.COMPLIANT

        return self._finding(rule, subject, state, outcomes, now, observable)

    # --- condition assessment ------------------------------------------------ #

    def _assess(
        self,
        condition: Condition,
        view: ObjectView,
        evidence: EvidenceRequirements,
        now: Instant,
        capability_gaps: frozenset[str],
        coverage: CoverageSummary | None,
    ) -> ConditionOutcome:
        """One condition against one object. Pure.

        The gates run in order of *how much they explain*: a capability gap tells
        an operator the rule can never be served here, which is more actionable
        than the staleness it would otherwise report, so it is checked first.
        """
        key = AttributeKey(condition.attribute)
        held = view.attributes.get(key)

        def unresolved(reason: UnknownReason) -> ConditionOutcome:
            return ConditionOutcome(
                attribute_key=condition.attribute,
                operator=condition.operator.value,
                expected=condition.expected,
                observed=held.value if held else None,
                unknown_reason=reason,
                observed_at=held.observed_at if held else None,
                evidence_ref=held.evidence_ref if held else None,
                message=condition.message,
            )

        if condition.attribute in capability_gaps:
            return unresolved(UnknownReason.CAPABILITY_GAP)

        if (
            evidence.require_full_coverage
            and coverage is not None
            and not coverage.fully_observable
        ):
            return unresolved(UnknownReason.COVERAGE_GAP)

        # `present` asks whether anything was observed, so absence answers it
        # rather than defeating it. Every other operator needs a value to compare.
        if condition.operator is Operator.PRESENT:
            return ConditionOutcome(
                attribute_key=condition.attribute,
                operator=condition.operator.value,
                expected=condition.expected,
                observed=held.value if held else None,
                satisfied=held is not None,
                observed_at=held.observed_at if held else None,
                evidence_ref=held.evidence_ref if held else None,
                message=condition.message,
            )

        if held is None:
            return unresolved(UnknownReason.ATTRIBUTE_ABSENT)

        # The platform looked and said it could not tell.
        #
        # Checked **before** staleness and before the comparison, because a
        # refusal is not a value: comparing it would fail the condition exactly
        # as a real negative does, and a person whose hands were inside a pot
        # would be reported as not wearing gloves. Absence of evidence must not
        # become evidence of absence, and this is the line where that happens.
        if condition.unknown_values and _as_text(held.value) in condition.unknown_values:
            return unresolved(UnknownReason.NOT_OBSERVABLE)

        if self._is_stale(held, evidence, now):
            return unresolved(UnknownReason.ATTRIBUTE_STALE)

        # §15's protected evaluation. The rule declared that its conclusion needs
        # corroboration; without it the evaluator declines rather than trusting a
        # claim the rule itself called insufficient.
        if self._uncorroborated(view, evidence, now):
            return unresolved(UnknownReason.EVIDENCE_UNVERIFIED)

        verdict = _compare(condition, held.value)
        if verdict is None:
            return unresolved(UnknownReason.VALUE_UNPARSEABLE)

        return ConditionOutcome(
            attribute_key=condition.attribute,
            operator=condition.operator.value,
            expected=condition.expected,
            observed=held.value,
            satisfied=verdict,
            observed_at=held.observed_at,
            evidence_ref=held.evidence_ref,
            message=condition.message,
        )

    @staticmethod
    def _is_stale(
        held: AttributeView, evidence: EvidenceRequirements, now: Instant
    ) -> bool:
        """Stale by the platform's horizon *or* by the rule's stricter one.

        Both, because they answer different questions: ``valid_until`` is the
        platform saying when it stops vouching for a value, and
        ``max_staleness`` is a rule saying how fresh it insists on being before
        it will act. The stricter wins.
        """
        if held.is_stale(now):
            return True
        if evidence.max_staleness is None:
            return False
        return now.ns - held.observed_at.ns > evidence.max_staleness.ns

    @staticmethod
    def _uncorroborated(
        view: ObjectView, evidence: EvidenceRequirements, now: Instant
    ) -> bool:
        """Whether required corroborating evidence is missing or stale.

        The corroborating attribute is whatever the verification policy produces
        — named by the document, never known to this code. Its presence and
        freshness is the whole test: a rule asks for corroboration, and the
        platform either published some or did not.
        """
        for key in evidence.requires:
            corroboration = view.attributes.get(AttributeKey(key))
            if corroboration is None or corroboration.is_stale(now):
                return True
        return False

    # --- assembly ------------------------------------------------------------ #

    def _finding(
        self,
        rule: Rule,
        subject: SubjectRef,
        state: ComplianceState,
        outcomes: tuple[ConditionOutcome, ...],
        now: Instant,
        observable: float,
    ) -> Finding:
        return Finding(
            finding_id=self._id_factory(rule, subject, now),
            rule_id=rule.rule_id,
            rule_version=rule.version,
            ruleset_version=self._rules.version,
            state=state,
            subject=subject,
            evaluated_at=now,
            conditions=outcomes,
            severity=rule.severity,
            coverage_fraction=observable,
            labels=dict(rule.labels),
        )


def _as_text(value: object) -> str:
    """A held value as the string a domain lists it as.

    Enum members arrive as plain strings, which is the case that matters. A
    boolean or a number stringifies harmlessly and simply will not match any
    declared refusal, which is the correct outcome — ``true`` is never a way of
    saying "I could not see".
    """
    return value if isinstance(value, str) else str(value)


def _deterministic_id(rule: Rule, subject: SubjectRef, now: Instant) -> str:
    """A finding id that a replay reproduces exactly.

    Derived from what the finding is *about* rather than from randomness or wall
    time, so re-evaluating the same rule against the same subject at the same
    instant yields the same id — which is what lets a caller deduplicate findings
    without a second store, and what makes the determinism test meaningful.
    """
    return f"{rule.rule_id}@{rule.version}:{subject.object_id}:{now.ns}"


def _compare(condition: Condition, held: object) -> bool | None:
    """Apply one operator. ``None`` means the comparison is not meaningful.

    ``None`` rather than ``False`` for a type mismatch: a rule comparing a string
    attribute with a number is an authoring error, and reporting it as a failed
    condition would present a broken rule as a real violation.
    """
    operator = condition.operator
    expected = condition.expected

    if operator is Operator.EQ:
        return held == expected
    if operator is Operator.NE:
        return held != expected
    if operator is Operator.IN:
        return held in tuple(expected)  # type: ignore[arg-type]
    if operator is Operator.NOT_IN:
        return held not in tuple(expected)  # type: ignore[arg-type]

    if operator.is_ordering:
        # `bool` is an `int` in Python, and ordering a boolean against a number
        # is never what a rule meant.
        if isinstance(held, bool) or isinstance(expected, bool):
            return None
        if not isinstance(held, int | float) or not isinstance(expected, int | float):
            return None
        if operator is Operator.GT:
            return held > expected
        if operator is Operator.GTE:
            return held >= expected
        if operator is Operator.LT:
            return held < expected
        return held <= expected

    return None  # pragma: no cover - PRESENT is handled before comparison


__all__ = ["ComplianceEvaluator"]
