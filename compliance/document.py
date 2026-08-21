"""Compliance rules, as data.

### Why the rule engine lives outside Vision OS

The platform refuses to hold this logic, in three places and on purpose. Its
attribute registry rejects any key containing ``compliant``, ``violation`` or
``alert``; a CI boundary test fails the build if those words appear as
identifiers inside its core; and its query language deliberately omits comparison
operators, with the note that *"V1 puts that in the consumer's rule engine, not
in a query language."*

This package **is** that consumer's rule engine. It is the one place in the
system where a threshold is allowed to mean something.

### Why there is no domain vocabulary in this file either

Not one attribute name, value, class or sentence appears here. ``hairnet``,
``cutting_board_category``, ``non_vegetarian`` and *"is not wearing a hairnet"*
are **data in a JSON document**, exactly as the platform's semantic policy keeps
them. Adding a use case is a file, not a release — and the same test that proves
it for the platform proves it here.

### What a rule may and may not do

A rule reads structured facts and compares them. It may not call a model, request
a crop, or reach into the platform for anything but a published observation. That
is not a guideline: the evaluator has no such collaborator to call, and an import
test asserts the package's whole dependency closure contains no model runtime and
no HTTP client.

### On confidence thresholds, which this deliberately does not support

A rule cannot filter on attribute confidence. Most attribute confidence in this
platform is ``SELF_REPORTED`` — a model's opinion about itself, which 02_VOM
§7.2 says *"is not a probability"* and which ``Confidence.comparable_with``
refuses to order at all. Offering ``min_confidence`` would invite exactly the
comparison the platform's type system exists to prevent, and it would fail
silently rather than loudly. A rule that wants stronger evidence declares
``requires`` instead, which is a statement about corroboration rather than about
an incomparable float.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vision_os.core.model.ids import CameraId, ClassId, SiteId, TenantId
from vision_os.core.model.timebase import Duration

#: Where a deployment names its active rule set. A path, or empty for none.
RULES_ENV = "COMPLIANCE_RULES"


class RuleDocumentError(ValueError):
    """A rule set was named that cannot be loaded. Raised at load time.

    At load rather than at evaluation: a malformed rule discovered on the first
    violation is a rule that has been silently not-evaluating since deployment.
    """


class Operator(enum.Enum):
    """How a held value is compared with an expected one. Closed set.

    Comparison operators appear here and nowhere in the platform. That asymmetry
    is the whole point: ``dwell_seconds > 300`` is a business threshold, and this
    is the layer that owns business thresholds.
    """

    EQ = "eq"
    NE = "ne"
    IN = "in"
    NOT_IN = "not_in"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    PRESENT = "present"
    """Holds when any value is held, whatever it is. The one operator for which
    an absent attribute is a *failure* rather than an unknown — asking whether
    something was observed is answerable by its absence."""

    @property
    def needs_expected(self) -> bool:
        return self is not Operator.PRESENT

    @property
    def is_ordering(self) -> bool:
        return self in (Operator.GT, Operator.GTE, Operator.LT, Operator.LTE)


@dataclass(frozen=True, slots=True)
class Condition:
    """One comparison against one registered attribute."""

    attribute: str
    operator: Operator
    expected: object = None
    message: str = ""
    """The document's own wording when this condition fails, rendered into the
    end-user sentence. Data, so the sentence carries no vocabulary from code and
    can be localised or rewritten without a release."""

    unknown_values: tuple[str, ...] = ()
    """Values that mean *"the platform could not tell"* rather than a fact.

    A vision attribute's domain often contains one — ``not_visible`` is the
    obvious case: the model looked, the body part was inside a pot or out of
    frame, and it said so. That is exactly the honest answer the observation
    layer is supposed to give, and it must not then be compared like a fact.

    Without this, ``hand_covering == "gloves"`` treats ``not_visible`` as a
    failed condition, and a person whose hands were never visible is reported as
    not wearing gloves. The reading is arithmetically correct and completely
    wrong: absence of evidence became evidence of absence.

    Listed per condition rather than inferred from the value, because only the
    rule author knows which of an enum's members are facts and which are
    refusals. ``unknown`` is a real answer to *"what kind of object is this?"*
    and a refusal in ``head_covering``; nothing in the value itself says which."""

    def __post_init__(self) -> None:
        if not self.attribute:
            raise RuleDocumentError("a condition must name an attribute")
        if self.operator.needs_expected and self.expected is None:
            raise RuleDocumentError(
                f"condition on '{self.attribute}' uses '{self.operator.value}' "
                f"with no expected value"
            )
        if self.operator in (Operator.IN, Operator.NOT_IN) and not isinstance(
            self.expected, list | tuple
        ):
            raise RuleDocumentError(
                f"condition on '{self.attribute}' uses '{self.operator.value}' "
                f"and needs a list of values"
            )


@dataclass(frozen=True, slots=True)
class EvidenceRequirements:
    """What the evidence behind this rule must satisfy before it is relied on.

    §15's *protected rule evaluation*, expressed as data. A rule declaring
    nothing here evaluates whatever the platform published; a rule declaring
    ``requires`` will not reach a verdict until corroboration exists, and says
    ``UNKNOWN`` in the meantime rather than trusting a claim it has declared
    insufficient.
    """

    max_staleness: Duration | None = None
    """How old a value may be before this rule declines to rely on it. Distinct
    from the attribute's own ``valid_until``: the platform states when a value
    expires, and a rule may additionally insist on something fresher."""

    requires: tuple[str, ...] = ()
    """Corroborating attributes that must be present and fresh for this rule's
    conclusion to be admissible — typically whatever the verification policy
    produces. Absent or stale corroboration yields ``EVIDENCE_UNVERIFIED``."""

    require_full_coverage: bool = False
    """Refuse to conclude while the platform reports it was not fully observing
    the subject's scope. Off by default: most rules concern what *was* seen, and
    only some are invalidated by what might have been missed."""


@dataclass(frozen=True, slots=True)
class RuleScope:
    """Where a rule applies. Ids only, exactly like the platform's demand scope."""

    tenant_id: TenantId = TenantId("")
    site_ids: tuple[SiteId, ...] = ()
    camera_ids: tuple[CameraId, ...] = ()

    def covers_camera(self, camera_id: CameraId) -> bool:
        """An empty camera list means every camera in scope, not none."""
        return not self.camera_ids or camera_id in self.camera_ids


@dataclass(frozen=True, slots=True)
class Rule:
    """One business requirement over observed facts.

    ``when`` and ``require`` together express both shapes the brief needs. A flat
    requirement declares only ``require``. A conditional one — *"if the food is X
    then the board must be X"* — puts the antecedent in ``when``, and a subject
    whose antecedent does not hold is ``NOT_APPLICABLE`` rather than compliant.
    """

    rule_id: str
    version: str
    require: tuple[Condition, ...]
    when: tuple[Condition, ...] = ()
    subject_classes: tuple[ClassId, ...] = ()
    scope: RuleScope = RuleScope()
    evidence: EvidenceRequirements = EvidenceRequirements()
    severity: str = ""
    """Opaque. Ordered by, never interpreted."""

    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise RuleDocumentError("a rule must carry a rule_id for traceability")
        if not self.version:
            raise RuleDocumentError(
                f"rule '{self.rule_id}' carries no version; an unversioned rule "
                f"makes every finding it produced unexplainable, because nobody "
                f"can tell which wording of it was applied"
            )
        if not self.require:
            raise RuleDocumentError(
                f"rule '{self.rule_id}' requires nothing; a rule that cannot fail "
                f"would report every subject compliant and inflate a pass rate"
            )

    @property
    def pinned(self) -> str:
        return f"{self.rule_id}@{self.version}"

    @property
    def attributes(self) -> tuple[str, ...]:
        """Every attribute this rule depends on, guards and requirements alike.

        Used to answer §20's question — *is what this rule needs actually being
        produced?* — before the rule is ever evaluated.
        """
        keys: list[str] = []
        for condition in (*self.when, *self.require):
            if condition.attribute not in keys:
                keys.append(condition.attribute)
        for key in self.evidence.requires:
            if key not in keys:
                keys.append(key)
        return tuple(keys)

    def covers_class(self, class_id: ClassId) -> bool:
        """Hierarchical, matching the platform's own class filters."""
        if not self.subject_classes:
            return True
        return any(
            class_id == allowed or class_id.startswith(f"{allowed}.")
            for allowed in self.subject_classes
        )


@dataclass(frozen=True, slots=True)
class RuleSet:
    """A deployment's rules, versioned as a whole.

    The set carries its own version alongside each rule's, because a finding must
    be reproducible and a rule can be evaluated identically under two different
    sets — the difference being which *other* rules ran beside it.
    """

    version: str = "1.0.0"
    rules: tuple[Rule, ...] = ()

    def for_class(self, class_id: ClassId) -> tuple[Rule, ...]:
        return tuple(rule for rule in self.rules if rule.covers_class(class_id))

    def get(self, rule_id: str) -> Rule | None:
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    @property
    def required_attributes(self) -> tuple[str, ...]:
        keys: list[str] = []
        for rule in self.rules:
            for key in rule.attributes:
                if key not in keys:
                    keys.append(key)
        return tuple(keys)

    def unproducible_against(self, producible: Sequence[str]) -> tuple[tuple[str, str], ...]:
        """Which rules depend on attributes the platform cannot produce.

        §20 made answerable before anything is evaluated. A deployment learns at
        startup that a rule can never reach a verdict here, instead of watching
        it return ``UNKNOWN`` forever and wondering which of six causes it is.
        """
        available = set(producible)
        gaps: list[tuple[str, str]] = []
        for rule in self.rules:
            for key in rule.attributes:
                if key not in available:
                    gaps.append((rule.rule_id, key))
        return tuple(gaps)

    # --- construction ------------------------------------------------------- #

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> RuleSet:
        entries = document.get("rules", ())
        if not isinstance(entries, list | tuple):
            raise RuleDocumentError("'rules' must be a list")
        rules = tuple(_rule_from(entry) for entry in entries)
        seen: set[str] = set()
        for rule in rules:
            if rule.rule_id in seen:
                raise RuleDocumentError(
                    f"rule '{rule.rule_id}' is declared twice; two rules sharing an "
                    f"id make a finding ambiguous about which one produced it"
                )
            seen.add(rule.rule_id)
        return cls(version=str(document.get("version", "1.0.0")), rules=rules)

    @classmethod
    def from_file(cls, path: Path | str) -> RuleSet:
        file = Path(path)
        if not file.is_file():
            raise RuleDocumentError(f"compliance rules not found at '{file}'")
        try:
            document = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuleDocumentError(f"'{file}' is not valid JSON: {exc}") from exc
        return cls.from_document(document)


def load_rules(
    path: Path | str | None = None, *, env: Mapping[str, str] | None = None
) -> RuleSet | None:
    """The active rule set, or ``None``.

    ``None`` is a supported configuration and never an error: it means this
    deployment evaluates no rules. Vision OS runs exactly as before, producing
    observations nobody has yet asked a question about.
    """
    import os

    source = os.environ if env is None else env
    chosen = str(path or source.get(RULES_ENV, "")).strip()
    if not chosen:
        return None
    return RuleSet.from_file(chosen)


# --- document parsing --------------------------------------------------------- #


def _rule_from(entry: Mapping[str, Any]) -> Rule:
    if not isinstance(entry, Mapping):
        raise RuleDocumentError("each rule must be an object")

    scope = entry.get("scope", {}) or {}
    evidence = entry.get("evidence", {}) or {}
    staleness = evidence.get("max_staleness_ms")

    return Rule(
        rule_id=str(entry.get("rule_id", "")).strip(),
        version=str(entry.get("version", "")).strip(),
        require=tuple(_condition_from(c) for c in entry.get("require", ())),
        when=tuple(_condition_from(c) for c in entry.get("when", ())),
        subject_classes=tuple(ClassId(str(c)) for c in entry.get("subject_classes", ())),
        scope=RuleScope(
            tenant_id=TenantId(str(scope.get("tenant_id", ""))),
            site_ids=tuple(SiteId(str(s)) for s in scope.get("site_ids", ())),
            camera_ids=tuple(CameraId(str(c)) for c in scope.get("camera_ids", ())),
        ),
        evidence=EvidenceRequirements(
            max_staleness=(
                None if staleness is None else Duration.from_millis(int(staleness))
            ),
            requires=tuple(str(k) for k in evidence.get("requires", ())),
            require_full_coverage=bool(evidence.get("require_full_coverage", False)),
        ),
        severity=str(entry.get("severity", "")),
        labels={str(k): str(v) for k, v in (entry.get("labels", {}) or {}).items()},
    )


def _condition_from(entry: Mapping[str, Any]) -> Condition:
    if not isinstance(entry, Mapping):
        raise RuleDocumentError("each condition must be an object")
    raw = str(entry.get("operator", "eq")).lower()
    try:
        operator = Operator(raw)
    except ValueError as exc:
        raise RuleDocumentError(
            f"'{raw}' is not a supported operator; supported: "
            f"{', '.join(o.value for o in Operator)}"
        ) from exc
    expected = entry.get("value")
    return Condition(
        attribute=str(entry.get("attribute", "")).strip(),
        operator=operator,
        expected=tuple(expected) if isinstance(expected, list) else expected,
        message=str(entry.get("message", "")),
        unknown_values=tuple(str(v) for v in entry.get("unknown_values", ())),
    )


__all__ = [
    "RULES_ENV",
    "Condition",
    "EvidenceRequirements",
    "Operator",
    "Rule",
    "RuleDocumentError",
    "RuleScope",
    "RuleSet",
    "load_rules",
]
