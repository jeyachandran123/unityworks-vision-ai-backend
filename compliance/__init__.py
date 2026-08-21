"""The compliance rule engine — Vision OS's consumer, not one of its modules.

This package sits **outside** the platform boundary and depends on it one way:
``compliance`` imports ``vision_os``; ``vision_os`` never imports ``compliance``.
That direction is asserted by a test, because the moment it reverses the platform
has acquired a business opinion.

### Why it cannot live inside

Vision OS refuses this logic in three places, each of them deliberate:

1. Its attribute registry rejects any key containing ``compliant``, ``violation``
   or ``alert`` — a business verdict cannot become an observation.
2. A CI boundary test fails the build if those words appear as identifiers
   anywhere in ``core/ kernel/ acquisition/ perception/ taxonomy/``.
3. Its query language deliberately omits comparison operators, noting that
   *"V1 puts that in the consumer's rule engine, not in a query language."*

The platform named this package before it existed. Building the rule engine
inside would have required weakening all three gates.

### The layering, in one line each

``document``
    Rules as data. No attribute name, value or sentence appears in code.

``evaluator``
    A pure function from facts to findings. Three-valued, deterministic, and
    unable to call a model because it holds no collaborator that could.

``finding``
    The verdict envelope, carrying its rule version, every condition outcome and
    the evidence handles that defend it.

``reader``
    The single scope-narrowed read path into the Observation API. Reads only.
"""

from .document import (
    RULES_ENV,
    Condition,
    EvidenceRequirements,
    Operator,
    Rule,
    RuleDocumentError,
    RuleScope,
    RuleSet,
    load_rules,
)
from .evaluator import ComplianceEvaluator
from .finding import (
    ComplianceState,
    ConditionOutcome,
    Finding,
    SubjectRef,
    UnknownReason,
)
from .reader import ObservationReader, ObservationSnapshot, subjects_by_camera

__all__ = [
    "RULES_ENV",
    "ComplianceEvaluator",
    "ComplianceState",
    "Condition",
    "ConditionOutcome",
    "EvidenceRequirements",
    "Finding",
    "ObservationReader",
    "ObservationSnapshot",
    "Operator",
    "Rule",
    "RuleDocumentError",
    "RuleScope",
    "RuleSet",
    "SubjectRef",
    "UnknownReason",
    "load_rules",
    "subjects_by_camera",
]
