"""Synthesis and state adapters — suppression policies, logs and sinks.

**Nothing outside this package and the composition root may name a suppression
policy, a log or a sink.** The platform holds P18, P19 and P20; which
implementation satisfies each is a configuration fact, exactly as Flow 2 keeps
YOLO invisible and Flow 6 keeps the VLM invisible.

The log adapters are **adapters, not M13**: §M13's single responsibility is
*"Describe what must persist and with what guarantees; implement none of it."*
Shipping an implementation behind one of its five contracts is the same act
Flow 2 performed for P25-P27 and Flow 4 for P21.
"""

from .decode import decode_observation
from .stores import (
    CollectingSink,
    FileObservationLog,
    InMemoryObservationLog,
    NullSink,
)
from .suppression import (
    ALWAYS_PUBLISH,
    AlwaysPublish,
    ExactSuppression,
    ThresholdSuppression,
)

#: Suppression policies selectable by configuration.
#:
#: A closed table, like the tracker and crop-strategy factories. A deployment
#: names a policy; it does not import one.
SUPPRESSION_FACTORIES = {
    "suppression.exact": ExactSuppression,
    "suppression.threshold": ThresholdSuppression,
    "suppression.always": AlwaysPublish,
}

__all__ = [
    "ALWAYS_PUBLISH",
    "SUPPRESSION_FACTORIES",
    "AlwaysPublish",
    "CollectingSink",
    "ExactSuppression",
    "FileObservationLog",
    "InMemoryObservationLog",
    "NullSink",
    "ThresholdSuppression",
    "decode_observation",
]
