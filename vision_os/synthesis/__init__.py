"""M11 Observation Builder — the single choke point for every published fact.

> **Single responsibility:** *Turn internal signals into published facts, and
> refuse anything that is not one.*

This package holds L5. The split mirrors every flow before it:

* ``builder.engine`` and ``runtime`` are the **module** — they assemble
  observations and implement the documented M11 public API.
* ``builder.validation`` and ``builder.suppression`` are the machinery, each with
  one responsibility and testable alone.

Nothing here imports an adapter. M11 holds ``SuppressionPolicyPort`` and
``ObservationSinkPort``; which implementation satisfies each is a composition
fact decided in ``synthesis_bootstrap``.
"""

from .builder.engine import (
    OBSERVATION_BUILDER_ID,
    BuildContext,
    ObservationBuilder,
    batch,
)
from .builder.suppression import (
    CameraSuppressionState,
    PublishedSignature,
    SuppressionStateStore,
    subject_key,
)
from .builder.validation import (
    CeilingGate,
    TaxonomyView,
    ceiling_violations,
)
from .runtime import (
    SYNTHESIS_RUNTIME_ID,
    SynthesisReport,
    SynthesisRuntime,
    SynthesisRuntimeStats,
)

__all__ = [
    "OBSERVATION_BUILDER_ID",
    "SYNTHESIS_RUNTIME_ID",
    "BuildContext",
    "CameraSuppressionState",
    "CeilingGate",
    "ObservationBuilder",
    "PublishedSignature",
    "SuppressionStateStore",
    "SynthesisReport",
    "SynthesisRuntime",
    "SynthesisRuntimeStats",
    "TaxonomyView",
    "batch",
    "ceiling_violations",
    "subject_key",
]
