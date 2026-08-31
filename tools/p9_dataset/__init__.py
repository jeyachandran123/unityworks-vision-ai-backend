"""P9 — the evaluation-grade PPE dataset: schema, validation, splits, manifests.

Offline tooling. Nothing here is imported by `app/`, `compliance/` or
`vision_os/`, and nothing here changes production inference. It builds and
guards the asset that future model decisions will be judged against.

The central rule this package exists to enforce, in code rather than in prose:

    NOT_VISIBLE is not ABSENT.

An annotation that says a region could not be seen may not also carry a decided
attribute state, and the validator rejects the combination rather than trusting
an annotator to remember.
"""

from .schema import (
    ANNOTATION_SCHEMA_VERSION,
    AttributeState,
    Observability,
    Region,
    RegionAnnotation,
    SubjectAnnotation,
)
from .validate import ValidationError, validate_manifest, validate_subject

__all__ = [
    "ANNOTATION_SCHEMA_VERSION",
    "AttributeState",
    "Observability",
    "Region",
    "RegionAnnotation",
    "SubjectAnnotation",
    "ValidationError",
    "validate_manifest",
    "validate_subject",
]
