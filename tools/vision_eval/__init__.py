"""Ground-truth annotation and evaluation for Vision OS.

A development toolchain, deliberately outside ``app/``. It imports Vision OS
types to read its output and is imported by no runtime code — the platform does
not know this exists, which is what keeps evaluation from becoming a dependency
of perception.

Nothing here trains a model, and nothing here writes a label. Ground truth comes
from a human looking at a frame; this package only stores it, splits it without
leakage, and scores predictions against it.
"""

from .dataset import Dataset, DatasetSplit, LeakageError, group_split, save_dataset
from .metrics import EvaluationReport, Failure, evaluate
from .schema import (
    AnnotatedFrame,
    AnnotatedSubject,
    AttributeState,
    BoundingBox,
    FailureCategory,
    PredictedFrame,
    PredictedSubject,
    load_annotations,
    save_annotations,
)

__all__ = [
    "AnnotatedFrame",
    "AnnotatedSubject",
    "AttributeState",
    "BoundingBox",
    "Dataset",
    "DatasetSplit",
    "EvaluationReport",
    "Failure",
    "FailureCategory",
    "LeakageError",
    "PredictedFrame",
    "PredictedSubject",
    "evaluate",
    "group_split",
    "load_annotations",
    "save_annotations",
]
