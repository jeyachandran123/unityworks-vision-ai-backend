"""Cropping adapters — the default policy, estimator and strategies.

**Nothing outside this package and the composition root may name a trigger
policy, a quality estimator, or a crop strategy.** The platform holds P12, P13
and P14; which implementation satisfies each is a configuration fact, exactly as
Flow 2 keeps YOLO invisible to the platform.

**No adapter here is named after a model.** §M8 STANDARDIZATION: *"No YOLO crop.
No CLIP crop. No Florence crop. No Qwen crop. No InternVL crop."* Mean
subtraction, channel-order flips and tensor layout are a model adapter's business
in M9, downstream of the canonical crop.
"""

from .quality import (
    DEFAULT_MIN_SCALE_PIXELS,
    AlwaysUsableEstimator,
    HeuristicQualityEstimator,
)
from .strategies import (
    DEFAULT_OUTPUT_SIZE,
    DEFAULT_PADDING,
    PaddedCropStrategy,
    PartFocusedCropStrategy,
    ReferenceCropExtractor,
    TightCropStrategy,
)
from .triggers import (
    DEFAULT_APPEARANCE_THRESHOLD,
    DEFAULT_LOW_CONFIDENCE,
    DEFAULT_REFRESH_INTERVAL,
    DefaultTriggerPolicy,
    ExplicitRequestPolicy,
)
from .verification import (
    CropRequirements,
    TrustConditions,
    VerificationPolicy,
    VerificationRule,
    VerificationRules,
)

#: Trigger policies selectable by configuration.
#:
#: A closed table, like the tracker factories. A deployment names a policy; it
#: does not import one.
TRIGGER_POLICY_FACTORIES = {
    "trigger.default": DefaultTriggerPolicy,
}

QUALITY_ESTIMATOR_FACTORIES = {
    "quality.heuristic": HeuristicQualityEstimator,
    "quality.always_usable": AlwaysUsableEstimator,
}

CROP_STRATEGY_FACTORIES = {
    "crop.tight": TightCropStrategy,
    "crop.padded": PaddedCropStrategy,
    # Part-focused. Spends the canonical crop on the region a question is
    # about instead of letterboxing a whole standing person into a square.
    "crop.part_focused": PartFocusedCropStrategy,
}

__all__ = [
    "CROP_STRATEGY_FACTORIES",
    "DEFAULT_APPEARANCE_THRESHOLD",
    "DEFAULT_LOW_CONFIDENCE",
    "DEFAULT_MIN_SCALE_PIXELS",
    "DEFAULT_OUTPUT_SIZE",
    "DEFAULT_PADDING",
    "DEFAULT_REFRESH_INTERVAL",
    "QUALITY_ESTIMATOR_FACTORIES",
    "TRIGGER_POLICY_FACTORIES",
    "AlwaysUsableEstimator",
    "CropRequirements",
    "DefaultTriggerPolicy",
    "ExplicitRequestPolicy",
    "HeuristicQualityEstimator",
    "PaddedCropStrategy",
    "PartFocusedCropStrategy",
    "ReferenceCropExtractor",
    "TightCropStrategy",
    "TrustConditions",
    "VerificationPolicy",
    "VerificationRule",
    "VerificationRules",
]
