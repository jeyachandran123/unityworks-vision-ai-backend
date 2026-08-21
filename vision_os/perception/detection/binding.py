"""What the Detection Engine holds instead of a model.

A ``DetectorBinding`` is everything needed to run one detector and to explain the
results it produces: the adapter, its declared capability, the model handle the
Model Manager granted, its taxonomy mapping and its calibration profile.

Bundling them makes the replaceability promise concrete. Swapping YOLO for
RT-DETR replaces one binding; the engine, the scheduler, the worker and every
consumer are untouched, because none of them holds anything model-specific.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.model.ids import AdapterId
from ...core.model.taxonomy import CoverageReport, TaxonomyMapping
from ...core.ports.detection import DetectorCapabilities, DetectorPort
from ...kernel.models.calibration import CalibrationProfile
from ...kernel.models.manager import ModelHandle


@dataclass(frozen=True, slots=True)
class DetectorBinding:
    """One activated detector, with everything needed to explain its output."""

    adapter_id: AdapterId
    adapter_version: str
    detector: DetectorPort
    capabilities: DetectorCapabilities
    model_handle: ModelHandle
    mapping: TaxonomyMapping
    coverage: CoverageReport
    role: str = "primary_detector"
    calibration: CalibrationProfile | None = None

    def __post_init__(self) -> None:
        if not self.coverage.valid:
            raise ValueError(
                f"detector '{self.adapter_id}' has an invalid taxonomy mapping: "
                f"{sorted(self.coverage.unknown_classes)}"
            )

    @property
    def model_id(self) -> str:
        return str(self.model_handle.model_id)

    @property
    def is_calibrated(self) -> bool:
        """Whether this detector's confidences are comparable with others.

        Stated rather than assumed: an uncalibrated detector still works, but its
        scores must not be ranked against another model's (02_VOM section 7.2).
        """
        return self.calibration is not None
