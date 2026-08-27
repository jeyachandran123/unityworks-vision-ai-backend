"""Perception adapters that are neither detector nor understander.

Kept out of ``adapters/cropping`` deliberately. That package is inside M8's
filesystem guard — *"a vocabulary guard can be worked around by naming a method
something else; an import guard cannot"* — and a model artefact has to be loaded
from somewhere. The crop path holds the **port**; loading the weights is a
composition-time act and lives here, exactly as the detector's does.
"""

from .pose import PoseRegionObservability, PoseThresholds

__all__ = ["PoseRegionObservability", "PoseThresholds"]
