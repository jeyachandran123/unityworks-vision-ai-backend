"""Executable port conformance kits (06_PORTS_AND_ADAPTERS section 5).

The mechanism that converts invariant V3 — "every model is replaceable" — from a
claim in a document into a gate in the loader.
"""

from __future__ import annotations

from .cropping_kits import (
    ALL_CROPPING_KITS,
    CROP_STRATEGY_KIT,
    QUALITY_ESTIMATOR_KIT,
    TRIGGER_POLICY_KIT,
)
from .detector_kit import DETECTOR_KIT, detector_kit_checks
from .exposure_kits import (
    ALL_EXPOSURE_KITS,
    API_TRANSPORT_KIT,
    AUTHORIZATION_KIT,
    EVIDENCE_STORE_KIT,
)
from .flow1_kits import ALL_FLOW1_KITS, flow1_registry, platform_registry
from .kit import (
    ConformanceCheck,
    ConformanceKit,
    ConformanceRegistry,
    ConformanceReport,
    KitSection,
)
from .model_kits import (
    ALL_MODEL_KITS,
    ARTIFACT_STORE_KIT,
    DEVICE_KIT,
    MODEL_RUNTIME_KIT,
)
from .registry_kits import IDENTITY_RESOLVER_KIT, OBJECT_STORE_KIT
from .synthesis_kits import (
    ALL_SYNTHESIS_KITS,
    OBSERVATION_LOG_KIT,
    OBSERVATION_SINK_KIT,
    SUPPRESSION_POLICY_KIT,
)
from .tracker_kit import DETERMINISM_CHECK, TRACKER_KIT
from .understanding_kits import (
    ALL_UNDERSTANDING_KITS,
    OUTPUT_COERCION_KIT,
    UNDERSTANDER_KIT,
)

__all__ = [
    "ALL_CROPPING_KITS",
    "ALL_EXPOSURE_KITS",
    "ALL_UNDERSTANDING_KITS",
    "ALL_FLOW1_KITS",
    "ALL_MODEL_KITS",
    "ALL_SYNTHESIS_KITS",
    "API_TRANSPORT_KIT",
    "ARTIFACT_STORE_KIT",
    "AUTHORIZATION_KIT",
    "CROP_STRATEGY_KIT",
    "DETECTOR_KIT",
    "DETERMINISM_CHECK",
    "DEVICE_KIT",
    "EVIDENCE_STORE_KIT",
    "IDENTITY_RESOLVER_KIT",
    "MODEL_RUNTIME_KIT",
    "OBJECT_STORE_KIT",
    "OBSERVATION_LOG_KIT",
    "OBSERVATION_SINK_KIT",
    "OUTPUT_COERCION_KIT",
    "QUALITY_ESTIMATOR_KIT",
    "SUPPRESSION_POLICY_KIT",
    "TRACKER_KIT",
    "TRIGGER_POLICY_KIT",
    "UNDERSTANDER_KIT",
    "ConformanceCheck",
    "ConformanceKit",
    "ConformanceRegistry",
    "ConformanceReport",
    "KitSection",
    "detector_kit_checks",
    "flow1_registry",
    "platform_registry",
]
