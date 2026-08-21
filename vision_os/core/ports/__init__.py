"""Port protocols — Flow 1 + Flow 2 subset (06_PORTS_AND_ADAPTERS).

A port is three things, not one::

    Port = Interface  +  Semantic Contract  +  Conformance Kit
           (shape)       (meaning)             (executable proof)

The interface lives here. The semantic contract lives in each protocol's
docstring. The conformance kit lives in ``vision_os.conformance`` and is run by
the Plugin Manager **before an adapter is activated** — which is what converts
"every model is replaceable" from a claim into a gate.

Implemented so far — 15 of the catalogue's 32 ports:

    Flow 1   P1 Source · P2 Decoder · P3 PrivacyMask · P4 ClockSync
             P5 AdmissionPolicy · P6 ChangeDetector · P7 Allocator
             P23 ConfigSource · P24 SecretProvider
             P29 EventTransport · P30 MetricsExport · Clock

    Flow 2   P8 Detector · P25 ArtifactStore · P26 ModelRuntime · P27 Device

Plus ``AdmittedFrameConsumer``, the pipeline continuation seam by which a later
flow resumes the admitted-frame path without Flow 1 knowing it exists.

The remaining 17 ports belong to later flows and are deliberately absent.
"""

from __future__ import annotations

from .acquisition import (
    CaptureEstimate,
    ClockSyncPort,
    DecodeOutcome,
    DecoderCapabilities,
    DecoderPort,
    MaskOutcome,
    PrivacyMaskPort,
    SourceCapabilities,
    SourceHandle,
    SourcePacket,
    SourcePort,
)
from .buffer import Allocation, AllocatorPort, AllocatorStats, WritableSlot
from .clock import Clock
from .configuration import ConfigSourcePort, SecretProviderPort
from .detection import (
    BatchProfile,
    DetectionRequest,
    DetectionResult,
    DetectorCapabilities,
    DetectorPort,
    FrameView,
    InputConstraints,
    NmsDeclaration,
    RawDetection,
)
from .models import (
    ArtifactRef,
    ArtifactStorePort,
    DeviceInfo,
    DeviceKind,
    DevicePort,
    LoadedModel,
    ModelRuntimePort,
)
from .observability import EventTransportPort, MetricsExportPort, MetricsSnapshotView
from .pipeline import AdmittedFrameConsumer
from .scheduling import (
    AdmissionContext,
    AdmissionPolicyPort,
    AdmissionVerdict,
    ChangeDetectorPort,
    ChangeVerdict,
    DropReason,
    Fidelity,
)

__all__ = [
    "AdmissionContext",
    "AdmissionPolicyPort",
    "AdmissionVerdict",
    "AdmittedFrameConsumer",
    "Allocation",
    "AllocatorPort",
    "AllocatorStats",
    "ArtifactRef",
    "ArtifactStorePort",
    "BatchProfile",
    "CaptureEstimate",
    "ChangeDetectorPort",
    "ChangeVerdict",
    "Clock",
    "ClockSyncPort",
    "ConfigSourcePort",
    "DecodeOutcome",
    "DecoderCapabilities",
    "DecoderPort",
    "DetectionRequest",
    "DetectionResult",
    "DetectorCapabilities",
    "DetectorPort",
    "DeviceInfo",
    "DeviceKind",
    "DevicePort",
    "DropReason",
    "EventTransportPort",
    "Fidelity",
    "FrameView",
    "InputConstraints",
    "LoadedModel",
    "MaskOutcome",
    "MetricsExportPort",
    "MetricsSnapshotView",
    "ModelRuntimePort",
    "NmsDeclaration",
    "PrivacyMaskPort",
    "RawDetection",
    "SecretProviderPort",
    "SourceCapabilities",
    "SourceHandle",
    "SourcePacket",
    "SourcePort",
    "WritableSlot",
]
