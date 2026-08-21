"""M5 Detection Engine and its execution substrate (Flow 2).

    DetectionEngine      find things in a frame
    DetectionNormalizer  turn adapter output into platform detections
    DetectionScheduler   gather frames from many cameras into batches
    DeviceWorker         execute one batch on one device
    DetectionManager     detector lifecycle: load, validate, activate, retire
    DetectionRuntime     own the layer's lifecycle; resume the admitted-frame path

Detection ends when standardized detections are emitted. There is no tracking, no
identity, no cropping, no understanding, and no state here — by construction.
"""

from __future__ import annotations

from .binding import DetectorBinding
from .engine import DetectionEngine, DetectionOutcome
from .manager import DetectionManager, DetectorFactory, DetectorRegistration
from .normalizer import DetectionNormalizer, NormalizationOutcome, NormalizationPolicy
from .runtime import DetectionRuntime, DetectionRuntimeStats
from .scheduler import BatchItem, BatchKey, BatchStats, DetectionScheduler
from .worker import DeviceWorker, WorkerStats

__all__ = [
    "BatchItem",
    "BatchKey",
    "BatchStats",
    "DetectionEngine",
    "DetectionManager",
    "DetectionNormalizer",
    "DetectionOutcome",
    "DetectionRuntime",
    "DetectionRuntimeStats",
    "DetectionScheduler",
    "DetectorBinding",
    "DetectorFactory",
    "DetectorRegistration",
    "DeviceWorker",
    "NormalizationOutcome",
    "NormalizationPolicy",
    "WorkerStats",
]
