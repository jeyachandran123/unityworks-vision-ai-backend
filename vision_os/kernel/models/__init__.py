"""M18 Model Manager — artifacts, devices, residency, versioning, calibration."""

from __future__ import annotations

from .calibration import (
    CalibrationMethod,
    CalibrationProfile,
    CalibrationRegistry,
)
from .devices import CPU_DEVICE_ID, DeviceBroker, DeviceReport, DeviceState, Reservation
from .manager import (
    ModelHandle,
    ModelManager,
    ModelSpec,
    ResidencyReport,
    RoleBinding,
    RolloutMode,
)

__all__ = [
    "CPU_DEVICE_ID",
    "CalibrationMethod",
    "CalibrationProfile",
    "CalibrationRegistry",
    "DeviceBroker",
    "DeviceReport",
    "DeviceState",
    "ModelHandle",
    "ModelManager",
    "ModelSpec",
    "ResidencyReport",
    "Reservation",
    "RoleBinding",
    "RolloutMode",
]
