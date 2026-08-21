"""L1 Acquisition — produce a correctly-identified, correctly-timestamped,
policy-masked frame stream.

Single responsibility for the layer as a whole: *get frames honestly. Know
nothing about what is in them.*

    M1 CameraManager        know what each camera is
    M2 VideoSourceManager   produce trustworthy frames
    M3 FrameScheduler       allocate scarce perception capacity
    M4 FrameBuffer          own pixel memory and its lifetime
"""

from __future__ import annotations

from .buffer import BufferStats, FrameBuffer, FrameLease, FrameSlot, PinHandle
from .camera_manager import CameraManager
from .scheduler import CameraRate, FrameScheduler, PressureReport
from .source_manager import (
    ActorState,
    EpochAllocator,
    EpochStore,
    InMemoryEpochStore,
    SourceActor,
    SourceBindings,
    SourceStatus,
    StreamStats,
    VideoSourceManager,
)

__all__ = [
    "ActorState",
    "BufferStats",
    "CameraManager",
    "CameraRate",
    "EpochAllocator",
    "EpochStore",
    "FrameBuffer",
    "FrameLease",
    "FrameScheduler",
    "FrameSlot",
    "InMemoryEpochStore",
    "PinHandle",
    "PressureReport",
    "SourceActor",
    "SourceBindings",
    "SourceStatus",
    "StreamStats",
    "VideoSourceManager",
]
