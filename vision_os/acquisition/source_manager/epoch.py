"""Stream epoch allocation (02_VISION_OBJECT_MODEL §4.1).

A ``StreamEpoch`` increments on every reconnect or reconfigure so that
``FrameRef`` stays genuinely unique for the deployment's lifetime.

The architecture requires that the last used epoch survive process restart:
"The only thing that must survive restart is the *last used epoch*, persisted
cheaply so epochs remain monotonic across restarts." Without it, a restart can
reuse an epoch and reintroduce exactly the ``FrameRef`` collision the epoch
exists to prevent.

The architecture is silent on *which* storage contract holds it, so this is a
module-private persistence protocol rather than a new entry in the platform port
catalogue — M2's own state, persisted by M2. The in-memory implementation is the
default; a file-backed one ships as an adapter.
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

from ...core.model.ids import CameraId, StreamEpoch


@runtime_checkable
class EpochStore(Protocol):
    """Module-private durable store for the highest epoch ever issued."""

    def last_epoch(self, camera_id: CameraId) -> StreamEpoch: ...

    def record_epoch(self, camera_id: CameraId, epoch: StreamEpoch) -> None: ...


class InMemoryEpochStore:
    """Non-durable default. Correct within a process lifetime."""

    __slots__ = ("_epochs", "_lock")

    def __init__(self) -> None:
        self._epochs: dict[CameraId, StreamEpoch] = {}
        self._lock = threading.Lock()

    def last_epoch(self, camera_id: CameraId) -> StreamEpoch:
        with self._lock:
            return self._epochs.get(camera_id, StreamEpoch(-1))

    def record_epoch(self, camera_id: CameraId, epoch: StreamEpoch) -> None:
        with self._lock:
            current = self._epochs.get(camera_id, StreamEpoch(-1))
            if epoch > current:
                self._epochs[camera_id] = epoch


class EpochAllocator:
    """Issues strictly increasing epochs, monotonic across restarts."""

    __slots__ = ("_store", "_lock")

    def __init__(self, store: EpochStore) -> None:
        self._store = store
        self._lock = threading.Lock()

    def next_epoch(self, camera_id: CameraId) -> StreamEpoch:
        with self._lock:
            nxt = StreamEpoch(self._store.last_epoch(camera_id) + 1)
            self._store.record_epoch(camera_id, nxt)
            return nxt
