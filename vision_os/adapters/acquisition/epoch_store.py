"""Durable stream-epoch storage (02_VOM §4.1, M2 state ownership).

The architecture requires the last used epoch to survive process restart:
without it, a restart can reuse an epoch and reintroduce exactly the ``FrameRef``
collision the epoch exists to prevent — a bug found in production months later,
by accident.

This is M2's own module-private persistence, not a platform port; see
``acquisition.source_manager.epoch`` for the rationale.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

from ...core.model.ids import CameraId, StreamEpoch


class JsonFileEpochStore:
    """Persist the highest epoch ever issued, per camera.

    Writes are atomic (temp file + replace) so that a crash mid-write cannot
    leave a truncated file that reads back as "no epochs ever issued" — which
    would silently defeat the entire mechanism.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._epochs: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt store must not prevent startup. Epochs restart, which is
            # safe because the allocator only ever increases from what it reads.
            return
        if isinstance(raw, dict):
            self._epochs = {str(k): int(v) for k, v in raw.items() if isinstance(v, int)}

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=".epochs-", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump(self._epochs, file)
            os.replace(temp_name, self._path)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise

    def last_epoch(self, camera_id: CameraId) -> StreamEpoch:
        with self._lock:
            return StreamEpoch(self._epochs.get(str(camera_id), -1))

    def record_epoch(self, camera_id: CameraId, epoch: StreamEpoch) -> None:
        with self._lock:
            key = str(camera_id)
            if epoch > self._epochs.get(key, -1):
                self._epochs[key] = int(epoch)
                self._flush()
