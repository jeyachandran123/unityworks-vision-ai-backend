"""The durable observation log this application runs.

The platform's `FileObservationLog` — same directory, same one-file-per-camera
JSON Lines, same append-only records, byte for byte — with the three costs
that grew with history taken out. `vision_os/` is untouched: this is an
adapter, which is where the platform expects deployment technology to live, and
it is gated at boot by the platform's own observation-log conformance kit, run
against a twin of *this* class.

### What was wrong

`FileObservationLog._load` runs on the first write to each camera after the
process starts. It read the camera's whole file, parsed every line as JSON and
kept every observation id it had ever written in a set — under the lock the
analysis thread appends through. Measured on the live deployment on
2026-09-25: 9.5 million observations in 10 GB, loaded at about 2,900 lines a
second, left perception blind for roughly 55 minutes after **every** restart.
No person was detected, so no finding, no incident and no alert; and the id set
held the whole history in memory for the life of the process.

`truncate`, the retention sweep's one tool, had the same shape: it parsed every
line, held every kept line in memory, rewrote the file and then forgot the
partition, so the next write paid the full load again.

### What this does instead

* **Start-up counts, it does not parse.** A partition's position is its number
  of records, which is its number of lines, counted in C over the raw bytes.
  Only the most recent records are parsed, to remember their ids.
* **Idempotency is kept for recent records.** Rejecting a re-appended
  observation matters for a batch retried moments after a failed write; no
  path re-sends an observation from days ago. Each partition remembers its most
  recent ids, a bounded number, instead of every id since the camera was added.
* **Retention streams.** Old records are skipped line by line into a new file
  that replaces the old one atomically; nothing is held in memory and a crash
  mid-way leaves the original intact.

### One difference, stated

The platform counted a partition's position as its *parseable* lines; this
counts its lines. They differ only in a file with a corrupt line in the middle
(a torn write that a later append ran on from), and there this count agrees
with `read()`, which has always indexed by line.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import tempfile
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from vision_os.adapters.synthesis import FileObservationLog
from vision_os.core.model.ids import CameraId, ObservationId
from vision_os.core.model.timebase import Instant

#: Observation ids each partition remembers, for rejecting a re-appended record.
#: A retry happens within seconds of the write it repeats; this is hours of a
#: busy camera, and a few megabytes.
RECENT_IDS = 50_000
#: How many of a partition's newest records start-up parses to recover them.
START_IDS = 5_000

_CHUNK = 8 * 1024 * 1024
_TAIL_BYTES = 16 * 1024 * 1024
_T_CAPTURE = re.compile(rb'"t_capture_ns":\s*(-?\d+)')


class RecentIds(set):
    """A set that keeps only its most recently added members.

    Drop-in for the plain set the platform's `append` reads and updates, so the
    inherited method works unchanged while memory stays bounded.
    """

    __slots__ = ("_capacity", "_order")

    def __init__(self, ids: Iterable[Any] = (), *, capacity: int = RECENT_IDS) -> None:
        super().__init__()
        self._capacity = max(1, int(capacity))
        self._order: deque[Any] = deque()
        self.update(ids)

    def add(self, item: Any) -> None:
        if item in self:
            return
        super().add(item)
        self._order.append(item)
        while len(self._order) > self._capacity:
            super().discard(self._order.popleft())

    def update(self, *iterables: Iterable[Any]) -> None:
        for iterable in iterables:
            for item in iterable:
                self.add(item)


def _count_lines(path: Path) -> int:
    """Complete lines — the records a reader will see.

    A torn final line has no newline and is not counted; the platform's own
    load skipped it too, as unparseable. Blank lines are never written.
    """
    count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            count += chunk.count(b"\n")
    return count


def _recent_ids(path: Path, *, limit: int) -> list[ObservationId]:
    """The ids of up to `limit` newest records, oldest first."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        start = max(0, size - _TAIL_BYTES)
        handle.seek(start)
        data = handle.read()
    lines = data.split(b"\n")
    if start > 0:
        lines = lines[1:]  # the first piece is the end of a line cut in half
    ids: list[ObservationId] = []
    for raw in lines[-(limit + 1) :]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            ids.append(ObservationId(json.loads(raw)["observation_id"]))
        except (ValueError, KeyError, TypeError):
            continue  # a torn line, as the platform's load would skip
    return ids[-limit:]


def _captured_before(raw: bytes, before_ns: int) -> bool | None:
    """Whether a record precedes `before_ns`; `None` for an unreadable line.

    The capture time is read without parsing the record when it appears once,
    which is how every record is written; anything else is parsed properly.
    """
    matches = _T_CAPTURE.findall(raw)
    if len(matches) == 1:
        return int(matches[0]) < before_ns
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    return int(record.get("t_capture_ns", 0)) < before_ns


class BoundedFileObservationLog(FileObservationLog):
    """`FileObservationLog` whose start-up, retention and memory do not grow
    with a camera's history."""

    def __init__(
        self,
        root: Path | str,
        *,
        recent_ids: int = RECENT_IDS,
        start_ids: int = START_IDS,
    ) -> None:
        super().__init__(root)
        self._recent_capacity = recent_ids
        self._start_ids = start_ids

    def _load(self, partition: CameraId) -> None:
        """Position from a line count; idempotency from the newest records."""
        if partition in self._ids:
            return
        path = self._path(partition)
        if path.exists():
            count = _count_lines(path)
            ids = _recent_ids(path, limit=self._start_ids)
        else:
            count, ids = 0, []
        self._ids[partition] = RecentIds(ids, capacity=self._recent_capacity)
        self._positions[partition] = count

    def truncate(self, partition: CameraId, before: Instant) -> int:
        """Remove records captured before `before`, streaming, never in memory."""
        with self._lock:
            path = self._path(partition)
            if not path.exists():
                return 0
            removed = 0
            dropped = 0
            fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as target, path.open("rb") as source:
                    for raw in source:
                        stripped = raw.strip()
                        if not stripped:
                            continue
                        old = _captured_before(stripped, before.ns)
                        if old is None:
                            dropped += 1  # unreadable; the platform dropped these too
                            continue
                        if old:
                            removed += 1
                            continue
                        target.write(stripped + b"\n")
                if removed or dropped:
                    os.replace(temporary, path)
                    # Positions renumber from the start of the new file.
                    self._ids.pop(partition, None)
                    self._positions.pop(partition, None)
                else:
                    os.unlink(temporary)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(temporary)
                raise
            return removed

    def for_conformance(self) -> tuple[BoundedFileObservationLog, Callable[[], None]]:
        """A disposable twin **of this class**, so the platform's gate certifies
        the adapter that actually runs rather than the one it replaced."""
        root = Path(tempfile.mkdtemp(prefix="conformance-log-"))
        twin = type(self)(root, recent_ids=self._recent_capacity, start_ids=self._start_ids)

        def dispose() -> None:
            shutil.rmtree(root, ignore_errors=True)

        return twin, dispose


__all__ = [
    "RECENT_IDS",
    "START_IDS",
    "BoundedFileObservationLog",
    "RecentIds",
]
