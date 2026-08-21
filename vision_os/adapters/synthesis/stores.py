"""Observation log and sink adapters — P20 and P19.

**These are adapters, not M13.** §M13's single responsibility is *"Describe what
must persist and with what guarantees; **implement none of it**"* — it owns no
state and is a set of contracts. Shipping an adapter behind one of those
contracts is the same act Flow 2 performed for P25–P27 and Flow 4 for P21.

Two logs ship:

``log.memory``
    Everything in a list. Correct, fast and volatile — the honest choice for a
    single-camera embedded box that accepts session-scoped history, and the only
    sane choice for a test. It says so through ``log_id``, so nobody mistakes it
    for durability.

``log.file``
    Append-only JSON Lines, one file per partition. 07_STATE §9.1 makes total log
    loss *"a critical incident"*, so a deployment claiming durability binds this
    (or something stronger) and replicates it.

Both are **idempotent by ``observation_id``** (L2), which is what makes
07_STATE §9.1's *"restart, replay from the last committed log position"* safe:
a retry after an uncertain outcome cannot double-count.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

from ...core.errors import LogUnavailableError
from ...core.model.ids import CameraId, LogPosition, ObservationId
from ...core.model.observation import Observation
from ...core.model.timebase import Instant
from ...core.ports.synthesis import LogAppendResult, SinkResult


class InMemoryObservationLog:
    """``log.memory`` — correct, fast, volatile.

    Not a fallback for failure: it is the honest implementation for a deployment
    that accepts session-scoped history, and it declares that through its id
    rather than pretending otherwise.
    """

    __slots__ = ("_ids", "_lock", "_records")

    def __init__(self) -> None:
        self._records: dict[CameraId, list[Observation]] = {}
        self._ids: dict[CameraId, set[ObservationId]] = {}
        self._lock = threading.Lock()

    @property
    def log_id(self) -> str:
        return "log.memory"

    def append(
        self, partition: CameraId, observations: Sequence[Observation]
    ) -> LogAppendResult:
        with self._lock:
            records = self._records.setdefault(partition, [])
            seen = self._ids.setdefault(partition, set())
            appended = 0
            duplicates: list[ObservationId] = []
            for observation in observations:
                if observation.observation_id in seen:
                    duplicates.append(observation.observation_id)
                    continue
                records.append(observation)
                seen.add(observation.observation_id)
                appended += 1
            return LogAppendResult(
                position=LogPosition(len(records)),
                appended=appended,
                duplicates=tuple(duplicates),
            )

    def read(
        self,
        partition: CameraId,
        *,
        start: LogPosition | None = None,
        end: LogPosition | None = None,
        limit: int = 1000,
    ) -> Iterator[Observation]:
        with self._lock:
            records = list(self._records.get(partition, ()))
        begin = int(start or 0)
        finish = int(end) if end is not None else len(records)
        return iter(records[begin:finish][:limit])

    def tail(
        self, partition: CameraId, *, start: LogPosition | None = None, limit: int = 1000
    ) -> Iterator[Observation]:
        """Follow from ``start`` to the current end (L7).

        A snapshot of the tail at call time, not a live cursor: the caller polls,
        advancing ``start`` by what it received. That keeps the adapter free of
        any notion of a waiting subscriber, which is what lets a file, a Kafka
        topic and an in-memory list all satisfy the same contract.
        """
        with self._lock:
            records = list(self._records.get(partition, ()))
        return iter(records[int(start or 0) :][:limit])

    def position(self, partition: CameraId) -> LogPosition:
        with self._lock:
            return LogPosition(len(self._records.get(partition, ())))

    def truncate(self, partition: CameraId, before: Instant) -> int:
        """Retention only. Removes a time-bounded **prefix**, never a middle.

        A prefix because the log is ordered: removing from the middle would
        break the position arithmetic that makes rebuild resumable, and 07_STATE
        §8.2 refuses rewriting history in any case.
        """
        with self._lock:
            records = self._records.get(partition)
            if not records:
                return 0
            keep_from = 0
            for index, observation in enumerate(records):
                if observation.t_capture.ns >= before.ns:
                    keep_from = index
                    break
            else:
                keep_from = len(records)
            removed = records[:keep_from]
            self._records[partition] = records[keep_from:]
            seen = self._ids.get(partition)
            if seen is not None:
                for observation in removed:
                    seen.discard(observation.observation_id)
            return len(removed)

    def __len__(self) -> int:
        return sum(len(records) for records in self._records.values())


class FileObservationLog:
    """``log.file`` — append-only JSON Lines, one file per partition.

    JSON Lines because the format's whole virtue here is that an append is a
    single write with no rewrite of what came before, which is exactly what an
    append-only log wants and what makes partial corruption recoverable — a torn
    last line loses one record rather than the file.

    Idempotency is held in memory: the id set is rebuilt on first touch by
    scanning the partition, so a restart re-derives it from the record itself
    rather than from a sidecar that could disagree.
    """

    __slots__ = ("_ids", "_lock", "_positions", "_root")

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._ids: dict[CameraId, set[ObservationId]] = {}
        self._positions: dict[CameraId, int] = {}
        self._lock = threading.Lock()

    @property
    def log_id(self) -> str:
        return "log.file"

    def _path(self, partition: CameraId) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(partition))
        return self._root / f"{safe}.jsonl"

    def _load(self, partition: CameraId) -> None:
        """Rebuild the id set and position from the file, once per partition."""
        if partition in self._ids:
            return
        seen: set[ObservationId] = set()
        count = 0
        path = self._path(partition)
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        # A torn final line from an interrupted write. Skipping
                        # it loses one record rather than the file, which is the
                        # reason for the line-oriented format.
                        continue
                    seen.add(ObservationId(record["observation_id"]))
                    count += 1
        self._ids[partition] = seen
        self._positions[partition] = count

    def append(
        self, partition: CameraId, observations: Sequence[Observation]
    ) -> LogAppendResult:
        with self._lock:
            self._load(partition)
            seen = self._ids[partition]
            fresh = [o for o in observations if o.observation_id not in seen]
            duplicates = tuple(
                o.observation_id for o in observations if o.observation_id in seen
            )

            if fresh:
                path = self._path(partition)
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("a", encoding="utf-8") as handle:
                        for observation in fresh:
                            handle.write(json.dumps(_encode(observation)) + "\n")
                        handle.flush()
                except OSError as exc:
                    raise LogUnavailableError(
                        f"cannot append to the observation log at {path}: {exc}",
                        partition=str(partition),
                    ) from exc
                seen.update(o.observation_id for o in fresh)
                self._positions[partition] += len(fresh)

            return LogAppendResult(
                position=LogPosition(self._positions[partition]),
                appended=len(fresh),
                duplicates=duplicates,
            )

    def read(
        self,
        partition: CameraId,
        *,
        start: LogPosition | None = None,
        end: LogPosition | None = None,
        limit: int = 1000,
    ) -> Iterator[Observation]:
        """Range-read in append order.

        Returns **references**, not full observations: a file log stores the
        envelope as JSON and rehydrating every field would require the whole
        object graph. What comes back is what a rebuild and a history query
        need — see ``_decode``'s note.
        """
        path = self._path(partition)
        if not path.exists():
            return iter(())
        begin = int(start or 0)
        finish = int(end) if end is not None else None
        out: list[Observation] = []
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index < begin:
                    continue
                if finish is not None and index >= finish:
                    break
                if len(out) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                decoded = _decode(line)
                if decoded is not None:
                    out.append(decoded)
        return iter(out)

    def tail(
        self, partition: CameraId, *, start: LogPosition | None = None, limit: int = 1000
    ) -> Iterator[Observation]:
        """Follow from ``start`` to the end of the file (L7).

        Delegates to ``read`` with no upper bound. A production adapter would
        hold the file open and follow appends; re-scanning is honest for a
        reference implementation and costs a caller nothing it cannot see, since
        ``limit`` bounds every call.
        """
        return self.read(partition, start=start, end=None, limit=limit)

    def position(self, partition: CameraId) -> LogPosition:
        with self._lock:
            self._load(partition)
            return LogPosition(self._positions[partition])

    def truncate(self, partition: CameraId, before: Instant) -> int:
        with self._lock:
            path = self._path(partition)
            if not path.exists():
                return 0
            kept: list[str] = []
            removed = 0
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                    except ValueError:
                        continue
                    if record.get("t_capture_ns", 0) < before.ns:
                        removed += 1
                        continue
                    kept.append(stripped)
            if removed:
                path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
                self._ids.pop(partition, None)
                self._positions.pop(partition, None)
            return removed


class CollectingSink:
    """``sink.collecting`` — keeps everything it is given, in order.

    For tests and for a single-node deployment that wants a tee it can inspect.
    Declares itself **not durable**, because obligation K5 exists so a tee to a
    dashboard is never mistaken for a system of record.
    """

    __slots__ = ("_lock", "_observations")

    def __init__(self) -> None:
        self._observations: list[Observation] = []
        self._lock = threading.Lock()

    @property
    def sink_id(self) -> str:
        return "sink.collecting"

    @property
    def durable(self) -> bool:
        return False

    def emit(self, observations: Sequence[Observation]) -> SinkResult:
        with self._lock:
            self._observations.extend(observations)
        return SinkResult(accepted=len(observations))

    @property
    def observations(self) -> tuple[Observation, ...]:
        with self._lock:
            return tuple(self._observations)

    def reset(self) -> None:
        """Forget everything collected.

        Exists for the composition root, which runs the conformance kit against
        the instance a deployment will actually use — a store can only be shown
        to store by storing. Without this, the first thing an operator saw in a
        freshly booted sink would be the kit's fixtures.
        """
        with self._lock:
            self._observations.clear()

    def __len__(self) -> int:
        return len(self._observations)


class NullSink:
    """``sink.null`` — accepts and discards.

    The honest no-op for a deployment that wants no tee. Declaring it explicitly
    beats binding nothing, because *"no sink configured"* and *"a sink that
    silently drops"* are different situations and only one is intended.
    """

    __slots__ = ("_count",)

    def __init__(self) -> None:
        self._count = 0

    @property
    def sink_id(self) -> str:
        return "sink.null"

    @property
    def durable(self) -> bool:
        return False

    def emit(self, observations: Sequence[Observation]) -> SinkResult:
        self._count += len(observations)
        return SinkResult(accepted=len(observations))

    @property
    def received(self) -> int:
        return self._count


def _encode(observation: Observation) -> dict:
    """The envelope's durable form.

    Lossy on the *heavy* nested objects — quality grades, the decision path, the
    evidence body — because 07_STATE §9 uses the log for rebuild and audit, and a
    full serializer is an M13 adapter concern.

    **Spatial payload is not among them, and originally was.** Dropping it made
    every presence record undecodable, because 02_VOM requires a presence
    observation to carry a position and the model rightly refuses to construct
    one without it. The failure was silent — each record decoded to ``None`` and
    the whole log read back empty — which is the exact shape of defect §9.1
    exists to prevent: *"replay from the last committed log position"* would have
    produced an empty world rather than the recorded one. A normalized bounding
    box is four floats, so there was never a size argument for omitting it.
    """
    return {
        "spatial": _encode_spatial(observation.spatial),
        "coverage": _encode_coverage(observation.coverage),
        "lifecycle_transition": (
            {
                "previous": observation.lifecycle_transition.previous.value,
                "current": observation.lifecycle_transition.current.value,
                "trigger": observation.lifecycle_transition.trigger,
            }
            if observation.lifecycle_transition
            else None
        ),
        "observation_id": str(observation.observation_id),
        "observation_type": observation.observation_type.value,
        "schema_version": observation.schema_version,
        "tenant_id": str(observation.tenant_id),
        "site_id": str(observation.site_id),
        "camera_id": str(observation.camera_id),
        "frame_ref": str(observation.frame_ref),
        "t_capture_ns": observation.t_capture.ns,
        "t_capture_unc_ns": observation.t_capture_unc.ns,
        "clock_quality": observation.clock_quality.value[0],
        "t_published_ns": observation.t_published.ns,
        "object_id": str(observation.object_id) if observation.object_id else None,
        "class_id": str(observation.class_id) if observation.class_id else None,
        "taxonomy_version": observation.taxonomy_version,
        "lifecycle_state": (
            observation.lifecycle_state.value if observation.lifecycle_state else None
        ),
        "measurement_basis": observation.measurement_basis.value,
        "confidence": (
            {
                "value": observation.confidence.value,
                "semantics": observation.confidence.semantics.value,
                "calibrated": observation.confidence.calibrated,
            }
            if observation.confidence
            else None
        ),
        "attributes": [
            {
                "key": str(a.key),
                "value": a.value,
                "schema_version": a.schema_version,
                "confidence": a.confidence.value,
                "confidence_semantics": a.confidence.semantics.value,
                "observed_at_ns": a.observed_at.ns,
            }
            for a in observation.attributes
        ],
        "evidence_id": (
            str(observation.evidence_ref.evidence_id)
            if observation.evidence_ref
            else None
        ),
        "provenance": {
            "producer_module": str(observation.provenance.producer_module),
            "producer_version": observation.provenance.producer_version,
            "config_revision": str(observation.provenance.config_revision),
            "model_id": (
                str(observation.provenance.model_id)
                if observation.provenance.model_id
                else None
            ),
            "model_artifact_hash": observation.provenance.model_artifact_hash,
        },
        "supersedes": str(observation.supersedes) if observation.supersedes else None,
        "lineage": [str(o) for o in observation.lineage],
        "demand_ids": [str(d) for d in observation.demand_ids],
    }


def _encode_spatial(spatial) -> dict | None:
    """The position, small enough to keep and required to decode.

    ``ground_point`` travels with its uncertainty or not at all — 02_VOM §6.2
    refuses a metre measurement whose error nobody recorded, and a decoder that
    reconstructed one without the other would manufacture false precision.
    """
    if spatial is None:
        return None
    box = spatial.bbox
    return {
        "frame_of_reference": spatial.frame_of_reference.value,
        "calibration_id": (
            str(spatial.calibration_id) if spatial.calibration_id else None
        ),
        "bbox": [box.x1, box.y1, box.x2, box.y2] if box is not None else None,
        "ground_point": (
            [spatial.ground_point.x, spatial.ground_point.y]
            if spatial.ground_point is not None and spatial.ground_uncertainty is not None
            else None
        ),
        "ground_uncertainty": (
            [
                spatial.ground_uncertainty.semi_major,
                spatial.ground_uncertainty.semi_minor,
                spatial.ground_uncertainty.orientation_rad,
            ]
            if spatial.ground_point is not None and spatial.ground_uncertainty is not None
            else None
        ),
    }


def _encode_coverage(window) -> dict | None:
    """The coverage window. Kept for the same reason as spatial payload.

    07_STATE §7.3 wants *"a query over any past window can reconstruct exactly
    what was observable then"*, and that query reads the log. Dropping the window
    would leave the platform unable to say what it could see, which is the one
    thing V8 requires it never lose.
    """
    if window is None:
        return None
    return {
        "status": window.status.value,
        "reason": window.reason.value,
        "since_ns": window.since.ns,
        "until_ns": window.until.ns if window.until is not None else None,
        "effective_rate": window.effective_rate,
        "regions_affected": list(window.regions_affected),
        "capability_gaps": [list(gap) for gap in window.capability_gaps],
    }


def _decode(line: str) -> Observation | None:
    """Rehydrate what the encoder wrote.

    Returns ``None`` for a record this adapter cannot reconstruct rather than
    raising: a rebuild reading a log written by an older schema must skip what it
    cannot read and continue, because halting would make an old log
    unrebuildable — the opposite of what a log is for.
    """
    from .decode import decode_observation

    return decode_observation(line)
