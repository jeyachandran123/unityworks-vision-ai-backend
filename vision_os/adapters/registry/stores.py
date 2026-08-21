"""``ObjectStorePort`` adapters — durable object state.

``07_STATE`` section 9.3 requires that **object identity survives a restart**
while tracks do not. These are the narrowest implementations that satisfy it
without building the Storage Interfaces module (M12), which belongs to a later
flow.

Two ship:

``InMemoryObjectStore``
    For tests and for embedded deployments that accept losing identity on
    restart. Honest about what it is: nothing survives the process.

``FileObjectStore``
    Atomic per-partition writes to the local filesystem, dependency-free. A
    partially written partition is worse than a lost one — it reloads as
    plausible corruption — so every write goes to a temporary file and is
    renamed into place, which is atomic on every filesystem the platform
    targets.

**Neither ever repairs.** A snapshot that fails to decode raises; it is never
silently downgraded to an empty partition, because that presents data loss as a
fresh start (obligation S3).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from ...core.errors import ObjectStoreError
from ...core.model.confidence import Confidence, ConfidenceSemantics
from ...core.model.ids import (
    BindingId,
    CameraId,
    ClassId,
    LocalTrackId,
    ObjectId,
    SiteId,
    TenantId,
    TrackerEpoch,
    TrackId,
)
from ...core.model.provenance import Provenance
from ...core.model.space import Box, FrameOfReference, SpatialInfo
from ...core.model.timebase import Instant
from ...core.model.visual_object import (
    BindingMethod,
    ClassObservation,
    LifecycleState,
    TrackBinding,
    VisualObject,
)
from ...core.ports.registry import PartitionSnapshot

SNAPSHOT_FORMAT_VERSION = 1


class InMemoryObjectStore:
    """Partitions held in memory. Nothing survives the process.

    Not a stub: an edge deployment that treats object identity as session-scoped
    is a legitimate configuration, and saying so plainly is better than a durable
    store nobody configured a path for.
    """

    def __init__(self) -> None:
        self._partitions: dict[CameraId, PartitionSnapshot] = {}
        self._lock = threading.Lock()

    @property
    def store_id(self) -> str:
        return "memory"

    def save(self, snapshot: PartitionSnapshot) -> None:
        with self._lock:
            self._partitions[snapshot.camera_id] = snapshot

    def load(self, camera_id: CameraId) -> PartitionSnapshot | None:
        with self._lock:
            return self._partitions.get(camera_id)

    def forget(self, camera_id: CameraId) -> None:
        with self._lock:
            self._partitions.pop(camera_id, None)

    def __len__(self) -> int:
        return len(self._partitions)


class FileObjectStore:
    """Atomic per-partition JSON on the local filesystem.

    JSON rather than pickle deliberately: a durable format that only one Python
    version can read is a migration problem waiting for the worst moment, and
    the object model is small enough that the encoding cost is irrelevant next
    to the persistence interval.
    """

    def __init__(self, directory: Path | str) -> None:
        self._directory = Path(directory)
        self._lock = threading.Lock()

    @property
    def store_id(self) -> str:
        return f"file:{self._directory}"

    def _path(self, camera_id: CameraId) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(camera_id))
        return self._directory / f"{safe}.json"

    def save(self, snapshot: PartitionSnapshot) -> None:
        """Write one partition atomically.

        Temp file plus rename: a crash mid-write leaves the previous snapshot
        intact rather than a truncated one that would reload as corruption.
        """
        payload = _encode_snapshot(snapshot)
        with self._lock:
            try:
                self._directory.mkdir(parents=True, exist_ok=True)
                handle, temporary = tempfile.mkstemp(
                    dir=str(self._directory), suffix=".tmp"
                )
                try:
                    with os.fdopen(handle, "w", encoding="utf-8") as stream:
                        json.dump(payload, stream)
                    os.replace(temporary, self._path(snapshot.camera_id))
                except BaseException:
                    Path(temporary).unlink(missing_ok=True)
                    raise
            except OSError as exc:
                raise ObjectStoreError(
                    f"could not persist partition '{snapshot.camera_id}': {exc}",
                    camera_id=str(snapshot.camera_id),
                ) from exc

    def load(self, camera_id: CameraId) -> PartitionSnapshot | None:
        path = self._path(camera_id)
        with self._lock:
            if not path.exists():
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                # Never downgrade to an empty partition (obligation S3): losing
                # every object silently is worse than failing loudly, because the
                # platform would report a fresh start rather than data loss.
                raise ObjectStoreError(
                    f"partition '{camera_id}' exists but could not be decoded: "
                    f"{exc}. Refusing to present data loss as a fresh start.",
                    camera_id=str(camera_id),
                ) from exc
        try:
            return _decode_snapshot(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise ObjectStoreError(
                f"partition '{camera_id}' decoded but is structurally invalid: {exc}",
                camera_id=str(camera_id),
            ) from exc

    def forget(self, camera_id: CameraId) -> None:
        with self._lock:
            self._path(camera_id).unlink(missing_ok=True)


# --- encoding ---------------------------------------------------------------- #


def _encode_snapshot(snapshot: PartitionSnapshot) -> dict[str, Any]:
    return {
        "format": SNAPSHOT_FORMAT_VERSION,
        "camera_id": str(snapshot.camera_id),
        "site_id": str(snapshot.site_id),
        "version": snapshot.version,
        "taken_at_ns": snapshot.taken_at.ns,
        "next_local_sequence": snapshot.next_local_sequence,
        "objects": [_encode_object(o) for o in snapshot.objects],
    }


def _decode_snapshot(payload: dict[str, Any]) -> PartitionSnapshot:
    version = payload.get("format")
    if version != SNAPSHOT_FORMAT_VERSION:
        raise ValueError(
            f"snapshot format {version} is not {SNAPSHOT_FORMAT_VERSION}; a "
            f"format change is a migration, never a silent reinterpretation"
        )
    return PartitionSnapshot(
        camera_id=CameraId(payload["camera_id"]),
        site_id=SiteId(payload["site_id"]),
        version=int(payload["version"]),
        taken_at=Instant(int(payload["taken_at_ns"])),
        objects=tuple(_decode_object(o) for o in payload["objects"]),
        next_local_sequence=int(payload.get("next_local_sequence", 0)),
    )


def _encode_object(obj: VisualObject) -> dict[str, Any]:
    return {
        "object_id": str(obj.object_id),
        "tenant_id": str(obj.tenant_id),
        "site_id": str(obj.site_id),
        "camera_id": str(obj.camera_id),
        "class_id": str(obj.class_id),
        "confidence": obj.confidence.value,
        "lifecycle": obj.lifecycle.value,
        "class_history": [
            {
                "class_id": str(c.class_id),
                "observed_at_ns": c.observed_at.ns,
                "confidence": c.confidence.value,
            }
            for c in obj.class_history
        ],
        "track_bindings": [_encode_binding(b) for b in obj.track_bindings],
        "spatial": _encode_spatial(obj.current_spatial),
        "spatial_history": [
            {"at_ns": t.ns, "spatial": _encode_spatial(s)} for t, s in obj.spatial_history
        ],
        "attributes": sorted(str(k) for k in obj.attributes),
        "first_seen_ns": obj.first_seen.ns,
        "last_seen_ns": obj.last_seen.ns,
        "last_confirmed_ns": obj.last_confirmed.ns,
        "observation_count": obj.observation_count,
        "merged_into": str(obj.merged_into) if obj.merged_into else None,
        "lineage": [str(o) for o in obj.lineage],
        "schema_version": obj.schema_version,
        "provenance": {
            "producer_module": str(obj.provenance.producer_module),
            "producer_version": obj.provenance.producer_version,
            "config_revision": str(obj.provenance.config_revision),
            "deterministic": obj.provenance.deterministic,
        },
    }


def _decode_object(payload: dict[str, Any]) -> VisualObject:
    provenance = payload["provenance"]
    return VisualObject(
        object_id=ObjectId(payload["object_id"]),
        tenant_id=TenantId(payload["tenant_id"]),
        site_id=SiteId(payload["site_id"]),
        camera_id=CameraId(payload["camera_id"]),
        class_id=ClassId(payload["class_id"]),
        confidence=Confidence.uncalibrated(
            float(payload["confidence"]), ConfidenceSemantics.IDENTITY
        ),
        lifecycle=LifecycleState(payload["lifecycle"]),
        class_history=tuple(
            ClassObservation(
                class_id=ClassId(c["class_id"]),
                observed_at=Instant(int(c["observed_at_ns"])),
                confidence=Confidence.uncalibrated(
                    float(c["confidence"]), ConfidenceSemantics.CLASSIFICATION
                ),
            )
            for c in payload["class_history"]
        ),
        track_bindings=tuple(_decode_binding(b) for b in payload["track_bindings"]),
        current_spatial=_decode_spatial(payload["spatial"]),
        spatial_history=tuple(
            (Instant(int(s["at_ns"])), _decode_spatial(s["spatial"]))
            for s in payload["spatial_history"]
        ),
        # Attributes are intentionally not restored: their values live in the
        # observation log, which is the system of record. Restoring keys without
        # values would present a stale claim as current.
        attributes={},
        first_seen=Instant(int(payload["first_seen_ns"])),
        last_seen=Instant(int(payload["last_seen_ns"])),
        last_confirmed=Instant(int(payload["last_confirmed_ns"])),
        observation_count=int(payload["observation_count"]),
        provenance=Provenance(
            producer_module=provenance["producer_module"],
            producer_version=provenance["producer_version"],
            config_revision=provenance["config_revision"],
            deterministic=bool(provenance.get("deterministic", False)),
        ),
        merged_into=(
            ObjectId(payload["merged_into"]) if payload.get("merged_into") else None
        ),
        lineage=tuple(ObjectId(o) for o in payload.get("lineage", ())),
        schema_version=payload.get("schema_version", "1.0.0"),
    )


def _encode_binding(binding: TrackBinding) -> dict[str, Any]:
    track = binding.track_id
    return {
        "binding_id": str(binding.binding_id),
        "track": {
            "camera_id": str(track.camera_id),
            "tracker_epoch": int(track.tracker_epoch),
            "local_id": int(track.local_id),
        },
        "bound_from_ns": binding.bound_from.ns,
        "bound_to_ns": binding.bound_to.ns if binding.bound_to else None,
        "confidence": binding.confidence.value if binding.confidence else None,
        "method": binding.method.value,
        "superseded_by": str(binding.superseded_by) if binding.superseded_by else None,
    }


def _decode_binding(payload: dict[str, Any]) -> TrackBinding:
    track = payload["track"]
    confidence = payload.get("confidence")
    return TrackBinding(
        binding_id=BindingId(payload["binding_id"]),
        track_id=TrackId(
            CameraId(track["camera_id"]),
            TrackerEpoch(int(track["tracker_epoch"])),
            LocalTrackId(int(track["local_id"])),
        ),
        bound_from=Instant(int(payload["bound_from_ns"])),
        bound_to=(
            Instant(int(payload["bound_to_ns"]))
            if payload.get("bound_to_ns") is not None
            else None
        ),
        confidence=(
            Confidence.uncalibrated(float(confidence), ConfidenceSemantics.IDENTITY)
            if confidence is not None
            else None
        ),
        method=BindingMethod(payload["method"]),
        superseded_by=(
            BindingId(payload["superseded_by"]) if payload.get("superseded_by") else None
        ),
    )


def _encode_spatial(spatial: SpatialInfo) -> dict[str, Any]:
    box = spatial.bbox
    return {
        "frame_of_reference": spatial.frame_of_reference.value,
        "bbox": [box.x1, box.y1, box.x2, box.y2] if box else None,
        "calibration_id": str(spatial.calibration_id) if spatial.calibration_id else None,
    }


def _decode_spatial(payload: dict[str, Any]) -> SpatialInfo:
    box = payload.get("bbox")
    return SpatialInfo(
        frame_of_reference=FrameOfReference(payload["frame_of_reference"]),
        bbox=Box(*box) if box else None,
        calibration_id=payload.get("calibration_id"),
    )


#: Keys the encoder writes. Used by the conformance kit to detect a field added
#: to ``VisualObject`` without being persisted — a silent durability gap.
ENCODED_OBJECT_KEYS: frozenset[str] = frozenset(
    {
        "object_id", "tenant_id", "site_id", "camera_id", "class_id", "confidence",
        "lifecycle", "class_history", "track_bindings", "spatial",
        "spatial_history", "attributes", "first_seen_ns", "last_seen_ns",
        "last_confirmed_ns", "observation_count", "merged_into", "lineage",
        "schema_version", "provenance",
    }
)
