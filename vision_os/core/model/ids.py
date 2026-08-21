"""Identifier construction (02_VISION_OBJECT_MODEL §4.1).

Identity is where vision systems most often accumulate permanent, invisible
corruption. Two constructions here carry outsized weight:

``FrameRef``
    ``(camera_id, stream_epoch, frame_seq)``. Every RTSP source eventually
    reconnects, and every naive implementation restarts frame numbering at zero
    — so frame 100 before the reconnect and frame 100 after it compare equal
    while describing different instants. The epoch makes reconnection explicit
    and keeps ``FrameRef`` genuinely unique for the deployment's lifetime.

``ULID``
    Time-sortable, mintable by any partition on any node without coordination.
    A central sequence would make identity allocation a distributed bottleneck
    at exactly the scale where it must not be.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import NewType

TenantId = NewType("TenantId", str)
SiteId = NewType("SiteId", str)
CameraId = NewType("CameraId", str)
RegionId = NewType("RegionId", str)
CalibrationId = NewType("CalibrationId", str)
ConfigRevision = NewType("ConfigRevision", str)
PluginId = NewType("PluginId", str)
PortId = NewType("PortId", str)
AdapterId = NewType("AdapterId", str)
ModuleId = NewType("ModuleId", str)
ProfileId = NewType("ProfileId", str)
PrivacyPolicyId = NewType("PrivacyPolicyId", str)

# --- perception identifiers (Flow 2 onward) ------------------------------- #

ModelId = NewType("ModelId", str)
"""Registry identity for a model. **Not a filename** — the same weights may live
at different paths on different nodes, and history must stay interpretable when
they move (02_VOM section 3)."""

ClassId = NewType("ClassId", str)
"""A platform Visual Taxonomy class, e.g. ``person`` or ``vehicle.forklift``.
Hierarchical by dotted path. Model-native label spaces never appear here."""

DetectionId = NewType("DetectionId", str)

StreamEpoch = NewType("StreamEpoch", int)
FrameSeq = NewType("FrameSeq", int)

# --- tracking identifiers (Flow 3) ---------------------------------------- #

TrackerEpoch = NewType("TrackerEpoch", int)
"""Monotonic per camera; +1 on every tracker reset. Tracks do not survive a
reset, so the epoch is what stops a recycled ``LocalTrackId`` from silently
appearing to continue a track it has nothing to do with."""

LocalTrackId = NewType("LocalTrackId", int)
"""A tracker's own counter, meaningful only inside one ``(camera, epoch)``."""

# --- registry identifiers (Flow 4) ---------------------------------------- #

ObjectId = NewType("ObjectId", str)
"""A ULID, **site-scoped and durable** (02_VOM section 4.1).

A ULID rather than a sequence because object identity must be mintable by any
partition on any node without coordination, and must sort by creation time for
efficient range queries. A central sequence would make the registry a distributed
bottleneck at exactly the scale where it must not be.

Minted by the Object Registry and by nothing else (01_LAYERED section 8:
*"Exactly one module may mint or retire an object identity"*)."""

BindingId = NewType("BindingId", str)
"""Identifies one track-to-object binding assertion, so a later revision can
supersede a specific claim rather than the whole object's history."""

AttributeKey = NewType("AttributeKey", str)
"""A registered attribute name. Registration is gated on visual neutrality
(02_VOM section 9.1)."""

# --- crop identifiers (Flow 5) -------------------------------------------- #

CropId = NewType("CropId", str)
"""**A content hash of the normalized crop pixels** (02_VOM section 4.1).

Content-addressed, not sequential, and the choice does real work: the same pixels
cropped twice must be one crop. That gives free deduplication, free cache keys,
free integrity checking, and an evidence reference that survives storage
migration. A sequential id would give none of those and would make two runs over
the same footage produce different evidence."""

DemandId = NewType("DemandId", str)
"""One consumer's declared need for attributes, in a scope, at a freshness."""

SubscriberId = NewType("SubscriberId", str)


# --- understanding identifiers (Flow 6) ----------------------------------- #

RequestId = NewType("RequestId", str)
"""One understanding request. Correlates a batch entry with its response.

Deliberately *not* derived from the crop: two demands may want different
attribute sets from the same crop, and they are different requests with different
costs, different prompts and different cache entries."""

PromptId = NewType("PromptId", str)
"""A prompt template, owned and versioned by M10.

The pair ``(prompt_id, version)`` is immutable once published — 04_MODULES §M10:
*"provenance is worthless if `prompt@3.2.0` means different things on different
days."*"""

PackId = NewType("PackId", str)
"""A prompt pack — the unit M10 loads and validates. A vertical's domain
knowledge enters the platform through packs, as **assets, not code**."""

BlobRef = NewType("BlobRef", str)
"""**A content hash of a stored byte payload** (02_VOM section 10.9).

Content-addressed for the same reasons as ``CropId``: identical model output
stored twice is one blob, the reference survives storage migration, and the
reference is valid the moment a store exists to resolve it. Flow 6 computes these
and carries the bytes; persisting them is M13's job through P22."""

EvidenceId = NewType("EvidenceId", str)
"""One evidence record. Minted by the producer of the evidence."""


# --- observation and state identifiers (Flow 7) --------------------------- #

ObservationId = NewType("ObservationId", str)
"""A ULID, **time-sortable** (02_VOM section 11).

Time-sortable because the observation log is the system of record and is read by
range: a lexicographic scan over ids is a chronological scan over facts, with no
secondary index. Minted by the Observation Builder and by nothing else — an
observation is a published fact, and only the choke point may publish."""

LogPosition = NewType("LogPosition", int)
"""A partition-local, monotonically increasing offset into the observation log.

07_STATE section 3.2 calls it *the projection watermark*: it makes "is the
projection caught up?" answerable, makes rebuild resumable, and makes snapshot
consistency expressible."""

PartitionVersion = NewType("PartitionVersion", int)
"""Monotonic per camera partition. Bumped on every applied observation, so a
reader holding a snapshot can say exactly which version it holds (07_STATE
section 5)."""


# --- ULID ---------------------------------------------------------------- #

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_LEN = 26


def new_ulid(now_ms: int | None = None) -> str:
    """Mint a lexicographically time-sortable 26-character ULID.

    Pure stdlib by design — ``core`` may not depend on third-party packages.
    Randomness comes from ``os.urandom`` rather than ``random`` so that IDs are
    not predictable from one another.

    Args:
        now_ms: Millisecond timestamp to encode. Defaults to the wall clock.
            Callers on a deterministic path pass an explicit value so that ID
            generation does not reintroduce hidden time (invariant V13).
    """
    timestamp = int(time.time() * 1000) if now_ms is None else now_ms
    randomness = int.from_bytes(os.urandom(10), "big")
    value = (timestamp << 80) | randomness
    out = [""] * _ULID_LEN
    for i in range(_ULID_LEN - 1, -1, -1):
        out[i] = _CROCKFORD[value & 0x1F]
        value >>= 5
    return "".join(out)


def ulid_timestamp_ms(ulid: str) -> int:
    """Recover the millisecond timestamp encoded in a ULID."""
    if len(ulid) != _ULID_LEN:
        raise ValueError(f"malformed ULID: expected {_ULID_LEN} chars, got {len(ulid)}")
    value = 0
    for char in ulid:
        index = _CROCKFORD.find(char.upper())
        if index < 0:
            raise ValueError(f"malformed ULID: illegal character {char!r}")
        value = (value << 5) | index
    return value >> 80


# --- FrameRef ------------------------------------------------------------ #


@dataclass(frozen=True, slots=True, order=True)
class FrameRef:
    """Globally unique, totally ordered reference to one decoded instant.

    Ordering is defined per camera. Comparing frames across cameras by ``FrameRef``
    is meaningless — cross-camera ordering uses ``t_capture`` with its declared
    uncertainty (02_VOM §5.2 rule 3).
    """

    camera_id: CameraId
    stream_epoch: StreamEpoch
    frame_seq: FrameSeq

    def __post_init__(self) -> None:
        if self.stream_epoch < 0:
            raise ValueError(f"stream_epoch must be non-negative, got {self.stream_epoch}")
        if self.frame_seq < 0:
            raise ValueError(f"frame_seq must be non-negative, got {self.frame_seq}")

    def follows_in_same_epoch(self, other: FrameRef) -> bool:
        """True when this frame is strictly later than ``other`` in the same epoch.

        The tracker and every order-dependent stage asserts on this rather than
        degrading quietly when frames arrive out of order (06_PORTS T1).
        """
        return (
            self.camera_id == other.camera_id
            and self.stream_epoch == other.stream_epoch
            and self.frame_seq > other.frame_seq
        )

    def __str__(self) -> str:
        return f"{self.camera_id}/e{self.stream_epoch}/f{self.frame_seq}"


# --- TrackId ------------------------------------------------------------- #


@dataclass(frozen=True, slots=True, order=True)
class TrackId:
    """``(camera_id, tracker_epoch, local_id)`` — 02_VOM section 4.1.

    **Composite on purpose.** A bare integer track id is the single most common
    route by which a camera-local, fragile, seconds-lived handle gets mistaken
    for durable identity: it compares equal across cameras, survives a tracker
    reset in appearance only, and reads like a primary key.

    Carrying the camera and the epoch inside the identifier makes every one of
    those mistakes a type-level impossibility rather than a convention. A track
    id from another camera cannot collide, and one from a previous epoch cannot
    compare equal to a live track (invariant V10, port obligation T3).

    A ``TrackId`` is **not an identity**. It answers "is this the same thing I
    saw a moment ago on this camera", never "who is this". Durable identity is
    ``ObjectId``, minted by the Object Registry, which is not this flow.
    """

    camera_id: CameraId
    tracker_epoch: TrackerEpoch
    local_id: LocalTrackId

    def __post_init__(self) -> None:
        if self.tracker_epoch < 0:
            raise ValueError(f"tracker_epoch must be non-negative, got {self.tracker_epoch}")
        if self.local_id < 0:
            raise ValueError(f"local_id must be non-negative, got {self.local_id}")

    def same_epoch_as(self, other: TrackId) -> bool:
        """Whether two ids are even comparable as continuity claims."""
        return self.camera_id == other.camera_id and self.tracker_epoch == other.tracker_epoch

    def __str__(self) -> str:
        return f"{self.camera_id}/t{self.tracker_epoch}/#{self.local_id}"
