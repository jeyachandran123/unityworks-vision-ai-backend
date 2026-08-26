"""The frames that decisions were actually made on, kept just long enough.

### The defect this exists for

A compliance pass runs on a timer. It reads an attribute observed some seconds
ago, decides, and opens an incident — and `ComplianceDriver._capture_evidence`
then photographed the room by taking the camera wall's *current* JPEG. Measured
on camera 13:

    attribute observed at   14:04:40Z   hand_covering = none
    incident opened at      14:05:29Z
    evidence frame stamped  14:05:17Z   ← 37 s after the observation

The verdict was real. The picture was of a kitchen the person had already left.
Its docstring was honest about this — *"not a replay of the exact frame the
verdict was computed from"* — but "the image that supports the alert" is a
product requirement, and a later photograph does not support anything.

### Why the frame was gone

Three places hold it, and all three let go before the incident opens:

* the platform's frame buffer leases pixels for extraction and the crop runtime
  calls `lease.release()` as soon as the crop is cut;
* a `Crop` keeps *its own* pixels, but only of the crop — never the scene;
* `CameraStream` keeps exactly one JPEG per camera, the newest.

### What this keeps, and what it refuses to keep

A small per-camera ring of **analysed** frames, encoded once as JPEG, plus the
object boxes that were cut from each. Only frames the analysis path actually
looked at — a handful per camera per minute, not the 25 fps the wall decodes.

It is **not** a recording. `MAX_FRAMES_PER_CAMERA` and `MAX_BYTES_PER_CAMERA`
are both hard, the oldest entry is dropped on overflow, and nothing here writes
to disk: durable storage stays the job of `EvidenceStore`, which already has
tenancy, permissions, audit and retention. This is the short window between a
frame being analysed and an incident being opened, and no longer.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from loguru import logger

#: Frames retained per camera. At the measured live analysis rate (~1.7 s per
#: frame, and slower with a VLM in the path) this is a couple of minutes of
#: decisions — comfortably longer than the gap between an observation and the
#: compliance pass that acts on it, and short enough that four cameras cost a
#: bounded and predictable amount of memory.
MAX_FRAMES_PER_CAMERA = 64

#: Second, independent ceiling. Frame count alone is not a memory bound: a
#: 4K camera's JPEG is an order of magnitude larger than this DVR's 960x576.
#: Whichever limit is reached first evicts.
MAX_BYTES_PER_CAMERA = 32 * 1024 * 1024

#: Quality for the retained JPEG. Higher than the wall's, because this image is
#: evidence somebody may have to defend rather than a thumbnail on a grid.
JPEG_QUALITY = 90

#: How many subjects in one frame may keep their **crop pixels**. Their boxes
#: are always kept — four floats each, and the highlight needs every one of
#: them. The pixels are the expensive part, and a frame with more people in it
#: than this is a frame where a gallery of thumbnails has stopped being useful
#: to an operator anyway.
MAX_CROPS_PER_FRAME = 12


@dataclass(slots=True)
class DecisionSubject:
    """One object that was cut out of this frame and asked about."""

    object_id: str
    #: Normalized box in the source frame — what a UI needs to draw the
    #: highlight, and what proves the crop came from where it says it did.
    #:
    #: This is the object's box **before** the crop strategy's padding. The
    #: crop image below therefore shows slightly more than this rectangle; the
    #: rectangle is what the platform said the object occupied, and it is the
    #: right thing to draw on the full frame.
    box: tuple[float, float, float, float]
    crop_jpeg: bytes | None = None
    crop_id: str = ""
    #: Whether this crop was actually sent to a model, as opposed to cut and
    #: then refused by the quality gate.
    sent_to_model: bool = False
    #: What the platform called this object — `person`. Recorded so evidence
    #: can be confined to objects the finding is actually about rather than to
    #: whatever else happened to be cut from the same frame.
    object_class: str = ""


@dataclass(slots=True)
class DecisionFrame:
    """One analysed frame, and the subjects taken from it."""

    camera_id: str
    frame_ref: str
    captured_at_ns: int
    width: int
    height: int
    jpeg: bytes
    subjects: dict[str, DecisionSubject] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        return len(self.jpeg) + sum(
            len(s.crop_jpeg or b"") for s in self.subjects.values()
        )


class DecisionFrameStore:
    """Bounded, in-memory, per-camera. Thread-safe.

    Written from the analysis worker thread and read from the API event loop
    when an incident opens, so every mutation takes the lock. The critical
    sections are dictionary operations on a bounded map — no I/O, no encoding
    — so this cannot become the event-loop starvation of Phase 10C.
    """

    __slots__ = (
        "_cameras",
        "_lock",
        "crops_dropped",
        "crops_retained",
        "evictions",
        "hits",
        "misses",
        "stores",
    )

    def __init__(self) -> None:
        self._cameras: dict[str, OrderedDict[str, DecisionFrame]] = {}
        self._lock = threading.Lock()
        self.stores = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.crops_retained = 0
        self.crops_dropped = 0

    # --- writing ----------------------------------------------------------- #

    def remember(
        self,
        *,
        camera_id: str,
        frame_ref: str,
        captured_at_ns: int,
        width: int,
        height: int,
        jpeg: bytes,
    ) -> None:
        """Keep one analysed frame. Cheap and never raises."""
        if not jpeg:
            return
        frame = DecisionFrame(
            camera_id=camera_id,
            frame_ref=frame_ref,
            captured_at_ns=captured_at_ns,
            width=width,
            height=height,
            jpeg=jpeg,
        )
        with self._lock:
            frames = self._cameras.setdefault(camera_id, OrderedDict())
            frames[frame_ref] = frame
            frames.move_to_end(frame_ref)
            self.stores += 1
            self._evict(frames)

    def attach_subject(
        self,
        *,
        camera_id: str,
        frame_ref: str,
        object_id: str,
        box: tuple[float, float, float, float],
        crop_jpeg: bytes | None = None,
        crop_id: str = "",
        sent_to_model: bool = False,
        object_class: str = "",
    ) -> bool:
        """Record that this object was cut from this frame.

        Returns False when the frame is no longer retained, which is a fact the
        caller should count rather than paper over: it means the analysis path
        fell far enough behind that the frame aged out of the window.

        Beyond `MAX_CROPS_PER_FRAME` the **box is still recorded** and the crop
        pixels are dropped. Losing the box would lose the highlight, which is
        the one thing an operator cannot reconstruct by looking harder.
        """
        with self._lock:
            frames = self._cameras.get(camera_id)
            frame = frames.get(frame_ref) if frames else None
            if frame is None:
                return False
            if crop_jpeg and object_id not in frame.subjects:
                with_crops = sum(1 for s in frame.subjects.values() if s.crop_jpeg)
                if with_crops >= MAX_CROPS_PER_FRAME:
                    crop_jpeg = None
                    self.crops_dropped += 1
            self.crops_retained += int(bool(crop_jpeg))
            frame.subjects[object_id] = DecisionSubject(
                object_id=object_id,
                box=box,
                crop_jpeg=crop_jpeg,
                crop_id=crop_id,
                sent_to_model=sent_to_model,
                object_class=object_class,
            )
            self._evict(frames)
            return True

    # --- reading ----------------------------------------------------------- #

    def get(self, camera_id: str, frame_ref: str) -> DecisionFrame | None:
        with self._lock:
            frames = self._cameras.get(camera_id)
            frame = frames.get(frame_ref) if frames else None
        if frame is None:
            self.misses += 1
        else:
            self.hits += 1
        return frame

    def latest_for_object(self, camera_id: str, object_id: str) -> DecisionFrame | None:
        """The most recent analysed frame this object was cut from.

        The compliance finding names an object and an observation time, not a
        frame. Walking newest-first and stopping at the first frame that
        actually contains this object is what ties the incident back to a frame
        the object was really in — rather than to whatever the camera is
        looking at now.
        """
        with self._lock:
            frames = self._cameras.get(camera_id)
            if not frames:
                self.misses += 1
                return None
            for frame in reversed(frames.values()):
                if object_id in frame.subjects:
                    self.hits += 1
                    return frame
        self.misses += 1
        return None

    def nearest_before(
        self, camera_id: str, object_id: str, at_ns: int, *, tolerance_ns: int
    ) -> DecisionFrame | None:
        """The newest frame containing this object at or before ``at_ns``.

        Used when a finding carries the instant its attribute was observed. A
        frame *after* the observation is never returned: that is exactly the
        "later room state" this module exists to stop, and returning one would
        reintroduce the defect through a different door.
        """
        best: DecisionFrame | None = None
        with self._lock:
            frames = self._cameras.get(camera_id)
            if not frames:
                self.misses += 1
                return None
            for frame in reversed(frames.values()):
                if object_id not in frame.subjects:
                    continue
                if frame.captured_at_ns > at_ns:
                    continue
                if at_ns - frame.captured_at_ns > tolerance_ns:
                    break  # older still, and frames only get older from here
                best = frame
                break
        if best is None:
            self.misses += 1
        else:
            self.hits += 1
        return best

    # --- bounds ------------------------------------------------------------ #

    def _evict(self, frames: OrderedDict[str, DecisionFrame]) -> None:
        """Caller holds the lock. Both ceilings, oldest first."""
        while len(frames) > MAX_FRAMES_PER_CAMERA:
            frames.popitem(last=False)
            self.evictions += 1
        total = sum(frame.size_bytes for frame in frames.values())
        while total > MAX_BYTES_PER_CAMERA and len(frames) > 1:
            _, dropped = frames.popitem(last=False)
            total -= dropped.size_bytes
            self.evictions += 1

    def stats(self) -> dict[str, int]:
        with self._lock:
            per_camera = {
                camera_id: len(frames) for camera_id, frames in self._cameras.items()
            }
            retained_bytes = sum(
                frame.size_bytes
                for frames in self._cameras.values()
                for frame in frames.values()
            )
        return {
            "cameras": len(per_camera),
            "frames_retained": sum(per_camera.values()),
            "retained_bytes": retained_bytes,
            "stores": self.stores,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "crops_retained": self.crops_retained,
            "crops_dropped": self.crops_dropped,
        }

    def clear(self) -> None:
        with self._lock:
            self._cameras.clear()


def encode_jpeg(payload: bytes, width: int, height: int) -> bytes:
    """BGR24 bytes to JPEG. Returns b"" rather than raising.

    Encoding happens on the analysis worker thread, never on the API loop.
    """
    if not payload or width <= 0 or height <= 0:
        return b""
    try:
        import io

        import numpy as np
        from PIL import Image

        expected = width * height * 3
        if len(payload) < expected:
            return b""
        array = np.frombuffer(payload[:expected], dtype=np.uint8).reshape(
            height, width, 3
        )
        buffer = io.BytesIO()
        Image.fromarray(array[:, :, ::-1]).save(
            buffer, format="JPEG", quality=JPEG_QUALITY
        )
        return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001 - evidence is never worth a frame
        logger.debug("decision frame encode failed: {}: {}", type(exc).__name__, exc)
        return b""


#: One per process, like the analysis worker beside it.
DECISION_FRAMES = DecisionFrameStore()

__all__ = [
    "DECISION_FRAMES",
    "DecisionFrame",
    "DecisionFrameStore",
    "DecisionSubject",
    "encode_jpeg",
]
