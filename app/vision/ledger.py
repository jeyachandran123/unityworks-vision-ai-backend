"""The frame ledger — the Frame-by-Frame backing store.

Every frame the source emitted, in order, as a **descriptor**. Never pixels: a
ledger that retained payloads would hold the whole video in memory a second time,
and the frames themselves are already released back to the platform's buffer the
moment processing finishes.

### Why a frame that produced nothing still gets an entry

The frame view could be assembled purely by grouping observations — the old demo
did exactly that, and it works right up to the interesting case. A frame that
produced *no* observation is invisible to that approach, and it is the single most
diagnostic frame in a run: the detector saw nothing, or the crop was refused, or
the model was never called. "Nothing happened here" and "this frame does not
exist" must not look the same, which is why the ledger records emission rather
than inferring it from output.

### What an entry may hold

Counts and references. `observation_count`, `detection_count`, `evidence_refs` —
never a box list, never a payload, never an attribute value. Those live in the
platform's own state and are read from it when a developer opens one frame. The
ledger's job is to say **which frames exist and which are worth opening**, and it
stays cheap enough to hold a whole replay because that is all it does.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

#: Entries retained per camera. A 12 fps replay of a five-minute clip is ~3,600
#: frames, so the default holds a whole run. Beyond it the oldest are discarded:
#: for a debugging tool the recent past is what matters.
DEFAULT_LEDGER_CAPACITY = 5_000


@dataclass(frozen=True, slots=True)
class FrameEntry:
    """One emitted frame, described.

    `captured_at_ns` is the camera's clock and `received_at_ns` is this process's.
    Both are kept because their difference is queue delay, and a finding is always
    about when the camera *saw* something — never about when a server got around
    to it.
    """

    camera_id: str
    sequence: int
    epoch: int
    frame_ref: str
    captured_at_ns: int
    received_at_ns: int
    width: int
    height: int
    source_kind: str
    #: Set once the frame has been through the handler.
    processed: bool = False
    #: Whether anything actually reported what this frame produced. Separate from
    #: `processed`, and the separation matters: a frame that ran through a
    #: pipeline nobody instrumented would otherwise report zero detections and
    #: zero observations, which reads exactly like a frame that genuinely
    #: produced nothing. "Not measured" is not "measured as none".
    counts_reported: bool = False
    processing_ms: float = 0.0
    detection_count: int = 0
    observation_count: int = 0
    object_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    error: str = ""

    def to_wire(self) -> dict[str, Any]:
        return {
            "frame_ref": self.frame_ref,
            "camera_id": self.camera_id,
            "sequence": self.sequence,
            "epoch": self.epoch,
            "captured_at_ns": self.captured_at_ns,
            "received_at_ns": self.received_at_ns,
            # Stated rather than left for the client to subtract, so every
            # consumer reads the same number.
            "queue_delay_ms": max(0.0, (self.received_at_ns - self.captured_at_ns) / 1_000_000),
            "width": self.width,
            "height": self.height,
            "source_kind": self.source_kind,
            "processed": self.processed,
            "processing_ms": self.processing_ms,
            "detection_count": self.detection_count,
            "observation_count": self.observation_count,
            "object_ids": list(self.object_ids),
            "evidence_refs": list(self.evidence_refs),
            "error": self.error,
            "counts_reported": self.counts_reported,
            # The flag that makes the timeline scannable: a frame that reached
            # the pipeline and produced nothing is a real and interesting state,
            # not a gap. `None` when nothing reported counts — the UI renders
            # that as unknown rather than as empty.
            "empty": (
                (self.observation_count == 0) if (self.processed and self.counts_reported) else None
            ),
        }


@dataclass(slots=True)
class _CameraLedger:
    entries: deque[FrameEntry] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_LEDGER_CAPACITY)
    )
    by_ref: dict[str, FrameEntry] = field(default_factory=dict)


class FrameLedger:
    """Frame descriptors per camera, bounded, thread-safe.

    Written from the session's consumer task and read from HTTP handlers, so
    every mutation takes the lock. The lock is never held across a platform call.
    """

    __slots__ = ("_capacity", "_cameras", "_lock")

    def __init__(self, capacity: int = DEFAULT_LEDGER_CAPACITY) -> None:
        self._capacity = capacity
        self._lock = threading.Lock()
        self._cameras: dict[str, _CameraLedger] = {}

    # -- writing --------------------------------------------------------------

    def record(
        self,
        *,
        camera_id: str,
        sequence: int,
        epoch: int,
        captured_at_ns: int,
        received_at_ns: int,
        width: int = 0,
        height: int = 0,
        source_kind: str = "replay",
    ) -> FrameEntry:
        """Note that a frame was emitted. Called before processing.

        Recorded on emission rather than on completion so that a frame which
        crashed the pipeline still appears — with `processed=False`, which is
        exactly the entry an engineer is looking for.
        """
        entry = FrameEntry(
            camera_id=camera_id,
            sequence=sequence,
            epoch=epoch,
            frame_ref=frame_ref_for(camera_id, epoch, sequence),
            captured_at_ns=captured_at_ns,
            received_at_ns=received_at_ns,
            width=width,
            height=height,
            source_kind=source_kind,
        )
        with self._lock:
            ledger = self._cameras.setdefault(camera_id, _CameraLedger())
            if len(ledger.entries) == ledger.entries.maxlen:
                # The deque evicts silently; the index must follow or it leaks.
                evicted = ledger.entries[0]
                ledger.by_ref.pop(evicted.frame_ref, None)
            ledger.entries.append(entry)
            ledger.by_ref[entry.frame_ref] = entry
        return entry

    def annotate(
        self,
        frame_ref: str,
        *,
        processing_ms: float | None = None,
        detection_count: int | None = None,
        observation_count: int | None = None,
        object_ids: tuple[str, ...] | None = None,
        evidence_refs: tuple[str, ...] | None = None,
        error: str | None = None,
    ) -> FrameEntry | None:
        """Record what the pipeline produced for a frame already noted.

        Returns `None` when the entry has aged out, which is not an error: a long
        soak legitimately outruns the ring, and the alternative — growing without
        bound to keep annotations for frames nobody will look at — is worse.
        """
        with self._lock:
            for ledger in self._cameras.values():
                existing = ledger.by_ref.get(frame_ref)
                if existing is None:
                    continue
                # Counts arriving from anywhere means somebody measured this
                # frame; only then may `empty` mean anything.
                reported = existing.counts_reported or (
                    detection_count is not None or observation_count is not None
                )
                updated = FrameEntry(
                    camera_id=existing.camera_id,
                    sequence=existing.sequence,
                    epoch=existing.epoch,
                    frame_ref=existing.frame_ref,
                    captured_at_ns=existing.captured_at_ns,
                    received_at_ns=existing.received_at_ns,
                    width=existing.width,
                    height=existing.height,
                    source_kind=existing.source_kind,
                    processed=True,
                    counts_reported=reported,
                    processing_ms=_pick(processing_ms, existing.processing_ms),
                    detection_count=_pick(detection_count, existing.detection_count),
                    observation_count=_pick(observation_count, existing.observation_count),
                    object_ids=_pick(object_ids, existing.object_ids),
                    evidence_refs=_pick(evidence_refs, existing.evidence_refs),
                    error=_pick(error, existing.error),
                )
                ledger.by_ref[frame_ref] = updated
                # `deque` has no random-access replace; rebuild in place. Bounded
                # by the ring, so this stays cheap.
                for index, candidate in enumerate(ledger.entries):
                    if candidate.frame_ref == frame_ref:
                        ledger.entries[index] = updated
                        break
                return updated
        return None

    # -- reading --------------------------------------------------------------

    def entries(
        self,
        *,
        camera_ids: tuple[str, ...] | None = None,
        offset: int = 0,
        limit: int = 200,
    ) -> tuple[list[FrameEntry], int]:
        """A page of frames, newest last, with the unpaged total.

        `camera_ids is None` is a tenant-wide grant; an **empty tuple is none**
        and returns nothing — the same three-state discipline the rest of the
        application uses, because an empty list must never read as a wildcard.
        """
        if camera_ids is not None and len(camera_ids) == 0:
            return [], 0

        with self._lock:
            collected: list[FrameEntry] = []
            for camera_id, ledger in self._cameras.items():
                if camera_ids is not None and camera_id not in camera_ids:
                    continue
                collected.extend(ledger.entries)

        collected.sort(key=lambda e: (e.captured_at_ns, e.sequence))
        total = len(collected)
        offset = max(0, offset)
        limit = min(max(limit, 1), 1000)
        return collected[offset : offset + limit], total

    def get(self, frame_ref: str) -> FrameEntry | None:
        with self._lock:
            for ledger in self._cameras.values():
                found = ledger.by_ref.get(frame_ref)
                if found is not None:
                    return found
        return None

    def neighbours(self, frame_ref: str) -> tuple[str | None, str | None]:
        """The previous and next frame on the same camera.

        Frame-by-Frame is a stepping tool, and stepping is the whole interaction.
        Computed here so the client never has to hold the whole timeline to know
        what comes next.
        """
        with self._lock:
            for ledger in self._cameras.values():
                if frame_ref not in ledger.by_ref:
                    continue
                ordered = sorted(ledger.entries, key=lambda e: (e.captured_at_ns, e.sequence))
                for index, entry in enumerate(ordered):
                    if entry.frame_ref != frame_ref:
                        continue
                    previous = ordered[index - 1].frame_ref if index > 0 else None
                    following = ordered[index + 1].frame_ref if index + 1 < len(ordered) else None
                    return previous, following
        return None, None

    def summary(self, *, camera_ids: tuple[str, ...] | None = None) -> dict[str, Any]:
        """Counts a developer needs before opening anything.

        `empty_frames` is the one worth reading first: frames that were processed
        and produced nothing. A run that is all empty frames has a detection or
        demand problem, and this is where that becomes visible in one number.
        """
        collected, total = self.entries(camera_ids=camera_ids, limit=1000)
        with self._lock:
            cameras = sorted(
                name for name in self._cameras if camera_ids is None or name in camera_ids
            )

        processed = [e for e in collected if e.processed]
        measured = [e for e in processed if e.counts_reported]
        return {
            "cameras": cameras,
            "frames_recorded": total,
            "frames_processed": len(processed),
            # Only measured frames are counted either way. An unmeasured frame is
            # neither productive nor empty, and forcing it into one of those
            # buckets would put a number on a thing nobody observed.
            "frames_measured": len(measured),
            "frames_with_observations": sum(1 for e in measured if e.observation_count > 0),
            "empty_frames": sum(1 for e in measured if e.observation_count == 0),
            "frames_with_errors": sum(1 for e in collected if e.error),
            "detections_total": sum(e.detection_count for e in measured),
            "observations_total": sum(e.observation_count for e in measured),
        }

    def clear(self) -> None:
        with self._lock:
            self._cameras.clear()


def _pick(incoming: Any, existing: Any) -> Any:
    """`None` means "not reported"; keep what the entry already had."""
    return existing if incoming is None else incoming


def frame_ref_for(camera_id: str, epoch: int, sequence: int) -> str:
    """The stable frame handle.

    `camera:epoch:sequence`. The epoch is in it because a sequence number resets
    on reconnect — without the epoch, frame 41 before a reconnect and frame 41
    after it would be the same reference to two different moments.
    """
    return f"{camera_id}:{epoch}:{sequence}"


__all__ = [
    "DEFAULT_LEDGER_CAPACITY",
    "FrameEntry",
    "FrameLedger",
    "frame_ref_for",
]
