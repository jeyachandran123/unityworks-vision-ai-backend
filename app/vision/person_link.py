"""One person, one tag: a new track id that picks up where an old one left off
keeps the old one's identity for alerting.

Measured on cam-13 (2026-10-22, 05:00-05:19 UTC): a person kept one platform
object id for a median 0.4 s, and 65 % of new ids started on the spot another
had left within 3 s - in 181 of 258 such hand-overs the new box overlapped the
old one by more than 70 %. Every new id opened a fresh alert, so one chef
walking past the camera became three alerts with three different tags.

The platform's tracker and registry live in `vision_os/` and are not changed
here. This links ids in the **application**, where alerts are made, using
geometry only - where and when, never what a person looks like:

* An object id seen before keeps the person it was linked to.
* A new object id inherits a person whose last sighting came **before the new
  id was first seen** (1 s slack), no more than `window_ns` earlier (10 s), at
  a box overlapping the new one by at least `min_iou` (0.3). The best overlap
  wins; a person is claimed by one id per pass.
* Otherwise the new object id starts a new person, named after itself.

Timing, not the platform's lifecycle labels, decides "gone": on cam-13 an id is
marked occluded in the same instant it is measured, and a chef standing still
behind a colleague was detected only every ~5 s - 3RQHQY last seen 05:18:06,
Q858TP first seen 05:18:12, X84R8K 05:18:17, all on the same box.

A wrong link would put one person's missing items on another's alert. The
three conditions together - gone before the newcomer arrived, recent, same
place - keep that rare; two people swapping places within ten seconds can
still fool it.
"""

from __future__ import annotations

from typing import Any

DEFAULT_WINDOW_NS = 10_000_000_000
DEFAULT_MIN_IOU = 0.3
#: How long a person is remembered at all. Longer than the link window so the
#: map from old object ids stays valid while their alert is still open.
FORGET_AFTER_NS = 600_000_000_000
#: A new id may be first seen in the same frame the old one was last measured.
SLACK_NS = 1_000_000_000
#: Seen this recently, an id is in view and its person is spoken for. Live
#: analysis runs at ~1.7 frames a second, so 1.5 s spans two or three frames.
IN_VIEW_NS = 1_500_000_000


def _box(obj: Any) -> tuple[float, float, float, float] | None:
    box = getattr(getattr(obj, "spatial", None), "bbox", None)
    if box is None:
        return None
    return (float(box.x1), float(box.y1), float(box.x2), float(box.y2))


def _ns(obj: Any, name: str, default: int) -> int:
    return int(getattr(getattr(obj, name, None), "ns", default))


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class PersonLinker:
    """Maps platform object ids to a stable person id, per organization and camera."""

    __slots__ = ("_min_iou", "_person_of", "_people", "_window_ns")

    def __init__(
        self, *, window_ns: int = DEFAULT_WINDOW_NS, min_iou: float = DEFAULT_MIN_IOU
    ) -> None:
        self._window_ns = window_ns
        self._min_iou = min_iou
        # (organization, object id) -> person id
        self._person_of: dict[tuple[str, str], str] = {}
        # (organization, camera, person id) -> (last box, last seen ns, last object id)
        self._people: dict[tuple[str, str, str], tuple[tuple[float, ...], int, str]] = {}

    def person(self, organization_id: str, object_id: str) -> str:
        """The person an object id belongs to; itself when never linked."""
        return self._person_of.get((organization_id, object_id), object_id)

    def link(self, snapshot: Any, *, organization_id: str, now_ns: int) -> dict[str, str]:
        """Link this pass's objects to people. Returns object id -> person id."""
        self._forget_before(now_ns - FORGET_AFTER_NS)
        objects = list(getattr(snapshot, "objects", None) or ())
        claimed: set[tuple[str, str, str]] = set()
        result: dict[str, str] = {}

        # Known ids first, so a person already accounted for is never handed to
        # a newcomer in the same pass.
        known = [o for o in objects if (organization_id, str(o.object_id)) in self._person_of]
        fresh = [o for o in objects if (organization_id, str(o.object_id)) not in self._person_of]
        for obj in known + fresh:
            object_id, camera = str(obj.object_id), str(obj.camera_id)
            box = _box(obj)
            last_ns = _ns(obj, "last_seen", now_ns)
            first_ns = _ns(obj, "first_seen", last_ns)
            person = self._person_of.get((organization_id, object_id))
            if person is None and box is not None:
                person = self._inherit(organization_id, camera, box, first_ns, claimed)
            person = person or object_id
            self._person_of[(organization_id, object_id)] = person
            slot = (organization_id, camera, person)
            if now_ns - last_ns <= IN_VIEW_NS:
                # Only an id seen just now speaks for its person; an old id
                # lingering in the state must not block its own successor.
                claimed.add(slot)
            previous = self._people.get(slot)
            if box is not None and (previous is None or last_ns >= previous[1]):
                self._people[slot] = (box, last_ns, object_id)
            result[object_id] = person
        return result

    def _inherit(self, organization_id, camera, box, first_ns, claimed) -> str | None:
        """The person this new id continues: gone before it arrived, recently,
        from the same place."""
        best: tuple[float, str] | None = None
        for slot, (last_box, last_ns, _) in self._people.items():
            org, cam, _person = slot
            if org != organization_id or cam != camera or slot in claimed:
                continue
            if not -SLACK_NS <= first_ns - last_ns <= self._window_ns:
                continue
            overlap = _iou(last_box, box)
            if overlap >= self._min_iou and (best is None or overlap > best[0]):
                best = (overlap, slot[2])
        return best[1] if best else None

    def _forget_before(self, cutoff_ns: int) -> None:
        stale = [slot for slot, (_, seen, _) in self._people.items() if seen < cutoff_ns]
        for slot in stale:
            del self._people[slot]
        live = {(org, person) for org, _, person in self._people}
        for key in [k for k, p in self._person_of.items() if (k[0], p) not in live]:
            del self._person_of[key]


__all__ = ["PersonLinker"]
