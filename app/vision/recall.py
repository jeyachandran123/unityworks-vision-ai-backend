"""A person's last clear answer survives a following "could not see".

The platform keeps one value per attribute and the latest look wins. v1 often
sees a bare hand in one crop and no hand at all in the next half-second, so the
`none` it saw was replaced by `not_visible` before any compliance pass read it:
on cam-12 (2026-10-22) a chef was seen without gloves at 04:40:54 and the alert
opened a minute later without gloves in it.

`not_visible` means "could not see now", not "now wearing". So when the latest
value is `not_visible`, the most recent *decided* value for the same object and
attribute is used instead, provided it is no older than `window_ns` — the PPE
rule's own `max_staleness_ms` (90 s), the age past which the rule would refuse
it anyway. A later decided value always replaces an earlier one, so a person
seen putting gloves on is judged on that.

Nothing is invented: every value used was observed, with its own observation
time and evidence reference, and the rule's staleness check still applies.
"""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import Any

#: The attribute value that means the model looked and could not see. Every
#: PPE attribute declares it, and the rules treat it as unknown, never absent.
UNSEEN = "not_visible"

#: The PPE rule's `evidence.max_staleness_ms`.
DEFAULT_WINDOW_NS = 90_000_000_000


class DecidedRecall:
    """Remembers each object's last decided value per attribute, briefly."""

    __slots__ = ("_seen", "_window_ns")

    def __init__(self, *, window_ns: int = DEFAULT_WINDOW_NS) -> None:
        self._window_ns = window_ns
        # (organization, object, attribute) -> AttributeView
        self._seen: dict[tuple[str, str, str], Any] = {}

    def apply(
        self, snapshot: Any, *, organization_id: str, now_ns: int, identity: Any = None
    ) -> Any:
        """The snapshot with `not_visible` replaced by a recent decided value.

        `identity` maps an object id to the person it belongs to (see
        `person_link`), so a clear answer survives the person's id changing.
        """
        if snapshot is None or not getattr(snapshot, "objects", None):
            return snapshot
        self._forget_before(now_ns - self._window_ns)
        who = identity or (lambda object_id: object_id)
        objects = tuple(self._object(obj, organization_id, who) for obj in snapshot.objects)
        return replace(snapshot, objects=objects) if is_dataclass(snapshot) else snapshot

    def _object(self, obj: Any, organization_id: str, who: Any) -> Any:
        attributes = dict(getattr(obj, "attributes", None) or {})
        changed = False
        for key, view in attributes.items():
            slot = (organization_id, who(str(obj.object_id)), str(key))
            if str(view.value) != UNSEEN:
                previous = self._seen.get(slot)
                if previous is None or view.observed_at.ns >= previous.observed_at.ns:
                    self._seen[slot] = view
                continue
            remembered = self._seen.get(slot)
            if (
                remembered is not None
                and remembered.observed_at.ns >= view.observed_at.ns - self._window_ns
            ):
                attributes[key] = remembered
                changed = True
        return replace(obj, attributes=attributes) if changed and is_dataclass(obj) else obj

    def _forget_before(self, cutoff_ns: int) -> None:
        stale = [slot for slot, view in self._seen.items() if view.observed_at.ns < cutoff_ns]
        for slot in stale:
            del self._seen[slot]


__all__ = ["DEFAULT_WINDOW_NS", "UNSEEN", "DecidedRecall"]
