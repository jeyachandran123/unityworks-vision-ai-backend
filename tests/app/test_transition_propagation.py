"""Track transitions must travel intact, not be re-derived from track states.

### The defect this pins

The tracker produces a complete `TrackUpdate` — `new`, `terminated`,
`coasting`, `recovered`, `associations`, `refused`, `unmatched_detections`, all
with real ids. `TrackingEngine.track` collapsed it to four integers, and the
L3→L4 bridge then rebuilt a `TrackUpdate` by inspecting each track's **state**:

    new=ids_in(TrackState.TENTATIVE)
    coasting=ids_in(TrackState.COASTING)

A state is not an event.

* `TENTATIVE` persists for `min_hits_to_confirm` frames, so one track was
  reported "new" on every one of those frames.
* A track created and confirmed in a single frame was never reported new at all.
* `terminated` and `recovered` had no counterpart among the counts, so they took
  their empty defaults on every frame — forever. The registry was told that
  nothing ever ended and nothing ever came back, which is precisely the
  information it needed to rebind a fragmented track to its own object.

The repair carries the tracker's own update through. These tests hold that
property at the seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.vision.bridges import TrackingToRegistryBridge


@dataclass
class _Outcome:
    """A `TrackingOutcome` shape, without importing the whole tracking stack."""

    camera_id: str = "cam-01"
    frame_ref: str = "cam-01/e1/f7"
    tracker_epoch: int = 1
    tracks: tuple = ()
    failed: bool = False
    reason: str = ""
    update: Any = None


class _Registry:
    def __init__(self) -> None:
        self.received: list[Any] = []

    async def on_tracked(self, camera_id, update) -> None:
        self.received.append(update)


class TestTheUpdateTravelsWhole:
    def test_the_tracker_s_own_update_is_forwarded_verbatim(self):
        """Not "a faithful reconstruction" — the same object.

        Anything less means some field is being re-derived, and every field this
        bridge re-derived was re-derived wrongly.
        """
        sentinel = object()
        forwarded = TrackingToRegistryBridge._to_update(_Outcome(update=sentinel))
        assert forwarded is sentinel

    def test_terminations_are_not_lost(self, monkeypatch):
        """The field that mattered most and was always empty.

        Without it the registry cannot know a track died this frame, and the
        re-entry repair in `tests/vision_os/registry/test_reentry_ordering.py`
        has nothing to act on.
        """
        from vision_os.core.model.track import BreakReason

        update = _StubUpdate(
            terminated=(("t-1", BreakReason.ASSOCIATION_FAILURE),),
            recovered=("t-2",),
            new=("t-3",),
        )
        forwarded = TrackingToRegistryBridge._to_update(_Outcome(update=update))
        assert forwarded.terminated == (("t-1", BreakReason.ASSOCIATION_FAILURE),)
        assert forwarded.recovered == ("t-2",)
        assert forwarded.new == ("t-3",)


@dataclass
class _StubUpdate:
    terminated: tuple = ()
    recovered: tuple = ()
    new: tuple = ()
    coasting: tuple = ()


class TestCreationIsAnEventNotAState:
    def test_a_tentative_track_is_not_reported_new_on_every_frame(self):
        """The old reconstruction's central error, stated as a property.

        A track sitting in TENTATIVE for three frames is created **once**. The
        tracker says so; the bridge no longer disagrees.
        """
        from vision_os.core.model.track import TrackState

        created_once = _StubUpdate(new=("t-1",))
        still_tentative = _StubUpdate(new=())

        first = TrackingToRegistryBridge._to_update(_Outcome(update=created_once))
        second = TrackingToRegistryBridge._to_update(_Outcome(update=still_tentative))

        assert first.new == ("t-1",)
        assert second.new == (), (
            "a track that merely remains TENTATIVE has not been created again"
        )
        # The state still exists and still means what it means; it is simply no
        # longer mistaken for a creation event.
        assert TrackState.TENTATIVE.value == "tentative"


class TestDegradationIsPreserved:
    def test_an_outcome_without_an_update_still_produces_one(self):
        """A failed frame carries no update. The bridge must still forward
        something, because a failed tracking frame is exactly when the registry
        should age its objects — dropping it would leave them looking fresher
        than the platform can justify."""
        forwarded = TrackingToRegistryBridge._to_update(
            _Outcome(update=None, failed=True, reason="capacity")
        )
        assert forwarded is not None
        assert forwarded.failed is True
        assert forwarded.reason == "capacity"

    def test_the_frame_reference_survives_the_fallback(self):
        """The rendering `cam-01/e1/f7` must parse back to the same frame, so a
        stored attribute stays traceable to a picture."""
        forwarded = TrackingToRegistryBridge._to_update(_Outcome(update=None))
        assert forwarded.frame_ref.stream_epoch == 1
        assert forwarded.frame_ref.frame_seq == 7


class TestTheEngineCarriesIt:
    def test_tracking_outcome_exposes_the_update(self):
        """The counts remain — consumers read them — but they are no longer the
        only thing that crosses the seam."""
        from vision_os.perception.tracking.engine import TrackingOutcome

        assert "update" in TrackingOutcome.__dataclass_fields__
        for count in ("created", "terminated", "recovered", "coasting"):
            assert count in TrackingOutcome.__dataclass_fields__, (
                f"{count} must stay, so existing readers are unaffected"
            )

    @pytest.mark.asyncio
    async def test_the_registry_receives_the_real_update(self):
        """End of the seam: what the tracker declared is what the registry gets."""
        import asyncio

        registry = _Registry()
        bridge = TrackingToRegistryBridge(registry)
        update = _StubUpdate(new=("t-9",))

        bridge(_Outcome(update=update))
        await asyncio.sleep(0)  # let the scheduled forward run

        assert registry.received == [update]
