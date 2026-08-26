"""A departed person must stop being a *current* compliance subject.

Measured on the running product before this was fixed:

    tracker      created 78    terminated 68     ← tracks die correctly
    registry     27 objects                      ← partition ages correctly
    Vision State 74 objects, ALL lifecycle=active
                 last seen: median 279 s, max 557 s

Three populations, disagreeing. Compliance reads the third:
`ComplianceDriver._subjects` queries `exposure.api`, which reads Vision State.
So a chef who left nine minutes ago was still a live compliance subject.

**The seam.** The registry ages objects in two places, and only one of them is
observable:

* `ingest()` — per frame — returns a `RegistryUpdate` carrying
  `lifecycle_changes`, which the runtime hands to its sink, which synthesis
  turns into LIFECYCLE observations, which Vision State projects. This works.
* `expire_stale()` — the scheduled sweep, and the *only* thing that ages a
  camera nobody is walking in front of — mutated the partition, published
  `ObjectLifecycleChanged` on the event bus, and returned a list of ids.
  **Nothing subscribes to that event**, and the sweep produced no
  `RegistryUpdate`, so none of its transitions ever reached Vision State.

An object swept ACTIVE → OCCLUDED → DORMANT → DEPARTED → EXPIRED and evicted
from the registry therefore left Vision State holding `active` forever — the
last value it was ever told.

These tests pin the propagation, not the horizons: the horizons were already
correct and are untouched.
"""

from __future__ import annotations

import pytest

from vision_os.core.model.ids import CameraId
from vision_os.core.model.space import Box
from vision_os.core.model.timebase import Duration
from vision_os.core.model.visual_object import LifecycleState

from tests.vision_os.registry.conftest import CAMERA, make_track, make_update

#: Where a second person stands: far enough that no tracker confuses them.
ELSEWHERE = Box(0.75, 0.05, 0.85, 0.45)
HERE = Box(0.30, 0.40, 0.40, 0.80)


def _present(lifecycle: LifecycleState | None) -> bool:
    """What the exposure API's default `StateFilter` calls present."""
    return lifecycle in (LifecycleState.PROVISIONAL, LifecycleState.ACTIVE,
                         LifecycleState.OCCLUDED)


class SweepHarness:
    """Registry runtime with a recording sink, driven on a virtual clock.

    Frame numbers are derived from the clock rather than counted independently.
    `conftest.at(seq)` maps a frame index onto a 5 fps timeline, and the
    registry ages objects by comparing `clock.now()` with `last_confirmed` — so
    a test that advances the clock while handing out its own sequence numbers
    silently ages everybody, including the person standing in full view.
    """

    FRAME_MS = 200

    def __init__(self, runtime, registry, clock):
        self.runtime = runtime
        self.registry = registry
        self.clock = clock
        self.updates: list = []
        runtime._sink = self.updates.append

    @property
    def seq(self) -> int:
        return self.clock.now().ns // (self.FRAME_MS * 1_000_000)

    async def see(self, boxes, *, camera=CAMERA):
        seq = self.seq
        tracks = [
            make_track(local=index, box=box, seq=seq, camera=camera)
            for index, box in enumerate(boxes)
        ]
        await self.runtime.on_tracked(camera, make_update(tracks, seq=seq, camera=camera))

    def advance(self, seconds: float) -> None:
        self.clock.advance(Duration(int(seconds * 1_000_000_000)))

    def sweep(self):
        """One scheduled maintenance pass, exactly as `_maintain` runs it."""
        self.runtime._expire()

    def registry_lifecycle(self, object_id):
        for partition in self.registry._partitions.values():
            record = partition.record_for(object_id)
            if record is not None:
                return record.lifecycle
        return None  # evicted

    def announced(self, object_id):
        """The last lifecycle state any sink update announced for this object.

        This is exactly what synthesis can build an observation from, and
        therefore exactly what Vision State can ever come to believe.
        """
        seen = None
        for update in self.updates:
            for changed_id, _previous, current in update.lifecycle_changes:
                if changed_id == object_id:
                    seen = current
        return seen

    def object_ids(self):
        ids = []
        for partition in self.registry._partitions.values():
            ids.extend(o.object_id for o in partition.objects())
        return ids


@pytest.fixture
async def harness(registry_runtime, registry, clock):
    await registry_runtime.start()
    try:
        yield SweepHarness(registry_runtime, registry, clock)
    finally:
        await registry_runtime.stop()


async def _confirm_one_person(harness, *, frames: int = 5) -> object:
    for _ in range(frames):
        harness.advance(0.2)
        await harness.see([HERE])
    ids = harness.object_ids()
    assert ids, "nobody was registered"
    return ids[0]


class TestTheSweepIsObservable:
    async def test_a_swept_transition_is_announced_to_the_sink(self, harness):
        """The defect, exactly.

        The person leaves, nothing else happens on the camera, and the
        scheduled sweep is the only thing that can age them. Whatever it does
        must be announced, or Vision State can never learn it.
        """
        object_id = await _confirm_one_person(harness)
        assert _present(harness.registry_lifecycle(object_id)), (
            "the person was not registered as present to begin with"
        )

        # The camera goes quiet — no more track updates at all, which is the
        # case the sweep exists for. One step at a time, so the object is
        # caught mid-decline rather than only at the far end.
        harness.advance(3.0)
        harness.sweep()

        swept = harness.registry_lifecycle(object_id)
        assert swept is not None, (
            "the object was evicted in a single step, so this test cannot show "
            "an intermediate transition being announced"
        )
        assert not _present(swept), (
            "the sweep did not age the object out of presence; this test is "
            "not exercising what it claims to"
        )
        assert harness.announced(object_id) == swept, (
            f"the registry moved the object to {swept} but never announced it — "
            f"Vision State still believes {harness.announced(object_id)}, and "
            f"that is what compliance reads"
        )

    async def test_the_object_stops_being_a_present_subject(self, harness):
        """The product consequence: a departed person is not a current subject."""
        object_id = await _confirm_one_person(harness)

        # Past the dormant horizon, with the camera quiet throughout.
        for _ in range(6):
            harness.advance(4.0)
            harness.sweep()

        announced = harness.announced(object_id)
        assert announced is not None, "no lifecycle change was ever announced"
        assert not _present(announced), (
            f"a person who left is still announced as {announced}, which the "
            f"exposure API's default filter counts as present"
        )

    async def test_eviction_is_announced_before_the_object_disappears(self, harness):
        """EXPIRED must be announced, not silently dropped.

        The sweep evicts on EXPIRED. If the announcement were built from the
        partition *after* eviction the object would already be gone, and Vision
        State would keep the last thing it was told — which is the bug.
        """
        object_id = await _confirm_one_person(harness)

        for _ in range(10):
            harness.advance(4.0)
            harness.sweep()

        assert harness.registry_lifecycle(object_id) is None, (
            "the object was never evicted, so this test proves nothing"
        )
        assert harness.announced(object_id) is LifecycleState.EXPIRED, (
            f"the object was evicted from the registry but Vision State was last "
            f"told {harness.announced(object_id)}"
        )

    async def test_a_quiet_camera_announces_nothing_it_did_not_change(self, harness):
        """The sweep must not chatter. Only real transitions are announced."""
        await _confirm_one_person(harness)
        harness.updates.clear()

        harness.sweep()   # no time has passed, so no horizon is due
        assert not any(u.lifecycle_changes for u in harness.updates), (
            "the sweep announced a change it did not make"
        )


class TestPresentPeopleAreUnaffected:
    async def test_a_person_still_being_seen_stays_active(self, harness):
        """The sweep must not age somebody who is standing right there."""
        object_id = await _confirm_one_person(harness)

        for _ in range(6):
            harness.advance(1.0)
            await harness.see([HERE])
            harness.sweep()

        assert _present(harness.registry_lifecycle(object_id)), (
            "a person in full view was aged out of presence"
        )
        announced = harness.announced(object_id)
        assert announced is None or _present(announced), (
            f"a person in full view was announced as {announced}"
        )

    async def test_person_a_leaving_does_not_disturb_person_b(self, harness):
        """A and B are independent. A departs; B is still working."""
        for _ in range(5):
            harness.advance(0.2)
            await harness.see([HERE, ELSEWHERE])

        ids = harness.object_ids()
        assert len(ids) >= 2, f"expected two people, registered {len(ids)}"

        # Only B keeps being seen, and the sweep runs throughout.
        for _ in range(6):
            harness.advance(1.0)
            await harness.see([ELSEWHERE])
            harness.sweep()

        live = [
            object_id for object_id in ids
            if _present(harness.registry_lifecycle(object_id))
        ]
        assert len(live) == 1, (
            f"expected exactly one person still present, found {len(live)}"
        )
