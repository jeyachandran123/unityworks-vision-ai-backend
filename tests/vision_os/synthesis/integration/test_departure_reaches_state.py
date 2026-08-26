"""A departed object must stop being a current subject, end to end.

`test_sweep_reaches_state.py` pins the registry seam — that the scheduled
horizon pass *announces* what it changed. This pins the consequence across the
whole chain the product actually depends on:

    registry horizon pass
        -> RegistryUpdate.lifecycle_changes
        -> Observation Builder
        -> Vision State
        -> the state query compliance runs

Measured on the running product before the fix: the registry held 27 objects
while the state query returned 74, every one of them `active`, the median last
seen 279 seconds earlier. `ComplianceDriver._subjects` runs that query, so all
74 were live compliance subjects.

Nothing here changes a horizon. The horizons were already right; the sweep was
mute.
"""

from __future__ import annotations

import asyncio

from vision_os.core.model.observation import ObservationType
from vision_os.core.model.timebase import Duration, Instant
from vision_os.core.model.visual_object import LifecycleState

from .test_end_to_end import CAMERA, run_full_pipeline, synthesis_document

#: What the exposure API's default `StateFilter` counts as present, and so what
#: `ComplianceDriver` will evaluate as a subject.
PRESENT = (LifecycleState.PROVISIONAL, LifecycleState.ACTIVE, LifecycleState.OCCLUDED)


def _held(synthesis, camera=CAMERA):
    snapshot = synthesis.state.snapshot()
    for camera_id, partition in snapshot.partitions.items():
        if camera_id == camera:
            return partition.objects
    return {}


def _current(synthesis, camera=CAMERA):
    """Objects a state query would return with the default filter."""
    return {
        object_id: state
        for object_id, state in _held(synthesis, camera).items()
        if state.lifecycle in PRESENT
    }


async def _settle(clock, registry_layer, synthesis, *, seconds: float, steps: int = 12):
    """Let the camera go quiet and the scheduled horizon pass do its work."""
    per_step = Duration(int(seconds / steps * 1_000_000_000))
    for _ in range(steps):
        clock.advance(per_step)
        registry_layer.runtime._expire()  # noqa: SLF001 - the scheduled pass
        for _ in range(20):
            await asyncio.sleep(0)


class TestDepartureReachesState:
    async def test_a_person_who_leaves_stops_being_a_current_subject(self, clock):
        """The whole point. History may remain; presence must not."""
        _, registry_layer, synthesis = await run_full_pipeline(
            clock, synthesis_document()
        )

        before = _current(synthesis)
        assert before, "nobody reached Vision State as present, so nothing is proven"

        await _settle(clock, registry_layer, synthesis, seconds=600.0)

        after = _current(synthesis)
        assert not after, (
            f"{len(after)} object(s) are still current subjects after ten minutes "
            f"of an empty camera: "
            f"{ {str(k): v.lifecycle.value for k, v in after.items()} }"
        )

    async def test_the_history_is_not_destroyed(self, clock):
        """Ageing is not deletion. §3: history may remain queryable.

        The objects stay in state with their observations; what changes is that
        they are no longer *present*, so a state query stops returning them and
        compliance stops treating them as subjects.
        """
        _, registry_layer, synthesis = await run_full_pipeline(
            clock, synthesis_document()
        )
        before = set(_held(synthesis))
        assert before

        await _settle(clock, registry_layer, synthesis, seconds=600.0)

        after = _held(synthesis)
        assert before <= set(after), (
            "objects vanished from state entirely; the fix must age them, not "
            "delete the record"
        )
        assert all(state.observation_count > 0 for state in after.values())

    async def test_the_transition_is_recorded_as_an_observation(self, clock):
        """Nothing is in state that was not first a published fact (07_STATE §1.1).

        The lifecycle change must arrive as a LIFECYCLE observation in the log,
        not by some side channel writing state directly — otherwise the record
        cannot explain why an object stopped being a subject.
        """
        _, registry_layer, synthesis = await run_full_pipeline(
            clock, synthesis_document()
        )
        await _settle(clock, registry_layer, synthesis, seconds=600.0)

        recorded = synthesis.state.observations_in(
            CAMERA, since=Instant(0), until=Instant(2**62)
        )
        assert recorded, "the observation log is empty; nothing is being recorded"

        departures = [
            observation
            for observation in recorded
            if observation.observation_type is ObservationType.LIFECYCLE
            and observation.lifecycle_transition is not None
            and observation.lifecycle_transition.current not in PRESENT
        ]
        assert departures, (
            "objects left the present set with no LIFECYCLE observation behind "
            "it — state changed without a published fact"
        )
