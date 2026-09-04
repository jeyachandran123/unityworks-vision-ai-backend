"""Re-entry binding must be reachable in the frame where fragmentation happens.

### The production failure this pins

    cam-12   767 incidents   767 distinct object_ids
    cam-13   703             703
    cam-11   392             392
    cam-14   128             128

`IncidentService.open` returned `created=False` **zero times in 1,990
incidents**. Deduplication was never failing; it was never being offered the same
identity twice, because the registry minted a fresh `ObjectId` every time a
track fragmented.

The cause was ordering, not thresholds. A fragmenting track dies and its
replacement is born in the *same* frame, and the absorb loop ran before the
ageing loop — so when `bind_reentry` was asked whether the newborn track might
be a known object, the predecessor was still `ACTIVE` and still bound. Those are
precisely the two conditions `bind_reentry` rejects, so it returned
`no_candidates` every time and `_mint` did the rest.

These tests hold both halves of the repair: fragmentation must rebind **and**
strangers must still refuse to merge. A registry that answered "same person" to
everything would pass the first half and be far worse than the bug.
"""

from __future__ import annotations

from vision_os.core.model.space import Box
from vision_os.core.model.track import BreakReason, TrackState
from vision_os.core.model.visual_object import LifecycleState

from tests.vision_os.registry.conftest import (
    CAMERA,

    make_track,
    make_update,
    track_id,
)


def _fragmenting(update, *, ended):
    """The same `TrackUpdate`, plus the terminations the tracker declared.

    The tracker has always known these ids. Until the transition contract was
    repaired they were collapsed to an integer count on the way to the registry,
    which is why the registry could not act on them.
    """
    from dataclasses import replace

    return replace(update, terminated=tuple((t, BreakReason.ASSOCIATION_FAILURE) for t in ended))


class TestFragmentationRebinds:
    def test_a_fragmented_track_rebinds_to_its_own_object(self, registry):
        """One person, one logical object, across a track id change.

        The heart of the repair. Frames 0-3 establish a confirmed object under
        track 1. On frame 4 the tracker gives up on track 1 and starts track 2
        on the same person in the same place — the exact shape of the
        association failure that fragments a walking subject.
        """
        box = Box(0.30, 0.40, 0.42, 0.80)
        for seq in range(4):
            registry.ingest(CAMERA, make_update([make_track(local=1, box=box, seq=seq)], seq=seq))

        first = registry.active(CAMERA)
        assert len(first) == 1
        original = first[0].object_id

        # Track 1 terminates and track 2 is born, in one frame, same position.
        fragment = make_update(
            [make_track(local=2, box=box, seq=4, state=TrackState.TENTATIVE, first_seq=4)],
            seq=4,
        )
        registry.ingest(CAMERA, _fragmenting(fragment, ended=[track_id(1)]))

        after = registry.active(CAMERA)
        assert len(after) == 1, "a fragmented track must not create a second person"
        assert after[0].object_id == original, (
            "the replacement track must rebind to the object it continues, "
            "not mint a new identity"
        )

    def test_the_object_is_released_before_absorption_not_after(self, registry):
        """The ordering itself, observed rather than inferred.

        If the release still happened after absorption the predecessor would be
        `ACTIVE` and bound at the moment re-entry ran, and a second object would
        exist by the end of the frame.
        """
        box = Box(0.30, 0.40, 0.42, 0.80)
        for seq in range(4):
            registry.ingest(CAMERA, make_update([make_track(local=1, box=box, seq=seq)], seq=seq))
        before = {o.object_id for o in registry.objects(CAMERA)}

        fragment = make_update(
            [make_track(local=2, box=box, seq=4, state=TrackState.TENTATIVE, first_seq=4)],
            seq=4,
        )
        registry.ingest(CAMERA, _fragmenting(fragment, ended=[track_id(1)]))

        assert {o.object_id for o in registry.objects(CAMERA)} == before, (
            "no new ObjectId may be minted when the predecessor is rebindable"
        )

    def test_rebinding_keeps_the_object_present_rather_than_resurrecting_it(self, registry):
        """A rebind is continuity, not a resurrection: the object stays present
        and keeps accumulating observations under one identity."""
        box = Box(0.30, 0.40, 0.42, 0.80)
        for seq in range(4):
            registry.ingest(CAMERA, make_update([make_track(local=1, box=box, seq=seq)], seq=seq))
        observations_before = registry.active(CAMERA)[0].observation_count

        fragment = make_update(
            [make_track(local=2, box=box, seq=4, state=TrackState.TENTATIVE, first_seq=4)],
            seq=4,
        )
        registry.ingest(CAMERA, _fragmenting(fragment, ended=[track_id(1)]))

        obj = registry.active(CAMERA)[0]
        assert obj.lifecycle.is_present
        assert obj.observation_count > observations_before


class TestFalseMergesAreStillRefused:
    """The other half. Making re-entry *reachable* must not make it *credulous*.

    Every guard that decides re-entry is untouched by this repair —
    `max_reentry_distance`, `max_reentry_gap`, `class_must_match`,
    `min_binding_confidence` and the `ambiguity_margin` refusal. These tests
    exist so a future change cannot quietly trade the duplicate-identity bug for
    a merged-identity one, which is the worse failure in a safety product.
    """

    def test_a_distant_newcomer_is_a_different_person(self, registry):
        """Someone appearing across the room when a track ends is not the same
        person. Distance is the guard, and it still holds."""
        near = Box(0.05, 0.40, 0.17, 0.80)
        for seq in range(4):
            registry.ingest(CAMERA, make_update([make_track(local=1, box=near, seq=seq)], seq=seq))
        original = registry.active(CAMERA)[0].object_id

        far = Box(0.80, 0.40, 0.92, 0.80)
        fragment = make_update(
            [make_track(local=2, box=far, seq=4, state=TrackState.TENTATIVE, first_seq=4)],
            seq=4,
        )
        registry.ingest(CAMERA, _fragmenting(fragment, ended=[track_id(1)]))

        ids = {o.object_id for o in registry.objects(CAMERA)}
        assert original in ids
        assert len(ids) == 2, "a distant track must not inherit another person's identity"

    def test_two_people_keep_two_identities_through_normal_tracking(self, registry):
        """The commonest scene in a kitchen. Nothing here should merge."""
        left = Box(0.10, 0.40, 0.22, 0.80)
        right = Box(0.70, 0.40, 0.82, 0.80)
        for seq in range(5):
            registry.ingest(
                CAMERA,
                make_update(
                    [
                        make_track(local=1, box=left, seq=seq),
                        make_track(local=2, box=right, seq=seq),
                    ],
                    seq=seq,
                ),
            )

        assert len({o.object_id for o in registry.active(CAMERA)}) == 2

    def test_a_different_class_never_inherits_an_identity(self, registry):
        """`class_must_match`. A person does not become a knife."""
        box = Box(0.30, 0.40, 0.42, 0.80)
        for seq in range(4):
            registry.ingest(CAMERA, make_update([make_track(local=1, box=box, seq=seq)], seq=seq))
        original = registry.active(CAMERA)[0].object_id

        fragment = make_update(
            [
                make_track(
                    local=2, box=box, seq=4, class_id="knife",
                    state=TrackState.TENTATIVE, first_seq=4,
                )
            ],
            seq=4,
        )
        registry.ingest(CAMERA, _fragmenting(fragment, ended=[track_id(1)]))

        ids = {o.object_id for o in registry.objects(CAMERA)}
        assert len(ids) == 2
        assert original in ids


class TestOrderingIsDeterministic:
    def test_the_same_frames_produce_the_same_identities_twice(
        self,
        registry,
        clock,
        bus,
        metrics,
        registry_config,
        lifecycle_policy,
        binding_policy,
        attribute_registry,
        registry_provenance,
    ):
        """The *decision* must not depend on dict ordering or on which record
        the engine happened to visit first (invariant V13).

        Asserted on the shape of the outcome, not on the literal ids: `ObjectId`
        is a monotonic ULID, so two registries in one process necessarily mint
        different strings for the same decision. Comparing the ids would test
        the id generator; comparing how many objects exist, and whether the
        fragment rebound, tests the ordering this file is about.
        """
        from vision_os.perception.registry.engine import ObjectRegistry

        from tests.vision_os.registry.conftest import SITE, TENANT

        second = ObjectRegistry(
            clock=clock,
            bus=bus,
            metrics=metrics,
            config=registry_config,
            tenant_id=TENANT,
            site_id=SITE,
            provenance=registry_provenance,
            lifecycle=lifecycle_policy,
            binding=binding_policy,
            attributes=attribute_registry,
        )
        box = Box(0.30, 0.40, 0.42, 0.80)

        def run(reg):
            for seq in range(4):
                reg.ingest(CAMERA, make_update([make_track(local=1, box=box, seq=seq)], seq=seq))
            fragment = make_update(
                [make_track(local=2, box=box, seq=4, state=TrackState.TENTATIVE, first_seq=4)],
                seq=4,
            )
            before = {o.object_id for o in reg.objects(CAMERA)}
            reg.ingest(CAMERA, _fragmenting(fragment, ended=[track_id(1)]))
            after = {o.object_id for o in reg.objects(CAMERA)}
            return (len(after), after == before)

        first_run = run(registry)
        assert first_run == run(second)
        # …and the decision itself is the one this file exists to produce.
        assert first_run == (1, True), "one object, rebound rather than re-minted"

    def test_a_release_with_no_terminations_changes_nothing(self, registry):
        """The release step must be inert on an ordinary frame — it acts only on
        transitions the tracker actually declared."""
        box = Box(0.30, 0.40, 0.42, 0.80)
        for seq in range(4):
            registry.ingest(CAMERA, make_update([make_track(local=1, box=box, seq=seq)], seq=seq))

        obj = registry.active(CAMERA)[0]
        assert obj.lifecycle is LifecycleState.ACTIVE
        assert len(registry.objects(CAMERA)) == 1
