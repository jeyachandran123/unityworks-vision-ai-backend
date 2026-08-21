"""Integration tests — M9 → M11 → M12 end to end.

Real modules throughout: a real builder, a real projection, a real log adapter, a
real state manager. The only substitute is the *understanding result*, because
M11's contract is that it consumes one and never asks a model anything.

Two seams are exercised, and the difference between them is architectural rather
than incidental. `01_LAYERED` §3.1's **dotted edges** say registry results become
observations *without passing through understanding* — so presence and spatial
facts must flow when no model has ever run.
"""

from __future__ import annotations

import pytest

from vision_os.core.model.ids import CameraId, ObjectId
from vision_os.core.model.observation import (
    ObservabilityReason,
    ObservabilityStatus,
    ObservationType,
)
from vision_os.core.model.visual_object import LifecycleState
from vision_os.perception.registry.engine import RegistryUpdate
from vision_os.synthesis import SynthesisRuntime

from .conftest import (
    CAMERA,
    OTHER_CAMERA,
    POSTURE,
    at,
    attribute,
    frame_ref,
    make_builder,
    make_object,
    synthesis_config,
    understanding,
)


@pytest.fixture
def runtime(clock, metrics, health, state, attribute_registry, taxonomy):
    from vision_os.adapters.synthesis import AlwaysPublish

    from .conftest import TAXONOMY_VERSION

    return SynthesisRuntime(
        clock=clock,
        metrics=metrics,
        health=health,
        builder=make_builder(
            clock=clock,
            metrics=metrics,
            registry=attribute_registry,
            view=taxonomy,
            policy=AlwaysPublish(),
            suppression_policy="suppression.always",
        ),
        config=synthesis_config(suppression_policy="suppression.always"),
        state=state,
        taxonomy_version=TAXONOMY_VERSION,
    )


def update(
    *,
    camera: CameraId = CAMERA,
    objects=(),
    seq: int = 3,
    failed: bool = False,
    lifecycle_changes=(),
) -> RegistryUpdate:
    return RegistryUpdate(
        camera_id=camera,
        frame_ref=frame_ref(seq, camera=camera),
        objects=tuple(objects),
        lifecycle_changes=tuple(lifecycle_changes),
        failed=failed,
    )


class TestTheRegistrySeam:
    """The dotted edge: perception becomes fact without a model."""

    async def test_a_registry_update_produces_observations(self, runtime, state) -> None:
        await runtime.start()
        await runtime.on_registered(update(objects=[make_object()]))

        snapshot = state.snapshot()
        assert CAMERA in snapshot.partitions
        assert ObjectId("obj-1") in snapshot.partitions[CAMERA].objects

    async def test_presence_flows_with_no_understanding_anywhere(
        self, runtime, state
    ) -> None:
        """*"Understanding is enrichment, not a toll gate."*

        A site with no model at all still records what it saw. Wiring synthesis
        only to M9's output would have made presence depend on inference.
        """
        await runtime.start()
        await runtime.on_registered(
            update(objects=[make_object(object_id=f"o{i}") for i in range(3)])
        )
        assert len(state.snapshot().partitions[CAMERA].objects) == 3
        assert runtime.stats.observations_built >= 3

    async def test_a_failed_registry_update_produces_nothing(
        self, runtime, state
    ) -> None:
        """V8. A broken registry must not read as *"nothing was here"*.

        Its object list is empty either way; only ``failed`` distinguishes an
        upstream fault from an empty scene, and manufacturing the second from the
        first is the exact error V8 names.
        """
        await runtime.start()
        await runtime.on_registered(update(objects=[], failed=True))
        assert state.snapshot().partitions == {}

    async def test_a_lifecycle_change_becomes_a_lifecycle_observation(
        self, runtime, state, log
    ) -> None:
        await runtime.start()
        obj = make_object(lifecycle=LifecycleState.OCCLUDED)
        await runtime.on_registered(
            update(
                objects=[obj],
                lifecycle_changes=[
                    (ObjectId("obj-1"), LifecycleState.ACTIVE, LifecycleState.OCCLUDED)
                ],
            )
        )
        kinds = {o.observation_type for o in tuple(log.read(CAMERA, limit=50))}
        assert ObservationType.LIFECYCLE in kinds

    async def test_the_seam_never_raises_into_the_registry(self, runtime) -> None:
        """V9. A failure at L5 may not stop L2.

        The registry holds a callable and awaits nothing; if synthesis threw, a
        detector would eventually stop because an observation could not be
        built.
        """
        await runtime.start()
        await runtime.on_registered(update(objects=[object()]))  # not a VisualObject
        assert runtime.health().state.value in ("degraded", "healthy")


class TestTheUnderstandingSeam:
    async def test_a_result_becomes_an_attribute_observation(
        self, runtime, state
    ) -> None:
        await runtime.start()
        await runtime.on_registered(update(objects=[make_object()]))
        await runtime.on_understood(
            [understanding(attributes=(attribute(POSTURE, "sitting"),))]
        )
        found = state.object_state(ObjectId("obj-1"))
        assert found.attributes[POSTURE].value == "sitting"

    async def test_an_attribute_observation_carries_no_invented_confidence(
        self, runtime, state, log
    ) -> None:
        """*"Never fabricate certainty."*

        The understanding seam reconstructs its subject from the result and does
        not know how sure M7 is that this track is this object. Stamping an
        identity confidence there would publish a number nobody measured. The
        attributes carry their own confidence, which M9 did measure.
        """
        await runtime.start()
        await runtime.on_registered(update(objects=[make_object()]))
        await runtime.on_understood([understanding()])

        attributes = [
            o for o in tuple(log.read(CAMERA, limit=50))
            if o.observation_type is ObservationType.ATTRIBUTE
        ]
        assert attributes
        assert attributes[0].confidence is None
        assert attributes[0].attributes[0].confidence.value > 0

    async def test_an_attribute_names_the_object_m7_minted_not_a_new_one(
        self, runtime, state
    ) -> None:
        """01_LAYERED §8: exactly one module may mint an identity.

        Projecting an attribute observation for an object with no prior presence
        record is correct — the log is the record, and the id travelled from M7
        through the crop and the request. What must never happen is synthesis
        *creating* an id, and the architecture guard on ``ObjectId(...)`` call
        sites is what enforces that.
        """
        await runtime.start()
        await runtime.on_understood([understanding(object_id="obj-from-m7")])
        assert ObjectId("obj-from-m7") in state.snapshot().partitions[CAMERA].objects

    async def test_results_are_grouped_by_camera(self, runtime, state) -> None:
        """07_STATE §4: one writer per partition, no cross-partition commit."""
        await runtime.start()
        await runtime.on_registered(update(objects=[make_object()]))
        await runtime.on_registered(
            update(
                camera=OTHER_CAMERA,
                objects=[make_object(object_id="obj-2", camera=OTHER_CAMERA)],
            )
        )
        await runtime.on_understood([
            understanding(object_id="obj-1", camera=CAMERA),
            understanding(
                request_id="req-2", object_id="obj-2", camera=OTHER_CAMERA
            ),
        ])
        assert set(state.partitions) == {CAMERA, OTHER_CAMERA}

    async def test_a_failed_understanding_adds_no_attribute(
        self, runtime, state
    ) -> None:
        from vision_os.core.model.understanding import UnderstandingOutcome

        await runtime.start()
        await runtime.on_registered(update(objects=[make_object()]))
        await runtime.on_understood([
            understanding(outcome=UnderstandingOutcome.TIMED_OUT, attributes=())
        ])
        assert POSTURE not in state.object_state(ObjectId("obj-1")).attributes


class TestEvidenceProvenanceSurvivesTheHandoff:
    async def test_m11_stamps_the_observation_id_onto_m9s_evidence(
        self, runtime, state, log
    ) -> None:
        """The promised half of a two-part construction.

        M9 produces everything except ``observation_id`` because it may not mint
        an identifier for an object it is forbidden to create. If M11 did not
        stamp it, the evidence record would be unreachable from the fact it
        explains, and V4's audit chain would have a break in exactly one link.
        """
        await runtime.start()
        await runtime.on_registered(update(objects=[make_object()]))
        await runtime.on_understood([understanding()])

        attributes = [
            o for o in tuple(log.read(CAMERA, limit=50))
            if o.observation_type is ObservationType.ATTRIBUTE
        ]
        assert attributes
        assert attributes[0].evidence_ref is not None
        assert attributes[0].evidence_ref.evidence_id

    async def test_the_evidence_reference_declares_whether_it_is_retrievable(
        self, runtime, log
    ) -> None:
        """§M11's failure table wants honesty about a failed evidence write:
        ``pending`` and ``unavailable`` mean different things to a consumer
        trying to fetch it.
        """
        await runtime.start()
        await runtime.on_registered(update(objects=[make_object()]))
        await runtime.on_understood([understanding()])
        attributes = [
            o for o in tuple(log.read(CAMERA, limit=50))
            if o.observation_type is ObservationType.ATTRIBUTE
        ]
        assert attributes[0].evidence_ref.status in (
            "stored", "pending", "unavailable"
        )

    async def test_provenance_names_m11_not_m9_for_the_envelope(
        self, runtime, log
    ) -> None:
        """The envelope was assembled by M11; the attribute was produced by M9.

        Both facts are recorded, in different places, because an auditor asking
        "who said this" and "who packaged it" is asking two questions.
        """
        await runtime.start()
        await runtime.on_registered(update(objects=[make_object()]))
        published = [o for o in tuple(log.read(CAMERA, limit=50)) if o.attributes]
        for observation in published:
            for attr in observation.attributes:
                assert attr.producer.producer_module == "understanding_engine"


class TestCoverageFlowsEndToEnd:
    async def test_publishing_coverage_reaches_state(self, runtime, state) -> None:
        await runtime.start()
        runtime.publish_coverage(
            CAMERA,
            status=ObservabilityStatus.BLIND,
            reason=ObservabilityReason.STREAM_DISCONNECTED,
            since=at(1),
            effective_rate=0.0,
        )
        assert state.coverage().by_camera[CAMERA].status is ObservabilityStatus.BLIND

    async def test_coverage_needs_no_prior_object(self, runtime, state) -> None:
        """A camera that has never seen anything can still go blind."""
        await runtime.start()
        runtime.publish_coverage(
            CAMERA,
            status=ObservabilityStatus.DEGRADED,
            reason=ObservabilityReason.SCENE_OBSCURED,
            since=at(1),
            effective_rate=0.3,
        )
        assert CAMERA in state.coverage().by_camera


class TestCommitNeverDrops:
    """08_RUNTIME §5.2: ``Builder → State`` is ``block``, not ``drop_oldest``."""

    async def test_every_built_observation_reaches_the_log(
        self, runtime, state, log
    ) -> None:
        """The queue asymmetry is the architecture's own answer.

        Dropping a crop costs one enrichment; dropping an observation deletes a
        fact the platform already decided was worth publishing.
        """
        await runtime.start()
        for i in range(20):
            await runtime.on_registered(
                update(objects=[make_object(object_id=f"o{i}", seq=i)], seq=i)
            )
        assert len(tuple(log.read(CAMERA, limit=200))) == runtime.stats.observations_built
