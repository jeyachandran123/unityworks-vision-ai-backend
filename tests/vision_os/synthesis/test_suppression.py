"""Change suppression — §M11's *"main performance feature, and a correctness
feature too"*.

The correctness half is the part worth guarding. A suppressed observation and a
missing one are indistinguishable to a consumer unless the heartbeat fires, so
these tests are as much about what suppression must **never** silence as about
the 10-50x reduction it exists to deliver.
"""

from __future__ import annotations

from vision_os.adapters.synthesis import (
    SUPPRESSION_FACTORIES,
    AlwaysPublish,
    ExactSuppression,
    ThresholdSuppression,
)
from vision_os.core.model.observation import (
    LifecycleTransition,
    ObservabilityReason,
    ObservabilityStatus,
    ObservationType,
)
from vision_os.core.model.timebase import Duration
from vision_os.core.model.visual_object import LifecycleState
from vision_os.synthesis.builder.suppression import (
    SuppressionStateStore,
    subject_key,
)

from .conftest import (
    CAMERA,
    OTHER_CAMERA,
    at,
    context,
    make_builder,
    make_object,
    presence_of,
    spatial,
)


class TestUnchangedFactsAreSuppressed:
    def test_the_second_identical_presence_says_nothing_new(self, builder) -> None:
        obj = make_object()
        assert builder.build_presence(obj, context(seq=3)) is not None
        assert builder.build_presence(obj, context(seq=4)) is None

    def test_a_stationary_object_does_not_republish_its_position(self, builder) -> None:
        """The canonical case §M11 names.

        A parked car observed at 5fps for an hour is 18,000 identical facts, and
        the 17,999 that say nothing are the whole reason this feature exists.
        """
        obj = make_object(position=spatial(0.4, 0.3))
        assert builder.build_spatial(obj, context(seq=1)) is not None
        published = sum(
            builder.build_spatial(obj, context(seq=seq)) is not None
            for seq in range(2, 30)
        )
        assert published == 0

    def test_a_moved_object_publishes_again(self, builder) -> None:
        assert builder.build_spatial(
            make_object(position=spatial(0.1, 0.1)), context(seq=1)
        ) is not None
        assert builder.build_spatial(
            make_object(position=spatial(0.7, 0.6)), context(seq=2)
        ) is not None

    def test_suppression_is_per_subject_not_per_camera(self, builder) -> None:
        """Two objects at the same position are two facts, not one."""
        assert builder.build_presence(make_object(object_id="a"), context()) is not None
        assert builder.build_presence(make_object(object_id="b"), context()) is not None

    def test_suppression_is_per_type(self, builder) -> None:
        """A presence fact does not suppress a spatial one about the same object."""
        obj = make_object()
        assert builder.build_presence(obj, context(seq=3)) is not None
        assert builder.build_spatial(obj, context(seq=3)) is not None


class TestTheHeartbeatMakesSuppressionSafe:
    """V8. Suppression without a heartbeat is silence indistinguishable from death."""

    def test_an_unchanged_fact_republishes_after_the_heartbeat(self) -> None:
        builder = make_builder(heartbeat_ms=1_000)
        obj = make_object()
        assert builder.build_presence(obj, context(seq=0)) is not None
        assert builder.build_presence(obj, context(seq=1)) is None
        # seq 10 is 2s later at 200ms frames — past the 1s heartbeat.
        assert builder.build_presence(obj, context(seq=10)) is not None

    def test_the_heartbeat_reason_says_why_it_published(self) -> None:
        """An operator asking "why did this republish" gets an answer, not a guess."""
        policy = ExactSuppression()
        observation = _presence()
        decision = policy.should_publish(
            observation,
            policy.signature(observation),
            elapsed=Duration.from_millis(60_000),
            heartbeat=Duration.from_millis(30_000),
        )
        assert decision.publish
        assert "heartbeat" in decision.reason


class TestSomeThingsAreNeverSuppressed:
    """The list is short and every entry is a V8 obligation."""

    def test_coverage_is_never_suppressed(self, builder) -> None:
        for seq in range(5):
            assert builder.build_coverage(
                context(seq=seq),
                status=ObservabilityStatus.BLIND,
                reason=ObservabilityReason.STREAM_DISCONNECTED,
                since=at(0),
                effective_rate=0.0,
            ) is not None

    def test_a_lifecycle_transition_is_never_suppressed(self, builder) -> None:
        """A transition is by definition a change.

        Suppressing one would mean the state machine's own history has a hole,
        and a projection replaying it would arrive somewhere different.
        """
        obj = make_object()
        first = builder.build_lifecycle(
            obj,
            LifecycleTransition(LifecycleState.PROVISIONAL, LifecycleState.ACTIVE),
            context(seq=1),
        )
        second = builder.build_lifecycle(
            obj,
            LifecycleTransition(LifecycleState.ACTIVE, LifecycleState.OCCLUDED),
            context(seq=2),
        )
        assert first is not None
        assert second is not None

    def test_identity_assertions_are_never_suppressed(self) -> None:
        from vision_os.adapters.synthesis import ALWAYS_PUBLISH

        assert ObservationType.COVERAGE in ALWAYS_PUBLISH
        assert ObservationType.LIFECYCLE in ALWAYS_PUBLISH
        assert ObservationType.IDENTITY in ALWAYS_PUBLISH

    def test_every_policy_honours_the_mandatory_set(self) -> None:
        """A policy that could suppress coverage would be a V8 hole a deployment
        could open by configuration.
        """
        observation = _coverage()
        for name, factory in SUPPRESSION_FACTORIES.items():
            policy = factory()
            decision = policy.should_publish(
                observation,
                policy.signature(observation),
                elapsed=Duration(0),
                heartbeat=Duration.from_millis(30_000),
            )
            assert decision.publish, f"{name} suppressed a coverage observation"


class TestSuppressionStateIsBounded:
    def test_the_store_evicts_rather_than_growing_without_limit(self) -> None:
        """Unbounded would grow with every object a camera has ever seen.

        §M11: *"brief duplication is harmless, missing data is not"* — so an
        evicted subject simply republishes.
        """
        store = SuppressionStateStore(capacity_per_camera=8)
        partition = store.partition(CAMERA)
        for i in range(64):
            observation = _presence(object_id=f"obj-{i}")
            partition.record(subject_key(observation), f"sig-{i}", observation)
        assert partition.tracked <= 8

    def test_partitions_do_not_share_state(self) -> None:
        """07_STATE §4. One camera's suppression must not silence another's."""
        store = SuppressionStateStore(capacity_per_camera=8)
        one = _presence(camera=CAMERA)
        store.partition(CAMERA).record(subject_key(one), "sig", one)
        assert store.partition(OTHER_CAMERA).last(subject_key(one)) is None

    def test_forgetting_a_camera_releases_its_state(self, builder) -> None:
        presence_of(builder)
        builder.forget_camera(CAMERA)
        assert builder.build_presence(make_object(), context(seq=4)) is not None


class TestThresholdSuppression:
    def test_a_sub_threshold_move_is_suppressed(self) -> None:
        """A bounding box jittering by a pixel is noise, not motion."""
        policy = ThresholdSuppression(position_threshold=0.2)
        first = _presence(position=spatial(0.400, 0.300))
        nudged = _presence(position=spatial(0.405, 0.302))
        decision = policy.should_publish(
            nudged,
            policy.signature(first),
            elapsed=Duration.from_millis(200),
            heartbeat=Duration.from_millis(30_000),
        )
        assert not decision.publish

    def test_a_real_move_publishes(self) -> None:
        policy = ThresholdSuppression(position_threshold=0.05)
        first = _presence(position=spatial(0.1, 0.1))
        moved = _presence(position=spatial(0.8, 0.7))
        decision = policy.should_publish(
            moved,
            policy.signature(first),
            elapsed=Duration.from_millis(200),
            heartbeat=Duration.from_millis(30_000),
        )
        assert decision.publish

    def test_the_policy_owns_its_signature(self) -> None:
        """P18's contract keeps the builder ignorant of what "changed" means.

        The builder stores one opaque string. A builder that understood the
        signature's structure would be holding suppression policy, and swapping
        the policy would then be a code change rather than a configuration one.
        """
        policy = ThresholdSuppression(position_threshold=0.2)
        assert isinstance(policy.signature(_presence()), str)


class TestSuppressionIsDeterministic:
    def test_the_same_inputs_give_the_same_decision(self) -> None:
        """V13. A replay must suppress exactly what the live run suppressed."""
        policy = ExactSuppression()
        observation = _presence()
        signature = policy.signature(observation)
        decisions = [
            policy.should_publish(
                observation,
                signature,
                elapsed=Duration.from_millis(100),
                heartbeat=Duration.from_millis(30_000),
            )
            for _ in range(5)
        ]
        assert all(d == decisions[0] for d in decisions)

    def test_the_signature_is_stable_across_calls(self) -> None:
        policy = ExactSuppression()
        observation = _presence()
        assert policy.signature(observation) == policy.signature(observation)

    def test_every_decision_carries_a_reason(self) -> None:
        """A suppression nobody can explain is a fact that vanished (V4)."""
        policy = ExactSuppression()
        observation = _presence()
        for previous in (None, policy.signature(observation), "stale"):
            decision = policy.should_publish(
                observation,
                previous,
                elapsed=Duration(0),
                heartbeat=Duration.from_millis(30_000),
            )
            assert decision.reason


class TestAlwaysPublishIsHonestAboutItself:
    def test_it_publishes_everything(self) -> None:
        policy = AlwaysPublish()
        observation = _presence()
        assert policy.should_publish(
            observation,
            policy.signature(observation),
            elapsed=Duration(0),
            heartbeat=Duration.from_millis(30_000),
        ).publish

    def test_its_id_says_what_it_does(self) -> None:
        """A deployment reading its own metrics should not have to guess why
        volume is 30x what the sizing assumed.
        """
        assert "always" in AlwaysPublish().policy_id


# --- helpers ------------------------------------------------------------------- #


def _presence(*, object_id: str = "obj-1", camera=CAMERA, position=None):
    """One published presence observation, built with suppression disabled."""
    observation = make_builder(policy=AlwaysPublish()).build_presence(
        make_object(object_id=object_id, camera=camera, position=position),
        context(camera=camera),
    )
    assert observation is not None
    return observation


def _coverage():
    return make_builder(policy=AlwaysPublish()).build_coverage(
        context(),
        status=ObservabilityStatus.DEGRADED,
        reason=ObservabilityReason.SCENE_OBSCURED,
        since=at(0),
        effective_rate=0.5,
    )
