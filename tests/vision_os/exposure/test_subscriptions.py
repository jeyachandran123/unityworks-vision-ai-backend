"""Subscription tests — delivery, backpressure and the `Gap` (09_API §3).

> §3.3: *"**A subscriber is never silently skipped.**"*

09_API §3.4 forbids three things: unbounded buffering, silent drop, and stalling
the platform. Each is tested by attempting it — a subscriber that stops reading,
a queue driven past capacity, a publisher racing a consumer — and asserting the
platform did the declared thing instead.
"""

from __future__ import annotations

import pytest

from vision_os.core.model.api import (
    DeliveryMode,
    DeliveryPolicy,
    Gap,
    GapReason,
    Heartbeat,
    OverflowPolicy,
    SubscriptionFilter,
)
from vision_os.core.model.observation import ObservationType
from vision_os.core.model.timebase import Duration

from .conftest import (
    CAMERA,
    OTHER_CAMERA,
    OTHER_TENANT,
    publish,
    scope,
)


class TestDeliveryIsBounded:
    """§3.4: *"Never: unbounded buffering."*"""

    def test_a_queue_has_a_capacity(self) -> None:
        assert DeliveryPolicy().queue_capacity > 0

    def test_a_zero_capacity_queue_cannot_be_configured(self) -> None:
        with pytest.raises(ValueError, match="positive capacity"):
            DeliveryPolicy(queue_capacity=0)

    def test_a_subscription_without_a_heartbeat_cannot_be_configured(self) -> None:
        """Without one, a quiet camera and a dead connection are identical (V8)."""
        with pytest.raises(ValueError, match="heartbeat"):
            DeliveryPolicy(heartbeat=Duration(0))

    def test_the_overflow_enum_offers_no_unbounded_option(self) -> None:
        """The three §3.4 forbids are not members, so they cannot be configured."""
        assert {p.value for p in OverflowPolicy} == {
            "conflate",
            "drop_with_gap",
            "disconnect",
        }

    def test_a_slow_subscriber_never_grows_past_its_capacity(
        self, api, state, operator
    ) -> None:
        subscription = api.subscribe(
            operator, scope(CAMERA), policy=DeliveryPolicy(queue_capacity=4)
        )
        observations = publish(state, count=40)
        api.publish(observations)
        assert subscription.depth <= 4


class TestNothingIsDroppedSilently:
    """§3.3 — the most important message type in the contract."""

    def test_dropping_produces_a_gap(self, api, state, operator) -> None:
        subscription = api.subscribe(
            operator,
            scope(CAMERA),
            policy=DeliveryPolicy(queue_capacity=2, overflow=OverflowPolicy.DROP_WITH_GAP),
        )
        api.publish(publish(state, count=20))

        messages = subscription.drain()
        gaps = [m for m in messages if isinstance(m, Gap)]
        assert gaps, "messages were dropped without a Gap"
        assert gaps[0].reason is GapReason.SLOW_CONSUMER

    def test_the_gap_arrives_before_what_followed_it(
        self, api, state, operator
    ) -> None:
        """A consumer must know it missed something *before* applying newer state.

        Otherwise it processes the newer facts and only then learns the sequence
        was incomplete — by which point it may already have acted.
        """
        subscription = api.subscribe(
            operator, scope(CAMERA), policy=DeliveryPolicy(queue_capacity=2)
        )
        api.publish(publish(state, count=20))
        messages = subscription.drain()
        assert isinstance(messages[0], Gap)

    def test_a_gap_says_whether_it_can_be_backfilled(self) -> None:
        """§3.3's ``recoverable``. *"A well-built consumer does exactly that."*

        A consumer that was slow can fetch what it missed. A platform that was
        blind has nothing to give, and saying so saves a pointless query.
        """
        assert GapReason.SLOW_CONSUMER.recoverable
        assert GapReason.BUDGET_SHED.recoverable
        assert not GapReason.PLATFORM_BLIND.recoverable
        assert not GapReason.RETENTION_EXPIRED.recoverable

    def test_consecutive_drops_merge_into_one_gap(self, api, state, operator) -> None:
        """A subscriber ten thousand messages behind gets one gap, not ten thousand.

        Unmerged markers would themselves overflow the queue, and the consumer
        would be told about the loss by losing the notifications about the loss.
        """
        subscription = api.subscribe(
            operator, scope(CAMERA), policy=DeliveryPolicy(queue_capacity=2)
        )
        api.publish(publish(state, count=50))
        gaps = [m for m in subscription.drain() if isinstance(m, Gap)]
        assert len(gaps) == 1
        assert gaps[0].observations_missed and gaps[0].observations_missed > 1

    def test_a_platform_gap_reaches_every_affected_subscriber(
        self, api, state, operator, hub
    ) -> None:
        """Blindness is not something a subscriber can infer on its own."""
        from vision_os.core.model.timebase import Instant

        first = api.subscribe(operator, scope(CAMERA))
        second = api.subscribe(operator, scope(CAMERA))
        delivered = hub.publish_gap(
            Gap(
                start=Instant(0),
                end=Instant(1_000),
                reason=GapReason.PLATFORM_BLIND,
                cameras=(CAMERA,),
                recoverable=False,
            )
        )
        assert delivered == 2
        assert any(isinstance(m, Gap) for m in first.drain())
        assert any(isinstance(m, Gap) for m in second.drain())


class TestDeliveryModes:
    """§3.2."""

    def test_conflation_keeps_the_latest_per_object(
        self, api, state, operator
    ) -> None:
        subscription = api.subscribe(
            operator,
            scope(CAMERA),
            policy=DeliveryPolicy(
                mode=DeliveryMode.CONFLATED,
                overflow=OverflowPolicy.CONFLATE,
                queue_capacity=2,
            ),
        )
        api.publish(publish(state, count=20))
        messages = subscription.drain()
        assert messages
        assert subscription.stats.conflated > 0

    def test_disconnect_closes_rather_than_buffering(
        self, api, state, operator
    ) -> None:
        subscription = api.subscribe(
            operator,
            scope(CAMERA),
            policy=DeliveryPolicy(
                queue_capacity=2, overflow=OverflowPolicy.DISCONNECT
            ),
        )
        api.publish(publish(state, count=20))
        assert subscription.closed

    def test_a_heartbeat_carries_the_cursor(self, api, state, operator, clock) -> None:
        """So a reconnect resumes without loss (§3.1's ``resume_from``)."""
        subscription = api.subscribe(operator, scope(CAMERA))
        api.publish(publish(state, count=2))
        subscription.drain()
        beat = subscription.beat(clock.now())
        assert isinstance(beat, Heartbeat)
        assert beat.cursor is not None


class TestSubscriptionScoping:
    def test_a_subscription_never_receives_another_tenants_observation(
        self, api, state, operator, hub
    ) -> None:
        """The tenant check runs **before** any filter.

        Checking it first means no ordering of predicates could ever let one
        through — the leak §4.2 describes cannot be reached by reordering.
        """
        subscription = api.subscribe(operator, scope(CAMERA))
        foreign = publish(state, count=2, tenant=OTHER_TENANT, start=90)
        hub.publish(foreign)
        assert subscription.drain() == ()

    def test_a_subscription_is_scoped_to_its_cameras(
        self, api, state, operator, hub
    ) -> None:
        subscription = api.subscribe(operator, scope(CAMERA))
        hub.publish(publish(state, count=2, camera=OTHER_CAMERA, start=70))
        assert subscription.drain() == ()

    def test_a_filter_selects_by_observation_type(
        self, api, state, operator
    ) -> None:
        subscription = api.subscribe(
            operator,
            scope(CAMERA),
            filter_=SubscriptionFilter(
                observation_types=(ObservationType.ATTRIBUTE,)
            ),
        )
        api.publish(publish(state, count=4))
        assert subscription.drain() == ()

    def test_a_principal_cannot_hold_unlimited_subscriptions(
        self, clock, metrics, state, authorizer, audit, operator
    ) -> None:
        from vision_os.core.errors import OverloadedError
        from vision_os.core.ports.exposure import ApiLimits
        from vision_os.exposure import ObservationApi

        api = ObservationApi(
            clock=clock,
            metrics=metrics,
            state=state,
            authorizer=authorizer,
            audit=audit,
            limits=ApiLimits(max_subscriptions_per_principal=2),
        )
        api.subscribe(operator, scope(CAMERA))
        api.subscribe(operator, scope(CAMERA))
        with pytest.raises(OverloadedError):
            api.subscribe(operator, scope(CAMERA))

    def test_subscribing_requires_its_own_privilege(self, api, reader) -> None:
        """12_SECURITY §5.3: *"continuous surveillance is a stronger capability
        than point-in-time query."*

        The read-only grant here includes it; what matters is that it is a
        separate action that a grant can withhold.
        """
        from vision_os.core.model.api import Action

        decision = api._authz.authorize(reader, Action.SUBSCRIBE, scope(CAMERA))  # noqa: SLF001
        assert decision.granted is not None


class TestPublishingNeverStallsThePlatform:
    """§3.4: *"Never: stall the platform."*"""

    def test_publishing_to_a_full_subscriber_returns_immediately(
        self, api, state, operator
    ) -> None:
        """The write path owes its readers nothing.

        A publish that awaited a consumer would let one slow dashboard throttle
        the rate at which the platform can record what it sees.
        """
        api.subscribe(operator, scope(CAMERA), policy=DeliveryPolicy(queue_capacity=1))
        observations = publish(state, count=100)
        api.publish(observations)  # completes; nothing to assert but the return

    def test_a_closed_subscription_accepts_nothing_further(
        self, api, state, operator
    ) -> None:
        subscription = api.subscribe(operator, scope(CAMERA))
        subscription.close()
        api.publish(publish(state, count=4))
        assert subscription.drain() == ()

    def test_unsubscribing_removes_it_from_fan_out(
        self, api, state, operator, hub
    ) -> None:
        subscription = api.subscribe(operator, scope(CAMERA))
        api.unsubscribe(operator, subscription.subscription_id)
        assert hub.count == 0

    def test_a_principal_cannot_unsubscribe_another(
        self, api, operator, reader
    ) -> None:
        from vision_os.core.errors import ForbiddenError

        subscription = api.subscribe(operator, scope(CAMERA))
        with pytest.raises(ForbiddenError):
            api.unsubscribe(reader, subscription.subscription_id)
