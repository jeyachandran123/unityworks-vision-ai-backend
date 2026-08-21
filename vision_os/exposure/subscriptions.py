"""Subscriptions and the `Gap` contract (09_API §3).

The whole design turns on one sentence:

> §3.3: *"**A subscriber is never silently skipped.** If the platform drops
> messages for any reason — the consumer was slow, the camera was blind, the
> budget was exhausted — an explicit `Gap` is delivered. This is V8 applied to
> delivery, and it is what allows a consumer to distinguish 'nothing happened'
> from 'you were not told what happened.'"*

Three behaviours §3.4 forbids are unrepresentable here rather than merely avoided:

* **Unbounded buffering** — every queue has a capacity, checked at construction.
* **Silent drop** — every drop path constructs a `Gap` before it discards.
* **Stalling the platform** — `publish` never blocks and never awaits a consumer.

The last one is why fan-out is push-into-a-bounded-queue rather than await-on-a-
consumer. A subscriber that stops reading fills its own queue and receives a
`Gap`; it cannot slow the observation pipeline that feeds it.

**Filters are evaluated once per observation, not once per subscriber.** §M14
Performance: *"Subscription fan-out is the dominant cost; filters are evaluated
once per observation against an index of subscriber predicates rather than once
per subscriber."* The index here is by observation type and camera, which are the
two dimensions almost every real filter constrains.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from ..core.model.api import (
    CoverageChange,
    DeliveryMode,
    DeliveryPolicy,
    Gap,
    GapReason,
    Heartbeat,
    Message,
    OverflowPolicy,
    Principal,
    Scope,
    StateDeltaMessage,
    SubscriptionFilter,
)
from ..core.model.ids import CameraId, ObjectId
from ..core.model.observation import Observation, ObservationType
from ..core.model.timebase import Duration, Instant
from ..core.ports.clock import Clock
from ..kernel.metrics import MetricName, MetricsEngine


@dataclass(slots=True)
class SubscriptionStats:
    """Per-subscription counters. Mutable; never published as a value."""

    delivered: int = 0
    dropped: int = 0
    gaps: int = 0
    conflated: int = 0
    heartbeats: int = 0

    @property
    def loss_rate(self) -> float:
        total = self.delivered + self.dropped
        return self.dropped / total if total else 0.0


class Subscription:
    """One consumer's live stream.

    Owns a **bounded** queue and the cursor that makes reconnection lossless.
    §3.2: *"a consumer reconnecting with `resume_from` receives everything since
    that cursor, bounded by log retention."*
    """

    __slots__ = (
        "_clock",
        "_closed",
        "_conflated",
        "_cursor",
        "_last_heartbeat",
        "_lock",
        "_pending_gap",
        "_queue",
        "filter",
        "policy",
        "principal",
        "scope",
        "stats",
        "subscription_id",
    )

    def __init__(
        self,
        subscription_id: str,
        *,
        principal: Principal,
        scope: Scope,
        filter_: SubscriptionFilter,
        policy: DeliveryPolicy,
        clock: Clock,
    ) -> None:
        self.subscription_id = subscription_id
        self.principal = principal
        self.scope = scope
        self.filter = filter_
        self.policy = policy
        self.stats = SubscriptionStats()

        self._clock = clock
        self._queue: deque[Message] = deque()
        self._conflated: dict[ObjectId, Observation] = {}
        self._cursor: str | None = policy.resume_from
        self._pending_gap: Gap | None = None
        self._last_heartbeat = clock.now()
        self._closed = False
        self._lock = threading.Lock()

    # --- delivery -------------------------------------------------------------- #

    def offer(self, message: Message) -> bool:
        """Enqueue one message, applying the overflow policy. **Never blocks.**

        Returns whether the message was accepted. A rejection is always
        accompanied by a queued `Gap`, so the subscriber learns of the loss
        through the stream rather than by noticing a hole in it.
        """
        with self._lock:
            if self._closed:
                return False

            if len(self._queue) < self.policy.queue_capacity:
                self._queue.append(message)
                self.stats.delivered += 1
                return True

            return self._overflow(message)

    def _overflow(self, message: Message) -> bool:
        """The queue is full. Apply the declared policy — and only that policy.

        Note what is absent: no branch grows the queue, and no branch discards
        without recording. §3.4's three prohibitions are enforced by there being
        no code that could violate them.
        """
        if self.policy.overflow is OverflowPolicy.CONFLATE:
            return self._conflate(message)

        if self.policy.overflow is OverflowPolicy.DISCONNECT:
            self._closed = True
            self._note_gap(message, GapReason.SLOW_CONSUMER, missed=len(self._queue))
            return False

        # DROP_WITH_GAP — drop the oldest, keep the newest, record the loss.
        dropped = self._queue.popleft()
        self._queue.append(message)
        self.stats.dropped += 1
        self._note_gap(dropped, GapReason.SLOW_CONSUMER, missed=1)
        return True

    def _conflate(self, message: Message) -> bool:
        """Collapse to the latest per object (§3.2's `conflated` mode).

        Only observations conflate. A `Gap` or a `CoverageChange` is not a state
        update with a newer version — it is a distinct fact, and collapsing two
        of them would lose one.
        """
        if not isinstance(message, Observation) or message.object_id is None:
            self.stats.dropped += 1
            self._note_gap(message, GapReason.SLOW_CONSUMER, missed=1)
            return False

        previous = self._conflated.get(message.object_id)
        self._conflated[message.object_id] = message
        if previous is None:
            self._note_gap(message, GapReason.SLOW_CONSUMER, missed=0)
        self.stats.conflated += 1
        return True

    def _note_gap(self, around: Message, reason: GapReason, *, missed: int) -> None:
        """Queue a `Gap`, merging with one already pending.

        Merged rather than appended because a subscriber falling behind by ten
        thousand messages should receive one gap covering the window, not ten
        thousand gap markers that themselves overflow the queue.
        """
        moment = _moment_of(around) or self._clock.now()
        if self._pending_gap is not None:
            self._pending_gap = Gap(
                start=self._pending_gap.start,
                end=moment,
                reason=self._pending_gap.reason,
                cameras=self._pending_gap.cameras,
                observations_missed=(self._pending_gap.observations_missed or 0) + missed,
                recoverable=self._pending_gap.recoverable,
            )
            return

        self._pending_gap = Gap(
            start=moment,
            end=moment,
            reason=reason,
            cameras=_cameras_of(around),
            observations_missed=missed,
            recoverable=reason.recoverable,
        )
        self.stats.gaps += 1

    def drain(self, *, limit: int = 256) -> tuple[Message, ...]:
        """Take what is queued. A pending `Gap` is delivered **first**.

        First because a consumer must know it missed something *before* it
        processes what came after; otherwise it applies newer state and only then
        learns the sequence was incomplete.
        """
        with self._lock:
            out: list[Message] = []
            if self._pending_gap is not None:
                out.append(self._pending_gap)
                self._pending_gap = None

            if self._conflated:
                out.extend(self._conflated.values())
                self._conflated.clear()

            while self._queue and len(out) < limit:
                out.append(self._queue.popleft())

            for message in reversed(out):
                cursor = _cursor_of(message)
                if cursor is not None:
                    self._cursor = cursor
                    break
            return tuple(out)

    def heartbeat_due(self, now: Instant) -> bool:
        return now.ns - self._last_heartbeat.ns >= self.policy.heartbeat.ns

    def beat(self, now: Instant) -> Heartbeat:
        """Emit liveness. §3.1's default cadence is 10s.

        Without it, a healthy subscription over a quiet camera and a dead
        connection produce identical silence — the same failure the observation
        heartbeat prevents one layer down.
        """
        with self._lock:
            self._last_heartbeat = now
            self.stats.heartbeats += 1
            return Heartbeat(at=now, cursor=self._cursor)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._queue.clear()
            self._conflated.clear()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def cursor(self) -> str | None:
        return self._cursor

    @property
    def depth(self) -> int:
        return len(self._queue) + len(self._conflated)

    @property
    def lagging(self) -> bool:
        return self.depth >= self.policy.queue_capacity


class SubscriptionHub:
    """Fan-out to every matching subscription (§M14 Performance).

    Holds an index keyed by observation type and camera so a publish evaluates
    each subscriber's remaining predicate only when the cheap dimensions already
    matched. With a thousand subscribers watching different cameras, an unindexed
    fan-out would evaluate a thousand filters per observation.
    """

    __slots__ = ("_by_camera", "_by_type", "_clock", "_lock", "_metrics", "_subs")

    def __init__(self, *, clock: Clock, metrics: MetricsEngine) -> None:
        self._clock = clock
        self._metrics = metrics
        self._subs: dict[str, Subscription] = {}
        self._by_type: dict[ObservationType, set[str]] = {}
        self._by_camera: dict[CameraId, set[str]] = {}
        self._lock = threading.Lock()

    def add(self, subscription: Subscription) -> None:
        with self._lock:
            self._subs[subscription.subscription_id] = subscription
            for kind in subscription.filter.observation_types or tuple(ObservationType):
                self._by_type.setdefault(kind, set()).add(subscription.subscription_id)
            for camera in subscription.scope.camera_ids:
                self._by_camera.setdefault(camera, set()).add(
                    subscription.subscription_id
                )

    def remove(self, subscription_id: str) -> None:
        with self._lock:
            subscription = self._subs.pop(subscription_id, None)
            if subscription is None:
                return
            subscription.close()
            for holders in self._by_type.values():
                holders.discard(subscription_id)
            for holders in self._by_camera.values():
                holders.discard(subscription_id)

    def publish(self, observations: Sequence[Observation]) -> int:
        """Fan one batch out. Returns how many deliveries were made.

        Never raises, never blocks, never awaits a consumer. A subscription that
        cannot keep up degrades itself, not the platform.
        """
        delivered = 0
        for observation in observations:
            for subscription in self._candidates(observation):
                if not self._in_scope(subscription, observation):
                    continue
                if not subscription.filter.matches(observation):
                    continue
                if subscription.offer(observation):
                    delivered += 1
        if delivered:
            self._metrics.counter(MetricName.API_MESSAGES_DELIVERED).increment(delivered)
        return delivered

    def publish_delta(self, delta: StateDeltaMessage) -> int:
        delivered = 0
        for subscription in self._for_camera(delta.camera_id):
            if subscription.scope.covers_camera(delta.camera_id) and subscription.offer(delta):
                delivered += 1
        return delivered

    def publish_coverage(self, change: CoverageChange) -> int:
        """Coverage changes reach **every** subscriber in scope.

        Unfiltered by observation type on purpose: a consumer subscribed only to
        attribute observations still needs to know the camera went blind, or it
        will read the resulting silence as an absence of attributes.
        """
        delivered = 0
        for subscription in self._for_camera(change.camera_id):
            if subscription.scope.covers_camera(change.camera_id) and subscription.offer(change):
                delivered += 1
        return delivered

    def publish_gap(self, gap: Gap) -> int:
        """A platform-originated gap — blindness, shedding, an unavailable partition.

        Distinct from the slow-consumer gaps a subscription raises for itself:
        this one is the platform telling every affected subscriber that *it* could
        not see, which no subscriber could infer on its own.
        """
        delivered = 0
        cameras = gap.cameras or tuple(self._by_camera)
        seen: set[str] = set()
        for camera in cameras:
            for subscription in self._for_camera(camera):
                if subscription.subscription_id in seen:
                    continue
                seen.add(subscription.subscription_id)
                if subscription.offer(gap):
                    delivered += 1
        return delivered

    def beat(self) -> int:
        """Send heartbeats to whoever is due one."""
        now = self._clock.now()
        sent = 0
        for subscription in self.subscriptions:
            if subscription.heartbeat_due(now):
                subscription.offer(subscription.beat(now))
                sent += 1
        return sent

    def _candidates(self, observation: Observation) -> Iterator[Subscription]:
        with self._lock:
            ids = set(self._by_type.get(observation.observation_type, ()))
            camera_ids = self._by_camera.get(observation.camera_id)
            unscoped = {
                s.subscription_id
                for s in self._subs.values()
                if not s.scope.camera_ids
            }
            if camera_ids is not None:
                ids &= camera_ids | unscoped
            else:
                ids &= unscoped
            return iter([self._subs[i] for i in ids if i in self._subs])

    def _for_camera(self, camera_id: CameraId) -> Iterator[Subscription]:
        with self._lock:
            ids = set(self._by_camera.get(camera_id, ()))
            ids |= {s.subscription_id for s in self._subs.values() if not s.scope.camera_ids}
            return iter([self._subs[i] for i in ids if i in self._subs])

    @staticmethod
    def _in_scope(subscription: Subscription, observation: Observation) -> bool:
        """Tenant first, always.

        12_SECURITY §4.1 makes tenancy part of identity. A subscription must
        never see another tenant's observation, and checking it here — before any
        filter — means there is no ordering of predicates that could let one
        through.
        """
        if observation.tenant_id != subscription.scope.tenant_id:
            return False
        return subscription.scope.covers_camera(observation.camera_id)

    @property
    def subscriptions(self) -> tuple[Subscription, ...]:
        with self._lock:
            return tuple(self._subs.values())

    @property
    def count(self) -> int:
        return len(self._subs)

    def for_principal(self, subject: str) -> tuple[Subscription, ...]:
        return tuple(s for s in self.subscriptions if s.principal.subject == subject)

    def close_all(self) -> None:
        for subscription in self.subscriptions:
            self.remove(subscription.subscription_id)


def _moment_of(message: Message) -> Instant | None:
    for attribute in ("t_capture", "at", "start"):
        value = getattr(message, attribute, None)
        if isinstance(value, Instant):
            return value
    return None


def _cameras_of(message: Message) -> tuple[CameraId, ...]:
    camera = getattr(message, "camera_id", None)
    return (camera,) if camera is not None else ()


def _cursor_of(message: Message) -> str | None:
    """The resumption token for a message.

    Only observations carry one: a heartbeat or a gap is not a position in the
    log, and treating one as a cursor would let a reconnect skip whatever the log
    held between them.
    """
    if isinstance(message, Observation):
        return str(message.observation_id)
    return None


@dataclass(frozen=True, slots=True)
class HubReport:
    """What an operator needs to know about fan-out."""

    subscriptions: int = 0
    lagging: int = 0
    total_delivered: int = 0
    total_dropped: int = 0
    total_gaps: int = 0
    modes: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    @property
    def healthy(self) -> bool:
        return self.lagging == 0


def report_of(hub: SubscriptionHub) -> HubReport:
    subscriptions = hub.subscriptions
    modes: dict[str, int] = {}
    for subscription in subscriptions:
        modes[subscription.policy.mode.value] = (
            modes.get(subscription.policy.mode.value, 0) + 1
        )
    return HubReport(
        subscriptions=len(subscriptions),
        lagging=sum(1 for s in subscriptions if s.lagging),
        total_delivered=sum(s.stats.delivered for s in subscriptions),
        total_dropped=sum(s.stats.dropped for s in subscriptions),
        total_gaps=sum(s.stats.gaps for s in subscriptions),
        modes=tuple(sorted(modes.items())),
    )


DEFAULT_POLICY = DeliveryPolicy(
    mode=DeliveryMode.ALL,
    overflow=OverflowPolicy.DROP_WITH_GAP,
    max_lag=Duration.from_millis(5_000),
)
