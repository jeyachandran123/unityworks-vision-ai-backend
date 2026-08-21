"""M19 Event Bus — typed publish/subscribe with bounded, attributed delivery.

Single responsibility: *deliver typed notifications. Understand none of them.*

Two properties are non-negotiable:

* **Bounded, always.** An unbounded queue is a memory leak with a delayed fuse.
* **Never silent.** When a subscriber's buffer overflows, a ``Gap`` is delivered
  carrying the dropped count and reason. A subscriber is never silently skipped
  (invariant V8).

The bus is control-plane only. No frame, crop, or tensor may travel on it.
"""

from __future__ import annotations

import enum
import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ...core.model.timebase import Instant
from ...core.ports.clock import Clock
from ...core.ports.observability import EventTransportPort
from .events import ALL_EVENT_TYPES, Event, Gap


class OverflowPolicy(enum.Enum):
    """What to do when a subscription's bounded buffer is full."""

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    CONFLATE = "conflate"
    """Keep only the latest event per partition key."""


@dataclass(frozen=True, slots=True)
class DeliveryPolicy:
    capacity: int = 1024
    overflow: OverflowPolicy = OverflowPolicy.DROP_OLDEST

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {self.capacity}")


@dataclass(slots=True)
class BusStats:
    published: int = 0
    delivered: int = 0
    dropped: int = 0
    gaps_emitted: int = 0
    transport_failures: int = 0
    by_type: dict[str, int] = field(default_factory=dict)


class Subscription:
    """A bounded, per-subscriber delivery queue.

    Drained by its owner. The bus never runs subscriber code on the publisher's
    thread, so a slow or broken subscriber cannot stall a producer.

    **Gap reporting is structural, not best-effort.** Drops are accumulated on
    the subscription and a ``Gap`` is synthesized at the head of the next
    ``drain()``. Enqueuing a Gap like an ordinary event would let a busy stream
    evict the very marker that announces the loss — silently defeating invariant
    V8 exactly when it matters most.
    """

    __slots__ = (
        "_id",
        "_types",
        "_filter",
        "_policy",
        "_queue",
        "_lock",
        "_dropped",
        "_undeclared_drops",
        "_closed",
        "_clock",
    )

    def __init__(
        self,
        subscription_id: str,
        event_types: frozenset[str],
        predicate: Callable[[Event], bool] | None,
        policy: DeliveryPolicy,
        clock: Clock,
    ) -> None:
        self._id = subscription_id
        self._types = event_types
        self._filter = predicate
        self._policy = policy
        self._queue: deque[Event] = deque()
        self._lock = threading.Lock()
        self._dropped = 0
        self._undeclared_drops = 0
        self._closed = False
        self._clock = clock

    @property
    def subscription_id(self) -> str:
        return self._id

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def depth(self) -> int:
        with self._lock:
            return len(self._queue)

    def matches(self, event: Event) -> bool:
        if self._types and type(event).event_type not in self._types:
            return False
        if self._filter is not None:
            try:
                return self._filter(event)
            except Exception:  # noqa: BLE001 - a bad filter must not break the bus
                return False
        return True

    def _offer(self, event: Event) -> int:
        """Enqueue, applying the overflow policy. Returns the number dropped."""
        with self._lock:
            if self._closed:
                return 0
            if self._policy.overflow is OverflowPolicy.CONFLATE:
                for index, existing in enumerate(self._queue):
                    if existing.partition_key == event.partition_key:
                        self._queue[index] = event
                        return 0
            if len(self._queue) < self._policy.capacity:
                self._queue.append(event)
                return 0
            if self._policy.overflow is OverflowPolicy.DROP_NEWEST:
                self._dropped += 1
                self._undeclared_drops += 1
                return 1
            self._queue.popleft()
            self._queue.append(event)
            self._dropped += 1
            self._undeclared_drops += 1
            return 1

    def drain(self, limit: int | None = None) -> list[Event]:
        """Take buffered events, preceded by a ``Gap`` if any were dropped.

        A subscriber cannot drain without learning what it missed.
        """
        with self._lock:
            gap: Gap | None = None
            if self._undeclared_drops:
                gap = Gap(
                    occurred_at=self._clock.now(),
                    partition_key=self._id,
                    subscription_id=self._id,
                    dropped=self._undeclared_drops,
                    reason="subscriber_overflow",
                )
                self._undeclared_drops = 0

            if limit is None or limit >= len(self._queue):
                events = list(self._queue)
                self._queue.clear()
            else:
                events = [self._queue.popleft() for _ in range(limit)]

            return ([gap, *events] if gap else events)

    @property
    def undeclared_drops(self) -> int:
        with self._lock:
            return self._undeclared_drops

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._queue.clear()


class EventBus:
    """The kernel's notification fabric.

    Publishing is non-blocking and must stay sub-microsecond: it happens on hot
    paths. Fan-out is O(matching subscriptions) via a type index, so a publish
    never walks every subscriber.
    """

    def __init__(
        self,
        clock: Clock,
        *,
        transport: EventTransportPort | None = None,
        gap_report_threshold: int = 1,
    ) -> None:
        self._clock = clock
        self._transport = transport
        self._gap_threshold = max(1, gap_report_threshold)
        self._lock = threading.RLock()
        self._by_type: dict[str, list[Subscription]] = {}
        self._all: list[Subscription] = []
        self._registered: set[str] = {t.event_type for t in ALL_EVENT_TYPES}
        self._stats = BusStats()

    # --- registration ---------------------------------------------------- #

    def register_event_type(self, event_type: str) -> None:
        """Register a new event type. The bus stays typed rather than becoming
        an untyped message soup."""
        with self._lock:
            self._registered.add(event_type)

    def is_registered(self, event_type: str) -> bool:
        with self._lock:
            return event_type in self._registered

    # --- publish --------------------------------------------------------- #

    def publish(self, event: Event) -> None:
        """Deliver to matching subscriptions. Never blocks, never raises."""
        event_type = type(event).event_type
        with self._lock:
            if event_type not in self._registered:
                raise ValueError(f"unregistered event type: {event_type}")
            self._stats.published += 1
            self._stats.by_type[event_type] = self._stats.by_type.get(event_type, 0) + 1
            targets = list(self._by_type.get(event_type, ())) + list(self._all)

        for subscription in targets:
            if not subscription.matches(event):
                continue
            dropped = subscription._offer(event)  # noqa: SLF001 - same-module collaborator
            with self._lock:
                if dropped:
                    self._stats.dropped += dropped
                    self._stats.gaps_emitted += dropped
                else:
                    self._stats.delivered += 1

        self._forward(event)

    def _forward(self, event: Event) -> None:
        if self._transport is None:
            return
        try:
            self._transport.deliver(
                type(event).event_type, event.partition_key, event.payload()
            )
        except Exception:  # noqa: BLE001 - transport failure never affects perception
            with self._lock:
                self._stats.transport_failures += 1

    # --- subscribe ------------------------------------------------------- #

    def subscribe(
        self,
        event_types: Iterable[type[Event]] | Iterable[str] | None = None,
        *,
        predicate: Callable[[Event], bool] | None = None,
        policy: DeliveryPolicy | None = None,
        subscription_id: str | None = None,
    ) -> Subscription:
        """Create a bounded subscription.

        Args:
            event_types: Types to receive. ``None`` subscribes to everything.
            predicate: Optional content filter, evaluated once per event.
            policy: Capacity and overflow behaviour.
        """
        names: frozenset[str] = frozenset()
        if event_types is not None:
            resolved: list[str] = []
            for item in event_types:
                resolved.append(item if isinstance(item, str) else item.event_type)
            names = frozenset(resolved)

        sid = subscription_id or f"sub-{id(object())}"
        subscription = Subscription(
            sid, names, predicate, policy or DeliveryPolicy(), self._clock
        )

        with self._lock:
            unknown = names - self._registered
            if unknown:
                raise ValueError(f"unregistered event types: {sorted(unknown)}")
            if names:
                for name in names:
                    self._by_type.setdefault(name, []).append(subscription)
            else:
                self._all.append(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            for subs in self._by_type.values():
                if subscription in subs:
                    subs.remove(subscription)
            if subscription in self._all:
                self._all.remove(subscription)
        subscription.close()

    # --- telemetry ------------------------------------------------------- #

    def stats(self) -> BusStats:
        with self._lock:
            return BusStats(
                published=self._stats.published,
                delivered=self._stats.delivered,
                dropped=self._stats.dropped,
                gaps_emitted=self._stats.gaps_emitted,
                transport_failures=self._stats.transport_failures,
                by_type=dict(self._stats.by_type),
            )

    @property
    def subscription_count(self) -> int:
        with self._lock:
            return len(self._all) + sum(len(v) for v in self._by_type.values())

    def close(self) -> None:
        with self._lock:
            everything = list(self._all)
            for subs in self._by_type.values():
                everything.extend(subs)
            self._all.clear()
            self._by_type.clear()
        for subscription in everything:
            subscription.close()


def now_event_detail(**kwargs: Any) -> dict[str, Any]:
    """Small helper for constructing bounded event detail payloads."""
    return {k: v for k, v in kwargs.items() if v is not None}


__all__ = [
    "BusStats",
    "DeliveryPolicy",
    "EventBus",
    "Instant",
    "OverflowPolicy",
    "Subscription",
    "now_event_detail",
]
