"""L7 Exposure — M14 Observation API.

> `01_LAYERED` §1.1: *"Serve state and observations to consumers, safely and under
> contract."* Explicitly not responsible for: **producing anything**.

This package is the platform's only external surface. Everything it serves comes
from M12's immutable snapshots; nothing it does can change a fact.
"""

from __future__ import annotations

from .api import ObservationApi
from .audit import AuditTrail, CountingAuditSink
from .demands import DemandIntake
from .subscriptions import Subscription, SubscriptionHub

__all__ = [
    "AuditTrail",
    "CountingAuditSink",
    "DemandIntake",
    "ObservationApi",
    "Subscription",
    "SubscriptionHub",
]
