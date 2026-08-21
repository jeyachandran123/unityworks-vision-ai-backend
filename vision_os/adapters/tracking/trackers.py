"""The shipped ``TrackerPort`` adapters.

Three trackers, three points on the accuracy/cost curve, one port. Swapping
between them is a change to one factory call and one config value; **no platform
module changes**, which is the property the whole port structure exists to buy.

| Adapter | Motion model | Association | Occlusion | Use |
|---|---|---|---|---|
| ``tracker.iou`` | none | single-stage greedy | none | Universal fallback |
| ``tracker.sort`` | linear | single-stage optimal | short | Default |
| ``tracker.bytetrack`` | linear | two-stage optimal | short | Crowded scenes |

The names appear here and in the composition root. They appear nowhere else in
the platform, and an architecture test enforces it.
"""

from __future__ import annotations

from ...perception.tracking.association import (
    AssociationPolicy,
    GreedyAssociator,
    OptimalAssociator,
)
from ...perception.tracking.lifecycle import LifecyclePolicy
from .geometric import GeometricConfig, GeometricTracker


def build_iou_tracker(
    *,
    lifecycle: LifecyclePolicy | None = None,
    association: AssociationPolicy | None = None,
    config_revision: str = "unset",
    history_length: int = 32,
) -> GeometricTracker:
    """``tracker.iou`` — pure geometry. **The universal fallback.**

    No motion model, no weights, no device, no optional import. One of only two
    components in the platform that can never be unavailable (10_RELIABILITY
    section 7.3): tracking degrades in *accuracy* when a better tracker fails,
    never in *availability*.

    Deliberately minimal. It exists to keep the pipeline running, and every
    property it lacks — motion prediction, occlusion handling — is one it would
    need a working model or tuning to provide.
    """
    return GeometricTracker(
        config=GeometricConfig(
            tracker_id="tracker.iou",
            use_motion_model=False,
            two_stage=False,
            handles_occlusion="none",
        ),
        lifecycle=lifecycle
        or LifecyclePolicy(min_hits_to_confirm=2, max_coast_frames=2, max_lost_frames=3),
        association=association or AssociationPolicy(),
        associator=GreedyAssociator(),
        config_revision=config_revision,
        history_length=history_length,
    )


def build_sort_tracker(
    *,
    lifecycle: LifecyclePolicy | None = None,
    association: AssociationPolicy | None = None,
    config_revision: str = "unset",
    history_length: int = 32,
) -> GeometricTracker:
    """``tracker.sort`` — linear motion prediction with optimal assignment.

    The sensible default: motion prediction lets a track survive the frame gaps
    the scheduler creates by design, and optimal assignment avoids the greedy
    failure where an early cheap match steals a detection a later track needed
    more.
    """
    return GeometricTracker(
        config=GeometricConfig(
            tracker_id="tracker.sort",
            use_motion_model=True,
            two_stage=False,
            handles_occlusion="short",
        ),
        lifecycle=lifecycle or LifecyclePolicy(),
        association=association or AssociationPolicy(),
        associator=OptimalAssociator(),
        config_revision=config_revision,
        history_length=history_length,
    )


def build_bytetrack_tracker(
    *,
    lifecycle: LifecyclePolicy | None = None,
    association: AssociationPolicy | None = None,
    config_revision: str = "unset",
    history_length: int = 32,
) -> GeometricTracker:
    """``tracker.bytetrack`` — two-stage association.

    The insight it is built on: a detection too weak to *start* a track is often
    still strong enough to *continue* one. A partially occluded person whose
    score drops to 0.3 would be discarded by a single-stage tracker, breaking the
    track and creating a new id when they re-emerge; using that weak detection to
    continue the existing track instead is the single largest reduction in
    fragmentation available without an appearance model.

    Costs one extra assignment pass over detections that were going to be thrown
    away, so it is nearly free.
    """
    return GeometricTracker(
        config=GeometricConfig(
            tracker_id="tracker.bytetrack",
            use_motion_model=True,
            two_stage=True,
            low_confidence_floor=0.1,
            high_confidence_floor=0.5,
            handles_occlusion="short",
        ),
        lifecycle=lifecycle or LifecyclePolicy(max_coast_frames=8, max_lost_frames=20),
        association=association or AssociationPolicy(),
        associator=OptimalAssociator(),
        config_revision=config_revision,
        history_length=history_length,
    )


#: Name to factory. The **only** mapping in the platform from a tracker name to
#: an implementation, consumed by the composition root alone.
TRACKER_FACTORIES = {
    "tracker.iou": build_iou_tracker,
    "tracker.sort": build_sort_tracker,
    "tracker.bytetrack": build_bytetrack_tracker,
}

#: The tracker used when a configured one fails to load or fails conformance.
#: Named as a constant so the fallback is a declared policy rather than a string
#: buried in an exception handler.
FALLBACK_TRACKER_ID = "tracker.iou"
