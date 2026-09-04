"""Demands: what the application actually asks the platform to look at.

### Why nothing was understood before this existed

A replay through the assembled stack produced `cropping.skipped = 5210`, every
one of them `no_demand`, and `understanding.results = 0`. That is the platform
working exactly as designed. M8 does not analyse what nobody asked about, which
is the whole reason cost scales with **demands × changes** rather than
**cameras × fps** — and with no demand registered, the correct number of model
calls is zero.

So this module is not an optimisation or a convenience. It is the missing half
of the economics: the platform provides the mechanism to spend nothing, and the
application has to state what it is willing to spend on.

### Demands come from policy, never from a request

A demand is a standing instruction that costs money on every change to every
matching object. It is derived from the deployment's loaded semantic policies —
the same documents that declare the attributes — so that "what we pay to look
at" and "what we are allowed to record" are the same decision, expressed once.

There is deliberately no HTTP route that registers a demand. `REGISTER_DEMAND`
exists as a permission and stays unwired: a console user who could register one
could spend the model budget directly, and the old validation harness's
`POST /demands` was a harness control that has no production meaning.

### The freshness window is the cost lever

`freshness` is how old an answer may be before the platform pays to re-ask. It
is what makes `FRESH_ENOUGH` fire, and `FRESH_ENOUGH` is the single largest
saving in a working deployment. Setting it too short spends money re-asking
about things that have not changed; setting it too long reports stale claims as
current. It belongs in configuration, and it is named there rather than
defaulted silently here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

#: The subscriber name that appears on every demand this application registers.
#: One name, so a demand's origin is legible in the registry and in metrics.
SUBSCRIBER = "unityworks.compliance"


@dataclass(slots=True)
class DemandAudit:
    """What was asked for, and what the platform said back."""

    attempted: int = 0
    accepted: int = 0
    rejected: int = 0
    attributes: tuple[str, ...] = ()
    classes: tuple[str, ...] = ()
    #: Attributes the platform admitted it cannot serve. Recorded because a
    #: demand that is accepted-but-unsatisfiable is the state most likely to be
    #: mistaken for one that is working.
    unsatisfiable: dict[str, str] = field(default_factory=dict)
    #: What the budget can actually sustain, which may be longer than requested.
    effective_freshness_ms: float | None = None
    requested_freshness_ms: float | None = None
    errors: list[str] = field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "attributes": list(self.attributes),
            "classes": list(self.classes),
            "unsatisfiable": dict(self.unsatisfiable),
            "requested_freshness_ms": self.requested_freshness_ms,
            "effective_freshness_ms": self.effective_freshness_ms,
            # True when the platform quietly widened the window. Surfaced because
            # a demand served at half the requested rate is not the demand that
            # was made.
            "freshness_downgraded": (
                self.effective_freshness_ms is not None
                and self.requested_freshness_ms is not None
                and self.effective_freshness_ms > self.requested_freshness_ms
            ),
            "errors": list(self.errors),
        }


def register_policy_demands(
    composition: Any,
    *,
    freshness_ms: int,
    camera_ids: tuple[str, ...] = (),
) -> DemandAudit:
    """Register one demand per policy, from the composition's own declarations.

    Args:
        camera_ids: Empty means **every camera in scope**, which is what a
            standing compliance policy means — unlike a *permission* scope,
            where empty means none. The two are opposite by design and the
            difference is stated here because it is the kind of thing that gets
            copied wrongly.

    Returns an audit rather than raising. A demand the platform refuses is a
    capability gap in this deployment, and the operator response is to look at
    what is bound — not to restart a process.
    """
    audit = DemandAudit()

    cropping = getattr(composition, "cropping", None)
    engine = getattr(cropping, "engine", None) or getattr(cropping, "manager", None)
    if engine is None or not hasattr(engine, "register_demand"):
        audit.errors.append("no cropping engine is bound; nothing can be demanded")
        return audit

    attributes = _declared_attributes(composition)
    classes = _demandable_classes(composition)
    audit.attributes = attributes
    audit.classes = classes
    audit.requested_freshness_ms = float(freshness_ms)

    if not attributes:
        # Not an error. A deployment with no declared attributes has nothing to
        # ask about, and asking anyway would register a demand that can only
        # ever be refused.
        audit.errors.append("no attributes are declared; no demand was registered")
        return audit

    from vision_os.core.model.demand import Demand, DemandScope, SubjectFilter
    from vision_os.core.model.ids import (
        AttributeKey,
        CameraId,
        ClassId,
        DemandId,
        SubscriberId,
    )
    from vision_os.kernel.clock import Duration

    cameras = tuple(CameraId(c) for c in camera_ids)

    # One demand per policy, built by the policy itself.
    #
    # This function has always been named `register_policy_demands` and has
    # always documented "one demand per policy". It nonetheless built a single
    # aggregate `Demand` by hand, and that hand-built object carried only
    # `class_ids` — dropping `lifecycle`, `min_confidence`, the trigger hints,
    # the priority class and the per-demand budget that every policy document
    # declares. A policy asking for `lifecycle: [active, occluded]` and
    # `min_confidence: 0.4` was parsed, carried and then discarded here, one
    # layer before the registry that would have enforced it.
    #
    # `SemanticPolicy.build_demand` is the platform's own translation and is
    # already complete. Using it means the policy contract reaches the registry
    # intact, and it means this application stops holding a second, poorer
    # opinion about what a policy means.
    demands: list[Demand] = []
    for policy in getattr(composition, "policies", ()) or ():
        build = getattr(policy, "build_demand", None)
        if not callable(build):
            continue
        try:
            demands.append(build(subscriber=SUBSCRIBER, cameras=cameras))
        except Exception as exc:  # noqa: BLE001 - one policy, not the boot
            audit.errors.append(
                f"policy {getattr(policy, 'policy_id', '?')}: "
                f"{type(exc).__name__}: {exc}"
            )

    if not demands:
        # No policy object could describe itself. The aggregate demand is kept
        # as the fallback so a deployment whose composition exposes attributes
        # but no policy objects behaves exactly as it did before.
        demands = [
            Demand(
                # Deterministic, so a restart re-registers the same demand
                # rather than accumulating a new one on every boot.
                demand_id=DemandId(f"{SUBSCRIBER}/standing"),
                subscriber=SubscriberId(SUBSCRIBER),
                scope=DemandScope(camera_ids=cameras),
                subject_filter=SubjectFilter(
                    class_ids=tuple(ClassId(c) for c in classes)
                ),
                required_attributes=tuple(AttributeKey(a) for a in attributes),
                freshness=Duration.from_millis(freshness_ms),
            )
        ]

    acknowledgement = None
    for demand in demands:
        audit.attempted += 1
        try:
            acknowledgement = engine.register_demand(demand)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            audit.rejected += 1
            audit.errors.append(f"{demand.demand_id}: {type(exc).__name__}: {exc}")
            logger.warning(
                "compliance demand {} was refused: {}: {}",
                demand.demand_id,
                type(exc).__name__,
                exc,
            )
            continue

        audit.accepted += 1
        _read_acknowledgement(acknowledgement, audit)

    if not audit.accepted:
        return audit

    logger.info(
        "standing demand registered — {} attribute(s) over {} class(es), " "freshness {} ms{}",
        len(attributes),
        len(classes) or "all",
        audit.effective_freshness_ms or freshness_ms,
        " (widened by the budget)" if audit.to_wire()["freshness_downgraded"] else "",
    )
    return audit


def _read_acknowledgement(acknowledgement: Any, audit: DemandAudit) -> None:
    """Record what the platform actually promised."""
    effective = getattr(acknowledgement, "effective_freshness", None)
    if effective is not None:
        audit.effective_freshness_ms = float(getattr(effective, "ns", 0)) / 1_000_000

    for entry in getattr(acknowledgement, "unsatisfiable", ()) or ():
        try:
            key, reason = entry
        except (TypeError, ValueError):
            continue
        audit.unsatisfiable[str(key)] = str(getattr(reason, "value", reason))

    if audit.unsatisfiable:
        # Loud. An attribute nobody can produce will never appear, and the
        # symptom downstream is a compliance rule that is permanently UNKNOWN
        # for a reason nothing else explains.
        logger.warning(
            "the platform cannot serve {} demanded attribute(s): {}",
            len(audit.unsatisfiable),
            audit.unsatisfiable,
        )


def _declared_attributes(composition: Any) -> tuple[str, ...]:
    """Attribute keys the registry has admitted through the neutrality gate."""
    declared = getattr(composition, "declared_attributes", ())
    return tuple(str(key) for key in declared)


def _demandable_classes(composition: Any) -> tuple[str, ...]:
    """Object classes the loaded policies name.

    Empty means "every class", which is the honest reading of a policy that does
    not restrict: a demand with no class filter matches whatever the detector
    finds. No class name is written here — `person`, `hairnet` and every other
    domain word live in policy documents (§31).
    """
    classes: list[str] = []
    for policy in getattr(composition, "policies", ()) or ():
        for name in getattr(policy, "object_classes", ()) or ():
            text = str(name)
            if text not in classes:
                classes.append(text)
    return tuple(classes)


__all__ = ["SUBSCRIBER", "DemandAudit", "register_policy_demands"]
