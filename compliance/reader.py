"""The one read path from Vision OS into the compliance layer.

Every fact this package evaluates arrives through here, and it arrives
**scope-narrowed**. 12_SECURITY §4.2 is explicit about why that matters:

> *"A query that fetches broadly and filters afterwards leaks whenever the filter
> has a bug, whenever an error path returns unfiltered data, whenever pagination
> interacts badly, and whenever a new code path forgets to apply it. Constructing
> the query already scoped means **there is no moment at which cross-tenant data
> exists in memory to leak**."*

The platform enforces that discipline inside its own API. A rule engine sitting
outside it could quietly reintroduce the leak by asking for a wide scope and
filtering the results itself, and no existing test would see it. So this module
is the only place in the package that touches the API, it always queries the
scope the authorizer returned, and it never filters afterwards.

**It reads and never writes.** There is no write path into Vision State — not
because one is guarded, but because none exists to call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from vision_os.core.model.api import (
    Action,
    CapabilitySummary,
    CoverageSummary,
    ObjectView,
    Principal,
    QueryOptions,
    Scope,
    StateFilter,
)
from vision_os.core.model.ids import EvidenceId
from vision_os.exposure.api import ObservationApi


@dataclass(frozen=True, slots=True)
class ObservationSnapshot:
    """What one read returned, with everything needed to interpret it.

    ``coverage`` and ``capabilities`` travel alongside the objects rather than
    being fetched separately, because a verdict reached without them is a verdict
    reached without knowing whether the platform could see. The platform makes
    ``coverage`` mandatory on its own result for exactly this reason; carrying it
    forward keeps the guarantee intact one layer further out.
    """

    objects: tuple[ObjectView, ...] = ()
    coverage: CoverageSummary = field(default_factory=CoverageSummary)
    capabilities: CapabilitySummary = field(default_factory=CapabilitySummary)
    complete: bool = True
    """Whether every requested partition answered. A caller concluding "nothing
    was wrong" must check this as well as ``coverage``."""

    @property
    def capability_gaps(self) -> tuple[str, ...]:
        """Attribute and class names the platform currently cannot produce.

        Read from the live capability report rather than assumed, because
        *"capability is live state, not documentation"* — a model evicted under
        memory pressure changes this, and a rule depending on what it produced
        should become ``UNKNOWN`` rather than silently never firing.
        """
        return tuple(subject for subject, _ in self.capabilities.gaps)


class ObservationReader:
    """Reads published facts for evaluation. The package's only I/O.

    Holds the API and a principal. It does not hold rules, does not evaluate, and
    does not decide — separating the read from the judgment is what lets the
    evaluator stay a pure function that a test can drive with hand-built views.
    """

    __slots__ = ("_api", "_principal")

    def __init__(self, api: ObservationApi, *, principal: Principal) -> None:
        self._api = api
        self._principal = principal

    @property
    def principal(self) -> Principal:
        return self._principal

    def read(
        self,
        scope: Scope,
        *,
        filter_: StateFilter | None = None,
        limit: int = 100,
    ) -> ObservationSnapshot:
        """Current state within ``scope``, as the platform narrowed it.

        ``include_provenance`` is left at its default of ``True``. The platform's
        own note on that default — *"explainability is not opt-out by accident"* —
        applies with more force here: a finding whose observation cannot name its
        producer is a finding nobody can audit.
        """
        # Keyword arguments, because ``query_state`` declares everything after
        # ``scope`` keyword-only. Passing them positionally raised a TypeError on
        # the first real call — the API's signature is deliberate and this is the
        # caller conforming to it, not the other way round.
        result = self._api.query_state(
            self._principal,
            scope,
            filter_=filter_ or StateFilter(),
            options=QueryOptions(limit=limit),
        )
        return ObservationSnapshot(
            objects=tuple(result.objects),
            coverage=result.coverage,
            capabilities=result.capabilities,
            complete=result.complete,
        )

    def capabilities(self, scope: Scope) -> CapabilitySummary:
        """What the platform can produce right now.

        Called at startup against a rule set's ``required_attributes`` so a
        deployment learns immediately that a rule can never reach a verdict here,
        rather than watching it return ``UNKNOWN`` forever.
        """
        return self._api.capabilities(self._principal, scope)

    def evidence(self, evidence_id: EvidenceId, *, purpose: str):
        """Resolve one evidence handle. **Requires a separate privilege.**

        Split from ``read`` deliberately, mirroring the platform's own split
        (obligation Z3): reading *"a person was here"* and viewing their image are
        categorically different acts, and a rule engine needs the first and must
        never be handed the second by default. ``purpose`` is mandatory because
        the platform requires a declared purpose for evidence access
        (12_SECURITY §5.4), and a reviewer resolving a crop is making a request
        the audit trail should record.
        """
        return self._api.get_evidence(self._principal, evidence_id, purpose=purpose)

    def authorized_actions(self) -> tuple[Action, ...]:
        """The read actions this reader is built to perform.

        Stated rather than assumed so a deployment can check its grants at
        startup. Evidence is absent: a reader that needs it asks for it
        explicitly, per call, with a purpose.
        """
        return (Action.READ_STATE, Action.READ_CAPABILITY, Action.READ_COVERAGE)


def subjects_by_camera(
    views: Sequence[ObjectView],
) -> dict[str, tuple[ObjectView, ...]]:
    """Group subjects by camera, in stable id order.

    Presentation helper for a frame-by-frame view. Stable order because a UI that
    renumbered *"Employee #1"* and *"Employee #2"* between two reads of the same
    frame would make its own labels untrustworthy.
    """
    grouped: dict[str, list[ObjectView]] = {}
    for view in sorted(views, key=lambda v: str(v.object_id)):
        grouped.setdefault(str(view.camera_id), []).append(view)
    return {camera: tuple(items) for camera, items in sorted(grouped.items())}


__all__ = ["ObservationReader", "ObservationSnapshot", "subjects_by_camera"]
