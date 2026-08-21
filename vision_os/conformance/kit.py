"""The conformance kit framework (06_PORTS_AND_ADAPTERS §5).

An interface constrains the shape of a call, not the meaning of its result. Two
detectors can implement the same interface perfectly and still break the platform
when swapped — different coordinate conventions, different NMS behaviour,
different treatment of "nothing found". So a port in this platform is three
things::

    Port = Interface  +  Semantic Contract  +  Conformance Kit
           (shape)       (meaning)             (executable proof)

**The kit is what converts invariant V3 from an aspiration into a gate.** The
Plugin Manager runs the fast subset before activating any adapter, so a mis-built
adapter is rejected at boot rather than in production data.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.model.ids import PortId


class KitSection(enum.Enum):
    """The five sections every kit has (06_PORTS §5.1)."""

    SHAPE = "shape"
    """Interface compliance, types, batch mapping."""

    SEMANTICS = "semantics"
    """The port's numbered obligations."""

    GOLDEN = "golden"
    """Correctness against a fixed reference corpus."""

    FAILURE = "failure"
    """Error behaviour under injected faults."""

    RESOURCE = "resource"
    """Declared resources are truthful."""

    @property
    def in_fast_subset(self) -> bool:
        """Sections run at plugin load.

        Deliberate: the fast subset costs seconds and catches the catastrophic
        class — coordinate conventions, vocabulary leakage, fabrication on
        failure — before a single real frame is processed.
        """
        return self in (KitSection.SHAPE, KitSection.SEMANTICS, KitSection.FAILURE)


@dataclass(frozen=True, slots=True)
class ConformanceCheck:
    """One executable obligation.

    ``run`` raises to fail. The obligation reference (``D1``, ``U2``, ...) ties
    each check back to the numbered semantic contract in the architecture.
    """

    name: str
    section: KitSection
    run: Callable[[Any], None]
    obligation: str = ""

    def execute(self, adapter: Any) -> str | None:
        """Returns ``None`` on pass, or a failure description."""
        try:
            self.run(adapter)
        except AssertionError as exc:
            return f"{self.qualified_name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - any raise is a failure
            return f"{self.qualified_name}: unexpected {type(exc).__name__}: {exc}"
        return None

    @property
    def qualified_name(self) -> str:
        prefix = f"[{self.obligation}] " if self.obligation else ""
        return f"{prefix}{self.section.value}/{self.name}"


@dataclass(frozen=True, slots=True)
class ConformanceReport:
    port_id: PortId
    kit_version: str
    passed: bool
    executed: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    fast_subset_only: bool = False

    def summary(self) -> str:
        state = "PASS" if self.passed else "FAIL"
        scope = "fast" if self.fast_subset_only else "full"
        return (
            f"{state} {self.port_id} kit@{self.kit_version} ({scope}): "
            f"{len(self.executed)} run, {len(self.failures)} failed, "
            f"{len(self.skipped)} skipped"
        )


@dataclass(frozen=True, slots=True)
class ConformanceKit:
    """An executable suite an adapter must pass before activation."""

    port_id: PortId
    version: str
    checks: tuple[ConformanceCheck, ...] = ()

    def run(self, adapter: Any, *, fast_only: bool = False) -> ConformanceReport:
        executed: list[str] = []
        failures: list[str] = []
        skipped: list[str] = []

        for check in self.checks:
            if fast_only and not check.section.in_fast_subset:
                skipped.append(check.qualified_name)
                continue
            executed.append(check.qualified_name)
            failure = check.execute(adapter)
            if failure is not None:
                failures.append(failure)

        return ConformanceReport(
            port_id=self.port_id,
            kit_version=self.version,
            passed=not failures,
            executed=tuple(executed),
            failures=tuple(failures),
            skipped=tuple(skipped),
            fast_subset_only=fast_only,
        )

    def sections_covered(self) -> frozenset[KitSection]:
        return frozenset(check.section for check in self.checks)


@dataclass(slots=True)
class ConformanceRegistry:
    """Maps a port to its kit. Injected into the Plugin Manager."""

    kits: dict[PortId, ConformanceKit] = field(default_factory=dict)

    def register(self, kit: ConformanceKit) -> None:
        self.kits[kit.port_id] = kit

    def get(self, port_id: PortId) -> ConformanceKit | None:
        return self.kits.get(port_id)

    def __contains__(self, port_id: object) -> bool:
        return port_id in self.kits
