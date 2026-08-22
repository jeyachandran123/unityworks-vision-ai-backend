"""The M9 → M7 write-back sink.

This is the seam Phase 6 spent nine sub-phases and five discarded hypotheses
finding. It is worth restating exactly what went wrong, because the shape of this
module is a direct response to it:

* Understanding produced 308 attributes. M7 refused **308 of 308**, because the
  two layers held *different* `AttributeRegistry` instances.
* Nothing downstream could tell. The understanding layer reported zero failures,
  the sink reported zero failures, and the platform re-asked the VLM for an
  answer it already held, on every frame, forever.
* The instrumentation that was supposed to catch it counted **attempts before the
  call**, and a bare ``except Exception: continue`` swallowed every rejection —
  so the diagnosis said "391 write-backs applied" when the true number was zero.

So this sink obeys three rules:

**Count after the call, never before.** An attempt is not an application.

**Never swallow an exception without recording its type.** Every rejection is
counted by its exception class. `except: continue` is what made a real defect
invisible for four sub-phases.

**Refuse to write anything a result did not actually establish.** A failed
outcome writes nothing. A missing object id writes nothing. An unregistered
attribute is rejected by M7 and stays rejected — this module never "repairs" it
by registering it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass(slots=True)
class WriteBackAudit:
    """What actually happened between M9 and M7.

    Every field is incremented **after** the operation it describes. The
    difference between `writeback_attempts` and `writebacks_applied` is the
    number this sink exists to make visible.
    """

    results_produced: int = 0
    results_failed: int = 0
    attributes_produced: int = 0
    writeback_attempts: int = 0
    writebacks_applied: int = 0
    writebacks_rejected: int = 0
    no_object_id: int = 0
    failed_outcome: int = 0
    sink_failures: int = 0
    #: Rejections by exception class. The field whose absence made the Phase 6.7
    #: measurement confidently wrong.
    rejection_kinds: Counter[str] = field(default_factory=Counter)
    #: A bounded sample of rejection messages, for a human reading DevTools.
    rejection_samples: list[str] = field(default_factory=list)

    @property
    def applied_rate(self) -> float:
        if self.writeback_attempts == 0:
            return 0.0
        return self.writebacks_applied / self.writeback_attempts

    def to_wire(self) -> dict[str, Any]:
        return {
            "results_produced": self.results_produced,
            "results_failed": self.results_failed,
            "attributes_produced": self.attributes_produced,
            "writeback_attempts": self.writeback_attempts,
            "writebacks_applied": self.writebacks_applied,
            "writebacks_rejected": self.writebacks_rejected,
            "no_object_id": self.no_object_id,
            "failed_outcome": self.failed_outcome,
            "sink_failures": self.sink_failures,
            "applied_rate": round(self.applied_rate, 4),
            "rejection_kinds": dict(self.rejection_kinds),
            "rejection_samples": list(self.rejection_samples[:5]),
        }


class RegistryWriteBackSink:
    """Holds understanding results in M7, through M7's own API.

    Callable, because that is the shape `build_understanding_layer` expects for
    `understanding_sink`.

    **It calls `RegistryEngine.apply_attribute`.** It never touches
    `VisualObject.attributes` directly — the registry's validation *is* the
    Semantic Ceiling at this seam, and bypassing it would let a model write a
    vocabulary no policy granted.
    """

    __slots__ = ("_audit", "_registry_engine")

    def __init__(self, registry_engine: Any) -> None:
        self._registry_engine = registry_engine
        self._audit = WriteBackAudit()

    @property
    def audit(self) -> WriteBackAudit:
        return self._audit

    def __call__(self, results: Any) -> None:
        self.emit(results)

    def emit(self, results: Any) -> None:
        """Apply every successful attribute from a batch of results."""
        batch = results if isinstance(results, list | tuple) else (results,)
        for result in batch:
            self._apply_one(result)

    def _apply_one(self, result: Any) -> None:
        self._audit.results_produced += 1

        # A failed understanding writes nothing. Not a partial value, not a
        # default — nothing. The platform's UNKNOWN is what an absent attribute
        # already means, and it is the honest answer.
        if not _succeeded(result):
            self._audit.results_failed += 1
            self._audit.failed_outcome += 1
            return

        object_id = getattr(result, "object_id", None)
        if object_id is None:
            # An attribute with no subject cannot be held against anything.
            self._audit.no_object_id += 1
            return

        for attribute in getattr(result, "attributes", ()) or ():
            self._audit.attributes_produced += 1
            self._write(object_id, attribute)

    def _write(self, object_id: Any, attribute: Any) -> None:
        try:
            self._registry_engine.apply_attribute(object_id, attribute)
        except Exception as exc:  # noqa: BLE001 - classified, never swallowed
            # The whole point. `except: continue` here is what made a total
            # write-back failure look like a total success.
            kind = type(exc).__name__
            self._audit.writeback_attempts += 1
            self._audit.writebacks_rejected += 1
            self._audit.rejection_kinds[kind] += 1
            if len(self._audit.rejection_samples) < 20:
                self._audit.rejection_samples.append(f"{kind}: {exc}")
            logger.warning(
                "M7 rejected attribute {} for {}: {}: {}",
                getattr(attribute, "key", "?"),
                object_id,
                kind,
                exc,
            )
            return

        # Counted only now, after the call returned. Phase 6.5 counted here-ish
        # and got it wrong by counting before.
        self._audit.writeback_attempts += 1
        self._audit.writebacks_applied += 1


def _succeeded(result: Any) -> bool:
    """Whether a result carries a usable answer.

    `UnderstandingResult.outcome` is an `UnderstandingOutcome`, and **only
    `SUCCEEDED` may be written**. The other members are all reasons not to:

        NO_ATTRIBUTES   the model answered and declared nothing
        REFUSED         the model declined
        TIMED_OUT       no answer arrived
        UNAVAILABLE     the provider could not be reached
        UNSUPPORTED     the adapter cannot produce what was asked for

    Every one of those must leave the attribute absent, which the platform
    already reads as UNKNOWN. Writing a value for any of them would convert
    "we do not know" into a fact — the exact conversion the whole three-valued
    design exists to prevent.

    An outcome this function cannot read returns **False**. A sink that guesses
    optimistically writes failures.
    """
    outcome = getattr(result, "outcome", None)
    if outcome is None:
        return False
    return str(getattr(outcome, "value", outcome)).lower() == "succeeded"


__all__ = ["RegistryWriteBackSink", "WriteBackAudit"]
