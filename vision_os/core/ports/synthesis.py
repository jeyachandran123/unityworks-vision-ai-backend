"""P18 ``SuppressionPolicyPort``, P19 ``ObservationSinkPort``, P20 ``ObservationLogPort``.

P18 and P19 belong to M11; P20 is one of M13's five storage contracts. M13's own
single responsibility is *"Describe what must persist and with what guarantees;
**implement none of it**"* — it owns no state and is *"a set of contracts"*. So
declaring the contract here and shipping adapters behind it is not implementing
M13; it is realizing a contract M13 already defines, exactly as Flow 2 did for
P25–P27 and Flow 4 for P21.

**Why suppression is a port and not a constant.** 04_MODULES §M11 calls change
suppression *"the main performance feature, and it is a correctness feature
too"*: without it a stationary object publishes an identical observation at full
frame rate forever, flooding storage and consumers with no information. But *what
counts as a change* is a deployment decision — exact match for a forensic mode,
positional threshold for a busy retail floor, semantic for a low-bandwidth edge
link. Hard-coding one would make the others a fork.

**Why the sink is a port.** §M11: *"in addition to state, an observation may be
teed to a message bus, a data lake, or a future learning pipeline. This is the
designed hook by which a training loop is added later without touching the
platform."*

**Why the log is a port.** 07_STATE §2 makes the log the system of record, and
§9.1 makes *"log loss (total)"* a critical incident. A single-node edge box and a
cloud cluster must differ only in which adapter is bound.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..model.ids import CameraId, LogPosition, ObservationId
from ..model.observation import Observation
from ..model.timebase import Duration, Instant

SUPPRESSION_POLICY_PORT_VERSION = "1.0.0"
OBSERVATION_SINK_PORT_VERSION = "1.0.0"
OBSERVATION_LOG_PORT_VERSION = "1.0.0"


# --- P18 SuppressionPolicyPort -------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SuppressionDecision:
    """Whether to publish, and why.

    ``reason`` is mandatory on both branches. A suppressed observation is a
    deliberate non-event, and a deployment tuning its suppression needs to know
    *which* rule fired — otherwise the only visible symptom of a
    misconfigured policy is a quiet platform.
    """

    publish: bool
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError(
                "a suppression decision must carry its reason; a quiet platform "
                "with no explanation is indistinguishable from a broken one"
            )


@runtime_checkable
class SuppressionPolicyPort(Protocol):
    """P18 — decide whether an observation says anything new.

    ### Semantic contract

    | # | Obligation |
    |---|---|
    | **S1** | The **first** observation for a subject always publishes. There is nothing to compare against, and suppressing it would mean an object could exist in the log only implicitly. |
    | **S2** | A **heartbeat** always publishes, regardless of change. §M11: *"a consumer must be able to distinguish 'unchanged' from 'stopped observing'"* (V8). |
    | **S3** | Deterministic: identical previous and candidate yield an identical decision (V13). |
    | **S4** | A **correction** (`supersedes` set) always publishes. A correction that said nothing new would be a correction nobody receives. |
    | **S5** | Never mutates either observation. Suppression is a read-only comparison. |
    | **S6** | Never interprets content. A policy may compare a position, a value or a hash; it may never decide an observation is *unimportant* on business grounds (V1). |
    | **S7** | ``signature`` is deterministic and content-only. Two observations differing solely in id or publication time must produce the same signature, or nothing is ever suppressed. |

    **The policy owns the signature.** The builder retains one opaque string per
    subject and hands it back; what that string *summarises* is the policy's
    business. A threshold policy quantizes position coarsely, an exact policy
    finely, and a semantic policy might hash only the attribute set — none of
    which the builder needs to know. Storing a builder-defined signature would
    force every policy to compare against a summary chosen by something else.
    """

    @property
    def policy_id(self) -> str:
        ...

    def signature(self, observation: Observation) -> str:
        """A content digest, by this policy's definition of *content*.

        Must exclude ``observation_id``, ``t_published`` and timing: those differ
        on every build, and including them would make every observation look
        changed — suppression that never suppresses (obligation S7).
        """
        ...

    def should_publish(
        self,
        candidate: Observation,
        previous_signature: str | None,
        *,
        elapsed: Duration,
        heartbeat: Duration,
    ) -> SuppressionDecision:
        """Decide. ``previous_signature`` is ``None`` for a never-published subject.

        A signature rather than the previous observation, because M11's
        suppression state is deliberately *"small, ephemeral, per-camera"* —
        retaining whole observations would make it as large as the projection.
        """
        ...


# --- P19 ObservationSinkPort ---------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SinkResult:
    """What a sink did with a batch.

    ``accepted`` may be fewer than offered without being an error: a sampling
    data-lake tee legitimately keeps a subset. What it may never be is *silently*
    fewer, which is why the count comes back.
    """

    accepted: int = 0
    rejected: tuple[tuple[ObservationId, str], ...] = ()
    detail: str = ""

    @property
    def complete(self) -> bool:
        return not self.rejected


@runtime_checkable
class ObservationSinkPort(Protocol):
    """P19 — receive published observations.

    Vision State is *a* sink, not *the* sink. A deployment may tee the same
    stream to a message bus and a data lake; a future learning pipeline attaches
    here and nothing in the platform changes.

    ### Semantic contract

    | # | Obligation |
    |---|---|
    | **K1** | **Never mutates an observation.** They are immutable facts (V5), and a sink that altered one would make two consumers disagree about a published fact. |
    | **K2** | Idempotent by ``observation_id``. At-least-once delivery is workable only if a repeat is harmless. |
    | **K3** | Failure is explicit and typed, never a silent partial success. |
    | **K4** | A slow sink must not block the platform. A sink that cannot keep up sheds and says so — one slow consumer may not stall perception. |
    | **K5** | Declares whether it is durable. A tee to a metrics dashboard is not a system of record and must not be mistaken for one. |
    """

    @property
    def sink_id(self) -> str:
        ...

    @property
    def durable(self) -> bool:
        """Whether acceptance implies durability (obligation K5)."""
        ...

    def emit(self, observations: Sequence[Observation]) -> SinkResult:
        """Receive a batch. Never mutates; never raises for a routine refusal."""
        ...


# --- P20 ObservationLogPort ----------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LogAppendResult:
    """The outcome of an append (§M13's ``append``).

    ``duplicates`` is separate from ``appended`` because §M13 makes append
    *"idempotent by `observation_id`, so retry after an uncertain outcome is
    always safe — which is what makes at-least-once delivery workable end to
    end."* A retry reporting duplicates is a success, not a partial failure.
    """

    position: LogPosition
    appended: int = 0
    duplicates: tuple[ObservationId, ...] = ()

    @property
    def total(self) -> int:
        return self.appended + len(self.duplicates)


@runtime_checkable
class ObservationLogPort(Protocol):
    """P20 — the immutable, append-only system of record.

    07_STATE §2: state is a projection *of this*. §9.1 lists the recovery for
    every failure mode, and in each row the log is authoritative — including
    *"projection bug: fix, rebuild into a shadow projection, atomic swap"*, which
    is only possible because the log survives the projection.

    ### Semantic contract

    | # | Obligation |
    |---|---|
    | **L1** | **Append-only.** No update, no delete. ``truncate`` exists for retention alone and removes only a time-bounded prefix. |
    | **L2** | **Idempotent by ``observation_id``.** Appending a known id is a no-op reported as a duplicate, never a second copy and never an error. |
    | **L3** | Positions are **monotonic per partition**, so a projection watermark is meaningful and rebuild is resumable. |
    | **L4** | ``read`` returns observations in append order. Order is the log's contract; a set-like store cannot satisfy it. |
    | **L5** | Failure is a typed result, never a silent partial success. A partial append is reported as such. |
    | **L6** | Partitions are independent. An append to one camera never blocks or affects another (07_STATE §4). |
    | **L7** | ``tail`` yields everything from a position onward and **never blocks on an empty partition**. A follow that hung on a quiet camera would make a subscription's liveness depend on the scene having something in it. |

    ### Why ``tail`` is separate from ``read``

    §M13's Public API lists both, and they answer different questions. ``read``
    is a *range* — bounded at both ends, used by rebuild and by historical
    queries. ``tail`` is a *follow* — bounded only at the start, used by
    subscription resumption. Serving a follow through ``read`` would force the
    caller to poll with a moving upper bound and to guess how far ahead the log
    had advanced.
    """

    @property
    def log_id(self) -> str:
        ...

    def append(
        self, partition: CameraId, observations: Sequence[Observation]
    ) -> LogAppendResult:
        """Append durably. Idempotent by ``observation_id`` (L2)."""
        ...

    def read(
        self,
        partition: CameraId,
        *,
        start: LogPosition | None = None,
        end: LogPosition | None = None,
        limit: int = 1000,
    ) -> Iterator[Observation]:
        """Range-read in append order (L4)."""
        ...

    def tail(
        self, partition: CameraId, *, start: LogPosition | None = None, limit: int = 1000
    ) -> Iterator[Observation]:
        """Live follow from a position onward (L7).

        §M13 specifies this alongside ``read`` because a subscription resuming
        from a cursor needs everything *since* that point without knowing where
        the log now ends. Returns immediately when there is nothing new — an
        empty iterator, never a block, because a camera watching an empty
        corridor is a normal state and not a reason to stall a subscriber.
        """
        ...

    def position(self, partition: CameraId) -> LogPosition:
        """The partition's current tail."""
        ...

    def truncate(self, partition: CameraId, before: Instant) -> int:
        """Retention only. Returns how many records were removed.

        Never a correctness operation: 07_STATE §8.2 distinguishes *tombstoning*
        from *rewriting*, and rewriting history *"would destroy the property that
        makes the log trustworthy in the first place."*
        """
        ...
