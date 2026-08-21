"""Vision OS error taxonomy (10_RELIABILITY_AND_FAILURE §2).

Every failure in the platform carries a **classification**, because the
classification — not the call site — determines the response. An unclassified
failure gets a guessed response, and guessed responses are how retry storms and
crash loops begin.

Adapters raise ``VisionOSError`` subclasses, or — where failure is an *expected*
outcome rather than an exceptional one — return a typed verdict carrying an
attributed reason (``AdmissionVerdict``, ``MaskOutcome``). They never raise bare
exceptions across a port boundary, and they never fabricate a plausible value on
failure (invariant V4, port obligation A4).
"""

from __future__ import annotations

import enum
from typing import Any


class FailureClass(enum.Enum):
    """10_RELIABILITY_AND_FAILURE §2 — the six failure classes."""

    TRANSIENT = "transient"
    """Self-resolving; a bounded retry is likely to succeed."""

    PERSISTENT = "persistent"
    """Will not self-resolve; retry is futile. Stop, fall back, alarm."""

    POISON = "poison"
    """A specific input reliably fails. Quarantine the input, keep the stream."""

    SYSTEMIC = "systemic"
    """A shared resource is affected; retry makes it worse. Shed load."""

    SILENT = "silent"
    """No error raised, but output is wrong or absent. Requires active detection."""

    BYZANTINE = "byzantine"
    """Confident, plausible, wrong output. Requires cross-checks and evidence."""


class VisionOSError(Exception):
    """Base class for every Vision OS failure.

    Carries a failure classification and structured context so that recovery
    logic never has to parse a message string.
    """

    failure_class: FailureClass = FailureClass.PERSISTENT

    def __init__(self, message: str, /, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context)

    @property
    def retryable(self) -> bool:
        """Explicit, never inferred by the caller (09_API_CONTRACTS §8)."""
        return self.failure_class is FailureClass.TRANSIENT

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"{type(self).__name__}({self.message!r}, class={self.failure_class.value})"


# --- configuration ------------------------------------------------------- #


class ConfigurationError(VisionOSError):
    """Configuration is invalid, missing, or internally inconsistent."""


class ValidationError(ConfigurationError):
    """A candidate configuration failed schema validation."""

    def __init__(self, message: str, /, violations: tuple[str, ...] = (), **context: Any) -> None:
        super().__init__(message, **context)
        self.violations = violations


class SecretResolutionError(ConfigurationError):
    """A secret reference could not be resolved. Never contains the secret."""


# --- plugins and ports --------------------------------------------------- #


class PluginError(VisionOSError):
    """A plugin could not be discovered, validated, loaded, or activated."""


class ManifestInvalidError(PluginError):
    """A plugin manifest is malformed or incomplete."""


class PortIncompatibleError(PluginError):
    """A plugin declares a port version outside the platform's supported range."""


class ConformanceFailedError(PluginError):
    """A plugin failed its port conformance kit and must not be activated (V3)."""

    def __init__(self, message: str, /, failures: tuple[str, ...] = (), **context: Any) -> None:
        super().__init__(message, **context)
        self.failures = failures


class SignatureInvalidError(PluginError):
    """Plugin signature verification failed. Fails closed (12_SECURITY §6)."""


# --- acquisition --------------------------------------------------------- #


class AcquisitionError(VisionOSError):
    """Base for the L1 acquisition layer."""


class ConnectFailedError(AcquisitionError):
    """A source could not be opened."""

    failure_class = FailureClass.TRANSIENT


class StreamLostError(AcquisitionError):
    """An open source stopped delivering frames."""

    failure_class = FailureClass.TRANSIENT


class DecodeError(AcquisitionError):
    """A frame could not be decoded."""

    failure_class = FailureClass.TRANSIENT


class UnsupportedCodecError(AcquisitionError):
    """No decoder can handle this stream. Fail provisioning loudly."""


class PrivacyMaskError(AcquisitionError):
    """Privacy masking failed.

    The **only** fail-closed path in the platform (12_SECURITY §2.1): the frame
    is dropped rather than emitted unmasked. A masking failure that proceeds is a
    compliance incident regardless of intent.
    """

    failure_class = FailureClass.SYSTEMIC


class NotSeekableError(AcquisitionError):
    """Seek requested on a source that does not support it."""


# --- buffering ----------------------------------------------------------- #


class BufferError(VisionOSError):
    """Base for frame buffer failures."""


class PoolExhaustedError(BufferError):
    """No buffer capacity is available."""

    failure_class = FailureClass.SYSTEMIC


class FrameUnavailableError(BufferError):
    """The requested frame is no longer resident (evicted or never published).

    A normal, expected outcome — not a bug. Callers degrade and count.
    """

    failure_class = FailureClass.TRANSIENT


class LeaseExpiredError(BufferError):
    """A lease was force-broken after exceeding its deadline."""


# --- registry / lookup --------------------------------------------------- #


class NotFoundError(VisionOSError):
    """A referenced entity does not exist."""


class UncalibratedError(VisionOSError):
    """An operation requiring calibration was attempted without one.

    Callers degrade to normalized image space rather than failing (V9).
    """


class CapacityExceededError(VisionOSError):
    """The runtime cannot accept more work of this kind."""

    failure_class = FailureClass.SYSTEMIC


class LifecycleError(VisionOSError):
    """An operation was attempted in an invalid lifecycle state."""


# --- models and devices (Flow 2) ----------------------------------------- #


class ModelError(VisionOSError):
    """Base for model artifact, runtime, and device failures."""


class ArtifactUnavailableError(ModelError):
    """An artifact could not be fetched. Transient; retry or use a cached copy."""

    failure_class = FailureClass.TRANSIENT


class ArtifactIntegrityError(ModelError):
    """An artifact's content hash did not match its declaration.

    A **supply-chain event**, not a network glitch. Fails closed and is never
    retried into success (12_SECURITY section 6).
    """


class ModelLoadError(ModelError):
    """A model artifact could not be loaded: corrupt, incompatible, unsupported."""


class ModelUnavailableError(ModelError):
    """No usable version of the requested model is resident or loadable."""

    failure_class = FailureClass.TRANSIENT


class DeviceUnavailableError(ModelError):
    """A required device is absent or has disappeared."""

    failure_class = FailureClass.TRANSIENT


class DeviceOutOfMemoryError(ModelError):
    """A device could not satisfy a memory reservation.

    Systemic: retrying unchanged makes it worse. The broker evicts by policy,
    retries once, then denies with a stated reason.
    """

    failure_class = FailureClass.SYSTEMIC


class LicenceViolationError(ModelError):
    """A model's licence forbids this deployment context.

    Checked at registration, never discovered in production.
    """


# --- detection (Flow 2) --------------------------------------------------- #


class DetectionError(VisionOSError):
    """Base for the detection layer."""


class DetectionFailedError(DetectionError):
    """A detector could not produce a result for a frame.

    Distinct from an empty result, which is a **valid, non-error** outcome
    (port obligation D5). Conflating the two is how "nothing was there" and
    "we could not look" become indistinguishable.
    """

    failure_class = FailureClass.TRANSIENT


class DetectionTimeoutError(DetectionError):
    """Inference exceeded its budget."""

    failure_class = FailureClass.TRANSIENT


class DetectorContractError(DetectionError):
    """An adapter violated its port contract at runtime.

    Byzantine: the adapter returned something structurally plausible but wrong —
    a mismatched batch length, a native label, a box outside normalized space.
    Detected and rejected rather than propagated.
    """

    failure_class = FailureClass.BYZANTINE


class DetectionQueueFullError(DetectionError):
    """The bounded inference queue is at capacity.

    Shedding rather than growing: an unbounded inference queue is a memory leak
    with a delayed fuse, and a frame that waits forever is worse than one dropped
    with an attributed reason (invariant V8).
    """

    failure_class = FailureClass.SYSTEMIC


class TaxonomyError(VisionOSError):
    """A taxonomy class or mapping is unknown, malformed, or inconsistent."""


# --- tracking (Flow 3) ---------------------------------------------------- #


class TrackingError(VisionOSError):
    """Base for the tracking layer."""


class TrackingFailedError(TrackingError):
    """A tracker could not process a frame.

    Distinct from an update that produced no tracks, which is valid. Conflating
    them makes "the tracker broke" and "nothing is in view" the same fact.
    """

    failure_class = FailureClass.TRANSIENT


class OutOfOrderFrameError(TrackingError):
    """A frame arrived before one already processed for this camera.

    **Loud on purpose** (03_MODULES M6, port obligation T1). Out-of-order frames
    corrupt tracking silently: the motion model integrates a negative time step,
    positions run backwards, and associations degrade in a way that looks like
    poor tracker quality rather than a pipeline bug. The architecture requires
    this be rejected and alarmed rather than absorbed.
    """

    failure_class = FailureClass.BYZANTINE


class IllegalTransitionError(TrackingError):
    """A track was asked to move between states that do not connect.

    Byzantine rather than transient: the lifecycle is a closed machine, so this
    can only mean a tracker or an adapter is constructing state incorrectly.
    """

    failure_class = FailureClass.BYZANTINE


class TrackerContractError(TrackingError):
    """An adapter violated its port contract at runtime.

    Reused track ids within an epoch, cross-camera state leakage, a predicted
    position presented as measured — structurally plausible, semantically wrong.
    """

    failure_class = FailureClass.BYZANTINE


class TrackerCapacityError(TrackingError):
    """The tracker's bounded track table is full.

    Bounded by design (port obligation T8): a crowd scene must degrade by
    refusing new tracks, never by growing without limit.
    """

    failure_class = FailureClass.SYSTEMIC


class RegistryError(VisionOSError):
    """Base for the Object Registry (M7)."""


class ObjectNotFoundError(RegistryError):
    """No object with this id exists in the addressed partition.

    Distinct from a merged object, which *is* found and reports where it went —
    history stays resolvable through ``merged_into`` (V5).
    """

    failure_class = FailureClass.PERSISTENT


class RegistryCapacityError(RegistryError):
    """The per-camera object population is at its cap.

    Bounded by design (03_MODULES M7): a runaway registry is a memory leak with
    a face. Provisional objects are shed first; when none remain, new objects
    are refused and the condition is alarmed rather than absorbed.
    """

    failure_class = FailureClass.SYSTEMIC


class OwnershipViolationError(RegistryError):
    """Something other than the owning partition tried to write an object.

    Byzantine rather than persistent: the registry is the sole writer by
    construction, so reaching this means a caller has bypassed the public API.
    """

    failure_class = FailureClass.BYZANTINE


class IdentityConflictError(RegistryError):
    """A merge or split was requested that would corrupt history.

    Merging an object into itself, merging across camera partitions
    synchronously, or splitting at a time outside an object's lifetime. Refused
    rather than repaired: silently fixing an identity error produces a plausible
    object nobody can trace.
    """

    failure_class = FailureClass.BYZANTINE


class AttributeRejectedError(RegistryError):
    """An attribute failed the neutrality gate (02_VOM section 9.1).

    The registry gate that operationalizes the Semantic Ceiling. ``is_employee``,
    ``is_compliant`` and ``wait_time_excessive`` are rejected here, and the
    rejection names the neutral counterpart to register instead.
    """

    failure_class = FailureClass.PERSISTENT


class ObjectStoreError(RegistryError):
    """Durable object state could not be read or written.

    Transient: the hot path never blocks on persistence, so a store failure
    degrades durability without stopping ingestion.
    """

    failure_class = FailureClass.TRANSIENT


class CropError(VisionOSError):
    """Base for the Crop Manager (M8)."""


class GateRejectedError(CropError):
    """A crop was refused by the quality gate.

    **Not a failure.** Refusing to send a hopeless crop to an expensive model is
    the gate working; the caller records the skip and retries when
    ``QUALITY_IMPROVED`` fires (03_MODULES M8 failure handling).
    """

    failure_class = FailureClass.TRANSIENT


class BudgetExhaustedError(CropError):
    """The understanding budget is spent for this window.

    Systemic: retrying makes it worse. The caller sheds by priority and publishes
    coverage so consumers know attributes are thinned rather than absent (V8).
    """

    failure_class = FailureClass.SYSTEMIC


class CropExtractionError(CropError):
    """Pixels could not be turned into a crop.

    Distinct from a gate rejection, which is a *judgment about quality*, and from
    ``FrameUnavailableError``, which is an *eviction*. This is a genuine fault in
    the extraction path.
    """

    failure_class = FailureClass.TRANSIENT


class DemandRejectedError(CropError):
    """A demand cannot be registered.

    The outermost ring of Semantic Ceiling enforcement: a demand naming an
    unregistered attribute is refused at registration with a pointer to the
    registration process, rather than accepted and silently never served
    (09_API_CONTRACTS section 4.2).
    """

    failure_class = FailureClass.PERSISTENT


class DemandNotFoundError(CropError):
    """No demand with this id is registered."""

    failure_class = FailureClass.PERSISTENT


# --- understanding (M9) --------------------------------------------------- #


class UnderstandingError(VisionOSError):
    """Base for the Understanding Engine (M9).

    Every subclass answers *"why did no attribute arrive"* with something a
    consumer can act on. 04_MODULES section M9 states the governing property:
    **understanding failure is never pipeline failure** — detection, tracking,
    identity and spatial observations continue unaffected, and only enrichment is
    lost.
    """


class UnderstandingFailedError(UnderstandingError):
    """The request could not be completed after retries and fallbacks.

    Carries the outcome so the caller records the right skip rather than
    inventing one. **Never** carries a value: a failed understanding that produced
    an attribute would be fabrication, which port obligation U2 calls *"the single
    most dangerous failure mode for a VLM-based system"*.
    """

    failure_class = FailureClass.TRANSIENT


class UnderstanderTimeoutError(UnderstandingError):
    """A model call exceeded its deadline.

    Transient by classification: 10_RELIABILITY section 4.3 retries once with
    backoff, then falls back, then gives up. The object simply has no attribute
    this round, and the skip is recorded.
    """

    failure_class = FailureClass.TRANSIENT


class UnderstanderUnavailableError(UnderstandingError):
    """No understander could be reached — circuit open or chain exhausted.

    Systemic: retrying makes it worse. 10_RELIABILITY section 7.2 requires the
    last link in a fallback chain to be **explicit unavailability, never a
    guess**, and this is that link.
    """

    failure_class = FailureClass.SYSTEMIC


class UnsupportedCapabilityError(UnderstandingError):
    """No bound understander declares the requested attributes.

    A **capability gap**, not a fault. The honest answer to a demand the platform
    cannot serve, so the consumer stops waiting for data that will never arrive
    (invariant V8).
    """

    failure_class = FailureClass.PERSISTENT


class OutputCoercionError(UnderstandingError):
    """Model output could not be parsed into the declared schema.

    Classified **POISON**. 04_MODULES section M9's table labels this *"Data"*,
    which is not one of the six canonical classes in 10_RELIABILITY section 2;
    ``POISON`` is the class whose documented response is exactly what M9 does —
    *"quarantine the input, continue the stream"*. The output goes to
    ``unstructured_note`` and **zero** attributes are emitted (02_VOM section
    9.3), while every other camera and every other object keeps working.
    """

    failure_class = FailureClass.POISON


class SchemaViolationError(UnderstandingError):
    """A field violates the attribute schema it claims to satisfy.

    Distinct from ``AttributeRejectedError``, which refuses an unregistered
    *key*. This refuses a registered key carrying an illegal *value* — a
    different operator response, because one means a prompt drifted and the other
    means a model formats badly.
    """

    failure_class = FailureClass.POISON


class PromptUnavailableError(UnderstandingError):
    """No prompt exists for the requested attribute set.

    04_MODULES section M10 returns ``NoSuitablePrompt`` here and M8 records a
    capability gap, so the consumer learns the demand is unsatisfiable rather
    than waiting on an answer nobody can render.
    """

    failure_class = FailureClass.PERSISTENT


# --- synthesis and state (M11, M12) --------------------------------------- #


class ObservationError(VisionOSError):
    """Base for the Observation Builder (M11)."""


class ValidationFailedError(ObservationError):
    """An observation was refused by the final semantic gate.

    04_MODULES section M11 is precise about which failures kill the envelope:
    missing provenance, timing or evidence. *"An unexplainable observation is
    worse than no observation — it is a fact nobody can audit (V4)."*

    An unregistered *attribute* does **not** raise this: the attribute is
    dropped, the observation survives, and the rejection is counted. Two
    failures, two responses.
    """

    failure_class = FailureClass.PERSISTENT


class TaxonomyMismatchError(ObservationError):
    """Producers disagree about the taxonomy version.

    Section M11: *"Reject with a clear diagnostic; this indicates a partial
    deployment and must be loud."* Two producers on different taxonomy versions
    are describing different worlds, and merging their output silently would
    produce observations nobody can interpret consistently.
    """

    failure_class = FailureClass.PERSISTENT


class StateError(VisionOSError):
    """Base for the Vision State Manager (M12)."""


class CommitFailedError(StateError):
    """Observations could not be appended to the log.

    10_RELIABILITY section 4.4 step 4: when the local buffer fills, the partition
    **stops accepting observations and is marked degraded** rather than dropping
    facts silently. *"Losing observations invisibly is a V8 violation of the
    worst kind."*
    """

    failure_class = FailureClass.SYSTEMIC


class LogUnavailableError(StateError):
    """The observation log cannot be reached.

    Systemic: the log is the system of record, and retrying against a dead store
    makes it worse. The partition degrades loudly.
    """

    failure_class = FailureClass.SYSTEMIC


class ProjectionError(StateError):
    """One observation could not be projected into state.

    Section M12: *"Quarantine that observation, continue the projection, alarm.
    One bad record must not stop the world."* Classified ``POISON`` because that
    is exactly the documented response — quarantine the input, keep the stream.
    """

    failure_class = FailureClass.POISON


class PartitionDegradedError(StateError):
    """The partition has stopped accepting observations.

    Raised on an attempt to append to a halted partition, so a caller learns the
    facts are not being recorded rather than believing they were.
    """

    failure_class = FailureClass.SYSTEMIC


class StateNotFoundError(NotFoundError):
    """No state exists for the requested object or partition.

    Distinct from a partition that exists and is empty: 07_STATE section 7.1 is
    built around a consumer being able to tell *"nothing there"* from *"we cannot
    see"*.
    """


class EmbeddingUnavailableError(TrackingError):
    """A tracker requiring appearance embeddings has no provider configured.

    Fails loudly rather than degrading to geometry, because a silent degradation
    would make a capability gap invisible to the consumer (invariant V8). Note
    that no embedding provider ships: appearance embeddings are C2 biometric
    data and are disabled by default (12_SECURITY section 4).
    """

    failure_class = FailureClass.PERSISTENT


# --- persistence (M13, Flow 8) -------------------------------------------- #


class PersistenceError(VisionOSError):
    """A storage contract failed.

    Named for the contract rather than the technology: §M13's whole purpose is
    that an edge box and a cloud cluster *"differ only in adapter selection"*, so
    a caller must be able to handle a storage failure without knowing whether the
    bytes were going to a disk or an object store.
    """


class EvidenceStoreError(PersistenceError):
    """An evidence write or read failed for a reason that is not absence.

    Absence is never an error: ``EvidenceStore.get`` reports ``NotFound``,
    ``Expired`` and ``Erased`` as *answers* (§M13), because collapsing them into
    an exception would make retention indistinguishable from data loss.
    """


class EvidenceQuotaExceededError(EvidenceStoreError):
    """The store is at its declared bound.

    Systemic rather than persistent: retrying makes it worse, and the correct
    response is to shed or to run retention, not to try again.
    """

    failure_class = FailureClass.SYSTEMIC


class ErasureFailedError(PersistenceError):
    """An erasure request could not be completed.

    Loud by necessity. A regulated deployment that believes an erasure succeeded
    when it did not has a compliance failure it cannot see, which is worse than
    an erasure that visibly failed and can be retried.
    """


# --- exposure (M14, Flow 8) ------------------------------------------------ #


class ApiError(VisionOSError):
    """Base for every failure a consumer can observe (09_API §8).

    Every subclass carries a stable ``code``. §8: *"Consumers must never infer
    retryability from a status code or a message string. Inferring it is how
    retry storms begin."* — so ``retryable`` comes from ``failure_class`` and is
    reported explicitly, never derived by the caller.
    """

    code: str = "INTERNAL"


class UnauthenticatedError(ApiError):
    """No valid credential was presented."""

    code = "UNAUTHENTICATED"


class ForbiddenError(ApiError):
    """Authenticated, but not permitted this action on this scope.

    12_SECURITY §5.3 separates ``read_observations`` from ``read_evidence``
    deliberately: *"Reading 'a person was here' and viewing their image are
    categorically different acts."*
    """

    code = "FORBIDDEN"


class TenantScopeViolationError(ForbiddenError):
    """A request reached outside its tenant.

    Always audited and alarmed. 12_SECURITY §4.2 requires scope be applied at
    query construction, so reaching this error means a code path constructed an
    unscoped query — a defect, not a permission problem.
    """

    code = "TENANT_SCOPE_VIOLATION"


class InvalidScopeError(ApiError):
    """The requested scope is malformed or names nothing."""

    code = "INVALID_SCOPE"


class WindowTooLargeError(ApiError):
    """The query window exceeds policy.

    09_API §M14: *"Reject with a bound and a cursor rather than degrading the
    service for everyone."*
    """

    code = "WINDOW_TOO_LARGE"


class UnsupportedVersionError(ApiError):
    """The consumer asked for a schema version this build does not serve.

    §M14: *"Reject with the supported set; never guess."* Guessing would serve a
    payload shaped for a contract the consumer does not implement.
    """

    code = "UNSUPPORTED_VERSION"


class PartitionUnavailableError(ApiError):
    """A partition in scope could not be read.

    Transient: 09_API §8 classes availability failures as retryable, and a
    consumer that backs off will usually succeed.
    """

    code = "PARTITION_UNAVAILABLE"
    failure_class = FailureClass.TRANSIENT


class OverloadedError(ApiError):
    """The API is shedding load.

    §M14: *"Shed load by priority; protect subscriptions over ad-hoc queries,
    because subscriptions are how live consumers stay correct."*
    """

    code = "OVERLOADED"
    failure_class = FailureClass.SYSTEMIC


class EvidenceExpiredError(ApiError):
    """The evidence existed and retention removed it.

    A distinct code from ``NOT_FOUND`` all the way to the consumer, because §M13
    refuses to let retention look like data loss at any layer.
    """

    code = "EXPIRED"


class SubscriptionError(ApiError):
    """A subscription could not be established or was terminated."""

    code = "SUBSCRIPTION_FAILED"
