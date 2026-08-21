"""Plugin manifests and the port registry (05_KERNEL M17, 06_PORTS §2).

A manifest is a plugin's declaration of what it is, what it implements, what it
needs, and what it can produce. Declaring capabilities **honestly** is adapter
obligation A1: it is what lets the platform report a capability gap immediately
rather than leaving a consumer waiting forever for data that will never arrive.

The port catalogue is closed in the same spirit as the object ontology: adding a
port is a deliberate, reviewed act, not something a plugin can do by asserting a
new name.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from ...core.model.ids import PluginId, PortId


class IsolationLevel(enum.Enum):
    """How a plugin is invoked (05_KERNEL M17 performance).

    The same plugin moves between levels **by configuration alone**, which is
    what allows a detector to run in-process on an edge box and on a remote
    inference server in a cluster with no code difference.
    """

    IN_PROCESS = "in_process"
    SUBPROCESS = "subprocess"
    REMOTE = "remote"


class PortCatalogue:
    """The 32 ports of 06_PORTS_AND_ADAPTERS §2.

    All are named so that manifests referencing a later-flow port fail
    validation with a clear message rather than a confusing lookup error. Only
    the Flow 1 subset is *bindable*; see ``FLOW1_PORTS``.
    """

    SOURCE = PortId("P1.SourcePort")
    DECODER = PortId("P2.DecoderPort")
    PRIVACY_MASK = PortId("P3.PrivacyMaskPort")
    CLOCK_SYNC = PortId("P4.ClockSyncPort")
    ADMISSION_POLICY = PortId("P5.AdmissionPolicyPort")
    CHANGE_DETECTOR = PortId("P6.ChangeDetectorPort")
    ALLOCATOR = PortId("P7.AllocatorPort")
    DETECTOR = PortId("P8.DetectorPort")
    TRACKER = PortId("P9.TrackerPort")
    EMBEDDING = PortId("P10.EmbeddingPort")
    IDENTITY_RESOLVER = PortId("P11.IdentityResolverPort")
    TRIGGER_POLICY = PortId("P12.TriggerPolicyPort")
    QUALITY_ESTIMATOR = PortId("P13.QualityEstimatorPort")
    CROP_STRATEGY = PortId("P14.CropStrategyPort")
    UNDERSTANDER = PortId("P15.UnderstanderPort")
    OUTPUT_COERCION = PortId("P16.OutputCoercionPort")
    PROMPT_SOURCE = PortId("P17.PromptSourcePort")
    SUPPRESSION_POLICY = PortId("P18.SuppressionPolicyPort")
    OBSERVATION_SINK = PortId("P19.ObservationSinkPort")
    OBSERVATION_LOG = PortId("P20.ObservationLogPort")
    STATE_STORE = PortId("P21.StateStorePort")
    EVIDENCE_STORE = PortId("P22.EvidenceStorePort")
    CONFIG_SOURCE = PortId("P23.ConfigSourcePort")
    SECRET_PROVIDER = PortId("P24.SecretProviderPort")
    ARTIFACT_STORE = PortId("P25.ArtifactStorePort")
    MODEL_RUNTIME = PortId("P26.ModelRuntimePort")
    DEVICE = PortId("P27.DevicePort")
    CALIBRATION = PortId("P28.CalibrationPort")
    EVENT_TRANSPORT = PortId("P29.EventTransportPort")
    METRICS_EXPORT = PortId("P30.MetricsExportPort")
    AUTHORIZATION = PortId("P31.AuthorizationPort")
    API_TRANSPORT = PortId("P32.ApiTransportPort")


ALL_PORTS: frozenset[PortId] = frozenset(
    value for key, value in vars(PortCatalogue).items() if not key.startswith("_")
)

#: Ports implemented by Flow 1.
FLOW1_PORTS: frozenset[PortId] = frozenset(
    {
        PortCatalogue.SOURCE,
        PortCatalogue.DECODER,
        PortCatalogue.PRIVACY_MASK,
        PortCatalogue.CLOCK_SYNC,
        PortCatalogue.ADMISSION_POLICY,
        PortCatalogue.CHANGE_DETECTOR,
        PortCatalogue.ALLOCATOR,
        PortCatalogue.CONFIG_SOURCE,
        PortCatalogue.SECRET_PROVIDER,
        PortCatalogue.EVENT_TRANSPORT,
        PortCatalogue.METRICS_EXPORT,
    }
)

#: Ports implemented by Flow 2 — detection and the model substrate that serves it.
FLOW2_PORTS: frozenset[PortId] = frozenset(
    {
        PortCatalogue.DETECTOR,
        PortCatalogue.ARTIFACT_STORE,
        PortCatalogue.MODEL_RUNTIME,
        PortCatalogue.DEVICE,
    }
)

#: Ports implemented by Flow 3 — tracking.
#:
#: ``EMBEDDING`` (P10) is deliberately **absent**. The port is defined so that a
#: tracker's ``requires_embeddings`` capability is meaningful, but appearance
#: embeddings are C2 biometric data, disabled by default (12_SECURITY section
#: 4.3). Making it bindable would let a deployment turn on the platform's most
#: invasive capability by adding a plugin, which is exactly the accident the
#: frontier exists to prevent.
FLOW3_PORTS: frozenset[PortId] = frozenset({PortCatalogue.TRACKER})

#: Ports implemented by Flow 4 — the Object Registry's durable state.
#:
#: ``IDENTITY_RESOLVER`` (P11) is deliberately **absent**. 15_ROADMAP section 3:
#: *"already specified, no implementations in Phase 1"*. M7's native
#: spatio-temporal binding is mandatory behaviour that needs no adapter; P11 is
#: the seam for replacing it with appearance-based or cross-camera strategies,
#: and cross-camera identity is classified C2 and policy-gated.
#:
#: ``STATE_STORE`` is bound here for the narrow purpose 07_STATE section 9.3
#: requires — persisting the object population so identity survives a restart.
#: It is not the Vision State projection, which is M13 at L6.
FLOW4_PORTS: frozenset[PortId] = frozenset({PortCatalogue.STATE_STORE})

#: Ports implemented by Flow 5 — the Crop Manager's attention machinery.
#:
#: All three ship with default adapters, unlike P11. §M8's Extension Points
#: section names each as replaceable: the trigger set is *"a default policy,
#: fully replaceable"*, quality estimation is *"heuristic sharpness/scale today;
#: learned quality predictors later"*, and crop strategies extend to multi-scale
#: and part-focused geometry.
#:
#: ``EVIDENCE_STORE`` (P22) is deliberately **absent**. M8 decides retention
#: *policy* and stamps it on the crop; persisting imagery is a different module's
#: job, and binding a store here would put a durable side effect inside the
#: platform's cheapest, hottest path.
FLOW5_PORTS: frozenset[PortId] = frozenset(
    {
        PortCatalogue.TRIGGER_POLICY,
        PortCatalogue.QUALITY_ESTIMATOR,
        PortCatalogue.CROP_STRATEGY,
    }
)

#: Ports implemented by Flow 6 — the Understanding Engine.
#:
#: ``UNDERSTANDER`` is one of the four ports 06_PORTS bolds as *"the ports that
#: make the platform a platform"*. Binding it is what lets a 7-billion-parameter
#: VLM and a 2-megabyte attribute head be interchangeable.
#:
#: ``PROMPT_SOURCE`` (P17) is deliberately **absent**: it belongs to M10, which
#: Flow 6 does not implement. M9 consumes prompts through a module seam, not
#: through a port it owns.
#:
#: ``EVIDENCE_STORE`` (P22) is also absent. M9 content-addresses its raw output
#: and carries the bytes; persisting them is M13's job, and binding a store here
#: would put a durable write inside the platform's most expensive path.
FLOW6_PORTS: frozenset[PortId] = frozenset(
    {
        PortCatalogue.UNDERSTANDER,
        PortCatalogue.OUTPUT_COERCION,
    }
)

#: Ports implemented by Flow 7 — synthesis and state.
#:
#: ``OBSERVATION_LOG`` (P20) is listed against M13 in the catalogue, and binding
#: it here is not implementing M13: §M13's single responsibility is *"Describe
#: what must persist and with what guarantees; **implement none of it**."* It
#: owns no state and is a set of contracts. Flow 2 bound M18's storage ports the
#: same way, and Flow 4 bound P21.
#:
#: ``EVIDENCE_STORE`` (P22) is deliberately **absent**. M11 stamps retention onto
#: the evidence reference and M9 content-addresses the payload; writing imagery
#: durably is still nobody's job in Phase 1, and binding a store here would put a
#: blob write on the observation hot path.
#:
#: ``API_TRANSPORT`` (P32) is absent because M14 is Flow 8. State exposes read
#: methods; exposing them over a wire is a different module's contract.
FLOW7_PORTS: frozenset[PortId] = frozenset(
    {
        PortCatalogue.SUPPRESSION_POLICY,
        PortCatalogue.OBSERVATION_SINK,
        PortCatalogue.OBSERVATION_LOG,
    }
)

#: Ports implemented by Flow 8 — storage interfaces and exposure.
#:
#: ``EVIDENCE_STORE`` (P22) becomes bindable here and nowhere earlier. Flows 5, 6
#: and 7 each left it unbound with a stated reason, and each reason was *"not this
#: flow's job"* rather than *"never"* — Flow 6's note reads *"persisting them is
#: M13's job."* Flow 8 **is** M13, so this is the earlier flows' expectation
#: arriving on schedule rather than a boundary crossed early.
#:
#: ``CALIBRATION`` (P28) stays **absent**. `06_PORTS` assigns it to M1 and M18,
#: neither of which is Flow 8's, and the Camera Manager's declared calibration
#: already satisfies 07_STATE without an adapter. Binding a port whose owning
#: module is not in scope is exactly what the frontier discipline forbids.
FLOW8_PORTS: frozenset[PortId] = frozenset(
    {
        PortCatalogue.EVIDENCE_STORE,
        PortCatalogue.AUTHORIZATION,
        PortCatalogue.API_TRANSPORT,
    }
)

#: Everything currently bindable. Binding anything else is rejected, because a
#: plugin for a port whose owning module does not exist yet cannot be activated —
#: which is how "no future flow is implemented early" stays enforceable.
#:
#: Two ports remain permanently unbindable in Phase 1, and neither is a frontier
#: matter: ``EMBEDDING`` (P10) and ``IDENTITY_RESOLVER`` (P11) are the biometric
#: and cross-camera-identity capabilities, disabled by default under
#: 12_SECURITY §4.3 and deferred to Phase 2 by 15_ROADMAP §3. ``PROMPT_SOURCE``
#: (P17) belongs to M10, which no flow has implemented.
BINDABLE_PORTS: frozenset[PortId] = (
    FLOW1_PORTS
    | FLOW2_PORTS
    | FLOW3_PORTS
    | FLOW4_PORTS
    | FLOW5_PORTS
    | FLOW6_PORTS
    | FLOW7_PORTS
    | FLOW8_PORTS
)


@dataclass(frozen=True, slots=True)
class VersionRange:
    """An inclusive-exclusive semantic version range, ``>=min <max``."""

    minimum: tuple[int, int, int]
    maximum: tuple[int, int, int]

    @classmethod
    def parse(cls, text: str) -> VersionRange:
        """Parse ``">=1.2 <2.0"``."""
        minimum = (0, 0, 0)
        maximum = (999, 0, 0)
        for token in text.split():
            if token.startswith(">="):
                minimum = _parse_version(token[2:])
            elif token.startswith("<"):
                maximum = _parse_version(token[1:])
            else:
                raise ValueError(f"malformed version range token: {token!r}")
        return cls(minimum, maximum)

    def contains(self, version: str) -> bool:
        parsed = _parse_version(version)
        return self.minimum <= parsed < self.maximum

    def __str__(self) -> str:
        return f">={_fmt(self.minimum)} <{_fmt(self.maximum)}"


def _parse_version(text: str) -> tuple[int, int, int]:
    parts = text.strip().split(".")
    while len(parts) < 3:
        parts.append("0")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise ValueError(f"malformed version: {text!r}") from exc


def _fmt(version: tuple[int, int, int]) -> str:
    return ".".join(str(v) for v in version)


@dataclass(frozen=True, slots=True)
class ResourceDeclaration:
    """What a plugin claims it needs. A declaration is a contract, not a hint."""

    device: str = "cpu"
    memory_bytes: int = 0
    vram_bytes: int = 0
    exclusive: bool = False


@dataclass(frozen=True, slots=True)
class PluginManifest:
    """A plugin's self-declaration (06_PORTS §3, 05_KERNEL M17)."""

    plugin_id: PluginId
    version: str
    port_id: PortId
    port_version_range: VersionRange
    platform_range: VersionRange
    isolation: IsolationLevel = IsolationLevel.IN_PROCESS
    resources: ResourceDeclaration = ResourceDeclaration()
    thread_safe: bool = True
    """Declared, and honoured by the runtime: a plugin declaring itself
    single-threaded gets a dedicated worker rather than an unsafe shared one."""

    deterministic: bool = True
    """V13 replay must know what to expect."""

    capabilities: dict[str, str] = field(default_factory=dict)
    """Published so capability gaps are detectable (V8)."""

    signature: str | None = None
    conformance_kit_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not self.plugin_id:
            raise ValueError("plugin_id is required")
        if not self.version:
            raise ValueError("version is required")
        _parse_version(self.version)
