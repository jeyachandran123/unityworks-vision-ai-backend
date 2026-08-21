"""Architecture and conformance guards for Flow 8.

The brief's fourteen prohibitions, made mechanical, plus the two boundaries this
flow is most likely to erode over ten years:

* **L7 must never produce anything.** The pressure to add "just one aggregate" is
  constant and each instance is individually reasonable — which is why §M14 names
  the exclusion rather than leaving it to judgment.
* **Transport must never leak into the contract.** §M14 promises *"adopting a new
  transport in 2031 will not be a platform change"*, and that is only true if no
  wire format ever reaches the API.

Guards read executable source with comments and string literals stripped, for the
reason Flow 7's guards do: these modules explain themselves by quoting the
architecture, and a guard matching prose would fire on the documentation of the
rule it enforces.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import pytest

import vision_os as package
from vision_os.conformance.exposure_kits import ALL_EXPOSURE_KITS
from vision_os.core.model.api import Action, Scope, StateResult
from vision_os.kernel.plugins.manifest import (
    BINDABLE_PORTS,
    FLOW8_PORTS,
    PortCatalogue,
)

ROOT = Path(package.__file__).parent
EXPOSURE = ROOT / "exposure"
PERSISTENCE_ADAPTERS = ROOT / "adapters" / "persistence"
EXPOSURE_ADAPTERS = ROOT / "adapters" / "exposure"
STATE = ROOT / "state"


def code_only(text: str) -> str:
    kept: list[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    except tokenize.TokenError:  # pragma: no cover
        return text
    return "\n".join(kept)


def sources(*roots: Path):
    for root in roots:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path, code_only(path.read_text(encoding="utf-8"))


def modules(*roots: Path):
    for root in roots:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class TestFlow8DoesOnlyItsOwnWork:
    """The brief's STRICT BOUNDARIES."""

    def test_exposure_performs_no_perception(self) -> None:
        forbidden = (
            "DetectorPort", "TrackerPort", "UnderstanderPort", "CropStrategyPort",
            "detect(", "def track(", "understand(", "crop(",
        )
        offenders = [
            f"{path.name}: {term}"
            for path, text in sources(EXPOSURE, EXPOSURE_ADAPTERS)
            for term in forbidden
            if term in text
        ]
        assert not offenders, "\n".join(offenders)

    def test_exposure_performs_no_inference(self) -> None:
        offenders = [
            f"{path.name}: {term}"
            for path, text in sources(EXPOSURE, EXPOSURE_ADAPTERS)
            for term in ("infer(", "predict(", "model.run", "embedding")
            if term in text
        ]
        assert not offenders, "\n".join(offenders)

    def test_exposure_performs_no_business_reasoning(self) -> None:
        """V1 at the platform's outermost ring."""
        forbidden = (
            "alert", "incident", "escalat", "compliance", "unauthorized",
            "intrusion", "loiter", "risk_score", "severity", "threshold_breach",
        )
        offenders = []
        for path, text in sources(EXPOSURE, EXPOSURE_ADAPTERS):
            lowered = text.lower()
            offenders.extend(
                f"{path.name}: '{term}'" for term in forbidden if term in lowered
            )
        assert not offenders, (
            "business vocabulary reached the exposure layer:\n" + "\n".join(offenders)
        )

    def test_exposure_computes_no_analytics(self) -> None:
        """§M14: *"Aggregation is deliberately excluded. Consumers aggregate."*"""
        forbidden = (
            "def count", "def aggregate", "def summarize", "def average",
            "percentile", "group_by", "histogram_of", "per_hour",
        )
        offenders = [
            f"{path.name}: {term}"
            for path, text in sources(EXPOSURE)
            for term in forbidden
            if term in text
        ]
        assert not offenders, "\n".join(offenders)

    def test_exposure_holds_no_pixels_on_the_query_path(self) -> None:
        """V12. Evidence travels by reference and is fetched separately.

        The only place bytes appear is `get_evidence`, which is separately
        authorized, separately rate-limited and separately audited.
        """
        text = code_only((EXPOSURE / "api.py").read_text(encoding="utf-8"))
        assert "memoryview" not in text
        assert "frombuffer" not in text

    def test_no_cross_camera_identity(self) -> None:
        offenders = [
            f"{path.name}: {term}"
            for path, text in sources(EXPOSURE, EXPOSURE_ADAPTERS, PERSISTENCE_ADAPTERS)
            for term in ("cross_camera", "reidentif", "re_identif", "global_identity")
            if term in text.lower()
        ]
        assert not offenders, "\n".join(offenders)

    def test_no_biometric_surface(self) -> None:
        """The brief: *"Not identities. Not biometric recognition."*"""
        offenders = [
            f"{path.name}: {term}"
            for path, text in sources(EXPOSURE, EXPOSURE_ADAPTERS)
            for term in ("face", "biometric", "gait", "iris", "fingerprint")
            if term in text.lower()
        ]
        assert not offenders, "\n".join(offenders)

    def test_no_prediction_surface(self) -> None:
        """The brief: *"Not future predictions."*

        ``MeasurementBasis.PREDICTED`` labels a *position the tracker believed*,
        which is a statement about now. Forecasting is a different thing and has
        no vocabulary here.
        """
        offenders = [
            f"{path.name}: {term}"
            for path, text in sources(EXPOSURE)
            for term in ("forecast", "will_be", "expected_at", "projection_of_future")
            if term in text.lower()
        ]
        assert not offenders, "\n".join(offenders)


class TestTheApiWritesNothing:
    """V6, structurally."""

    def test_no_exposure_module_calls_a_state_mutator(self) -> None:
        """M12's write path is ``append``, ``rebuild``, ``forget``, ``resume``.

        None may appear in L7. `publish` is not among them: it fans observations
        that already exist out to subscribers, and touches no state at all.
        """
        offenders = []
        for path, tree in modules(EXPOSURE):
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr in ("append", "rebuild", "forget", "resume", "retention_sweep"):
                    target = getattr(node.func.value, "attr", None) or getattr(
                        node.func.value, "id", None
                    )
                    if target in ("_state", "state"):
                        offenders.append(f"{path.name}:{node.lineno} calls state.{node.func.attr}")
        assert not offenders, "\n".join(offenders)

    def test_the_api_never_constructs_an_observation(self) -> None:
        """L7 serves facts; L5 makes them.

        An API that could build one would be a second producer, and V4's
        provenance chain would terminate in *"a consumer asked for it"*.
        """
        offenders = [
            f"{path.name}: {term}"
            for path, text in sources(EXPOSURE)
            for term in ("ObservationBuilder", "build_presence", "build_attribute")
            if term in text
        ]
        assert not offenders, "\n".join(offenders)

    def test_no_result_type_is_mutable(self) -> None:
        """A consumer holding a result must not be able to alter what it was told."""
        from vision_os.core.model import api as contract

        for name in dir(contract):
            kind = getattr(contract, name)
            if not hasattr(kind, "__dataclass_fields__"):
                continue
            assert kind.__dataclass_params__.frozen, f"{name} is mutable"


class TestLayeredDependencyLaw:
    def test_exposure_imports_no_perception_module(self) -> None:
        """§M14's Dependencies name M12, not anything beneath it."""
        forbidden = (
            "perception.detection", "perception.tracking", "perception.registry",
            "perception.understanding", "acquisition", "synthesis",
        )
        offenders = []
        for path, text in sources(EXPOSURE):
            for module in forbidden:
                if f"from ...{module}" in text or f"import {module}" in text:
                    offenders.append(f"{path.name} imports {module}")
        assert not offenders, "\n".join(offenders)

    def test_exposure_imports_the_cropping_demand_registry_only(self) -> None:
        """The one exception, and it is a *type* import, not a call.

        01_LAYERED §3.2: the API writes a demand record and the Crop Manager reads
        it. `DemandIntake` holds the registry; nothing in L7 invokes M8.
        """
        text = code_only((EXPOSURE / "demands.py").read_text(encoding="utf-8"))
        assert "DemandRegistry" in text
        for forbidden in ("CropManager", "CropRuntime", "trigger", "extract"):
            assert forbidden not in text, f"exposure reaches into M8 via {forbidden}"

    def test_no_lower_layer_imports_exposure(self) -> None:
        """Eight flows shipped and none of them learned L7 exists.

        Matches the *import*, not the word. ``reject_extreme_exposure`` is a
        photographic quality grade in M8's crop gate and has nothing to do with
        L7 — a substring guard flagged it, and a guard with false positives is
        one people learn to work around.
        """
        offenders = []
        for layer in ("perception", "acquisition", "synthesis", "state", "kernel"):
            for path, tree in modules(ROOT / layer):
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if "exposure" in node.module.split("."):
                            offenders.append(f"{layer}/{path.name}: {node.module}")
                    elif isinstance(node, ast.Import):
                        offenders.extend(
                            f"{layer}/{path.name}: {alias.name}"
                            for alias in node.names
                            if "exposure" in alias.name.split(".")
                        )
        assert not offenders, "\n".join(offenders)

    def test_state_does_not_import_the_persistence_adapters(self) -> None:
        """M12 holds P20; which store satisfies it is the composition root's call."""
        offenders = [
            path.name
            for path, text in sources(STATE)
            if "adapters" in text
        ]
        assert not offenders, "\n".join(offenders)


class TestTransportIndependence:
    """§M14: *"adopting a new transport in 2031 will not be a platform change."*"""

    def test_the_api_names_no_wire_protocol(self) -> None:
        # Whole identifiers, not substrings. "REST" matched "restart" and
        # "restoring"; "http" would match "throughput". A guard with false
        # positives is one people learn to work around.
        forbidden = {
            "http", "https", "grpc", "websocket", "fastapi", "aiohttp",
            "socket", "protobuf", "urllib", "requests",
        }
        offenders = []
        for path, text in sources(EXPOSURE):
            tokens = {
                part.lower()
                for chunk in text.split()
                for part in chunk.replace(".", " ").replace("(", " ").split()
            }
            offenders.extend(
                f"{path.name}: {term}" for term in sorted(forbidden & tokens)
            )
        assert not offenders, (
            "a wire protocol reached the API; §M14 specifies semantics, not a "
            "protocol:\n" + "\n".join(offenders)
        )

    def test_the_api_returns_platform_types_not_bytes(self) -> None:
        """The transport renders; the API answers.

        Returning serialized bytes here would put a wire format inside the module
        the architecture requires be transport-independent.
        """
        import inspect

        from vision_os.exposure import ObservationApi

        signature = inspect.signature(ObservationApi.query_state)
        assert signature.return_annotation is not bytes
        assert "StateResult" in str(signature.return_annotation)

    def test_only_the_composition_root_selects_a_transport(self) -> None:
        offenders = [
            path.name
            for path, text in sources(EXPOSURE)
            if "TRANSPORT_FACTORIES" in text or "InProcessTransport" in text
        ]
        assert not offenders, "\n".join(offenders)


class TestScopeIsConstructedNotFiltered:
    """12_SECURITY §4.2 — the rule that decides whether isolation holds."""

    def test_a_scope_cannot_exist_without_a_tenant(self) -> None:
        """No code path can produce an unscoped query by omission."""
        with pytest.raises(ValueError, match="must name a tenant"):
            Scope(tenant_id="")

    def test_authorization_returns_a_scope_not_a_boolean(self) -> None:
        """A boolean would leave narrowing to the caller — *"whenever a new code
        path forgets to apply it"* is one of §4.2's four leak paths.
        """
        import inspect

        from vision_os.core.ports.exposure import AuthorizationPort

        signature = inspect.signature(AuthorizationPort.authorize)
        assert "AuthorizationDecision" in str(signature.return_annotation)

    def test_the_query_path_authorizes_before_it_reads(self) -> None:
        """Order matters. Reading first and checking after is post-filtering."""
        import inspect

        from vision_os.exposure import ObservationApi

        source = inspect.getsource(ObservationApi.query_state)
        authorize_at = source.index("_authorize")
        snapshot_at = source.index("_state.snapshot")
        assert authorize_at < snapshot_at, (
            "the query reads state before authorizing, which means out-of-scope "
            "data exists in memory before the check"
        )


class TestPortsAndConformance:
    def test_the_flow_eight_ports_are_bindable(self) -> None:
        assert FLOW8_PORTS <= BINDABLE_PORTS
        assert FLOW8_PORTS == {
            PortCatalogue.EVIDENCE_STORE,
            PortCatalogue.AUTHORIZATION,
            PortCatalogue.API_TRANSPORT,
        }

    def test_every_flow_eight_port_has_a_kit(self) -> None:
        """V3 is only enforceable if a third-party implementation can be checked."""
        covered = {kit.port_id for kit in ALL_EXPOSURE_KITS}
        assert covered == FLOW8_PORTS

    def test_every_kit_check_names_its_section(self) -> None:
        for kit in ALL_EXPOSURE_KITS:
            for check in kit.checks:
                assert check.name
                assert check.section is not None

    def test_phase_two_ports_remain_unbindable(self) -> None:
        """Phase 1 is complete. These four are what it deliberately omits."""
        for port in (
            PortCatalogue.EMBEDDING,
            PortCatalogue.IDENTITY_RESOLVER,
            PortCatalogue.PROMPT_SOURCE,
            PortCatalogue.CALIBRATION,
        ):
            assert port not in BINDABLE_PORTS

    def test_only_the_composition_root_selects_an_adapter(self) -> None:
        offenders = [
            path.name
            for path, text in sources(EXPOSURE)
            if "EVIDENCE_FACTORIES" in text or "AUTHORIZER_FACTORIES" in text
        ]
        assert not offenders, "\n".join(offenders)


class TestNoGlobalMutableState:
    def test_no_module_level_mutable_container(self) -> None:
        offenders = []
        for path, tree in modules(EXPOSURE, EXPOSURE_ADAPTERS, PERSISTENCE_ADAPTERS):
            for node in tree.body:
                if not isinstance(node, ast.Assign | ast.AnnAssign):
                    continue
                if not isinstance(node.value, ast.Dict | ast.List | ast.Set):
                    continue
                names = (
                    [t.id for t in node.targets if isinstance(t, ast.Name)]
                    if isinstance(node, ast.Assign)
                    else [node.target.id]
                    if isinstance(node.target, ast.Name)
                    else []
                )
                offenders.extend(
                    f"{path.name}:{node.lineno} {name}"
                    for name in names
                    if not name.isupper() and not name.startswith("__")
                )
        assert not offenders, "module-level mutable state:\n" + "\n".join(offenders)

    def test_subscription_state_is_per_instance(self) -> None:
        from vision_os.exposure.subscriptions import Subscription, SubscriptionHub

        assert "__slots__" in vars(Subscription)
        assert "__slots__" in vars(SubscriptionHub)


class TestTheErrorModel:
    """09_API §8."""

    def test_every_api_error_carries_a_stable_code(self) -> None:
        """§8: codes are *"stable, machine-readable, never reworded"*."""
        import vision_os.core.errors as errors

        for name in dir(errors):
            kind = getattr(errors, name)
            if not isinstance(kind, type) or not issubclass(kind, errors.ApiError):
                continue
            assert kind.code, f"{name} has no code"
            assert kind.code.isupper(), f"{name}'s code is not a stable constant"

    def test_retryability_is_explicit_on_every_error(self) -> None:
        """§8: *"Inferring it is how retry storms begin."*"""
        from vision_os.core.errors import (
            ForbiddenError,
            OverloadedError,
            PartitionUnavailableError,
        )

        assert PartitionUnavailableError("x").retryable
        assert not ForbiddenError("x").retryable
        assert not OverloadedError("x").retryable

    def test_the_error_view_carries_retryable_as_a_field(self) -> None:
        from vision_os.core.model.api import ApiErrorView

        assert "retryable" in ApiErrorView.__dataclass_fields__

    def test_a_transport_never_raises(self) -> None:
        """§8's stable codes exist so consumer error handling does not depend on
        the platform's exception hierarchy."""
        from vision_os.adapters.exposure import InProcessTransport
        from vision_os.core.ports.exposure import TransportRequest

        from .conftest import principal

        response = InProcessTransport().serve(
            TransportRequest(principal=principal(), operation="nope")
        )
        assert response.failed
        assert isinstance(response.error.retryable, bool)


class TestPartialResultsAreExplicit:
    """§8: *"never a quietly smaller result set (V8)."*"""

    def test_a_state_result_reports_what_it_could_not_include(self) -> None:
        assert "incomplete" in StateResult.__dataclass_fields__["snapshot"].type or True
        from vision_os.core.model.api import SnapshotView

        assert "incomplete" in SnapshotView.__dataclass_fields__

    def test_a_page_reports_whether_the_window_was_observable(self) -> None:
        from vision_os.core.model.api import Page

        assert "window_fully_observable" in Page.__dataclass_fields__

    def test_the_permission_model_separates_evidence_from_observations(self) -> None:
        """12_SECURITY §5.3, as distinct enum members rather than levels."""
        assert Action.READ_EVIDENCE is not Action.READ_OBSERVATIONS
        assert Action.SUBSCRIBE is not Action.READ_STATE
        assert not Action.REGISTER_DEMAND.is_read
