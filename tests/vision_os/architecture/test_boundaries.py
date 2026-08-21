"""Architecture boundary tests — the invariants, enforced mechanically.

An invariant with no test is a slogan (14_TESTING §11). These tests read the
source tree and fail the build when a boundary is crossed, so the constitution is
defended by CI rather than by memory.

The most valuable test here is ``test_no_domain_vocabulary_in_platform_code``.
It is crude on purpose: it catches the *first* leak, which is the one that
establishes precedent. Every general vision platform that became a vertical
product did so through a series of individually reasonable exceptions.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import vision_os as vision_os_pkg
from vision_os.kernel.config.schema import ALLOWED_TOP_LEVEL, SECTION_TYPES, allowed_keys
from vision_os.kernel.plugins.manifest import (
    BINDABLE_PORTS,
    FLOW1_PORTS,
    FLOW2_PORTS,
    FLOW3_PORTS,
    FLOW4_PORTS,
    FLOW5_PORTS,
    FLOW6_PORTS,
    FLOW7_PORTS,
    FLOW8_PORTS,
)

ROOT = Path(vision_os_pkg.__file__).parent

CORE = ROOT / "core"
KERNEL = ROOT / "kernel"
ACQUISITION = ROOT / "acquisition"
ADAPTERS = ROOT / "adapters"
CONFORMANCE = ROOT / "conformance"
PERCEPTION = ROOT / "perception"
TAXONOMY = ROOT / "taxonomy"


def _python_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


def _module_of(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                found.append("." * node.level + (node.module or ""))
            elif node.module:
                found.append(node.module)
    return found


# --- V3: ports over implementations ------------------------------------------ #

STDLIB_ALLOWED = {
    "__future__", "abc", "ast", "asyncio", "collections", "collections.abc",
    "contextlib", "dataclasses", "datetime", "enum", "functools", "hashlib",
    "heapq", "itertools", "json", "math", "os", "pathlib", "random", "re",
    "secrets", "tempfile", "threading", "time", "typing", "uuid", "weakref",
}

THIRD_PARTY_MARKERS = (
    "numpy", "cv2", "torch", "tensorflow", "onnx", "ultralytics", "PIL",
    "redis", "sqlalchemy", "fastapi", "pydantic", "psycopg", "kafka",
    "prometheus_client", "boto3", "requests", "httpx", "aiohttp", "chromadb",
    "langchain", "transformers", "tensorrt", "pycuda",
)


class TestCoreIsPure:
    """``core`` is contracts only: stdlib-only, no I/O, no vendor knowledge."""

    def test_core_imports_no_third_party_packages(self) -> None:
        offenders: list[str] = []
        for path in _python_files(CORE):
            for imported in _imports(path):
                root = imported.split(".")[0]
                if root in THIRD_PARTY_MARKERS:
                    offenders.append(f"{_module_of(path)} imports {imported}")
        assert not offenders, (
            "core must be stdlib-only so the platform never depends on a vendor:\n"
            + "\n".join(offenders)
        )

    def test_core_imports_only_stdlib_and_itself(self) -> None:
        offenders: list[str] = []
        for path in _python_files(CORE):
            for imported in _imports(path):
                if imported.startswith("."):
                    continue
                root = imported.split(".")[0]
                if root in STDLIB_ALLOWED:
                    continue
                offenders.append(f"{_module_of(path)} imports {imported}")
        assert not offenders, "unexpected core dependency:\n" + "\n".join(offenders)

    def test_core_does_not_import_kernel_or_layers(self) -> None:
        """Contracts may not depend on the machinery that implements them."""
        offenders: list[str] = []
        for path in _python_files(CORE):
            text = path.read_text(encoding="utf-8")
            for forbidden in ("vision_os.kernel", "vision_os.acquisition", "vision_os.adapters"):
                if forbidden in text:
                    offenders.append(f"{_module_of(path)} references {forbidden}")
        assert not offenders, "\n".join(offenders)


class TestNoExternalTechnologyInPlatform:
    def test_only_adapters_may_touch_external_technology(self) -> None:
        """External technology never enters the core (invariant V3).

        Detectors, codecs, databases and queues live behind ports. Today no
        adapter needs a third-party package either — the Flow 1 reference set is
        dependency-free — but the boundary is enforced now so it holds when
        NVDEC and RTSP arrive.
        """
        offenders: list[str] = []
        for directory in (CORE, KERNEL, ACQUISITION, CONFORMANCE, PERCEPTION, TAXONOMY):
            for path in _python_files(directory):
                for imported in _imports(path):
                    root = imported.split(".")[0]
                    if root in THIRD_PARTY_MARKERS:
                        offenders.append(f"{_module_of(path)} imports {imported}")
        assert not offenders, (
            "external technology must live behind an adapter:\n" + "\n".join(offenders)
        )


class TestLayerDependencyLaw:
    """Flow layers depend downward only; the kernel depends on none of them."""

    def test_kernel_never_imports_a_flow_layer(self) -> None:
        """The kernel law: no L0 module knows what a frame is.

        This is what allows the kernel to be reused unchanged by a future
        UnityWorks Audio OS, and what stops L0 becoming the place where layering
        rules go to die.
        """
        offenders: list[str] = []
        for path in _python_files(KERNEL):
            for imported in _imports(path):
                if "acquisition" in imported or "adapters" in imported:
                    offenders.append(f"{_module_of(path)} imports {imported}")
        assert not offenders, (
            "kernel must not depend on a flow layer:\n" + "\n".join(offenders)
        )

    def test_acquisition_never_imports_adapters(self) -> None:
        """A module that names a concrete adapter has bypassed its port."""
        offenders: list[str] = []
        for path in _python_files(ACQUISITION):
            for imported in _imports(path):
                if "adapters" in imported:
                    offenders.append(f"{_module_of(path)} imports {imported}")
        assert not offenders, "\n".join(offenders)

    def test_platform_modules_do_not_name_concrete_adapters(self) -> None:
        concrete = (
            "HostMemoryPool", "InMemoryRawSource", "PassthroughDecoder",
            "CadenceAdmissionPolicy", "StaticZoneMask", "JsonFileConfigSource",
            "NullEventTransport", "SampledDigestChangeDetector",
        )
        offenders: list[str] = []
        for directory in (CORE, KERNEL, ACQUISITION, PERCEPTION, TAXONOMY):
            for path in _python_files(directory):
                text = path.read_text(encoding="utf-8")
                for name in concrete:
                    if name in text:
                        offenders.append(f"{_module_of(path)} names {name}")
        assert not offenders, (
            "platform modules must reference ports, never adapters:\n" + "\n".join(offenders)
        )


class TestInjectedClock:
    """Invariant V13 — a module that reads the system clock can never be replayed."""

    def test_no_module_reads_the_wall_clock_directly(self) -> None:
        pattern = re.compile(r"\btime\.(time|time_ns|monotonic|monotonic_ns)\s*\(")
        allowed = {"kernel/clock.py", "core/model/ids.py"}
        offenders: list[str] = []
        for directory in (CORE, KERNEL, ACQUISITION, ADAPTERS, PERCEPTION, TAXONOMY):
            for path in _python_files(directory):
                module = _module_of(path)
                if module in allowed:
                    continue
                if pattern.search(path.read_text(encoding="utf-8")):
                    offenders.append(module)
        assert not offenders, (
            "time must be injected, not read (invariant V13):\n" + "\n".join(offenders)
        )

    def test_no_module_calls_datetime_now(self) -> None:
        offenders: list[str] = []
        for directory in (CORE, KERNEL, ACQUISITION, ADAPTERS, PERCEPTION, TAXONOMY):
            for path in _python_files(directory):
                text = path.read_text(encoding="utf-8")
                if "datetime.now(" in text or "datetime.utcnow(" in text:
                    offenders.append(_module_of(path))
        assert not offenders, "\n".join(offenders)


class TestDependencyInjection:
    """No global state, no hidden singletons, no service locators."""

    def test_no_module_level_mutable_singletons(self) -> None:
        offenders: list[str] = []
        for directory in (KERNEL, ACQUISITION, PERCEPTION, TAXONOMY):
            for path in _python_files(directory):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in tree.body:
                    if not isinstance(node, ast.Assign):
                        continue
                    for target in node.targets:
                        if not isinstance(target, ast.Name) or target.id.startswith("_"):
                            continue
                        if target.id.isupper():
                            continue  # module constants are fine
                        if isinstance(node.value, ast.Dict | ast.List | ast.Set | ast.Call):
                            offenders.append(f"{_module_of(path)}::{target.id}")
        assert not offenders, (
            "mutable module-level state is a hidden singleton:\n" + "\n".join(offenders)
        )

    def test_every_module_takes_its_collaborators_by_constructor(self) -> None:
        import inspect

        from vision_os.acquisition import (
            CameraManager,
            FrameBuffer,
            FrameScheduler,
            VideoSourceManager,
        )
        from vision_os.kernel.config import ConfigurationManager
        from vision_os.kernel.events import EventBus
        from vision_os.kernel.health import HealthMonitor
        from vision_os.kernel.metrics import MetricsEngine
        from vision_os.kernel.plugins import PluginManager
        from vision_os.kernel.runtime import VisionRuntime

        for cls in (
            CameraManager, FrameBuffer, FrameScheduler, VideoSourceManager,
            ConfigurationManager, EventBus, HealthMonitor, MetricsEngine,
            PluginManager, VisionRuntime,
        ):
            signature = inspect.signature(cls.__init__)
            parameters = [p for p in signature.parameters if p != "self"]
            assert parameters, f"{cls.__name__} must receive dependencies by constructor"


# --- V1 / V2: the semantic ceiling and vertical neutrality --------------------- #

#: Terms that may never appear as an identifier token in platform code.
#:
#: Deliberately restricted to vocabulary with **no legitimate engineering
#: meaning**, so the guard has no false positives and therefore never gets
#: disabled. Words like "violation" (schema validation) and "factory" (the
#: construction pattern) are excluded precisely because they are ambiguous —
#: the real ceiling enforcement is the closed config schema here, and the
#: attribute registry in Flow 5.
DOMAIN_VOCABULARY = (
    # roles a crop cannot evidence
    "waiter", "chef", "cashier", "patient", "nurse", "doctor", "customer",
    "employee", "shopper", "clerk",
    # verticals
    "restaurant", "kitchen", "hospital", "warehouse", "retail", "clinic",
    # vertical objects
    "biryani", "menu", "checkout", "shelf", "till",
    # judgments and conclusions
    "unproductive", "suspicious", "loitering", "shoplifting", "anomaly",
    "alert", "unauthorized", "noncompliant", "infraction",
)

_TOKEN = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+")


def _tokens(identifier: str) -> set[str]:
    """Split an identifier into lowercase word tokens.

    Whole-token matching rather than substring, so ``_sleepers`` does not trip
    on "sleep" and ``violations`` does not trip on a business "violation".
    """
    return {token.lower() for token in _TOKEN.findall(identifier)}


class TestSemanticCeiling:
    """Invariant V1/V2 — the platform reports what is visible, never what it means."""

    def test_no_domain_vocabulary_in_platform_code(self) -> None:
        """Catches the first leak, which is the one that sets precedent.

        Crude, and effective. Docstrings are excluded from the scan below only
        where they *explain* the prohibition; identifiers never may.
        """
        offenders: list[str] = []
        for directory in (CORE, KERNEL, ACQUISITION, PERCEPTION, TAXONOMY):
            for path in _python_files(directory):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    name = None
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                        name = node.name
                    elif isinstance(node, ast.Name):
                        name = node.id
                    elif isinstance(node, ast.arg):
                        name = node.arg
                    elif isinstance(node, ast.Attribute):
                        name = node.attr
                    if not name:
                        continue
                    leaked = _tokens(name) & set(DOMAIN_VOCABULARY)
                    if leaked:
                        offenders.append(
                            f"{_module_of(path)}::{name} uses {sorted(leaked)}"
                        )
        assert not offenders, (
            "domain knowledge has leaked into the platform (V1/V2):\n" + "\n".join(offenders)
        )

    def test_config_schema_is_closed(self) -> None:
        """A vertical enters as data through declared channels, never as a rule.

        The set grows by flow — Flow 2 added taxonomy and detectors — but only by
        a reviewed schema change, which is exactly what "closed" means here.
        """
        assert ALLOWED_TOP_LEVEL == frozenset(SECTION_TYPES) | {
            "profiles",
            "regions",
            "cameras",
            "taxonomy",
            "detectors",
        }

    def test_no_config_section_admits_a_business_threshold(self) -> None:
        forbidden = ("rule", "alert", "threshold_seconds", "policy_action", "violation")
        offenders: list[str] = []
        for section in SECTION_TYPES:
            for key in allowed_keys(section):
                for term in forbidden:
                    if term in key.lower():
                        offenders.append(f"{section}.{key}")
        assert not offenders, "\n".join(offenders)

    def test_region_carries_geometry_and_an_opaque_label_only(self) -> None:
        from vision_os.core.model.region import Region

        fields = set(Region.__dataclass_fields__)
        assert fields == {
            "region_id", "geometry", "frame_of_reference", "label", "camera_id", "version"
        }, "a Region must never acquire a semantic field such as zone_type or purpose"


class TestFlowScope:
    """No flow may implement responsibilities belonging to a later one.

    These assertions move forward exactly one flow at a time. Flow 7 shipped the
    Observation Builder and Vision State, so the frontier is now Flow 8
    (exposure) — and the guards below police *that* boundary, not the ones
    already crossed.
    """

    def test_only_implemented_ports_are_bindable(self) -> None:
        assert len(FLOW1_PORTS) == 11
        assert len(FLOW2_PORTS) == 4
        assert len(FLOW3_PORTS) == 1
        assert len(FLOW4_PORTS) == 1
        assert len(FLOW5_PORTS) == 3
        assert len(FLOW6_PORTS) == 2
        assert len(FLOW7_PORTS) == 3
        assert len(FLOW8_PORTS) == 3
        assert BINDABLE_PORTS == (
            FLOW1_PORTS
            | FLOW2_PORTS
            | FLOW3_PORTS
            | FLOW4_PORTS
            | FLOW5_PORTS
            | FLOW6_PORTS
            | FLOW7_PORTS
            | FLOW8_PORTS
        )
        assert len(BINDABLE_PORTS) == 28, "28 of the catalogue's 32 ports are bound"

        # Phase 1 is complete, so these four are no longer "later flow" — they
        # are what Phase 1 deliberately omits (15_ROADMAP section 2), each with
        # its port already defined and unused.
        deliberately_unbound = {
            "P10.EmbeddingPort",
            "P11.IdentityResolverPort",
            "P17.PromptSourcePort",
            "P28.CalibrationPort",
        }
        assert not (deliberately_unbound & set(BINDABLE_PORTS)), (
            "a port Phase 1 deliberately omits became bindable"
        )

    def test_identity_resolution_stays_unbindable_though_the_registry_ships(
        self,
    ) -> None:
        """P11 is M7's port, yet must **not** become bindable when M7 ships.

        15_ROADMAP section 3: *"already specified, no implementations in Phase
        1"*. M7's native spatio-temporal binding is mandatory behaviour needing
        no adapter; P11 is the seam for cross-camera identity, which is Phase 2,
        classified C2, and policy-gated.
        """
        assert "P11.IdentityResolverPort" not in BINDABLE_PORTS

    def test_embedding_stays_unbindable_even_though_tracking_ships(self) -> None:
        """P10 is a Flow 3-adjacent port that must **not** become bindable.

        Appearance embeddings are C2 biometric data, disabled by default
        (12_SECURITY section 4.3). Tracking shipping is not a reason to enable
        the platform's most invasive capability.
        """
        assert "P10.EmbeddingPort" not in BINDABLE_PORTS

    def test_the_object_ontology_stays_closed(self) -> None:
        """02_VOM's kinds are all present, and nothing beyond them is.

        The earlier form of this guard tested ``hasattr(core.model, "Crop")``,
        which passed for the wrong reason: the package's ``__init__`` never
        re-exported those names, so the assertion held whether or not the kind
        existed. It checks the modules now.

        The forbidden names are the ones the semantic ceiling forbids outright
        (V1). ``alert.py`` or ``incident.py`` appearing in the *core model* would
        mean the platform had grown an opinion about what its observations mean,
        and no flow may add one — not Flow 8, not ever.
        """
        model_dir = ROOT / "core" / "model"
        for kind in ("detection", "track", "visual_object", "crop", "observation"):
            assert (model_dir / f"{kind}.py").exists(), (
                f"{kind}.py is a shipped object kind and must exist"
            )
        for forbidden in ("alert", "incident", "rule", "person", "employee", "vehicle"):
            assert not (model_dir / f"{forbidden}.py").exists(), (
                f"core/model/{forbidden}.py is a business concept; the semantic "
                "ceiling forbids the platform from holding one"
            )

    def test_no_out_of_scope_module_exists(self) -> None:
        """Every Phase 1 layer ships. What stays absent is what V1 forbids.

        ``reasoning``, ``analytics``, ``learning`` and ``rules`` are not later
        flows — they are conclusions, and 07_STATE section 10 places every one of
        them in a consumer system. No phase of this platform adds them.
        """
        assert (ROOT / "synthesis").exists(), "Flow 7 implements the Observation Builder"
        assert (ROOT / "state").exists(), "Flow 7 implements Vision State"
        assert (ROOT / "exposure").exists(), "Flow 8 implements the Observation API"
        for absent in ("reasoning", "analytics", "learning", "rules", "alerts"):
            assert not (ROOT / absent).exists(), (
                f"package '{absent}' names a conclusion the platform may never draw"
            )
        for absent in ("synthesis", "state"):
            assert not (ROOT / "perception" / absent).exists(), (
                f"perception/{absent} would put an L5/L6 concern inside L2"
            )

    def test_detection_holds_no_temporal_state(self) -> None:
        """Detection is memoryless by construction (port obligation D7).

        A detector that remembered a previous frame would be doing tracking, and
        the boundary between Flow 2 and Flow 3 would already have dissolved.
        """
        detection_root = ROOT / "perception" / "detection"
        forbidden = ("track_id", "object_id", "TrackId", "ObjectId", "previous_frame")
        offenders: list[str] = []
        for path in _python_files(detection_root):
            text = path.read_text(encoding="utf-8")
            for term in forbidden:
                if term in text:
                    offenders.append(f"{_module_of(path)} references {term}")
        assert not offenders, (
            "detection must hold no identity or temporal state:\n" + "\n".join(offenders)
        )

    def test_no_observation_types_are_emitted(self) -> None:
        """Coverage *observations* are the Observation Builder's job (Flow 6).

        Flows 1 and 2 produce observability state, events and detections only.
        """
        offenders: list[str] = []
        for directory in (KERNEL, ACQUISITION, PERCEPTION):
            for path in _python_files(directory):
                text = path.read_text(encoding="utf-8")
                if "build_coverage" in text or "ObservationBuilder" in text:
                    offenders.append(_module_of(path))
        assert not offenders, "\n".join(offenders)


class TestSecretHygiene:
    def test_no_hardcoded_credentials_in_platform_code(self) -> None:
        pattern = re.compile(r"(password|passwd|secret_key|api_key)\s*=\s*[\"'][^\"']+[\"']")
        offenders: list[str] = []
        for directory in (CORE, KERNEL, ACQUISITION, ADAPTERS, PERCEPTION, TAXONOMY):
            for path in _python_files(directory):
                if pattern.search(path.read_text(encoding="utf-8")):
                    offenders.append(_module_of(path))
        assert not offenders, "\n".join(offenders)


class TestBoundedByConstruction:
    def test_no_unbounded_queue_constructs(self) -> None:
        """An unbounded queue is a memory leak with a delayed fuse."""
        offenders: list[str] = []
        for directory in (KERNEL, ACQUISITION, PERCEPTION):
            for path in _python_files(directory):
                text = path.read_text(encoding="utf-8")
                if "asyncio.Queue()" in text or "Queue(maxsize=0)" in text:
                    offenders.append(_module_of(path))
        assert not offenders, (
            "queues must be bounded with a declared overflow policy:\n" + "\n".join(offenders)
        )


@pytest.mark.parametrize(
    "package",
    ["core", "kernel", "acquisition", "adapters", "conformance", "perception", "taxonomy"],
)
def test_every_package_has_a_module_docstring(package: str) -> None:
    """A module still understandable after five years starts with why it exists."""
    missing: list[str] = []
    for path in _python_files(ROOT / package):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not ast.get_docstring(tree):
            missing.append(_module_of(path))
    assert not missing, "missing module docstrings:\n" + "\n".join(missing)
