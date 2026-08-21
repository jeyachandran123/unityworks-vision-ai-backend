"""Architecture guards specific to the detection layer.

The generic boundary tests already police core purity, the dependency law, the
injected clock and domain vocabulary across every package including perception.
These add the guarantees that are *about detection*: that it holds no memory, no
identity and no meaning, that YOLO is invisible above the adapter, and that the
Semantic Ceiling still holds now that the platform names object classes.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import vision_os as vision_os_pkg
from vision_os.core.model.detection import Detection
from vision_os.core.ports.detection import DetectorPort, RawDetection
from vision_os.kernel.plugins.manifest import BINDABLE_PORTS, PortCatalogue

ROOT = Path(vision_os_pkg.__file__).parent
PERCEPTION = ROOT / "perception"
DETECTION = PERCEPTION / "detection"
KERNEL = ROOT / "kernel"
CORE = ROOT / "core"
TAXONOMY_PKG = ROOT / "taxonomy"
ADAPTERS = ROOT / "adapters"


def _files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


def _module_of(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _identifiers(path: Path) -> set[str]:
    """Every name a module *defines or uses* — excluding prose.

    Docstrings naming example implementations are documentation, not coupling;
    what matters is whether code can reach a concrete type.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            names.update(alias.name for alias in node.names)
    return names


def _tokens(identifier: str) -> set[str]:
    return {part.lower() for part in identifier.replace(".", "_").split("_") if part}


class TestPlatformDoesNotKnowYoloExists:
    """Invariant V3, at its most concrete."""

    def test_no_platform_module_names_a_detector_vendor(self) -> None:
        vendors = (
            "yolo", "ultralytics", "rtdetr", "rt_detr", "dino", "detectron",
            "mmdet", "torchvision", "openvino", "tensorrt",
        )
        offenders: list[str] = []
        for directory in (CORE, KERNEL, PERCEPTION, TAXONOMY_PKG):
            for path in _files(directory):
                for identifier in _identifiers(path):
                    lowered = identifier.lower()
                    for vendor in vendors:
                        if vendor in lowered:
                            offenders.append(
                                f"{_module_of(path)}::{identifier} names '{vendor}'"
                            )
        assert not offenders, (
            "the platform must not be able to reach a concrete detector:\n"
            + "\n".join(offenders)
        )

    def test_only_the_composition_root_names_a_concrete_detector(self) -> None:
        """One function in the codebase may say the word YOLO."""
        named: list[str] = []
        for path in _files(ROOT):
            module = _module_of(path)
            if module.startswith("adapters/") or module == "detection_bootstrap.py":
                continue
            if "YoloDetector" in path.read_text(encoding="utf-8"):
                named.append(module)
        assert not named, "\n".join(named)

    def test_detection_layer_imports_no_adapter(self) -> None:
        offenders: list[str] = []
        for path in _files(PERCEPTION):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if "adapters" in node.module:
                        offenders.append(f"{_module_of(path)} imports {node.module}")
                elif isinstance(node, ast.ImportFrom) and node.level:
                    continue
        assert not offenders, "\n".join(offenders)

    def test_engine_holds_a_port_not_a_model(self) -> None:
        """The engine's detector is typed as the protocol, never a concrete class."""
        from vision_os.perception.detection.binding import DetectorBinding

        annotation = DetectorBinding.__dataclass_fields__["detector"].type
        assert "DetectorPort" in str(annotation)


class TestDetectionIsMemoryless:
    """Port obligation D7, and the Flow 2 / Flow 3 boundary."""

    def test_detection_carries_no_identity(self) -> None:
        fields = set(Detection.__dataclass_fields__)
        for forbidden in ("track_id", "object_id", "identity", "history", "age_frames"):
            assert forbidden not in fields

    def test_raw_detection_carries_no_identity(self) -> None:
        fields = set(RawDetection.__dataclass_fields__)
        for forbidden in ("track_id", "object_id", "previous"):
            assert forbidden not in fields

    def test_detector_port_has_no_temporal_method(self) -> None:
        """A detector that could be told about the past would be a tracker."""
        methods = {
            name
            for name, _ in inspect.getmembers(DetectorPort, inspect.isfunction)
            if not name.startswith("_")
        }
        assert methods == {"capabilities", "detect", "warm", "health"}

    def test_detection_layer_holds_no_cross_frame_state(self) -> None:
        """Names that would only exist if something were remembered per frame.

        Deliberately specific. A broad token like "last" would flag
        ``_last_report_ns`` — a health-reporting timestamp that says nothing
        about frames — and a guard with false positives is one people learn to
        ignore.
        """
        forbidden = {
            "track", "tracks", "track_id", "tracker", "trajectory",
            "object_id", "previous_frame", "last_frame", "frame_history",
            "prior_detections", "frame_memory",
        }
        offenders: list[str] = []
        for path in _files(DETECTION):
            for identifier in _identifiers(path):
                normalized = identifier.lower().lstrip("_")
                if normalized in forbidden:
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, (
            "detection must not remember across frames:\n" + "\n".join(offenders)
        )


class TestSemanticCeilingWithClasses:
    """Naming object classes is exactly where V1 gets tested hardest."""

    def test_taxonomy_rejects_roles_and_judgments(self) -> None:
        from vision_os.kernel.config.schema import _FORBIDDEN_CLASS_TOKENS

        for term in ("patient", "employee", "customer", "suspect", "violation"):
            assert term in _FORBIDDEN_CLASS_TOKENS

    def test_no_business_conclusion_in_the_detection_contract(self) -> None:
        """A detection states what is visible, never what it implies."""
        fields = set(Detection.__dataclass_fields__)
        for forbidden in (
            "alert", "violation", "severity", "risk", "priority",
            "action", "recommendation", "compliant",
        ):
            assert forbidden not in fields

    def test_detection_layer_uses_no_domain_vocabulary(self) -> None:
        domain = (
            "restaurant", "kitchen", "hospital", "warehouse", "retail",
            "patient", "customer", "employee", "waiter", "shopper",
            "violation", "alert", "suspicious", "loitering",
        )
        offenders: list[str] = []
        for path in _files(DETECTION):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                name = None
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    name = node.name
                elif isinstance(node, ast.Name):
                    name = node.id
                elif isinstance(node, ast.arg):
                    name = node.arg
                if not name:
                    continue
                tokens = {t.lower() for t in name.replace(".", "_").split("_")}
                leaked = tokens & set(domain)
                if leaked:
                    offenders.append(f"{_module_of(path)}::{name} uses {sorted(leaked)}")
        assert not offenders, "\n".join(offenders)

    def test_there_is_no_false_detection_metric(self) -> None:
        """Whether a detection is false requires ground truth the platform lacks.

        The platform counts what it *rejected* and why, which is knowable.
        """
        from vision_os.kernel.metrics.names import ALL_METRIC_NAMES, MetricName

        assert not any("false" in name for name in ALL_METRIC_NAMES)
        assert MetricName.DETECTIONS_REJECTED in ALL_METRIC_NAMES


class TestModelManagerKnowsNothingOfVision:
    """The kernel law: M18 knows weights, memory, devices and versions."""

    def test_model_manager_has_no_vision_vocabulary(self) -> None:
        vision_terms = ("detection", "detector", "frame", "camera", "track", "crop")
        offenders: list[str] = []
        for path in _files(KERNEL / "models"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                name = None
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    name = node.name
                elif isinstance(node, ast.arg):
                    name = node.arg
                if not name:
                    continue
                tokens = {t.lower() for t in name.split("_")}
                leaked = tokens & set(vision_terms)
                if leaked:
                    offenders.append(f"{_module_of(path)}::{name} uses {sorted(leaked)}")
        assert not offenders, (
            "M18 must serve model kinds that do not exist yet:\n" + "\n".join(offenders)
        )

    def test_model_manager_imports_no_flow_layer(self) -> None:
        offenders: list[str] = []
        for path in _files(KERNEL / "models"):
            text = path.read_text(encoding="utf-8")
            for forbidden in ("perception", "acquisition", "adapters", "taxonomy"):
                if f"vision_os.{forbidden}" in text or f"...{forbidden}" in text:
                    offenders.append(f"{_module_of(path)} references {forbidden}")
        assert not offenders, "\n".join(offenders)


class TestFlowScope:
    def test_detection_ports_are_bindable(self) -> None:
        for port in (
            PortCatalogue.DETECTOR,
            PortCatalogue.ARTIFACT_STORE,
            PortCatalogue.MODEL_RUNTIME,
            PortCatalogue.DEVICE,
        ):
            assert port in BINDABLE_PORTS

    def test_phase_two_ports_remain_unbindable(self) -> None:
        """Phase 1 is complete. These four stay unbound, each for its own reason.

        ``EMBEDDING`` and ``IDENTITY_RESOLVER`` are the biometric and
        cross-camera-identity capabilities, disabled by default (12_SECURITY
        section 4.3, 15_ROADMAP Phase 2). ``PROMPT_SOURCE`` belongs to M10, which
        no flow implemented. ``CALIBRATION`` belongs to M1 and M18.
        """
        for port in (
            PortCatalogue.EMBEDDING,
            PortCatalogue.IDENTITY_RESOLVER,
            PortCatalogue.PROMPT_SOURCE,
            PortCatalogue.CALIBRATION,
        ):
            assert port not in BINDABLE_PORTS

    def test_detection_does_not_import_exposure(self) -> None:
        """L2 must never learn L7 exists.

        The API reads state; detection produces boxes. An import here would be
        the sharpest possible violation of the dependency law.
        """
        offenders = [
            path.name
            for path in (ROOT / "perception" / "detection").rglob("*.py")
            if "exposure" in path.read_text(encoding="utf-8")
        ]
        assert not offenders, "; ".join(offenders)

    def test_detection_does_not_import_tracking(self) -> None:
        """Flow 2 must not learn that Flow 3 exists."""
        offenders: list[str] = []
        for path in _files(DETECTION):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and "tracking" in node.module:
                    offenders.append(f"{_module_of(path)} imports {node.module}")
        assert not offenders, "\n".join(offenders)

    def test_perception_emits_no_observations(self) -> None:
        """Observations are Flow 6. L2 emits detections, tracks, objects, events.

        Matched on identifiers rather than substrings: ``ClassObservation`` and
        ``MotionObservation`` are unrelated types whose names merely contain the
        word, and a guard that flags them is one people learn to suppress.
        """
        forbidden = {"Observation", "ObservationBuilder", "ObservationId", "Evidence"}
        offenders: list[str] = []
        for path in _files(PERCEPTION):
            for identifier in _identifiers(path) & forbidden:
                offenders.append(f"{_module_of(path)} references {identifier}")
        assert not offenders, "\n".join(offenders)

    def test_detection_writes_no_state(self) -> None:
        """Vision State is Flow 7; detection persists nothing."""
        forbidden = {
            "statestore", "observationlog", "visionstate", "evidencestore",
            "persist", "checkpoint",
        }
        offenders: list[str] = []
        for path in _files(DETECTION):
            for identifier in _identifiers(path):
                normalized = identifier.lower().lstrip("_")
                if normalized in forbidden:
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, (
            "detection persists nothing; Vision State is Flow 7:\n" + "\n".join(offenders)
        )


class TestSeamIsMinimal:
    """Flow 1 gained exactly one optional collaborator, typed as a protocol."""

    def test_runtime_holds_the_protocol_not_a_detection_type(self) -> None:
        from vision_os.kernel.runtime import runtime as runtime_module

        source = Path(runtime_module.__file__).read_text(encoding="utf-8")
        assert "AdmittedFrameConsumer" in source
        for forbidden in ("DetectionEngine", "DetectionRuntime", "Detection("):
            assert forbidden not in source, (
                f"the Flow 1 runtime must not name {forbidden}"
            )

    def test_consumer_is_optional(self) -> None:
        from vision_os.kernel.runtime import VisionRuntime

        signature = inspect.signature(VisionRuntime.__init__)
        parameter = signature.parameters["admitted_frame_consumer"]
        assert parameter.default is None, (
            "detection must be opt-in, so a Flow 1 deployment is unchanged"
        )

    def test_seam_carries_a_reference_not_a_frame(self) -> None:
        """Control-plane sized, so the seam works across a process boundary (V12)."""
        from vision_os.core.ports.pipeline import AdmittedFrameConsumer

        signature = inspect.signature(AdmittedFrameConsumer.on_admitted)
        assert "frame_ref" in signature.parameters
        assert "frame" not in signature.parameters


@pytest.mark.parametrize("package", ["perception", "taxonomy"])
def test_every_new_module_has_a_docstring(package: str) -> None:
    missing: list[str] = []
    for path in _files(ROOT / package):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not ast.get_docstring(tree):
            missing.append(_module_of(path))
    assert not missing, "\n".join(missing)
