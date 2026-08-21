"""Architecture guards specific to the Crop Manager.

The generic boundary tests already police core purity, the dependency law, the
injected clock and domain vocabulary across every package including perception.
These add the guarantees that are *about M8*: that it understands nothing,
identifies nobody, writes no state, and preprocesses for no particular model.

The brief for this flow is unusually explicit about the last one:

> *"No YOLO crop. No CLIP crop. No Florence crop. No Qwen crop. No InternVL
> crop."*

so ``test_no_model_specific_preprocessing`` reads the source for those names and
for the operations that would imply them.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import vision_os as vision_os_pkg
from vision_os.core.model.crop import Crop, CropRequest
from vision_os.core.ports.cropping import TriggerCandidate
from vision_os.kernel.plugins.manifest import BINDABLE_PORTS, PortCatalogue

ROOT = Path(vision_os_pkg.__file__).parent
PERCEPTION = ROOT / "perception"
CROPPING = PERCEPTION / "cropping"
ADAPTERS = ROOT / "adapters" / "cropping"
CORE = ROOT / "core"
KERNEL = ROOT / "kernel"


def _files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


def _module_of(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _identifiers(path: Path) -> set[str]:
    """Every name a module defines or uses — excluding prose.

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


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.append(node.module)
        elif isinstance(node, ast.ImportFrom) and node.level:
            found.append("." * node.level + (node.module or ""))
    return found


class TestM8UnderstandsNothing:
    """The L3/L4 boundary. M8 prepares evidence; it never reads it."""

    def test_no_inference_vocabulary(self) -> None:
        # ``reason`` is deliberately absent: it is the field name on every
        # ``TriggerReason``, ``SkipReason`` and ``GateRejection``, and those are
        # what make V8 true. The forbidden verb is ``reasoning`` — drawing a
        # conclusion — not recording why something happened.
        forbidden = {
            "classify", "classifier", "caption", "captioning", "ocr",
            "embed", "embedding", "infer", "inference", "predict", "prediction",
            "vlm", "prompt", "understand", "understanding", "recognise",
            "recognize", "reasoning", "reason_about", "learn", "learned",
        }
        offenders: list[str] = []
        for path in _files(CROPPING) + _files(ADAPTERS):
            for identifier in _identifiers(path):
                normalized = identifier.lower().lstrip("_")
                if normalized in forbidden:
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, (
            "M8 prepares evidence and never interprets it:\n" + "\n".join(offenders)
        )

    def test_no_identity_vocabulary(self) -> None:
        """*"M8 owns images. Not identities."*

        No biometric recognition, no face recognition, no person recognition, no
        identity persistence.
        """
        forbidden = {
            "face", "faces", "biometric", "identity", "identify", "person_id",
            "gallery", "reid", "re_id", "match_identity", "enrol", "enroll",
        }
        offenders: list[str] = []
        for path in _files(CROPPING) + _files(ADAPTERS):
            for identifier in _identifiers(path):
                normalized = identifier.lower().lstrip("_")
                if normalized in forbidden:
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, (
            "M8 owns images, not identities:\n" + "\n".join(offenders)
        )

    def test_no_observation_or_business_vocabulary(self) -> None:
        forbidden = {
            "observation", "observations", "alert", "alerts", "violation",
            "incident", "severity", "risk", "employee", "customer", "staff",
            "compliance", "verdict",
        }
        offenders: list[str] = []
        for path in _files(CROPPING) + _files(ADAPTERS):
            for identifier in _identifiers(path):
                normalized = identifier.lower().lstrip("_")
                if normalized in forbidden:
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, (
            "M8 builds no observations and reaches no business conclusions:\n"
            + "\n".join(offenders)
        )

    def test_the_crop_carries_no_interpretation(self) -> None:
        """A crop is pixels plus provenance, never a claim about them."""
        fields = set(Crop.__dataclass_fields__)
        for forbidden in (
            "label", "class_name", "caption", "description", "attributes",
            "embedding", "identity", "alert", "conclusion",
        ):
            assert forbidden not in fields, f"Crop must not carry '{forbidden}'"

    def test_the_trigger_candidate_carries_no_meaning(self) -> None:
        """A policy cannot breach the ceiling because it is never handed the
        material to breach it with."""
        fields = set(TriggerCandidate.__dataclass_fields__)
        for forbidden in (
            "region_label", "region_type", "class_name", "zone_purpose",
            "business_context", "role",
        ):
            assert forbidden not in fields


class TestNoModelSpecificPreprocessing:
    """*"No YOLO crop. No CLIP crop. No Florence crop."* — one crop format."""

    def test_no_model_name_appears_in_the_crop_path(self) -> None:
        vendors = (
            "yolo", "clip", "florence", "qwen", "internvl", "llava", "siglip",
            "dinov", "blip", "openclip", "ultralytics", "sam2",
        )
        offenders: list[str] = []
        for path in _files(CROPPING) + _files(ADAPTERS) + [CORE / "ports" / "cropping.py"]:
            for identifier in _identifiers(path):
                lowered = identifier.lower()
                for vendor in vendors:
                    if vendor in lowered:
                        offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, (
            "the crop format is model-agnostic; preprocessing belongs to the "
            "model adapter downstream:\n" + "\n".join(offenders)
        )

    def test_no_normalization_vocabulary(self) -> None:
        """Mean subtraction and tensor layout belong to M9's model adapter.

        A crop that arrives pre-normalized for one model is unusable by another,
        which quietly undoes the whole point of a canonical format.
        """
        forbidden = {
            "mean_subtract", "imagenet_mean", "imagenet_std", "normalize_tensor",
            "to_tensor", "nchw", "nhwc", "chw", "hwc", "std_dev_normalize",
        }
        offenders: list[str] = []
        for path in _files(CROPPING) + _files(ADAPTERS):
            for identifier in _identifiers(path):
                if identifier.lower().lstrip("_") in forbidden:
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, "\n".join(offenders)

    def test_the_crop_size_is_one_configured_value(self) -> None:
        from vision_os.kernel.config.schema import CroppingSection

        settings = CroppingSection()
        assert settings.crop_width > 0 and settings.crop_height > 0
        fields = set(CroppingSection.__dataclass_fields__)
        for per_model in ("yolo_size", "clip_size", "vlm_size", "sizes_by_model"):
            assert per_model not in fields


class TestM8OwnsNoState:
    """*"M8 does NOT own Vision State. M8 owns crop lifecycle only."*"""

    def test_no_durable_store_is_held(self) -> None:
        """Trigger state is ephemeral and node-local by §M8's own words.

        Persisting it would create a second writer of something the registry
        already owns, and a stale record after a restart would suppress the
        analysis the restart made necessary.
        """
        # ``restore`` is absent from this set on purpose: ``DemandRegistry.restore``
        # returns a throttled demand to active service, which is a lifecycle
        # transition in memory and not a read from disk.
        forbidden = {"save", "load", "persist", "flush", "snapshot"}
        offenders: list[str] = []
        for path in _files(CROPPING):
            for identifier in _identifiers(path):
                if identifier.lower().lstrip("_") in forbidden:
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, (
            "M8's state is ephemeral; a restart costs one round of FIRST_SIGHT "
            "and nothing more:\n" + "\n".join(offenders)
        )

    def test_nothing_in_the_crop_path_touches_the_filesystem(self) -> None:
        """The stronger form of the same rule, checked structurally.

        A vocabulary guard can be worked around by naming a method something
        else; an import guard cannot, because writing a file requires reaching
        for something that writes files.
        """
        forbidden_modules = {"pathlib", "os", "shutil", "sqlite3", "pickle", "shelve"}
        offenders: list[str] = []
        for path in _files(CROPPING) + _files(ADAPTERS):
            for module in _imports(path):
                if module.split(".")[0] in forbidden_modules:
                    offenders.append(f"{_module_of(path)} imports {module}")
        assert not offenders, (
            "M8 holds no durable store; persisting trigger state would create a "
            "second writer of what the registry already owns:\n"
            + "\n".join(offenders)
        )

    def test_no_store_port_is_held(self) -> None:
        offenders: list[str] = []
        for path in _files(CROPPING) + _files(ADAPTERS):
            source = path.read_text(encoding="utf-8")
            for port in ("ObjectStorePort", "EvidenceStorePort", "StateStorePort"):
                if port in source:
                    offenders.append(f"{_module_of(path)} names {port}")
        assert not offenders, "\n".join(offenders)

    def test_the_crop_manager_never_writes_an_object(self) -> None:
        """M7 is the only writer of Vision Objects (Flow 4's ownership rule)."""
        offenders: list[str] = []
        for path in _files(CROPPING) + _files(ADAPTERS):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ("VisualObject", "Attribute", "ObjectId"):
                        offenders.append(f"{_module_of(path)} constructs {node.func.id}")
        assert not offenders, (
            "only M7 may mint objects and only M9 may produce attributes:\n"
            + "\n".join(offenders)
        )

    def test_no_object_id_is_minted(self) -> None:
        offenders: list[str] = []
        for path in _files(CROPPING) + _files(ADAPTERS):
            if "new_ulid" in path.read_text(encoding="utf-8"):
                offenders.append(_module_of(path))
        # The demand registry mints DemandIds, which are M8's own.
        assert offenders in ([], ["perception/cropping/demands.py"]), (
            f"unexpected id minting in {offenders}"
        )


class TestLayering:
    def test_cropping_imports_no_adapter(self) -> None:
        offenders: list[str] = []
        for path in _files(CROPPING):
            for module in _imports(path):
                if "adapters" in module and not module.startswith("."):
                    offenders.append(f"{_module_of(path)} imports {module}")
        assert not offenders, "\n".join(offenders)

    def test_cropping_does_not_import_the_tracking_or_detection_modules(self) -> None:
        """M8 consumes objects from M7. It never reaches past them.

        Reaching back to a track would re-introduce the fragile identity V10
        exists to keep separate from the durable one.

        Scoped to the *perception layer* modules. ``core.model.detection`` is a
        shared model — it is where ``QualityGrades`` lives, and quality is a
        platform vocabulary the whole pipeline shares, not a detector's private
        type.
        """
        offenders: list[str] = []
        for path in _files(CROPPING):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                target = node.module
                if node.level:
                    # Relative: ``..tracking.engine`` reaches a sibling layer,
                    # ``...core.model.detection`` reaches the shared model.
                    if target.startswith(("tracking", "detection")):
                        offenders.append(f"{_module_of(path)} imports {target}")
                elif "perception.tracking" in target or "perception.detection" in target:
                    offenders.append(f"{_module_of(path)} imports {target}")
        assert not offenders, "\n".join(offenders)

    def test_the_registry_does_not_import_cropping(self) -> None:
        """The dependency runs one way. M7 holds a callable and learns nothing."""
        offenders: list[str] = []
        for path in _files(PERCEPTION / "registry"):
            for module in _imports(path):
                if "cropping" in module:
                    offenders.append(f"{_module_of(path)} imports {module}")
        assert not offenders, "\n".join(offenders)

    def test_the_kernel_does_not_import_cropping(self) -> None:
        offenders: list[str] = []
        for path in _files(KERNEL):
            for module in _imports(path):
                if "cropping" in module:
                    offenders.append(f"{_module_of(path)} imports {module}")
        assert not offenders, "\n".join(offenders)

    def test_only_the_composition_root_names_a_concrete_adapter(self) -> None:
        """One module in the codebase may say ``HeuristicQualityEstimator``."""
        named: list[str] = []
        for path in _files(ROOT):
            module = _module_of(path)
            if module.startswith("adapters/") or module == "cropping_bootstrap.py":
                continue
            source = path.read_text(encoding="utf-8")
            for concrete in (
                "HeuristicQualityEstimator",
                "PaddedCropStrategy",
                "TightCropStrategy",
                "ReferenceCropExtractor",
                "DefaultTriggerPolicy",
            ):
                if concrete in source:
                    named.append(f"{module} names {concrete}")
        assert not named, "\n".join(named)

    def test_the_engine_holds_ports_not_implementations(self) -> None:
        from vision_os.perception.cropping import CropManager

        signature = inspect.signature(CropManager.__init__)
        for name in ("policy", "estimator", "strategy", "extractor"):
            assert name in signature.parameters, f"{name} must be injected"


class TestSemanticCeiling:
    def test_priority_is_an_opaque_string(self) -> None:
        """The platform orders by it and never interprets it (V1/V2)."""
        annotation = str(CropRequest.__dataclass_fields__["priority_class"].type)
        assert "str" in annotation
        assert "enum" not in annotation.lower(), (
            "a priority *enum* would be the platform declaring which classes may "
            "exist, which is a business decision"
        )

    def test_no_region_semantics_reach_the_crop_path(self) -> None:
        forbidden = {"zone_type", "region_type", "region_purpose", "zone_purpose"}
        offenders: list[str] = []
        for path in _files(CROPPING) + _files(ADAPTERS):
            for identifier in _identifiers(path):
                if identifier.lower() in forbidden:
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, "\n".join(offenders)

    def test_the_config_section_has_no_business_slot(self) -> None:
        from vision_os.kernel.config.schema import CroppingSection

        fields = set(CroppingSection.__dataclass_fields__)
        for forbidden in (
            "important_regions", "vip_classes", "alert_on", "roles",
            "always_analyse", "compliance_mode",
        ):
            assert forbidden not in fields


class TestFlowScope:
    """No flow may implement responsibilities belonging to a later one.

    Flow 5 shipped the Crop Manager, so the frontier is now Flow 6.
    """

    def test_the_three_crop_ports_are_bindable(self) -> None:
        for port in (
            PortCatalogue.TRIGGER_POLICY,
            PortCatalogue.QUALITY_ESTIMATOR,
            PortCatalogue.CROP_STRATEGY,
        ):
            assert port in BINDABLE_PORTS

    def test_phase_two_ports_remain_unbindable(self) -> None:
        """Phase 1 is complete; these four are what it deliberately omits.

        Not a moving frontier any more. 15_ROADMAP section 2 lists each as
        omitted with the port already defined and unused.
        """
        for port in (
            PortCatalogue.EMBEDDING,
            PortCatalogue.IDENTITY_RESOLVER,
            PortCatalogue.PROMPT_SOURCE,
            PortCatalogue.CALIBRATION,
        ):
            assert port not in BINDABLE_PORTS, f"{port} became bindable early"

    def test_cropping_does_not_bind_the_evidence_store_itself(self) -> None:
        """M8 decides retention policy; persisting imagery is M13's job.

        P22 became bindable in Flow 8, which is M13 — but M8 must still not reach
        it. Binding a store *here* would put a durable side effect inside the
        platform's cheapest, hottest path, which is what this guard has always
        been about. The check moved from "the port is unbound" to "this layer
        does not touch it", because the first stopped being true and the second
        is what actually mattered.
        """
        offenders = [
            path.name
            for path in (ROOT / "perception" / "cropping").rglob("*.py")
            if "EvidenceStore" in path.read_text(encoding="utf-8")
        ]
        assert not offenders, "; ".join(offenders)

    def test_the_biometric_ports_stay_unbindable(self) -> None:
        """A standing guard, not a frontier one (12_SECURITY section 4.3)."""
        assert PortCatalogue.EMBEDDING not in BINDABLE_PORTS
        assert PortCatalogue.IDENTITY_RESOLVER not in BINDABLE_PORTS

    def test_no_phase_two_module_exists(self) -> None:
        """L7 shipped in Flow 8; what stays absent is Phase 2 and beyond."""
        assert (ROOT / "exposure").exists(), "Flow 8 implements the Observation API"
        for absent in ("reasoning", "analytics", "learning", "rules"):
            assert not (ROOT / absent).exists(), (
                f"package '{absent}' names a conclusion the platform may never draw"
            )

    def test_cropping_does_not_import_synthesis(self) -> None:
        """M11 ships, but M8 must not learn it exists.

        L3 hands crops upward and never hears what became of them. An import
        here would be the beginning of attention deciding what is worth
        publishing, which is the Observation Builder's judgement, not M8's.
        """
        offenders = []
        for path in (ROOT / "perception" / "cropping").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for forbidden in ("synthesis", "vision_state", "observation"):
                if f"import {forbidden}" in source or f"from ...{forbidden}" in source:
                    offenders.append(f"{path.name} imports {forbidden}")
        assert not offenders, "\n".join(offenders)

    def test_cropping_does_not_import_understanding(self) -> None:
        """M9 ships, but M8 must not learn it exists.

        The dependency runs cropping-to-understanding, never the reverse: a crop
        manager that knew what a model concluded would be making attention
        decisions on semantic grounds, which is the ceiling breached from below.
        """
        offenders: list[str] = []
        for path in _files(CROPPING) + _files(ADAPTERS):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and "understanding" in node.module
                ):
                    offenders.append(f"{_module_of(path)} imports {node.module}")
        assert not offenders, "\n".join(offenders)


class TestEarlierFlowsUnchanged:
    def test_flow_one_does_not_import_cropping(self) -> None:
        offenders: list[str] = []
        for directory in (ROOT / "acquisition", KERNEL):
            for path in _files(directory):
                for module in _imports(path):
                    if "cropping" in module:
                        offenders.append(f"{_module_of(path)} imports {module}")
        assert not offenders, "\n".join(offenders)

    def test_detection_does_not_import_cropping(self) -> None:
        offenders: list[str] = []
        for path in _files(PERCEPTION / "detection"):
            for module in _imports(path):
                if "cropping" in module:
                    offenders.append(f"{_module_of(path)} imports {module}")
        assert not offenders, "\n".join(offenders)
