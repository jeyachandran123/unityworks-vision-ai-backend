"""Architecture and security guards specific to the Understanding Engine.

The generic boundary tests already police core purity, the dependency law, the
injected clock and domain vocabulary. These add the guarantees that are *about
M9*: that it detects nothing, tracks nothing, registers nothing, observes
nothing, writes no state, names no model, and authors no prompt.

The brief for this flow is explicit about the last two:

> *Never hard-code model names. Never couple M9 to any AI model.*
> *M9 consumes prompts. M9 never creates prompts.*

so ``test_no_model_name_appears_in_the_platform`` and
``test_no_prompt_text_lives_in_the_engine`` read the source for exactly that.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import vision_os as vision_os_pkg
from vision_os.core.model.understanding import (
    UnderstandingRequest,
    UnderstandingResult,
)
from vision_os.core.ports.understanding import (
    CropView,
    UnderstandingPortRequest,
)
from vision_os.kernel.plugins.manifest import BINDABLE_PORTS, PortCatalogue

ROOT = Path(vision_os_pkg.__file__).parent
PERCEPTION = ROOT / "perception"
UNDERSTANDING = PERCEPTION / "understanding"
ADAPTERS = ROOT / "adapters" / "understanding"
CORE = ROOT / "core"
KERNEL = ROOT / "kernel"


def _files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


def _module_of(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _identifiers(path: Path) -> set[str]:
    """Every name a module defines or uses — excluding prose.

    Docstrings naming example models are documentation, not coupling; what
    matters is whether code can reach a concrete type.
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
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(("." * node.level if node.level else "") + node.module)
    return found


class TestM9PerformsNoPerception:
    def test_no_detection_or_tracking_vocabulary(self) -> None:
        """M9 receives a crop. It never finds one, never follows one."""
        forbidden = {
            "detect", "detector", "detection", "nms", "bbox", "anchor",
            "track", "tracker", "tracking", "trajectory", "kalman", "association",
            "iou", "bytetrack",
        }
        offenders: list[str] = []
        for path in _files(UNDERSTANDING) + _files(ADAPTERS):
            for identifier in _identifiers(path):
                if identifier.lower().lstrip("_") in forbidden:
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, (
            "M9 consumes crops; it performs no detection and no tracking:\n"
            + "\n".join(offenders)
        )

    def test_no_crop_generation_vocabulary(self) -> None:
        """M9 reads a crop it did not make and cannot re-make."""
        forbidden = {
            "extract_crop", "letterbox", "resize", "pad", "padding", "rectify",
            "crop_strategy", "gate", "quality_gate", "trigger",
        }
        offenders: list[str] = []
        for path in _files(UNDERSTANDING) + _files(ADAPTERS):
            for identifier in _identifiers(path):
                if identifier.lower().lstrip("_") in forbidden:
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, (
            "crop preparation belongs to M8:\n" + "\n".join(offenders)
        )

    def test_no_object_registration_vocabulary(self) -> None:
        """M7 is the only writer of Vision Objects (Flow 4's ownership rule)."""
        # ``merge`` and ``split`` are deliberately absent as bare tokens: they
        # are M7 *operations on objects*, but they are also ordinary verbs — the
        # JSON coercion strategy splits declared fields from undeclared ones and
        # that is not object surgery. The registry-shaped names are what matter.
        forbidden = {
            "mint", "register_object", "bind_track", "merge_objects",
            "split_object", "object_registry", "expire_stale", "apply_attribute",
        }
        offenders: list[str] = []
        for path in _files(UNDERSTANDING) + _files(ADAPTERS):
            for identifier in _identifiers(path):
                if identifier.lower().lstrip("_") in forbidden:
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, "\n".join(offenders)

    def test_the_engine_never_constructs_a_platform_object(self) -> None:
        """AST scan. M9 produces attribute *values*; M7 holds them."""
        forbidden = {"VisualObject", "Track", "Detection", "Crop", "ObjectId"}
        offenders: list[str] = []
        for path in _files(UNDERSTANDING) + _files(ADAPTERS):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in forbidden:
                        offenders.append(f"{_module_of(path)} constructs {node.func.id}")
        assert not offenders, (
            "M9 may name these types; it may not create them:\n" + "\n".join(offenders)
        )


class TestM9BuildsNoObservations:
    def test_no_observation_vocabulary(self) -> None:
        """`01_LAYERED` §1.2: L4/L5 is one of the three boundaries systems
        collapse, and the consequence is publishing *"whatever the VLM said"*."""
        # ``publish`` alone is absent: it is the Event Bus's method, and M9
        # publishing a *control-plane alarm* is required rather than forbidden.
        # What must not exist is publishing a **fact**.
        forbidden = {
            "observation", "observations", "observation_id", "envelope",
            "emit_observation", "publish_observation", "observation_builder",
        }
        offenders: list[str] = []
        for path in _files(UNDERSTANDING) + _files(ADAPTERS):
            for identifier in _identifiers(path):
                if identifier.lower().lstrip("_") in forbidden:
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, (
            "M11 is the single choke point for published facts:\n"
            + "\n".join(offenders)
        )

    def test_the_result_carries_no_observation_id(self) -> None:
        """M9 may not mint an identifier for an object it cannot create.

        02_VOM §10.9 declares ``observation_id`` on the completed evidence
        record; M11 stamps it when it assembles the observation this evidence
        explains.
        """
        from vision_os.core.model.understanding import UnderstandingEvidence

        assert "observation_id" not in UnderstandingEvidence.__dataclass_fields__
        assert "observation_id" not in UnderstandingResult.__dataclass_fields__

    def test_no_business_vocabulary(self) -> None:
        forbidden = {
            "alert", "alerts", "violation", "incident", "severity", "risk",
            "employee", "customer", "staff", "compliance", "verdict", "rule",
            "threshold_breach", "escalate",
        }
        offenders: list[str] = []
        for path in _files(UNDERSTANDING) + _files(ADAPTERS):
            for identifier in _identifiers(path):
                if identifier.lower().lstrip("_") in forbidden:
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, (
            "M9 reaches no business conclusion:\n" + "\n".join(offenders)
        )

    def test_the_result_carries_no_judgment_field(self) -> None:
        fields = set(UnderstandingResult.__dataclass_fields__)
        for forbidden in (
            "alert", "severity", "verdict", "conclusion", "action", "recommendation",
        ):
            assert forbidden not in fields


class TestM9OwnsNoWorldState:
    def test_it_holds_no_durable_store(self) -> None:
        """§M9: *"Owns no world state. Every call is a pure function of
        (crop, prompt, model)."*"""
        forbidden_modules = {"pathlib", "os", "shutil", "sqlite3", "pickle", "shelve"}
        offenders: list[str] = []
        for path in _files(UNDERSTANDING) + _files(ADAPTERS):
            for module in _imports(path):
                if module.split(".")[0] in forbidden_modules:
                    offenders.append(f"{_module_of(path)} imports {module}")
        assert not offenders, "\n".join(offenders)

    def test_no_store_port_is_held(self) -> None:
        offenders: list[str] = []
        for path in _files(UNDERSTANDING) + _files(ADAPTERS):
            source = path.read_text(encoding="utf-8")
            for port in ("ObjectStorePort", "EvidenceStorePort", "StateStorePort"):
                if port in source:
                    offenders.append(f"{_module_of(path)} names {port}")
        assert not offenders, (
            "persisting evidence is M13's job through P22:\n" + "\n".join(offenders)
        )

    def test_no_vision_state_vocabulary(self) -> None:
        forbidden = {"vision_state", "state_store", "projection", "materialize"}
        offenders: list[str] = []
        for path in _files(UNDERSTANDING) + _files(ADAPTERS):
            for identifier in _identifiers(path):
                if identifier.lower().lstrip("_") in forbidden:
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, "\n".join(offenders)


class TestNoModelCoupling:
    def test_no_model_name_appears_in_the_platform(self) -> None:
        """*"Never hard-code model names. Never couple M9 to any AI model."*"""
        vendors = (
            "qwen", "gemma", "internvl", "llava", "gpt4", "gpt-4", "gpt41",
            "claude", "gemini", "openai", "anthropic", "florence", "blip",
            "siglip", "clip", "minicpm", "phi3", "pixtral",
        )
        offenders: list[str] = []
        for path in _files(UNDERSTANDING) + [CORE / "ports" / "understanding.py"]:
            for identifier in _identifiers(path):
                lowered = identifier.lower()
                for vendor in vendors:
                    if vendor in lowered:
                        offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, (
            "the platform must not be able to reach a concrete model:\n"
            + "\n".join(offenders)
        )

    def test_no_vlm_ships_as_an_adapter(self) -> None:
        """Binding a real model needs weights, a runtime and a device — M18's
        concern and a deployment's choice."""
        for path in _files(ADAPTERS):
            for identifier in _identifiers(path):
                lowered = identifier.lower()
                for vendor in ("qwen", "llava", "gpt4", "claude", "gemini"):
                    assert vendor not in lowered, (
                        f"{_module_of(path)} ships a concrete VLM adapter"
                    )

    def test_the_engine_holds_ports_not_implementations(self) -> None:
        from vision_os.perception.understanding import UnderstandingEngine

        signature = inspect.signature(UnderstandingEngine.__init__)
        for name in ("router", "prompts", "coercion", "attributes"):
            assert name in signature.parameters, f"{name} must be injected"

    def test_only_the_composition_root_names_a_reference_adapter(self) -> None:
        named: list[str] = []
        for path in _files(ROOT):
            module = _module_of(path)
            if module.startswith("adapters/") or module == "understanding_bootstrap.py":
                continue
            source = path.read_text(encoding="utf-8")
            for concrete in (
                "ScriptedUnderstander",
                "StaticAttributeHead",
                "UnavailableUnderstander",
                "JsonCoercion",
                "StaticPromptProvider",
            ):
                if concrete in source:
                    named.append(f"{module} names {concrete}")
        assert not named, "\n".join(named)


class TestNoPromptAuthoring:
    def test_no_prompt_text_lives_in_the_engine(self) -> None:
        """*"M9 consumes prompts. M9 never creates prompts."*

        A prompt in code is a prompt with no version, no declared output schema
        and no load-time validation — bypassing gate 2 of the ceiling
        (00_CHARTER §4.3).
        """
        instruction_words = (
            "describe the",
            "you are a",
            "answer in json",
            "what is the person",
            "identify the",
            "analyze the image",
            "analyse the image",
        )
        offenders: list[str] = []
        for path in _files(UNDERSTANDING):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    lowered = node.value.lower()
                    for phrase in instruction_words:
                        if phrase in lowered:
                            offenders.append(
                                f"{_module_of(path)} contains prompt-like text: "
                                f"{node.value[:60]!r}"
                            )
        assert not offenders, "\n".join(offenders)

    def test_the_engine_resolves_prompts_through_the_seam(self) -> None:
        from vision_os.perception.understanding import engine as engine_module

        source = inspect.getsource(engine_module)
        assert "self._prompts.resolve" in source
        assert "self._prompts.render" in source

    def test_the_prompt_source_port_is_not_bindable(self) -> None:
        """P17 belongs to M10, which Flow 6 does not implement."""
        assert PortCatalogue.PROMPT_SOURCE not in BINDABLE_PORTS


class TestSemanticCeiling:
    def test_the_port_request_carries_no_subject(self) -> None:
        """An adapter is handed pixels and a question, never a subject.

        A model that could be told *who* it is looking at could accumulate
        something about them, which is what 12_SECURITY forbids structurally.
        """
        fields = set(UnderstandingPortRequest.__dataclass_fields__)
        for forbidden in ("object_id", "track_id", "tenant_id", "person_id"):
            assert forbidden not in fields

    def test_the_crop_view_carries_no_identity(self) -> None:
        fields = set(CropView.__dataclass_fields__)
        for forbidden in ("object_id", "track_id", "tenant_id", "identity"):
            assert forbidden not in fields

    def test_the_rendering_context_names_no_subject(self) -> None:
        """A prompt that could name the subject could be asked about the subject,
        and the ceiling would have a hole shaped like a template variable.

        Checked against the **keys the method actually returns**, not its source
        text: the docstring explains which fields are deliberately absent, and a
        substring scan would flag the explanation as the violation.
        """
        from vision_os.perception.understanding import engine as engine_module

        from .conftest import build_engine, make_request

        exposed = set(
            engine_module.UnderstandingEngine._context(None, make_request())
        )
        for forbidden in ("object_id", "tenant_id", "site_id", "region_label", "camera_id"):
            assert forbidden not in exposed, (
                f"the rendering context exposes '{forbidden}' to a prompt"
            )
        assert exposed == {
            "class_id",
            "requested_attributes",
            "prior_attributes",
            "quality",
        }, "the context is exactly what 04_MODULES section M10 renders with"
        assert build_engine is not None

    def test_the_config_section_has_no_business_slot(self) -> None:
        from vision_os.kernel.config.schema import UnderstandingSection

        fields = set(UnderstandingSection.__dataclass_fields__)
        for forbidden in (
            "prompts", "prompt_text", "attributes", "rules", "alert_on",
            "model_name", "vlm", "roles",
        ):
            assert forbidden not in fields

    def test_the_request_carries_no_business_context(self) -> None:
        fields = set(UnderstandingRequest.__dataclass_fields__)
        for forbidden in ("region_label", "business_context", "role", "purpose"):
            assert forbidden not in fields


class TestNoBiometrics:
    def test_no_face_or_identity_code_exists(self) -> None:
        """*"No biometric recognition. No face identification. No identity
        persistence."*"""
        forbidden = (
            "face_detect", "facial", "landmark", "iris", "gait", "fingerprint",
            "voiceprint", "recognise_person", "recognize_person", "gallery",
            "enrol", "enroll", "reid", "re_identify",
        )
        offenders: list[str] = []
        for path in _files(UNDERSTANDING) + _files(ADAPTERS):
            source = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                if token in source:
                    offenders.append(f"{path.name} contains '{token}'")
        assert not offenders, "\n".join(offenders)

    def test_the_biometric_ports_stay_unbindable(self) -> None:
        assert PortCatalogue.EMBEDDING not in BINDABLE_PORTS
        assert PortCatalogue.IDENTITY_RESOLVER not in BINDABLE_PORTS

    def test_no_embedding_is_produced_or_stored(self) -> None:
        offenders: list[str] = []
        for path in _files(UNDERSTANDING) + _files(ADAPTERS):
            for identifier in _identifiers(path):
                if identifier.lower().lstrip("_") in {"embedding", "embeddings", "vector_store"}:
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, "\n".join(offenders)


class TestLayering:
    def test_understanding_imports_no_adapter(self) -> None:
        offenders: list[str] = []
        for path in _files(UNDERSTANDING):
            for module in _imports(path):
                if "adapters" in module and not module.startswith("."):
                    offenders.append(f"{_module_of(path)} imports {module}")
        assert not offenders, "\n".join(offenders)

    def test_understanding_does_not_import_earlier_perception_modules(self) -> None:
        """M9 consumes crops from M8 and the registry's *attribute vocabulary*.

        It must not reach into tracking or detection: reaching past M8 would
        re-introduce decisions two layers below it.
        """
        offenders: list[str] = []
        for path in _files(UNDERSTANDING):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                if node.level and node.module.startswith(("tracking", "detection", "cropping")):
                    offenders.append(f"{_module_of(path)} imports {node.module}")
        assert not offenders, "\n".join(offenders)

    def test_earlier_flows_do_not_import_understanding(self) -> None:
        """The dependency runs one way. M8 holds a callable it never types."""
        offenders: list[str] = []
        for directory in (
            ROOT / "acquisition",
            KERNEL,
            PERCEPTION / "detection",
            PERCEPTION / "tracking",
            PERCEPTION / "registry",
            PERCEPTION / "cropping",
        ):
            for path in _files(directory):
                for module in _imports(path):
                    if "understanding" in module:
                        offenders.append(f"{_module_of(path)} imports {module}")
        assert not offenders, "\n".join(offenders)

    def test_core_remains_stdlib_only(self) -> None:
        for path in [
            CORE / "ports" / "understanding.py",
            CORE / "model" / "understanding.py",
        ]:
            for module in _imports(path):
                if module.startswith("."):
                    continue
                assert module.split(".")[0] in {
                    "__future__", "collections", "collections.abc", "dataclasses",
                    "enum", "typing", "hashlib", "json",
                }, f"{_module_of(path)} imports {module}"


class TestFlowScope:
    """Flow 6 shipped M9, so the frontier is now Flow 7."""

    def test_the_understanding_ports_are_bindable(self) -> None:
        assert PortCatalogue.UNDERSTANDER in BINDABLE_PORTS
        assert PortCatalogue.OUTPUT_COERCION in BINDABLE_PORTS

    def test_phase_two_ports_remain_unbindable(self) -> None:
        """``PROMPT_SOURCE`` is M10's, and M10 was never implemented.

        M9 consumes prompts through a module seam rather than a port it owns, so
        the Prompt Manager can arrive without disturbing understanding — which is
        why this port stayed unbound through eight flows without anything
        breaking.
        """
        for port in (
            PortCatalogue.EMBEDDING,
            PortCatalogue.IDENTITY_RESOLVER,
            PortCatalogue.PROMPT_SOURCE,
            PortCatalogue.CALIBRATION,
        ):
            assert port not in BINDABLE_PORTS, f"{port} became bindable early"

    def test_no_perception_level_exposure_exists(self) -> None:
        """L7 shipped, but never inside L2."""
        assert not (PERCEPTION / "synthesis").exists()
        assert not (PERCEPTION / "exposure").exists()

    def test_understanding_does_not_import_synthesis(self) -> None:
        """M9 gained a consumer and must not have noticed.

        §M9's boundary is *"produces understanding results; does not decide what
        they mean for the site"*. Importing the Observation Builder would let
        understanding reach the module that makes that decision, and the first
        convenience — asking the builder whether a result is worth publishing
        before spending a model call on it — would put suppression policy inside
        L4.
        """
        offenders = []
        for path in (PERCEPTION / "understanding").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for forbidden in ("synthesis", "vision_state", "state.manager"):
                if f"import {forbidden}" in source or f"{forbidden} import" in source:
                    offenders.append(f"{path.name} imports {forbidden}")
        assert not offenders, "\n".join(offenders)

    def test_no_temporal_adapter_is_bound(self) -> None:
        """Temporal understanding is Phase 3. The *contract* accepts a sequence;
        nothing shipped claims to handle one (15_ROADMAP §4)."""
        from vision_os.adapters.understanding import (
            ScriptedUnderstander,
            StaticAttributeHead,
        )

        from .conftest import POSTURE

        for adapter in (
            ScriptedUnderstander(producible=(POSTURE,)),
            StaticAttributeHead(attribute=POSTURE, value="standing"),
        ):
            assert not adapter.capabilities().supports_temporal, (
                f"{adapter.adapter_id} declares temporal support; Phase 3 is not "
                f"implemented and declaring it would route sequences to a model "
                f"that reads one frame"
            )

    def test_the_port_contract_admits_sequences(self) -> None:
        """The shape ships so Phase 3 needs *"no contract change"*."""
        annotation = str(UnderstandingPortRequest.__dataclass_fields__["crops"].type)
        assert "tuple" in annotation and "CropView" in annotation
