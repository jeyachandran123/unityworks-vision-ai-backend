"""Architecture and boundary guards for Flow 7.

The brief's fifteen prohibitions, made mechanical. Every one of these is
something a reasonable-looking future edit could introduce, and a comment saying
"don't" would not stop it.

The frontier guards here police **Flow 8**, because Flow 7 shipped. Guards for
boundaries already crossed belong to the flow that crossed them.
"""

from __future__ import annotations

import ast
import inspect
import io
import tokenize
from pathlib import Path

import pytest

import vision_os as package
from vision_os.core.model.observation import Observation, ObservationType
from vision_os.core.model.vision_state import ObjectState, RegionState, StateSnapshot
from vision_os.kernel.plugins.manifest import (
    BINDABLE_PORTS,
    FLOW7_PORTS,
    PortCatalogue,
)

ROOT = Path(package.__file__).parent
SYNTHESIS = ROOT / "synthesis"
STATE = ROOT / "state"


def sources(*roots: Path):
    """Every module under these roots, as ``(path, executable source)``.

    **Comments and string literals are stripped.** These modules explain
    themselves by quoting the architecture, so ``engine.py`` legitimately
    contains the sentence *"There is no severity, no alert, no threshold"* and
    ``suppression.py`` quotes §M7's *"a memory leak with a face"*. A guard that
    matched prose would fire on the documentation of the very rule it enforces —
    and the obvious fix, deleting the sentence, would make the code worse.

    So the guards read what the interpreter reads. A module that *does* the
    forbidden thing still has the identifier in its code.
    """
    for root in roots:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path, code_only(path.read_text(encoding="utf-8"))


def modules(*roots: Path):
    """Every module as ``(path, parsed tree)``.

    Separate from ``sources`` because an AST needs real source: the stripped
    text ``sources`` yields is deliberately not valid Python.
    """
    for root in roots:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            yield path, ast.parse(text, filename=str(path))


def code_only(text: str) -> str:
    """Source with comments, docstrings and string literals removed."""
    kept: list[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    except tokenize.TokenError:  # pragma: no cover - a syntax error fails elsewhere
        return text
    return "\n".join(kept)


class TestFlow7DoesOnlyItsOwnWork:
    """The brief's STRICT BOUNDARIES, one test per prohibition family."""

    def test_synthesis_performs_no_detection_or_tracking(self) -> None:
        forbidden = ("DetectorPort", "TrackerPort", "detect(", "def track(")
        offenders = [
            f"{path.name}: {term}"
            for path, text in sources(SYNTHESIS, STATE)
            for term in forbidden
            if term in text
        ]
        assert not offenders, "\n".join(offenders)

    def test_synthesis_performs_no_inference(self) -> None:
        """*"It never asks the AI another question."*

        M11 receives a validated result. A model call here would mean a fact
        could change between being decided and being recorded, and the evidence
        would explain a different answer from the one published.
        """
        forbidden = ("UnderstanderPort", "understand(", "infer(", "predict(", "model.run")
        offenders = [
            f"{path.name}: {term}"
            for path, text in sources(SYNTHESIS, STATE)
            for term in forbidden
            if term in text
        ]
        assert not offenders, "\n".join(offenders)

    def test_synthesis_generates_no_prompts(self) -> None:
        offenders = [
            f"{path.name}: {term}"
            for path, text in sources(SYNTHESIS, STATE)
            for term in ("PromptSourcePort", "render_prompt", "PromptTemplate")
            if term in text
        ]
        assert not offenders, "\n".join(offenders)

    def test_synthesis_holds_no_pixels(self) -> None:
        """V12. Pixels stay local; observations are control-plane sized.

        An observation carrying imagery would make every downstream hop a data
        transfer, and the evidence reference exists precisely so it does not.
        """
        offenders = [
            f"{path.name}: {term}"
            for path, text in sources(SYNTHESIS, STATE)
            for term in ("memoryview", "pixels", "np.ndarray", "frombuffer")
            if term in text
        ]
        assert not offenders, "\n".join(offenders)

    def test_synthesis_performs_no_business_reasoning(self) -> None:
        """V1. The vocabulary a rule would need does not exist here."""
        forbidden = (
            "alert", "incident", "violation_rule", "threshold_breach", "escalat",
            "compliance", "authorized", "unauthorized", "intrusion", "loiter",
            "risk_score", "severity",
        )
        offenders = []
        for path, text in sources(SYNTHESIS, STATE):
            lowered = text.lower()
            for term in forbidden:
                if term in lowered:
                    offenders.append(f"{path.name}: '{term}'")
        assert not offenders, (
            "business vocabulary reached the synthesis or state layer:\n"
            + "\n".join(offenders)
        )

    def test_synthesis_does_no_cross_camera_identity(self) -> None:
        """12_SECURITY §4.3 and 07_STATE §4: the camera is the partition.

        Correlating identities across cameras is Phase 2, classified C2, and
        policy-gated. A join here would enable it by accident.
        """
        offenders = [
            f"{path.name}: {term}"
            for path, text in sources(SYNTHESIS, STATE)
            for term in ("cross_camera", "reidentif", "re_identif", "global_identity")
            if term in text.lower()
        ]
        assert not offenders, "\n".join(offenders)

    def test_synthesis_computes_no_analytics(self) -> None:
        """07_STATE §6.1: *"History exists for perception, not for analytics."*"""
        offenders = [
            f"{path.name}: {term}"
            for path, text in sources(SYNTHESIS, STATE)
            for term in ("dwell_time", "average_", "percentile", "aggregate_over", "report_for")
            if term in text.lower()
        ]
        assert not offenders, "\n".join(offenders)

    def test_no_biometric_persistence(self) -> None:
        """The brief: *"No biometric persistence. No facial recognition."*"""
        offenders = [
            f"{path.name}: {term}"
            for path, text in sources(SYNTHESIS, STATE)
            for term in ("embedding", "face", "biometric", "gait", "iris")
            if term in text.lower()
        ]
        assert not offenders, "\n".join(offenders)


class TestLayeredDependencyLaw:
    """01_LAYERED: dependencies run one way only."""

    def test_synthesis_does_not_import_a_lower_layer_runtime(self) -> None:
        """L5 may hold L2's *types*, never its machinery.

        Importing a detection or tracking runtime would let synthesis drive
        perception, and the two would no longer be separately deployable.
        """
        forbidden = (
            "perception.detection", "perception.tracking", "perception.cropping",
            "perception.understanding", "acquisition",
        )
        offenders = []
        for path, text in sources(SYNTHESIS, STATE):
            for module in forbidden:
                if f"from ...{module}" in text or f"import {module}" in text:
                    offenders.append(f"{path.name} imports {module}")
        assert not offenders, "\n".join(offenders)

    def test_state_does_not_import_the_synthesis_package(self) -> None:
        """M12 consumes observations, not the module that builds them.

        The dependency runs L5 → L6. Reversing it would make the projection
        unable to replay a log written by a different builder version, which is
        exactly what a rebuild after an upgrade must do.

        ``core.ports.synthesis`` is a different thing and is allowed: a port is a
        contract owned by neither side, which is the whole reason it is a port.
        The guard names the package, not the word.
        """
        offenders = []
        for path, tree in modules(STATE):
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    head = node.module.split(".")[0]
                    if head == "synthesis" or node.module.startswith("vision_os.synthesis"):
                        offenders.append(f"{path.name}: {node.module}")
                    if node.level >= 2 and node.module.startswith("synthesis"):
                        offenders.append(f"{path.name}: relative {node.module}")
        assert not offenders, "\n".join(offenders)

    def test_the_builder_does_not_import_the_state_manager(self) -> None:
        """M11 publishes; it does not know where facts land.

        A builder holding the manager could read state to decide what to
        publish, and suppression would silently become a state query.
        """
        text = (SYNTHESIS / "builder" / "engine.py").read_text(encoding="utf-8")
        assert "VisionStateManager" not in text
        assert "state.manager" not in text

    def test_lower_layers_do_not_import_synthesis(self) -> None:
        """Flow 7 attached to four earlier flows and none of them noticed."""
        offenders = []
        for layer in ("perception", "acquisition"):
            for path, text in sources(ROOT / layer):
                for term in ("synthesis", "vision_state", "VisionStateManager"):
                    if term in text:
                        offenders.append(f"{layer}/{path.name}: {term}")
        assert not offenders, "\n".join(offenders)


class TestPortsNotImplementations:
    """V3. The builder holds protocols; the composition root picks classes."""

    def test_the_builder_names_no_concrete_adapter(self) -> None:
        text = (SYNTHESIS / "builder" / "engine.py").read_text(encoding="utf-8")
        for concrete in ("ExactSuppression", "ThresholdSuppression", "AlwaysPublish"):
            assert concrete not in text, f"the builder names {concrete}"

    def test_the_state_manager_names_no_concrete_log(self) -> None:
        text = (STATE / "manager.py").read_text(encoding="utf-8")
        for concrete in ("InMemoryObservationLog", "FileObservationLog"):
            assert concrete not in text

    def test_only_the_composition_root_selects_adapters(self) -> None:
        """One place decides. Two would drift, and the second would be found by
        an operator wondering why configuration had no effect.
        """
        offenders = [
            path.name
            for path, text in sources(SYNTHESIS, STATE)
            if "SUPPRESSION_FACTORIES" in text
        ]
        assert not offenders, "\n".join(offenders)

    def test_the_flow_seven_ports_are_bindable(self) -> None:
        assert FLOW7_PORTS <= BINDABLE_PORTS
        assert len(FLOW7_PORTS) == 3


class TestFlowEightRemainsUnbuilt:
    """The frontier moved to Flow 8; these guard *that* boundary."""

    def test_phase_two_ports_remain_unbindable(self) -> None:
        """Phase 1 is complete; the frontier has stopped moving.

        Four ports stay unbound, and none of them is waiting for a flow:
        ``EMBEDDING`` and ``IDENTITY_RESOLVER`` are Phase 2 and policy-gated,
        ``PROMPT_SOURCE`` belongs to M10, ``CALIBRATION`` to M1 and M18.
        """
        for port in (
            PortCatalogue.EMBEDDING,
            PortCatalogue.IDENTITY_RESOLVER,
            PortCatalogue.PROMPT_SOURCE,
            PortCatalogue.CALIBRATION,
        ):
            assert port not in BINDABLE_PORTS, f"{port} became bindable early"

    def test_the_biometric_ports_stay_unbindable(self) -> None:
        """A standing guard, and now a permanent one (12_SECURITY §4.3).

        Every flow has shipped. If these were ever going to become bindable in
        Phase 1, it would have happened by now — so this stops being a frontier
        guard and becomes the boundary itself.
        """
        assert PortCatalogue.EMBEDDING not in BINDABLE_PORTS
        assert PortCatalogue.IDENTITY_RESOLVER not in BINDABLE_PORTS

    def test_synthesis_does_not_import_exposure(self) -> None:
        """L5 must not learn L7 exists.

        M11 publishes facts. Whether anyone is subscribed is not its concern, and
        an import here would let suppression start depending on who is listening.
        """
        offenders = [
            f"{path.name}: exposure"
            for path, text in sources(SYNTHESIS, STATE)
            if "exposure" in text
        ]
        assert not offenders, "\n".join(offenders)

    def test_state_exposes_no_transport(self) -> None:
        """M14 serves state over a wire. M12 returns values.

        Putting a transport here would make the projection's read path depend on
        a network stack, and 07_STATE §5.1's O(1) snapshot claim would stop being
        about memory.
        """
        offenders = [
            f"{path.name}: {term}"
            for path, text in sources(STATE)
            for term in ("fastapi", "aiohttp", "FastAPI", "@app.get", "Router")
            if term in text
        ]
        assert not offenders, "\n".join(offenders)


class TestImmutabilityIsStructural:
    """V5, enforced by the type rather than by convention."""

    @pytest.mark.parametrize(
        "kind", [Observation, ObjectState, RegionState, StateSnapshot]
    )
    def test_every_record_type_is_frozen_and_slotted(self, kind) -> None:
        assert kind.__dataclass_params__.frozen, f"{kind.__name__} is mutable"
        assert hasattr(kind, "__slots__"), f"{kind.__name__} can grow a field"

    def test_no_module_mutates_an_observation_in_place(self) -> None:
        """``replace`` produces a new value; assignment would edit history."""
        offenders = []
        for path, tree in modules(SYNTHESIS, STATE):
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in ("observation", "candidate", "published")
                    ):
                        offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, "\n".join(offenders)


class TestNoGlobalMutableState:
    """The brief: *"no global mutable state"*. V13 depends on it."""

    def test_no_module_level_mutable_container(self) -> None:
        """A module-level dict shared across cameras would couple partitions
        that the architecture requires to be independent, and two runs of the
        same log would then differ by what ran before them.
        """
        offenders = []
        for path, tree in modules(SYNTHESIS, STATE):
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
                    # SHOUTING names are declared constants. ``__all__`` is a
                    # list by language requirement and is not state — it has
                    # no cased characters, so ``isupper()`` alone rejects it.
                    if not name.isupper() and not name.startswith("__")
                )
        assert not offenders, (
            "module-level mutable state:\n" + "\n".join(offenders)
        )

    def test_the_builder_holds_its_state_per_instance(self) -> None:
        from vision_os.synthesis import ObservationBuilder

        assert "__slots__" in vars(ObservationBuilder)
        assert "_suppression" in ObservationBuilder.__slots__


class TestTheGateIsTheOnlyWayIn:
    def test_every_builder_method_routes_through_finish_or_the_gate(self) -> None:
        """One entry point, so a new observation type cannot bypass the ceiling.

        A ``build_x`` that assembled and returned without validating would be a
        hole in the platform's last constitutional check, and it would look
        entirely reasonable in review.
        """
        from vision_os.synthesis import ObservationBuilder

        builders = [
            name
            for name in dir(ObservationBuilder)
            if name.startswith("build_")
        ]
        assert len(builders) == 7, f"expected the seven §M11 builders, got {builders}"

        for name in builders:
            source = inspect.getsource(getattr(ObservationBuilder, name))
            assert "_finish" in source or "_count" in source, (
                f"{name} neither validates nor counts; a build path that skips "
                f"the gate is a hole in the semantic ceiling"
            )

    def test_there_is_exactly_one_observation_type_enum(self) -> None:
        """Two would drift, and the second would not be gated."""
        assert len(tuple(ObservationType)) == 7
