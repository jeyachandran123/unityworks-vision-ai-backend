"""The compliance layer's boundaries, enforced mechanically.

An invariant with no test is a slogan. These read the source tree and fail the
build when a boundary is crossed, which is the only way the two most important
promises in this package stay true:

* a rule **never** calls a model, and
* the platform **never** learns that compliance exists.

Both are stated in prose in a dozen places. Only these tests make them binding.

Deliberately crude, in the same spirit as the platform's own
``test_no_domain_vocabulary_in_platform_code``: they catch the *first* leak,
which is the one that establishes precedent.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import compliance as compliance_pkg
import vision_os as vision_os_pkg

COMPLIANCE = Path(compliance_pkg.__file__).parent
VISION_OS = Path(vision_os_pkg.__file__).parent


def _python_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Identity of every docstring constant, so prose can be excluded from a scan.

    Attribute docstrings — a bare string following an assignment, which this
    codebase uses heavily — count too: they are documentation by convention even
    though ``ast.get_docstring`` does not see them.
    """
    found: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        previous_was_assignment = False
        for index, statement in enumerate(body):
            is_string_expr = (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
            if is_string_expr and (index == 0 or previous_was_assignment):
                found.add(id(statement.value))
            previous_was_assignment = isinstance(statement, ast.AnnAssign | ast.Assign)
    return found


def _imports(path: Path) -> list[str]:
    """Absolute and relative module names imported by one file."""
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


#: What the evaluator is allowed to reach for. Everything here is either a value
#: type or the platform's one published read surface.
ALLOWED_VISION_OS = (
    "vision_os.core.model",
    "vision_os.exposure.api",
)

#: Anything that could make an evaluation non-deterministic, expensive, or
#: dependent on something other than the facts it was handed.
FORBIDDEN_MARKERS = (
    "torch", "tensorflow", "onnx", "onnxruntime", "ultralytics", "cv2", "numpy",
    "transformers", "openai", "ollama", "anthropic", "langchain",
    "requests", "httpx", "aiohttp", "urllib", "socket", "http",
    "random", "secrets",
)

#: Platform internals a rule engine must never touch. Reaching into any of these
#: would mean the compliance layer had acquired a perception responsibility.
FORBIDDEN_VISION_OS = (
    "vision_os.adapters",
    "vision_os.perception",
    "vision_os.acquisition",
    "vision_os.synthesis",
    "vision_os.kernel",
)

STDLIB_ALLOWED = {
    "__future__", "abc", "collections", "collections.abc", "dataclasses",
    "datetime", "enum", "functools", "itertools", "json", "math", "os",
    "pathlib", "re", "typing",
}


class TestTheRuleEngineCannotPerceive:
    """Tests 18 and 19: a rule never calls a model and never creates a crop.

    Proved by dependency closure rather than by inspection. There is no code
    path from an evaluation to an inference because there is no collaborator
    that could perform one.
    """

    def test_no_module_imports_an_inference_or_network_dependency(self) -> None:
        offenders: list[str] = []
        for path in _python_files(COMPLIANCE):
            for imported in _imports(path):
                root = imported.split(".")[0]
                if root in FORBIDDEN_MARKERS:
                    offenders.append(f"{path.name} imports '{imported}'")

        assert not offenders, (
            "the compliance layer must be a deterministic function of rules and "
            "facts; these imports make it something else:\n" + "\n".join(offenders)
        )

    def test_no_module_reaches_into_platform_internals(self) -> None:
        offenders: list[str] = []
        for path in _python_files(COMPLIANCE):
            for imported in _imports(path):
                for forbidden in FORBIDDEN_VISION_OS:
                    if imported.startswith(forbidden):
                        offenders.append(f"{path.name} imports '{imported}'")

        assert not offenders, (
            "a rule engine reaching into perception has taken responsibility for "
            "a layer that is not its own:\n" + "\n".join(offenders)
        )

    def test_platform_imports_are_confined_to_value_types_and_the_api(self) -> None:
        offenders: list[str] = []
        for path in _python_files(COMPLIANCE):
            for imported in _imports(path):
                if not imported.startswith("vision_os"):
                    continue
                if not any(imported.startswith(ok) for ok in ALLOWED_VISION_OS):
                    offenders.append(f"{path.name} imports '{imported}'")

        assert not offenders, (
            f"the compliance layer may reach only {list(ALLOWED_VISION_OS)}; "
            f"found:\n" + "\n".join(offenders)
        )

    def test_only_the_reader_touches_the_api(self) -> None:
        """One read path, so a scope narrowing cannot be forgotten in a second.

        12_SECURITY section 4.2 designs the leak out by constructing every query
        already scoped. A rule engine with two read paths has two places to get
        that right and one place to get it wrong.
        """
        offenders: list[str] = []
        for path in _python_files(COMPLIANCE):
            if path.name in ("reader.py", "__init__.py"):
                continue
            for imported in _imports(path):
                if "exposure" in imported:
                    offenders.append(f"{path.name} imports '{imported}'")

        assert not offenders, (
            "only reader.py may hold the Observation API:\n" + "\n".join(offenders)
        )

    def test_the_evaluator_holds_no_clock(self) -> None:
        """``now`` is a parameter. A clock read inside would make the same rule
        against the same facts produce two different findings."""
        source = (COMPLIANCE / "evaluator.py").read_text(encoding="utf-8")

        assert "import time" not in source
        assert "datetime.now" not in source
        assert "Clock" not in source

    def test_the_evaluator_imports_nothing_that_creates_a_crop(self) -> None:
        source = (COMPLIANCE / "evaluator.py").read_text(encoding="utf-8")

        for forbidden in ("Crop", "CropRequest", "UnderstandingRequest", "Understander"):
            assert forbidden not in source, (
                f"'{forbidden}' appears in the evaluator; a rule evaluates "
                f"structured facts and never asks for pixels"
            )


class TestTheDependencyRunsOneWay:
    """``compliance -> vision_os``, never the reverse.

    The moment it reverses, the platform has acquired a business opinion — and
    every guarantee that rests on its neutrality becomes a matter of trust
    rather than of structure.
    """

    def test_no_platform_module_imports_the_compliance_layer(self) -> None:
        offenders: list[str] = []
        for path in _python_files(VISION_OS):
            for imported in _imports(path):
                if "compliance" in imported.split("."):
                    offenders.append(
                        f"{path.relative_to(VISION_OS)} imports '{imported}'"
                    )

        assert not offenders, (
            "Vision OS must not know its consumers exist:\n" + "\n".join(offenders)
        )

    def test_the_platform_still_rejects_a_compliance_attribute(self) -> None:
        """The neutrality gate is the reason this package exists at all.

        If this ever passes, the argument for a separate package has evaporated
        and someone should be told rather than left to discover it.
        """
        from vision_os.core.errors import AttributeRejectedError
        from vision_os.perception.registry.attributes import check_neutrality

        for key in ("is_compliant", "ppe_violation", "safety_alert"):
            with pytest.raises(AttributeRejectedError):
                check_neutrality(key, "the crop shows what is being claimed here")


class TestNoDomainVocabularyInCode:
    """Rules are data here too, exactly as the platform's semantic policy is.

    The words this package is *about* — the attributes, the values, the sentences
    a reviewer reads — live in a JSON document. Adding a use case is a file, not
    a release, and this is what proves it.
    """

    DOMAIN_VOCABULARY = (
        "waiter", "chef", "cashier", "patient", "nurse", "doctor", "customer",
        "employee", "shopper", "clerk",
        "restaurant", "kitchen", "hospital", "warehouse", "retail", "clinic",
        "hairnet", "glove", "apron", "helmet", "biryani", "menu", "checkout",
        "vegetarian", "shelf", "till",
    )

    def test_no_domain_word_appears_as_an_identifier_or_a_literal(self) -> None:
        """Identifiers and behavioural string literals, but not docstrings.

        Docstrings are excluded because prose *explaining* the prohibition has to
        be able to name what it is prohibiting — the platform's own version of
        this test draws the line the same way. String **literals** are included
        where the platform's is not, because a hard-coded value in a comparison
        is a use case in code no less than a variable name is.
        """
        import re

        token = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+")
        offenders: list[str] = []

        for path in _python_files(COMPLIANCE):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            docstrings = _docstring_nodes(tree)

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
                elif (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings
                ):
                    name = node.value
                if not name:
                    continue
                leaked = {t.lower() for t in token.findall(name)} & set(
                    self.DOMAIN_VOCABULARY
                )
                if leaked:
                    offenders.append(
                        f"{path.name}::{name[:60]!r} uses {sorted(leaked)}"
                    )

        assert not offenders, (
            "a use case has leaked into the rule engine; it belongs in the rule "
            "document:\n" + "\n".join(offenders)
        )

    def test_a_use_case_nobody_has_heard_of_works_end_to_end(self) -> None:
        """The test that actually matters: invent a vocabulary and drive it.

        Nothing in this repository has heard of any word below, and the engine
        reaches a correct verdict anyway — which is what "configuration-driven"
        has to mean to be worth claiming.
        """
        from compliance import ComplianceEvaluator, ComplianceState, RuleSet

        from .conftest import NOW, attribute, subject

        rules = RuleSet.from_document(
            {
                "version": "1",
                "rules": [
                    {
                        "rule_id": "invented.use.case.v1",
                        "version": "3.2.1",
                        "subject_classes": ["quirion"],
                        "require": [
                            {
                                "attribute": "flange_alignment",
                                "operator": "eq",
                                "value": "nominal",
                                "message": "has a misaligned flange",
                            }
                        ],
                    }
                ],
            }
        )
        view = subject(
            class_id="quirion",
            attributes={"flange_alignment": attribute("flange_alignment", "skewed")},
        )
        finding = ComplianceEvaluator(rules).evaluate_object(view, now=NOW)[0]

        assert finding.state is ComplianceState.VIOLATION
        assert finding.describe().endswith("has a misaligned flange")
