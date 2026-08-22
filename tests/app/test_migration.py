"""Migration invariants — the properties that make this a migration rather than a copy.

Every test here would have passed trivially before the move and would fail if the
move regressed. They are cheap, and each one guards a specific way the migration
could be quietly undone later.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("app", "vision_os", "compliance", "tools")


def python_files(*roots: str) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        base = REPO / root
        if not base.is_dir():
            continue
        found.extend(
            p for p in base.rglob("*.py") if "__pycache__" not in p.parts and ".venv" not in p.parts
        )
    return sorted(found)


class TestPackageBoundary:
    def test_vision_os_imports_from_its_own_name(self) -> None:
        """The platform resolves as `vision_os`, not `app.vision_os`."""
        import vision_os

        assert Path(vision_os.__file__).resolve().parent == REPO / "vision_os"

    def test_no_module_references_the_old_package_name(self) -> None:
        """`app.vision_os` and `app.compliance` are gone from every source file.

        1,102 occurrences of `app.vision_os` and 9 of `app.compliance` were
        rewritten. A single survivor would import from a repository that this one
        must not depend on.
        """
        offenders = [
            str(path.relative_to(REPO))
            for path in python_files(*SOURCE_DIRS)
            if "app.vision_os" in path.read_text(encoding="utf-8")
            or "app.compliance" in path.read_text(encoding="utf-8")
        ]
        assert offenders == []

    def test_there_is_exactly_one_vision_os_package(self) -> None:
        """No `app/vision_os/`, no second copy, no compatibility shim.

        Two canonical implementations is the failure this repository exists to
        end: the platform lived inside an unrelated application and was reached
        by path injection. Recreating it here in any form would restore that.
        """
        assert not (REPO / "app" / "vision_os").exists()

        # `tests/vision_os` is the platform's own suite and shares the name
        # deliberately — see `tests/__init__.py` for why that is safe and what
        # makes it so. A *package* copy is what this test looks for.
        roots = [
            p
            for p in REPO.glob("*/vision_os")
            if ".venv" not in p.parts and p.parent.name != "tests"
        ]
        assert roots == [], f"a second vision_os package exists: {roots}"


class TestNoSiblingRepositoryDependency:
    def test_no_source_file_mutates_sys_path(self) -> None:
        """`sys.path` manipulation is how the validation harness reached the
        platform. It must not exist here.

        Parsed rather than grepped, so that the phrase appearing in a docstring —
        as it does, in this file and in the migration report — is not a false
        positive. Only a real call counts.
        """
        offenders: list[str] = []
        for path in python_files(*SOURCE_DIRS):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover - would fail elsewhere first
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                target = node.func
                if not isinstance(target, ast.Attribute):
                    continue
                if target.attr not in {"insert", "append", "extend"}:
                    continue
                value = target.value
                if (
                    isinstance(value, ast.Attribute)
                    and value.attr == "path"
                    and isinstance(value.value, ast.Name)
                    and value.value.id == "sys"
                ):
                    offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
        assert offenders == [], f"sys.path is mutated at: {offenders}"

    def test_no_source_file_names_a_sibling_repository(self) -> None:
        """Nothing points at atlas/backend, the console, the demo or the old frontend."""
        forbidden = (
            "atlas/backend",
            "atlas\\backend",
            "vision_os_validation_console",
            "vision_os_demo",
            "vosvc_harness",
        )
        offenders: list[str] = []
        for path in python_files(*SOURCE_DIRS):
            text = path.read_text(encoding="utf-8")
            for needle in forbidden:
                if needle in text:
                    offenders.append(f"{path.relative_to(REPO)} -> {needle}")
        assert offenders == []

    def test_the_platform_imports_in_a_subprocess_with_no_extra_path(self) -> None:
        """`import vision_os` works with `sys.path` untouched and cwd elsewhere.

        Run out-of-tree so that the repository root being the working directory
        cannot be what makes it succeed. This is the property that makes the
        install reproducible on a machine where no other Atlas repository exists.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import vision_os, compliance; "
                "from vision_os.exposure.api import ObservationApi; "
                "print(vision_os.__file__)",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO.parent),
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert str(REPO / "vision_os") in result.stdout


class TestVisionOsIsUnchanged:
    def test_the_platform_uses_relative_imports_only(self) -> None:
        """No absolute self-reference inside the platform.

        This is *why* the rename needed no edit inside `vision_os/`: 201 files,
        every internal import relative. The property is worth keeping — it is
        what makes the package movable at all.
        """
        offenders: list[str] = []
        for path in python_files("vision_os"):
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(("from vision_os", "import vision_os")):
                    offenders.append(f"{path.relative_to(REPO)}: {stripped}")
        assert offenders == []

    def test_core_imports_no_third_party_library(self) -> None:
        """`vision_os.core` is stdlib-only, by contract.

        `core/model/space.py` states it: *"Geometry here is pure Python. core may
        not import numpy or OpenCV."* It is what lets a deployment install the
        platform without a CV stack.
        """
        banned = {"numpy", "onnxruntime", "torch", "cv2", "av", "PIL", "httpx", "requests"}
        offenders: list[str] = []
        for path in python_files("vision_os"):
            if "core" not in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module.split(".")[0]]
                else:
                    continue
                for name in names:
                    if name in banned:
                        offenders.append(f"{path.relative_to(REPO)}: {name}")
        assert offenders == []


class TestApplicationDependencyDirection:
    def test_the_platform_never_imports_the_application(self) -> None:
        """`app` imports `vision_os`; `vision_os` never imports `app`.

        The moment it reverses, the platform has acquired a business opinion.
        """
        offenders: list[str] = []
        for path in python_files("vision_os", "compliance"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("app"):
                    offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "app" or alias.name.startswith("app."):
                            offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
        assert offenders == []

    def test_compliance_depends_on_the_platform_and_not_the_reverse(self) -> None:
        import compliance
        import vision_os

        assert Path(compliance.__file__).parent == REPO / "compliance"
        assert Path(vision_os.__file__).parent == REPO / "vision_os"


class TestConfigurationDocuments:
    """The domain arrives as data. These files are the domain."""

    @pytest.mark.parametrize(
        "relative",
        [
            "config/policies/kitchen-safety.example.json",
            "config/policies/object-identity.example.json",
            "config/policies/verification.example.json",
            "config/rules/site-safety.example.json",
        ],
    )
    def test_document_migrated(self, relative: str) -> None:
        assert (REPO / relative).is_file()

    def test_the_production_detector_weights_came_along(self) -> None:
        weights = REPO / "models" / "yolov8n.onnx"
        assert weights.is_file()
        assert weights.stat().st_size > 1_000_000
