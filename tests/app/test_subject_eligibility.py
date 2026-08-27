"""Layer A — the per-attribute quality floor the policy measured and nothing used.

`kitchen-safety` has declared, for as long as it has existed:

    head_covering  min_scale_pixels: 130.0   max_blur: 0.5
    hand_covering  min_scale_pixels: 150.0   max_blur: 0.85

with a comment recording that they were *"Calibrated against
datasets/kitchen-01 (15 frames, 43 annotated subjects, human visual
inspection). These are measured floors, not guesses"*.

`QualityGate` has always accepted them, and its own docstring describes the
failure they prevent: *"a whole-person crop 60px tall is a fine subject for
'what colour is the garment'; the head band inside it is 27px and cannot answer
'is the head covered'."*

Nothing carried the floors from the document to the gate. Every crop was judged
against the deployment default of 48px, and the difference is silent — a
too-small head produces a confident answer rather than a rejection.

The measured consequences of wiring them, over the two real populations:

* **0 of 43** human-confirmed people fall below the head floor, so the guard
  cannot damage true detection (S13).
* **12 of 284** live subjects do, so those stop costing a model call (S16).

What it does **not** do is caught by the last test here, deliberately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.vision.composition import load_policies
from app.vision.understanding import _quality_floors

CONFIG = Path(__file__).resolve().parents[2] / "config"
POLICIES = load_policies(
    f"{CONFIG / 'policies' / 'kitchen-safety.example.json'},"
    f"{CONFIG / 'policies' / 'object-identity.example.json'}"
)


class TestTheFloorsReachTheGate:
    def test_the_composition_root_collects_the_declared_floors(self) -> None:
        """The wiring itself. Absent this, `build_cropping_layer` receives
        `None` and every attribute is judged against one global default."""
        floors = _quality_floors(POLICIES)
        assert floors is not None
        assert floors["head_covering"]["min_scale_pixels"] == 130.0
        assert floors["head_covering"]["max_blur"] == 0.5

    def test_the_floors_are_attribute_specific(self) -> None:
        """S4: head visibility and hand visibility are different questions, and
        one global rule has to be wrong for one of them."""
        floors = _quality_floors(POLICIES)
        assert floors["hand_covering"]["min_scale_pixels"] == 150.0
        assert (
            floors["hand_covering"]["min_scale_pixels"]
            != floors["head_covering"]["min_scale_pixels"]
        )

    def test_nothing_restates_the_geometry_in_application_code(self) -> None:
        """S4: do not duplicate crop geometry in application code. The helper
        reads the document and names no threshold of its own.

        Parsed with `ast` and checked against the *executable* body only. An
        earlier version compared raw text and failed on the docstring above,
        which names the numbers precisely so nobody has to hunt for them —
        documentation naming a value is the opposite of code hard-coding it.
        """
        import ast

        source = (
            Path(__file__).resolve().parents[2] / "app" / "vision" / "understanding.py"
        ).read_text(encoding="utf-8")
        helper = next(
            node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "_quality_floors"
        )
        body = list(helper.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]  # drop the docstring
        code = "\n".join(ast.unparse(node) for node in body)

        assert "130" not in code
        assert "min_scale_pixels" not in code
        assert "max_blur" not in code
        assert "quality_floors" in code, "it must read what the document declares"

    def test_the_gate_applies_the_strictest_floor_for_a_shared_crop(self) -> None:
        """One crop answers head, face and hands at once. A crop good enough for
        the laxest question but not the strictest would otherwise answer both."""
        from vision_os.perception.cropping.gate import GateThresholds, QualityGate

        gate = QualityGate(
            GateThresholds(min_scale_pixels=48.0),
            per_attribute={
                "head_covering": GateThresholds(min_scale_pixels=130.0),
                "hand_covering": GateThresholds(min_scale_pixels=150.0),
            },
        )
        chosen = gate.thresholds_for(("head_covering", "hand_covering"))
        assert chosen.min_scale_pixels == 150.0


class TestItCannotDamageTrueDetection:
    """S13: this is not a make-alerts-disappear task."""

    def test_no_human_confirmed_person_falls_below_the_head_floor(self) -> None:
        """Measured over all 43 annotated subjects. `scale_pixels` is object
        height in source pixels, and the smallest real person in kitchen-01 is
        142.9px against a floor of 130."""
        dataset = Path(
            "c:/Users/Jayachandran/ProjectsAndDocs/atlas/backend/datasets/kitchen-01"
        )
        if not dataset.is_dir():
            pytest.skip("kitchen-01 dataset is not present in this checkout")

        ann = json.loads(
            (dataset / "annotations" / "kitchen-01.json").read_text(encoding="utf-8")
        )
        floor = _quality_floors(POLICIES)["head_covering"]["min_scale_pixels"]
        heights = [
            (s["box"]["y2"] - s["box"]["y1"]) * 576
            for f in ann["frames"] for s in f["subjects"]
        ]
        assert len(heights) == 43
        assert min(heights) >= floor, (
            f"the declared floor {floor} would reject a human-confirmed person "
            f"({min(heights):.1f}px) — that is a regression, not a guard"
        )


class TestWhatTheGuardDoesNotDo:
    def test_the_floor_does_not_catch_the_proven_false_subjects(self) -> None:
        """**Recorded so the barrier is not mistaken for a fix.**

        Five live subjects were proven false by opening their stored decision
        frames — two showed an empty kitchen. Every one of them is well above
        the head floor, because they are *large* boxes containing no person.

        No geometric measure separated them: area, width, height, aspect, edge
        distance and head-band pixels were all measured across both populations
        and all overlap. Layer A is defence in depth against unusable crops, not
        a detector-precision fix, and pretending otherwise is what would let the
        real failure keep shipping.
        """
        floor = _quality_floors(POLICIES)["head_covering"]["min_scale_pixels"]
        proven_false_heights_px = {
            "502c12883c": 232.5, "74002b9470": 209.0, "26994b451a": 148.6,
            "716dde9a21": 231.9, "a5d81072ba": 278.5,
        }
        admitted = [k for k, px in proven_false_heights_px.items() if px >= floor]
        assert admitted == list(proven_false_heights_px), (
            "if this now fails, the floor changed and the false-subject "
            "measurement must be re-run before it is trusted"
        )
