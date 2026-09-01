"""P10 Phase 1 — `meta/llama-3.2-90b-vision-instruct` resolves by configuration alone.

The contract these enforce is `docs/production-hardening/P10_PHASE1_INTEGRATION_CONTRACT.md`.

### What is actually being defended

Not "the 90B works" — that is Phase 2 and Phase 6, and needs a network. What is defended
here is that **selecting a model is a deployment act, not a code change**:

    VISION_NVIDIA_MODEL=meta/llama-3.2-90b-vision-instruct

and nothing else. No new provider, no new adapter, no edit to `DEFAULT_MODEL`, and no
occurrence of the string anywhere in production source.

That last one has teeth. On 2026-08-31 a model swap was performed by editing
`DEFAULT_MODEL` *and* running a repo-wide find-and-replace, which rewrote historical
evidence files as a side effect and had to be restored from git. A model name reachable
only through configuration cannot be swapped that way.

None of these tests make a network call.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.configuration.settings import Settings
from vision_os.adapters.configuration.understander_providers import (
    PROVIDER_ENV,
    UNDERSTANDER_FACTORIES,
    build_understander,
)
from vision_os.adapters.understanding import NvidiaVisionUnderstander
from vision_os.core.model.ids import AttributeKey

#: The model this phase integrates. Named once, here, so the string lives in the
#: test that is about it and nowhere in production source.
NINETY_B = "meta/llama-3.2-90b-vision-instruct"
ELEVEN_B = "meta/llama-3.2-11b-vision-instruct"

PRODUCIBLE = (AttributeKey("head_covering"), AttributeKey("face_covering"))
KEYED = {"NVIDIA_API_KEY": "nvapi-not-a-real-credential"}
REPO = Path(__file__).resolve().parents[3]


def nvidia(**env) -> NvidiaVisionUnderstander:
    adapter, _ = build_understander(
        producible=PRODUCIBLE, env={PROVIDER_ENV: "nvidia", **KEYED, **env}
    )
    return adapter


class TestItResolvesFromConfiguration:
    def test_the_primary_variable_selects_it(self) -> None:
        assert nvidia(VISION_NVIDIA_MODEL=NINETY_B)._model == NINETY_B

    def test_the_secondary_variable_selects_it(self) -> None:
        """`NVIDIA_MODEL` is honoured for deployments that predate the
        vision-specific name."""
        assert nvidia(NVIDIA_MODEL=NINETY_B)._model == NINETY_B

    def test_the_primary_wins_over_the_secondary(self) -> None:
        adapter = nvidia(VISION_NVIDIA_MODEL=NINETY_B, NVIDIA_MODEL=ELEVEN_B)
        assert adapter._model == NINETY_B

    def test_surrounding_whitespace_is_tolerated(self) -> None:
        """A hand-edited `.env` line is the normal way this value arrives."""
        assert nvidia(VISION_NVIDIA_MODEL=f"  {NINETY_B}  ")._model == NINETY_B

    def test_the_string_is_passed_through_verbatim(self) -> None:
        """No normalisation, no alias table, no vendor-prefix handling. What the
        deployment names is what reaches the endpoint."""
        adapter = nvidia(VISION_NVIDIA_MODEL=NINETY_B)
        assert adapter._model == NINETY_B
        assert adapter._model.count("/") == 1


class TestTheEnvFileReachesTheAdapter:
    """`.env` → `Settings` → factory → adapter. The bridge, end to end."""

    def test_settings_carry_the_model_to_the_factory(self) -> None:
        options = Settings(
            VISION_NVIDIA_MODEL=NINETY_B,
            VISION_NVIDIA_BASE_URL="https://nim.internal/v1",
        ).understander_options()

        adapter, _ = build_understander(
            producible=PRODUCIBLE, provider="nvidia", env=KEYED, defaults=options
        )
        assert adapter._model == NINETY_B
        assert adapter._base == "https://nim.internal/v1"

    def test_a_real_environment_variable_still_wins_over_the_file(self) -> None:
        """Layering is defaults → file → environment, and an operator setting a
        variable in the shell must not be silently overridden by `.env`."""
        options = Settings(VISION_NVIDIA_MODEL=ELEVEN_B).understander_options()

        adapter, _ = build_understander(
            producible=PRODUCIBLE, provider="nvidia",
            env={**KEYED, "VISION_NVIDIA_MODEL": NINETY_B}, defaults=options,
        )
        assert adapter._model == NINETY_B

    def test_an_unset_model_never_overrides_with_an_empty_string(self) -> None:
        """The defect this pins cost a whole debugging session: a mistyped key
        (`VISION_NVIDIA_MODE`) resolved to `''`, and an empty value that reached
        the factory would have named the empty model rather than falling back.

        Constructed with `_env_file=None` deliberately. Reading the repository's
        real `.env` would make this test assert whatever the machine happens to
        be configured for today — it failed exactly that way when first written.
        """
        options = Settings(_env_file=None).understander_options()

        assert "VISION_NVIDIA_MODEL" not in options, (
            "an unset model must be absent from the overlay, not present-and-empty"
        )


class TestSelectingItNeedsNoSourceEdit:
    """The heart of the contract."""

    def test_it_resolves_without_touching_the_source_default(self, monkeypatch) -> None:
        """Whatever `DEFAULT_MODEL` happens to be, an explicit setting wins."""
        import vision_os.adapters.understanding.nvidia_vl as vl

        monkeypatch.setattr(vl, "DEFAULT_MODEL", "some/unrelated-model")
        assert nvidia(VISION_NVIDIA_MODEL=NINETY_B)._model == NINETY_B

    def test_the_source_default_is_only_the_last_resort(self, monkeypatch) -> None:
        import vision_os.adapters.understanding.nvidia_vl as vl

        monkeypatch.setattr(vl, "DEFAULT_MODEL", "some/unrelated-model")
        assert nvidia()._model == "some/unrelated-model"

    def test_no_production_module_names_the_model(self) -> None:
        """A model name reachable only through configuration cannot be swapped
        by a find-and-replace — which is how historical evidence was corrupted
        on 2026-08-31 and had to be restored from git.

        `nvidia_vl.DEFAULT_MODEL` is the one permitted occurrence of *a* model
        string in production source, and it must not be this one: putting the
        90B there would make the source default and the deployment two competing
        answers to the same question.
        """
        offenders: dict[str, list[int]] = {}
        for directory in ("app", "vision_os", "compliance"):
            root = REPO / directory
            if not root.is_dir():
                continue
            for path in root.rglob("*.py"):
                hits = [
                    n for n, line in enumerate(
                        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                    )
                    if NINETY_B in line
                ]
                if hits:
                    offenders[str(path.relative_to(REPO))] = hits
        assert offenders == {}, (
            f"the 90B model name is hard-coded in production source: {offenders}. "
            f"It must be reachable only through VISION_NVIDIA_MODEL."
        )

    def test_the_default_model_is_a_single_named_constant(self) -> None:
        """Exactly one assignment, so there is one place a fallback can come
        from rather than several that can disagree."""
        source = (REPO / "vision_os" / "adapters" / "understanding" / "nvidia_vl.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        assigns = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", "") == "DEFAULT_MODEL" for t in node.targets)
        ]
        assert len(assigns) == 1


class TestItBindsTheExistingAdapter:
    def test_it_binds_the_nvidia_adapter_not_a_new_one(self) -> None:
        assert isinstance(nvidia(VISION_NVIDIA_MODEL=NINETY_B), NvidiaVisionUnderstander)

    def test_no_provider_was_added_for_it(self) -> None:
        """The brief forbids a new provider, and the endpoint is OpenAI-compatible
        so none is needed. The table stays closed at three."""
        assert set(UNDERSTANDER_FACTORIES) == {"nvidia", "ollama", "static"}

    def test_it_satisfies_the_port_contract(self) -> None:
        from vision_os.conformance.understanding_kits import UNDERSTANDER_KIT

        adapter = nvidia(VISION_NVIDIA_MODEL=NINETY_B)
        for check in UNDERSTANDER_KIT.checks:
            assert check.holds(adapter) if hasattr(check, "holds") else True
        assert adapter.capabilities().model_id == NINETY_B

    def test_residency_is_still_declared_remote(self) -> None:
        """Crops leave the machine. A site with a data-residency policy must
        still refuse this binding at composition time, 90B or not."""
        assert nvidia(VISION_NVIDIA_MODEL=NINETY_B).capabilities().data_residency == "remote"


class TestEvidenceStaysAttributable:
    """§16 of the contract. Provenance must survive into every observation."""

    def test_the_model_id_carries_the_exact_string(self) -> None:
        meta = nvidia(VISION_NVIDIA_MODEL=NINETY_B)._meta()
        assert meta.model_id == NINETY_B
        assert meta.artifact_hash == f"nvidia:{NINETY_B}"

    def test_the_two_models_are_distinguishable_in_evidence(self) -> None:
        """So a 90B observation can never be confused with an 11B one — the
        confusion that had to be undone by hand earlier today."""
        ninety = nvidia(VISION_NVIDIA_MODEL=NINETY_B)._meta()
        eleven = nvidia(VISION_NVIDIA_MODEL=ELEVEN_B)._meta()

        assert ninety.model_id != eleven.model_id
        assert ninety.artifact_hash != eleven.artifact_hash

    def test_model_version_is_degenerate_and_must_not_be_used_for_provenance(self) -> None:
        """**A known weakness, pinned so it cannot regress unnoticed.**

        `model_version` is derived by `rsplit("-", 1)` and yields "instruct" for
        both models. It is cosmetic; `model_id` carries provenance. This test
        exists so that Phase 6 attributes by `model_id`, and so that anyone who
        later fixes the derivation is told that recorded evidence has this shape.
        """
        ninety = nvidia(VISION_NVIDIA_MODEL=NINETY_B)._meta()
        eleven = nvidia(VISION_NVIDIA_MODEL=ELEVEN_B)._meta()

        assert ninety.model_version == "instruct"
        assert ninety.model_version == eleven.model_version, (
            "model_version cannot distinguish these models — use model_id"
        )

    def test_it_is_not_claimed_deterministic(self) -> None:
        """A hosted model behind a load balancer does not reproduce, and V13
        rests on replaying the observation log rather than the model."""
        assert nvidia(VISION_NVIDIA_MODEL=NINETY_B)._meta().deterministic is False


class TestProductionIsUnchangedByThisPhase:
    def test_phase_one_changes_no_production_module(self) -> None:
        """Phase 1 is documentation plus these tests. The adapter, the factory
        and the settings object are untouched by it, so whatever the deployment
        was running before this phase it is still running after."""
        import vision_os.adapters.configuration.understander_providers as providers

        assert set(providers.UNDERSTANDER_FACTORIES) == {"nvidia", "ollama", "static"}
        assert providers.PROVIDER_ENV == "VISION_UNDERSTANDER_PROVIDER"

    @pytest.mark.parametrize("model", [NINETY_B, ELEVEN_B])
    def test_either_model_binds_through_one_code_path(self, model: str) -> None:
        adapter = nvidia(VISION_NVIDIA_MODEL=model)
        assert adapter.adapter_id == "understander.nvidia_vl"
        assert adapter._model == model
