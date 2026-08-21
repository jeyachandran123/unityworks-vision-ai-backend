"""Coercion strategies, the prompt provider, and the reference understanders.

The coercion tests are where 02_VOM §9.3 lives:

> *This text is **inspectable but never promoted**. It is not queryable as fact,
> never enters Vision State as an attribute, and cannot be filtered on by the
> API.*

Every test below that asserts something lands in ``unparsed`` is asserting that
the platform kept what the model said without believing it.
"""

from __future__ import annotations

import json

import pytest

from vision_os.adapters.understanding import (
    JsonCoercion,
    KeyValueCoercion,
    PassthroughCoercion,
    PromptTemplate,
    ScriptedAnswer,
    StaticPromptProvider,
)
from vision_os.core.errors import PromptUnavailableError
from vision_os.core.model.ids import AttributeKey, ClassId, PromptId
from vision_os.core.ports.understanding import OutputSchema

from ..conftest import (
    HEADWEAR,
    HEIGHT,
    PERSON,
    POSTURE,
    UNREGISTERED,
    VEHICLE,
    build_registry,
    scripted,
)

SCHEMA = OutputSchema(fields=(POSTURE, HEADWEAR))


class TestJsonCoercion:
    def test_clean_json_parses(self) -> None:
        result = JsonCoercion().coerce('{"posture": "standing"}', schema=SCHEMA)
        assert result.parsed == {"posture": "standing"}
        assert result.unparsed is None

    def test_json_embedded_in_prose_is_found(self) -> None:
        """A model told to answer in JSON usually does — and then explains itself.

        Finding the object inside is parsing. Anything more would be inventing.
        """
        text = 'Looking at the image: {"posture": "sitting"} — the subject is seated.'
        result = JsonCoercion().coerce(text, schema=SCHEMA)
        assert result.parsed == {"posture": "sitting"}
        assert result.reparsed
        assert result.unparsed == text, "the surrounding prose is preserved"

    def test_undeclared_fields_go_to_unparsed(self) -> None:
        """The model volunteered; the platform records without believing."""
        result = JsonCoercion().coerce(
            '{"posture": "standing", "is_violation": true}', schema=SCHEMA
        )
        assert result.parsed == {"posture": "standing"}
        assert "is_violation" in result.unparsed

    def test_malformed_json_is_preserved_not_repaired(self) -> None:
        """Repairing would mean guessing what the model meant to say, and a
        guessed attribute is indistinguishable from a real one downstream."""
        text = '{"posture": "stand'
        result = JsonCoercion().coerce(text, schema=SCHEMA)
        assert not result.parsed
        assert result.unparsed == text

    def test_an_array_is_not_an_object(self) -> None:
        """``["standing"]`` has not said *which field* is standing."""
        result = JsonCoercion().coerce('["standing"]', schema=SCHEMA)
        assert not result.parsed

    def test_prose_alone_parses_nothing(self) -> None:
        result = JsonCoercion().coerce("The subject appears upright.", schema=SCHEMA)
        assert not result.parsed
        assert result.unparsed == "The subject appears upright."

    def test_empty_text_parses_nothing(self) -> None:
        assert JsonCoercion().coerce("", schema=SCHEMA).parsed == {}
        assert JsonCoercion().coerce("   ", schema=SCHEMA).unparsed is None

    def test_a_runaway_generation_is_preserved_not_scanned(self) -> None:
        """Bounded so coercion never becomes the expensive step."""
        text = "x" * 200_000
        result = JsonCoercion().coerce(text, schema=SCHEMA)
        assert not result.parsed
        assert result.unparsed == text

    def test_coercion_is_deterministic(self) -> None:
        text = '{"posture": "standing", "extra": 1}'
        first = JsonCoercion().coerce(text, schema=SCHEMA)
        second = JsonCoercion().coerce(text, schema=SCHEMA)
        assert first.parsed == second.parsed
        assert first.unparsed == second.unparsed

    @pytest.mark.parametrize(
        "text", ["", "{", "null", "[1,2]", "\x00\x01", '{"a": ', "}{"]
    )
    def test_it_never_raises(self, text) -> None:
        JsonCoercion().coerce(text, schema=SCHEMA)


class TestKeyValueCoercion:
    def test_declared_keys_parse(self) -> None:
        result = KeyValueCoercion().coerce("posture: standing", schema=SCHEMA)
        assert result.parsed == {"posture": "standing"}

    def test_undeclared_keys_go_to_unparsed(self) -> None:
        result = KeyValueCoercion().coerce(
            "posture: standing\nis_violation: true", schema=SCHEMA
        )
        assert result.parsed == {"posture": "standing"}
        assert "is_violation" in result.unparsed

    def test_booleans_and_numbers_convert(self) -> None:
        result = KeyValueCoercion().coerce("headwear_present: true", schema=SCHEMA)
        assert result.parsed == {"headwear_present": True}

    def test_yes_stays_a_string(self) -> None:
        """The rejection is *information* — a prompt that should have said
        "answer true or false". Mapping it would hide the prompt bug forever."""
        result = KeyValueCoercion().coerce("headwear_present: yes", schema=SCHEMA)
        assert result.parsed == {"headwear_present": "yes"}

    def test_prose_lines_are_preserved(self) -> None:
        result = KeyValueCoercion().coerce(
            "posture: standing\nI am not certain about this.", schema=SCHEMA
        )
        assert result.parsed == {"posture": "standing"}
        assert "not certain" in result.unparsed

    def test_no_natural_language_extraction(self) -> None:
        """Deliberately narrow. Inferring a field from a sentence would make the
        coercion layer a second understanding layer with no schema and no
        evidence."""
        result = KeyValueCoercion().coerce(
            "The person is clearly standing upright", schema=SCHEMA
        )
        assert not result.parsed

    def test_an_empty_separator_is_refused(self) -> None:
        with pytest.raises(ValueError):
            KeyValueCoercion(separator="")


class TestPassthroughCoercion:
    def test_it_parses_nothing_and_preserves_everything(self) -> None:
        result = PassthroughCoercion().coerce('{"posture": "standing"}', schema=SCHEMA)
        assert not result.parsed
        assert result.unparsed == '{"posture": "standing"}'


class TestThePromptProvider:
    def test_it_resolves_a_covering_prompt(self, prompts) -> None:
        resolved = prompts.resolve((POSTURE,), class_id=PERSON, model_family="any")
        assert resolved == (PromptId("person.posture"), "1.0.0")

    def test_it_prefers_the_narrowest_covering_prompt(self, prompts) -> None:
        """A prompt asking for one attribute is cheaper than one asking for three,
        and 04_MODULES §M10 evaluates packs on token count for exactly this
        reason."""
        resolved = prompts.resolve((POSTURE,), class_id=PERSON, model_family="any")
        assert resolved[0] == PromptId("person.posture")

    def test_a_class_scoped_prompt_beats_a_universal_one(self) -> None:
        provider = StaticPromptProvider(
            (
                PromptTemplate(
                    prompt_id=PromptId("universal"),
                    version="1.0.0",
                    template="Describe {class_id}.",
                    output_keys=(POSTURE,),
                ),
                PromptTemplate(
                    prompt_id=PromptId("person.specific"),
                    version="1.0.0",
                    template="Describe the person.",
                    output_keys=(POSTURE,),
                    applies_to=(PERSON,),
                ),
            )
        )
        resolved = provider.resolve((POSTURE,), class_id=PERSON, model_family="any")
        assert resolved[0] == PromptId("person.specific")

    def test_no_covering_prompt_returns_none(self, prompts) -> None:
        """``NoSuitablePrompt`` is a **normal** outcome that becomes a capability
        gap; raising would make an expected answer look like a fault."""
        assert prompts.resolve((UNREGISTERED,), class_id=PERSON, model_family="a") is None

    def test_a_class_mismatch_returns_none(self, prompts) -> None:
        assert prompts.resolve((POSTURE,), class_id=VEHICLE, model_family="a") is None

    def test_a_class_independent_prompt_applies_everywhere(self, prompts) -> None:
        resolved = prompts.resolve((HEIGHT,), class_id=VEHICLE, model_family="a")
        assert resolved == (PromptId("generic.geometry"), "2.1.0")

    def test_resolution_is_deterministic(self, prompts) -> None:
        picks = {
            prompts.resolve((POSTURE,), class_id=PERSON, model_family="a")
            for _ in range(10)
        }
        assert len(picks) == 1

    def test_rendering_pins_the_version(self, prompts) -> None:
        rendered = prompts.render(
            PromptId("person.posture"), "1.0.0", {"class_id": "person"}
        )
        assert rendered.pinned == "person.posture@1.0.0"
        assert rendered.content_hash.startswith("sha256:")

    def test_rendering_an_unknown_prompt_raises(self, prompts) -> None:
        with pytest.raises(PromptUnavailableError, match="not declared"):
            prompts.render(PromptId("ghost"), "1.0.0", {})

    def test_a_missing_context_variable_fails_the_request(self, prompts) -> None:
        """§M10: *"Fail the single request, count... never crash the engine."*

        Rendering a sentence with a hole in it would produce a question a model
        then answers confidently.
        """
        with pytest.raises(PromptUnavailableError, match="failed to render"):
            prompts.render(PromptId("person.posture"), "1.0.0", {})

    def test_a_published_version_is_immutable(self, prompts) -> None:
        """*"Provenance is worthless if `prompt@3.2.0` means different things on
        different days."*"""
        with pytest.raises(ValueError, match="immutable"):
            prompts.add(
                PromptTemplate(
                    prompt_id=PromptId("person.posture"),
                    version="1.0.0",
                    template="A completely different instruction.",
                    output_keys=(POSTURE,),
                )
            )

    def test_re_adding_an_identical_template_is_allowed(self, prompts) -> None:
        prompts.add(prompts.templates[0])

    def test_a_prompt_with_no_output_keys_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no output keys"):
            PromptTemplate(
                prompt_id=PromptId("p"),
                version="1.0.0",
                template="Describe.",
                output_keys=(),
            )

    def test_unregistered_declarations_are_caught_at_load(self) -> None:
        """A narrowed form of the ceiling's second gate: a broken prompt fails
        before it costs a model call."""
        provider = StaticPromptProvider(
            (
                PromptTemplate(
                    prompt_id=PromptId("bad"),
                    version="1.0.0",
                    template="Is this a violation?",
                    output_keys=(UNREGISTERED,),
                ),
            )
        )
        violations = provider.validate_against(build_registry())
        assert violations
        assert "is_violation" in violations[0]

    def test_a_valid_pack_reports_no_violations(self, prompts) -> None:
        assert prompts.validate_against(build_registry()) == ()

    def test_declared_attributes_are_enumerable(self, prompts) -> None:
        assert POSTURE in prompts.declared_attributes()


class TestReferenceUnderstanders:
    def test_the_scripted_understander_filters_undeclared_fields(self) -> None:
        """A **U1-compliant** adapter does the platform's work for it.

        The engine's gate still runs — defence in depth — but a well-behaved
        adapter never presents an undeclared field as structured output.
        """
        adapter = scripted(
            ScriptedAnswer(fields={str(POSTURE): "standing", str(UNREGISTERED): True}),
            producible=(POSTURE,),
        )
        from ..conftest import make_request
        from .test_engine import build_result  # noqa: F401 - import parity

        response = adapter.understand(_port_request())
        assert str(UNREGISTERED) not in response.structured
        assert str(UNREGISTERED) in response.unparsed
        assert make_request is not None

    def test_it_preserves_raw_output(self) -> None:
        adapter = scripted(ScriptedAnswer(fields={str(POSTURE): "standing"}))
        response = adapter.understand(_port_request())
        assert json.loads(response.raw_output)["posture"] == "standing"

    def test_running_past_the_script_returns_nothing(self) -> None:
        """Not wrapping: a test that ran past its script should see *no answer*,
        never the first answer again."""
        adapter = scripted(ScriptedAnswer(fields={str(POSTURE): "standing"}))
        adapter.understand(_port_request())
        second = adapter.understand(_port_request())
        assert not second.structured

    def test_a_specialized_head_declares_one_attribute(self, head) -> None:
        assert head.capabilities().producible_attributes == (HEADWEAR,)
        assert head.capabilities().cost_class < 0.1

    def test_a_head_asked_the_wrong_question_says_nothing(self, head) -> None:
        """It does not answer the wrong question (U1)."""
        response = head.understand(_port_request(fields=(POSTURE,)))
        assert not response.structured

    def test_the_unavailable_understander_always_raises(self, unavailable) -> None:
        from vision_os.core.errors import UnderstanderUnavailableError

        with pytest.raises(UnderstanderUnavailableError):
            unavailable.understand(_port_request())

    def test_batch_is_total_even_on_failure(self) -> None:
        adapter = scripted(
            ScriptedAnswer(raise_timeout=True),
            ScriptedAnswer(fields={str(POSTURE): "standing"}),
        )
        requests = [_port_request(suffix=str(i)) for i in range(2)]
        responses = adapter.understand_batch(requests)
        assert set(responses) == {r.request_id for r in requests}


def _port_request(*, suffix: str = "1", fields: tuple[AttributeKey, ...] = (POSTURE,)):
    from vision_os.core.model.ids import CropId, RequestId
    from vision_os.core.ports.understanding import (
        CropView,
        RenderedPrompt,
        UnderstandingPortRequest,
    )

    prompt = RenderedPrompt(
        prompt_id=PromptId("p"),
        version="1.0.0",
        text="Describe.",
        output_schema=OutputSchema(fields=fields),
    )
    return UnderstandingPortRequest(
        request_id=RequestId(f"req-{suffix}"),
        crops=(
            CropView(
                crop_id=CropId("crop-1"),
                pixels=memoryview(bytes(8 * 8 * 3)),
                width=8,
                height=8,
            ),
        ),
        prompt=prompt,
        output_schema=prompt.output_schema,
        context={"class_id": str(ClassId("person"))},
    )
