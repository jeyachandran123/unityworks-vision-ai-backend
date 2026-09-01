"""The parser against **MiniMax M3's** actual output, recorded live.

### Why this file exists separately

`test_vlm_model_replacement.py` already tests the parser against a braceless
reply. That sample is **Llama's** — recorded from
`meta/llama-3.2-11b-vision-instruct`, which answered a "respond with JSON"
prompt with the object body and no braces at all. It stays attributed there.

MiniMax M3 does something different: it fences. Every successful reply observed
on 2026-08-31 came back inside a ```json block, which is a shape the parser
already handles — but "already handles" was an assumption until it was checked
against the model actually deployed, and §10 of the live-hardening brief says
not to use one model's samples as evidence for another.

### The samples

`FENCED` and `FENCED_THREE_KEYS` are **verbatim** from `minimaxai/minimax-m3`
against the shipped kitchen prompt and real kitchen-01 crops on 2026-08-31.
The rest are constructed degradations, and are labelled as such.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.understanding.payload import extract_json, split_by_schema
from vision_os.core.model.ids import AttributeKey
from vision_os.core.ports.understanding import OutputSchema

#: Verbatim from `minimaxai/minimax-m3`, 2026-08-31, kitchen-01 f00060/s0.
FENCED = (
    "```json\n"
    "{\n"
    '  "head_covering": "hairnet",\n'
    '  "face_covering": "not_visible",\n'
    '  "hand_covering": "not_visible"\n'
    "}\n"
    "```"
)

#: Verbatim from `minimaxai/minimax-m3`, 2026-08-31, kitchen-01 f00180/s0.
FENCED_THREE_KEYS = (
    "```json\n"
    "{\n"
    '  "head_covering": "cap",\n'
    '  "face_covering": "not_visible",\n'
    '  "hand_covering": "not_visible"\n'
    "}\n"
    "```"
)

SCHEMA = OutputSchema(
    fields=(
        AttributeKey("head_covering"),
        AttributeKey("face_covering"),
        AttributeKey("hand_covering"),
    ),
    strict=False,
)


class TestTheShapeMiniMaxActuallyReturns:
    def test_a_fenced_object_parses(self) -> None:
        assert extract_json(FENCED) == {
            "head_covering": "hairnet",
            "face_covering": "not_visible",
            "hand_covering": "not_visible",
        }

    def test_the_second_recorded_reply_parses(self) -> None:
        assert extract_json(FENCED_THREE_KEYS)["head_covering"] == "cap"

    def test_all_three_ppe_keys_survive_the_schema_split(self) -> None:
        """The keys must reach the registry, not just the parser."""
        structured, leftover = split_by_schema(extract_json(FENCED), SCHEMA)

        assert set(structured) == {"head_covering", "face_covering", "hand_covering"}
        assert leftover in (None, {}, "")

    def test_a_refusal_value_is_carried_through_unchanged(self) -> None:
        """`not_visible` is the answer the whole uncertainty design rests on. If
        the parser dropped it the rule would see an absent attribute instead of
        an observed refusal — both UNKNOWN, but for different reasons, and §17
        requires those to stay distinguishable."""
        structured, _ = split_by_schema(extract_json(FENCED), SCHEMA)
        assert structured["hand_covering"] == "not_visible"


class TestDegradedShapes:
    """Constructed, not recorded. Each is a way a reply can arrive broken."""

    def test_bare_json_parses(self) -> None:
        assert extract_json('{"head_covering": "none"}') == {"head_covering": "none"}

    def test_leading_and_trailing_whitespace_parses(self) -> None:
        assert extract_json('\n\n  {"head_covering": "cap"}  \n\t') == {
            "head_covering": "cap"
        }

    def test_a_fence_without_a_language_tag_parses(self) -> None:
        assert extract_json('```\n{"head_covering": "hood"}\n```') == {
            "head_covering": "hood"
        }

    def test_prose_around_the_object_parses(self) -> None:
        text = 'Looking at the image:\n{"head_covering": "hairnet"}\nHope that helps.'
        assert extract_json(text) == {"head_covering": "hairnet"}

    def test_a_missing_field_is_absent_not_invented(self) -> None:
        """The parser must not fill a gap. An absent key becomes an absent
        attribute, which the rule reads as UNKNOWN — never as a value."""
        structured, _ = split_by_schema(
            extract_json('{"head_covering": "none"}'), SCHEMA
        )
        assert structured == {"head_covering": "none"}
        assert "hand_covering" not in structured

    def test_an_unknown_field_does_not_contaminate_the_schema(self) -> None:
        decoded = extract_json(
            '{"head_covering": "none", "mood": "cheerful", "confidence": 0.9}'
        )
        structured, leftover = split_by_schema(decoded, SCHEMA)

        assert structured == {"head_covering": "none"}
        assert "mood" not in structured

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "I cannot determine that from this image.",
            "{not json at all",
            '{"head_covering": ',
            "```json\n{broken\n```",
        ],
        ids=["empty", "blank", "prose-only", "unclosed", "truncated", "fenced-broken"],
    )
    def test_an_unusable_reply_returns_none(self, text: str) -> None:
        """`None`, so `understand()` records it as unparseable and returns a
        refusal. Anything that produced a partial dict here would let a
        half-read answer become an attribute."""
        assert extract_json(text) is None


class TestNoShapeBecomesCompliance:
    """§24: no malformed or unavailable inference may silently become a value."""

    @pytest.mark.parametrize(
        "text", ["", "I cannot tell.", "{broken", "```json\n{oops\n```"]
    )
    def test_nothing_unusable_yields_a_ppe_value(self, text: str) -> None:
        decoded = extract_json(text)
        assert decoded is None
        # And the only path from here is `structured={}` — asserted in the
        # adapter's own refusal tests rather than re-derived here.

    def test_a_parsed_object_never_defaults_a_missing_key(self) -> None:
        """The failure mode this guards: a parser that returned
        `{"hand_covering": "gloves"}` for a reply that never mentioned hands
        would manufacture compliance out of silence."""
        structured, _ = split_by_schema(extract_json('{"face_covering": "mask"}'), SCHEMA)
        assert structured == {"face_covering": "mask"}
        assert structured.get("hand_covering") is None
        assert structured.get("head_covering") is None
