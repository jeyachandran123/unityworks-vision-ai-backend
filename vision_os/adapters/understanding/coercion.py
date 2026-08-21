"""Output coercion strategies — P16.

> 04_MODULES §M9 responsibility 4: *"Coerce raw model output into the declared
> schema, and quarantine what does not fit."*

Two strategies ship. Both obey X2 absolutely: **anything not parsed appears in
``unparsed``**. A strategy that dropped a fragment would be discarding model
output, and 02_VOM §9.3 is explicit that what does not coerce is *preserved* —
inspectable, never promoted.

The escalation is deliberate: strict JSON first, because a model with structured
output support produces exactly that and there is nothing to guess at; then a
lenient pass that finds a JSON object inside prose, because a model told to
answer in JSON usually does so *and then explains itself*. Anything past that is
not parsing, it is inventing, and X1 forbids it.
"""

from __future__ import annotations

import json
import re

from ...core.ports.understanding import CoercionResult, OutputSchema

#: Longest raw text a strategy will scan, in characters.
#:
#: A runaway generation must not turn coercion into the expensive step. Text past
#: the bound is preserved as ``unparsed`` rather than parsed — which is the honest
#: outcome, since a structured answer that far into a response is not an answer.
MAX_SCAN_CHARS = 64_000

_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class JsonCoercion:
    """Strict JSON, then a bounded search for an object inside prose.

    Never repairs. A truncated or malformed object is left to the quarantine
    path: repairing it would mean guessing what the model meant to say, and a
    guessed attribute is indistinguishable from a real one downstream — exactly
    the failure U2 exists to prevent.
    """

    __slots__ = ()

    @property
    def strategy_id(self) -> str:
        return "coercion.json"

    def coerce(self, raw: str, *, schema: OutputSchema) -> CoercionResult:
        """Parse ``raw`` against the schema. **Never raises** (obligation X4)."""
        text = (raw or "").strip()
        if not text:
            return CoercionResult(strategy_used=self.strategy_id)
        if len(text) > MAX_SCAN_CHARS:
            return CoercionResult(
                unparsed=text,
                strategy_used=self.strategy_id,
            )

        parsed = _load_object(text)
        if parsed is not None:
            return self._split(parsed, text, schema, reparsed=False)

        # A model asked for JSON often answers in JSON and then adds a sentence.
        # Finding the object inside is parsing; anything more would be inventing.
        match = _OBJECT_PATTERN.search(text)
        if match:
            embedded = _load_object(match.group(0))
            if embedded is not None:
                return self._split(embedded, text, schema, reparsed=True)

        return CoercionResult(unparsed=text, strategy_used=self.strategy_id)

    def _split(
        self,
        parsed: dict,
        original: str,
        schema: OutputSchema,
        *,
        reparsed: bool,
    ) -> CoercionResult:
        """Separate declared fields from everything else.

        Undeclared keys go to ``unparsed`` as text rather than being dropped
        (X2) — and they are *not* returned as parsed fields, because U1 says an
        adapter returns what the schema declared *"and nothing else"*. The
        platform still sees them, in the note, where a human can notice a model
        volunteering something nobody asked for.
        """
        declared = {str(key) for key in schema.fields}
        kept = {key: value for key, value in parsed.items() if key in declared}
        extra = {key: value for key, value in parsed.items() if key not in declared}

        unparsed: str | None = None
        if extra:
            unparsed = json.dumps(extra, sort_keys=True, default=str)
        elif reparsed:
            # The prose around the object is itself worth preserving: it is often
            # where a model explains a hedge that the structured answer hides.
            unparsed = original

        return CoercionResult(
            parsed=kept,
            unparsed=unparsed,
            strategy_used=self.strategy_id,
            reparsed=reparsed,
        )


class KeyValueCoercion:
    """``key: value`` lines, for models that will not emit JSON.

    Deliberately narrow. It reads lines of the form ``key: value`` and nothing
    else — no natural-language extraction, no synonym matching, no inference
    about what a sentence "probably means". Those would all be the coercion layer
    quietly becoming a second understanding layer with no schema and no evidence.
    """

    __slots__ = ("_separator",)

    def __init__(self, separator: str = ":") -> None:
        if not separator:
            raise ValueError("a key/value separator cannot be empty")
        self._separator = separator

    @property
    def strategy_id(self) -> str:
        return "coercion.keyvalue"

    def coerce(self, raw: str, *, schema: OutputSchema) -> CoercionResult:
        text = (raw or "").strip()
        if not text or len(text) > MAX_SCAN_CHARS:
            return CoercionResult(
                unparsed=text or None, strategy_used=self.strategy_id
            )

        declared = {str(key) for key in schema.fields}
        parsed: dict[str, object] = {}
        leftovers: list[str] = []

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            key, separator, value = stripped.partition(self._separator)
            key = key.strip()
            if not separator or key not in declared:
                leftovers.append(stripped)
                continue
            parsed[key] = _scalar(value.strip())

        return CoercionResult(
            parsed=parsed,
            unparsed="\n".join(leftovers) if leftovers else None,
            strategy_used=self.strategy_id,
        )


class PassthroughCoercion:
    """Parses nothing; preserves everything.

    For an adapter with native structured output, where a second parse would be
    a second interpretation of an answer that was already unambiguous. Also the
    honest baseline a conformance kit checks the others against.
    """

    __slots__ = ()

    @property
    def strategy_id(self) -> str:
        return "coercion.passthrough"

    def coerce(self, raw: str, *, schema: OutputSchema) -> CoercionResult:
        text = (raw or "").strip()
        return CoercionResult(
            unparsed=text or None, strategy_used=self.strategy_id
        )


def _load_object(text: str) -> dict | None:
    """Parse a JSON object, or ``None``. Arrays and scalars are not objects.

    A model returning ``["standing"]`` has not answered *which field* is
    ``standing``, and guessing would attribute a value to a key nobody named.
    """
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _scalar(text: str) -> object:
    """Interpret a bare token conservatively.

    Only unambiguous literals convert. ``"yes"`` stays a string, because a schema
    expecting a boolean will reject it with ``WRONG_TYPE`` and that rejection is
    *information* — a prompt that should have said "answer true or false". A
    coercion layer that helpfully mapped it would hide the prompt bug forever.
    """
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("null", "none", ""):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text
