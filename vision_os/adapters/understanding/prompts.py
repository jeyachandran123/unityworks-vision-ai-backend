"""A minimal prompt provider — **a stand-in for M10, not an implementation of it**.

M9 cannot run without something answering *"which prompt covers these attributes,
and what output schema does it declare"*. M10 Prompt Manager is that something,
and M10 is a separate module in a later flow.

What this provides is the narrow consumer-side contract M9 uses — `resolve`,
`render`, `schema_of` — served from prompts declared in configuration. What it
deliberately does **not** provide, because these are M10's responsibilities and
implementing them here would be building M10 early:

* prompt **packs** and their load-time validation;
* the **neutrality gate** on declared output keys (00_CHARTER §4.3 gate 2);
* A/B and **shadow** variants;
* **model-family** variant resolution;
* hot reload with copy-on-write catalogue swap;
* content-hash detection of a mutated published version.

It does keep the two properties M9's correctness depends on, because losing
either would make the engine untestable rather than merely limited:

**Versions are pinned and immutable.** A rendered prompt carries
`prompt_id@version` and a content hash, so provenance means something even
though the catalogue is small.

**Declared output keys are checked against the Attribute Schema Registry.** Not
the full neutrality gate — that is registration, and registration is gate 1's job
— but a prompt declaring an unregistered key is refused at load rather than
producing rejections at inference time. M10 will do this more thoroughly; doing
none of it here would let a broken prompt look fine until it cost money.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...core.errors import PromptUnavailableError
from ...core.model.ids import AttributeKey, ClassId, PromptId
from ...core.ports.understanding import OutputSchema, RenderedPrompt

#: The provider's identity, chosen so nobody mistakes it for M10 in a log line.
PROVIDER_ID = "prompt.static"


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """One declared prompt. Immutable once constructed.

    ``applies_to`` empty means "any class" — the degenerate case, not an
    omission, since some attributes are genuinely class-independent.
    """

    prompt_id: PromptId
    version: str
    template: str
    output_keys: tuple[AttributeKey, ...]
    applies_to: tuple[ClassId, ...] = ()
    max_output_tokens: int = 512

    def __post_init__(self) -> None:
        if not self.prompt_id or not self.version:
            raise ValueError("a prompt template requires an id and a version")
        if not self.output_keys:
            raise ValueError(
                f"prompt '{self.prompt_id}' declares no output keys; a prompt with "
                f"no declared schema cannot be validated against the registry, "
                f"which is the second of the ceiling's three gates"
            )
        if "{" not in self.template and "}" not in self.template:
            return
        # A template with braces must be renderable; catching it here means a
        # bad template fails at load rather than on the first paying request.
        try:
            self.template.format_map(_Blanks())
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"prompt '{self.prompt_id}@{self.version}' has a malformed "
                f"template: {exc}"
            ) from exc

    @property
    def pinned(self) -> str:
        return f"{self.prompt_id}@{self.version}"

    def covers(self, attributes: Sequence[AttributeKey]) -> bool:
        declared = set(self.output_keys)
        return bool(attributes) and all(key in declared for key in attributes)

    def applies(self, class_id: ClassId) -> bool:
        if not self.applies_to:
            return True
        return any(
            class_id == allowed or class_id.startswith(f"{allowed}.")
            for allowed in self.applies_to
        )


class _Blanks(dict):
    """Renders any placeholder as empty, for load-time template validation."""

    def __missing__(self, key: str) -> str:  # noqa: D105
        return ""


class StaticPromptProvider:
    """Serves declared prompts. The M10 seam, satisfied minimally.

    Deterministic in every respect: resolution walks templates in a stable order
    and returns the **most specific** match, so two identical requests always get
    the same prompt and a replay reproduces the same provenance (V13).
    """

    __slots__ = ("_templates",)

    def __init__(self, templates: Sequence[PromptTemplate] = ()) -> None:
        self._templates: dict[tuple[PromptId, str], PromptTemplate] = {}
        for template in templates:
            self.add(template)

    def add(self, template: PromptTemplate) -> None:
        key = (template.prompt_id, template.version)
        existing = self._templates.get(key)
        if existing is not None and existing != template:
            raise ValueError(
                f"prompt '{template.pinned}' is already declared with different "
                f"content; published versions are immutable, because provenance "
                f"is worthless if a pinned version means different things on "
                f"different days (04_MODULES section M10)"
            )
        self._templates[key] = template

    def validate_against(self, registry) -> tuple[str, ...]:
        """Refuse prompts declaring unregistered keys. Returns the violations.

        A narrowed form of M10's gate 2. The full gate also runs the neutrality
        check on the *declaration*; here the key must simply exist in the
        registry, which already passed neutrality at registration. Doing this at
        load means a broken prompt fails before it costs a model call.
        """
        violations: list[str] = []
        for template in self._ordered():
            for key in template.output_keys:
                if registry.get(key) is None:
                    violations.append(
                        f"{template.pinned} declares unregistered attribute '{key}'"
                    )
        return tuple(violations)

    # --- the M10 contract M9 uses ------------------------------------------------ #

    def resolve(
        self,
        attributes: Sequence[AttributeKey],
        *,
        class_id: ClassId,
        model_family: str,
    ) -> tuple[PromptId, str] | None:
        """Which prompt covers this attribute set for this class.

        ``None`` rather than an exception: 04_MODULES §M10 makes
        ``NoSuitablePrompt`` a normal outcome that becomes a capability gap, and
        raising would make an expected answer look like a fault.

        ``model_family`` is accepted and **ignored** — family variants are M10's,
        and pretending to resolve them here would hide their absence.
        """
        candidates = [
            template
            for template in self._ordered()
            if template.applies(class_id) and template.covers(attributes)
        ]
        if not candidates:
            return None
        # Most specific first: a class-scoped prompt beats a universal one, then
        # the narrowest output schema, then the pinned name for stability.
        best = min(
            candidates,
            key=lambda t: (0 if t.applies_to else 1, len(t.output_keys), t.pinned),
        )
        return (best.prompt_id, best.version)

    def render(
        self, prompt_id: PromptId, version: str, context: Mapping[str, Any]
    ) -> RenderedPrompt:
        """Render a pinned prompt with context.

        Raises:
            PromptUnavailableError: unknown prompt, or a render error. §M10:
                *"Fail the single request, count, fall back to the previous prompt
                version; never crash the engine."* The engine treats this as a
                capability gap for this request only.
        """
        template = self._templates.get((prompt_id, version))
        if template is None:
            raise PromptUnavailableError(
                f"prompt '{prompt_id}@{version}' is not declared",
                prompt_id=str(prompt_id),
                version=version,
            )
        try:
            text = template.template.format_map(_Rendering(context))
        except Exception as exc:  # noqa: BLE001 - a template may fail many ways
            raise PromptUnavailableError(
                f"prompt '{template.pinned}' failed to render: {exc}",
                prompt_id=str(prompt_id),
                version=version,
            ) from exc

        return RenderedPrompt(
            prompt_id=template.prompt_id,
            version=template.version,
            text=text,
            output_schema=OutputSchema(fields=template.output_keys),
            content_hash=_hash(text),
            model_family="",
            max_output_tokens=template.max_output_tokens,
        )

    def schema_of(self, prompt_id: PromptId, version: str) -> OutputSchema:
        template = self._templates.get((prompt_id, version))
        if template is None:
            raise PromptUnavailableError(
                f"prompt '{prompt_id}@{version}' is not declared",
                prompt_id=str(prompt_id),
                version=version,
            )
        return OutputSchema(fields=template.output_keys)

    # --- access -------------------------------------------------------------------- #

    def _ordered(self) -> list[PromptTemplate]:
        return [self._templates[key] for key in sorted(self._templates)]

    @property
    def provider_id(self) -> str:
        return PROVIDER_ID

    @property
    def templates(self) -> tuple[PromptTemplate, ...]:
        return tuple(self._ordered())

    def declared_attributes(self) -> frozenset[AttributeKey]:
        keys: set[AttributeKey] = set()
        for template in self._templates.values():
            keys.update(template.output_keys)
        return frozenset(keys)

    def __len__(self) -> int:
        return len(self._templates)


class _Rendering(dict):
    """Context lookup that names a missing key rather than swallowing it.

    A prompt referring to a variable nobody supplied is a template bug, and §M10
    wants it to fail the single request loudly rather than render a sentence with
    a hole in it that a model then answers confidently.
    """

    def __init__(self, context: Mapping[str, Any]) -> None:
        super().__init__(context)

    def __missing__(self, key: str) -> str:
        raise KeyError(f"template variable '{key}' was not supplied in context")


def _hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()
