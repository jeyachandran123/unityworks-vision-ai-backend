"""Use-case semantic policy — a document, loaded onto existing seams.

### Why this is a loader and not a pipeline stage

Everything a policy needs to express already exists as a platform contract:

| A policy declares | The platform already has |
|---|---|
| semantic definitions | ``AttributeSchema`` → ``AttributeRegistry`` |
| which classes it applies to | ``SubjectFilter.class_ids`` |
| how fresh an answer must be | ``Demand.freshness`` |
| when to re-look | ``TriggerHint`` |
| how much it may spend | ``DemandBudget`` |
| what may come back | ``OutputSchema`` |
| the question, versioned | ``PromptTemplate`` → P17 |
| what is accepted | ``AttributeValidator`` |

So a "rules engine" would be a second implementation of seven things that work.
This module reads a document and calls those APIs. It adds no stage, no port and
no decision: the Understanding Engine never learns that policies exist.

### Why there is no domain vocabulary in this file

Not one attribute name, enum value, class name or prompt sentence appears here.
`kitchen`, `hairnet`, `apron`, `shirt_colour` are **data in a JSON file**, and a
deployment adds a use case by writing one — no change to this module, the VLM
adapters, the detector, the engine or the harness.

The test that matters is `test_a_new_use_case_needs_no_code_change`: it invents a
semantic field nothing in this repository has heard of and drives it end to end.

### The ceiling still holds

A policy can *ask* for anything. It cannot *grant* anything. Every attribute it
declares still passes the registry's neutrality gate at registration, the
`OutputSchema` still bounds what an adapter may return, and `AttributeValidator`
still re-checks every field against the registry afterwards. A policy widens the
question; it never widens what the platform will accept as true.

### No policy is a supported configuration

Absent a policy: no attributes registered, no demands raised, therefore no
understanding requests and no model calls. Detection, tracking, the registry,
presence and spatial observations, Vision State and the Observation API all run
exactly as before. That is a platform without a semantic use case, not a broken
one, and it is why no default domain is ever invented.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...core.errors import ConfigurationError
from ...core.model.demand import (
    Demand,
    DemandBudget,
    DemandScope,
    SubjectFilter,
    TriggerHint,
)
from ...core.model.ids import (
    AttributeKey,
    CameraId,
    ClassId,
    DemandId,
    PromptId,
    SubscriberId,
)
from ...core.model.timebase import Duration, Instant

#: Where a deployment names its active policy. A path, or empty for none.
POLICY_ENV = "VISION_SEMANTIC_POLICY"

_VALUE_TYPES = {"enum", "bool", "scalar", "text", "count"}


@dataclass(frozen=True, slots=True)
class SemanticRequirement:
    """One thing a policy wants known about an object.

    ``justification`` is not decoration: ``AttributeSchema`` requires it, and the
    registry's neutrality gate is the reason a policy cannot smuggle a judgment
    in as an attribute. It must describe **visible evidence** — what a camera can
    see — and a policy whose justification is a business rule will read as one to
    the human reviewing it.
    """

    key: str
    value_type: str
    justification: str
    values: tuple[str, ...] = ()
    validity_ms: int = 30_000
    question: str = ""
    """Optional per-field wording. Rendered into the prompt when present."""

    min_quality: Mapping[str, float] | None = None
    """Quality floors a crop must clear before this attribute is asked at all.

    Keys are ``GateThresholds`` field names — ``min_scale_pixels``,
    ``max_blur``, ``max_truncation``, ``max_occlusion``. ``None`` means the
    deployment's default gate applies.

    Declared per attribute because **what counts as usable depends on the
    question**. A whole-person crop 60px tall answers "what colour is the
    garment" perfectly well; the head band inside it is 27px and cannot answer
    "is the head covered". A single global floor has to be wrong for one of
    them, and it fails silently either way — too low and a blurred region
    returns a confident answer, too high and every distant subject goes
    unexamined.

    These are **calibration candidates**, not production values. The right
    numbers come from measuring annotated footage, and until that measurement
    exists a deployment should treat any value here as provisional."""

    output_size: tuple[int, int] | None = None
    """Canonical crop size for evidence answering this attribute. ``None`` means
    the deployment default.

    Separate from ``evidence_region`` because they fix different losses.
    The region decides *what* is in the crop; this decides *how much detail*
    survives being resampled into it. Narrowing to a body part stops spending the
    canvas on the rest of the subject, but that band is still resampled to the
    canonical size — so a small feature can arrive as a few dozen pixels either
    way, and be read as absent.

    Measured on annotated footage (``datasets/kitchen-01``), raising one
    attribute's crop from 224 to 448 moved its accuracy from 23.3% to 74.4% with
    an identical model, prompt, region and detector.

    Declared per attribute rather than raised globally because vision tokens
    scale with **area**: 448 costs 4x the tokens of 224, and paying that on
    every question to fix one would be a cost with no measured return."""

    evidence_region: tuple[float, float] | None = None
    """Where on the subject this attribute is visible, as ``(top, height)``
    fractions of its bounding box. ``None`` means the whole subject.

    Geometry, not meaning. P14's obligation C5 permits a strategy to choose
    geometry from what is being asked — *"a head region for a person"* — and
    forbids it using what the class means to a business. This is that permission
    exercised from the document rather than from code: the policy that declares
    an attribute is the only thing that knows which part of a subject answers it,
    and ``crop.part_focused`` reads it without learning what the attribute is.

    Absent this, a tall subject letterboxes into the square canonical crop with
    most of the image as black bar, and the region a question is about lands in a
    few dozen pixels — measurably too few for a model to answer from, which is
    what real footage showed."""

    def __post_init__(self) -> None:
        if not self.key:
            raise ConfigurationError("a semantic requirement needs a key")
        if self.value_type not in _VALUE_TYPES:
            raise ConfigurationError(
                f"'{self.key}' declares value_type '{self.value_type}'; "
                f"supported: {', '.join(sorted(_VALUE_TYPES))}"
            )
        if self.value_type == "enum" and not self.values:
            raise ConfigurationError(
                f"'{self.key}' is an enum with no values. An unconstrained enum "
                f"is a text field wearing a type, and the registry refuses it"
            )
        if not self.justification.strip():
            raise ConfigurationError(
                f"'{self.key}' declares no neutrality justification. The registry "
                f"requires one, and it is what keeps a business rule from being "
                f"registered as an observation"
            )


@dataclass(frozen=True, slots=True)
class SemanticPolicy:
    """One use case, as data.

    Carries no behaviour beyond turning itself into platform objects. Two
    policies differ only in their contents, which is the property that makes a
    new deployment a file rather than a release.
    """

    policy_id: str
    version: str
    object_classes: tuple[str, ...]
    requirements: tuple[SemanticRequirement, ...]
    prompt_id: str = ""
    prompt_version: str = "1.0.0"
    prompt_preamble: str = ""
    max_output_tokens: int = 256
    freshness_ms: int = 30_000
    trigger_hints: tuple[str, ...] = ("on_first_sight", "on_change")
    priority_class: str = ""
    max_calls_per_hour: int | None = None
    max_cost_per_hour: float | None = None
    min_confidence: float | None = None
    lifecycle: tuple[str, ...] = ()
    scene: str = ""
    """Free-text label for the deployment's own use. The platform never reads
    it — a scene that changed behaviour would be a domain branch."""

    labels: Mapping[str, str] = field(default_factory=dict)

    # --- construction ------------------------------------------------------- #

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> SemanticPolicy:
        try:
            requirements = tuple(
                SemanticRequirement(
                    key=str(entry["key"]),
                    value_type=str(entry.get("type", "enum")).lower(),
                    justification=str(entry.get("justification", "")),
                    values=tuple(str(v) for v in entry.get("values", ())),
                    validity_ms=int(entry.get("validity_ms", 30_000)),
                    question=str(entry.get("question", "")),
                    evidence_region=_region(entry.get("evidence_region")),
                    min_quality=_quality(entry.get("min_quality")),
                    output_size=_output_size(entry.get("output_size")),
                )
                for entry in document.get("attributes", ())
            )
        except KeyError as exc:
            raise ConfigurationError(f"attribute entry missing {exc}") from exc

        if not requirements:
            raise ConfigurationError(
                "a policy that requires nothing would raise a demand for nothing "
                "and cost a model call to discover it"
            )

        demand = document.get("demand", {})
        prompt = document.get("prompt", {})
        scope = document.get("scope", {})
        budget = demand.get("budget", {})

        policy_id = str(document.get("policy_id", "")).strip()
        if not policy_id:
            raise ConfigurationError("a policy must carry a policy_id for provenance")

        return cls(
            policy_id=policy_id,
            version=str(document.get("version", "1.0.0")),
            object_classes=tuple(str(c) for c in scope.get("object_classes", ())),
            requirements=requirements,
            prompt_id=str(prompt.get("prompt_id", "")) or f"{policy_id}.attributes",
            prompt_version=str(prompt.get("version", "1.0.0")),
            prompt_preamble=str(prompt.get("preamble", "")),
            max_output_tokens=int(prompt.get("max_output_tokens", 256)),
            freshness_ms=int(demand.get("freshness_ms", 30_000)),
            trigger_hints=tuple(str(t) for t in demand.get("triggers", ("on_first_sight", "on_change"))),
            priority_class=str(demand.get("priority_class", "")),
            max_calls_per_hour=_optional_int(budget.get("max_calls_per_hour")),
            max_cost_per_hour=_optional_float(budget.get("max_cost_per_hour")),
            min_confidence=_optional_float(scope.get("min_confidence")),
            lifecycle=tuple(str(s) for s in scope.get("lifecycle", ())),
            scene=str(document.get("scene", "")),
            labels={str(k): str(v) for k, v in document.get("labels", {}).items()},
        )

    @classmethod
    def from_file(cls, path: Path | str) -> SemanticPolicy:
        file = Path(path)
        if not file.is_file():
            raise ConfigurationError(f"semantic policy not found at '{file}'")
        try:
            document = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"'{file}' is not valid JSON: {exc}") from exc
        return cls.from_document(document)

    # --- projection onto platform contracts --------------------------------- #

    @property
    def attribute_keys(self) -> tuple[AttributeKey, ...]:
        return tuple(AttributeKey(r.key) for r in self.requirements)

    @property
    def class_ids(self) -> tuple[ClassId, ...]:
        return tuple(ClassId(c) for c in self.object_classes)

    @property
    def quality_floors(self) -> dict[str, dict[str, float]]:
        """Every declared quality floor, keyed by attribute.

        Handed to the gate at composition. Attributes that declared none are
        absent rather than mapped to the default, so the gate can tell "the
        document said nothing" from "the document said the default".
        """
        return {
            r.key: dict(r.min_quality)
            for r in self.requirements
            if r.min_quality
        }

    @property
    def output_sizes(self) -> dict[str, tuple[int, int]]:
        """Every declared canonical crop size, keyed by attribute.

        Handed to ``crop.part_focused`` at composition. Attributes that declared
        none are absent rather than mapped to the default, so the strategy can
        tell "the document said nothing" from "the document said the default".
        """
        return {
            r.key: r.output_size
            for r in self.requirements
            if r.output_size is not None
        }

    @property
    def evidence_regions(self) -> dict[str, tuple[float, float]]:
        """Every declared region, keyed by attribute. Handed to ``crop.part_focused``.

        Attributes that declared none are absent rather than mapped to the whole
        box, so a strategy can tell "the document said nothing" from "the
        document said all of it".
        """
        return {
            r.key: r.evidence_region
            for r in self.requirements
            if r.evidence_region is not None
        }

    def register_attributes(self, registry) -> tuple[AttributeKey, ...]:
        """Put every requirement through the registry's own gate.

        The registry validates: an enum with no domain is refused, a missing
        justification is refused. Nothing is force-registered — a policy that
        cannot pass the ceiling's first gate does not load.
        """
        from ...core.model.timebase import Duration as _Duration
        from ...perception.registry.attributes import (
            AttributeSchema,
            AttributeValueType,
        )

        applies_to = self.class_ids
        registered: list[AttributeKey] = []
        for requirement in self.requirements:
            registry.register(
                AttributeSchema(
                    key=AttributeKey(requirement.key),
                    value_type=AttributeValueType(requirement.value_type),
                    domain=requirement.values,
                    neutrality_justification=requirement.justification,
                    applies_to=applies_to,
                    validity=_Duration.from_millis(requirement.validity_ms),
                )
            )
            registered.append(AttributeKey(requirement.key))
        return tuple(registered)

    def build_prompt_template(self):
        """One versioned prompt bound to one output schema.

        The template is assembled from the policy's own wording. A prompt naming
        an unregistered key is refused at load — the ceiling's second gate — so a
        policy whose prompt and attributes disagree fails before it costs a call.
        """
        from ..understanding.prompts import PromptTemplate

        return PromptTemplate(
            prompt_id=PromptId(self.prompt_id),
            version=self.prompt_version,
            template=self.render_prompt(),
            output_keys=self.attribute_keys,
            applies_to=self.class_ids,
            max_output_tokens=self.max_output_tokens,
        )

    def render_prompt(self) -> str:
        """Assemble the question from the policy's data.

        No literal braces: the platform validates a template by rendering it with
        ``format_map``, so a JSON brace is read as a format placeholder and the
        prompt is rejected at load. Constrained decoding is what guarantees JSON
        anyway; the prompt only names the keys.
        """
        preamble = self.prompt_preamble.strip() or (
            "You are a visual attribute extractor for a video analytics platform. "
            "Look at the cropped image of a {class_id} and report ONLY what is "
            "directly visible. Do not infer intent, identity, role, emotion, or "
            "the purpose of any activity."
        )

        lines = [
            preamble,
            f"Respond with JSON containing exactly these {len(self.requirements)} keys:",
        ]
        for requirement in self.requirements:
            lines.append(f"  {json.dumps(requirement.key)}: {_expected(requirement)}")
            if requirement.question:
                lines.append(f"      {requirement.question.strip()}")
        lines.append(
            "Never answer outside the listed options, and never guess to fill a key."
        )
        return "\n".join(lines)

    def build_demand(
        self,
        *,
        subscriber: str,
        cameras: Sequence[CameraId] = (),
        now: Instant | None = None,
    ) -> Demand:
        """The consumer's declared need, as the platform models it.

        Everything that controls **how often a model is called** lives here and
        nowhere else: freshness, triggers and budget are the policy's, and the
        platform decides what it can actually sustain. No adapter is consulted,
        and no adapter could override it.
        """
        return Demand(
            demand_id=DemandId(f"{self.policy_id}@{self.version}"),
            subscriber=SubscriberId(subscriber),
            scope=DemandScope(camera_ids=tuple(cameras)),
            subject_filter=SubjectFilter(
                class_ids=self.class_ids,
                lifecycle=self.lifecycle,
                min_confidence=self.min_confidence,
            ),
            required_attributes=self.attribute_keys,
            freshness=Duration.from_millis(self.freshness_ms),
            trigger_hints=tuple(_trigger(hint) for hint in self.trigger_hints),
            priority_class=self.priority_class,
            budget=DemandBudget(
                max_calls_per_hour=self.max_calls_per_hour,
                max_cost_per_hour=self.max_cost_per_hour,
            ),
            registered_at=now or Instant(0),
            labels={
                # Provenance for every observation this demand causes: which
                # policy asked, and which version of it.
                "policy_id": self.policy_id,
                "policy_version": self.version,
                **self.labels,
            },
        )


def load_policy(
    path: Path | str | None = None, *, env: Mapping[str, str] | None = None
) -> SemanticPolicy | None:
    """The active policy, or ``None``.

    ``None`` is a supported configuration and never an error: it means this
    deployment has no semantic use case, so no attributes are registered, no
    demands are raised and no model is called. Inventing a default domain here
    would make every deployment a kitchen.
    """
    import os

    source = os.environ if env is None else env
    chosen = str(path or source.get(POLICY_ENV, "")).strip()
    if not chosen:
        return None
    return SemanticPolicy.from_file(chosen)


def _expected(requirement: SemanticRequirement) -> str:
    if requirement.value_type == "enum":
        return "one of [" + ", ".join(json.dumps(v) for v in requirement.values) + "]"
    if requirement.value_type == "bool":
        return "true or false"
    if requirement.value_type == "count":
        return "a whole number"
    if requirement.value_type == "scalar":
        return "a number"
    return "a short string"


def _trigger(name: str) -> TriggerHint:
    try:
        return TriggerHint(name.strip().lower())
    except ValueError as exc:
        raise ConfigurationError(
            f"'{name}' is not a trigger hint; supported: "
            f"{', '.join(t.value for t in TriggerHint)}"
        ) from exc


def _region(value: Any) -> tuple[float, float] | None:
    """Parse ``{"top": 0.0, "height": 0.45}``. Refuses a band outside the box.

    Raises at load rather than clamping silently: a region reaching past the
    subject would crop context the document did not ask for, and a typo there
    should be visible while someone is looking at the file.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConfigurationError("evidence_region must be an object with top and height")
    try:
        top = float(value["top"])
        height = float(value["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"evidence_region needs numeric 'top' and 'height': {exc}"
        ) from exc
    if not 0.0 <= top <= 1.0 or not 0.0 < height <= 1.0 or top + height > 1.0001:
        raise ConfigurationError(
            f"evidence_region (top={top}, height={height}) does not lie inside the "
            f"subject box; both are fractions of it"
        )
    return top, height


def _quality(value: Any) -> dict[str, float] | None:
    """Parse a ``min_quality`` block. Refuses a field the gate does not have.

    Refusing at load rather than ignoring: a misspelled threshold that silently
    did nothing would leave a deployment believing it had tightened a floor it
    had not, and the symptom — a confident answer from bad evidence — looks like
    a model problem rather than a typo.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConfigurationError("min_quality must be an object of threshold names")
    allowed = {
        "min_scale_pixels", "max_truncation", "max_occlusion",
        "max_blur", "max_crowding",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ConfigurationError(
            f"min_quality names {sorted(unknown)}, which the quality gate has no "
            f"threshold for; supported: {sorted(allowed)}"
        )
    return {str(k): float(v) for k, v in value.items()}


#: Largest canonical crop side a document may ask for.
#:
#: Vision tokens scale with area, so an unbounded value in a config file is an
#: unbounded bill and an unbounded latency. 1024 is far above anything measured
#: to help and low enough that a typo — 4480 for 448 — is refused while someone
#: is looking at the file rather than discovered on an invoice.
MAX_OUTPUT_SIZE = 1024


def _output_size(value: Any) -> tuple[int, int] | None:
    """Parse ``448`` or ``{"width": 448, "height": 448}``.

    A bare integer is accepted because square is the overwhelmingly common case
    and ``"output_size": 448`` is what someone writes when they mean it.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigurationError("output_size must be a number or {width, height}")
    if isinstance(value, (int, float)):
        width = height = int(value)
    elif isinstance(value, Mapping):
        try:
            width, height = int(value["width"]), int(value["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"output_size needs integer 'width' and 'height': {exc}"
            ) from exc
    else:
        raise ConfigurationError("output_size must be a number or {width, height}")

    if width <= 0 or height <= 0:
        raise ConfigurationError(f"output_size ({width}x{height}) must be positive")
    if width > MAX_OUTPUT_SIZE or height > MAX_OUTPUT_SIZE:
        raise ConfigurationError(
            f"output_size ({width}x{height}) exceeds the {MAX_OUTPUT_SIZE}px ceiling; "
            f"vision tokens scale with area, so this is a cost limit, not a capability one"
        )
    return width, height


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


__all__ = [
    "POLICY_ENV",
    "SemanticPolicy",
    "SemanticRequirement",
    "load_policy",
]
