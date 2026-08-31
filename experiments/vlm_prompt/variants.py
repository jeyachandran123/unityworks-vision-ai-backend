"""The four prompt/evidence formulations under test, and how each is read back.

**One variable.** Every variant sends byte-identical crop pixels through the
production adapter at the production temperature and `max_side`. What differs is
the instruction text and — for D — the inference-time metadata prepended to it.

**Nothing here is production.** No variant is written into `config/policies/`,
and the shipped prompt is read from the policy document rather than restated, so
Variant A cannot drift away from the thing it is the control for.

### Reading the answers back

Variant A speaks the production domain: `none | hairnet | cap | hood | other |
not_visible`. B, C and D speak a decomposed schema. Both are collapsed onto the
same three states the ground truth uses, by one mapping applied to every variant:

```
PRESENT      a covering was seen
ABSENT       the region was seen and carried no covering
NOT_VISIBLE  the region could not be assessed
None         nothing parseable  ->  UNKNOWN downstream, never a value
```

The collapse is deliberate and lossy in one direction only: it discards *which*
covering was seen. The shipped rule tests `head_covering != none`, so that
distinction changes no verdict, and keeping it would make A and B/C/D
incomparable for no gain.

**Uncertainty is never optimism.** `UNCERTAIN` in any position maps to
`NOT_VISIBLE`, which the rule already treats as UNKNOWN. A variant cannot buy a
better score by hedging: hedging costs it recall on the observable class.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

#: The three states ground truth uses, and the only ones scored.
PRESENT = "present"
ABSENT = "absent"
NOT_VISIBLE = "not_visible"

#: Values the production domain treats as a covering.
_COVERINGS = {"hairnet", "cap", "hood", "other"}


@dataclass(frozen=True, slots=True)
class Variant:
    """One formulation, its prompt, its token budget and its reader."""

    id: str
    title: str
    hypothesis: str
    prompt: str
    parse: Callable[[str, dict[str, Any]], tuple[str | None, dict[str, Any]]]
    max_output_tokens: int = 128
    #: Whether the prompt is assembled with per-subject metadata (Variant D).
    templated: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #


def _load(text: str) -> dict[str, Any] | None:
    """Parse, or return ``None``. Never optimistic.

    Uses the platform's own recovery so a brace-less body is read exactly as
    production reads one — a real behaviour of this model, and excluding it would
    make the experiment measure a parser rather than a prompt.
    """
    from vision_os.adapters.understanding.payload import extract_json

    decoded = extract_json(text)
    return decoded if isinstance(decoded, dict) else None


def parse_production(text: str, _meta: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Variant A. The shipped domain, collapsed onto the scored states."""
    decoded = _load(text)
    if decoded is None:
        return None, {"reason": "unparseable"}
    raw = decoded.get("head_covering")
    if not isinstance(raw, str):
        return None, {"reason": "key absent"}
    value = raw.strip().lower()
    if value == "not_visible":
        return NOT_VISIBLE, {"raw": value}
    if value == "none":
        return ABSENT, {"raw": value}
    if value in _COVERINGS:
        return PRESENT, {"raw": value}
    return None, {"reason": "out of domain", "raw": value}


def parse_decomposed(text: str, _meta: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Variants B, C, D. Observability first, attribute only if observable.

    The asymmetry is the point: a decided attribute on a non-visible region is
    **refused here**, not honoured. A model that says "NOT_VISIBLE, and also the
    head is bare" has contradicted itself, and the safe reading of a
    contradiction is that nothing was established.
    """
    decoded = _load(text)
    if decoded is None:
        return None, {"reason": "unparseable"}

    observability = str(decoded.get("observability", "")).strip().upper()
    attribute = str(decoded.get("attribute", "")).strip().upper()
    detail = {
        "observability": observability,
        "attribute": attribute,
        "evidence_reason": str(decoded.get("evidence_reason", ""))[:160],
    }

    if observability not in {"VISIBLE", "NOT_VISIBLE", "UNCERTAIN"}:
        return None, {**detail, "reason": "observability out of domain"}
    if observability != "VISIBLE":
        return NOT_VISIBLE, detail

    if attribute == "PRESENT":
        return PRESENT, detail
    if attribute == "ABSENT":
        return ABSENT, detail
    if attribute in {"UNCERTAIN", "NOT_EVALUATED", ""}:
        # Visible but undecided. Not a covering and not an absence — the honest
        # reading is that nothing was established, which the rule treats as
        # UNKNOWN. Scored against a `present` truth this costs recall, which is
        # the correct price for hedging.
        return NOT_VISIBLE, {**detail, "reason": "visible but undecided"}
    return None, {**detail, "reason": "attribute out of domain"}


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_SCHEMA_BLOCK = """Respond with JSON containing exactly these keys:
  "observability": one of "VISIBLE", "NOT_VISIBLE", "UNCERTAIN"
  "attribute": one of "PRESENT", "ABSENT", "UNCERTAIN", "NOT_EVALUATED"
  "evidence_reason": a short phrase naming what you actually saw
Answer with the JSON object only. No prose before or after it."""

VARIANT_B_PROMPT = f"""You are inspecting one cropped CCTV image of a person. Answer in two \
separate steps, and do not merge them.

STEP 1 - OBSERVABILITY. Look at the TOP OF THE HEAD only. Decide whether that \
region is sufficiently visible in THIS image to be judged.
  "VISIBLE" - you can see the top of the head clearly enough to say what is on it.
  "NOT_VISIBLE" - the top of the head is out of frame, cut off by the crop edge, \
turned fully away, hidden behind the body or an object, in shadow, blurred, or \
too small to judge.
  "UNCERTAIN" - you genuinely cannot decide which of the two above applies.

STEP 2 - ATTRIBUTE. Do this ONLY if step 1 was "VISIBLE". If step 1 was \
"NOT_VISIBLE" or "UNCERTAIN", answer "NOT_EVALUATED" and stop.
  "PRESENT" - a net, cap, hood, scarf or other fabric covers any part of the top \
of the head. Answer PRESENT even if loose hair is also visible below or around \
it; that is the normal appearance of a head covering, not an absence.
  "ABSENT" - you can see bare hair or scalp on top of the head with nothing on it.
  "UNCERTAIN" - the region is visible but you cannot tell.

A region you could not see is NEVER "ABSENT". Not seeing a covering and seeing \
that there is no covering are different findings, and only the second is ABSENT.

Do not infer from the uniform, the room, the person's apparent role, or what \
someone in this setting would normally wear.

{_SCHEMA_BLOCK}"""

VARIANT_C_PROMPT = f"""You are a visual evidence reporter examining one cropped CCTV \
image. Work through the image in this order and report only what the pixels support.

1. INSPECT. Look at the image you were given. Do not imagine any part of the \
scene that was not supplied.
2. LOCATE. Find the top of the head of the person in this crop. If you cannot \
find it, say so - that is a complete and useful answer.
3. OBSERVABILITY. State whether that region is visible enough to judge.
4. EVIDENCE. If and only if it is visible, state what you can actually see on it.
5. REPORT. Produce the structured result below.

Three rules govern this task:

  ABSENCE OF EVIDENCE IS NOT EVIDENCE OF ABSENCE. If the top of the head is not \
in this crop, the correct answer is that it was not visible - never that it was \
uncovered.

  UNSEEN AREAS MUST NOT BE INFERRED. A visible shoulder, uniform, hairline or \
piece of equipment tells you nothing about the top of the head. Neither does the \
kind of room this is.

  DO NOT GUESS. You are being graded on refusing to answer questions this image \
cannot answer, not on filling every field. An honest refusal scores full marks.

A covering is any net, cap, hood, scarf or fabric on the top of the head. Loose \
hair visible below or around a covering is normal and does not make the head \
uncovered.

{_SCHEMA_BLOCK}"""

#: Variant D prepends genuine inference-time metadata to Variant C's body.
#:
#: Every field below is available at inference time from the crop request. None
#: of it is derived from ground truth, from the annotation file, or from any
#: model's previous answer. `observability_signal` is the pose producer's own
#: verdict for this subject — a real signal the platform already computes at
#: this point in the pipeline (P33), and the thing this variant tests the value
#: of surfacing to the model.
VARIANT_D_HEADER = """EVIDENCE CONTEXT
  attribute_under_evaluation: head_covering
  body_region: top of the head
  evidence_region: upper {region_pct}% of the detected person box
  crop_resolution: {width}x{height}
  evidence_source: fixed CCTV camera, kitchen, single frame
  subject_reference: {subject_ref}
  upstream_observability_signal: {observability_signal}

The upstream signal is a geometric estimate from a separate pose model. It is \
advisory and is sometimes wrong. Use it as a prior, not as an answer: if the \
image plainly disagrees with it, report what you can see.

"""

VARIANT_D_PROMPT = VARIANT_D_HEADER + VARIANT_C_PROMPT


def production_prompt() -> str:
    """Variant A's text, read from the shipped policy document.

    Read rather than restated so the control cannot silently drift away from
    production. If the policy changes, this experiment's control changes with it
    and the run manifest records the new content hash.
    """
    from pathlib import Path

    from vision_os.adapters.configuration.semantic_policy import SemanticPolicy

    root = Path(__file__).resolve().parents[2]
    document = json.loads(
        (root / "config" / "policies" / "kitchen-safety.example.json").read_text(
            encoding="utf-8"
        )
    )
    return SemanticPolicy.from_document(document).render_prompt()


def all_variants() -> dict[str, Variant]:
    return {
        "A": Variant(
            id="A",
            title="Production baseline (control)",
            hypothesis="The shipped prompt, unchanged. Everything else is measured against this.",
            prompt=production_prompt(),
            parse=parse_production,
            max_output_tokens=128,
            notes=(
                "Read from config/policies/kitchen-safety.example.json v2.1.0.",
                "128 output tokens, as production declares.",
            ),
        ),
        "B": Variant(
            id="B",
            title="Observability + attribute decomposition",
            hypothesis=(
                "The model conflates 'I cannot see the head' with 'the head is bare'. "
                "Forcing observability to be decided FIRST, as a separate answer, "
                "should let it refuse without having to also produce a covering."
            ),
            prompt=VARIANT_B_PROMPT,
            parse=parse_decomposed,
            max_output_tokens=256,
            notes=("Two explicit steps. Step 2 is gated on step 1.",),
        ),
        "C": Variant(
            id="C",
            title="Evidence-first structured",
            hypothesis=(
                "Naming the reasoning order and stating the inference rules explicitly "
                "('absence of evidence is not evidence of absence') should suppress "
                "inference from context that the decomposition alone does not."
            ),
            prompt=VARIANT_C_PROMPT,
            parse=parse_decomposed,
            max_output_tokens=256,
            notes=("Same output schema as B. Differs only in instruction framing.",),
        ),
        "D": Variant(
            id="D",
            title="Evidence-first + structured inference-time context",
            hypothesis=(
                "Supplying the metadata the platform already holds at inference time — "
                "including the pose observability signal — should improve agreement "
                "further, or reveal that the model over-trusts a supplied prior."
            ),
            prompt=VARIANT_D_PROMPT,
            parse=parse_decomposed,
            max_output_tokens=256,
            templated=True,
            notes=(
                "C's body with a metadata header. The ONLY difference from C.",
                "No ground truth, no expected answer, no annotation text.",
                "The pose signal is real and available at this point in the pipeline.",
            ),
        ),
    }
