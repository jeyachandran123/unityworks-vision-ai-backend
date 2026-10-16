"""``attr.region_labels`` — a specialised head answering from labelled regions.

06_PORTS on the ``attr.*`` adapters: *"They are not VLMs at all, and they prove
the abstraction is at the right altitude."* 11_PERFORMANCE §7's migration — a
VLM discovers an attribute, its evidence trains a head, the head takes over in
production — is a routing change only because a head and a VLM satisfy the same
port. This is that head for models which label **regions** of an image: a
fine-tuned network that reports "this part of the crop shows label X with score
s" for a closed set of labels.

### What it is handed, and why

* ``score_regions`` — the bound model, as a callable from crop pixels to
  ``(label, score, x1, y1, x2, y2)`` tuples in normalised crop coordinates. The
  configuration seam opens the weights and builds it; this package reads no file
  and names no runtime (§M9: *"owns no world state"*).
* ``label_values`` — per attribute, which label means which registered value.
  This is the whole vocabulary, and it is **data**: the deployment's mapping
  file says what a label means, so no domain word appears in code.

### How it answers

For each attribute the request asks about, the strongest label mapped to that
attribute wins, counting only regions scoring at least ``min_score`` whose
centre lies inside the crop's middle — a crop is padded around its subject and
neighbours intrude at the edges. When nothing mapped is seen, the answer is
``unseen_value`` (``not_visible``): the model did not find the part, so nothing
may be claimed about it.

### What it refuses

An unreadable crop, a sequence of crops, or a failing model is a **refusal**,
never ``not_visible`` (U2): "could not be seen" is a claim about the scene, and
pixels nobody read support no claim at all.

### Measured

kitchen-01, 41 human-labelled subjects, 2026-10-16, with the fine-tuned v1
model behind it: head 87.8 % and hands 92.7 % agreement with the human labels,
and no attribute answered as its absent value where the truth was present or
not visible. The VLM it replaced scored 56.1 % and 78.0 % with 12 such answers.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence

from ...core.model.ids import AttributeKey, ModelId, RequestId
from ...core.model.understanding import CostEstimate, ModelMeta, Timing
from ...core.ports.understanding import (
    UnderstanderCapabilities,
    UnderstandingPortRequest,
    UnderstandingPortResponse,
)

#: ``(label, score, x1, y1, x2, y2)``, coordinates normalised to the crop.
Region = tuple[str, float, float, float, float, float]

#: The bound model: ``(pixels, width, height, colour_space) -> regions``.
ScoreRegions = Callable[[memoryview, int, int, str], Sequence[Region]]


class RegionLabelHead:
    """Answers attributes from the labelled regions a local model reports."""

    __slots__ = (
        "_calls",
        "_capabilities",
        "_hash",
        "_id",
        "_labels",
        "_margin",
        "_min_score",
        "_score",
        "_unseen",
        "_version",
    )

    def __init__(
        self,
        *,
        score_regions: ScoreRegions,
        label_values: Mapping[str, Mapping[str, str]],
        producible: Sequence[AttributeKey],
        model_id: str,
        artifact_hash: str,
        model_version: str = "1",
        min_score: float = 0.4,
        centre_margin: float = 0.15,
        unseen_value: str = "not_visible",
        adapter_id: str = "attr.region_labels",
        cost_class: float = 0.05,
    ) -> None:
        if not label_values:
            raise ValueError(
                "a region head needs a label mapping; without one it can answer "
                "nothing and would never be routed to"
            )
        if not 0.0 <= centre_margin < 0.5:
            raise ValueError("centre_margin must be in [0, 0.5)")
        answerable = tuple(key for key in producible if str(key) in label_values)
        if not answerable:
            raise ValueError(
                "none of the producible attributes has a label mapping; the head "
                f"maps {sorted(label_values)}"
            )
        self._score = score_regions
        self._labels = {str(k): dict(v) for k, v in label_values.items()}
        self._min_score = float(min_score)
        self._margin = float(centre_margin)
        self._unseen = unseen_value
        self._id = adapter_id
        self._hash = artifact_hash
        self._version = model_version
        self._calls = 0
        self._capabilities = UnderstanderCapabilities(
            producible_attributes=answerable,
            model_id=ModelId(model_id),
            max_crops_per_request=1,
            supports_structured_output=True,
            supports_temporal=False,
            supports_batching=False,
            max_batch_size=1,
            cost_class=cost_class,
            deterministic=True,
            data_residency="local",
        )

    # --- port surface --------------------------------------------------------- #

    @property
    def adapter_id(self) -> str:
        return self._id

    @property
    def calls(self) -> int:
        return self._calls

    def capabilities(self) -> UnderstanderCapabilities:
        return self._capabilities

    def understand(self, request: UnderstandingPortRequest) -> UnderstandingPortResponse:
        self._calls += 1
        if len(request.crops) != 1:
            return self._refusal(
                f"one crop per request; got {len(request.crops)} — this head reads a "
                "single image and would describe one instant as a span"
            )
        crop = request.crops[0]
        if crop.width <= 0 or crop.height <= 0 or len(crop.pixels) < crop.width * crop.height * 3:
            return self._refusal("the crop's pixels cannot be read")

        asked = [str(key) for key in request.output_schema.fields if str(key) in self._labels]
        if not asked:
            # Asked only about attributes it has no labels for. Silence, not a
            # guess (U1).
            return UnderstandingPortResponse(model_meta=self._meta(), timing=Timing())

        try:
            regions = list(self._score(crop.pixels, crop.width, crop.height, crop.colour_space))
        except Exception as exc:  # noqa: BLE001 - a failed model is a refusal (U2)
            return self._refusal(f"the model failed: {type(exc).__name__}: {exc}")

        kept = [region for region in regions if self._counts(region)]
        structured: dict[str, str] = {}
        confidence: dict[str, float] = {}
        for key in asked:
            values = self._labels[key]
            best: tuple[str, float] | None = None
            for label, score, *_ in kept:
                if label in values and (best is None or score > best[1]):
                    best = (label, score)
            if best is None:
                structured[key] = self._unseen
            else:
                structured[key] = values[best[0]]
                confidence[key] = best[1]

        raw = {
            "regions": [[label, round(score, 4), *(round(v, 4) for v in box)] for label, score, *box in kept],
            "answers": structured,
        }
        return UnderstandingPortResponse(
            structured=structured,
            field_confidence=confidence or None,
            raw_output=json.dumps(raw, sort_keys=True).encode(),
            model_meta=self._meta(),
            timing=Timing(batch_size=1),
        )

    def understand_batch(
        self, requests: Sequence[UnderstandingPortRequest]
    ) -> Mapping[RequestId, UnderstandingPortResponse]:
        return {request.request_id: self.understand(request) for request in requests}

    def estimate_cost(self, request: UnderstandingPortRequest) -> CostEstimate:
        covered = tuple(key for key in request.output_schema.fields if str(key) in self._labels)
        return CostEstimate(
            cost_units=self._capabilities.cost_class,
            model_id=self._capabilities.model_id,
            attributes_covered=covered,
            attributes_uncovered=tuple(
                key for key in request.output_schema.fields if str(key) not in self._labels
            ),
        )

    # --- internals ------------------------------------------------------------- #

    def _counts(self, region: Region) -> bool:
        """Strong enough, and centred inside the crop's middle."""
        _, score, x1, y1, x2, y2 = region
        if score < self._min_score:
            return False
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        low, high = self._margin, 1.0 - self._margin
        return low <= cx <= high and low <= cy <= high

    def _refusal(self, reason: str) -> UnderstandingPortResponse:
        return UnderstandingPortResponse(
            refused=True, refusal_reason=reason, model_meta=self._meta(), timing=Timing()
        )

    def _meta(self) -> ModelMeta:
        return ModelMeta(
            model_id=self._capabilities.model_id,
            model_version=self._version,
            artifact_hash=self._hash,
            adapter_id=self._id,
            deterministic=True,
        )


__all__ = ["Region", "RegionLabelHead", "ScoreRegions"]
