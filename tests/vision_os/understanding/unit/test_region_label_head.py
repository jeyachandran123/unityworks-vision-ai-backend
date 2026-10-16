"""A specialised head that answers attributes from labelled regions in a crop.

06_PORTS calls the ``attr.*`` heads *"the point of this port's design"*: a small
local model that produces a registered attribute, satisfying the same port as a
VLM so that switching between them is a routing change. This head is handed a
``score_regions`` callable — the model is bound at the configuration seam, so
this package reads no file and names no model — plus a mapping from the
model's labels to attribute values. Nothing here knows what the labels mean.

Measured against kitchen-01's 41 human-labelled subjects (2026-10-16) with the
fine-tuned v1 model behind it: head 87.8 % and hands 92.7 % agreement, and no
attribute answered as its absent value where the truth was present or not
visible. The VLM it replaces scored 56.1 % and 78.0 %, with 12 such answers.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.understanding.region_head import RegionLabelHead
from vision_os.conformance import UNDERSTANDER_KIT
from vision_os.core.model.ids import AttributeKey, CropId, PromptId, RequestId
from vision_os.core.ports.understanding import (
    CropView,
    OutputSchema,
    RenderedPrompt,
    UnderstandingPortRequest,
)

HEAD = AttributeKey("head_part")
HANDS = AttributeKey("hand_part")
OTHER = AttributeKey("unrelated")

LABELS = {
    "head_part": {"cover_on": "covered", "cover_off": "bare"},
    "hand_part": {"glove_on": "covered", "glove_off": "bare"},
}


def head(regions=(), **kwargs) -> RegionLabelHead:
    """`regions`: what the stand-in model reports, as (label, score, x1, y1, x2, y2)."""
    defaults = dict(
        score_regions=lambda pixels, width, height, colour_space: list(regions),
        label_values=LABELS,
        producible=(HEAD, HANDS, OTHER),
        model_id="stand-in",
        artifact_hash="blake2b:test",
    )
    defaults.update(kwargs)
    return RegionLabelHead(**defaults)


def request(
    *fields: AttributeKey, crops: int = 1, pixels: bytes | None = None
) -> UnderstandingPortRequest:
    prompt = RenderedPrompt(
        prompt_id=PromptId("p"), version="1", text="-", output_schema=OutputSchema(fields=fields)
    )
    return UnderstandingPortRequest(
        request_id=RequestId("r"),
        crops=tuple(
            CropView(
                crop_id=CropId(f"c{i}"),
                pixels=memoryview(bytes(16 * 16 * 3) if pixels is None else pixels),
                width=16,
                height=16,
                colour_space="rgb24",
            )
            for i in range(crops)
        ),
        prompt=prompt,
        output_schema=prompt.output_schema,
    )


CENTRE = (0.4, 0.4, 0.6, 0.6)


def test_a_label_in_the_middle_answers_its_attribute() -> None:
    response = head([("cover_off", 0.8, *CENTRE)]).understand(request(HEAD))
    assert dict(response.structured) == {"head_part": "bare"}
    assert response.field_confidence == {"head_part": pytest.approx(0.8)}
    assert response.raw_output


def test_the_stronger_of_two_contradicting_labels_wins() -> None:
    regions = [("cover_off", 0.5, *CENTRE), ("cover_on", 0.7, *CENTRE)]
    assert head(regions).understand(request(HEAD)).structured == {"head_part": "covered"}


def test_nothing_seen_is_not_visible_never_an_absence() -> None:
    """The model found no covering and no bare head: it could not see one."""
    assert head([]).understand(request(HEAD)).structured == {"head_part": "not_visible"}


def test_a_label_near_the_edge_belongs_to_a_neighbour() -> None:
    """Crops are padded around the subject, and neighbours intrude there."""
    edge = ("cover_off", 0.9, 0.0, 0.0, 0.1, 0.1)
    assert head([edge]).understand(request(HEAD)).structured == {"head_part": "not_visible"}


def test_a_weak_label_is_ignored() -> None:
    weak = ("cover_off", 0.2, *CENTRE)
    answer = head([weak], min_score=0.4).understand(request(HEAD)).structured
    assert answer == {"head_part": "not_visible"}


def test_only_the_asked_attributes_are_answered() -> None:
    """U1. A head crop is not asked about hands, so it says nothing of them."""
    regions = [("cover_on", 0.8, *CENTRE), ("glove_off", 0.8, *CENTRE)]
    assert head(regions).understand(request(HEAD)).structured == {"head_part": "covered"}


def test_an_attribute_it_has_no_labels_for_is_left_unanswered() -> None:
    assert head([("cover_on", 0.8, *CENTRE)]).understand(request(OTHER)).structured == {}


def test_it_declares_only_what_its_labels_can_answer() -> None:
    produced = {str(k) for k in head().capabilities().producible_attributes}
    assert produced == {"head_part", "hand_part"}


def test_an_unreadable_crop_is_refused_not_answered() -> None:
    """U2: 'not_visible' about pixels nobody could read would be fabricated."""
    response = head([("cover_on", 0.9, *CENTRE)]).understand(request(HEAD, pixels=b""))
    assert response.refused and not response.structured


def test_a_sequence_of_crops_is_refused() -> None:
    response = head([("cover_on", 0.9, *CENTRE)]).understand(request(HEAD, crops=3))
    assert response.refused and not response.structured


def test_a_failing_model_is_a_refusal() -> None:
    def broken(*_):
        raise RuntimeError("session gone")

    response = head(score_regions=broken).understand(request(HEAD))
    assert response.refused and not response.structured


def test_labels_must_be_configured() -> None:
    with pytest.raises(ValueError):
        head(label_values={})


def test_it_passes_the_platform_understander_kit() -> None:
    report = UNDERSTANDER_KIT.run(head([("cover_on", 0.9, *CENTRE)]))
    assert report.passed, report.failures
