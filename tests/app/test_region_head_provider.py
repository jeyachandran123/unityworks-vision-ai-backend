"""The 'region_head' understander: v1's regions answer the PPE attributes.

v1 is a YOLO26n fine-tuned on kitchen CCTV to label hairnet / no_hairnet,
glove / no_glove, face_mask / no_face_mask, apron / no_apron and person. Run on
the pipeline's own evidence crops and scored against kitchen-01's 41 human-
labelled subjects (2026-10-16):

    head_covering   v1 87.8 %   VLM 56.1 %
    hand_covering   v1 92.7 %   VLM 78.0 %
    head answered "none" where the truth was present or not visible:
                    v1 0        VLM 12

About 0.3 s per crop on this CPU through the platform's ONNX session, against
5-11 s and a paid call per VLM request.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision_os.adapters.configuration.understander_providers import (
    REGION_HEAD_LABELS_ENV,
    REGION_HEAD_WEIGHTS_ENV,
    ProviderConfigurationError,
    build_understander,
)
from vision_os.core.model.ids import AttributeKey, CropId, PromptId, RequestId
from vision_os.core.ports.understanding import (
    CropView,
    OutputSchema,
    RenderedPrompt,
    UnderstandingPortRequest,
)

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "models" / "v1.onnx"
LABELS = REPO / "config" / "understanding" / "region-head.kitchen.json"
KEYS = tuple(
    AttributeKey(k)
    for k in ("head_covering", "face_covering", "hand_covering", "apron", "visible_object_kind")
)
requires_weights = pytest.mark.skipif(not WEIGHTS.is_file(), reason="v1.onnx absent")

ENV = {REGION_HEAD_WEIGHTS_ENV: str(WEIGHTS), REGION_HEAD_LABELS_ENV: str(LABELS)}


def build(env=ENV):
    adapter, _ = build_understander(producible=KEYS, provider="region_head", env=env)
    return adapter


def ask(adapter, image: Image.Image, *fields: str):
    rgb = image.convert("RGB")
    prompt = RenderedPrompt(
        prompt_id=PromptId("p"),
        version="1",
        text="-",
        output_schema=OutputSchema(fields=tuple(AttributeKey(f) for f in fields)),
    )
    return adapter.understand(
        UnderstandingPortRequest(
            request_id=RequestId("r"),
            crops=(
                CropView(
                    crop_id=CropId("c"),
                    pixels=memoryview(np.asarray(rgb).tobytes()),
                    width=rgb.width,
                    height=rgb.height,
                    colour_space="rgb24",
                ),
            ),
            prompt=prompt,
            output_schema=prompt.output_schema,
        )
    )


@requires_weights
def test_it_binds_and_declares_the_mapped_attributes() -> None:
    adapter = build()
    assert adapter.adapter_id == "attr.region_labels"
    produced = {str(k) for k in adapter.capabilities().producible_attributes}
    # visible_object_kind has no labels in this model, so it is not claimed.
    assert produced == {"head_covering", "face_covering", "hand_covering", "apron"}
    assert adapter.capabilities().data_residency == "local"


def pipeline_crop(
    frame: Image.Image, box: tuple[float, float, float, float], attribute: str
) -> Image.Image:
    """The evidence crop the production strategy cuts for `attribute`: the
    policy's band, padded, fitted onto the policy's canvas — the image the
    understander is actually handed. A hand-cut band is a different picture."""
    from vision_os.adapters.configuration.semantic_policy import SemanticPolicy
    from vision_os.adapters.cropping import PartFocusedCropStrategy
    from vision_os.core.model.ids import ClassId
    from vision_os.core.model.space import Box

    policy = SemanticPolicy.from_file(REPO / "config" / "policies" / "kitchen-safety.example.json")
    strategy = PartFocusedCropStrategy(
        regions=policy.evidence_regions, output_sizes=dict(policy.output_sizes)
    )
    w, h = frame.size
    plan = strategy.plan(
        box=Box(*box),
        class_id=ClassId("person"),
        source_width=w,
        source_height=h,
        attributes=(AttributeKey(attribute),),
    )
    b = plan.padded_box
    cut = frame.convert("RGB").crop((int(b.x1 * w), int(b.y1 * h), int(b.x2 * w), int(b.y2 * h)))
    canvas = Image.new("RGB", (plan.output_width, plan.output_height), (0, 0, 0))
    cut.thumbnail((plan.output_width, plan.output_height), Image.LANCZOS)
    canvas.paste(
        cut, ((plan.output_width - cut.width) // 2, (plan.output_height - cut.height) // 2)
    )
    return canvas


@requires_weights
def test_a_hairnet_is_seen_on_a_real_kitchen_crop() -> None:
    """kitchen-01 f00060 s0, annotated 'blue hairnet', in the head crop the
    pipeline would hand over."""
    frame = Image.open(REPO / "datasets" / "kitchen-01" / "frames" / "f00060.jpg")
    crop = pipeline_crop(frame, (0.3607, 0.1619, 0.5332, 0.8272), "head_covering")
    response = ask(build(), crop, "head_covering")
    assert not response.refused
    assert response.structured == {"head_covering": "hairnet"}


@requires_weights
def test_an_input_size_contradicting_the_graph_is_refused() -> None:
    """A 448 crop in a 640 graph is enlarged by nearest neighbour: 73.2 %
    against 87.8 % on kitchen-01's heads. Silent, so refused at composition."""
    from vision_os.adapters.configuration.understander_providers import REGION_HEAD_INPUT_ENV

    with pytest.raises(ProviderConfigurationError, match="exported for 448"):
        build(env={**ENV, REGION_HEAD_INPUT_ENV: "640"})


@requires_weights
def test_an_empty_wall_is_not_visible_not_bare() -> None:
    blank = Image.new("RGB", (448, 448), (180, 180, 180))
    response = ask(build(), blank, "head_covering", "hand_covering")
    assert response.structured == {"head_covering": "not_visible", "hand_covering": "not_visible"}


@requires_weights
def test_settings_carry_the_region_head_from_file_to_the_factory() -> None:
    """`.env` reaches the settings object, never `os.environ`; without this
    bridge the provider would be named and then refuse to bind for want of the
    weights path it was given."""
    from app.configuration.settings import Settings

    options = Settings(
        VISION_REGION_HEAD_WEIGHTS=str(WEIGHTS), VISION_REGION_HEAD_LABELS=str(LABELS)
    ).understander_options()
    adapter, _ = build_understander(
        producible=KEYS, provider="region_head", env={}, defaults=options
    )
    assert adapter.adapter_id == "attr.region_labels"


def test_both_settings_are_required() -> None:
    with pytest.raises(ProviderConfigurationError):
        build(env={REGION_HEAD_WEIGHTS_ENV: str(WEIGHTS)})


def test_missing_weights_fail_at_composition(tmp_path) -> None:
    with pytest.raises(ProviderConfigurationError, match="weights not found"):
        build(env={**ENV, REGION_HEAD_WEIGHTS_ENV: str(tmp_path / "absent.onnx")})


@requires_weights
def test_a_label_the_model_cannot_report_is_refused(tmp_path) -> None:
    """A misspelt label would never match, and every answer would read as
    not visible — a silent outage. Refused while someone is watching the log."""
    bad = tmp_path / "labels.json"
    bad.write_text('{"attributes": {"head_covering": {"hairnett": "hairnet"}}}')
    with pytest.raises(ProviderConfigurationError, match="hairnett"):
        build(env={**ENV, REGION_HEAD_LABELS_ENV: str(bad)})


@requires_weights
def test_the_shipped_floors_hold_the_face_labels_higher(tmp_path) -> None:
    """0.3 overall, 0.2 for the face and apron labels (the demo setting):
    measured on kitchen-01 for head and hands, checked by eye on live frames
    for faces (see the mapping file). A floor for a label the model cannot report is refused like a
    misspelt mapping, and a floor outside (0, 1) is refused outright."""
    import json

    shipped = json.loads(LABELS.read_text(encoding="utf-8"))
    assert shipped["min_score"] == 0.3
    assert shipped["label_min_scores"] == {
        "face_mask": 0.4,
        "no_face_mask": 0.2,
        "apron": 0.4,
        "no_apron": 0.2,
    }

    bad = tmp_path / "labels.json"
    bad.write_text(json.dumps({**shipped, "label_min_scores": {"no_face_maks": 0.4}}))
    with pytest.raises(ProviderConfigurationError, match="no_face_maks"):
        build(env={**ENV, REGION_HEAD_LABELS_ENV: str(bad)})

    bad.write_text(json.dumps({**shipped, "min_score": 4}))
    with pytest.raises(ProviderConfigurationError, match="between 0 and 1"):
        build(env={**ENV, REGION_HEAD_LABELS_ENV: str(bad)})
