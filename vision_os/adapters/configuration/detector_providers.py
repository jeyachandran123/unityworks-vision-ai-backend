"""Which detector a deployment binds — resolved here, like the understander.

The sibling of ``understander_providers``, and for the same reasons: a closed
factory table keyed by configuration, environment read at this seam rather than
inside an adapter, and the platform never told the name. ``build_detection_layer``
takes a ``detector_factory`` and calls it with a declaration; what that factory
returns is a composition decision, not a platform one.

### Why the class list is not in this file

``open_onnx_detector_session`` reads the names out of the graph's own metadata,
and the ``TaxonomyMapping`` built here is derived from that list. A COCO constant
written into this repository would silently mislabel every object the day
somebody binds a model trained on something else — and a mislabelled detection is
worse than a missing one, because it is wrong with full confidence and it
propagates into tracking, cropping, prompts and observations before anyone
notices.

So the model declares what it can name, this module maps those names onto
platform classes, and a deployment may narrow the result but never invent it.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...core.model.ids import AdapterId, ClassId, ModelId
from ...core.model.taxonomy import MappingEntry, TaxonomyMapping, UnmappedPolicy
from ...core.ports.cropping import LabelSpaceView

#: Selector. ``yolo`` is the default because a scripted detector must never be
#: what a demo or a deployment runs by accident: it answers with the same box on
#: every frame regardless of the pixels, and everything downstream — tracking,
#: crops, prompts, attributes, observations — is then real machinery operating on
#: a fiction.
PROVIDER_ENV = "VISION_DETECTOR_PROVIDER"
WEIGHTS_ENV = "VISION_DETECTOR_WEIGHTS"
CLASSES_ENV = "VISION_DETECTOR_CLASSES"
CONFIDENCE_ENV = "VISION_DETECTOR_CONFIDENCE"
IOU_ENV = "VISION_DETECTOR_IOU"

DEFAULT_PROVIDER = "yolo"
DEFAULT_WEIGHTS = "models/yolov8n.onnx"

#: How a detector's label space behaves when it meets something outside it.
#:
#: This is the fact nothing in the platform used to state, and its absence is
#: what let a pen be reported as a toothbrush with no qualification anywhere.
#:
#: A **closed-set** detector scores a fixed vocabulary and returns the argmax. It
#: has no way to answer "none of these": presented with an object it was never
#: trained on, it must still distribute probability across the words it knows,
#: and the winner is the nearest neighbour in that space rather than an
#: identification. That is not a defect to be tuned away with a threshold — a
#: pen scores 0.454 as `toothbrush`, comfortably above any threshold that still
#: admits real detections.
#:
#: An **open-vocabulary** detector is scored against labels supplied at query
#: time and can be asked whether a specific concept is present, so absence from
#: its answer carries information a closed-set absence does not.
#:
#: Declaring which kind is bound lets consumers qualify a class claim instead of
#: presenting it as an identity. It is a capability statement, in the same family
#: as `producible_classes`, and it obeys the same rule: declared honestly, by the
#: adapter that knows.
CLOSED_SET = "closed_set"
OPEN_VOCABULARY = "open_vocabulary"


class DetectorConfigurationError(ValueError):
    """A detector was named that cannot be built. Raised at composition time."""


def resolve_detector_provider(env: Mapping[str, str] | None = None) -> str:
    source = os.environ if env is None else env
    return ((source.get(PROVIDER_ENV) or "").strip().lower()) or DEFAULT_PROVIDER


def default_weights_path() -> Path:
    """``backend/models/yolov8n.onnx``, relative to this package."""
    return Path(__file__).resolve().parents[4] / DEFAULT_WEIGHTS


def _setting(env: Mapping[str, str], name: str, default: str = "") -> str:
    return (env.get(name) or "").strip() or default


class BoundDetector:
    """What a composition root needs to wire a detector into the platform.

    The detector alone is not enough: the config document must declare the same
    class list and the same artifact hash, and the Crop Manager and the
    Observation API must be told which classes exist. Returning them together
    stops those four places drifting apart — which they did, silently, when the
    taxonomy said ``person`` and the detector could produce eighty things.
    """

    __slots__ = ("adapter_id", "classes", "detector", "mappings", "model_id",
                 "model_version", "artifact_hash", "artifact_path", "note",
                 "label_space_kind", "native_labels", "native_label_space")

    def __init__(
        self,
        *,
        detector: Any,
        adapter_id: str,
        classes: tuple[ClassId, ...],
        mappings: tuple[MappingEntry, ...],
        model_id: str,
        model_version: str,
        artifact_hash: str,
        artifact_path: str,
        note: str,
        label_space_kind: str,
        native_labels: tuple[str, ...],
        native_label_space: str,
    ) -> None:
        self.detector = detector
        # Carried rather than read off the detector: `DetectorPort` declares
        # `capabilities`, `detect` and `warm` and nothing else. `adapter_id` is
        # an understander concept, and reaching for one here would work by
        # accident on adapters that happen to have it.
        self.adapter_id = adapter_id
        self.classes = classes
        self.mappings = mappings
        self.model_id = model_id
        self.model_version = model_version
        self.artifact_hash = artifact_hash
        self.artifact_path = artifact_path
        self.note = note
        # Whether this detector can decline to name something, and the exact
        # vocabulary it is choosing from. Both travel to consumers so a class
        # claim can be qualified rather than read as an identification.
        self.label_space_kind = label_space_kind
        self.native_labels = native_labels
        self.native_label_space = native_label_space

    @property
    def is_closed_set(self) -> bool:
        return self.label_space_kind == CLOSED_SET

    def capability_gaps(self) -> tuple[tuple[str, str], ...]:
        """Capability facts for `CapabilitySummary.gaps`.

        `gaps` is the platform's existing typed channel for "here is something I
        cannot do", and it already reaches consumers through the Observation API.
        A closed-set label space belongs there exactly: the detector cannot
        report an object outside its vocabulary, and a consumer that does not
        know this will read every class name as an identification.
        """
        if not self.is_closed_set:
            return ()
        return (
            (
                "detector.label_space",
                f"{CLOSED_SET}:{self.native_label_space or 'custom'}:"
                f"{len(self.native_labels)}",
            ),
            (
                "detector.vocabulary",
                ",".join(self.native_labels),
            ),
        )

    def declaration(self, *, detector_id: str, artifact_uri: str) -> dict[str, Any]:
        """The config-document entry describing this binding.

        Written before ``build_platform`` reads configuration, because the
        platform reads it once at construction: mutating the document afterwards
        leaves a running platform bound to a detector list it never saw.
        """
        return {
            "detector_id": detector_id,
            "adapter_id": self.adapter_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "artifact_uri": artifact_uri,
            "artifact_hash": self.artifact_hash,
            "role": "primary_detector",
            "mappings": [
                {"native_label": entry.native_label, "class_id": str(entry.class_id)}
                for entry in self.mappings
            ],
        }


def label_space_view(bound: BoundDetector) -> LabelSpaceView:
    """The bound detector's capability, in the form M8's policies consume.

    The same fact ``capability_gaps()`` publishes to consumers, handed to the
    Crop Manager so a trigger policy can *act* on it instead of a consumer merely
    reading about it afterwards. Derived here, at the only place that holds both
    the detector and its declaration, so the two cannot drift.

    ``bound.classes`` rather than ``bound.native_labels``: a policy compares this
    against a candidate's platform ``class_id``, and a native label must never
    escape its adapter (port obligation D2).
    """
    return LabelSpaceView(
        kind=bound.label_space_kind,
        producible_classes=frozenset(bound.classes),
    )


def _build_yolo(*, clock, env: Mapping[str, str], **_: Any) -> BoundDetector:
    from ..detection.yolo import YoloDetector
    from ..models.runtimes import open_onnx_detector_session

    path = Path(_setting(env, WEIGHTS_ENV) or default_weights_path())
    if not path.is_file():
        raise DetectorConfigurationError(
            f"detector weights not found at '{path}'. Set {WEIGHTS_ENV} to an ONNX "
            f"detection graph. Binding a scripted detector instead would answer "
            f"with the same box on every frame, and nothing downstream could tell."
        )

    session = open_onnx_detector_session(
        str(path),
        conf_threshold=float(_setting(env, CONFIDENCE_ENV, "0.25")),
        iou_threshold=float(_setting(env, IOU_ENV, "0.45")),
    )

    # The model's own names, optionally narrowed. `UnmappedPolicy.DROP` then
    # discards anything outside the selection at the adapter boundary, so a
    # narrowed deployment never sees a class its taxonomy does not define.
    declared = list(session.class_names())
    wanted = {
        name.strip().lower()
        for name in _setting(env, CLASSES_ENV).split(",")
        if name.strip()
    }
    selected = [name for name in declared if not wanted or name.lower() in wanted]
    if not selected:
        raise DetectorConfigurationError(
            f"{CLASSES_ENV} selected none of the model's {len(declared)} classes; "
            f"a detector that can produce nothing would never be routed to"
        )

    mappings = tuple(
        MappingEntry(native_label=name, class_id=ClassId(_platform_class(name)))
        for name in selected
    )
    model_id = path.stem
    artifact_hash = f"blake2b:{_digest(path)}"

    detector = YoloDetector(
        clock=clock,
        session=session,
        mapping=TaxonomyMapping(
            adapter_id=AdapterId("detector.yolo"),
            model_id=ModelId(model_id),
            entries=mappings,
            unmapped_policy=UnmappedPolicy.DROP,
            native_label_space="coco" if len(declared) == 80 else "custom",
        ),
        model_id=ModelId(model_id),
        model_version="1.0.0",
        artifact_hash=artifact_hash,
    )

    return BoundDetector(
        detector=detector,
        adapter_id="detector.yolo",
        classes=tuple(dict.fromkeys(entry.class_id for entry in mappings)),
        mappings=mappings,
        model_id=model_id,
        model_version="1.0.0",
        artifact_hash=artifact_hash,
        artifact_path=str(path),
        note=(
            f"yolo ({model_id}, {len(selected)} of {len(declared)} classes"
            + (", narrowed by configuration" if wanted else "")
            + ", closed-set)"
        ),
        # An exported detection graph has a fixed classification head. It scores
        # exactly these labels and returns the best of them; there is no index
        # meaning "not one of these". That is a property of the artifact, not a
        # configuration choice, so it is declared here rather than made settable.
        label_space_kind=CLOSED_SET,
        native_labels=tuple(selected),
        native_label_space="coco" if len(declared) == 80 else "custom",
    )


def _build_reference(*, clock, env: Mapping[str, str], **_: Any) -> BoundDetector:
    """The scripted detector. **For fixtures, never for a demo or a deployment.**

    It answers with a fixed box on every frame regardless of the pixels, which
    makes it invaluable for testing the letterbox inverse without a GPU and
    actively misleading anywhere else. Selecting it is therefore explicit, and
    the note says plainly what it is so the model panel cannot imply otherwise.
    """
    from ..detection.reference import ReferenceDetector, ScriptedDetection
    from ...core.model.space import Box

    person = ClassId("person")
    return BoundDetector(
        detector=ReferenceDetector(
            clock=clock,
            producible_classes=(person,),
            script=(ScriptedDetection(person, Box(0.3, 0.15, 0.62, 0.9), 0.92),),
        ),
        adapter_id="detector.reference",
        classes=(person,),
        mappings=(MappingEntry(native_label="person", class_id=person),),
        model_id="reference-detector",
        model_version="1.0.0",
        artifact_hash=f"blake2b:{hashlib.blake2b(b'reference', digest_size=32).hexdigest()}",
        artifact_path="",
        note="reference (scripted — a fixed box, not real detection)",
        label_space_kind=CLOSED_SET,
        native_labels=("person",),
        native_label_space="scripted",
    )


def _build_open_vocabulary(*, clock, env: Mapping[str, str], **_: Any) -> BoundDetector:
    """The open-vocabulary slot. **Declared, and deliberately unimplemented.**

    A registered provider that refuses is worth more than an absent one. It
    states, in the one table a deployment actually reads, that this platform has
    a place for a detector whose vocabulary is supplied at query time — and that
    nothing here will pretend to be one.

    The alternative was to let a closed-set model stand in. That is precisely the
    failure this whole seam exists to prevent: a COCO detector asked about a pen
    answers `toothbrush` at 0.454, and dressing that up as open-vocabulary
    detection would launder a nearest-neighbour guess into an identification.

    Binding a real one — OWL-ViT, Grounding DINO, YOLO-World — is a factory entry
    and an adapter, and touches no Vision OS module: `DetectorPort` already
    accommodates it (its own docstring names Grounding DINO among the
    implementations it expects), the mapping already absorbs a native label
    space, and the capability declaration already has somewhere to say what kind
    of label space is bound.
    """
    raise DetectorConfigurationError(
        f"{PROVIDER_ENV}='{OPEN_VOCABULARY}' names a capability this deployment "
        f"has not bound. Open-vocabulary detection needs an adapter whose labels "
        f"are supplied at query time (OWL-ViT, Grounding DINO, YOLO-World); no "
        f"such adapter is registered. The bundled '{DEFAULT_PROVIDER}' provider "
        f"is closed-set and cannot substitute: asked about an object outside its "
        f"vocabulary it returns the nearest word it knows, which is a guess "
        f"wearing the clothes of an identification."
    )


#: A closed table, like the understander, crop-strategy and coercion factories.
DETECTOR_FACTORIES: Mapping[str, Any] = {
    "yolo": _build_yolo,
    "reference": _build_reference,
    OPEN_VOCABULARY: _build_open_vocabulary,
}


def build_detector(
    *,
    clock,
    provider: str | None = None,
    env: Mapping[str, str] | None = None,
) -> BoundDetector:
    """Build the configured detector, or say clearly why not.

    Raises:
        DetectorConfigurationError: the provider is unknown, or its weights are
            missing. Both are composition-time failures, and neither falls back
            to the scripted detector: a demo that silently degraded to a fixed
            box would show a working pipeline built on a constant.
    """
    source = dict(os.environ if env is None else env)
    name = (provider or resolve_detector_provider(source)).strip().lower()
    factory = DETECTOR_FACTORIES.get(name)
    if factory is None:
        raise DetectorConfigurationError(
            f"{PROVIDER_ENV}='{name}' is not a registered detector provider. "
            f"Available: {', '.join(sorted(DETECTOR_FACTORIES))}."
        )
    return factory(clock=clock, env=source)


def _platform_class(native_label: str) -> str:
    """COCO labels contain spaces; platform class ids use underscores."""
    return native_label.strip().lower().replace(" ", "_")


def _digest(path: Path) -> str:
    hasher = hashlib.blake2b(digest_size=32)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


__all__ = [
    "CLASSES_ENV",
    "CLOSED_SET",
    "OPEN_VOCABULARY",
    "DEFAULT_PROVIDER",
    "DETECTOR_FACTORIES",
    "PROVIDER_ENV",
    "WEIGHTS_ENV",
    "BoundDetector",
    "DetectorConfigurationError",
    "build_detector",
    "default_weights_path",
    "label_space_view",
    "resolve_detector_provider",
]
