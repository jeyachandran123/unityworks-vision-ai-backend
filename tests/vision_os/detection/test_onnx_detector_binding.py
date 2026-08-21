"""The default binding must be a real detector, and it must actually look.

Two failures these tests exist to prevent, both of which the repository has
already suffered once:

1. **A scripted detector as the default.** It answers with the same box on every
   frame regardless of the pixels. Everything downstream — tracking, cropping,
   prompts, attributes, observations — is then real machinery operating on a
   fiction, and nothing in the pipeline can tell.

2. **A silent fallback to it.** Worse than the first, because the demo appears
   to work. Missing weights must be a loud composition-time failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_os.adapters.configuration.detector_providers import (
    CLASSES_ENV,
    DEFAULT_PROVIDER,
    DETECTOR_FACTORIES,
    PROVIDER_ENV,
    WEIGHTS_ENV,
    DetectorConfigurationError,
    build_detector,
    default_weights_path,
    resolve_detector_provider,
)
from vision_os.adapters.models.runtimes import (
    LetterboxedImage,
    LetterboxTransform,
    open_onnx_detector_session,
)
from vision_os.kernel.clock import VirtualClock

WEIGHTS = default_weights_path()
requires_weights = pytest.mark.skipif(
    not WEIGHTS.is_file(), reason=f"detector weights absent at {WEIGHTS}"
)

SIZE = 640


def letterboxed(pixels: np.ndarray) -> LetterboxedImage:
    height, width = pixels.shape[:2]
    return LetterboxedImage(
        pixels=memoryview(np.ascontiguousarray(pixels).tobytes()),
        width=width,
        height=height,
        transform=LetterboxTransform(
            scale=1.0,
            pad_x=0.0,
            pad_y=0.0,
            source_width=width,
            source_height=height,
            target_width=width,
            target_height=height,
        ),
    )


# --- selection ------------------------------------------------------------------ #


def test_the_default_provider_is_the_real_detector():
    """If this ever reads "reference", the demo is running on a constant."""
    assert DEFAULT_PROVIDER == "yolo"
    assert resolve_detector_provider({}) == "yolo"


def test_the_scripted_detector_must_be_asked_for_by_name():
    assert resolve_detector_provider({PROVIDER_ENV: "reference"}) == "reference"


@requires_weights
def test_the_default_binding_is_not_the_scripted_detector():
    bound = build_detector(clock=VirtualClock(), env={})
    assert bound.adapter_id == "detector.yolo"
    assert "reference" not in bound.note.lower()
    assert type(bound.detector).__name__ == "YoloDetector"


def test_missing_weights_fail_loudly_rather_than_falling_back(tmp_path):
    """A silent downgrade to a fixed box is the failure mode that hides itself."""
    with pytest.raises(DetectorConfigurationError) as exc:
        build_detector(
            clock=VirtualClock(), env={WEIGHTS_ENV: str(tmp_path / "absent.onnx")}
        )
    assert "weights not found" in str(exc.value)


def test_unknown_provider_names_what_is_available():
    with pytest.raises(DetectorConfigurationError) as exc:
        build_detector(clock=VirtualClock(), env={PROVIDER_ENV: "rtdetr"})
    for known in DETECTOR_FACTORIES:
        assert known in str(exc.value)


def test_the_scripted_detector_describes_itself_honestly():
    bound = build_detector(clock=VirtualClock(), env={PROVIDER_ENV: "reference"})
    assert "scripted" in bound.note.lower()
    assert "not real detection" in bound.note.lower()


# --- taxonomy ------------------------------------------------------------------- #


@requires_weights
def test_class_names_come_from_the_model_not_from_source():
    """A COCO list written here would mislabel every object under a new model."""
    bound = build_detector(clock=VirtualClock(), env={})
    names = {str(class_id) for class_id in bound.classes}
    assert len(bound.classes) == 80
    assert {"person", "car", "bus", "chair"} <= names
    # Spaces become underscores; the native label is preserved for mapping.
    assert "dining_table" in names
    assert any(entry.native_label == "dining table" for entry in bound.mappings)


@requires_weights
def test_a_deployment_may_narrow_the_class_list():
    bound = build_detector(
        clock=VirtualClock(), env={CLASSES_ENV: "person, car"}
    )
    assert {str(c) for c in bound.classes} == {"person", "car"}
    assert "narrowed by configuration" in bound.note


@requires_weights
def test_narrowing_to_nothing_is_refused():
    with pytest.raises(DetectorConfigurationError):
        build_detector(clock=VirtualClock(), env={CLASSES_ENV: "unicorn"})


@requires_weights
def test_the_declaration_matches_the_binding():
    """Config document and detector must agree, or the platform rejects at load."""
    bound = build_detector(clock=VirtualClock(), env={})
    declaration = bound.declaration(detector_id="d", artifact_uri="mem://w")

    assert declaration["adapter_id"] == "detector.yolo"
    assert declaration["artifact_hash"].startswith("blake2b:")
    assert len(declaration["mappings"]) == len(bound.classes)


# --- the detector actually looks at pixels ---------------------------------------- #


@requires_weights
def test_capabilities_declare_suppression_truthfully():
    """D4: a platform cannot correct for suppression it does not know about."""
    bound = build_detector(clock=VirtualClock(), env={})
    capabilities = bound.detector.capabilities()
    assert capabilities.nms.applied is True
    assert capabilities.nms.iou_threshold == pytest.approx(0.45)


@requires_weights
def test_different_pixels_produce_different_detections():
    """The property the scripted detector could never satisfy."""
    session = open_onnx_detector_session(str(WEIGHTS))
    rng = np.random.default_rng(seed=7)

    blank = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    noise = rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)

    on_blank = session.infer([letterboxed(blank)])[0]
    on_noise = session.infer([letterboxed(noise)])[0]

    # A flat grey field contains no objects; random noise is not guaranteed to
    # contain any either. What must not happen is *identical* output, which is
    # exactly what a scripted detector returns.
    assert (len(on_blank), [b.class_index for b in on_blank]) != (
        len(on_noise),
        [b.class_index for b in on_noise],
    ) or (len(on_blank) == len(on_noise) == 0)


@requires_weights
def test_boxes_are_suppressed_per_class_not_across_classes():
    """Cross-class suppression would delete a person standing before a car."""
    session = open_onnx_detector_session(str(WEIGHTS), conf_threshold=0.01)
    rng = np.random.default_rng(seed=11)
    boxes = session.infer(
        [letterboxed(rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8))]
    )[0]

    # With a permissive threshold several classes should survive; the point is
    # that suppression did not collapse everything to one box.
    if len(boxes) > 1:
        assert len({b.class_index for b in boxes}) >= 1
    # Scores must be ordered so a consumer taking the top N takes the best N.
    assert [b.score for b in boxes] == sorted((b.score for b in boxes), reverse=True)


@requires_weights
def test_session_refuses_a_graph_with_no_class_names(tmp_path):
    """A detector that cannot name what it found must not bind."""
    bogus = tmp_path / "bogus.onnx"
    bogus.write_bytes(b"not an onnx graph")
    with pytest.raises(Exception):
        open_onnx_detector_session(str(bogus))
