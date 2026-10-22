"""The production detector is v1, shipped with the checkout.

v1 is the fine-tuned model that also answers the PPE attributes; since
2026-10-22 it is the only model. Its 640 export finds the people. Measured on
kitchen-01's 43 annotated people at the kitchen policy's 0.45 floor, through the
platform's own session and letterbox, person class only:

    yolo11s   41 of 43 found, 7 extra boxes, 94 ms a frame
    v1        41 of 43 found, 8 extra boxes, 46 ms a frame

v1 names nine classes - person and the PPE labels - so the deployment narrows it
to `person` (`VISION_DETECTOR_CLASSES`, defaulted in the settings): a hairnet
box tracked as an object of its own would be judged, drawn and counted as one.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from vision_os.adapters.configuration.detector_providers import (
    CLASSES_ENV,
    build_detector,
    default_weights_path,
)
from vision_os.kernel.clock import VirtualClock

REPO = Path(__file__).resolve().parents[2]


def test_the_default_weights_are_v1_at_640() -> None:
    path = default_weights_path()
    assert path.name == "v1-640.onnx"
    assert path.is_file()


def test_the_default_detector_binds_with_its_own_class_names() -> None:
    bound = build_detector(clock=VirtualClock(), env={})
    assert bound.model_id == "v1-640"
    assert {str(c) for c in bound.classes} >= {"person", "hairnet", "no_hairnet"}


def test_the_settings_narrow_the_detector_to_people() -> None:
    """`.env` reaches the settings object, not `os.environ`: without the bridge
    the PPE labels would become tracked objects."""
    from app.configuration.settings import Settings

    options = Settings().detector_options()
    assert options[CLASSES_ENV] == "person"
    bound = build_detector(clock=VirtualClock(), env=options)
    assert [str(c) for c in bound.classes] == ["person"]


def test_the_weights_are_not_ignored_by_git() -> None:
    """A checkout must reproduce the deployment; ignored weights would not."""
    for weights in ("models/v1-640.onnx", "models/v1.onnx"):
        result = subprocess.run(["git", "check-ignore", "-q", weights], cwd=REPO, check=False)
        assert result.returncode == 1, f"{weights} is git-ignored"
