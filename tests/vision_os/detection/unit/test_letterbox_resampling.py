"""The detector's letterbox averages when it shrinks a frame.

A 1920x1080 camera fitted into a 640 network is a scale of one third. Taken by
nearest neighbour, that keeps one pixel in nine and discards the rest, so a
person becomes whatever pixels happened to land on the sampling grid.

Measured on 30 PCC2 decision frames (2026-10-15) at the kitchen policy's 0.4
confidence floor: yolov8n found 73 people through a sampled letterbox and 82
through an averaged one; yolo11s, 90 and 93. The pose adapter met the same
defect and fixed it the same way (`adapters/perception/pose.py::_resample`).
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_os.adapters.detection.letterbox import compute_transform
from vision_os.adapters.models.runtimes import OnnxDetectorSession


def fit(frame: np.ndarray, size: int = 640) -> tuple[np.ndarray, object]:
    height, width = frame.shape[:2]
    transform = compute_transform(
        source_width=width, source_height=height, target_width=size, target_height=size
    )
    return OnnxDetectorSession._letterbox(frame, transform, np), transform


def checkerboard(height: int, width: int) -> np.ndarray:
    """One-pixel black/white squares: the finest detail an image can hold."""
    ys, xs = np.indices((height, width))
    board = ((ys + xs) % 2 * 255).astype(np.uint8)
    return np.repeat(board[:, :, None], 3, axis=2)


def test_shrinking_averages_detail_instead_of_sampling_it() -> None:
    canvas, transform = fit(checkerboard(1080, 1920))
    top, left = round(transform.pad_y), round(transform.pad_x)
    inner = canvas[top + 2 : top + 356, left + 2 : left + 638].astype(np.float32)
    # Averaged, a checkerboard is mid grey everywhere. Sampled, it is black and
    # white noise with a standard deviation near 127.
    assert inner.std() < 20
    assert inner.mean() == pytest.approx(127.5, abs=10)


def test_the_canvas_and_padding_are_unchanged() -> None:
    canvas, transform = fit(np.full((1080, 1920, 3), 200, dtype=np.uint8))
    assert canvas.shape == (640, 640, 3)
    top = round(transform.pad_y)
    assert (canvas[: top - 1] == 114).all()
    assert (canvas[top + 1 : top + 359] == 200).all()


def test_a_flat_region_keeps_its_exact_value() -> None:
    """Averaging must not shift colour, or every detection would see a tint."""
    frame = np.zeros((576, 960, 3), dtype=np.uint8)
    frame[:, :, 0], frame[:, :, 1], frame[:, :, 2] = 30, 140, 220
    canvas, transform = fit(frame)
    top = round(transform.pad_y)
    middle = canvas[top + 10 : top + 300, 10:630]
    assert (middle[:, :, 0] == 30).all() and (middle[:, :, 2] == 220).all()


def test_the_accelerated_and_plain_paths_agree(monkeypatch) -> None:
    """OpenCV is an optional accelerator; without it the numpy path must give
    the same picture, or detection would differ between two installs."""
    import sys

    from vision_os.adapters.models.runtimes import _area_resample

    pytest.importorskip("cv2")
    rng = np.random.default_rng(seed=3)
    frame = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    accelerated = _area_resample(frame, 360, 640, np).astype(np.int16)
    monkeypatch.setitem(sys.modules, "cv2", None)
    plain = _area_resample(frame, 360, 640, np).astype(np.int16)
    assert np.abs(accelerated - plain).max() <= 1


def test_a_frame_smaller_than_the_network_is_still_fitted() -> None:
    canvas, _ = fit(np.full((240, 320, 3), 50, dtype=np.uint8))
    assert canvas.shape == (640, 640, 3)
    assert (canvas[320, 320] == 50).all()
