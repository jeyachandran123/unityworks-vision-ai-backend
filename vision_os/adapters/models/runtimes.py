"""P26 ``ModelRuntimePort`` — turn a verified artifact into an executable session.

This port is what makes an adapter *family* portable. The same YOLO adapter runs
on ultralytics locally and on ONNX at the edge, because the parts that matter for
correctness — letterboxing, coordinate inversion, taxonomy mapping — live in the
adapter, and only the tensor call lives here.

``DetectorSession`` is the narrow contract the detection adapters consume: give
it letterboxed images, get back boxes in letterboxed pixel space. Everything
about inverting that space is the adapter's job and is therefore testable without
a GPU.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from time import sleep
from typing import Protocol, runtime_checkable

from ...core.errors import ModelLoadError
from ...core.ports.models import LoadedModel


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """How a source image was fitted into the model's input square.

    Retained so the inverse is exact rather than approximate. Recording the
    transform is what makes two models comparable: evaluating them on
    differently-letterboxed crops is not a fair comparison, and without this
    field nobody finds out.
    """

    scale: float
    pad_x: float
    pad_y: float
    source_width: int
    source_height: int
    target_width: int
    target_height: int

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"letterbox scale must be positive, got {self.scale}")
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("letterbox source dimensions must be positive")


@dataclass(frozen=True, slots=True)
class LetterboxedImage:
    """One image prepared for inference, with the transform that produced it."""

    pixels: memoryview
    width: int
    height: int
    transform: LetterboxTransform


@dataclass(frozen=True, slots=True)
class RawBox:
    """A model's native output: letterboxed pixel space, native class index."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_index: int


@runtime_checkable
class DetectorSession(Protocol):
    """An executable detection model."""

    def infer(
        self, images: Sequence[LetterboxedImage]
    ) -> Sequence[Sequence[RawBox]]:
        """Run inference. Results map 1:1 and in order to ``images``."""
        ...

    def class_names(self) -> Sequence[str]:
        """The model's native label space, indexed by class index."""
        ...

    def close(self) -> None: ...


# --- scripted runtime ------------------------------------------------------- #


@dataclass(slots=True)
class ScriptedSession:
    """A deterministic session that replays scripted boxes.

    The platform's reference detection runtime: dependency-free, exactly
    reproducible, and therefore usable in CI without a GPU or a model file. It is
    what makes the letterbox-inverse tests — the highest-value tests in the
    detection layer — runnable everywhere.
    """

    script: dict[str, Sequence[RawBox]] = field(default_factory=dict)
    default: Sequence[RawBox] = ()
    names: Sequence[str] = ("person", "car")
    calls: int = 0
    fail_after: int = 0
    """Raise on the Nth call onwards, to exercise the failure ladder."""

    stall_after: int = 0
    stall_seconds: float = 0.25
    """Block for ``stall_seconds`` from the Nth call, to exercise the inference
    timeout. Bounded deliberately: a test that sleeps for half a minute to prove
    a timeout works is a test nobody will keep running."""

    closed: bool = False

    def infer(self, images: Sequence[LetterboxedImage]) -> Sequence[Sequence[RawBox]]:
        self.calls += 1
        if self.fail_after and self.calls >= self.fail_after:
            raise RuntimeError("scripted inference failure")
        if self.stall_after and self.calls >= self.stall_after:
            sleep(self.stall_seconds)
        return [self._for(image) for image in images]

    def _for(self, image: LetterboxedImage) -> Sequence[RawBox]:
        key = f"{image.transform.source_width}x{image.transform.source_height}"
        return self.script.get(key, self.default)

    def class_names(self) -> Sequence[str]:
        return self.names

    def close(self) -> None:
        self.closed = True


class ScriptedRuntime:
    """Loads ``ScriptedSession`` instances. The reference P26 adapter."""

    def __init__(
        self,
        *,
        session_factory=None,
        vram_bytes: int = 0,
        warmup_ms: float = 1.0,
    ) -> None:
        self._session_factory = session_factory or (lambda: ScriptedSession())
        self._vram_bytes = vram_bytes
        self._warmup_ms = warmup_ms
        self._lock = threading.Lock()
        self.loaded: list[LoadedModel] = []

    @property
    def runtime_id(self) -> str:
        return "scripted"

    def supports(self, artifact_path: str, precision: str) -> bool:
        return True

    def load(
        self,
        *,
        model_id: str,
        version: str,
        artifact_path: str,
        artifact_hash: str,
        device_id: str,
        precision: str,
        options: dict[str, str] | None = None,
    ) -> LoadedModel:
        session = self._session_factory()
        loaded = LoadedModel(
            model_id=model_id,
            version=version,
            artifact_hash=artifact_hash,
            device_id=device_id,
            precision=precision,
            session=session,
            vram_bytes=self._vram_bytes,
            warmup_ms=self._warmup_ms,
            metadata={"runtime": self.runtime_id},
        )
        with self._lock:
            self.loaded.append(loaded)
        return loaded

    def unload(self, loaded: LoadedModel) -> None:
        session = loaded.session
        if isinstance(session, ScriptedSession):
            session.close()
        with self._lock:
            if loaded in self.loaded:
                self.loaded.remove(loaded)


# --- ultralytics runtime ----------------------------------------------------- #


class UltralyticsSession:
    """Wraps an ultralytics model behind ``DetectorSession``.

    Deliberately thin. Everything that decides whether a box lands in the right
    place lives in the YOLO *adapter*, not here, so the correctness-critical code
    is exercised in CI while this wrapper is the only part that needs a GPU.
    """

    def __init__(self, model, device_id: str, half: bool) -> None:
        self._model = model
        self._device_id = device_id
        self._half = half
        self._lock = threading.Lock()

    def infer(self, images: Sequence[LetterboxedImage]) -> Sequence[Sequence[RawBox]]:
        try:
            import numpy as np  # noqa: PLC0415 - optional, adapter-scoped
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError("numpy is required by the ultralytics runtime") from exc

        batch = [
            np.frombuffer(image.pixels, dtype=np.uint8).reshape(
                image.height, image.width, 3
            )
            for image in images
        ]
        with self._lock:
            predictions = self._model.predict(
                batch, verbose=False, device=self._device_id, half=self._half
            )

        results: list[list[RawBox]] = []
        for prediction in predictions:
            boxes: list[RawBox] = []
            container = getattr(prediction, "boxes", None)
            if container is not None:
                for row, score, class_index in zip(
                    container.xyxy.tolist(),
                    container.conf.tolist(),
                    container.cls.tolist(),
                    strict=False,
                ):
                    boxes.append(
                        RawBox(
                            x1=float(row[0]),
                            y1=float(row[1]),
                            x2=float(row[2]),
                            y2=float(row[3]),
                            score=float(score),
                            class_index=int(class_index),
                        )
                    )
            results.append(boxes)
        return results

    def class_names(self) -> Sequence[str]:
        names = getattr(self._model, "names", {})
        if isinstance(names, dict):
            return [names[key] for key in sorted(names)]
        return list(names)

    def close(self) -> None:
        self._model = None


class UltralyticsRuntime:
    """Loads YOLO weights through ultralytics.

    The import is deferred to ``load`` so a deployment without ultralytics starts
    normally and simply cannot bind this runtime — an absent optional dependency
    is a capability gap, not a startup failure.
    """

    def __init__(self, *, warmup_enabled: bool = True) -> None:
        self._warmup_enabled = warmup_enabled

    @property
    def runtime_id(self) -> str:
        return "ultralytics"

    def supports(self, artifact_path: str, precision: str) -> bool:
        if precision not in ("fp32", "fp16"):
            return False
        return artifact_path.endswith((".pt", ".pth", ".engine", ".onnx"))

    def load(
        self,
        *,
        model_id: str,
        version: str,
        artifact_path: str,
        artifact_hash: str,
        device_id: str,
        precision: str,
        options: dict[str, str] | None = None,
    ) -> LoadedModel:
        try:
            from ultralytics import YOLO  # noqa: PLC0415 - optional dependency
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                "the ultralytics runtime is not installed; bind a different "
                "ModelRuntimePort or install the optional dependency",
                model_id=model_id,
            ) from exc

        try:
            model = YOLO(artifact_path)
            if device_id != "cpu":
                model.to(device_id)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"ultralytics failed to load '{artifact_path}': {exc}",
                model_id=model_id,
            ) from exc

        session = UltralyticsSession(model, device_id, half=precision == "fp16")
        return LoadedModel(
            model_id=model_id,
            version=version,
            artifact_hash=artifact_hash,
            device_id=device_id,
            precision=precision,
            session=session,
            metadata={"runtime": self.runtime_id, "artifact": artifact_path},
        )

    def unload(self, loaded: LoadedModel) -> None:
        session = loaded.session
        if isinstance(session, UltralyticsSession):
            session.close()


# --- ONNX Runtime ------------------------------------------------------------ #


class OnnxDetectorSession:
    """Wraps an ONNX detection graph behind ``DetectorSession``.

    The sibling of ``UltralyticsSession``, and thin for the same reason: every
    decision that determines *where a box lands* — letterboxing, the inverse,
    taxonomy translation — belongs to the adapter and is tested without a model
    file. What lives here is the tensor call plus the two things only a runtime
    can know: how this graph's output tensor is laid out, and what its class
    indices mean.

    Both of those are worth stating, because getting either wrong produces boxes
    that are subtly and undetectably wrong rather than absent.

    **The export carries no NMS.** A YOLOv8 ONNX graph emits every anchor —
    ``(1, 84, 8400)`` — and suppression is the caller's job. It happens here and
    is declared upward through the adapter's ``NmsDeclaration``, because
    obligation D4 is explicit that a platform cannot correct for suppression it
    does not know about. Skipping it floods tracking with duplicates of every
    object.

    **The graph wants RGB; the platform speaks BGR24.** ``InputConstraints``
    declares ``bgr24`` and the crop pipeline honours it, so the swap belongs at
    this boundary. Omitting it raises no error and costs roughly half the
    model's accuracy — red and blue trade places and nothing ever says so.
    """

    __slots__ = ("_conf", "_input", "_iou", "_lock", "_names", "_session")

    def __init__(
        self,
        session,
        *,
        names: Sequence[str],
        input_name: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self._session = session
        self._names = tuple(names)
        self._input = input_name
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._lock = threading.Lock()

    def infer(self, images: Sequence[LetterboxedImage]) -> Sequence[Sequence[RawBox]]:
        """Run inference. Results map 1:1 and in order to ``images``.

        One image per call: the exported graph fixes its batch dimension at 1.
        The adapter declares ``max_batch_size`` accordingly rather than batching
        into a shape this graph cannot accept.
        """
        try:
            import numpy as np  # noqa: PLC0415 - optional, adapter-scoped
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError("numpy is required by the ONNX runtime") from exc

        results: list[list[RawBox]] = []
        for image in images:
            frame = np.frombuffer(image.pixels, dtype=np.uint8).reshape(
                image.height, image.width, 3
            )
            # Apply the transform the adapter computed. P8 hands a runtime the
            # *source* pixels plus a `LetterboxTransform` describing how they
            # should be fitted; carrying it out belongs to whoever knows the
            # graph's input shape, which is here. `UltralyticsSession` gets away
            # with passing frames through only because ultralytics letterboxes
            # internally, and `ScriptedSession` ignores pixels entirely — so this
            # obligation had never been exercised before an exported graph with a
            # fixed 640x640 input arrived and rejected everything.
            fitted = self._letterbox(frame, image.transform, np)

            # BGR24 -> RGB, HWC -> CHW, uint8 -> float32 in [0, 1].
            tensor = fitted[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            tensor = np.ascontiguousarray(tensor)[None, ...]

            with self._lock:
                raw = self._session.run(None, {self._input: tensor})[0]

            results.append(self._decode(raw, np))
        return results

    @staticmethod
    def _letterbox(frame, transform, np):
        """Scale by ``transform.scale`` and centre on a padded canvas.

        Nearest-neighbour, and deliberately so: it needs no imaging library, and
        the adapter's inverse is pure arithmetic over the same transform, so
        whatever this does to coordinates is undone exactly. A smoother
        resampler would change the pixels a model sees without changing where a
        box lands.

        The 114 grey fill is YOLO's convention. It matters only in that the
        padding must be a constant the model was trained to ignore.
        """
        source_h, source_w = frame.shape[:2]
        target_w = int(transform.target_width)
        target_h = int(transform.target_height)

        scaled_w = max(1, int(round(source_w * transform.scale)))
        scaled_h = max(1, int(round(source_h * transform.scale)))

        rows = np.minimum(
            (np.arange(scaled_h) / transform.scale).astype(np.int32), source_h - 1
        )
        cols = np.minimum(
            (np.arange(scaled_w) / transform.scale).astype(np.int32), source_w - 1
        )
        resized = frame[rows][:, cols]

        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        top = int(round(transform.pad_y))
        left = int(round(transform.pad_x))
        # Clipped rather than assumed: a rounded pad plus a rounded scale can
        # exceed the canvas by a pixel, and an exception here would read as a
        # model failure rather than an off-by-one.
        height = min(scaled_h, target_h - top)
        width = min(scaled_w, target_w - left)
        if height > 0 and width > 0:
            canvas[top : top + height, left : left + width] = resized[:height, :width]
        return canvas

    def _decode(self, raw, np) -> list[RawBox]:
        """``(1, 84, 8400)`` -> suppressed boxes in letterboxed pixel space.

        Rows are ``[cx, cy, w, h, *class_scores]``. There is no objectness
        channel in v8: the class score *is* the confidence.
        """
        predictions = raw[0].T  # (8400, 84)
        if predictions.shape[1] < 5:
            return []

        scores_by_class = predictions[:, 4:]
        class_indices = scores_by_class.argmax(axis=1)
        scores = scores_by_class[np.arange(scores_by_class.shape[0]), class_indices]

        keep = scores >= self._conf
        if not keep.any():
            return []

        boxes = predictions[keep, :4]
        scores = scores[keep]
        class_indices = class_indices[keep]

        # Centre-form -> corner-form, still in letterboxed pixels. The adapter
        # inverts the letterbox; this function must not.
        half_w = boxes[:, 2] / 2.0
        half_h = boxes[:, 3] / 2.0
        corners = np.stack(
            [
                boxes[:, 0] - half_w,
                boxes[:, 1] - half_h,
                boxes[:, 0] + half_w,
                boxes[:, 1] + half_h,
            ],
            axis=1,
        )

        out: list[RawBox] = []
        # Per class, because suppressing across classes would delete a person
        # standing in front of a car — a real detection lost to a rule that only
        # ever meant to remove duplicates of the same thing.
        for class_index in np.unique(class_indices):
            mask = class_indices == class_index
            for row, score in self._suppress(corners[mask], scores[mask], np):
                out.append(
                    RawBox(
                        x1=float(row[0]),
                        y1=float(row[1]),
                        x2=float(row[2]),
                        y2=float(row[3]),
                        score=float(score),
                        class_index=int(class_index),
                    )
                )

        # Highest confidence first, so a consumer taking the top N takes the
        # best N, and two runs over one frame agree on the order (V13).
        out.sort(key=lambda box: box.score, reverse=True)
        return out

    def _suppress(self, boxes, scores, np):
        """Greedy IoU suppression within one class."""
        order = scores.argsort()[::-1]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        kept: list[tuple] = []

        while order.size:
            best = order[0]
            kept.append((boxes[best], scores[best]))
            if order.size == 1:
                break
            rest = order[1:]

            x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
            y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
            x2 = np.minimum(boxes[best, 2], boxes[rest, 2])
            y2 = np.minimum(boxes[best, 3], boxes[rest, 3])
            overlap = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
            union = areas[best] + areas[rest] - overlap
            iou = np.where(union > 0, overlap / union, 0.0)

            order = rest[iou <= self._iou]

        return kept

    def class_names(self) -> Sequence[str]:
        return self._names

    def close(self) -> None:
        self._session = None


def open_onnx_detector_session(
    artifact_path: str,
    *,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    providers: Sequence[str] | None = None,
) -> OnnxDetectorSession:
    """Load an ONNX detection graph and read its own class list.

    The class names come from the graph's metadata rather than from a constant
    in this repository. A hard-coded COCO list silently mislabels every object
    the moment someone swaps in a model trained on anything else, and a
    mislabelled detection is worse than a missing one: it is wrong with full
    confidence, and it propagates into tracking, cropping, prompts and
    observations before anybody notices.
    """
    try:
        import ast  # noqa: PLC0415
        import onnxruntime  # noqa: PLC0415 - optional, adapter-scoped
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError("onnxruntime is required by the ONNX runtime") from exc

    try:
        session = onnxruntime.InferenceSession(
            artifact_path, providers=list(providers or ["CPUExecutionProvider"])
        )
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"onnxruntime failed to load '{artifact_path}': {exc}") from exc

    declared = session.get_modelmeta().custom_metadata_map.get("names", "")
    try:
        parsed = ast.literal_eval(declared) if declared else {}
    except (ValueError, SyntaxError):
        parsed = {}

    if isinstance(parsed, dict):
        names = [str(parsed[key]) for key in sorted(parsed)]
    elif isinstance(parsed, (list, tuple)):
        names = [str(name) for name in parsed]
    else:
        names = []

    if not names:
        raise ModelLoadError(
            f"'{artifact_path}' declares no class names in its metadata. Without "
            f"them a class index cannot be translated, and a detector that "
            f"cannot name what it found must not be bound."
        )

    return OnnxDetectorSession(
        session,
        names=names,
        input_name=session.get_inputs()[0].name,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
