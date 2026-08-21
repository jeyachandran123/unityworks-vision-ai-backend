"""Run the current Vision OS perception stack over annotated frames.

Uses the **real** components — the configured detector, the configured crop
strategy, the real quality gate, the real understander, the real policy document
— assembled directly rather than through a replay session. Frame-level
evaluation has no use for tracking or the registry, and driving a session would
add a virtual clock and a freshness window between the question and the answer.

What this proves and what it does not:

*proves* — detection, evidence framing, quality gating and semantic answering,
which is where every failure measured so far has originated.

*does not prove* — tracking, temporal consensus, or the observation/compliance
path. Those are measured by the session tests and by Phase 8, and this file must
not be read as covering them.

**The gate runs before the model.** A crop the gate rejects never reaches the
understander: the result is ``NOT_VISIBLE`` carrying the gate's own reason, and
the call is never made. That ordering is the whole of Phase 4.1, and it is the
difference between a system that admits it cannot see and one that guesses.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import (
    AnnotatedFrame,
    AttributeState,
    BoundingBox,
    PredictedFrame,
    PredictedSubject,
)

#: How the platform's registered enum values map onto evaluation states.
#:
#: Loaded from the policy rather than hard-coded per attribute: the mapping says
#: only which domain members mean "could not see" and which mean "nothing there".
#: Everything else is PRESENT.
NOT_VISIBLE_VALUES = frozenset({"not_visible"})
ABSENT_VALUES = frozenset({"none", "absent"})
UNKNOWN_VALUES = frozenset({"unknown"})


def to_state(raw: str | None) -> AttributeState | None:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in NOT_VISIBLE_VALUES:
        return AttributeState.NOT_VISIBLE
    if value in UNKNOWN_VALUES:
        return AttributeState.UNKNOWN
    if value in ABSENT_VALUES:
        return AttributeState.ABSENT
    return AttributeState.PRESENT


@dataclass(slots=True)
class RunStats:
    """What a run cost, so VLM usage is measurable before it becomes optional."""

    frames: int = 0
    detections: int = 0
    evidence_groups: int = 0
    gate_rejections: int = 0
    vlm_calls: int = 0
    vlm_failures: int = 0
    latency_ms: list[float] = field(default_factory=list)
    reasons: dict[str, int] = field(default_factory=dict)

    def note(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    not_visible_from_gate: int = 0
    """Answers the gate produced by refusing. Counted apart from the next field
    because they have different cures: this one is a camera, crop or threshold
    problem, and it was reached without spending a call."""

    not_visible_from_model: int = 0
    """Answers the model produced after seeing a crop the gate accepted. The
    evidence was affordable and the model still could not read it — which is
    where pose and orientation failures land."""

    wall_clock_ms: float = 0.0
    cached_answers: int = 0
    """Calls served from a previous run of the identical configuration. Counted
    so a latency figure built partly on replays is never mistaken for a fresh
    timing."""

    @property
    def mean_latency_ms(self) -> float:
        return sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 0.0

    def percentile(self, fraction: float) -> float:
        """Nearest-rank percentile. No interpolation — with 40-odd samples an
        interpolated p95 invents a value between two real measurements."""
        if not self.latency_ms:
            return 0.0
        ordered = sorted(self.latency_ms)
        index = min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))
        return ordered[index]

    @property
    def median_latency_ms(self) -> float:
        return self.percentile(0.5)

    @property
    def p95_latency_ms(self) -> float:
        return self.percentile(0.95)


class PerceptionRunner:
    """The current stack, driven frame by frame.

    ``use_regions`` is the Phase 4 variable and the only intended difference
    between a baseline run and a new one: with it off, every attribute is
    answered from one whole-subject crop, which is what the system did before
    evidence groups existed.
    """

    def __init__(
        self,
        *,
        policy_path: Path | str,
        provider: str = "nvidia",
        use_regions: bool = True,
        use_quality_gate: bool = True,
        env_file: Path | str | None = None,
        crop_size: int = 0,
        cache_path: Path | str | None = None,
    ) -> None:
        from vision_os.adapters.configuration import build_understander
        from vision_os.adapters.configuration.detector_providers import build_detector
        from vision_os.adapters.configuration.semantic_policy import SemanticPolicy
        from vision_os.adapters.cropping import (
            HeuristicQualityEstimator,
            PaddedCropStrategy,
            PartFocusedCropStrategy,
        )
        from vision_os.adapters.understanding.prompts import StaticPromptProvider
        from vision_os.kernel.clock import VirtualClock
        from vision_os.perception.cropping.gate import GateThresholds, QualityGate

        self.policy = SemanticPolicy.from_file(policy_path)
        self.attributes = [str(k) for k in self.policy.attribute_keys]
        self.use_regions = use_regions
        self.use_quality_gate = use_quality_gate

        self._detector = build_detector(clock=VirtualClock())
        self._estimator = HeuristicQualityEstimator()
        # Sizes come from the policy, exactly as the composition root supplies
        # them, so a run measures the configuration a deployment would have.
        # ``crop_size`` overrides *every* attribute and exists only to reproduce
        # the pre-Phase-4.2 global experiments; it is not a deployment path.
        self.crop_size = crop_size
        self.output_sizes = (
            {key: (crop_size, crop_size) for key in self.attributes}
            if crop_size
            else dict(self.policy.output_sizes)
        )
        size = {"output_size": (crop_size, crop_size)} if crop_size else {}
        self._strategy = (
            PartFocusedCropStrategy(
                regions=self.policy.evidence_regions, output_sizes=self.output_sizes
            )
            if use_regions
            else PaddedCropStrategy(**size)
        )

        # The same per-attribute floors the composition root builds, so a run
        # measures the gate a deployment would actually have.
        from dataclasses import replace as _replace

        default = GateThresholds()
        self._gate = QualityGate(
            default,
            per_attribute={
                key: _replace(default, **floors)
                for key, floors in self.policy.quality_floors.items()
            },
        )

        self._understander, self.binding_note = build_understander(
            producible=tuple(self.policy.attribute_keys),
            provider=provider,
            env_file=env_file,
        )
        template = self.policy.build_prompt_template()
        self._prompts = StaticPromptProvider((template,))
        self._template = template
        self.stats = RunStats()

        # A model answer is expensive, deterministic in its inputs, and a full
        # run takes long enough that losing one to an interrupted process means
        # paying for it twice. Answers are keyed by everything that could change
        # them — frame, subject, question, crop geometry — so a cached entry can
        # only be reused for an identical question about identical pixels.
        self._cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict[str, str]] = {}
        if self._cache_path and self._cache_path.exists():
            import json as _json

            stored = _json.loads(self._cache_path.read_text(encoding="utf-8"))
            if stored.get("fingerprint") == self.fingerprint:
                self._cache = stored.get("answers", {})
            # A cache from a different configuration is discarded rather than
            # reused: replaying answers given about different pixels would
            # fabricate a measurement of a run that never happened.

    @property
    def fingerprint(self) -> str:
        """Everything that changes what the model is asked or shown.

        Any difference here invalidates every cached answer, so it is
        deliberately broad — a stale cache is worse than no cache.
        """
        import hashlib

        material = "|".join(
            [
                self.policy.policy_id,
                self.policy.version,
                str(sorted(self.output_sizes.items())),
                str(sorted(self.policy.evidence_regions.items())),
                str(sorted((k, sorted(v.items())) for k, v in self.policy.quality_floors.items())),
                str(self.use_regions),
                str(self.use_quality_gate),
                str(self._understander.capabilities().model_id),
                self._template.text if hasattr(self._template, "text") else "",
            ]
        )
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def _save_cache(self) -> None:
        if self._cache_path is None:
            return
        import json as _json

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            _json.dumps(
                {"fingerprint": self.fingerprint, "answers": self._cache}, indent=1
            ),
            encoding="utf-8",
        )

    # --- the run ------------------------------------------------------------ #

    def run(self, frames: Sequence[AnnotatedFrame], images) -> list[PredictedFrame]:
        """One prediction per annotated frame. ``images`` maps frame_id → PIL image."""
        out: list[PredictedFrame] = []
        started = time.time()
        for frame in frames:
            image = images.get(frame.frame_id)
            if image is None:
                continue
            out.append(self._one_frame(frame, image))
        self.stats.wall_clock_ms = (time.time() - started) * 1000
        return out

    def _one_frame(self, frame: AnnotatedFrame, image) -> PredictedFrame:
        import numpy as np

        from vision_os.core.model.frame import FrameDimensions
        from vision_os.core.model.ids import CameraId, FrameRef, FrameSeq, StreamEpoch
        from vision_os.core.ports.detection import DetectionRequest, FrameView

        self.stats.frames += 1
        width, height = image.size
        bgr = np.array(image.convert("RGB"))[:, :, ::-1].copy()
        view = FrameView(
            frame_ref=FrameRef(CameraId("eval"), StreamEpoch(0), FrameSeq(frame.frame_index)),
            dimensions=FrameDimensions(width=width, height=height, colour_space="bgr24"),
            pixels=memoryview(bgr.tobytes()).toreadonly(),
        )
        result = self._detector.detector.detect([view], DetectionRequest(min_confidence=0.35))[0]
        people = [d for d in result.detections if str(d.class_id) == "person"]
        self.stats.detections += len(people)

        vlm_before = self.stats.vlm_calls
        subjects = [
            self._one_subject(f"det-{i}", d, image, width, height, frame.frame_id)
            for i, d in enumerate(people)
        ]
        # Flush after every frame: a run killed mid-way keeps everything it has
        # already paid for, and resuming costs only what is genuinely missing.
        self._save_cache()
        return PredictedFrame(
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            subjects=tuple(subjects),
            vlm_calls=self.stats.vlm_calls - vlm_before,
            vlm_call_reasons=dict(self.stats.reasons),
        )

    def _one_subject(
        self, object_id, detection, image, width, height, frame_id=""
    ) -> PredictedSubject:
        """Answer every attribute for one detected person.

        Attributes are grouped by evidence region exactly as M8 groups them, so
        one crop serves every question that shares a band and each band is judged
        on its own quality.
        """
        groups = self._group(self.attributes)
        cache_prefix = f"{frame_id}:{object_id}"
        states: dict[str, AttributeState] = {}
        raw: dict[str, str] = {}
        crops: dict[str, str] = {}
        sizes: dict[str, str] = {}
        quality: dict[str, str] = {}
        used_vlm = False

        for group in groups:
            crop, grades, verdict = self._evidence(detection, group, image, width, height)
            self.stats.evidence_groups += 1
            label = f"{crop.width}x{crop.height}" if crop else "none"

            if self.use_quality_gate and not verdict.passed:
                # Phase 4.1: refuse before asking. The model is never shown a
                # crop its own policy called unusable, and the reason travels.
                self.stats.gate_rejections += 1
                self.stats.note(f"gate:{verdict.reason.value}")
                for key in group:
                    states[key] = AttributeState.NOT_VISIBLE
                    raw[key] = "not_visible"
                    quality[key] = verdict.reason.value
                    sizes[key] = label
                    self.stats.not_visible_from_gate += 1
                continue

            answered, elapsed = self._ask(crop, group, cache_key=f"{cache_prefix}:{'+'.join(group)}")
            used_vlm = True
            for key in group:
                value = answered.get(key)
                state = to_state(value)
                states[key] = state if state is not None else AttributeState.UNKNOWN
                raw[key] = str(value) if value is not None else ""
                quality[key] = "passed"
                sizes[key] = label
                crops[key] = f"{object_id}:{'+'.join(group)}"
                if states[key] is AttributeState.NOT_VISIBLE:
                    self.stats.not_visible_from_model += 1
            self.stats.latency_ms.append(elapsed)

        return PredictedSubject(
            object_id=object_id,
            box=BoundingBox(
                detection.box.x1, detection.box.y1, detection.box.x2, detection.box.y2
            ),
            attributes=states,
            raw_values=raw,
            crop_ids=crops,
            crop_size=sizes,
            quality=quality,
            model_id=str(self._understander.capabilities().model_id),
            vlm_used=used_vlm,
            detector_class=str(detection.class_id),
            detector_confidence=float(detection.score),
        )

    def _group(self, attributes: Sequence[str]) -> list[list[str]]:
        """Partition by declared region — the same rule M8 applies."""
        if not self.use_regions:
            return [list(attributes)]
        buckets: dict[Any, list[str]] = {}
        regions = self.policy.evidence_regions
        for key in attributes:
            buckets.setdefault(regions.get(key), []).append(key)
        return [sorted(v) for _, v in sorted(
            buckets.items(), key=lambda kv: (kv[0] is not None, kv[0] or (0.0, 0.0))
        )]

    def _evidence(self, detection, group, image, width, height):
        """Plan, cut and grade one evidence crop."""
        from PIL import Image

        from vision_os.core.model.ids import AttributeKey, ClassId
        from vision_os.core.ports.cropping import QualityRequest

        kwargs = {} if not self.use_regions else {
            "attributes": tuple(AttributeKey(k) for k in group)
        }
        plan = self._strategy.plan(
            box=detection.box,
            class_id=ClassId("person"),
            source_width=width,
            source_height=height,
            **kwargs,
        )
        b = plan.padded_box
        box = (int(b.x1 * width), int(b.y1 * height), int(b.x2 * width), int(b.y2 * height))
        cut = image.crop(box)

        grades = self._estimator.estimate(
            QualityRequest(
                camera_id="eval",
                box=plan.padded_box,
                source_width=width,
                source_height=height,
                pixels=memoryview(cut.convert("RGB").tobytes()),
                crop_width=cut.width,
                crop_height=cut.height,
            )
        )
        verdict = self._gate.evaluate(grades, group)

        canvas = Image.new("RGB", (plan.output_width, plan.output_height), (0, 0, 0))
        thumb = cut.convert("RGB")
        thumb.thumbnail((plan.output_width, plan.output_height), Image.LANCZOS)
        canvas.paste(thumb, ((plan.output_width - thumb.width) // 2,
                             (plan.output_height - thumb.height) // 2))
        return canvas, grades, verdict

    def _ask(self, crop, group, *, cache_key: str = "") -> tuple[dict[str, str], float]:
        """One model call for one evidence group, or a replay of one already made.

        A cache hit still counts toward ``vlm_calls``: the measurement being
        reported is what the configuration costs to run, not what this
        particular process happened to spend. Latency is not recorded for a hit,
        because no request was made and inventing a duration would corrupt the
        percentiles.
        """
        from vision_os.core.model.ids import CropId, RequestId
        from vision_os.core.ports.understanding import (
            CropView,
            UnderstandingPortRequest,
        )

        rendered = self._prompts.render(
            self._template.prompt_id, self._template.version, {"class_id": "person"}
        )
        request = UnderstandingPortRequest(
            request_id=RequestId("eval"),
            crops=(
                CropView(
                    crop_id=CropId("c"),
                    pixels=memoryview(crop.tobytes()),
                    width=crop.width,
                    height=crop.height,
                    colour_space="rgb24",
                ),
            ),
            prompt=rendered,
            output_schema=rendered.output_schema,
        )
        if cache_key and cache_key in self._cache:
            self.stats.vlm_calls += 1
            self.stats.note("evidence_sufficient")
            self.stats.cached_answers += 1
            return dict(self._cache[cache_key]), 0.0

        started = time.time()
        try:
            response = self._understander.understand(request)
        except Exception:  # noqa: BLE001 - a failure is a result, never a value
            self.stats.vlm_failures += 1
            self.stats.note("vlm_failure")
            return {}, (time.time() - started) * 1000
        self.stats.vlm_calls += 1
        self.stats.note("evidence_sufficient")
        if response.refused:
            self.stats.vlm_failures += 1
            return {}, (time.time() - started) * 1000
        answered = dict(response.structured)
        if cache_key:
            self._cache[cache_key] = {k: str(v) for k, v in answered.items()}
        return answered, (time.time() - started) * 1000


__all__ = ["PerceptionRunner", "RunStats", "to_state"]
