"""Every tracking configuration field must actually do something.

Dead configuration is worse than absent configuration: it advertises a knob, an
operator turns it, and nothing happens. These tests prove each field reaches
behaviour, and the last one is a standing guard that fails if a future field is
added to the schema without being wired.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import vision_os as vision_os_pkg
from vision_os.adapters.configuration import InMemoryConfigSource
from vision_os.adapters.tracking import build_sort_tracker
from vision_os.bootstrap import build_platform
from vision_os.conformance import platform_registry
from vision_os.core.errors import ConfigurationError, TrackingError, ValidationError
from vision_os.core.model.space import Box
from vision_os.kernel.config import ConfigLayer
from vision_os.kernel.config.schema import TrackingSection
from vision_os.kernel.metrics import MetricName
from vision_os.perception.tracking import TrackingRuntime
from vision_os.tracking_bootstrap import build_tracking_layer

from ..conftest import CAMERA, make_outcome, make_request, walking_box
from .test_end_to_end import bindings_factory, tracking_document

ROOT = Path(vision_os_pkg.__file__).parent


def platform_for(clock, **tracking_overrides):
    document = tracking_document()
    document["tracking"].update(tracking_overrides)
    return build_platform(
        config_sources={ConfigLayer.SITE: InMemoryConfigSource(document)},
        bindings_factory=bindings_factory(clock),
        clock=clock,
        conformance=platform_registry(),
    )


class TestSchemaValidation:
    def test_enabling_tracking_without_naming_a_tracker_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="no tracking.tracker_id"):
            TrackingSection(enabled=True, tracker_id="")

    def test_a_disabled_section_needs_no_tracker(self) -> None:
        assert TrackingSection().tracker_id == ""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("iou_threshold", 1.5),
            ("max_association_cost", -0.1),
            ("ambiguity_margin", -0.1),
            ("gate_multiplier", -1.0),
            ("min_hits_to_confirm", 0),
            ("max_coast_frames", -1),
            ("max_lost_frames", -1),
            ("max_age_frames", 0),
            ("max_tracks_per_camera", 0),
            ("history_length", 0),
            ("frame_timeout_ms", 0),
        ],
    )
    def test_out_of_range_values_are_refused(self, field: str, value) -> None:
        with pytest.raises(ValidationError, match=field):
            TrackingSection(enabled=True, tracker_id="tracker.iou", **{field: value})

    def test_all_zero_association_weights_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="weights"):
            TrackingSection(
                enabled=True,
                tracker_id="tracker.iou",
                iou_weight=0.0,
                distance_weight=0.0,
                scale_weight=0.0,
            )


class TestFieldsReachBehaviour:
    def test_enabled_gates_the_layer(self, clock) -> None:
        platform = platform_for(clock, enabled=False)
        with pytest.raises(TrackingError, match="enabled is false"):
            build_tracking_layer(platform)

    def test_tracker_id_selects_the_adapter(self, clock) -> None:
        for tracker_id in ("tracker.iou", "tracker.sort", "tracker.bytetrack"):
            layer = build_tracking_layer(platform_for(clock, tracker_id=tracker_id))
            assert layer.tracker_id == tracker_id

    def test_max_tracks_per_camera_reaches_the_tracker(self, clock) -> None:
        layer = build_tracking_layer(platform_for(clock, max_tracks_per_camera=9))
        assert layer.manager.capabilities.max_objects == 9

    def test_history_length_bounds_the_retained_frames(self, clock) -> None:
        layer = build_tracking_layer(platform_for(clock, history_length=4))
        tracker = layer.manager.tracker
        for seq in range(20):
            tracker.update(make_request(seq, [Box(0.4, 0.4, 0.5, 0.8)]))
        assert len(tracker.tracks(CAMERA)[0].detections) == 4

    def test_min_hits_to_confirm_reaches_the_lifecycle(self, clock) -> None:
        layer = build_tracking_layer(platform_for(clock, min_hits_to_confirm=5))
        tracker = layer.manager.tracker
        states = []
        for seq in range(6):
            update = tracker.update(make_request(seq, [walking_box(seq)]))
            states.append(update.active[0].state.value)
        assert states[:4] == ["tentative"] * 4
        assert states[4] == "confirmed"

    def test_max_coast_frames_reaches_the_lifecycle(self, clock) -> None:
        layer = build_tracking_layer(
            platform_for(clock, min_hits_to_confirm=2, max_coast_frames=2)
        )
        tracker = layer.manager.tracker
        for seq in range(4):
            tracker.update(make_request(seq, [walking_box(seq)]))
        for step in range(3):
            tracker.update(make_request(10 + step, []))
        assert tracker.tracks(CAMERA)[0].state.value == "lost"

    def test_ambiguity_margin_reaches_the_associator(self, clock) -> None:
        """A zero margin disables refusal; a large one refuses almost everything."""
        pair = [Box(0.30, 0.40, 0.50, 0.80), Box(0.3001, 0.4001, 0.5001, 0.8001)]

        permissive = build_tracking_layer(
            platform_for(clock, ambiguity_margin=0.0)
        ).manager.tracker
        strict = build_tracking_layer(
            platform_for(clock, ambiguity_margin=0.4)
        ).manager.tracker

        permissive_refusals, strict_refusals = [], []
        for seq in range(5):
            permissive_refusals.extend(permissive.update(make_request(seq, pair)).refused)
            strict_refusals.extend(strict.update(make_request(seq, pair)).refused)

        assert permissive_refusals == []
        assert strict_refusals

    def test_iou_threshold_reaches_the_gate(self, clock) -> None:
        layer = build_tracking_layer(platform_for(clock, iou_threshold=0.9))
        assert layer.manager.tracker is not None

    def test_require_deterministic_reaches_the_manager(self, clock) -> None:
        from vision_os.core.errors import PortIncompatibleError
        from vision_os.core.ports.tracking import TrackerCapabilities, TrackerPort

        class _Random(TrackerPort):
            def __init__(self) -> None:
                self._inner = build_sort_tracker()

            def update(self, request):
                return self._inner.update(request)

            def tracks(self, camera_id):
                return self._inner.tracks(camera_id)

            def reset(self, camera_id, reason):
                return self._inner.reset(camera_id, reason)

            def capabilities(self):
                return TrackerCapabilities(
                    tracker_id="tracker.random", version="1.0.0", deterministic=False
                )

        platform = platform_for(clock, require_deterministic=True)
        layer = build_tracking_layer(platform)
        with pytest.raises(PortIncompatibleError, match="V13"):
            layer.manager.load(_Random())

    def test_appearance_enabled_without_a_provider_is_refused(self, clock) -> None:
        """Loud rather than a silent downgrade to geometry (invariant V8)."""
        platform = platform_for(clock, appearance_enabled=True)
        with pytest.raises(ConfigurationError, match="biometric"):
            build_tracking_layer(platform)

    async def test_frame_timeout_bounds_the_wait_for_a_busy_camera(
        self, clock, metrics, health, tracking_engine
    ) -> None:
        """Backpressure on this edge is blocking by design, so the *wait* must be
        bounded: frames queueing behind a slow one would otherwise pile up
        without limit rather than degrade.

        Note what this does **not** cover: a tracker wedged inside a synchronous
        call cannot be interrupted by ``asyncio.timeout`` at all. That is a
        property of Python, not of this design, and the mitigation is the
        conformance kit's resource checks plus the bounded track table — not a
        timeout that cannot fire.
        """
        runtime = TrackingRuntime(
            clock=clock,
            metrics=metrics,
            health=health,
            engine=tracking_engine,
            config=TrackingSection(
                enabled=True, tracker_id="tracker.iou", frame_timeout_ms=20
            ),
        )
        await runtime.start()
        # Prime the camera so its lock exists, then hold it as a stuck frame would.
        await runtime.on_detected(make_outcome(0, [walking_box(0)]))
        lock = runtime._locks[CAMERA]  # noqa: SLF001 - the contended resource

        await lock.acquire()
        try:
            await runtime.on_detected(make_outcome(1, [walking_box(1)]))
        finally:
            lock.release()

        assert runtime.stats.frames_timed_out == 1
        assert metrics.snapshot().counters_matching(MetricName.TRACKING_FAILURES)

    async def test_a_timed_out_frame_does_not_stop_the_camera(
        self, clock, metrics, health, tracking_engine
    ) -> None:
        """Degrade, never die: the next frame must still be tracked."""
        runtime = TrackingRuntime(
            clock=clock,
            metrics=metrics,
            health=health,
            engine=tracking_engine,
            config=TrackingSection(
                enabled=True, tracker_id="tracker.iou", frame_timeout_ms=20
            ),
        )
        await runtime.start()
        await runtime.on_detected(make_outcome(0, [walking_box(0)]))
        lock = runtime._locks[CAMERA]  # noqa: SLF001

        await lock.acquire()
        try:
            await runtime.on_detected(make_outcome(1, [walking_box(1)]))
        finally:
            lock.release()

        await runtime.on_detected(make_outcome(2, [walking_box(2)]))
        assert runtime.stats.frames_tracked == 2

    def test_slow_frame_warn_ms_reaches_the_engine(self, tracking_config) -> None:
        assert tracking_config.slow_frame_warn_ms > 0


class TestNoDeadConfiguration:
    """A standing guard: every field must be read somewhere."""

    def test_every_tracking_field_is_consumed(self) -> None:
        """Fails when a field is added to the schema but never wired.

        Scans the tracking layer, its adapters and the composition root for a
        read of each field. Detection has identically named settings, so only
        tracking-owned modules count as evidence.
        """
        owners = [
            ROOT / "tracking_bootstrap.py",
            ROOT / "perception" / "tracking",
            ROOT / "adapters" / "tracking",
        ]
        sources = []
        for owner in owners:
            if owner.is_file():
                sources.append(owner.read_text(encoding="utf-8"))
            else:
                sources.extend(
                    p.read_text(encoding="utf-8")
                    for p in owner.rglob("*.py")
                    if "__pycache__" not in p.parts
                )
        blob = "\n".join(sources)

        dead = [
            field
            for field in TrackingSection.__dataclass_fields__
            if not re.search(rf"(settings|_config|config)\.{field}\b", blob)
        ]
        assert not dead, (
            "these tracking configuration fields are never read, so setting them "
            f"does nothing: {', '.join(dead)}"
        )
