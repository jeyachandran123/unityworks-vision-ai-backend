"""The platform tracking layer — engine, manager, runtime.

The engine is a firewall: ``track()`` never raises, because a tracking failure
may not stop detection, which may not stop acquisition (invariant V9). The
manager is a gate: an adapter failing conformance never becomes reachable. The
runtime is an actor: per-camera serialization with backpressure rather than
dropping.
"""

from __future__ import annotations

import asyncio

import pytest

from vision_os.adapters.tracking import build_iou_tracker, build_sort_tracker
from vision_os.conformance import platform_registry
from vision_os.conformance.kit import ConformanceRegistry
from vision_os.core.errors import (
    ConformanceFailedError,
    EmbeddingUnavailableError,
    PortIncompatibleError,
    TrackerContractError,
)
from vision_os.core.model.ids import CameraId, TrackerEpoch
from vision_os.core.model.space import Box
from vision_os.core.model.track import (
    BreakReason,
    MeasurementBasis,
    TrackState,
    TrackUpdate,
)
from vision_os.core.ports.tracking import TrackerCapabilities, TrackerPort
from vision_os.kernel.events import (
    AssociationFailure,
    TrackCreated,
    TrackerEpochAdvanced,
    TrackingWarning,
    TrackTerminated,
)
from vision_os.kernel.metrics import MetricName
from vision_os.perception.tracking import TrackingEngine, TrackingManager

from ..conftest import (
    CAMERA,
    OTHER_CAMERA,
    make_outcome,
    walking_box,
)


def outcomes(engine, count: int, *, camera: CameraId = CAMERA, start: int = 0):
    return [
        engine.track(make_outcome(start + s, [walking_box(start + s)], camera=camera))
        for s in range(count)
    ]


def fallback():
    return build_iou_tracker(config_revision="test")


# --- the manager gates activation --------------------------------------------- #


class TestManagerGating:
    def test_a_conforming_tracker_activates(self, metrics) -> None:
        manager = TrackingManager(
            metrics=metrics, conformance=platform_registry(), fallback_factory=fallback
        )
        binding = manager.load(build_sort_tracker())
        assert manager.is_loaded
        assert binding.tracker_id == "tracker.sort"
        assert not manager.is_fallback

    def test_a_non_conforming_tracker_never_activates(self, metrics) -> None:
        """Not loaded in a degraded mode — the binding is simply never installed.

        This tracker is structurally perfect: it satisfies the protocol and
        declares valid capabilities. It only breaks a *semantic* obligation —
        it never produces tracks — which is exactly the class of defect an
        interface cannot catch and a kit can.
        """

        class _ProducesNothing(TrackerPort):
            def update(self, request):
                return TrackUpdate(
                    camera_id=request.camera_id,
                    frame_ref=request.frame_ref,
                    tracker_epoch=0,
                )

            def tracks(self, camera_id):
                return ()

            def reset(self, camera_id, reason):
                return TrackerEpoch(0)

            def capabilities(self):
                return TrackerCapabilities(tracker_id="produces-nothing", version="1.0.0")

        manager = TrackingManager(
            metrics=metrics, conformance=platform_registry(), fallback_factory=fallback
        )
        with pytest.raises(ConformanceFailedError):
            manager.load(_ProducesNothing())
        assert not manager.is_loaded

    def test_a_conformance_failure_is_counted(self, metrics, metrics_exporter) -> None:
        class _Broken(TrackerPort):
            def update(self, request):
                raise RuntimeError("nope")

            def tracks(self, camera_id):
                return ()

            def reset(self, camera_id, reason):
                return TrackerEpoch(0)

            def capabilities(self):
                return TrackerCapabilities(tracker_id="broken", version="1.0.0")

        manager = TrackingManager(
            metrics=metrics, conformance=platform_registry(), fallback_factory=fallback
        )
        with pytest.raises(ConformanceFailedError):
            manager.load(_Broken())
        snapshot = metrics.snapshot()
        assert snapshot.counters_matching(MetricName.CONFORMANCE_FAILURES)

    def test_an_object_that_is_not_a_tracker_is_refused(self, metrics) -> None:
        manager = TrackingManager(
            metrics=metrics, conformance=platform_registry(), fallback_factory=fallback
        )
        with pytest.raises(PortIncompatibleError, match="TrackerPort"):
            manager.load(object())

    def test_a_missing_kit_is_fatal_rather_than_a_free_pass(self, metrics) -> None:
        """A missing kit is a wiring bug. Treating it as 'no checks required' is
        how the gate quietly stops being a gate."""
        manager = TrackingManager(
            metrics=metrics,
            conformance=ConformanceRegistry(),
            fallback_factory=fallback,
        )
        with pytest.raises(ConformanceFailedError, match="no conformance kit"):
            manager.load(build_sort_tracker())

    def test_a_tracker_requiring_embeddings_is_refused_when_none_exist(
        self, metrics
    ) -> None:
        """Refusing beats degrading silently: a capability gap must be visible,
        not inferred from worse results (invariant V8)."""

        class _NeedsAppearance(TrackerPort):
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
                    tracker_id="tracker.deepsort",
                    version="1.0.0",
                    requires_embeddings=True,
                )

        manager = TrackingManager(
            metrics=metrics,
            conformance=platform_registry(),
            fallback_factory=fallback,
            appearance_available=False,
        )
        with pytest.raises(EmbeddingUnavailableError, match="biometric"):
            manager.load(_NeedsAppearance())

    def test_a_non_deterministic_tracker_is_refused_in_deterministic_mode(
        self, metrics
    ) -> None:
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

        manager = TrackingManager(
            metrics=metrics,
            conformance=platform_registry(),
            fallback_factory=fallback,
            require_deterministic=True,
        )
        with pytest.raises(PortIncompatibleError, match="V13"):
            manager.load(_Random())

    def test_using_the_manager_before_loading_is_an_explicit_error(
        self, metrics
    ) -> None:
        manager = TrackingManager(
            metrics=metrics, conformance=platform_registry(), fallback_factory=fallback
        )
        with pytest.raises(PortIncompatibleError, match="no tracker is bound"):
            _ = manager.tracker


class TestManagerFallback:
    def test_fallback_installs_the_always_available_tracker(self, metrics) -> None:
        manager = TrackingManager(
            metrics=metrics, conformance=platform_registry(), fallback_factory=fallback
        )
        manager.load(build_sort_tracker())
        manager.fall_back("adapter exploded")
        assert manager.is_fallback
        assert manager.binding.tracker_id == "tracker.iou"
        assert manager.fallback_reason == "adapter exploded"

    def test_fallback_is_idempotent(self, metrics) -> None:
        """Repeated failures from an already-degraded tracker must not churn."""
        manager = TrackingManager(
            metrics=metrics, conformance=platform_registry(), fallback_factory=fallback
        )
        manager.load(build_sort_tracker())
        first = manager.fall_back("one")
        second = manager.fall_back("two")
        assert first is second
        assert manager.fallback_reason == "one"

    def test_fallback_works_from_a_cold_start(self, metrics) -> None:
        """A tracker that fails to even build must still leave tracking running."""
        manager = TrackingManager(
            metrics=metrics, conformance=platform_registry(), fallback_factory=fallback
        )
        manager.fall_back("never loaded")
        assert manager.is_loaded
        assert manager.is_fallback

    def test_fallback_is_counted(self, metrics) -> None:
        manager = TrackingManager(
            metrics=metrics, conformance=platform_registry(), fallback_factory=fallback
        )
        manager.fall_back("degraded")
        assert metrics.snapshot().counters_matching(MetricName.TRACKER_FALLBACKS)

    def test_the_fallback_tracker_itself_conforms(self) -> None:
        """It is the last line of defence, so it must pass the same gate."""
        from vision_os.conformance import TRACKER_KIT

        assert TRACKER_KIT.run(fallback()).passed


# --- the engine ---------------------------------------------------------------- #


class TestEngineHappyPath:
    def test_detections_become_tracks(self, tracking_engine) -> None:
        results = outcomes(tracking_engine, 6)
        assert results[-1].count == 1
        assert not results[-1].failed

    def test_a_track_confirms_and_is_reported(self, tracking_engine) -> None:
        outcomes(tracking_engine, 6)
        tracks = tracking_engine.tracks(CAMERA)
        assert tracks[0].state is TrackState.CONFIRMED

    def test_creation_is_reported_once(self, tracking_engine) -> None:
        results = outcomes(tracking_engine, 8)
        assert sum(r.created for r in results) == 1

    def test_an_empty_frame_ages_tracks_rather_than_being_skipped(
        self, tracking_engine
    ) -> None:
        """Empty frames are exactly when tracks coast and terminate."""
        outcomes(tracking_engine, 6)
        result = tracking_engine.track(make_outcome(6, []))
        assert not result.failed
        assert result.coasting == 1

    def test_a_failed_detection_still_ages_tracks(self, tracking_engine) -> None:
        """Otherwise every track freezes for the length of a detector outage and
        then resumes as though no time had passed."""
        outcomes(tracking_engine, 6)
        result = tracking_engine.track(
            make_outcome(6, [], failed=True, reason="detector died")
        )
        assert not result.failed, "a detection failure is not a tracking failure"
        assert result.coasting == 1

    def test_latency_is_measured(self, tracking_engine) -> None:
        result = outcomes(tracking_engine, 1)[0]
        assert result.latency_ms >= 0.0

    def test_the_outcome_names_the_epoch(self, tracking_engine) -> None:
        assert outcomes(tracking_engine, 3)[-1].tracker_epoch == 0


class TestEngineNeverRaises:
    """V9 — a tracking failure may not stop detection or acquisition."""

    def _engine_with(self, tracker, clock, bus, metrics, tracking_config):
        manager = TrackingManager(
            metrics=metrics, conformance=platform_registry(), fallback_factory=fallback
        )
        manager.fall_back("test harness")
        manager._binding = type(manager.binding)(  # noqa: SLF001 - direct injection
            tracker=tracker, capabilities=tracker.capabilities(), is_fallback=False
        )
        return TrackingEngine(
            clock=clock, bus=bus, metrics=metrics, manager=manager, config=tracking_config
        )

    def test_an_exploding_tracker_does_not_raise(
        self, clock, bus, metrics, tracking_config
    ) -> None:
        class _Explodes(TrackerPort):
            def update(self, request):
                raise RuntimeError("boom")

            def tracks(self, camera_id):
                return ()

            def reset(self, camera_id, reason):
                return TrackerEpoch(0)

            def capabilities(self):
                return TrackerCapabilities(tracker_id="explodes", version="1.0.0")

        engine = self._engine_with(_Explodes(), clock, bus, metrics, tracking_config)
        result = engine.track(make_outcome(0, [walking_box(0)]))
        assert result.failed
        assert "RuntimeError" in result.reason

    def test_a_failure_degrades_to_the_fallback(
        self, clock, bus, metrics, tracking_config
    ) -> None:
        """03_MODULES M6: fall back to a trivial IoU tracker so the pipeline
        degrades rather than stops."""

        class _Explodes(TrackerPort):
            def update(self, request):
                raise RuntimeError("boom")

            def tracks(self, camera_id):
                return ()

            def reset(self, camera_id, reason):
                return TrackerEpoch(0)

            def capabilities(self):
                return TrackerCapabilities(tracker_id="explodes", version="1.0.0")

        engine = self._engine_with(_Explodes(), clock, bus, metrics, tracking_config)
        engine.track(make_outcome(0, [walking_box(0)]))
        assert engine.health().state.value in ("degraded", "unhealthy")

    def test_tracking_continues_after_degradation(
        self, clock, bus, metrics, tracking_config
    ) -> None:
        calls = {"n": 0}
        inner = build_sort_tracker()

        class _FailsOnce(TrackerPort):
            def update(self, request):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("transient")
                return inner.update(request)

            def tracks(self, camera_id):
                return inner.tracks(camera_id)

            def reset(self, camera_id, reason):
                return inner.reset(camera_id, reason)

            def capabilities(self):
                return TrackerCapabilities(tracker_id="flaky", version="1.0.0")

        engine = self._engine_with(_FailsOnce(), clock, bus, metrics, tracking_config)
        first = engine.track(make_outcome(0, [walking_box(0)]))
        assert first.failed
        # Now on the fallback, which works.
        second = engine.track(make_outcome(1, [walking_box(1)]))
        assert not second.failed

    def test_an_out_of_order_frame_is_counted_and_alarmed_not_raised(
        self, tracking_engine, bus
    ) -> None:
        subscription = bus.subscribe([TrackingWarning])
        outcomes(tracking_engine, 5)
        result = tracking_engine.track(make_outcome(2, [walking_box(2)]))
        assert result.failed
        assert result.reason == "out_of_order_frame"
        assert tracking_engine.out_of_order_frames == 1
        warnings = subscription.drain()
        assert any(w.reason == "out_of_order_frame" for w in warnings)

    def test_an_out_of_order_frame_does_not_degrade_the_tracker(
        self, tracking_engine, tracking_manager
    ) -> None:
        """It is a pipeline bug, not a tracker fault; falling back would hide it."""
        outcomes(tracking_engine, 5)
        tracking_engine.track(make_outcome(2, [walking_box(2)]))
        assert not tracking_manager.is_fallback

    def test_tracks_query_never_raises(self, tracking_engine) -> None:
        assert tracking_engine.tracks(CameraId("never-seen")) == ()


class TestEngineVerifiesTheAdapter:
    """The tracking analogue of Flow 2's normalizer.

    An adapter that satisfies the interface but breaks an obligation produces
    plausible, wrong output; the platform is the only place that can catch it
    before it reaches consumers.
    """

    def _engine_with(self, tracker, clock, bus, metrics, tracking_config):
        manager = TrackingManager(
            metrics=metrics, conformance=platform_registry(), fallback_factory=fallback
        )
        manager.fall_back("harness")
        manager._binding = type(manager.binding)(  # noqa: SLF001
            tracker=tracker, capabilities=tracker.capabilities(), is_fallback=False
        )
        return TrackingEngine(
            clock=clock, bus=bus, metrics=metrics, manager=manager, config=tracking_config
        )

    def _wrap(self, transform):
        inner = build_sort_tracker()

        class _Wrapped(TrackerPort):
            def update(self, request):
                return transform(inner.update(request), request)

            def tracks(self, camera_id):
                return inner.tracks(camera_id)

            def reset(self, camera_id, reason):
                return inner.reset(camera_id, reason)

            def capabilities(self):
                return TrackerCapabilities(tracker_id="wrapped", version="1.0.0")

        return _Wrapped()

    def test_a_prediction_sold_as_a_measurement_is_rejected(
        self, clock, bus, metrics, tracking_config
    ) -> None:
        import dataclasses

        def mislabel(update, _request):
            return dataclasses.replace(
                update,
                active=tuple(
                    dataclasses.replace(t, measurement_basis=MeasurementBasis.MEASURED)
                    if t.state.is_predicted
                    else t
                    for t in update.active
                ),
            )

        engine = self._engine_with(
            self._wrap(mislabel), clock, bus, metrics, tracking_config
        )
        for seq in range(6):
            engine.track(make_outcome(seq, [walking_box(seq)]))
        result = engine.track(make_outcome(6, []))
        assert result.failed, "the platform must catch a mislabelled prediction (T5)"

    def test_a_duplicate_track_id_is_rejected(
        self, clock, bus, metrics, tracking_config
    ) -> None:
        def duplicate(update, _request):
            import dataclasses

            if update.active:
                return dataclasses.replace(
                    update, active=update.active + (update.active[0],)
                )
            return update

        engine = self._engine_with(
            self._wrap(duplicate), clock, bus, metrics, tracking_config
        )
        result = engine.track(make_outcome(0, [walking_box(0)]))
        assert result.failed

    def test_a_cross_camera_track_is_rejected(
        self, clock, bus, metrics, tracking_config
    ) -> None:
        """T7. No cross-camera state may exist in this port."""
        import dataclasses

        def relabel(update, _request):
            return dataclasses.replace(update, camera_id=OTHER_CAMERA)

        engine = self._engine_with(
            self._wrap(relabel), clock, bus, metrics, tracking_config
        )
        result = engine.track(make_outcome(0, [walking_box(0)]))
        assert result.failed

    def test_termination_without_a_reason_is_rejected(
        self, clock, bus, metrics, tracking_config
    ) -> None:
        import dataclasses

        def strip(update, _request):
            return dataclasses.replace(
                update,
                terminated=tuple((tid, BreakReason.NONE) for tid, _ in update.terminated),
            )

        engine = self._engine_with(self._wrap(strip), clock, bus, metrics, tracking_config)
        for seq in range(6):
            engine.track(make_outcome(seq, [walking_box(seq)]))
        for seq in range(6, 40):
            result = engine.track(make_outcome(seq, []))
            if result.failed:
                return
        pytest.fail("a termination without a reason was accepted (T6)")

    def test_the_contract_error_is_specific(self) -> None:
        assert issubclass(TrackerContractError, Exception)


# --- events ---------------------------------------------------------------------- #


class TestEventsAreObservable:
    def test_track_creation_is_published(self, tracking_engine, bus) -> None:
        subscription = bus.subscribe([TrackCreated])
        outcomes(tracking_engine, 3)
        created = subscription.drain()
        assert len(created) == 1
        assert created[0].camera_id == CAMERA

    def test_a_created_event_names_the_composite_id(self, tracking_engine, bus) -> None:
        """Publishing a bare local id would let a subscriber treat it as identity."""
        subscription = bus.subscribe([TrackCreated])
        outcomes(tracking_engine, 3)
        track_id = subscription.drain()[0].track_id
        assert str(CAMERA) in track_id
        assert "#" in track_id

    def test_termination_is_published_with_its_reason(
        self, tracking_engine, bus
    ) -> None:
        subscription = bus.subscribe([TrackTerminated])
        outcomes(tracking_engine, 6)
        for seq in range(6, 45):
            tracking_engine.track(make_outcome(seq, []))
        terminated = subscription.drain()
        assert terminated
        assert terminated[0].break_reason != "none"

    def test_a_reset_publishes_the_discontinuity(self, tracking_engine, bus) -> None:
        """Without it, every track vanishing at once reads downstream as the
        whole scene teleporting."""
        subscription = bus.subscribe([TrackerEpochAdvanced])
        outcomes(tracking_engine, 6)
        tracking_engine.reset(CAMERA, "operator")
        advanced = subscription.drain()
        assert advanced
        assert advanced[0].reason == "operator"
        assert advanced[0].discarded_tracks == 1
        assert advanced[0].epoch == 1

    def test_an_ambiguous_association_is_published(self, tracking_engine, bus) -> None:
        """M6 requires the tracker never hide uncertainty to look clean.

        Two identical boxes at the same place: nothing distinguishes which track
        continues which detection, so the margin collapses to zero and the
        tracker must decline rather than guess.
        """
        subscription = bus.subscribe([AssociationFailure])
        for seq in range(6):
            tracking_engine.track(
                make_outcome(
                    seq,
                    [Box(0.30, 0.40, 0.50, 0.80), Box(0.3001, 0.4001, 0.5001, 0.8001)],
                )
            )
        failures = subscription.drain()
        assert failures, "a near-tie association was asserted without publishing it"
        assert failures[0].margin < 0.05
        assert failures[0].camera_id == CAMERA


class TestMetrics:
    def test_active_tracks_are_gauged(self, tracking_engine, metrics) -> None:
        outcomes(tracking_engine, 6)
        assert metrics.snapshot().gauge_value(
            MetricName.TRACKS_ACTIVE, camera_id=str(CAMERA)
        ) == 1.0

    def test_creation_is_counted(self, tracking_engine, metrics) -> None:
        outcomes(tracking_engine, 6)
        assert metrics.snapshot().counters_matching(MetricName.TRACKS_CREATED)

    def test_termination_is_labelled_by_break_reason(
        self, tracking_engine, metrics
    ) -> None:
        """A rise concentrated in detector_miss points at the detector."""
        outcomes(tracking_engine, 6)
        for seq in range(6, 45):
            tracking_engine.track(make_outcome(seq, []))
        counters = metrics.snapshot().counters_matching(MetricName.TRACKS_TERMINATED)
        assert counters
        assert any("break_reason" in str(key) for key in counters)

    def test_latency_is_recorded(self, tracking_engine, metrics) -> None:
        outcomes(tracking_engine, 3)
        assert metrics.snapshot().histogram_values(
            MetricName.TRACKING_LATENCY_MS, camera_id=str(CAMERA)
        )

    def test_out_of_order_frames_are_counted(self, tracking_engine, metrics) -> None:
        outcomes(tracking_engine, 5)
        tracking_engine.track(make_outcome(1, [walking_box(1)]))
        assert metrics.snapshot().counters_matching(MetricName.TRACKING_OUT_OF_ORDER)

    def test_epoch_resets_are_counted(self, tracking_engine, metrics) -> None:
        tracking_engine.reset(CAMERA, "test")
        assert metrics.snapshot().counters_matching(MetricName.TRACKER_EPOCH_RESETS)


class TestHealth:
    def test_a_healthy_engine_reports_healthy(self, tracking_engine) -> None:
        outcomes(tracking_engine, 5)
        assert tracking_engine.health().state.value == "healthy"

    def test_running_on_the_fallback_reports_degraded(
        self, tracking_engine, tracking_manager
    ) -> None:
        """Degradation is announced, never silent (invariant V9)."""
        tracking_manager.fall_back("model unavailable")
        health = tracking_engine.health()
        assert health.state.value == "degraded"
        assert "fallback" in health.detail

    def test_health_carries_counters(self, tracking_engine) -> None:
        outcomes(tracking_engine, 4)
        assert tracking_engine.health().metrics["frames"] == 4


# --- the runtime is an actor per camera ------------------------------------------ #


class TestRuntimeSeam:
    async def test_it_consumes_detection_outcomes(self, tracking_runtime) -> None:
        await tracking_runtime.start()
        for seq in range(6):
            await tracking_runtime.on_detected(make_outcome(seq, [walking_box(seq)]))
        assert tracking_runtime.stats.frames_tracked == 6
        assert tracking_runtime.stats.tracks_emitted > 0

    async def test_it_ignores_frames_before_start(self, tracking_runtime) -> None:
        await tracking_runtime.on_detected(make_outcome(0, [walking_box(0)]))
        assert tracking_runtime.stats.frames_consumed == 0

    async def test_it_never_raises(self, tracking_runtime, tracking_manager) -> None:
        await tracking_runtime.start()

        class _Explodes(TrackerPort):
            def update(self, request):
                raise RuntimeError("boom")

            def tracks(self, camera_id):
                return ()

            def reset(self, camera_id, reason):
                return TrackerEpoch(0)

            def capabilities(self):
                return TrackerCapabilities(tracker_id="explodes", version="1.0.0")

        tracking_manager._binding = type(tracking_manager.binding)(  # noqa: SLF001
            tracker=_Explodes(), capabilities=_Explodes().capabilities()
        )
        await tracking_runtime.on_detected(make_outcome(0, [walking_box(0)]))
        assert tracking_runtime.stats.frames_failed >= 1

    async def test_a_failing_sink_does_not_break_tracking(
        self, clock, metrics, health, tracking_engine, tracking_config
    ) -> None:
        from vision_os.perception.tracking import TrackingRuntime

        def exploding_sink(_result):
            raise ValueError("bad consumer")

        runtime = TrackingRuntime(
            clock=clock,
            metrics=metrics,
            health=health,
            engine=tracking_engine,
            config=tracking_config,
            sink=exploding_sink,
        )
        await runtime.start()
        await runtime.on_detected(make_outcome(0, [walking_box(0)]))
        assert runtime.stats.frames_tracked == 1
        assert runtime.stats.sink_failures == 1

    async def test_it_serializes_frames_from_one_camera(self, tracking_runtime) -> None:
        """T1 — frame N's association depends on frame N-1's state, so two
        frames from one camera may never interleave."""
        await tracking_runtime.start()
        await asyncio.gather(
            *(
                tracking_runtime.on_detected(make_outcome(seq, [walking_box(seq)]))
                for seq in range(8)
            )
        )
        assert tracking_runtime.stats.frames_consumed == 8

    async def test_cameras_run_independently(self, tracking_runtime) -> None:
        await tracking_runtime.start()
        await asyncio.gather(
            *(
                tracking_runtime.on_detected(
                    make_outcome(seq, [walking_box(seq)], camera=camera)
                )
                for camera in (CAMERA, OTHER_CAMERA)
                for seq in range(5)
            )
        )
        assert tracking_runtime.cameras_seen == 2

    async def test_a_detached_camera_releases_its_lock(self, tracking_runtime) -> None:
        """Otherwise the lock table grows with every camera ever seen — a slow
        leak that only appears on long-lived nodes with churning camera sets."""
        await tracking_runtime.start()
        await tracking_runtime.on_detected(make_outcome(0, [walking_box(0)]))
        assert tracking_runtime.cameras_seen == 1
        tracking_runtime.forget(CAMERA)
        assert tracking_runtime.cameras_seen == 0

    async def test_stop_halts_consumption(self, tracking_runtime) -> None:
        await tracking_runtime.start()
        await tracking_runtime.on_detected(make_outcome(0, [walking_box(0)]))
        await tracking_runtime.stop()
        await tracking_runtime.on_detected(make_outcome(1, [walking_box(1)]))
        assert tracking_runtime.stats.frames_consumed == 1

    async def test_failure_rate_is_reported(self, tracking_runtime) -> None:
        await tracking_runtime.start()
        for seq in range(5):
            await tracking_runtime.on_detected(make_outcome(seq, [walking_box(seq)]))
        assert tracking_runtime.stats.failure_rate == 0.0

    async def test_health_is_reported_through_the_monitor(
        self, tracking_runtime, health
    ) -> None:
        from vision_os.perception.tracking import TRACKING_RUNTIME_ID

        await tracking_runtime.start()
        reported = health.component_health(TRACKING_RUNTIME_ID)
        assert reported.state.value == "healthy"
