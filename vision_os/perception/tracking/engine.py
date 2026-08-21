"""M6 Tracking Engine — is this the same thing I saw a moment ago?

> **Single responsibility:** *Associate detections across time within one camera.
> Never assert durable identity — that is M7.*

The engine holds a ``TrackerPort`` and never learns what implements it. What it
owns is everything the *platform* must guarantee regardless of which tracker is
bound:

* **ordering** — per-camera monotonicity, asserted and alarmed (T1);
* **epoch discipline** — a reset mints a new epoch and the discontinuity is
  published, never inferred;
* **contract verification** — the adapter's output is checked before it is
  published, the way Flow 2's normalizer checks a detector's;
* **degradation** — a tracker that fails is replaced by the always-available
  geometric fallback rather than stopping the pipeline (V9);
* **explainability** — every transition becomes an event with its reason.

``track()`` **never raises.** A tracking failure may not stop detection, which
may not stop acquisition. This is the same firewall discipline as Flow 2's
engine, for the same reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...core.errors import (
    OutOfOrderFrameError,
    TrackerCapacityError,
    TrackerContractError,
    TrackingError,
)
from ...core.model.detection import DetectionOutcome
from ...core.model.health import ComponentHealth, HealthState
from ...core.model.ids import CameraId, ModuleId, TrackerEpoch, TrackId
from ...core.model.timebase import Duration, Instant
from ...core.model.track import (
    BreakReason,
    MeasurementBasis,
    Track,
    TrackState,
    TrackUpdate,
)
from ...core.ports.clock import Clock
from ...core.ports.tracking import TrackingRequest
from ...kernel.config.schema import TrackingSection
from ...kernel.events import (
    AssociationFailure,
    EventBus,
    TrackCreated,
    TrackerEpochAdvanced,
    TrackingWarning,
    TrackLost,
    TrackRecovered,
    TrackTerminated,
    TrackUpdated,
)
from ...kernel.metrics import MetricName, MetricsEngine
from .manager import TrackingManager

TRACKING_ENGINE_ID = ModuleId("tracking_engine")


@dataclass(frozen=True, slots=True)
class TrackingOutcome:
    """What one frame did to a camera's tracks.

    ``failed`` distinguishes "tracking could not run" from "tracking ran and
    there is nothing here". Conflating them would make a broken tracker look
    like an empty scene (invariant V8).
    """

    camera_id: CameraId
    frame_ref: str
    tracker_epoch: int
    tracks: tuple[Track, ...] = ()
    created: int = 0
    terminated: int = 0
    recovered: int = 0
    coasting: int = 0
    failed: bool = False
    reason: str = ""
    latency_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.tracks)

    @property
    def confirmed(self) -> tuple[Track, ...]:
        return tuple(t for t in self.tracks if t.state is TrackState.CONFIRMED)


class TrackingEngine:
    """Converts standardized detections into standardized tracked objects."""

    def __init__(
        self,
        *,
        clock: Clock,
        bus: EventBus,
        metrics: MetricsEngine,
        manager: TrackingManager,
        config: TrackingSection,
    ) -> None:
        self._clock = clock
        self._bus = bus
        self._metrics = metrics
        self._manager = manager
        self._config = config

        self._last_time: dict[CameraId, Instant] = {}
        self._frames = 0
        self._failures = 0
        self._out_of_order = 0
        self._degraded_reason = ""

    # --- the public surface -------------------------------------------------- #

    def track(self, outcome: DetectionOutcome) -> TrackingOutcome:
        """Advance tracking for one frame. **Never raises.**

        A detection failure is *not* a tracking failure: when the detector could
        not look, tracking still ages its tracks, because an unmeasured frame is
        exactly when a track should coast. Skipping the frame would freeze every
        track for the duration of a detector outage and then resume as though no
        time had passed.
        """
        camera_id = outcome.frame_ref.camera_id
        started = self._clock.monotonic().ns

        try:
            update = self._run(outcome)
        except OutOfOrderFrameError as exc:
            # A pipeline bug. Loud, counted, alarmed — never absorbed (T1).
            self._out_of_order += 1
            self._metrics.counter(
                MetricName.TRACKING_OUT_OF_ORDER, camera_id=str(camera_id)
            ).increment()
            self._bus.publish(
                TrackingWarning(
                    occurred_at=self._clock.now(),
                    camera_id=camera_id,
                    reason="out_of_order_frame",
                    detail=str(exc),
                )
            )
            return self._failure(outcome, "out_of_order_frame", started)
        except TrackerCapacityError as exc:
            self._metrics.counter(
                MetricName.TRACKER_CAPACITY_REFUSALS, camera_id=str(camera_id)
            ).increment()
            return self._failure(outcome, f"capacity: {exc}", started)
        except (TrackerContractError, TrackingError) as exc:
            return self._degrade(outcome, exc, started)
        except Exception as exc:  # noqa: BLE001 - the engine is a firewall
            return self._degrade(outcome, exc, started)

        self._frames += 1
        latency_ms = (self._clock.monotonic().ns - started) / 1_000_000
        self._record(camera_id, update, latency_ms)
        self._publish(camera_id, update)

        return TrackingOutcome(
            camera_id=camera_id,
            frame_ref=str(outcome.frame_ref),
            tracker_epoch=update.tracker_epoch,
            tracks=update.active,
            created=len(update.new),
            terminated=len(update.terminated),
            recovered=len(update.recovered),
            coasting=len(update.coasting),
            latency_ms=latency_ms,
        )

    def tracks(self, camera_id: CameraId) -> Sequence[Track]:
        """Current live tracks. Never includes terminated ones."""
        try:
            return self._manager.tracker.tracks(camera_id)
        except Exception:  # noqa: BLE001 - a read must not raise
            return ()

    def reset(self, camera_id: CameraId, reason: str) -> TrackerEpoch:
        """Discard a camera's tracker state and publish the discontinuity.

        Consumers must *see* a reset. Without the event, every track vanishing
        at once and new ids appearing reads downstream as the entire scene
        teleporting (03_MODULES M6 failure handling).
        """
        previous = self._epoch_of(camera_id)
        discarded = len(self.tracks(camera_id))
        epoch = self._manager.tracker.reset(camera_id, reason)
        self._last_time.pop(camera_id, None)

        self._metrics.counter(
            MetricName.TRACKER_EPOCH_RESETS, camera_id=str(camera_id)
        ).increment()
        self._bus.publish(
            TrackerEpochAdvanced(
                occurred_at=self._clock.now(),
                camera_id=camera_id,
                previous_epoch=previous,
                epoch=epoch,
                reason=reason,
                discarded_tracks=discarded,
            )
        )
        return epoch

    def health(self) -> ComponentHealth:
        state = HealthState.HEALTHY
        detail = "tracking"
        if self._manager.is_fallback:
            state = HealthState.DEGRADED
            detail = f"running on fallback tracker: {self._manager.fallback_reason}"
        elif self._degraded_reason:
            state = HealthState.DEGRADED
            detail = self._degraded_reason
        return ComponentHealth(
            component_id=TRACKING_ENGINE_ID,
            state=state,
            reported_at=self._clock.now(),
            detail=detail,
            metrics={
                "frames": float(self._frames),
                "failures": float(self._failures),
                "out_of_order": float(self._out_of_order),
            },
        )

    # --- internals ------------------------------------------------------------ #

    def _run(self, outcome: DetectionOutcome) -> TrackUpdate:
        camera_id = outcome.frame_ref.camera_id
        now = self._clock.now()
        elapsed = self._elapsed(camera_id, now)

        request = TrackingRequest(
            camera_id=camera_id,
            frame_ref=outcome.frame_ref,
            timestamp=now,
            elapsed=elapsed,
            detections=outcome.detections,
            embeddings=None,
        )
        update = self._manager.tracker.update(request)
        self._verify(update, request)
        self._last_time[camera_id] = now
        return update

    def _elapsed(self, camera_id: CameraId, now: Instant) -> Duration:
        """Real elapsed time since this camera's previous processed frame.

        Port obligation T2. The scheduler drops frames by design, so a motion
        model integrating over frame count produces velocities whose meaning
        changes with system load.
        """
        previous = self._last_time.get(camera_id)
        if previous is None:
            return Duration(0)
        return Duration(max(0, now.ns - previous.ns))

    def _verify(self, update: TrackUpdate, request: TrackingRequest) -> None:
        """Check the adapter's output before publishing it.

        The tracking analogue of Flow 2's normalizer. An adapter that satisfies
        the interface but breaks an obligation produces plausible, wrong output,
        and the platform is the only place that can catch it before it reaches
        consumers.
        """
        if update.camera_id != request.camera_id:
            raise TrackerContractError(
                f"tracker returned an update for camera {update.camera_id} when "
                f"asked about {request.camera_id} (port obligation T7)"
            )

        seen: set[TrackId] = set()
        for track in update.active:
            if track.track_id in seen:
                raise TrackerContractError(
                    f"tracker returned duplicate track id {track.track_id} in one "
                    f"frame; ids are unique within an epoch (port obligation T3)"
                )
            seen.add(track.track_id)

            if track.camera_id != request.camera_id:
                raise TrackerContractError(
                    f"track {track.track_id} belongs to camera {track.camera_id} but "
                    f"was returned for {request.camera_id}; no cross-camera state may "
                    f"exist in this port (port obligation T7)"
                )
            if track.state is TrackState.TERMINATED:
                raise TrackerContractError(
                    f"track {track.track_id} is terminated but was returned as active"
                )
            # T5: the single check that catches a predicted position sold as a
            # measurement — the corruption V8 exists to prevent, and one no
            # consumer could detect on its own.
            if track.state.is_predicted and (
                track.measurement_basis is MeasurementBasis.MEASURED
            ):
                raise TrackerContractError(
                    f"track {track.track_id} is {track.state.value} but reports a "
                    f"MEASURED position; coasting must be explicitly marked "
                    f"(port obligation T5)"
                )

        for track_id, break_reason in update.terminated:
            if break_reason is BreakReason.NONE:
                raise TrackerContractError(
                    f"track {track_id} terminated without a break_reason "
                    f"(port obligation T6)"
                )

    def _record(self, camera_id: CameraId, update: TrackUpdate, latency_ms: float) -> None:
        label = str(camera_id)
        self._metrics.gauge(MetricName.TRACKS_ACTIVE, camera_id=label).set(
            float(update.active_count)
        )
        self._metrics.gauge(MetricName.TRACKS_COASTING, camera_id=label).set(
            float(len(update.coasting))
        )
        self._metrics.histogram(MetricName.TRACKING_LATENCY_MS, camera_id=label).record(
            latency_ms
        )
        self._metrics.counter(
            MetricName.TRACKING_FRAMES_PROCESSED, camera_id=label
        ).increment()

        if update.new:
            self._metrics.counter(MetricName.TRACKS_CREATED, camera_id=label).increment(
                len(update.new)
            )
        if update.recovered:
            self._metrics.counter(
                MetricName.TRACKS_RECOVERED, camera_id=label
            ).increment(len(update.recovered))

        for _track_id, break_reason in update.terminated:
            # Labelled by reason: a rise concentrated in `detector_miss` points
            # at the detector, not the tracker (02_VOM section 10.5).
            self._metrics.counter(
                MetricName.TRACKS_TERMINATED,
                camera_id=label,
                break_reason=break_reason.value,
            ).increment()

        if latency_ms > self._config.slow_frame_warn_ms:
            self._metrics.counter(
                MetricName.TRACKING_FAILURES, camera_id=label, reason="slow"
            ).increment()

    def _publish(self, camera_id: CameraId, update: TrackUpdate) -> None:
        """Turn transitions into events. Every transition is observable."""
        now = self._clock.now()
        by_id = {track.track_id: track for track in update.active}

        for track_id in update.new:
            track = by_id.get(track_id)
            self._bus.publish(
                TrackCreated(
                    occurred_at=now,
                    camera_id=camera_id,
                    track_id=str(track_id),
                    tracker_epoch=update.tracker_epoch,
                    class_id=str(track.class_id) if track else "",
                    frame_ref=str(update.frame_ref),
                )
            )

        for track_id in update.recovered:
            track = by_id.get(track_id)
            self._bus.publish(
                TrackRecovered(
                    occurred_at=now,
                    camera_id=camera_id,
                    track_id=str(track_id),
                    coasted_frames=track.coast_frames if track else 0,
                    previous_state=TrackState.COASTING.value,
                )
            )

        for track_id in update.coasting:
            track = by_id.get(track_id)
            if track is not None and track.state is TrackState.LOST:
                self._bus.publish(
                    TrackLost(
                        occurred_at=now,
                        camera_id=camera_id,
                        track_id=str(track_id),
                        coasted_frames=track.coast_frames,
                        break_reason=track.break_reason.value,
                    )
                )

        for track_id, break_reason in update.terminated:
            self._bus.publish(
                TrackTerminated(
                    occurred_at=now,
                    camera_id=camera_id,
                    track_id=str(track_id),
                    break_reason=break_reason.value,
                )
            )

        for association in update.associations:
            track = by_id.get(association.track_id)
            if track is None:
                continue
            self._bus.publish(
                TrackUpdated(
                    occurred_at=now,
                    camera_id=camera_id,
                    track_id=str(association.track_id),
                    state=track.state.value,
                    association_confidence=association.confidence.value,
                    measurement_basis=track.measurement_basis.value,
                )
            )
        # A refused association produces no ``Association`` and usually no live
        # track either — the track is terminated in the same frame. Reporting
        # only what *was* associated would therefore hide exactly the cases M6
        # cares most about: the near-ties the tracker declined rather than risk
        # an ID switch. *"The tracker never hides uncertainty to look clean."*
        for refusal in update.refused:
            self._metrics.counter(
                MetricName.ASSOCIATION_FAILURES, camera_id=str(camera_id)
            ).increment()
            self._bus.publish(
                AssociationFailure(
                    occurred_at=now,
                    camera_id=camera_id,
                    track_id=str(refusal.track_id),
                    best_cost=refusal.best_cost,
                    runner_up_cost=refusal.runner_up_cost,
                    margin=refusal.margin,
                )
            )

    def _degrade(self, outcome: DetectionOutcome, exc: Exception, started: int) -> TrackingOutcome:
        """A tracker failed. Fall back rather than stop (V9).

        03_MODULES M6: *"Fall back to a trivial IoU tracker (always available,
        no model needed) so the pipeline degrades rather than stops."*
        """
        camera_id = outcome.frame_ref.camera_id
        reason = f"{type(exc).__name__}: {exc}"
        self._degraded_reason = reason
        self._metrics.counter(
            MetricName.TRACKING_FAILURES, camera_id=str(camera_id), reason="adapter"
        ).increment()
        self._manager.fall_back(reason)
        self._bus.publish(
            TrackingWarning(
                occurred_at=self._clock.now(),
                camera_id=camera_id,
                reason="tracker_failure",
                detail=reason,
            )
        )
        return self._failure(outcome, reason, started)

    def _failure(
        self, outcome: DetectionOutcome, reason: str, started: int
    ) -> TrackingOutcome:
        self._failures += 1
        return TrackingOutcome(
            camera_id=outcome.frame_ref.camera_id,
            frame_ref=str(outcome.frame_ref),
            tracker_epoch=self._epoch_of(outcome.frame_ref.camera_id),
            failed=True,
            reason=reason,
            latency_ms=(self._clock.monotonic().ns - started) / 1_000_000,
        )

    def _epoch_of(self, camera_id: CameraId) -> int:
        live = self.tracks(camera_id)
        return live[0].tracker_epoch if live else 0

    @property
    def frames_processed(self) -> int:
        return self._frames

    @property
    def failures(self) -> int:
        return self._failures

    @property
    def out_of_order_frames(self) -> int:
        return self._out_of_order
