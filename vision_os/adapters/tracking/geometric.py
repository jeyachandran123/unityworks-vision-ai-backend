"""The shared geometric tracking core, behind P9.

Three shipped adapters are thin configurations of this one engine:

``tracker.iou``
    Pure geometry, no motion model, single-stage association. The **universal
    fallback** — no weights, no device, cannot become unavailable
    (10_RELIABILITY section 7.3).

``tracker.sort``
    Adds linear motion prediction and optimal assignment.

``tracker.bytetrack``
    Adds a second association stage over low-confidence detections.

Sharing one core is deliberate. Lifecycle correctness, id uniqueness, bounded
memory and honest coasting are obligations *every* tracker owes (T3, T5, T6, T8),
and re-implementing them per adapter is how one adapter ends up quietly violating
one of them. What differs between trackers is which signals populate the cost
matrix and how the matrix is resolved — so that is what varies here, and nothing
else.

**The platform never imports this module.** It holds ``TrackerPort``; only the
composition root names a concrete tracker.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...core.errors import OutOfOrderFrameError, TrackerCapacityError
from ...core.model.confidence import Confidence, ConfidenceSemantics
from ...core.model.detection import Detection
from ...core.model.ids import (
    AdapterId,
    CameraId,
    ConfigRevision,
    ModuleId,
    TrackerEpoch,
    TrackId,
)
from ...core.model.provenance import Provenance
from ...core.model.space import Box, FrameOfReference, Point, SpatialInfo
from ...core.model.timebase import Duration, Instant
from ...core.model.track import (
    Association,
    AssociationMethod,
    BreakReason,
    MeasurementBasis,
    MotionEstimate,
    MotionState,
    RefusedAssociation,
    Track,
    TrackEvidence,
    TrackState,
    TrackUpdate,
)
from ...core.ports.tracking import (
    AssignmentResult,
    AssociationPort,
    MotionObservation,
    MotionPredictorPort,
    Prediction,
    TrackerCapabilities,
    TrackingRequest,
)
from ...perception.tracking.association import (
    AssociationPolicy,
    CostMatrixBuilder,
    GreedyAssociator,
)
from ...perception.tracking.lifecycle import (
    LifecycleMachine,
    LifecyclePolicy,
)
from ...perception.tracking.table import TrackRecord, TrackTable
from .motion import (
    VELOCITY_NOISE_FLOOR,
    LinearPredictor,
    StationaryPredictor,
    heading_of,
    speed_of,
)

TRACKER_MODULE = ModuleId("tracking_engine")

#: Displacement per second below which an object is called stationary. Above the
#: velocity noise floor so detector jitter alone cannot make a parked car "move".
MOTION_FLOOR = 0.01

#: Consecutive frames of consistent behaviour before the motion state flips.
#: Hysteresis: one still frame does not make a walking person stationary.
MOTION_HYSTERESIS = 3

#: Heading reversals within the retained window that make motion "erratic".
ERRATIC_DIRECTION_CHANGES = 3


@dataclass(frozen=True, slots=True)
class GeometricConfig:
    """What distinguishes one geometric tracker from another."""

    tracker_id: str
    version: str = "1.0.0"
    use_motion_model: bool = True
    two_stage: bool = False
    """Second association pass over low-confidence detections. The idea behind
    ByteTrack: a detection too weak to start a track is often still strong
    enough to *continue* one, and using it materially reduces fragmentation
    through partial occlusion."""

    low_confidence_floor: float = 0.1
    high_confidence_floor: float = 0.5
    """Detections at or above this may start new tracks. Below it they may only
    continue existing ones (two-stage mode)."""

    handles_occlusion: str = "short"


class GeometricTracker:
    """A ``TrackerPort`` implementation over pure geometry and motion.

    **Per-camera state, strictly sequential** (T1, T7). One table per camera,
    no cross-camera structure of any kind — a fact the conformance kit verifies
    by driving two cameras and asserting their id spaces stay independent.
    """

    def __init__(
        self,
        *,
        config: GeometricConfig,
        lifecycle: LifecyclePolicy | None = None,
        association: AssociationPolicy | None = None,
        associator: AssociationPort | None = None,
        config_revision: str = "unset",
        history_length: int = 32,
    ) -> None:
        self._config = config
        self._lifecycle_policy = lifecycle or LifecyclePolicy()
        self._association_policy = association or AssociationPolicy()
        self._machine = LifecycleMachine(self._lifecycle_policy)
        self._matrix = CostMatrixBuilder(self._association_policy)
        self._associator = associator or GreedyAssociator()
        self._config_revision = config_revision
        self._history_length = history_length

        self._tables: dict[CameraId, TrackTable] = {}
        self._last_frame: dict[CameraId, tuple[int, int]] = {}
        self._last_time: dict[CameraId, Instant] = {}
        self._epochs: dict[CameraId, TrackerEpoch] = {}

    # --- port surface ------------------------------------------------------ #

    def capabilities(self) -> TrackerCapabilities:
        return TrackerCapabilities(
            tracker_id=self._config.tracker_id,
            version=self._config.version,
            requires_embeddings=False,
            handles_occlusion=self._config.handles_occlusion,
            max_objects=self._lifecycle_policy.max_tracks_per_camera,
            supports_ground_plane_tracking=False,
            deterministic=True,
            state_per_camera_bytes=self._lifecycle_policy.max_tracks_per_camera * 512,
        )

    def tracks(self, camera_id: CameraId) -> Sequence[Track]:
        table = self._tables.get(camera_id)
        if table is None:
            return ()
        now = self._last_time.get(camera_id, Instant(0))
        return tuple(self._project(record, now) for record in table.records())

    def reset(self, camera_id: CameraId, reason: str) -> TrackerEpoch:
        """Discard state and mint a new epoch (T7)."""
        epoch = TrackerEpoch(self._epochs.get(camera_id, TrackerEpoch(0)) + 1)
        self._epochs[camera_id] = epoch
        table = self._tables.get(camera_id)
        if table is not None:
            table.reset(epoch)
        self._last_frame.pop(camera_id, None)
        self._last_time.pop(camera_id, None)
        return epoch

    def update(self, request: TrackingRequest) -> TrackUpdate:
        """Advance one frame. The whole tracker in one call."""
        camera_id = request.camera_id
        self._assert_ordering(camera_id, request)

        table = self._table_for(camera_id)
        elapsed = self._elapsed_for(camera_id, request)

        records = table.records()
        predictions = self._predict_all(records, elapsed)

        high, low = self._split_detections(request.detections)
        result = self._associate(records, predictions, request.detections, high)

        matched, refused = self._filter_ambiguous(result, records)

        if self._config.two_stage and low:
            matched, result = self._second_stage(
                records, predictions, request.detections, low, matched, result
            )

        update = self._apply(
            table=table,
            records=records,
            request=request,
            elapsed=elapsed,
            matched=matched,
            refused=refused,
            result=result,
            high=high,
        )

        self._last_frame[camera_id] = (
            request.frame_ref.stream_epoch,
            request.frame_ref.frame_seq,
        )
        self._last_time[camera_id] = request.timestamp
        return update

    # --- ordering ---------------------------------------------------------- #

    def _assert_ordering(self, camera_id: CameraId, request: TrackingRequest) -> None:
        """T1 — reject out-of-order frames loudly.

        An out-of-order frame integrates a negative time step, runs positions
        backwards, and degrades association in a way that looks like poor
        tracker quality rather than the pipeline bug it is. The architecture
        requires this be alarmed, never absorbed (03_MODULES M6).
        """
        previous = self._last_frame.get(camera_id)
        if previous is None:
            return
        current = (request.frame_ref.stream_epoch, request.frame_ref.frame_seq)
        if current <= previous:
            raise OutOfOrderFrameError(
                f"frame {request.frame_ref} arrived after epoch/seq {previous}; "
                f"per-camera ordering is a pipeline guarantee and its violation "
                f"corrupts tracking silently (port obligation T1)",
                camera_id=str(camera_id),
                received=str(request.frame_ref),
                last_processed=f"e{previous[0]}/f{previous[1]}",
            )

    def _elapsed_for(self, camera_id: CameraId, request: TrackingRequest) -> Duration:
        """Real elapsed time, never frame count (T2).

        Prefers the caller's measurement and falls back to the timestamp delta.
        The platform drops frames by design, so a per-frame step would make
        every velocity a function of system load.
        """
        if request.elapsed.ns > 0:
            return request.elapsed
        previous = self._last_time.get(camera_id)
        if previous is None:
            return Duration(0)
        return Duration(max(0, request.timestamp.ns - previous.ns))

    def _table_for(self, camera_id: CameraId) -> TrackTable:
        table = self._tables.get(camera_id)
        if table is None:
            table = TrackTable(
                camera_id,
                epoch=self._epochs.setdefault(camera_id, TrackerEpoch(0)),
                max_tracks=self._lifecycle_policy.max_tracks_per_camera,
                history_length=self._history_length,
            )
            self._tables[camera_id] = table
        return table

    # --- association ------------------------------------------------------- #

    def _split_detections(
        self, detections: Sequence[Detection]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Indices of high- and low-confidence detections.

        Single-stage trackers put everything above the low floor in ``high``, so
        the same code path serves both modes.
        """
        config = self._config
        high: list[int] = []
        low: list[int] = []
        for index, detection in enumerate(detections):
            score = detection.confidence.value
            if not config.two_stage:
                if score >= config.low_confidence_floor:
                    high.append(index)
                continue
            if score >= config.high_confidence_floor:
                high.append(index)
            elif score >= config.low_confidence_floor:
                low.append(index)
        return tuple(high), tuple(low)

    def _predict_all(
        self, records: Sequence[TrackRecord], elapsed: Duration
    ) -> tuple[Prediction, ...]:
        """Predict each track forward from its **last measurement**.

        The horizon is ``since_measurement + elapsed``, not ``elapsed``. A
        motion model extrapolates from the last position it observed, so a track
        coasting for five frames must be predicted five frames forward — using
        one frame's elapsed would anchor the gate to a stale position and lose
        every moving object across a gap, while working perfectly on stationary
        ones.
        """
        predictions: list[Prediction] = []
        for record in records:
            horizon = Duration(record.since_measurement_ns + elapsed.ns)
            try:
                predictions.append(record.predictor.predict(horizon))
            except ValueError:
                box = record.box
                predictions.append(Prediction(box.x1, box.y1, box.x2, box.y2))
        return tuple(predictions)

    def _associate(self, records, predictions, detections, indices):
        boxes = [detections[i].spatial.bbox for i in indices]
        scores = [detections[i].confidence.value for i in indices]
        candidates = self._matrix.build(
            predictions=predictions,
            detection_boxes=[b for b in boxes if b is not None],
            detection_scores=scores,
        )
        result = self._associator.assign(
            track_count=len(records),
            detection_count=len(indices),
            candidates=candidates,
            max_cost=self._association_policy.max_cost,
        )
        return _resolve(result, indices)

    def _filter_ambiguous(self, result, records):
        """Refuse matches whose margin over the runner-up is too small.

        03_MODULES M6: *"Prefer terminating a track over a wrong association"*
        and *"the tracker never hides uncertainty to look clean."* A track whose
        best and second-best candidates are nearly tied is exactly the ID-switch
        case, and coasting through it is recoverable where a wrong bind is not.

        Returns the surviving matches and a **record of each refusal**, carrying
        both costs. The refused track is usually terminated in the same frame, so
        this is the only place both numbers still exist to be reported.
        """
        margin = self._association_policy.ambiguity_margin
        if margin <= 0.0:
            return dict(result.matches), ()

        matched: dict[int, int] = {}
        refusals: list[RefusedAssociation] = []
        for track_index, detection_index in result.matches.items():
            runner_up = result.runner_up.get(track_index)
            won = result.costs.get((track_index, detection_index), 0.0)
            if runner_up is not None and (runner_up - won) < margin:
                refusals.append(
                    RefusedAssociation(
                        track_id=records[track_index].track_id,
                        best_cost=won,
                        runner_up_cost=runner_up,
                    )
                )
                continue
            matched[track_index] = detection_index
        return matched, tuple(refusals)

    def _second_stage(self, records, predictions, detections, low, matched, result):
        """Associate leftover tracks against low-confidence detections.

        A detection too weak to *start* a track is often strong enough to
        *continue* one — the observation ByteTrack is built on, and the largest
        single reduction in fragmentation through partial occlusion.
        """
        leftover = [i for i in range(len(records)) if i not in matched]
        if not leftover:
            return matched, result

        sub_predictions = [predictions[i] for i in leftover]
        second = self._associate(
            [records[i] for i in leftover], sub_predictions, detections, low
        )
        combined_costs = dict(result.costs)
        combined_runner = dict(result.runner_up)
        for local_index, detection_index in second.matches.items():
            track_index = leftover[local_index]
            matched[track_index] = detection_index
            combined_costs[(track_index, detection_index)] = second.costs.get(
                (local_index, detection_index), 0.0
            )
        return matched, _Resolved(matched, combined_costs, combined_runner)

    # --- state application -------------------------------------------------- #

    def _apply(
        self, *, table, records, request, elapsed, matched, refused, result, high
    ) -> TrackUpdate:
        now = request.timestamp
        detections = request.detections

        new_ids: list[TrackId] = []
        terminated: list[tuple[TrackId, BreakReason]] = []
        coasting: list[TrackId] = []
        recovered: list[TrackId] = []
        associations: list[Association] = []
        matched_detections = set(matched.values())
        refused_ids = {r.track_id for r in refused}

        for track_index, record in enumerate(records):
            detection_index = matched.get(track_index)
            if detection_index is not None:
                transition = self._on_hit(
                    record, detections[detection_index], elapsed, now, request, result,
                    track_index, detection_index,
                )
                associations.append(
                    Association(
                        track_id=record.track_id,
                        detection_index=detection_index,
                        confidence=Confidence(
                            value=record.association_confidence,
                            semantics=ConfidenceSemantics.ASSOCIATION,
                            raw_score=record.association_confidence,
                        ),
                        method=record.association_method,
                        cost=record.association_cost,
                    )
                )
                if transition.is_recovery:
                    recovered.append(record.track_id)
            else:
                break_reason = (
                    BreakReason.ASSOCIATION_FAILURE
                    if record.track_id in refused_ids
                    else self._miss_reason(record, detections)
                )
                transition = self._on_miss(record, now, break_reason, elapsed)

            if transition.is_terminal:
                record.break_reason = transition.break_reason
                terminated.append((record.track_id, transition.break_reason))
                table.remove(record.track_id)
            elif record.state is TrackState.COASTING or record.state is TrackState.LOST:
                coasting.append(record.track_id)

        for detection_index in high:
            if detection_index in matched_detections:
                continue
            created = self._spawn(table, detections[detection_index], now, request)
            if created is not None:
                new_ids.append(created.track_id)

        active = tuple(self._project(record, now) for record in table.records())

        return TrackUpdate(
            camera_id=request.camera_id,
            frame_ref=request.frame_ref,
            tracker_epoch=table.epoch,
            active=active,
            new=tuple(new_ids),
            terminated=tuple(terminated),
            coasting=tuple(coasting),
            recovered=tuple(recovered),
            associations=tuple(associations),
            refused=tuple(refused),
            unmatched_detections=tuple(
                i for i in range(len(detections)) if i not in matched_detections
            ),
        )

    def _on_hit(
        self, record, detection, elapsed, now, request, result, track_index, detection_index
    ):
        box = detection.spatial.bbox
        if box is None:
            raise ValueError("a detection reaching the tracker must carry a box")

        record.age_frames += 1
        record.hit_count += 1
        transition = self._machine.on_hit(
            state=record.state, hit_count=record.hit_count, age_frames=record.age_frames
        )
        record.state = transition.current
        record.box = box
        record.class_id = detection.class_id
        record.last_seen = now
        record.last_updated = now
        record.break_reason = BreakReason.NONE
        record.history.append(request.frame_ref)

        # The observation's elapsed is time since the last *measurement*, which
        # across a coast is longer than one frame. Feeding the frame delta here
        # would inflate the velocity estimate by the length of the gap.
        measured_gap = Duration(record.since_measurement_ns + elapsed.ns)
        record.predictor.observe(
            MotionObservation(box.x1, box.y1, box.x2, box.y2, measured_gap)
        )
        record.coast_frames = 0
        record.since_measurement_ns = 0

        cost = result.costs.get((track_index, detection_index), 0.0)
        runner_up = result.runner_up.get(track_index)
        record.association_cost = cost
        record.runner_up_cost = runner_up
        record.association_confidence = _confidence_from_cost(cost, runner_up)
        record.association_method = (
            AssociationMethod.MOTION_GATED_IOU
            if self._config.use_motion_model
            else AssociationMethod.IOU
        )
        self._update_motion_state(record, elapsed)
        return transition

    def _on_miss(self, record, now, break_reason, elapsed: Duration):
        record.age_frames += 1
        record.coast_frames += 1
        record.since_measurement_ns += elapsed.ns
        transition = self._machine.on_miss(
            state=record.state,
            coast_frames=record.coast_frames,
            age_frames=record.age_frames,
            break_reason=break_reason,
        )
        record.state = transition.current
        record.last_updated = now
        record.break_reason = transition.break_reason
        # Association confidence decays while coasting: a position five frames
        # unmeasured is a weaker claim than one measured this frame, and saying
        # so is the whole point of T5.
        record.association_confidence *= 0.7
        return transition

    def _miss_reason(self, record, detections) -> BreakReason:
        """Why this track was not measured.

        A track at the frame edge most likely left; one in open view with other
        detections present was probably occluded; otherwise the detector missed
        it. Diagnostic only — it never changes tracking behaviour, but it is what
        makes a regression attributable (02_VOM section 10.5).
        """
        box = record.box
        at_edge = box.x1 <= 0.02 or box.y1 <= 0.02 or box.x2 >= 0.98 or box.y2 >= 0.98
        if at_edge:
            return BreakReason.EXIT
        if detections:
            return BreakReason.OCCLUSION
        return BreakReason.DETECTOR_MISS

    def _spawn(self, table, detection, now, request):
        box = detection.spatial.bbox
        if box is None:
            return None

        # Tenancy comes from the detection, never from configuration: it is
        # declared per camera, so a node serving two tenants must carry each
        # track's own rather than stamping one value across all of them.
        def _create():
            return table.create(
                class_id=detection.class_id,
                box=box,
                predictor=self._new_predictor(),
                now=now,
                frame_ref=request.frame_ref,
                tenant_id=detection.tenant_id,
                site_id=detection.site_id,
            )

        try:
            return _create()
        except TrackerCapacityError:
            # Bounded by design (T8). Evict the weakest and retry once; if the
            # table is full of confirmed tracks, the new detection is refused
            # and counted rather than growing memory.
            if table.evict_weakest() is None:
                return None
            try:
                return _create()
            except TrackerCapacityError:
                return None

    def _new_predictor(self) -> MotionPredictorPort:
        return LinearPredictor() if self._config.use_motion_model else StationaryPredictor()

    def _update_motion_state(self, record: TrackRecord, elapsed: Duration) -> None:
        """Classify motion with hysteresis.

        Descriptive only. ``stationary`` is a fact about pixels — it is never
        ``loitering``, which is a judgment the Semantic Ceiling rejects (V1).
        """
        velocity = record.predictor.velocity()
        speed = speed_of(velocity)
        heading = heading_of(velocity)

        if heading is not None and record.last_heading is not None:
            difference = abs(heading - record.last_heading)
            difference = min(difference, 360.0 - difference)
            if difference > 90.0:
                record.direction_changes += 1
        if heading is not None:
            record.last_heading = heading

        if speed < MOTION_FLOOR:
            record.still_frames += 1
            record.moving_frames = 0
        else:
            record.moving_frames += 1
            record.still_frames = 0

        if record.direction_changes >= ERRATIC_DIRECTION_CHANGES:
            record.motion_state = MotionState.ERRATIC
        elif record.still_frames >= MOTION_HYSTERESIS:
            record.motion_state = MotionState.STATIONARY
        elif record.moving_frames >= MOTION_HYSTERESIS:
            record.motion_state = MotionState.MOVING
        elif record.hit_count < MOTION_HYSTERESIS:
            record.motion_state = MotionState.UNKNOWN

    # --- projection --------------------------------------------------------- #

    def _project(self, record: TrackRecord, now: Instant) -> Track:
        """Freeze a mutable record into the published immutable ``Track``."""
        predicted = record.state.is_predicted
        box = record.box
        if predicted:
            # Same horizon the association gate used, so the position a consumer
            # reads is the position the tracker actually believes — not the last
            # measurement wearing a "predicted" label.
            try:
                prediction = record.predictor.predict(
                    Duration(record.since_measurement_ns)
                )
                box = Box(
                    max(0.0, min(1.0, prediction.x1)),
                    max(0.0, min(1.0, prediction.y1)),
                    max(0.0, min(1.0, prediction.x2)),
                    max(0.0, min(1.0, prediction.y2)),
                )
            except (ValueError, TypeError):
                box = record.box

        velocity = record.predictor.velocity()
        acceleration = record.predictor.acceleration()
        speed = speed_of(velocity)

        motion = MotionEstimate(
            velocity=_point(velocity),
            acceleration=_point(acceleration) if acceleration is not None else None,
            heading_degrees=heading_of(velocity),
            speed=speed if speed >= VELOCITY_NOISE_FLOOR else 0.0,
            # Grows with time unmeasured, not with frames unmeasured: a
            # five-second gap is a weaker claim than five fast frames.
            uncertainty=(record.since_measurement_ns / 1_000_000_000) * 0.05,
        )

        return Track(
            track_id=record.track_id,
            camera_id=record.track_id.camera_id,
            tenant_id=record.tenant_id,
            site_id=record.site_id,
            state=record.state,
            class_id=record.class_id,
            confidence=Confidence(
                value=max(0.0, min(1.0, record.association_confidence)),
                semantics=ConfidenceSemantics.ASSOCIATION,
                raw_score=max(0.0, min(1.0, record.association_confidence)),
            ),
            spatial=SpatialInfo(frame_of_reference=FrameOfReference.NORMALIZED, bbox=box),
            measurement_basis=(
                MeasurementBasis.PREDICTED if predicted else MeasurementBasis.MEASURED
            ),
            motion=motion,
            motion_state=record.motion_state,
            first_seen=record.first_seen,
            last_seen=record.last_seen,
            last_updated=record.last_updated,
            age_frames=record.age_frames,
            hit_count=record.hit_count,
            coast_frames=record.coast_frames,
            detections=tuple(record.history),
            evidence=TrackEvidence(
                association_method=record.association_method,
                association_cost=record.association_cost,
                runner_up_cost=record.runner_up_cost,
                gated_candidates=record.gated_candidates,
            ),
            provenance=Provenance(
                producer_module=TRACKER_MODULE,
                producer_version="1.0.0",
                config_revision=ConfigRevision(self._config_revision),
                adapter_id=AdapterId(self._config.tracker_id),
                adapter_version=self._config.version,
                deterministic=True,
            ),
            break_reason=record.break_reason,
        )


def _point(vector: tuple[float, float]) -> Point:
    return Point(vector[0], vector[1])


def _confidence_from_cost(cost: float, runner_up: float | None) -> float:
    """Association confidence from cost and margin.

    Two signals, not one. A cheap match that barely beat its runner-up is *not*
    a confident association, and reporting only ``1 - cost`` would hide exactly
    the ambiguity that T4 requires be published.
    """
    base = max(0.0, min(1.0, 1.0 - cost))
    if runner_up is None:
        return base
    margin = max(0.0, runner_up - cost)
    # Full credit at a margin of 0.3; proportionally less below that.
    return base * max(0.0, min(1.0, margin / 0.3))


@dataclass(frozen=True, slots=True)
class _Resolved:
    """An assignment expressed in the frame's original detection indices.

    The association stage works over a *subset* of detections (high-confidence
    only, or low-confidence only on the second pass), so its indices are local
    to that subset. Translating once, here, keeps every downstream consumer
    working in one index space — mixing the two is a subtle and very
    hard-to-find source of wrong bindings.
    """

    matches: dict[int, int]
    costs: dict[tuple[int, int], float]
    runner_up: dict[int, float]


def _resolve(result: AssignmentResult, indices: Sequence[int]) -> _Resolved:
    return _Resolved(
        matches={t: indices[d] for t, d in result.matches},
        costs={(t, indices[d]): result.costs.get((t, d), 0.0) for t, d in result.matches},
        runner_up=dict(result.runner_up),
    )


