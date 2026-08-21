"""Rehydrating an observation from its durable form.

Separated from the store so the round trip is testable without touching a
filesystem, and so the **loss** is documented in one place rather than implied by
what the encoder happened to omit.

**What survives a round trip**, and why these and not more: identity, time,
subject, position, coverage, lifecycle transition, content, provenance and
lineage. 07_STATE §9 uses the log for two things — rebuilding a projection and
answering *"what did the system report at 09:14 and why"* — and both need exactly
this set.

Position and coverage are here because **the model requires them**. 02_VOM
refuses to construct a presence observation with no position, so a decoder that
dropped it produced ``None`` for every record and the log read back empty — the
precise failure §9.1 exists to prevent, arrived at by being *too* conservative
about what to keep. A normalized box is four floats; there was never a size
argument for omitting it.

**What does not survive**: `QualityGrades`, the decision path, the identity
assertion and the evidence body. Each is either large, reconstructable from its
own store, or an M13 adapter's concern, and none of them is required for an
observation to be constructed.

A deployment needing full-fidelity replay binds a richer P20 adapter. The port is
where that choice belongs.
"""

from __future__ import annotations

import json

from ...core.model.confidence import Confidence, ConfidenceSemantics
from ...core.model.ids import (
    AttributeKey,
    CalibrationId,
    CameraId,
    ClassId,
    ConfigRevision,
    EvidenceId,
    FrameRef,
    FrameSeq,
    ModuleId,
    ObjectId,
    ObservationId,
    SiteId,
    StreamEpoch,
    TenantId,
)
from ...core.model.observation import (
    CoverageWindow,
    EvidenceRef,
    LifecycleTransition,
    MeasurementBasis,
    ObservabilityReason,
    ObservabilityStatus,
    Observation,
    ObservationType,
)
from ...core.model.provenance import Provenance
from ...core.model.space import (
    Box,
    Ellipse,
    FrameOfReference,
    Point,
    SpatialInfo,
)
from ...core.model.timebase import ClockQuality, Duration, Instant
from ...core.model.understanding import Timing
from ...core.model.visual_object import Attribute, LifecycleState


def decode_observation(line: str) -> Observation | None:
    """Parse one JSON Lines record. ``None`` when it cannot be reconstructed.

    ``None`` rather than an exception: a rebuild reading a log written by an
    older schema must skip what it cannot read and continue. Halting would make
    an old log unrebuildable, which is the opposite of what a log is for.
    """
    try:
        record = json.loads(line)
    except ValueError:
        return None

    try:
        return _build(record)
    except (KeyError, ValueError, TypeError):
        return None


def _build(record: dict) -> Observation:
    camera_id = CameraId(record["camera_id"])
    return Observation(
        observation_id=ObservationId(record["observation_id"]),
        observation_type=ObservationType(record["observation_type"]),
        tenant_id=TenantId(record["tenant_id"]),
        site_id=SiteId(record["site_id"]),
        camera_id=camera_id,
        frame_ref=_frame_ref(record["frame_ref"], camera_id),
        t_capture=Instant(record["t_capture_ns"]),
        t_capture_unc=Duration(record.get("t_capture_unc_ns", 0)),
        clock_quality=_clock_quality(record.get("clock_quality")),
        t_published=Instant(record.get("t_published_ns", record["t_capture_ns"])),
        provenance=_provenance(record["provenance"]),
        timing=Timing(total_ms=0.01),
        object_id=ObjectId(record["object_id"]) if record.get("object_id") else None,
        class_id=ClassId(record["class_id"]) if record.get("class_id") else None,
        taxonomy_version=record.get("taxonomy_version", ""),
        lifecycle_state=(
            LifecycleState(record["lifecycle_state"])
            if record.get("lifecycle_state")
            else None
        ),
        confidence=_confidence(record.get("confidence")),
        attributes=tuple(
            _attribute(raw, record["provenance"]) for raw in record.get("attributes", ())
        ),
        measurement_basis=MeasurementBasis(record.get("measurement_basis", "measured")),
        evidence_ref=(
            EvidenceRef(evidence_id=EvidenceId(record["evidence_id"]))
            if record.get("evidence_id")
            else None
        ),
        coverage=_coverage(record.get("coverage")),
        lifecycle_transition=_transition(record.get("lifecycle_transition")),
        identity=None,
        spatial=_spatial(record.get("spatial")),
        supersedes=(
            ObservationId(record["supersedes"]) if record.get("supersedes") else None
        ),
        lineage=tuple(ObservationId(o) for o in record.get("lineage", ())),
        schema_version=record.get("schema_version", "1.0.0"),
    )


def _spatial(record: dict | None) -> SpatialInfo | None:
    """Rebuild the position.

    Not optional decoration: 02_VOM requires a presence or spatial observation to
    carry one, so a decoder that returned ``None`` here would make every such
    record unconstructable and the log would read back empty.
    """
    if not record:
        return None
    box = record.get("bbox")
    point = record.get("ground_point")
    ellipse = record.get("ground_uncertainty")
    return SpatialInfo(
        frame_of_reference=FrameOfReference(record["frame_of_reference"]),
        bbox=Box(box[0], box[1], box[2], box[3]) if box else None,
        calibration_id=(
            CalibrationId(record["calibration_id"])
            if record.get("calibration_id")
            else None
        ),
        ground_point=Point(point[0], point[1]) if point and ellipse else None,
        ground_uncertainty=(
            Ellipse(ellipse[0], ellipse[1], ellipse[2]) if point and ellipse else None
        ),
    )


def _coverage(record: dict | None) -> CoverageWindow | None:
    """Rebuild the observability window (07_STATE §7.3)."""
    if not record:
        return None
    return CoverageWindow(
        status=ObservabilityStatus(record["status"]),
        reason=ObservabilityReason(record["reason"]),
        since=Instant(record["since_ns"]),
        until=Instant(record["until_ns"]) if record.get("until_ns") else None,
        effective_rate=record.get("effective_rate", 1.0),
        regions_affected=tuple(record.get("regions_affected", ())),
        capability_gaps=tuple(
            tuple(gap) for gap in record.get("capability_gaps", ())
        ),
    )


def _transition(record: dict | None) -> LifecycleTransition | None:
    if not record:
        return None
    return LifecycleTransition(
        previous=LifecycleState(record["previous"]),
        current=LifecycleState(record["current"]),
        trigger=record.get("trigger", ""),
    )


def _frame_ref(text: str, camera_id: CameraId) -> FrameRef:
    """Parse ``camera/eN/fM`` back into a typed reference.

    Parsing a stringified reference is exactly the hazard Flow 5 removed from the
    registry seam, and it is acceptable here only because this *is* the
    deserialization boundary — the point at which text becomes types is the one
    place text is the honest input.
    """
    parts = text.split("/")
    epoch = int(parts[1][1:]) if len(parts) > 1 and parts[1].startswith("e") else 0
    seq = int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith("f") else 0
    return FrameRef(camera_id, StreamEpoch(epoch), FrameSeq(seq))


def _clock_quality(value: str | None) -> ClockQuality:
    if not value:
        return ClockQuality.UNKNOWN
    for quality in ClockQuality:
        if quality.value[0] == value:
            return quality
    # An unknown quality string from a future schema degrades to UNKNOWN, which
    # carries maximal uncertainty — the honest reading of a value this build
    # cannot interpret.
    return ClockQuality.UNKNOWN


def _confidence(raw: dict | None) -> Confidence | None:
    if not raw:
        return None
    return Confidence(
        value=float(raw["value"]),
        semantics=ConfidenceSemantics(raw["semantics"]),
        calibrated=False,
        raw_score=float(raw["value"]),
    )


def _attribute(raw: dict, provenance: dict) -> Attribute:
    return Attribute(
        key=AttributeKey(raw["key"]),
        schema_version=raw.get("schema_version", "1.0.0"),
        value=raw["value"],
        confidence=Confidence(
            value=float(raw["confidence"]),
            semantics=ConfidenceSemantics(raw["confidence_semantics"]),
            calibrated=False,
            raw_score=float(raw["confidence"]),
        ),
        observed_at=Instant(raw["observed_at_ns"]),
        producer=_provenance(provenance),
    )


def _provenance(raw: dict) -> Provenance:
    return Provenance(
        producer_module=ModuleId(raw["producer_module"]),
        producer_version=raw.get("producer_version", "1.0.0"),
        config_revision=ConfigRevision(raw["config_revision"]),
        model_id=raw.get("model_id"),
        model_artifact_hash=raw.get("model_artifact_hash"),
    )
