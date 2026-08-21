"""The Camera object (02_VISION_OBJECT_MODEL §10.1).

A stable viewpoint, so that everything observed through it stays interpretable
years later.

``camera_id`` stability is a hard rule: it is the partition key for state, the
prefix of every ``FrameRef``, and the join key for years of history. Physically
replacing a camera keeps the id; moving it to a new viewpoint mints a new one and
a new calibration, because the old history no longer describes the same view.

Credentials are **references** into a secret store, never values. A Camera record
is written to config repositories, logs, diagnostics, and support bundles, and
must be safe in all of them (12_SECURITY §9.1).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from .ids import (
    CalibrationId,
    CameraId,
    PrivacyPolicyId,
    ProfileId,
    RegionId,
    SiteId,
    TenantId,
)
from .space import Calibration


class SourceSemantics(enum.Enum):
    """Declared behaviour of a source (01_LAYERED §5.3).

    This single property changes the behaviour of the Scheduler, the Buffer, and
    the Runtime clock, and is the mechanism by which one pipeline serves both
    production streaming and deterministic replay. Nothing else branches on it.
    """

    REALTIME = "realtime"
    """Live. Latency is protected; frames are dropped to meet budget."""

    ARCHIVAL = "archival"
    """Recorded. Completeness is protected; the producer blocks. Reproducible."""

    DISCRETE = "discrete"
    """One-shot images. Immediate, complete, reproducible."""

    @property
    def may_drop_frames(self) -> bool:
        return self is SourceSemantics.REALTIME

    @property
    def is_deterministic(self) -> bool:
        return self is not SourceSemantics.REALTIME


class CameraStatus(enum.Enum):
    PROVISIONED = "provisioned"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    DEGRADED = "degraded"
    BLIND = "blind"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """How to reach a source. Never contains a secret value."""

    uri: str
    transport: str
    credential_ref: str | None = None
    """A *reference* resolved against the secret provider at connect time."""

    options: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        lowered = self.uri.lower()
        if "@" in lowered and "://" in lowered:
            authority = lowered.split("://", 1)[1].split("/", 1)[0]
            if ":" in authority.split("@", 1)[0]:
                raise ValueError(
                    "inline credentials in source URI are forbidden; use credential_ref"
                )


@dataclass(frozen=True, slots=True)
class NativeProfile:
    """What the source natively delivers."""

    width: int
    height: int
    fps: float
    codec: str
    colour_space: str = "bgr24"


@dataclass(frozen=True, slots=True)
class PipelineProfile:
    """How a camera should be processed.

    Resource- and capability-shaped only. ``priority_class`` is an **opaque**
    label the platform orders by and never interprets: "process the kitchen more
    often because it matters more" is a business priority and belongs to the
    consumer (03_MODULES M3).
    """

    profile_id: ProfileId
    target_fps: float
    max_in_flight: int = 4
    priority_class: str = "default"
    inference_width: int = 640
    inference_height: int = 640

    def __post_init__(self) -> None:
        if self.target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {self.target_fps}")
        if self.max_in_flight < 1:
            raise ValueError(f"max_in_flight must be >= 1, got {self.max_in_flight}")


@dataclass(frozen=True, slots=True)
class Camera:
    """The authoritative record for one viewpoint."""

    camera_id: CameraId
    tenant_id: TenantId
    site_id: SiteId
    source_spec: SourceSpec
    source_semantics: SourceSemantics
    native_profile: NativeProfile
    pipeline_profile: PipelineProfile
    calibration: Calibration | None = None
    privacy_policy_id: PrivacyPolicyId | None = None
    region_ids: tuple[RegionId, ...] = ()
    status: CameraStatus = CameraStatus.PROVISIONED
    labels: dict[str, str] = field(default_factory=dict)
    """Opaque operational tags (rack=A3). No platform logic may branch on these —
    the pressure valve that stops operational metadata becoming domain logic."""

    @property
    def calibration_id(self) -> CalibrationId | None:
        return self.calibration.calibration_id if self.calibration else None

    def with_status(self, status: CameraStatus) -> Camera:
        """Return a copy with a new status. Records are immutable."""
        return Camera(
            camera_id=self.camera_id,
            tenant_id=self.tenant_id,
            site_id=self.site_id,
            source_spec=self.source_spec,
            source_semantics=self.source_semantics,
            native_profile=self.native_profile,
            pipeline_profile=self.pipeline_profile,
            calibration=self.calibration,
            privacy_policy_id=self.privacy_policy_id,
            region_ids=self.region_ids,
            status=status,
            labels=dict(self.labels),
        )

    def with_calibration(self, calibration: Calibration) -> Camera:
        return Camera(
            camera_id=self.camera_id,
            tenant_id=self.tenant_id,
            site_id=self.site_id,
            source_spec=self.source_spec,
            source_semantics=self.source_semantics,
            native_profile=self.native_profile,
            pipeline_profile=self.pipeline_profile,
            calibration=calibration,
            privacy_policy_id=self.privacy_policy_id,
            region_ids=self.region_ids,
            status=self.status,
            labels=dict(self.labels),
        )
