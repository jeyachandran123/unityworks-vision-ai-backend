"""M1 Camera Manager — know what each camera is; never touch its bytes.

Owns the identity, calibration, and operating profile of every viewpoint. It is
the platform's registry of viewpoints, not its connection layer: it never
connects, decodes, or streams.

Concurrency model is **read-mostly with copy-on-write snapshots**. Readers take
an immutable snapshot pointer; writers build a new version and swap atomically.
No reader ever blocks and no reader ever sees a torn record — which matters
because this module is consulted on the hot path (~3000 times a second at 100
cameras x 30 fps) while being written rarely (a recalibration a month).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace

from ...core.errors import NotFoundError, UncalibratedError, ValidationError
from ...core.model.camera import (
    Camera,
    CameraStatus,
    NativeProfile,
    PipelineProfile,
    SourceSemantics,
    SourceSpec,
)
from ...core.model.ids import (
    CalibrationId,
    CameraId,
    PrivacyPolicyId,
    ProfileId,
    RegionId,
    SiteId,
    TenantId,
)
from ...core.model.region import Region
from ...core.model.space import Calibration, Ellipse, FrameOfReference, Homography, Point, Polygon
from ...core.ports.clock import Clock
from ...kernel.config.schema import (
    CalibrationDeclaration,
    CameraDeclaration,
    ProfileDeclaration,
    RegionDeclaration,
)
from ...kernel.events import CameraChanged, EventBus, ViewpointDriftSuspected


@dataclass(frozen=True, slots=True)
class _Registry:
    """An immutable snapshot of the whole registry. Swapped atomically."""

    cameras: dict[CameraId, Camera]
    regions: dict[RegionId, Region]
    profiles: dict[ProfileId, PipelineProfile]
    calibration_history: dict[CameraId, tuple[Calibration, ...]]


class CameraManager:
    """The authoritative record for every viewpoint."""

    def __init__(self, *, clock: Clock, bus: EventBus) -> None:
        self._clock = clock
        self._bus = bus
        self._write_lock = threading.Lock()
        self._snapshot = _Registry(cameras={}, regions={}, profiles={}, calibration_history={})

    # --- provisioning ------------------------------------------------------ #

    def load_declarations(
        self,
        *,
        cameras: tuple[CameraDeclaration, ...],
        profiles: tuple[ProfileDeclaration, ...],
        regions: tuple[RegionDeclaration, ...],
    ) -> None:
        """Build the registry from validated configuration.

        Fails fast: an unresolvable profile or region reference is a provisioning
        error at startup, never a surprise at first frame.
        """
        with self._write_lock:
            profile_map = {ProfileId(p.profile_id): _build_profile(p) for p in profiles}
            region_map = {RegionId(r.region_id): _build_region(r) for r in regions}
            camera_map: dict[CameraId, Camera] = {}
            history: dict[CameraId, tuple[Calibration, ...]] = {}

            for declaration in cameras:
                camera = _build_camera(declaration, profile_map, region_map)
                camera_map[camera.camera_id] = camera
                if camera.calibration is not None:
                    history[camera.camera_id] = (camera.calibration,)

            self._snapshot = _Registry(
                cameras=camera_map,
                regions=region_map,
                profiles=profile_map,
                calibration_history=history,
            )

    def provision(self, camera: Camera) -> Camera:
        """Add or replace a single camera record."""
        with self._write_lock:
            current = self._snapshot
            for region_id in camera.region_ids:
                if region_id not in current.regions:
                    raise ValidationError(
                        f"camera '{camera.camera_id}' references undeclared region '{region_id}'"
                    )
            cameras = dict(current.cameras)
            cameras[camera.camera_id] = camera
            history = dict(current.calibration_history)
            if camera.calibration is not None:
                history[camera.camera_id] = (camera.calibration,)
            self._snapshot = replace(current, cameras=cameras, calibration_history=history)
        self._publish(camera.camera_id, "provisioned")
        return camera

    def retire(self, camera_id: CameraId, reason: str = "") -> None:
        with self._write_lock:
            current = self._snapshot
            camera = current.cameras.get(camera_id)
            if camera is None:
                raise NotFoundError(f"unknown camera '{camera_id}'", camera_id=str(camera_id))
            cameras = dict(current.cameras)
            cameras[camera_id] = camera.with_status(CameraStatus.RETIRED)
            self._snapshot = replace(current, cameras=cameras)
        self._publish(camera_id, f"retired:{reason}")

    # --- reads (hot path, lock-free) ---------------------------------------- #

    def get(self, camera_id: CameraId) -> Camera:
        camera = self._snapshot.cameras.get(camera_id)
        if camera is None:
            raise NotFoundError(f"unknown camera '{camera_id}'", camera_id=str(camera_id))
        return camera

    def try_get(self, camera_id: CameraId) -> Camera | None:
        return self._snapshot.cameras.get(camera_id)

    def list(
        self,
        *,
        tenant_id: TenantId | None = None,
        site_id: SiteId | None = None,
        status: CameraStatus | None = None,
    ) -> tuple[Camera, ...]:
        cameras = self._snapshot.cameras.values()
        return tuple(
            camera
            for camera in cameras
            if (tenant_id is None or camera.tenant_id == tenant_id)
            and (site_id is None or camera.site_id == site_id)
            and (status is None or camera.status is status)
        )

    def resolve_profile(self, camera_id: CameraId) -> PipelineProfile:
        """O(1) snapshot read with no allocation. Must never appear in a profile."""
        return self.get(camera_id).pipeline_profile

    def regions_of(self, camera_id: CameraId) -> tuple[Region, ...]:
        snapshot = self._snapshot
        camera = snapshot.cameras.get(camera_id)
        if camera is None:
            raise NotFoundError(f"unknown camera '{camera_id}'", camera_id=str(camera_id))
        return tuple(
            snapshot.regions[rid] for rid in camera.region_ids if rid in snapshot.regions
        )

    def get_calibration(self, camera_id: CameraId) -> Calibration:
        calibration = self.get(camera_id).calibration
        if calibration is None:
            raise UncalibratedError(
                f"camera '{camera_id}' has no calibration", camera_id=str(camera_id)
            )
        return calibration

    def calibration_history(self, camera_id: CameraId) -> tuple[Calibration, ...]:
        """Historical observations remain interpretable under their own version."""
        return self._snapshot.calibration_history.get(camera_id, ())

    # --- calibration -------------------------------------------------------- #

    def recalibrate(self, camera_id: CameraId, calibration: Calibration) -> CalibrationId:
        """Mint a new calibration version. Rejects a degenerate homography.

        The previous version stays in force if the candidate is invalid — a bad
        calibration must not blind a working camera.
        """
        if calibration.homography is not None and calibration.homography.is_degenerate():
            raise ValidationError(
                f"calibration '{calibration.calibration_id}' has a degenerate homography; "
                f"the previous calibration stays in force"
            )
        with self._write_lock:
            current = self._snapshot
            camera = current.cameras.get(camera_id)
            if camera is None:
                raise NotFoundError(f"unknown camera '{camera_id}'", camera_id=str(camera_id))
            cameras = dict(current.cameras)
            cameras[camera_id] = camera.with_calibration(calibration)
            history = dict(current.calibration_history)
            history[camera_id] = (*history.get(camera_id, ()), calibration)
            self._snapshot = replace(current, cameras=cameras, calibration_history=history)
        self._publish(camera_id, f"recalibrated:{calibration.calibration_id}")
        return calibration.calibration_id

    def project_to_ground(self, camera_id: CameraId, point: Point) -> tuple[Point, Ellipse]:
        """Project a normalized image point onto the ground plane.

        Raises ``UncalibratedError`` when no calibration exists; callers degrade
        to normalized space and simply omit ground fields (invariant V9).
        """
        calibration = self.get_calibration(camera_id)
        if not calibration.can_project_to_ground:
            raise UncalibratedError(
                f"camera '{camera_id}' calibration '{calibration.calibration_id}' "
                f"cannot project to ground",
                camera_id=str(camera_id),
            )
        return calibration.project_to_ground(point)

    def report_viewpoint_drift(self, camera_id: CameraId, evidence: str) -> None:
        """Flag suspected movement.

        Marks the calibration ``suspect`` — ground projections continue with
        inflated uncertainty. It does **not** auto-invalidate: a false positive
        must not blind a site (03_MODULES M1).
        """
        with self._write_lock:
            current = self._snapshot
            camera = current.cameras.get(camera_id)
            if camera is None or camera.calibration is None:
                return
            suspect = replace(camera.calibration, suspect=True)
            cameras = dict(current.cameras)
            cameras[camera_id] = camera.with_calibration(suspect)
            self._snapshot = replace(current, cameras=cameras)

        self._bus.publish(
            ViewpointDriftSuspected(
                occurred_at=self._clock.now(),
                partition_key=str(camera_id),
                camera_id=camera_id,
                evidence=evidence,
            )
        )

    # --- status ------------------------------------------------------------- #

    def set_status(self, camera_id: CameraId, status: CameraStatus) -> None:
        changed = False
        with self._write_lock:
            current = self._snapshot
            camera = current.cameras.get(camera_id)
            if camera is None:
                raise NotFoundError(f"unknown camera '{camera_id}'", camera_id=str(camera_id))
            if camera.status is status:
                return
            cameras = dict(current.cameras)
            cameras[camera_id] = camera.with_status(status)
            self._snapshot = replace(current, cameras=cameras)
            changed = True
        if changed:
            self._publish(camera_id, f"status:{status.value}")

    def _publish(self, camera_id: CameraId, change: str) -> None:
        self._bus.publish(
            CameraChanged(
                occurred_at=self._clock.now(),
                partition_key=str(camera_id),
                camera_id=camera_id,
                change=change,
            )
        )

    @property
    def count(self) -> int:
        return len(self._snapshot.cameras)


# --- declaration builders -------------------------------------------------- #


def _build_profile(declaration: ProfileDeclaration) -> PipelineProfile:
    return PipelineProfile(
        profile_id=ProfileId(declaration.profile_id),
        target_fps=declaration.target_fps,
        max_in_flight=declaration.max_in_flight,
        priority_class=declaration.priority_class,
        inference_width=declaration.inference_width,
        inference_height=declaration.inference_height,
    )


def _build_region(declaration: RegionDeclaration) -> Region:
    return Region(
        region_id=RegionId(declaration.region_id),
        geometry=Polygon(tuple(Point(x, y) for x, y in declaration.vertices)),
        frame_of_reference=FrameOfReference(declaration.frame_of_reference),
        label=declaration.label,
        camera_id=CameraId(declaration.camera_id) if declaration.camera_id else None,
        version=declaration.version,
    )


def _build_calibration(declaration: CalibrationDeclaration) -> Calibration:
    homography = None
    if declaration.homography is not None:
        rows = declaration.homography
        if len(rows) != 3 or any(len(row) != 3 for row in rows):
            raise ValidationError(
                f"calibration '{declaration.calibration_id}' homography must be 3x3"
            )
        homography = Homography(
            matrix=(
                (rows[0][0], rows[0][1], rows[0][2]),
                (rows[1][0], rows[1][1], rows[1][2]),
                (rows[2][0], rows[2][1], rows[2][2]),
            )
        )
        if homography.is_degenerate():
            raise ValidationError(
                f"calibration '{declaration.calibration_id}' homography is degenerate"
            )
    return Calibration(
        calibration_id=CalibrationId(declaration.calibration_id),
        homography=homography,
        ground_uncertainty_at_unit_distance=declaration.ground_uncertainty_at_unit_distance,
    )


def _build_camera(
    declaration: CameraDeclaration,
    profiles: dict[ProfileId, PipelineProfile],
    regions: dict[RegionId, Region],
) -> Camera:
    profile = profiles.get(ProfileId(declaration.profile_id))
    if profile is None:
        raise ValidationError(
            f"camera '{declaration.camera_id}' references undeclared profile "
            f"'{declaration.profile_id}'"
        )
    for region_id in declaration.region_ids:
        if RegionId(region_id) not in regions:
            raise ValidationError(
                f"camera '{declaration.camera_id}' references undeclared region '{region_id}'"
            )
    return Camera(
        camera_id=CameraId(declaration.camera_id),
        tenant_id=TenantId(declaration.tenant_id),
        site_id=SiteId(declaration.site_id),
        source_spec=SourceSpec(
            uri=declaration.uri,
            transport=declaration.transport,
            credential_ref=declaration.credential_ref,
            options=declaration.source_options,
        ),
        source_semantics=SourceSemantics(declaration.source_semantics),
        native_profile=NativeProfile(
            width=declaration.width,
            height=declaration.height,
            fps=declaration.fps,
            codec=declaration.codec,
            colour_space=declaration.colour_space,
        ),
        pipeline_profile=profile,
        calibration=(
            _build_calibration(declaration.calibration) if declaration.calibration else None
        ),
        privacy_policy_id=(
            PrivacyPolicyId(declaration.privacy_policy_id)
            if declaration.privacy_policy_id
            else None
        ),
        region_ids=tuple(RegionId(r) for r in declaration.region_ids),
        labels=dict(declaration.labels),
    )
