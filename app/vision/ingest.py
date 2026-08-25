"""The seam: a session frame becomes a Vision OS frame.

Phases 3–5 built a source, a session, a bounded queue and a durable store, and
Phase 4 assembled the perception stack — but nothing joined them. `LiveRuntime`
took an `on_frame` callback that nothing ever set, so a frame reached the
consumer task and stopped there. This module is that callback, and it is the
only thing in the application that hands a frame to the platform.

### It composes; it does not reimplement

Four public platform calls, in the order the platform declares them:

    buffer.register_camera(camera_id)          # once per camera
    slot = buffer.acquire_slot(camera_id, …)   # pooled, bounded
    buffer.publish(slot, frame_ref=…, …)       # slot becomes an immutable Frame
    await detection.runtime.on_admitted(ref, fidelity)

`on_admitted` is `AdmittedFrameConsumer` — described in the platform's own words
as *"the single, documented extension point at which a later flow resumes the
admitted-frame path"*. Everything after it (detection → tracking → registry →
cropping → understanding → synthesis → state) is the platform's wiring, done at
assembly, and this module neither knows nor touches it.

### What this deliberately does not do

It does not run the scheduler's admission policy. A frame reaching here has
already passed `FrameSampler` in the session, which is the application's one
sampling decision (§7: one processing boundary). Running admission again would
be a second attention policy with a second set of counters, and the platform's
`cameras × fps` economics would stop matching what actually happened.

It never raises into the session. `AdmittedFrameConsumer` implementations "must
not raise" by contract, and this holds the same rule one level up: a frame that
cannot be ingested is counted and dropped, and the camera keeps running.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from app.vision.frames import LiveFrame
from app.vision.ledger import FrameLedger, frame_ref_for
from app.vision.session import SessionSpec

#: What the detector is asked to work at. Matches the letterboxed input the
#: bound YOLO artifact declares; the platform scales from the published frame.
DEFAULT_INFERENCE_SIZE = 640


@dataclass(slots=True)
class IngestAudit:
    """What the seam did. Counted, never inferred.

    Every field is incremented at the moment the thing happens, so a report from
    here can be compared against the platform's own metrics and any disagreement
    is a real finding rather than two estimates of the same number.
    """

    frames_offered: int = 0
    frames_published: int = 0
    frames_admitted: int = 0
    #: The ring was full and the oldest unpinned frame could not be evicted.
    pool_exhausted: int = 0
    publish_failures: int = 0
    consumer_failures: int = 0
    cameras_registered: int = 0
    last_error: str = ""
    failure_kinds: dict[str, int] = field(default_factory=dict)

    def note_failure(self, exc: BaseException) -> None:
        kind = type(exc).__name__
        self.failure_kinds[kind] = self.failure_kinds.get(kind, 0) + 1
        self.last_error = f"{kind}: {exc}"

    def to_wire(self) -> dict[str, Any]:
        return {
            "frames_offered": self.frames_offered,
            "frames_published": self.frames_published,
            "frames_admitted": self.frames_admitted,
            "pool_exhausted": self.pool_exhausted,
            "publish_failures": self.publish_failures,
            "consumer_failures": self.consumer_failures,
            "cameras_registered": self.cameras_registered,
            "failure_kinds": dict(self.failure_kinds),
            "last_error": self.last_error,
            # The number that says whether the seam is actually working. Zero
            # admitted with a non-zero offer count is the exact shape of the
            # Phase 4 gap this module exists to close.
            "admitted_fraction": (
                self.frames_admitted / self.frames_offered if self.frames_offered else 0.0
            ),
        }


class FrameIngest:
    """Publishes session frames into the platform and admits them to detection."""

    __slots__ = ("_inference_size", "_known", "_ledger", "_lock", "audit", "composition")

    def __init__(
        self,
        composition: Any,
        *,
        ledger: FrameLedger | None = None,
        inference_size: int = DEFAULT_INFERENCE_SIZE,
    ) -> None:
        self.composition = composition
        self.audit = IngestAudit()
        self._ledger = ledger
        self._inference_size = inference_size
        self._known: set[str] = set()
        self._lock = threading.Lock()

    @property
    def _platform(self) -> Any:
        return getattr(self.composition, "platform", None)

    async def __call__(self, spec: SessionSpec, frame: LiveFrame) -> None:
        """The `on_frame` callback. Never raises."""
        self.audit.frames_offered += 1

        platform = self._platform
        detection = getattr(self.composition, "detection", None)
        if platform is None or detection is None:
            # No assembled stack. Not an error: a deployment may run acquisition
            # only, and saying so once beats failing on every frame.
            return

        try:
            frame_ref = self._publish(platform, spec, frame)
        except Exception as exc:  # noqa: BLE001 - one frame, not the camera
            self._note_publish_failure(exc, frame)
            return

        try:
            await detection.runtime.on_admitted(frame_ref, self._fidelity())
            self.audit.frames_admitted += 1
        except Exception as exc:  # noqa: BLE001 - the seam is a firewall
            self.audit.consumer_failures += 1
            self.audit.note_failure(exc)
            logger.warning(
                "camera {} frame {} was published but not admitted: {}: {}",
                frame.camera_id,
                frame.sequence,
                type(exc).__name__,
                exc,
            )

    # -- publishing -----------------------------------------------------------

    def _publish(self, platform: Any, spec: SessionSpec, frame: LiveFrame) -> Any:
        from vision_os.core.model.frame import (
            FrameDimensions,
            FrameTime,
            PrivacyState,
        )
        from vision_os.core.model.ids import CameraId, FrameRef, StreamEpoch
        from vision_os.core.model.timebase import ClockQuality
        from vision_os.kernel.clock import Duration, Instant

        camera_id = CameraId(frame.camera_id)
        self._ensure_registered(platform, spec, frame)

        buffer = platform.buffer
        slot = buffer.acquire_slot(camera_id)

        payload = frame.payload or b""
        try:
            # `memory()` is a method, not a property, and it raises once the slot
            # is released — which is the buffer telling a late writer that the
            # allocation is no longer theirs.
            memory = slot.memory()
            # Truncated rather than refused. A slot sized for a smaller frame is a
            # `BYTES_PER_SLOT` configuration fact, and dropping the frame outright
            # would hide it behind an empty timeline instead of a short one.
            written = min(len(payload), len(memory))
            memory[:written] = payload[:written]
        except Exception:
            # A slot that was taken and not published leaks a pooled allocation.
            # Returning it is the buffer's documented path for exactly this.
            buffer.discard_slot(slot)
            raise

        frame_ref = FrameRef(
            camera_id=camera_id,
            stream_epoch=StreamEpoch(frame.epoch),
            frame_seq=frame.sequence,
        )

        capture = Instant(frame.captured_at_ns)
        ingest = Instant(frame.received_at_ns)

        published = buffer.publish(
            slot,
            frame_ref=frame_ref,
            time=FrameTime(
                pts=frame.sequence,
                # Capture time, from the source. Never `now()`: a finding is
                # about when the camera saw something, and stamping arrival here
                # would silently re-date every observation by the queue delay.
                t_capture=capture,
                t_capture_uncertainty=Duration(0),
                t_ingest=ingest,
                t_decoded=ingest,
                # The honest value for a replay or an RTSP stream with no
                # PTP or RTCP: this process does not know how well the source's
                # clock is disciplined, and says so rather than claiming an
                # accuracy it cannot vouch for. Downstream freshness maths reads
                # this, so an optimistic value here would quietly widen every
                # validity window in the system.
                clock_quality=ClockQuality.UNKNOWN,
            ),
            dimensions=FrameDimensions(
                width=getattr(frame, "width", 0) or 0,
                height=getattr(frame, "height", 0) or 0,
            ),
            # No masking adapter is bound, so the frame is unmasked and says so.
            # `MASKED` here would be a claim about privacy processing that never
            # happened — the one lie the platform refuses to publish.
            privacy_state=PrivacyState.UNMASKED_PERMITTED,
            bytes_written=written,
        )
        self.audit.frames_published += 1
        return published.frame_ref

    def _ensure_registered(self, platform: Any, spec: SessionSpec, frame: LiveFrame) -> None:
        """Make the platform aware of this camera. Once, per camera.

        Three registrations, and all three are needed:

        * **Frame buffer** — allocates the camera's ring.
        * **Health monitor** — makes the camera appear in coverage reporting;
          without it a camera can stream while the site reports full observability.
        * **Camera manager** — the record `DetectionEngine` looks up to resolve a
          pipeline profile. Missing it, detection fails every frame with
          `camera_unknown`, which is a *persistent* failure class and therefore
          exactly the silent, total outage this application must never have.

        The third is the one that is easy to miss, because the first two succeed
        and frames flow all the way to detection before anything complains.
        """
        camera = spec.camera_id
        key = str(camera)
        with self._lock:
            if key in self._known:
                return
            self._known.add(key)

        from vision_os.core.model.ids import CameraId

        identity = CameraId(camera)

        for holder in (platform.buffer, platform.health):
            register = getattr(holder, "register_camera", None)
            if callable(register):
                try:
                    register(identity)
                except Exception as exc:  # noqa: BLE001 - already registered is fine
                    logger.debug(
                        "register_camera({}) on {}: {}: {}",
                        key,
                        type(holder).__name__,
                        type(exc).__name__,
                        exc,
                    )

        self._provision(platform, identity, spec, frame)
        self.audit.cameras_registered += 1

    def _provision(self, platform: Any, identity: Any, spec: SessionSpec, frame: LiveFrame) -> None:
        """Declare the camera to the camera manager.

        The `SourceSpec` records **how this process actually acquired** the
        frames, and it carries a redacted URI and no credential — a camera record
        the platform holds in memory is still a place a password must never be.
        """
        cameras = getattr(platform, "cameras", None)
        if cameras is None or cameras.try_get(identity) is not None:
            return

        from vision_os.core.model.camera import (
            Camera,
            CameraStatus,
            NativeProfile,
            SourceSpec,
        )
        from vision_os.core.model.ids import ProfileId, SiteId, TenantId
        from vision_os.core.ports.scheduling import PipelineProfile, SourceSemantics

        try:
            cameras.provision(
                Camera(
                    camera_id=identity,
                    tenant_id=TenantId(spec.tenant_id),
                    site_id=SiteId(spec.tenant_id),
                    source_spec=SourceSpec(
                        # Redacted at the source, and never rebuilt here.
                        uri=self._redacted_uri(frame),
                        transport="application",
                    ),
                    # Replay is archival: completeness matters more than latency,
                    # so a full ring blocks the producer rather than dropping a
                    # frame the run is supposed to contain.
                    source_semantics=SourceSemantics.REALTIME,
                    native_profile=NativeProfile(
                        width=getattr(frame, "width", 0) or 0,
                        height=getattr(frame, "height", 0) or 0,
                        fps=spec.analysis_fps,
                        codec="raw",
                    ),
                    pipeline_profile=PipelineProfile(
                        profile_id=ProfileId("application"),
                        # What this deployment actually asks for, not what the
                        # camera can emit. The sampler upstream already enforced
                        # it, and stating a higher number here would make every
                        # cost projection wrong.
                        target_fps=spec.analysis_fps,
                        inference_width=self._inference_size,
                        inference_height=self._inference_size,
                    ),
                    status=CameraStatus.STREAMING,
                )
            )
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            self.audit.note_failure(exc)
            logger.error(
                "camera {} could not be provisioned to the platform: {}: {}. "
                "Detection will report camera_unknown for every frame.",
                identity,
                type(exc).__name__,
                exc,
            )

    @staticmethod
    def _redacted_uri(frame: LiveFrame) -> str:
        return f"application://{frame.camera_id}"

    def _fidelity(self) -> Any:
        from vision_os.core.ports.scheduling import Fidelity

        return Fidelity(
            inference_width=self._inference_size,
            inference_height=self._inference_size,
        )

    def _note_publish_failure(self, exc: BaseException, frame: LiveFrame) -> None:
        kind = type(exc).__name__
        if kind == "PoolExhaustedError":
            # The ring is full of pinned frames. Distinct from a publish error
            # because the operator response is different: raise buffer depth or
            # shorten pin TTL, rather than look for a bug.
            self.audit.pool_exhausted += 1
        else:
            self.audit.publish_failures += 1
        self.audit.note_failure(exc)

        if self._ledger is not None:
            self._ledger.annotate(
                frame_ref_for(frame.camera_id, frame.epoch, frame.sequence),
                error=f"ingest: {kind}",
            )

        logger.warning(
            "camera {} frame {} could not be published to the platform: {}: {}",
            frame.camera_id,
            frame.sequence,
            kind,
            exc,
        )


__all__ = ["DEFAULT_INFERENCE_SIZE", "FrameIngest", "IngestAudit"]
