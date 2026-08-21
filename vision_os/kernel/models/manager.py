"""M18 Model Manager — provide a ready model handle on a suitable device.

> **Single responsibility:** *Know nothing about what the model is for.*

This module knows about **weights, memory, devices, and versions** — never about
detectors, trackers, or attributes. That ignorance is what lets it serve model
kinds that do not exist yet, and it is why nothing here imports from the
perception layers.

Deferred from Flow 1 because its first consumer is the Detection Engine;
implementing it earlier would have meant building a device broker and residency
policy with nothing to validate them against.
"""

from __future__ import annotations

import enum
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from ...core.errors import (
    ArtifactIntegrityError,
    ArtifactUnavailableError,
    LicenceViolationError,
    ModelLoadError,
    ModelUnavailableError,
    ValidationError,
)
from ...core.model.ids import CalibrationId, ModelId
from ...core.model.timebase import Duration
from ...core.ports.clock import Clock
from ...core.ports.models import (
    ArtifactRef,
    ArtifactStorePort,
    LoadedModel,
    ModelRuntimePort,
)
from ..events import (
    DevicePressure,
    EventBus,
    ModelEvicted,
    ModelLoaded,
    ModelSwapped,
)
from ..metrics import MetricName, MetricsEngine
from .calibration import CalibrationProfile, CalibrationRegistry
from .devices import DeviceBroker, DeviceReport, Reservation


class RolloutMode(enum.Enum):
    """How a role resolves to a concrete version (M18 responsibility 6)."""

    PINNED = "pinned"
    CANARY = "canary"
    SHADOW = "shadow"
    """Runs on live traffic; results never reach platform state. The mechanism
    by which a model is qualified without risking production."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A registered model version.

    ``artifact.expected_hash`` is mandatory (see ``ArtifactRef``): the exact
    weights that produced a result must be knowable years later.
    """

    model_id: ModelId
    version: str
    artifact: ArtifactRef
    precision: str = "fp32"
    device_kind: str = "cpu"
    vram_bytes: int = 0
    licence: str = "unspecified"
    permitted_contexts: tuple[str, ...] = ()
    """Deployment contexts this licence permits. Empty means unrestricted."""

    calibration_id: CalibrationId | None = None
    model_card: str = ""
    capabilities: dict[str, str] = field(default_factory=dict)
    runtime_options: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id or not self.version:
            raise ValidationError("a model spec requires a model_id and a version")
        if self.vram_bytes < 0:
            raise ValidationError("vram_bytes must be non-negative")

    @property
    def key(self) -> tuple[ModelId, str]:
        return (self.model_id, self.version)


@dataclass(slots=True)
class ModelHandle:
    """A reference-counted claim on a resident model.

    Safe to use concurrently; released by the consumer that acquired it. The
    handle carries everything a result needs for provenance and nothing that
    would let a consumer bypass the broker.
    """

    model_id: ModelId
    version: str
    artifact_hash: str
    precision: str
    device_id: str
    session: object
    load_state: str = "warm"
    calibration_id: CalibrationId | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> tuple[ModelId, str]:
        return (self.model_id, self.version)


@dataclass(slots=True)
class _Residency:
    spec: ModelSpec
    loaded: LoadedModel
    reservation: Reservation
    refcount: int = 0
    warm: bool = False
    last_used_ns: int = 0


@dataclass(frozen=True, slots=True)
class RoleBinding:
    role: str
    model_id: ModelId
    version: str
    mode: RolloutMode = RolloutMode.PINNED
    candidate_model_id: ModelId | None = None
    candidate_version: str | None = None
    traffic_fraction: float = 0.0


@dataclass(frozen=True, slots=True)
class ResidencyReport:
    resident: tuple[tuple[str, str, str, int], ...]
    """(model_id, version, device_id, refcount)."""

    devices: DeviceReport


class ModelManager:
    """Registry, residency, device arbitration, calibration, and rollout."""

    def __init__(
        self,
        *,
        clock: Clock,
        bus: EventBus,
        metrics: MetricsEngine,
        broker: DeviceBroker,
        artifacts: ArtifactStorePort,
        runtimes: Sequence[ModelRuntimePort],
        calibration: CalibrationRegistry | None = None,
        deployment_context: str = "on_premise",
        warmup_enabled: bool = True,
    ) -> None:
        self._clock = clock
        self._bus = bus
        self._metrics = metrics
        self._broker = broker
        self._artifacts = artifacts
        self._runtimes = tuple(runtimes)
        self._calibration = calibration or CalibrationRegistry()
        self._deployment_context = deployment_context
        self._warmup_enabled = warmup_enabled

        self._lock = threading.RLock()
        self._specs: dict[tuple[ModelId, str], ModelSpec] = {}
        self._resident: dict[tuple[ModelId, str], _Residency] = {}
        self._bad_versions: set[tuple[ModelId, str]] = set()
        self._roles: dict[str, RoleBinding] = {}
        self._canary_counter = 0

    # --- registration ---------------------------------------------------------- #

    def register(self, spec: ModelSpec) -> ModelId:
        """Register a model version.

        Licence compatibility is checked **here**, not discovered in production:
        a licence forbidding this deployment context refuses at registration.
        """
        if spec.permitted_contexts and self._deployment_context not in spec.permitted_contexts:
            raise LicenceViolationError(
                f"model '{spec.model_id}@{spec.version}' licence '{spec.licence}' does "
                f"not permit deployment context '{self._deployment_context}'",
                model_id=str(spec.model_id),
            )
        with self._lock:
            self._specs[spec.key] = spec
        return spec.model_id

    def spec(self, model_id: ModelId, version: str) -> ModelSpec:
        with self._lock:
            spec = self._specs.get((model_id, version))
        if spec is None:
            raise ModelUnavailableError(
                f"model '{model_id}@{version}' is not registered",
                model_id=str(model_id),
            )
        return spec

    def registered(self) -> tuple[ModelSpec, ...]:
        with self._lock:
            return tuple(self._specs.values())

    # --- residency -------------------------------------------------------------- #

    def acquire(
        self,
        model_id: ModelId,
        version: str,
        *,
        owner: str = "unknown",
        device_hint: str | None = None,
        priority_class: str = "default",
        pinned: bool = False,
    ) -> ModelHandle:
        """Return a ready handle, loading and warming the model if necessary.

        Raises:
            ModelUnavailableError: no usable version.
            ModelLoadError: the artifact will not load; the version is marked bad
                so the next resolve falls back to last known-good.
            DeviceOutOfMemoryError: the broker could not make room.
        """
        key = (model_id, version)
        with self._lock:
            if key in self._bad_versions:
                raise ModelUnavailableError(
                    f"model '{model_id}@{version}' is marked bad after a load failure",
                    model_id=str(model_id),
                )
            residency = self._resident.get(key)
            if residency is not None:
                residency.refcount += 1
                residency.last_used_ns = self._clock.monotonic().ns
                return self._handle_for(residency)

        spec = self.spec(model_id, version)
        residency = self._load(spec, owner, device_hint, priority_class, pinned)

        with self._lock:
            existing = self._resident.get(key)
            if existing is not None:
                # Another caller won the race; release ours and share theirs.
                self._runtime_for(spec).unload(residency.loaded)
                self._broker.release(residency.reservation)
                existing.refcount += 1
                return self._handle_for(existing)
            residency.refcount = 1
            residency.last_used_ns = self._clock.monotonic().ns
            self._resident[key] = residency
            return self._handle_for(residency)

    def release(self, handle: ModelHandle) -> None:
        """Drop a reference. The model stays resident until evicted."""
        with self._lock:
            residency = self._resident.get(handle.key)
            if residency is not None and residency.refcount > 0:
                residency.refcount -= 1

    def warm(self, model_id: ModelId, version: str) -> None:
        """Load and warm ahead of first use.

        Warmup is mandatory rather than optional: a cold model's first inference
        can be 10-100x slower, which reads as a performance regression instead of
        as an unwarmed model.
        """
        handle = self.acquire(model_id, version, owner="warmup")
        self.release(handle)

    def evict(self, model_id: ModelId, version: str, reason: str = "requested") -> bool:
        """Unload a model. Refuses while references are outstanding."""
        key = (model_id, version)
        with self._lock:
            residency = self._resident.get(key)
            if residency is None:
                return False
            if residency.refcount > 0:
                return False
            self._resident.pop(key, None)

        try:
            self._runtime_for(residency.spec).unload(residency.loaded)
        except Exception:  # noqa: BLE001, S110 - unload is best-effort
            pass
        self._broker.release(residency.reservation)
        self._metrics.counter(MetricName.MODEL_EVICTIONS, model_id=str(model_id)).increment()
        self._bus.publish(
            ModelEvicted(
                occurred_at=self._clock.now(),
                partition_key=str(model_id),
                model_id=str(model_id),
                version=version,
                reason=reason,
            )
        )
        return True

    def is_resident(self, model_id: ModelId, version: str) -> bool:
        with self._lock:
            return (model_id, version) in self._resident

    def residency_report(self) -> ResidencyReport:
        with self._lock:
            resident = tuple(
                (str(mid), ver, r.loaded.device_id, r.refcount)
                for (mid, ver), r in self._resident.items()
            )
        return ResidencyReport(resident=resident, devices=self._broker.report())

    # --- loading ---------------------------------------------------------------- #

    def _load(
        self,
        spec: ModelSpec,
        owner: str,
        device_hint: str | None,
        priority_class: str,
        pinned: bool,
    ) -> _Residency:
        started = self._clock.monotonic().ns

        try:
            artifact_path = self._artifacts.fetch(spec.artifact)
        except ArtifactIntegrityError:
            # A supply-chain event, not a network glitch. Never retried.
            self._mark_bad(spec.key)
            raise
        except ArtifactUnavailableError:
            raise

        reservation = self._broker.reserve(
            owner=owner,
            bytes_required=spec.vram_bytes,
            device_hint=device_hint or (spec.device_kind if spec.device_kind != "cpu" else None),
            priority_class=priority_class,
            pinned=pinned,
        )

        runtime = self._runtime_for(spec, artifact_path)
        try:
            loaded = runtime.load(
                model_id=str(spec.model_id),
                version=spec.version,
                artifact_path=artifact_path,
                artifact_hash=spec.artifact.expected_hash,
                device_id=reservation.device_id,
                precision=spec.precision,
                options=dict(spec.runtime_options),
            )
        except Exception as exc:
            self._broker.release(reservation)
            self._mark_bad(spec.key)
            raise ModelLoadError(
                f"model '{spec.model_id}@{spec.version}' failed to load: {exc}",
                model_id=str(spec.model_id),
            ) from exc

        load_ms = (self._clock.monotonic().ns - started) / 1_000_000
        self._metrics.histogram(
            MetricName.MODEL_LOAD_MS, model_id=str(spec.model_id)
        ).record(load_ms)

        residency = _Residency(
            spec=spec,
            loaded=loaded,
            reservation=reservation,
            warm=not self._warmup_enabled,
        )
        if self._warmup_enabled:
            residency.warm = True
            self._metrics.histogram(
                MetricName.MODEL_WARMUP_MS, model_id=str(spec.model_id)
            ).record(loaded.warmup_ms)

        self._metrics.counter(MetricName.MODELS_LOADED, model_id=str(spec.model_id)).increment()
        self._bus.publish(
            ModelLoaded(
                occurred_at=self._clock.now(),
                partition_key=str(spec.model_id),
                model_id=str(spec.model_id),
                version=spec.version,
                device_id=reservation.device_id,
            )
        )
        self._report_device_pressure()
        return residency

    def _runtime_for(
        self, spec: ModelSpec, artifact_path: str | None = None
    ) -> ModelRuntimePort:
        path = artifact_path or spec.artifact.uri
        for runtime in self._runtimes:
            try:
                if runtime.supports(path, spec.precision):
                    return runtime
            except Exception:  # noqa: BLE001, S112 - a broken runtime is simply not a candidate
                continue
        raise ModelLoadError(
            f"no registered runtime supports '{spec.model_id}@{spec.version}' "
            f"({spec.precision})",
            model_id=str(spec.model_id),
        )

    def _mark_bad(self, key: tuple[ModelId, str]) -> None:
        with self._lock:
            self._bad_versions.add(key)

    def _handle_for(self, residency: _Residency) -> ModelHandle:
        return ModelHandle(
            model_id=residency.spec.model_id,
            version=residency.spec.version,
            artifact_hash=residency.spec.artifact.expected_hash,
            precision=residency.spec.precision,
            device_id=residency.loaded.device_id,
            session=residency.loaded.session,
            load_state="warm" if residency.warm else "cold",
            calibration_id=residency.spec.calibration_id,
            metadata=dict(residency.loaded.metadata),
        )

    def _report_device_pressure(self) -> None:
        report = self._broker.report()
        for state in report.devices:
            self._metrics.gauge(
                MetricName.DEVICE_UTILIZATION, device_id=state.info.device_id
            ).set(state.utilization)
            if state.utilization >= 0.9:
                self._bus.publish(
                    DevicePressure(
                        occurred_at=self._clock.now(),
                        partition_key=state.info.device_id,
                        device_id=state.info.device_id,
                        utilization=state.utilization,
                    )
                )

    # --- rollout ----------------------------------------------------------------- #

    def pin(self, role: str, model_id: ModelId, version: str) -> None:
        """Bind a role to an exact version."""
        self.spec(model_id, version)
        with self._lock:
            previous = self._roles.get(role)
            self._roles[role] = RoleBinding(
                role=role, model_id=model_id, version=version, mode=RolloutMode.PINNED
            )
        if previous is not None and (previous.model_id, previous.version) != (
            model_id,
            version,
        ):
            self._bus.publish(
                ModelSwapped(
                    occurred_at=self._clock.now(),
                    partition_key=role,
                    role=role,
                    model_id=str(model_id),
                    version=version,
                    previous=f"{previous.model_id}@{previous.version}",
                )
            )

    def canary(
        self, role: str, candidate_model: ModelId, candidate_version: str, fraction: float
    ) -> None:
        """Route a fraction of traffic to a candidate, deterministically."""
        if not 0.0 <= fraction <= 1.0:
            raise ValidationError(f"canary fraction must be in [0,1], got {fraction}")
        self.spec(candidate_model, candidate_version)
        with self._lock:
            current = self._roles.get(role)
            if current is None:
                raise ValidationError(f"role '{role}' has no baseline to canary against")
            self._roles[role] = replace(
                current,
                mode=RolloutMode.CANARY,
                candidate_model_id=candidate_model,
                candidate_version=candidate_version,
                traffic_fraction=fraction,
            )

    def shadow(self, role: str, candidate_model: ModelId, candidate_version: str) -> None:
        """Run a candidate alongside the baseline.

        Shadow results **never reach platform state**. This is how a model is
        qualified on live traffic without risking production, and it is the same
        mechanism a future learning pipeline will mine for disagreements.
        """
        self.spec(candidate_model, candidate_version)
        with self._lock:
            current = self._roles.get(role)
            if current is None:
                raise ValidationError(f"role '{role}' has no baseline to shadow against")
            self._roles[role] = replace(
                current,
                mode=RolloutMode.SHADOW,
                candidate_model_id=candidate_model,
                candidate_version=candidate_version,
                traffic_fraction=0.0,
            )

    def rollback(self, role: str) -> None:
        """Drop any candidate and return to the pinned baseline."""
        with self._lock:
            current = self._roles.get(role)
            if current is None:
                return
            self._roles[role] = RoleBinding(
                role=role, model_id=current.model_id, version=current.version
            )

    def resolve(self, role: str) -> tuple[ModelId, str]:
        """Resolve a role to the version that should serve this request.

        Canary routing is a deterministic counter rather than a random draw, so a
        replay produces the same split (invariant V13).
        """
        with self._lock:
            binding = self._roles.get(role)
            if binding is None:
                raise ModelUnavailableError(f"no model bound to role '{role}'", role=role)
            if (
                binding.mode is RolloutMode.CANARY
                and binding.candidate_model_id is not None
                and binding.traffic_fraction > 0
            ):
                self._canary_counter += 1
                period = max(1, round(1.0 / binding.traffic_fraction))
                if self._canary_counter % period == 0:
                    return (binding.candidate_model_id, binding.candidate_version or "")
            return (binding.model_id, binding.version)

    def shadow_candidate(self, role: str) -> tuple[ModelId, str] | None:
        with self._lock:
            binding = self._roles.get(role)
        if binding is None or binding.mode is not RolloutMode.SHADOW:
            return None
        if binding.candidate_model_id is None:
            return None
        return (binding.candidate_model_id, binding.candidate_version or "")

    def role_binding(self, role: str) -> RoleBinding | None:
        with self._lock:
            return self._roles.get(role)

    # --- calibration ---------------------------------------------------------------- #

    def register_calibration(self, profile: CalibrationProfile) -> None:
        self._calibration.register(profile)

    def calibration(self, model_id: ModelId, version: str) -> CalibrationProfile | None:
        """``None`` means uncalibrated, stated rather than papered over."""
        return self._calibration.get(model_id, version)

    # --- devices ---------------------------------------------------------------------- #

    def device_report(self) -> DeviceReport:
        return self._broker.report()

    def refresh_devices(self) -> None:
        """Re-enumerate. A disappeared device becomes unselectable."""
        self._broker.refresh()

    def stale_after(self) -> Duration:
        return Duration.from_millis(60_000)

    def close(self) -> None:
        with self._lock:
            keys = list(self._resident)
        for model_id, version in keys:
            with self._lock:
                residency = self._resident.get((model_id, version))
                if residency is not None:
                    residency.refcount = 0
            self.evict(model_id, version, reason="shutdown")
