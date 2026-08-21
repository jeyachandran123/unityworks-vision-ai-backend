"""M16 Configuration Manager — resolve and validate configuration. Interpret nothing.

Single responsibility. The **only** component that reads the outside world for
settings; every other module receives a validated, typed slice by injection,
which is what makes every module constructible in a test with a literal config.

Three properties do the real work:

* **Layered with traceable origin.** Every effective value knows which layer set
  it, because "why is this camera running at 2 fps?" must not become an
  afternoon of archaeology.
* **Immutable snapshots.** A module holds its slice for the duration of an
  operation and never observes a torn configuration — a camera pipeline reading
  cadence from one revision and budget from another produces behaviour nobody
  can reproduce.
* **Failed reload keeps the current revision.** Never apply partial
  configuration; never degrade a running system for a bad reload.
"""

from __future__ import annotations

import enum
import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from ...core.errors import ConfigurationError, ValidationError
from ...core.model.ids import ConfigRevision
from ...core.model.timebase import Duration, Instant
from ...core.ports.clock import Clock
from ...core.ports.configuration import ConfigSourcePort, SecretProviderPort
from .schema import (
    SECTION_TYPES,
    ApiSection,
    BufferSection,
    CalibrationDeclaration,
    CameraDeclaration,
    ClockMode,
    CroppingSection,
    DeploymentProfile,
    DetectionSection,
    DetectorDeclaration,
    EffectiveConfig,
    HealthSection,
    MappingEntryDeclaration,
    MetricsSection,
    ModelsSection,
    PlatformSection,
    ProfileDeclaration,
    RegionDeclaration,
    RegistrySection,
    RuntimeSection,
    SchedulerSection,
    SourceSection,
    StateSection,
    StorageSection,
    SynthesisSection,
    TaxonomyClassDeclaration,
    TrackingSection,
    UnderstandingSection,
    validate,
)


class ConfigLayer(enum.IntEnum):
    """Precedence: later layers override earlier ones (05_KERNEL M16)."""

    DEFAULTS = 0
    DEPLOYMENT = 1
    TENANT = 2
    SITE = 3
    CAMERA = 4
    OVERRIDE = 5
    """Time-boxed, operational, always audited."""


@dataclass(frozen=True, slots=True)
class ValueOrigin:
    """Which layer set a value, and to what."""

    path: str
    layer: ConfigLayer
    source_id: str
    value: Any


@dataclass(frozen=True, slots=True)
class LayerDocument:
    layer: ConfigLayer
    source_id: str
    document: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReloadResult:
    revision: ConfigRevision
    changed_paths: tuple[str, ...]
    requires_restart: tuple[str, ...]
    """Changes that were resolved but cannot take effect until restart. Reported
    precisely — never silently ignore a change an operator made."""


@dataclass(frozen=True, slots=True)
class OverrideHandle:
    path: str
    expires_at: Instant
    actor: str


#: Sections whose changes cannot take effect without a restart (08_RUNTIME §7.2).
NON_RELOADABLE_SECTIONS: frozenset[str] = frozenset({"platform", "buffer", "runtime"})
NON_RELOADABLE_KEYS: frozenset[str] = frozenset(
    {
        "metrics.max_label_cardinality",
        # Detection's execution substrate is built at boot: changing queue depth
        # or worker batching mid-flight would strand in-flight requests.
        "detection.queue_capacity",
        "detection.max_batch_size",
        "models.artifact_cache_dir",
        "models.allow_cpu_fallback",
    }
)


class ConfigurationManager:
    """Resolve layered configuration into validated, typed, immutable slices."""

    def __init__(
        self,
        *,
        clock: Clock,
        sources: dict[ConfigLayer, ConfigSourcePort] | None = None,
        secrets: SecretProviderPort | None = None,
        defaults: dict[str, Any] | None = None,
    ) -> None:
        self._clock = clock
        self._sources = dict(sources or {})
        self._secrets = secrets
        self._lock = threading.RLock()
        self._layers: list[LayerDocument] = []
        self._merged: dict[str, Any] = {}
        self._effective: EffectiveConfig | None = None
        self._revision: ConfigRevision = ConfigRevision("cfg-uninitialised")
        self._origins: dict[str, ValueOrigin] = {}
        self._watchers: list[Callable[[ConfigRevision], None]] = []
        self._overrides: dict[str, tuple[Any, Instant, str]] = {}
        self._history: list[tuple[ConfigRevision, Instant]] = []

        if defaults:
            self._layers.append(LayerDocument(ConfigLayer.DEFAULTS, "builtin-defaults", defaults))

    # --- loading ---------------------------------------------------------- #

    def load(self) -> ConfigRevision:
        """Load every configured source and resolve. Fails fast and loudly.

        Raises:
            ValidationError: with the precise path and expectation. Never boot
                into a half-valid state.
        """
        documents: list[LayerDocument] = [
            layer for layer in self._layers if layer.layer is ConfigLayer.DEFAULTS
        ]
        for layer, source in sorted(self._sources.items()):
            try:
                raw = source.load()
            except ConfigurationError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalise adapter failures
                raise ConfigurationError(
                    f"config source '{source.source_id}' failed to load: {exc}",
                    layer=layer.name,
                ) from exc
            documents.append(LayerDocument(layer, source.source_id, raw or {}))

        with self._lock:
            self._layers = documents
            return self._resolve_locked()

    def reload(self) -> ReloadResult:
        """Re-resolve from sources.

        On validation failure the current revision stays in force and the error
        is raised — never apply partial configuration to a running system.
        """
        with self._lock:
            previous_merged = dict(self._merged)
            previous_revision = self._revision
            previous_layers = list(self._layers)
            try:
                revision = self.load()
            except (ValidationError, ConfigurationError):
                self._layers = previous_layers
                self._merged = previous_merged
                self._revision = previous_revision
                raise

            changed = _changed_paths(previous_merged, self._merged)
            requires_restart = tuple(
                path
                for path in changed
                if path.split(".", 1)[0] in NON_RELOADABLE_SECTIONS or path in NON_RELOADABLE_KEYS
            )
            watchers = list(self._watchers)

        for watcher in watchers:
            try:
                watcher(revision)
            except Exception:  # noqa: BLE001, S112 - a bad watcher must not break reload
                continue

        return ReloadResult(
            revision=revision,
            changed_paths=changed,
            requires_restart=requires_restart,
        )

    def _resolve_locked(self) -> ConfigRevision:
        merged: dict[str, Any] = {}
        origins: dict[str, ValueOrigin] = {}

        for layer_doc in sorted(self._layers, key=lambda d: d.layer):
            _deep_merge(merged, layer_doc.document, origins, layer_doc, prefix="")

        now = self._clock.now()
        for path, (value, expires_at, actor) in list(self._overrides.items()):
            if expires_at.ns <= now.ns:
                self._overrides.pop(path, None)
                continue
            _set_path(merged, path, value)
            origins[path] = ValueOrigin(path, ConfigLayer.OVERRIDE, f"override:{actor}", value)

        violations = validate(merged)
        if violations:
            raise ValidationError(
                f"configuration invalid: {len(violations)} violation(s)",
                violations=tuple(violations),
            )

        effective = _build_effective(merged)
        revision = _compute_revision(merged)

        self._merged = merged
        self._origins = origins
        self._effective = effective
        self._revision = revision
        self._history.append((revision, now))
        if len(self._history) > 128:
            del self._history[:-128]
        return revision

    # --- access ----------------------------------------------------------- #

    def effective(self) -> EffectiveConfig:
        with self._lock:
            if self._effective is None:
                raise ConfigurationError("configuration has not been loaded")
            return self._effective

    def revision(self) -> ConfigRevision:
        with self._lock:
            return self._revision

    def platform(self) -> PlatformSection:
        return self.effective().platform

    def buffer(self) -> BufferSection:
        return self.effective().buffer

    def scheduler(self) -> SchedulerSection:
        return self.effective().scheduler

    def source(self) -> SourceSection:
        return self.effective().source

    def health(self) -> HealthSection:
        return self.effective().health

    def metrics(self) -> MetricsSection:
        return self.effective().metrics

    def runtime(self) -> RuntimeSection:
        return self.effective().runtime

    def detection(self) -> DetectionSection:
        return self.effective().detection

    def models(self) -> ModelsSection:
        return self.effective().models

    def tracking(self) -> TrackingSection:
        return self.effective().tracking

    def registry(self) -> RegistrySection:
        return self.effective().registry

    def cropping(self) -> CroppingSection:
        return self.effective().cropping

    def understanding(self) -> UnderstandingSection:
        return self.effective().understanding

    def synthesis(self) -> SynthesisSection:
        return self.effective().synthesis

    def state(self) -> StateSection:
        return self.effective().state

    def storage(self) -> StorageSection:
        """M13's adapter selection and retention policy (Flow 8)."""
        return self.effective().storage

    def api(self) -> ApiSection:
        """M14's operating envelope (Flow 8)."""
        return self.effective().api

    def taxonomy(self) -> tuple[TaxonomyClassDeclaration, ...]:
        return self.effective().taxonomy

    def detectors(self) -> tuple[DetectorDeclaration, ...]:
        return self.effective().detectors

    def cameras(self) -> tuple[CameraDeclaration, ...]:
        return self.effective().cameras

    def regions(self) -> tuple[RegionDeclaration, ...]:
        return self.effective().regions

    def profiles(self) -> tuple[ProfileDeclaration, ...]:
        return self.effective().profiles

    # --- diagnostics ------------------------------------------------------ #

    def explain(self, path: str) -> ValueOrigin:
        """Which layer set this value, and to what.

        Without this, "why is this camera running at 2 fps?" is archaeology.
        """
        with self._lock:
            origin = self._origins.get(path)
            if origin is None:
                raise ConfigurationError(f"no configured value at path '{path}'", path=path)
            return origin

    def history(self) -> tuple[tuple[ConfigRevision, Instant], ...]:
        with self._lock:
            return tuple(self._history)

    def validate_candidate(self, document: dict[str, Any]) -> tuple[str, ...]:
        return validate(document)

    # --- secrets ---------------------------------------------------------- #

    def resolve_secret(self, reference: str | None) -> str | None:
        """Resolve a credential reference. Never places the value in the tree."""
        if reference is None:
            return None
        if self._secrets is None:
            raise ConfigurationError(
                "a credential_ref is declared but no secret provider is configured",
                reference=reference,
            )
        return self._secrets.resolve(reference)

    # --- overrides -------------------------------------------------------- #

    def override(self, path: str, value: Any, ttl: Duration, actor: str) -> OverrideHandle:
        """Apply a time-boxed operational override. Always audited, always expires."""
        with self._lock:
            expires_at = self._clock.now().plus(ttl)
            self._overrides[path] = (value, expires_at, actor)
            self._resolve_locked()
            return OverrideHandle(path=path, expires_at=expires_at, actor=actor)

    def clear_override(self, path: str) -> None:
        with self._lock:
            self._overrides.pop(path, None)
            self._resolve_locked()

    # --- watching --------------------------------------------------------- #

    def watch(self, callback: Callable[[ConfigRevision], None]) -> None:
        with self._lock:
            self._watchers.append(callback)


# --- merge helpers -------------------------------------------------------- #


def _deep_merge(
    target: dict[str, Any],
    incoming: dict[str, Any],
    origins: dict[str, ValueOrigin],
    layer_doc: LayerDocument,
    prefix: str,
) -> None:
    for key, value in incoming.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value, origins, layer_doc, prefix=f"{path}.")
        elif isinstance(value, dict):
            target[key] = {}
            _deep_merge(target[key], value, origins, layer_doc, prefix=f"{path}.")
        else:
            target[key] = value
            origins[path] = ValueOrigin(path, layer_doc.layer, layer_doc.source_id, value)


def _set_path(document: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = document
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


def _flatten(document: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in document.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value if not isinstance(value, list) else json.dumps(value, default=str)
    return flat


def _changed_paths(before: dict[str, Any], after: dict[str, Any]) -> tuple[str, ...]:
    flat_before = _flatten(before)
    flat_after = _flatten(after)
    keys = set(flat_before) | set(flat_after)
    return tuple(sorted(k for k in keys if flat_before.get(k) != flat_after.get(k)))


def _compute_revision(document: dict[str, Any]) -> ConfigRevision:
    payload = json.dumps(document, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).hexdigest()
    return ConfigRevision(f"cfg-{digest}")


# --- typed construction --------------------------------------------------- #


def _build_section(section_type: type, raw: dict[str, Any] | None) -> Any:
    if not raw:
        return section_type()
    kwargs: dict[str, Any] = {}
    assert is_dataclass(section_type)
    for field_info in fields(section_type):
        if field_info.name not in raw:
            continue
        value = raw[field_info.name]
        if field_info.type in ("DeploymentProfile", DeploymentProfile):
            value = DeploymentProfile(value) if not isinstance(value, DeploymentProfile) else value
        elif field_info.type in ("ClockMode", ClockMode):
            value = ClockMode(value) if not isinstance(value, ClockMode) else value
        kwargs[field_info.name] = value
    return section_type(**kwargs)


def _build_effective(merged: dict[str, Any]) -> EffectiveConfig:
    sections = {
        name: _build_section(section_type, merged.get(name))
        for name, section_type in SECTION_TYPES.items()
    }
    return EffectiveConfig(
        platform=sections["platform"],
        buffer=sections["buffer"],
        scheduler=sections["scheduler"],
        source=sections["source"],
        health=sections["health"],
        metrics=sections["metrics"],
        runtime=sections["runtime"],
        detection=sections["detection"],
        models=sections["models"],
        tracking=sections["tracking"],
        registry=sections["registry"],
        cropping=sections["cropping"],
        understanding=sections["understanding"],
        synthesis=sections["synthesis"],
        state=sections["state"],
        storage=sections["storage"],
        api=sections["api"],
        profiles=tuple(_build_profile(p) for p in merged.get("profiles", []) or []),
        regions=tuple(_build_region(r) for r in merged.get("regions", []) or []),
        cameras=tuple(_build_camera(c) for c in merged.get("cameras", []) or []),
        taxonomy=tuple(
            _build_taxonomy_class(t) for t in merged.get("taxonomy", []) or []
        ),
        detectors=tuple(_build_detector(d) for d in merged.get("detectors", []) or []),
    )


def _build_taxonomy_class(raw: dict[str, Any]) -> TaxonomyClassDeclaration:
    return TaxonomyClassDeclaration(
        class_id=raw["class_id"],
        geometry_kinds=tuple(raw.get("geometry_kinds", ("box",)) or ("box",)),
        description=raw.get("description", ""),
        status=raw.get("status", "active"),
        superseded_by=raw.get("superseded_by"),
    )


def _build_detector(raw: dict[str, Any]) -> DetectorDeclaration:
    return DetectorDeclaration(
        detector_id=raw["detector_id"],
        adapter_id=raw["adapter_id"],
        model_id=raw["model_id"],
        model_version=raw["model_version"],
        artifact_uri=raw["artifact_uri"],
        artifact_hash=raw["artifact_hash"],
        role=raw.get("role", "primary_detector"),
        precision=raw.get("precision", "fp32"),
        device_kind=raw.get("device_kind", "cpu"),
        vram_bytes=int(raw.get("vram_bytes", 0)),
        licence=raw.get("licence", "unspecified"),
        permitted_contexts=tuple(raw.get("permitted_contexts", ()) or ()),
        native_label_space=raw.get("native_label_space", ""),
        unmapped_policy=raw.get("unmapped_policy", "drop"),
        mappings=tuple(
            MappingEntryDeclaration(
                native_label=entry["native_label"],
                class_id=entry["class_id"],
                mapping_confidence=float(entry.get("mapping_confidence", 1.0)),
                notes=entry.get("notes", ""),
            )
            for entry in raw.get("mappings", ()) or ()
        ),
        calibration_id=raw.get("calibration_id"),
        runtime_options=tuple(
            (str(k), str(v)) for k, v in (raw.get("runtime_options", {}) or {}).items()
        ),
        enabled=bool(raw.get("enabled", True)),
    )


def _build_profile(raw: dict[str, Any]) -> ProfileDeclaration:
    return ProfileDeclaration(
        profile_id=raw["profile_id"],
        target_fps=float(raw["target_fps"]),
        max_in_flight=int(raw.get("max_in_flight", 4)),
        priority_class=raw.get("priority_class", "default"),
        inference_width=int(raw.get("inference_width", 640)),
        inference_height=int(raw.get("inference_height", 640)),
    )


def _build_region(raw: dict[str, Any]) -> RegionDeclaration:
    return RegionDeclaration(
        region_id=raw["region_id"],
        label=raw.get("label", ""),
        vertices=tuple((float(x), float(y)) for x, y in raw.get("vertices", ())),
        frame_of_reference=raw.get("frame_of_reference", "normalized"),
        camera_id=raw.get("camera_id"),
        version=raw.get("version", "1.0.0"),
    )


def _build_camera(raw: dict[str, Any]) -> CameraDeclaration:
    calibration_raw = raw.get("calibration")
    calibration = None
    if calibration_raw:
        homography = calibration_raw.get("homography")
        calibration = CalibrationDeclaration(
            calibration_id=calibration_raw["calibration_id"],
            homography=(
                tuple(tuple(float(v) for v in row) for row in homography) if homography else None
            ),
            ground_uncertainty_at_unit_distance=float(
                calibration_raw.get("ground_uncertainty_at_unit_distance", 0.05)
            ),
        )
    return CameraDeclaration(
        camera_id=raw["camera_id"],
        tenant_id=raw["tenant_id"],
        site_id=raw["site_id"],
        uri=raw["uri"],
        transport=raw["transport"],
        source_semantics=raw["source_semantics"],
        profile_id=raw["profile_id"],
        width=int(raw.get("width", 1920)),
        height=int(raw.get("height", 1080)),
        fps=float(raw.get("fps", 25.0)),
        codec=raw.get("codec", "raw"),
        colour_space=raw.get("colour_space", "bgr24"),
        credential_ref=raw.get("credential_ref"),
        privacy_policy_id=raw.get("privacy_policy_id"),
        region_ids=tuple(raw.get("region_ids", ()) or ()),
        calibration=calibration,
        labels=dict(raw.get("labels", {}) or {}),
        source_options=tuple((str(k), str(v)) for k, v in (raw.get("source_options", {}) or {}).items()),
    )
