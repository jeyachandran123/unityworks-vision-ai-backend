"""Architecture guards for the Object Registry.

These convert the Flow 4 boundaries from claims in a report into rules the build
enforces. The central ones are the two the brief calls constitutionally critical:
**Track belongs to M6, Object belongs to M7, never merged**, and **M7 is the sole
canonical owner of Vision Objects**.

Everything scans the AST rather than raw text. Docstrings naming later modules
are documentation, not coupling; what matters is whether code can reach a type.
"""

from __future__ import annotations

import ast
from pathlib import Path

import vision_os as vision_os_pkg
from vision_os.core.model.visual_object import LifecycleState, VisualObject
from vision_os.kernel.plugins.manifest import BINDABLE_PORTS, PortCatalogue

ROOT = Path(vision_os_pkg.__file__).parent
PERCEPTION = ROOT / "perception"
REGISTRY = PERCEPTION / "registry"
TRACKING = PERCEPTION / "tracking"
DETECTION = PERCEPTION / "detection"
KERNEL = ROOT / "kernel"
CORE = ROOT / "core"
ADAPTERS = ROOT / "adapters"

REGISTRY_ROOT = "registry_bootstrap.py"


def _files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


def _module_of(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _identifiers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            names.update(alias.name for alias in node.names)
    return names


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


class TestTrackAndObjectStaySeparate:
    """The boundary the brief calls constitutionally critical."""

    def test_no_track_field_exists_on_the_object(self) -> None:
        fields = set(VisualObject.__dataclass_fields__)
        for forbidden in (
            "track_id", "tracker_epoch", "coast_frames", "hit_count",
            "age_frames", "break_reason", "motion", "motion_state",
            "measurement_basis", "association_confidence",
        ):
            assert forbidden not in fields, (
                f"'{forbidden}' belongs to Track (M6); merging it into "
                f"VisualObject collapses the boundary V10 exists to hold"
            )

    def test_the_registry_does_not_re_associate_detections(self) -> None:
        """M7 consumes tracks. Re-associating detections would be doing M6's job."""
        offenders: list[str] = []
        for path in _files(REGISTRY):
            for identifier in _identifiers(path):
                if identifier in ("Detection", "DetectionOutcome", "RawDetection"):
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, "\n".join(offenders)

    def test_the_registry_never_imports_the_tracking_layer(self) -> None:
        """It consumes ``TrackUpdate`` from the object model, not M6 itself."""
        offenders: list[str] = []
        for path in _files(REGISTRY):
            for module in _imports(path):
                if "perception.tracking" in module or module.endswith("tracking.engine"):
                    offenders.append(f"{_module_of(path)} imports {module}")
        assert not offenders, "\n".join(offenders)

    def test_tracking_never_imports_the_registry(self) -> None:
        """The dependency runs one way. A tracker that knew about objects would
        be re-deciding identity."""
        offenders: list[str] = []
        for path in _files(TRACKING) + _files(ADAPTERS / "tracking"):
            for module in _imports(path):
                if "registry" in module:
                    offenders.append(f"{_module_of(path)} imports {module}")
        assert not offenders, "\n".join(offenders)

    def test_a_track_id_is_not_an_object_id(self) -> None:
        from vision_os.core.model.ids import (
            CameraId,
            LocalTrackId,
            ObjectId,
            TrackerEpoch,
            TrackId,
        )

        track = TrackId(CameraId("cam-01"), TrackerEpoch(0), LocalTrackId(1))
        assert not isinstance(track, str), "a TrackId is composite, not an id string"
        assert isinstance(ObjectId("01JB0000000000000000000001"), str)


class TestCanonicalOwnership:
    def test_the_object_is_immutable(self) -> None:
        assert VisualObject.__dataclass_params__.frozen

    def test_no_module_outside_the_registry_writes_an_object(self) -> None:
        """Future modules consume objects; they never mutate them.

        ``dataclasses.replace`` on a ``VisualObject`` outside the registry would
        be a second writer producing a divergent copy.
        """
        offenders: list[str] = []
        for path in _files(ROOT):
            module = _module_of(path)
            if module.startswith(("perception/registry/", "adapters/registry/", "conformance/")):
                continue
            text = path.read_text(encoding="utf-8")
            if "replace(" in text and "VisualObject" in text:
                offenders.append(module)
        assert not offenders, "\n".join(offenders)

    def test_the_partition_holds_no_lock(self) -> None:
        """Safety comes from the actor owning it, not from locks inside it.

        A lock here would suggest concurrent writers are expected, licensing
        exactly the design the sharding model exists to prevent.
        """
        text = (REGISTRY / "partition.py").read_text(encoding="utf-8")
        assert "threading" not in text
        assert "Lock()" not in text

    def test_object_ids_are_minted_from_the_injected_clock(self) -> None:
        """V13 — identity generation must not reintroduce hidden time.

        ``new_ulid`` encodes a timestamp. Calling it without ``now_ms`` reads the
        wall clock, which would make object ids unreplayable and a deterministic
        run non-reproducible.
        """
        offenders: list[str] = []
        for path in _files(REGISTRY):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "new_ulid"
                ):
                    supplied = {kw.arg for kw in node.keywords}
                    if "now_ms" not in supplied:
                        offenders.append(f"{_module_of(path)}:{node.lineno}")
        assert not offenders, (
            "these call sites mint an id from the wall clock rather than the "
            "injected one:\n" + "\n".join(offenders)
        )


class TestNoIdentityBeyondObjects:
    """12_SECURITY — M7 owns Objects, not identities."""

    def test_no_person_or_biometric_field_exists(self) -> None:
        fields = set(VisualObject.__dataclass_fields__)
        for forbidden in (
            "person_id", "name", "identity", "face_id", "biometric_id",
            "embedding", "descriptor", "gallery_id", "global_id",
        ):
            assert forbidden not in fields

    def test_no_registry_module_names_a_biometric_concept(self) -> None:
        forbidden = {
            "face", "faces", "biometric", "biometrics", "reidentification",
            "reid", "gallery", "fingerprint", "iris", "gait",
        }
        offenders: list[str] = []
        for path in _files(REGISTRY) + _files(ADAPTERS / "registry"):
            for identifier in _identifiers(path):
                tokens = {
                    t for t in identifier.lower().replace(".", "_").split("_") if t
                }
                for word in tokens & forbidden:
                    offenders.append(f"{_module_of(path)}::{identifier} uses '{word}'")
        assert not offenders, "\n".join(offenders)

    def test_the_embedding_port_stays_unbindable(self) -> None:
        assert PortCatalogue.EMBEDDING not in BINDABLE_PORTS

    def test_the_identity_resolver_port_stays_unbindable(self) -> None:
        """P11 is M7's port, yet has no implementations in Phase 1."""
        assert PortCatalogue.IDENTITY_RESOLVER not in BINDABLE_PORTS

    def test_no_cross_camera_structure_exists(self) -> None:
        offenders: list[str] = []
        for path in _files(REGISTRY):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            lowered = target.attr.lower()
                            for word in ("global_", "cross_camera", "site_objects"):
                                if word in lowered:
                                    offenders.append(f"{_module_of(path)}::{target.attr}")
        assert not offenders, "\n".join(offenders)

    def test_an_object_id_is_site_scoped_not_global(self) -> None:
        """02_VOM section 4.1 — cross-site identity is a federation concern."""
        fields = set(VisualObject.__dataclass_fields__)
        assert "site_id" in fields
        assert "global_identity" not in fields


class TestSemanticCeiling:
    def test_no_domain_vocabulary_in_the_registry(self) -> None:
        domain = {
            "customer", "staff", "employee", "waiter", "shopper", "patient",
            "nurse", "doctor", "intruder", "suspect", "queueing", "loitering",
            "abandoned", "suspicious", "theft", "checkout", "till", "alert",
            "violation", "compliance", "incident", "crowded", "overcrowded",
        }
        offenders: list[str] = []
        for path in _files(REGISTRY) + _files(ADAPTERS / "registry"):
            for identifier in _identifiers(path):
                tokens = {
                    t for t in identifier.lower().replace(".", "_").split("_") if t
                }
                for word in tokens & domain:
                    offenders.append(f"{_module_of(path)}::{identifier} uses '{word}'")
        assert not offenders, "\n".join(offenders)

    def test_region_occupancy_carries_no_judgment(self) -> None:
        """07_STATE section 3.3 — counting only, no threshold."""
        from vision_os.perception.registry.regions import RegionOccupancy

        fields = set(RegionOccupancy.__dataclass_fields__)
        for forbidden in (
            "is_crowded", "exceeds_capacity", "queue_forming", "capacity",
            "threshold", "alert", "status",
        ):
            assert forbidden not in fields

    def test_the_attribute_gate_exists_and_rejects(self) -> None:
        """14_TESTING section 6 names the registry gate as a V1 enforcement point."""
        import pytest

        from vision_os.core.errors import AttributeRejectedError
        from vision_os.core.model.ids import AttributeKey
        from vision_os.perception.registry.attributes import check_neutrality

        with pytest.raises(AttributeRejectedError):
            check_neutrality(AttributeKey("is_employee"), "The uniform indicates a job")

    def test_no_metric_claims_a_judgment(self) -> None:
        from vision_os.kernel.metrics import MetricName

        for attribute in dir(MetricName):
            if attribute.startswith("_"):
                continue
            value = getattr(MetricName, attribute)
            if not isinstance(value, str) or "registry" not in value:
                continue
            for forbidden in ("false", "wrong", "incorrect", "accuracy", "violation"):
                assert forbidden not in value, (
                    f"{attribute} = '{value}' claims a judgment the platform "
                    f"cannot make without ground truth (V1)"
                )

    def test_no_business_event_type_exists(self) -> None:
        from vision_os.kernel.events import ALL_EVENT_TYPES

        for event in ALL_EVENT_TYPES:
            if not event.event_type.startswith("registry."):
                continue
            for forbidden in ("alert", "violation", "incident", "threshold"):
                assert forbidden not in event.event_type


class TestLayerBoundaries:
    def test_the_registry_produces_no_attributes(self) -> None:
        """M7 holds; M9 produces. Holding is storage, producing is inference."""
        offenders: list[str] = []
        for path in _files(REGISTRY):
            for identifier in _identifiers(path):
                if identifier in ("UnderstanderPort", "PromptManager", "understand"):
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, "\n".join(offenders)

    def test_the_registry_builds_no_observations(self) -> None:
        """Schema and ceiling enforcement is M11's single choke point."""
        offenders: list[str] = []
        forbidden = {"Observation", "ObservationBuilder", "ObservationId", "Evidence"}
        for path in _files(REGISTRY):
            for identifier in _identifiers(path) & forbidden:
                offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, "\n".join(offenders)

    def test_the_registry_writes_no_vision_state(self) -> None:
        """Vision State is projected from the observation log by M13 at L6."""
        offenders: list[str] = []
        for path in _files(REGISTRY):
            for module in _imports(path):
                lowered = module.lower()
                if "observation_log" in lowered or lowered.endswith(".state"):
                    offenders.append(f"{_module_of(path)} imports {module}")
        assert not offenders, "\n".join(offenders)

    def test_the_registry_never_touches_pixels(self) -> None:
        """V12 — M7 consumes control-plane data only."""
        offenders: list[str] = []
        for path in _files(REGISTRY):
            for identifier in _identifiers(path):
                if identifier in (
                    "Frame", "FrameBuffer", "PixelBuffer", "pixels", "Crop", "CropId"
                ):
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, "\n".join(offenders)

    def test_the_registry_imports_no_adapter(self) -> None:
        offenders: list[str] = []
        for path in _files(REGISTRY):
            for module in _imports(path):
                if "adapters" in module:
                    offenders.append(f"{_module_of(path)} imports {module}")
        assert not offenders, "\n".join(offenders)

    def test_core_imports_no_flow_layer(self) -> None:
        offenders: list[str] = []
        for path in _files(CORE):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module or node.level:
                    continue
                for forbidden in ("perception", "acquisition", "kernel", "adapters"):
                    if forbidden in node.module:
                        offenders.append(f"{_module_of(path)} imports {node.module}")
        assert not offenders, "\n".join(offenders)

    def test_only_the_composition_root_selects_a_store(self) -> None:
        named: list[str] = []
        for path in _files(ROOT):
            module = _module_of(path)
            if module.startswith(("adapters/", "conformance/")) or module == REGISTRY_ROOT:
                continue
            text = path.read_text(encoding="utf-8")
            for symbol in ("FileObjectStore", "InMemoryObjectStore"):
                if symbol in text:
                    named.append(f"{module} names {symbol}")
        assert not named, "\n".join(named)


class TestDeterminismAndTime:
    def test_no_registry_module_reads_the_wall_clock(self) -> None:
        offenders: list[str] = []
        for path in _files(REGISTRY):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in (
                    "time", "time_ns", "monotonic_ns", "perf_counter", "utcnow"
                ):
                    if isinstance(node.value, ast.Name) and node.value.id in (
                        "time", "datetime"
                    ):
                        offenders.append(f"{_module_of(path)} reads {node.value.id}.{node.attr}")
        assert not offenders, "\n".join(offenders)

    def test_no_registry_module_uses_unseeded_randomness(self) -> None:
        offenders: list[str] = []
        for path in _files(REGISTRY):
            for module in _imports(path):
                if module in ("random", "secrets"):
                    offenders.append(f"{_module_of(path)} imports {module}")
        assert not offenders, "\n".join(offenders)

    def test_the_lifecycle_is_a_closed_set(self) -> None:
        assert len(LifecycleState) == 7


class TestFlowScope:
    def test_the_state_store_port_is_bindable(self) -> None:
        assert PortCatalogue.STATE_STORE in BINDABLE_PORTS

    def test_phase_two_ports_remain_unbindable(self) -> None:
        """Phase 1 is complete; these are what it deliberately omits."""
        for port in (
            PortCatalogue.EMBEDDING,
            PortCatalogue.IDENTITY_RESOLVER,
            PortCatalogue.PROMPT_SOURCE,
            PortCatalogue.CALIBRATION,
        ):
            assert port not in BINDABLE_PORTS, f"{port} became bindable before its flow"

    def test_the_registry_does_not_import_exposure(self) -> None:
        """M7 gained a second consumer in Flow 8 and must still not have noticed."""
        offenders = [
            path.name
            for path in (ROOT / "perception" / "registry").rglob("*.py")
            if "exposure" in path.read_text(encoding="utf-8")
        ]
        assert not offenders, "; ".join(offenders)

    def test_the_registry_does_not_import_synthesis(self) -> None:
        """M7 gained a consumer in Flow 7 and must not have noticed.

        The registry publishes through a callable it was given. If it imported
        the Observation Builder, the dependency would run downward — L2 knowing
        about L5 — and the registry could no longer be tested, or deployed,
        without the layer above it.
        """
        offenders = []
        for path in (ROOT / "perception" / "registry").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for forbidden in ("synthesis", "vision_state"):
                if f"import {forbidden}" in source or f"{forbidden} import" in source:
                    offenders.append(f"{path.name} imports {forbidden}")
        assert not offenders, "\n".join(offenders)

    def test_the_registry_output_is_only_objects(self) -> None:
        from vision_os.perception.registry import RegistryUpdate

        annotation = str(RegistryUpdate.__dataclass_fields__["objects"].type)
        assert "VisualObject" in annotation

    def test_the_registry_update_names_no_later_flow_type(self) -> None:
        from vision_os.perception.registry import RegistryUpdate

        fields = set(RegistryUpdate.__dataclass_fields__)
        for forbidden in ("observations", "crops", "triggers", "claims"):
            assert forbidden not in fields


class TestEarlierFlowsUnchanged:
    def test_flow_one_does_not_import_the_registry(self) -> None:
        offenders: list[str] = []
        for directory in (ROOT / "acquisition", KERNEL):
            for path in _files(directory):
                for module in _imports(path):
                    if "registry" in module and "plugins" not in module:
                        offenders.append(f"{_module_of(path)} imports {module}")
        assert not offenders, "\n".join(offenders)

    def test_flow_two_does_not_import_the_registry(self) -> None:
        offenders: list[str] = []
        for path in _files(DETECTION):
            for module in _imports(path):
                if "registry" in module:
                    offenders.append(f"{_module_of(path)} imports {module}")
        assert not offenders, "\n".join(offenders)

    def test_the_detection_seam_is_untouched(self) -> None:
        import inspect

        from vision_os.core.ports.pipeline import DetectionConsumer

        signature = inspect.signature(DetectionConsumer.on_detected)
        assert "outcome" in signature.parameters

    def test_the_admitted_frame_seam_is_untouched(self) -> None:
        import inspect

        from vision_os.core.ports.pipeline import AdmittedFrameConsumer

        source = inspect.getsource(AdmittedFrameConsumer)
        assert "frame_ref" in source
        assert "VisualObject" not in source

    def test_the_tracker_port_is_untouched(self) -> None:
        from vision_os.core.ports.tracking import TrackerPort

        surface = {
            name
            for name in dir(TrackerPort)
            if not name.startswith("_") and callable(getattr(TrackerPort, name, None))
        }
        assert surface == {"update", "tracks", "reset", "capabilities"}
