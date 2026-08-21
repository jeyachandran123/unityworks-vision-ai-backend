"""Architecture guards specific to the tracking layer.

These convert the Flow 3 constraints from claims in a report into rules the build
enforces. The central one mirrors Flow 2's: **the platform does not know
ByteTrack exists**, proved the same way YOLO's invisibility was proved.

Everything scans the AST rather than raw text. Docstrings naming example
implementations are documentation, not coupling; what matters is whether code can
reach a concrete type.
"""

from __future__ import annotations

import ast
from pathlib import Path

import vision_os as vision_os_pkg
from vision_os.core.model.track import Track, TrackState, TrackUpdate
from vision_os.core.ports.tracking import TrackerPort
from vision_os.kernel.plugins.manifest import BINDABLE_PORTS, PortCatalogue

ROOT = Path(vision_os_pkg.__file__).parent
PERCEPTION = ROOT / "perception"
TRACKING = PERCEPTION / "tracking"
KERNEL = ROOT / "kernel"
CORE = ROOT / "core"
ADAPTERS = ROOT / "adapters"

#: The composition root is the one module allowed to name a tracker.
TRACKING_ROOT = "tracking_bootstrap.py"


def _files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


def _module_of(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _identifiers(path: Path) -> set[str]:
    """Every name a module defines or uses — excluding prose."""
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


def _attribute_targets(path: Path) -> set[str]:
    """``self.x`` names assigned anywhere in the module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    names.add(target.attr)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Attribute):
            if isinstance(node.target.value, ast.Name) and node.target.value.id == "self":
                names.add(node.target.attr)
    return names


class TestPlatformDoesNotKnowByteTrackExists:
    """Invariant V3, exactly as Flow 2 proved it for YOLO."""

    def test_no_platform_module_names_a_tracker_vendor(self) -> None:
        """Vendor names, as whole identifier tokens.

        ``sort`` is deliberately absent from this list: it is also a builtin
        method, and a rule that flags ``list.sort()`` is a rule people delete.
        The SORT *tracker* is caught by
        ``test_only_the_composition_root_names_a_concrete_tracker`` instead,
        which matches the factory symbol rather than an English word.
        """
        vendors = (
            "bytetrack",
            "byte_track",
            "deepsort",
            "deep_sort",
            "botsort",
            "bot_sort",
            "ocsort",
            "oc_sort",
            "strongsort",
            "motr",
            "fairmot",
            "norfair",
            "ultralytics",
        )
        offenders: list[str] = []
        for directory in (CORE, KERNEL, PERCEPTION):
            for path in _files(directory):
                for identifier in _identifiers(path):
                    lowered = identifier.lower()
                    for vendor in vendors:
                        if vendor in lowered:
                            offenders.append(
                                f"{_module_of(path)}::{identifier} names '{vendor}'"
                            )
        assert not offenders, (
            "the platform must not be able to reach a concrete tracker:\n"
            + "\n".join(offenders)
        )

    def test_only_the_composition_root_names_a_concrete_tracker(self) -> None:
        """One module in the codebase may name a tracker implementation."""
        named: list[str] = []
        symbols = (
            "build_bytetrack_tracker",
            "build_sort_tracker",
            "build_iou_tracker",
            "GeometricTracker",
            "TRACKER_FACTORIES",
            "tracker.bytetrack",
            "tracker.sort",
        )
        for path in _files(ROOT):
            module = _module_of(path)
            if module.startswith("adapters/") or module == TRACKING_ROOT:
                continue
            text = path.read_text(encoding="utf-8")
            for symbol in symbols:
                if symbol in text:
                    named.append(f"{module} names {symbol}")
        assert not named, "\n".join(named)

    def test_the_tracking_layer_imports_no_adapter(self) -> None:
        offenders: list[str] = []
        for path in _files(TRACKING):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and "adapters" in node.module:
                    offenders.append(f"{_module_of(path)} imports {node.module}")
        assert not offenders, "\n".join(offenders)

    def test_the_manager_holds_a_port_not_a_tracker(self) -> None:
        from vision_os.perception.tracking.manager import TrackerBinding

        annotation = TrackerBinding.__dataclass_fields__["tracker"].type
        assert "TrackerPort" in str(annotation)

    def test_swapping_trackers_touches_only_configuration(self) -> None:
        """The whole mapping from a name to an implementation is one dict."""
        from vision_os.adapters.tracking import TRACKER_FACTORIES

        assert set(TRACKER_FACTORIES) == {
            "tracker.iou",
            "tracker.sort",
            "tracker.bytetrack",
        }


class TestTrackingOwnsNoIdentity:
    """Invariant V10 — the boundary between M6 and M7."""

    def test_a_track_carries_no_object_id(self) -> None:
        assert "object_id" not in Track.__dataclass_fields__

    def test_a_track_carries_no_person_or_identity_field(self) -> None:
        forbidden = {
            "person_id",
            "identity",
            "identity_id",
            "name",
            "face_id",
            "global_id",
            "global_track_id",
            "biometric_id",
        }
        assert not (forbidden & set(Track.__dataclass_fields__))

    def test_no_tracking_module_mints_an_object_id(self) -> None:
        """Durable identity is minted by the Object Registry alone (M7)."""
        offenders: list[str] = []
        for path in _files(TRACKING) + _files(ADAPTERS / "tracking"):
            identifiers = {i.lower() for i in _identifiers(path)}
            for forbidden in ("objectid", "object_id", "identityassertion"):
                if forbidden in identifiers:
                    offenders.append(f"{_module_of(path)} references {forbidden}")
        assert not offenders, "\n".join(offenders)

    def test_no_tracking_module_uses_identity_confidence(self) -> None:
        """``IDENTITY`` semantics belong to M7; a track asserts ``ASSOCIATION``."""
        offenders: list[str] = []
        for path in _files(TRACKING) + _files(ADAPTERS / "tracking"):
            text = path.read_text(encoding="utf-8")
            if "ConfidenceSemantics.IDENTITY" in text:
                offenders.append(_module_of(path))
        assert not offenders, "\n".join(offenders)

    def test_the_identity_resolver_port_is_not_bindable(self) -> None:
        """Cross-camera identity is P11, Phase 2. Defined, unused, unbindable."""
        assert PortCatalogue.IDENTITY_RESOLVER not in BINDABLE_PORTS

    def test_a_track_id_cannot_be_compared_across_cameras(self) -> None:
        """Structural, not conventional: the camera is inside the identifier."""
        from vision_os.core.model.ids import CameraId, LocalTrackId, TrackerEpoch, TrackId

        first = TrackId(CameraId("cam-01"), TrackerEpoch(0), LocalTrackId(5))
        second = TrackId(CameraId("cam-02"), TrackerEpoch(0), LocalTrackId(5))
        assert first != second


class TestNoCrossCameraState:
    """Port obligation T7."""

    def test_no_tracking_module_holds_a_cross_camera_structure(self) -> None:
        offenders: list[str] = []
        for path in _files(TRACKING) + _files(ADAPTERS / "tracking"):
            for attribute in _attribute_targets(path):
                lowered = attribute.lower()
                for forbidden in ("global_", "site_tracks", "all_tracks", "cross_camera"):
                    if forbidden in lowered:
                        offenders.append(f"{_module_of(path)}::self.{attribute}")
        assert not offenders, "\n".join(offenders)

    def test_the_track_table_is_scoped_to_one_camera(self) -> None:
        from vision_os.perception.tracking.table import TrackTable

        assert "camera_id" in TrackTable.__init__.__code__.co_varnames

    def test_the_embedding_port_is_not_bindable(self) -> None:
        """Appearance embeddings are C2 biometric data, disabled by default
        (12_SECURITY section 4.3). Making the port bindable would let a
        deployment enable the platform's most invasive capability by dropping in
        a plugin."""
        assert PortCatalogue.EMBEDDING not in BINDABLE_PORTS

    def test_no_embedding_adapter_ships(self) -> None:
        adapters = ADAPTERS / "tracking"
        for path in _files(adapters):
            text = path.read_text(encoding="utf-8").lower()
            assert "class embedding" not in text
            assert "def embed(" not in text


class TestSemanticCeiling:
    """Invariant V1 — tracking states what it sees, never what it means."""

    def test_no_domain_vocabulary_in_the_tracking_layer(self) -> None:
        """Words that could only be a business judgment.

        Deliberately excludes ambiguous ones. ``table`` is a data structure here
        and a restaurant fixture elsewhere; ``order`` is frame ordering. A guard
        that flags ``TrackTable`` and ``out_of_order`` teaches people to
        suppress it, and a suppressed guard protects nothing.
        """
        domain = {
            "customer",
            "staff",
            "employee",
            "waiter",
            "shopper",
            "queueing",
            "loitering",
            "dwelling",
            "abandoned",
            "suspicious",
            "intruder",
            "theft",
            "shoplifting",
            "patient",
            "nurse",
            "doctor",
            "checkout",
            "till",
            "alert",
            "violation",
            "compliance",
        }
        offenders: list[str] = []
        for path in _files(TRACKING) + _files(ADAPTERS / "tracking"):
            for identifier in _identifiers(path):
                tokens = {
                    t for t in identifier.lower().replace(".", "_").split("_") if t
                }
                for word in tokens & domain:
                    offenders.append(f"{_module_of(path)}::{identifier} uses '{word}'")
        assert not offenders, "\n".join(offenders)

    def test_motion_states_are_descriptive_not_judgmental(self) -> None:
        from vision_os.core.model.track import MotionState

        assert {s.value for s in MotionState} == {
            "stationary",
            "moving",
            "erratic",
            "unknown",
        }

    def test_no_metric_claims_a_wrong_association(self) -> None:
        """Whether an association was *wrong* needs ground truth the platform
        does not have. What it can count is what it refused."""
        from vision_os.kernel.metrics import MetricName

        for attribute in dir(MetricName):
            if attribute.startswith("_"):
                continue
            value = getattr(MetricName, attribute)
            if not isinstance(value, str) or "tracking" not in value:
                continue
            for forbidden in ("false", "wrong", "incorrect", "id_switch", "error_rate"):
                assert forbidden not in value, (
                    f"{attribute} = '{value}' claims a judgment the platform "
                    f"cannot make without ground truth (invariant V1)"
                )

    def test_the_break_reason_set_is_diagnostic_not_interpretive(self) -> None:
        from vision_os.core.model.track import BreakReason

        assert {r.value for r in BreakReason} == {
            "none",
            "occlusion",
            "exit",
            "detector_miss",
            "association_failure",
            "epoch_reset",
        }


class TestFlowScope:
    """Flow 3 ends when tracked objects are emitted."""

    def test_the_tracker_port_is_bindable(self) -> None:
        assert PortCatalogue.TRACKER in BINDABLE_PORTS

    def test_phase_two_ports_remain_unbindable(self) -> None:
        """Phase 1 is complete; the frontier has stopped moving."""
        for port in (
            PortCatalogue.EMBEDDING,
            PortCatalogue.IDENTITY_RESOLVER,
            PortCatalogue.PROMPT_SOURCE,
            PortCatalogue.CALIBRATION,
        ):
            assert port not in BINDABLE_PORTS, f"{port} became bindable before its flow"

    def test_tracking_does_not_import_exposure(self) -> None:
        offenders = [
            path.name
            for path in (ROOT / "perception" / "tracking").rglob("*.py")
            if "exposure" in path.read_text(encoding="utf-8")
        ]
        assert not offenders, "; ".join(offenders)

    def test_tracking_does_not_import_the_crop_manager(self) -> None:
        """M8 ships, but M6 must not learn it exists.

        The dependency runs cropping-to-tracking through the registry, never the
        reverse: a tracker that knew what was being cropped would be making
        attention decisions, which belong two layers up.
        """
        offenders: list[str] = []
        for path in _files(TRACKING) + _files(ADAPTERS / "tracking"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and "cropping" in node.module:
                    offenders.append(f"{_module_of(path)} imports {node.module}")
        assert not offenders, "\n".join(offenders)

    def test_tracking_does_not_import_the_registry(self) -> None:
        """M7 ships, but M6 must not learn it exists.

        The dependency runs registry-to-tracking, never the reverse: a tracker
        that knew about objects would be re-deciding identity, which is exactly
        the fusion V10 exists to prevent.
        """
        offenders: list[str] = []
        for path in _files(TRACKING) + _files(ADAPTERS / "tracking"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and "registry" in node.module:
                    offenders.append(f"{_module_of(path)} imports {node.module}")
        assert not offenders, "\n".join(offenders)

    def test_tracking_never_names_an_object_id(self) -> None:
        """``ObjectId`` is minted by M7 alone (01_LAYERED section 8)."""
        offenders: list[str] = []
        for path in _files(TRACKING) + _files(ADAPTERS / "tracking"):
            for identifier in _identifiers(path):
                if identifier in ("ObjectId", "VisualObject", "ObjectRegistry"):
                    offenders.append(f"{_module_of(path)}::{identifier}")
        assert not offenders, "\n".join(offenders)

    def test_tracking_emits_no_observations(self) -> None:
        """The platform ``Observation`` type is Flow 6 and must not appear here.

        ``MotionObservation`` is excluded by name: it is a measurement fed to a
        motion model, unrelated to the observation envelope.
        """
        offenders: list[str] = []
        forbidden_types = {"Observation", "ObservationBuilder", "ObservationId"}
        for path in _files(TRACKING) + _files(ADAPTERS / "tracking"):
            for identifier in _identifiers(path) & forbidden_types:
                offenders.append(f"{_module_of(path)} references {identifier}")
        assert not offenders, "\n".join(offenders)

    def test_tracking_writes_no_vision_state(self) -> None:
        offenders: list[str] = []
        for path in _files(TRACKING) + _files(ADAPTERS / "tracking"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    lowered = node.module.lower()
                    if "state_store" in lowered or lowered.endswith(".state"):
                        offenders.append(f"{_module_of(path)} imports {node.module}")
        assert not offenders, "\n".join(offenders)

    def test_tracking_produces_only_tracks(self) -> None:
        """The engine's single output type."""
        from vision_os.perception.tracking import TrackingOutcome

        annotation = str(TrackingOutcome.__dataclass_fields__["tracks"].type)
        assert "Track" in annotation

    def test_the_track_update_names_no_later_flow_type(self) -> None:
        fields = set(TrackUpdate.__dataclass_fields__)
        for forbidden in ("observations", "crops", "attributes", "objects"):
            assert forbidden not in fields


class TestDeterminismAndTime:
    def test_no_tracking_module_reads_the_wall_clock(self) -> None:
        """Invariant V13. Time is injected, never sampled."""
        offenders: list[str] = []
        for path in _files(TRACKING) + _files(ADAPTERS / "tracking"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in (
                    "time",
                    "time_ns",
                    "monotonic_ns",
                    "perf_counter",
                    "utcnow",
                    "now",
                ):
                    if isinstance(node.value, ast.Name) and node.value.id in (
                        "time",
                        "datetime",
                    ):
                        offenders.append(f"{_module_of(path)} reads {node.value.id}.{node.attr}")
        assert not offenders, "\n".join(offenders)

    def test_no_tracking_module_uses_unseeded_randomness(self) -> None:
        offenders: list[str] = []
        for path in _files(TRACKING) + _files(ADAPTERS / "tracking"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ("random", "secrets"):
                            offenders.append(f"{_module_of(path)} imports {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module in (
                    "random",
                    "secrets",
                ):
                    offenders.append(f"{_module_of(path)} imports from {node.module}")
        assert not offenders, "\n".join(offenders)

    def test_the_lifecycle_is_a_closed_set(self) -> None:
        assert len(TrackState) == 5


class TestPortPurity:
    def test_the_tracker_port_has_no_identity_method(self) -> None:
        for forbidden in ("identify", "recognize", "reidentify", "match_across", "embed"):
            assert not hasattr(TrackerPort, forbidden)

    def test_the_tracker_port_surface_is_the_documented_four(self) -> None:
        """03_MODULES M6 public API, implemented verbatim."""
        surface = {
            name
            for name in dir(TrackerPort)
            if not name.startswith("_") and callable(getattr(TrackerPort, name, None))
        }
        assert surface == {"update", "tracks", "reset", "capabilities"}

    def test_core_ports_import_no_flow_layer(self) -> None:
        """Core is the innermost layer and depends on nothing above it.

        Relative imports are skipped: ``from .acquisition import ...`` inside
        ``core/ports`` names a sibling *port module*, not the acquisition flow
        layer, and conflating the two would flag correct code.
        """
        offenders: list[str] = []
        for path in _files(CORE):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                if node.level:  # relative — a sibling within core
                    continue
                for forbidden in ("perception", "acquisition", "kernel", "adapters"):
                    if forbidden in node.module:
                        offenders.append(f"{_module_of(path)} imports {node.module}")
        assert not offenders, "\n".join(offenders)


class TestFlowOneAndTwoUnchanged:
    """The seams stay minimal and optional."""

    def test_the_detection_seam_carries_an_outcome_not_a_tracker(self) -> None:
        """The seam names Flow 2's output type and no Flow 3 type.

        Checked on the signature rather than the source, because the docstring
        legitimately says "Detection-to-Tracking" — prose describing the seam is
        not coupling to it.
        """
        import inspect

        from vision_os.core.model.detection import DetectionOutcome
        from vision_os.core.ports.pipeline import DetectionConsumer

        signature = inspect.signature(DetectionConsumer.on_detected)
        annotation = signature.parameters["outcome"].annotation
        assert annotation in (DetectionOutcome, "DetectionOutcome")

        namespace = vars(DetectionConsumer)
        assert not any("Track" in str(v) and "Tracking" not in str(v) for v in namespace)

    def test_the_detection_consumer_is_optional(self) -> None:
        """Flow 2's behaviour with no consumer must be exactly what it was."""
        import inspect

        from vision_os.perception.detection import DetectionRuntime

        signature = inspect.signature(DetectionRuntime.__init__)
        assert signature.parameters["consumer"].default is None

    def test_the_admitted_frame_seam_is_untouched(self) -> None:
        import inspect

        from vision_os.core.ports.pipeline import AdmittedFrameConsumer

        source = inspect.getsource(AdmittedFrameConsumer)
        assert "frame_ref" in source
        assert "Track" not in source

    def test_flow_two_modules_do_not_import_tracking(self) -> None:
        """Detection must not learn that tracking exists."""
        offenders: list[str] = []
        for path in _files(PERCEPTION / "detection"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if "tracking" in node.module:
                        offenders.append(f"{_module_of(path)} imports {node.module}")
        assert not offenders, "\n".join(offenders)

    def test_flow_one_modules_do_not_import_tracking(self) -> None:
        offenders: list[str] = []
        for directory in (ROOT / "acquisition", KERNEL):
            for path in _files(directory):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if "tracking" in node.module or "perception" in node.module:
                            offenders.append(f"{_module_of(path)} imports {node.module}")
        assert not offenders, "\n".join(offenders)
