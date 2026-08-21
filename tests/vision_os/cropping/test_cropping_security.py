"""Security and privacy for the Crop Manager — 12_SECURITY.

> §M8 SECURITY: *"M8 owns images. Not identities. No biometric recognition. No
> face recognition. No person recognition. No identity persistence."*

Three properties are load-bearing, and each fails silently if untested:

**Every crop is C1 Imagery**, classified rather than inferred. A crop whose class
is inferred downstream is a crop that ends up in an unclassified path the first
time someone forgets.

**Every cache key includes the tenant** (12_SECURITY section 4). A cache keyed on
content alone lets one tenant's crop satisfy another tenant's request — a
cross-tenant data path wearing the disguise of an optimization.

**Pixels are data-plane and node-local** (V12). The crop reference travels; the
imagery does not.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import vision_os as vision_os_pkg
from vision_os.core.model.crop import Crop, PrivacyClass, RetentionMode
from vision_os.core.model.timebase import Duration
from vision_os.perception.cropping import CropDeduplicationCache

from .conftest import (
    OTHER_TENANT,
    TENANT,
    frame_context,
    make_demand,
    make_object,
    sharp_frame,
)

ROOT = Path(vision_os_pkg.__file__).parent
CROPPING = ROOT / "perception" / "cropping"
ADAPTERS = ROOT / "adapters" / "cropping"


def _files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


class TestDataClassification:
    def test_every_crop_is_c1_imagery(self, manager) -> None:
        """Fixed, not inferred (12_SECURITY section 3)."""
        manager.register_demand(make_demand())
        frame = frame_context()
        request = manager.evaluate([make_object()], frame).requests[0]
        crop = manager.extract(request, pixels=sharp_frame(), frame=frame)
        assert crop.privacy_class is PrivacyClass.C1_IMAGERY

    def test_no_crop_is_ever_classified_biometric(self, manager) -> None:
        """C2 is disabled by default; M8 must not be a route to it."""
        manager.register_demand(make_demand())
        frame = frame_context()
        request = manager.evaluate([make_object()], frame).requests[0]
        crop = manager.extract(request, pixels=sharp_frame(), frame=frame)
        assert crop.privacy_class is not PrivacyClass.C2_BIOMETRIC

    def test_the_privacy_class_is_not_configurable(self) -> None:
        from vision_os.kernel.config.schema import CroppingSection

        fields = set(CroppingSection.__dataclass_fields__)
        for forbidden in ("privacy_class", "data_class", "classification"):
            assert forbidden not in fields, (
                "a deployment that could downgrade a crop's classification could "
                "route imagery into an unclassified path by editing a file"
            )


class TestTenantIsolation:
    def test_every_cache_key_includes_the_tenant(self) -> None:
        """12_SECURITY section 4, checked at the key function itself."""
        mine = CropDeduplicationCache.key(TENANT, "identical-pixels")
        theirs = CropDeduplicationCache.key(OTHER_TENANT, "identical-pixels")
        assert mine != theirs
        assert str(TENANT) in mine[0]

    def test_one_tenants_crop_never_satisfies_anothers(self) -> None:
        from vision_os.core.model.ids import CropId

        cache = CropDeduplicationCache(capacity=16)
        cache.put(TENANT, "same-bytes", CropId("crop-a"))
        assert cache.get(OTHER_TENANT, "same-bytes") is None

    def test_erasure_reaches_the_cache(self) -> None:
        """A reference left alive after its data was deleted is a leak."""
        from vision_os.core.model.ids import CropId

        cache = CropDeduplicationCache(capacity=16)
        cache.put(TENANT, "a", CropId("a"))
        cache.put(TENANT, "b", CropId("b"))
        cache.put(OTHER_TENANT, "c", CropId("c"))
        assert cache.forget_tenant(TENANT) == 2
        assert len(cache) == 1

    def test_a_crop_carries_its_tenant(self, manager) -> None:
        manager.register_demand(make_demand())
        frame = frame_context()
        request = manager.evaluate([make_object(tenant=OTHER_TENANT)], frame).requests[0]
        crop = manager.extract(request, pixels=sharp_frame(), frame=frame)
        assert crop.tenant_id == OTHER_TENANT


class TestRetention:
    def test_ephemeral_is_the_default(self) -> None:
        """The cheapest and least exposed setting, chosen unless overridden."""
        from vision_os.kernel.config.schema import CroppingSection

        assert CroppingSection().retention_mode == "ephemeral"

    def test_evidence_retention_is_bounded(self) -> None:
        """12_SECURITY section 3 bounds C1 imagery at 24-72 hours."""
        from vision_os.kernel.config.schema import CroppingSection

        settings = CroppingSection()
        assert settings.evidence_ttl_ms <= 72 * 3_600_000

    def test_never_persist_is_expressible(self) -> None:
        """12_SECURITY section 2.3's no-evidence mode."""
        from vision_os.kernel.config.schema import CroppingSection

        settings = CroppingSection(retention_mode="never_persist")
        assert RetentionMode(settings.retention_mode) is RetentionMode.NEVER_PERSIST

    def test_an_unknown_retention_mode_is_refused(self) -> None:
        from vision_os.core.errors import ValidationError
        from vision_os.kernel.config.schema import CroppingSection

        with pytest.raises(ValidationError, match="retention_mode"):
            CroppingSection(retention_mode="forever")

    def test_the_manager_stamps_the_configured_retention(
        self, manager, cropping_config
    ) -> None:
        manager.register_demand(make_demand())
        frame = frame_context()
        request = manager.evaluate([make_object()], frame).requests[0]
        crop = manager.extract(request, pixels=sharp_frame(), frame=frame)
        assert crop.retention is RetentionMode(cropping_config.retention_mode)


class TestTheDataPlaneStaysLocal:
    def test_a_crop_reference_carries_no_pixels(self, manager) -> None:
        """Invariant V12, made mechanical."""
        manager.register_demand(make_demand())
        frame = frame_context()
        request = manager.evaluate([make_object()], frame).requests[0]
        crop = manager.extract(request, pixels=sharp_frame(), frame=frame)
        assert crop.pixels is not None
        assert crop.without_pixels().pixels is None

    def test_the_crop_event_surface_carries_no_imagery(self) -> None:
        """A lossy control-plane bus is the wrong transport for megabytes."""
        from vision_os.kernel.events import (
            BudgetExhausted,
            CapabilityGap,
            GateRejectionSpike,
        )

        for event in (BudgetExhausted, GateRejectionSpike, CapabilityGap):
            fields = set(event.__dataclass_fields__)
            for forbidden in ("pixels", "crop", "image", "payload", "bytes"):
                assert forbidden not in fields, f"{event.__name__} leaks imagery"

    def test_no_crop_event_type_exists(self) -> None:
        """Announcing every crop would put data-plane traffic on the bus."""
        from vision_os.kernel.events import ALL_EVENT_TYPES

        names = {event.event_type for event in ALL_EVENT_TYPES}
        assert "cropping.crop_produced" not in names


class TestNoBiometrics:
    def test_no_face_or_recognition_code_exists(self) -> None:
        forbidden = (
            "face_detect", "facial", "landmark", "keypoint_face", "iris",
            "gait", "fingerprint", "voiceprint", "recognise_person",
        )
        offenders: list[str] = []
        for path in _files(CROPPING) + _files(ADAPTERS):
            source = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                if token in source:
                    offenders.append(f"{path.name} contains '{token}'")
        assert not offenders, (
            "M8 owns images, not identities:\n" + "\n".join(offenders)
        )

    def test_the_embedding_port_is_not_reachable_from_cropping(self) -> None:
        """Appearance embeddings are C2, disabled by default."""
        offenders: list[str] = []
        for path in _files(CROPPING) + _files(ADAPTERS):
            source = path.read_text(encoding="utf-8")
            if "EmbeddingPort" in source or "IdentityResolverPort" in source:
                offenders.append(path.name)
        assert not offenders, "\n".join(offenders)

    def test_the_appearance_signal_is_a_scalar_not_a_vector(self) -> None:
        """A delta is a measurement. A vector would be an appearance template —
        which is C2 biometric data by 12_SECURITY's own definition.
        """
        from vision_os.core.ports.cropping import TriggerCandidate

        annotation = str(
            TriggerCandidate.__dataclass_fields__["appearance_delta"].type
        )
        assert "float" in annotation
        assert "tuple" not in annotation and "Sequence" not in annotation

    def test_no_gallery_or_enrolment_state_exists(self) -> None:
        offenders: list[str] = []
        for path in _files(CROPPING):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and any(
                    token in node.name.lower()
                    for token in ("gallery", "enrol", "template", "probe")
                ):
                    offenders.append(f"{path.name}::{node.name}")
        assert not offenders, "\n".join(offenders)


class TestNoIdentityPersistence:
    def test_trigger_state_holds_no_identity(self) -> None:
        """It holds an ``ObjectId`` — M7's handle — and nothing about a person."""
        from vision_os.perception.cropping import ObjectTriggerState

        fields = set(ObjectTriggerState.__slots__)
        for forbidden in (
            "person_id", "identity", "name", "embedding", "face", "gallery_id",
        ):
            assert forbidden not in fields

    def test_the_crop_names_an_object_not_a_person(self) -> None:
        fields = set(Crop.__dataclass_fields__)
        assert "object_id" in fields
        for forbidden in ("person_id", "identity_id", "global_id", "subject_id"):
            assert forbidden not in fields

    def test_evidence_ttl_bounds_how_long_an_image_lives(self) -> None:
        from ..cropping.conftest import at

        crop = _evidence_crop(ttl_ms=3_600_000)
        assert crop.expires_at() is not None
        assert crop.expires_at().ns == at(3).ns + 3_600_000_000_000
        assert Duration.from_millis(3_600_000).ns > 0


def _evidence_crop(*, ttl_ms: int) -> Crop:
    from .unit.test_crop_model import make_crop

    return make_crop(
        retention=RetentionMode.EVIDENCE,
        retention_ttl=Duration.from_millis(ttl_ms),
    )
