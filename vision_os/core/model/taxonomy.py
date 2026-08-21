"""The Visual Taxonomy (02_VISION_OBJECT_MODEL section 8).

Detector A emits COCO's 80 classes. Detector B emits Open Images' 600. Detector C
is a custom fine-tune emitting ``pallet``, ``forklift``, ``carton``. If those
label spaces reach the pipeline, every downstream component and every consumer is
coupled to the current model choice, and swapping detectors becomes a data
migration.

> **The platform owns a versioned Visual Taxonomy. Model-native label spaces are
> an adapter concern and never escape the adapter.**

**The taxonomy is domain-neutral.** ``person``, ``vehicle.forklift``,
``container.tray`` are visual kinds any observer would name. ``staff_member``,
``patient``, ``customer`` are *roles* — rejected under invariant V1, because no
crop evidences a role. A consumer assigns roles from observations plus its own
knowledge.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from .ids import AdapterId, ClassId, ModelId


class GeometryKind(enum.Enum):
    """What may be asserted about a class."""

    BOX = "box"
    ORIENTED_BOX = "oriented_box"
    MASK = "mask"
    KEYPOINTS = "keypoints"


class ClassStatus(enum.Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"


class UnmappedPolicy(enum.Enum):
    """What an adapter does with a native label it cannot map."""

    DROP = "drop"
    EMIT_AS_UNKNOWN = "emit_as_unknown"


#: The class emitted under ``EMIT_AS_UNKNOWN``. Present in every taxonomy so that
#: an unmapped detection is *visible* rather than silently discarded — the
#: platform would otherwise under-report and never say why (invariant V8).
UNKNOWN_CLASS = ClassId("unknown")


@dataclass(frozen=True, slots=True)
class TaxonomyClass:
    """One visual kind.

    Hierarchical by dotted path, so specificity can vary by model: a generic
    detector emits ``vehicle``, a specialized one ``vehicle.forklift``, and a
    consumer asking for ``vehicle`` matches both. This is what lets a model
    upgrade *increase* specificity without breaking existing queries.
    """

    class_id: ClassId
    taxonomy_version: str
    geometry_kinds: tuple[GeometryKind, ...] = (GeometryKind.BOX,)
    status: ClassStatus = ClassStatus.ACTIVE
    superseded_by: ClassId | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.class_id:
            raise ValueError("class_id is required")
        if self.class_id.startswith(".") or self.class_id.endswith("."):
            raise ValueError(f"malformed hierarchical class_id: {self.class_id!r}")
        if self.status is ClassStatus.SUPERSEDED and self.superseded_by is None:
            raise ValueError(
                f"class '{self.class_id}' is superseded but names no successor; "
                f"classes are deprecated, never deleted, so history stays readable"
            )
        if not self.geometry_kinds:
            raise ValueError(f"class '{self.class_id}' must declare a geometry kind")

    @property
    def parent(self) -> ClassId | None:
        """The immediate ancestor, derived from the dotted path."""
        if "." not in self.class_id:
            return None
        return ClassId(self.class_id.rsplit(".", 1)[0])

    @property
    def ancestry(self) -> tuple[ClassId, ...]:
        """Every ancestor from broadest to this class, inclusive."""
        parts = self.class_id.split(".")
        return tuple(ClassId(".".join(parts[: i + 1])) for i in range(len(parts)))

    def is_a(self, ancestor: ClassId) -> bool:
        """Hierarchical match: ``vehicle.forklift`` is a ``vehicle``."""
        return ancestor in self.ancestry


@dataclass(frozen=True, slots=True)
class MappingEntry:
    """One native label bound to one platform class."""

    native_label: str
    class_id: ClassId
    mapping_confidence: float = 1.0
    notes: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.mapping_confidence <= 1.0:
            raise ValueError(
                f"mapping_confidence must be in [0,1], got {self.mapping_confidence}"
            )


@dataclass(frozen=True, slots=True)
class TaxonomyMapping:
    """Owned by the adapter, validated by the platform (02_VOM section 8.2).

    Absorbing the native label space here is what makes detectors interchangeable.
    A mapping that references a class the taxonomy does not define fails
    validation at load, not at first frame.
    """

    adapter_id: AdapterId
    model_id: ModelId
    entries: tuple[MappingEntry, ...]
    unmapped_policy: UnmappedPolicy = UnmappedPolicy.DROP
    native_label_space: str = ""

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for entry in self.entries:
            if entry.native_label in seen:
                raise ValueError(
                    f"mapping for '{self.adapter_id}' declares native label "
                    f"'{entry.native_label}' twice"
                )
            seen.add(entry.native_label)

    def lookup(self, native_label: str) -> MappingEntry | None:
        for entry in self.entries:
            if entry.native_label == native_label:
                return entry
        return None

    @property
    def producible_classes(self) -> tuple[ClassId, ...]:
        """Which platform classes this model can produce.

        Published so a consumer demanding a class the site cannot produce gets an
        explicit capability gap rather than silence (invariant V8).
        """
        return tuple(dict.fromkeys(entry.class_id for entry in self.entries))


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """What a mapping can and cannot produce, against a taxonomy."""

    adapter_id: AdapterId
    model_id: ModelId
    producible: tuple[ClassId, ...]
    absent: tuple[ClassId, ...]
    unknown_classes: tuple[ClassId, ...] = field(default_factory=tuple)
    """Classes the mapping names that the taxonomy does not define. Non-empty
    means the mapping is invalid and must not load."""

    @property
    def valid(self) -> bool:
        return not self.unknown_classes

    def can_produce(self, class_id: ClassId) -> bool:
        """Hierarchical: a mapping producing ``vehicle.forklift`` covers a
        request for ``vehicle``."""
        return any(
            produced == class_id or produced.startswith(f"{class_id}.")
            for produced in self.producible
        )
