"""The Visual Taxonomy registry (02_VISION_OBJECT_MODEL section 8).

A platform-owned, versioned asset registry — not a flow layer. The Detection
Engine is its first consumer (M5's declared dependency, "Taxonomy registry");
Tracking, Understanding and the Observation Builder consume the same instance in
later flows.

Its job is to make one guarantee enforceable: **a model-native label never
escapes an adapter.** A mapping that names a class the taxonomy does not define
fails validation at load, not at first frame.

The registry is domain-neutral by construction. It stores visual kinds and their
hierarchy; it holds no roles, no policies, and no thresholds, because a class
that cannot be evidenced by a crop has no place in it (invariant V1).
"""

from __future__ import annotations

import threading
from dataclasses import replace

from ..core.errors import TaxonomyError
from ..core.model.ids import AdapterId, ClassId, ModelId
from ..core.model.taxonomy import (
    UNKNOWN_CLASS,
    ClassStatus,
    CoverageReport,
    GeometryKind,
    TaxonomyClass,
    TaxonomyMapping,
)

DEFAULT_TAXONOMY_VERSION = "1.0.0"


class TaxonomyRegistry:
    """Read-mostly registry of visual kinds and adapter mappings.

    Concurrency mirrors the Camera Manager: copy-on-write snapshots swapped
    atomically, so lookups on the detection hot path never block and never
    observe a half-applied reload.
    """

    def __init__(self, *, version: str = DEFAULT_TAXONOMY_VERSION) -> None:
        self._write_lock = threading.Lock()
        self._version = version
        self._classes: dict[ClassId, TaxonomyClass] = {}
        self._mappings: dict[AdapterId, TaxonomyMapping] = {}
        self._register_unknown()

    def _register_unknown(self) -> None:
        """``unknown`` always exists.

        Without it an adapter using ``EMIT_AS_UNKNOWN`` could not honour its own
        declared policy, and unmapped detections would vanish silently.
        """
        self._classes[UNKNOWN_CLASS] = TaxonomyClass(
            class_id=UNKNOWN_CLASS,
            taxonomy_version=self._version,
            description="An object the loaded model could not map to a platform class.",
        )

    # --- classes ------------------------------------------------------------ #

    @property
    def version(self) -> str:
        return self._version

    def register_class(self, taxonomy_class: TaxonomyClass) -> None:
        """Add a class. Its ancestors must already exist.

        Requiring ancestors first is what keeps the hierarchy connected: a
        ``vehicle.forklift`` with no ``vehicle`` would silently fail every
        consumer query for ``vehicle``.
        """
        with self._write_lock:
            parent = taxonomy_class.parent
            if parent is not None and parent not in self._classes:
                raise TaxonomyError(
                    f"cannot register '{taxonomy_class.class_id}': its parent "
                    f"'{parent}' is not defined",
                    class_id=str(taxonomy_class.class_id),
                )
            if taxonomy_class.superseded_by is not None:
                if taxonomy_class.superseded_by not in self._classes:
                    raise TaxonomyError(
                        f"'{taxonomy_class.class_id}' is superseded by "
                        f"'{taxonomy_class.superseded_by}', which is not defined"
                    )
            classes = dict(self._classes)
            classes[taxonomy_class.class_id] = taxonomy_class
            self._classes = classes

    def register_classes(self, classes: tuple[TaxonomyClass, ...]) -> None:
        """Register in ancestry order, so declaration order does not matter."""
        for taxonomy_class in sorted(classes, key=lambda c: c.class_id.count(".")):
            self.register_class(taxonomy_class)

    def get(self, class_id: ClassId) -> TaxonomyClass:
        taxonomy_class = self._classes.get(class_id)
        if taxonomy_class is None:
            raise TaxonomyError(f"unknown taxonomy class '{class_id}'", class_id=str(class_id))
        return taxonomy_class

    def has(self, class_id: ClassId) -> bool:
        return class_id in self._classes

    def classes(self) -> tuple[TaxonomyClass, ...]:
        return tuple(self._classes.values())

    def resolve(self, class_id: ClassId) -> ClassId:
        """Follow ``superseded_by`` forward so renames stay queryable.

        Classes are deprecated, never deleted: historical records must remain
        interpretable, and a query for an old name must still find the data.
        """
        seen: set[ClassId] = set()
        current = class_id
        while True:
            taxonomy_class = self._classes.get(current)
            if taxonomy_class is None or taxonomy_class.superseded_by is None:
                return current
            if current in seen:
                raise TaxonomyError(f"supersession cycle at '{current}'")
            seen.add(current)
            current = taxonomy_class.superseded_by

    def is_a(self, class_id: ClassId, ancestor: ClassId) -> bool:
        """Hierarchical match. ``vehicle.forklift`` is a ``vehicle``."""
        return class_id == ancestor or class_id.startswith(f"{ancestor}.")

    def descendants(self, ancestor: ClassId) -> tuple[ClassId, ...]:
        return tuple(
            class_id for class_id in self._classes if self.is_a(class_id, ancestor)
        )

    def supports_geometry(self, class_id: ClassId, kind: GeometryKind) -> bool:
        return kind in self.get(class_id).geometry_kinds

    def deprecate(self, class_id: ClassId, superseded_by: ClassId | None = None) -> None:
        with self._write_lock:
            existing = self._classes.get(class_id)
            if existing is None:
                raise TaxonomyError(f"unknown taxonomy class '{class_id}'")
            if superseded_by is not None and superseded_by not in self._classes:
                raise TaxonomyError(f"successor '{superseded_by}' is not defined")
            classes = dict(self._classes)
            classes[class_id] = replace(
                existing,
                status=(
                    ClassStatus.SUPERSEDED if superseded_by else ClassStatus.DEPRECATED
                ),
                superseded_by=superseded_by,
            )
            self._classes = classes

    # --- mappings ------------------------------------------------------------ #

    def validate_mapping(self, mapping: TaxonomyMapping) -> CoverageReport:
        """Check a mapping against the taxonomy without registering it.

        ``unknown_classes`` being non-empty means the mapping is invalid. The
        Plugin Manager refuses to activate an adapter whose mapping does not
        validate, so a typo'd class fails at load rather than producing
        detections nobody can query.
        """
        producible: list[ClassId] = []
        unknown: list[ClassId] = []
        for entry in mapping.entries:
            if entry.class_id in self._classes:
                if entry.class_id not in producible:
                    producible.append(entry.class_id)
            elif entry.class_id not in unknown:
                unknown.append(entry.class_id)

        absent = tuple(
            class_id
            for class_id in self._classes
            if class_id != UNKNOWN_CLASS
            and not any(
                produced == class_id or produced.startswith(f"{class_id}.")
                for produced in producible
            )
        )
        return CoverageReport(
            adapter_id=mapping.adapter_id,
            model_id=mapping.model_id,
            producible=tuple(producible),
            absent=absent,
            unknown_classes=tuple(unknown),
        )

    def register_mapping(self, mapping: TaxonomyMapping) -> CoverageReport:
        """Validate and register. Raises rather than registering an invalid mapping."""
        report = self.validate_mapping(mapping)
        if not report.valid:
            raise TaxonomyError(
                f"mapping for adapter '{mapping.adapter_id}' names "
                f"{len(report.unknown_classes)} class(es) the taxonomy does not "
                f"define: {sorted(report.unknown_classes)}",
                adapter_id=str(mapping.adapter_id),
            )
        with self._write_lock:
            mappings = dict(self._mappings)
            mappings[mapping.adapter_id] = mapping
            self._mappings = mappings
        return report

    def mapping_for(self, adapter_id: AdapterId) -> TaxonomyMapping | None:
        return self._mappings.get(adapter_id)

    def coverage(self, adapter_id: AdapterId) -> CoverageReport | None:
        mapping = self._mappings.get(adapter_id)
        return self.validate_mapping(mapping) if mapping else None

    def producible_classes(self) -> tuple[ClassId, ...]:
        """Every class any registered mapping can currently produce.

        This is the platform's published capability surface: a consumer asking
        for something absent here gets an explicit gap rather than silence (V8).
        """
        produced: list[ClassId] = []
        for mapping in self._mappings.values():
            for class_id in mapping.producible_classes:
                if class_id not in produced:
                    produced.append(class_id)
        return tuple(produced)

    def capability_gap(self, requested: tuple[ClassId, ...]) -> tuple[ClassId, ...]:
        """Which requested classes no loaded model can produce."""
        producible = self.producible_classes()
        return tuple(
            class_id
            for class_id in requested
            if not any(
                produced == class_id or produced.startswith(f"{class_id}.")
                for produced in producible
            )
        )

    def model_for(self, adapter_id: AdapterId) -> ModelId | None:
        mapping = self._mappings.get(adapter_id)
        return mapping.model_id if mapping else None
