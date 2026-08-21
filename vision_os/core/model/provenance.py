"""Provenance and timing (02_VISION_OBJECT_MODEL section 3).

``model_artifact_hash`` and ``config_revision`` are mandatory, not optional. Six
months after deployment someone will ask why the platform said what it said on a
specific day; without the exact weights and the exact configuration that question
is unanswerable and every regression investigation becomes archaeology.

These two fields are also a security property: if a compromised artifact is later
discovered, the exact set of records it produced is *queryable* rather than
assumed total (12_SECURITY section 6.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ids import AdapterId, ConfigRevision, ModelId, ModuleId


@dataclass(frozen=True, slots=True)
class Provenance:
    """Who asserted this, with what, under which configuration."""

    producer_module: ModuleId
    producer_version: str
    config_revision: ConfigRevision
    adapter_id: AdapterId | None = None
    adapter_version: str | None = None
    model_id: ModelId | None = None
    model_version: str | None = None
    model_artifact_hash: str | None = None
    deterministic: bool = False

    def __post_init__(self) -> None:
        if not self.producer_module:
            raise ValueError("provenance requires a producer_module")
        if not self.config_revision:
            raise ValueError(
                "provenance requires a config_revision; without it no result is "
                "reproducible (invariant V4)"
            )
        if self.model_id is not None and self.model_artifact_hash is None:
            raise ValueError(
                f"model '{self.model_id}' named without its artifact hash; the exact "
                f"weights that produced a result are mandatory (invariant V4)"
            )

    @property
    def detector_name(self) -> str | None:
        """Human-facing detector identity, for reports and dashboards."""
        return str(self.adapter_id) if self.adapter_id else None

    @property
    def detector_version(self) -> str | None:
        return self.adapter_version


@dataclass(frozen=True, slots=True)
class InferenceTiming:
    """Where the wall-clock went, in milliseconds.

    Retained per result rather than aggregated only into metrics: when one camera
    is slow the question is always "slow where?", and an aggregate cannot answer
    it for a specific frame.
    """

    queued_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    batch_size: int = 1
    device_id: str = "cpu"
    model_load_state: str = "warm"
    """``warm`` | ``cold``. A cold first inference can be 10-100x slower, and
    without this field that reads as a performance regression."""

    @property
    def total_ms(self) -> float:
        return (
            self.queued_ms + self.preprocess_ms + self.inference_ms + self.postprocess_ms
        )


@dataclass(frozen=True, slots=True)
class ModelMeta:
    """What an adapter declares about the model that produced a result."""

    model_id: ModelId
    model_version: str
    artifact_hash: str
    precision: str = "fp32"
    device_id: str = "cpu"
    extra: dict[str, str] = field(default_factory=dict)
