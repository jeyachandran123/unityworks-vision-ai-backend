"""The device worker (08_RUNTIME_AND_THREADING section 4).

Single responsibility: *execute one batch on one device, and never let an
adapter's failure escape as anything but a typed result.*

One worker per device. The adapter itself need not be thread-safe, which widely
broadens the set of usable third-party models: a detector that declares itself
single-threaded gets a dedicated worker rather than an unsafe shared one.

Inference is synchronous and CPU/GPU-bound, so it runs in a thread executor. That
keeps the event loop free for the dozens of camera actors awaiting their results
— which is exactly how batches form in the first place.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from ...core.errors import DetectionFailedError, DetectorContractError
from ...core.model.health import ComponentHealth, HealthState
from ...core.ports.clock import Clock
from ...core.ports.detection import (
    DetectionRequest,
    DetectionResult,
    DetectorPort,
    FrameView,
)


@dataclass(frozen=True, slots=True)
class WorkerStats:
    batches: int = 0
    frames: int = 0
    failures: int = 0
    total_inference_ms: float = 0.0

    @property
    def mean_inference_ms(self) -> float:
        return self.total_inference_ms / self.batches if self.batches else 0.0


class DeviceWorker:
    """Runs batches against one detector bound to one device."""

    def __init__(
        self,
        *,
        clock: Clock,
        detector: DetectorPort,
        device_id: str,
        single_threaded: bool = True,
    ) -> None:
        self._clock = clock
        self._detector = detector
        self._device_id = device_id
        self._stats = WorkerStats()
        # A single-threaded adapter gets exclusive access rather than an unsafe
        # shared one; the manifest declares which it is and the platform honours it.
        self._gate = asyncio.Semaphore(1 if single_threaded else 4)

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def detector(self) -> DetectorPort:
        return self._detector

    @property
    def stats(self) -> WorkerStats:
        return self._stats

    async def execute(
        self, views: Sequence[FrameView], request: DetectionRequest
    ) -> Sequence[DetectionResult]:
        """Run one batch.

        Raises:
            DetectionFailedError: the adapter could not produce results. Never a
                fabricated result — a plausible wrong answer is worse than an
                admitted failure, because nothing downstream can detect it.
            DetectorContractError: the adapter returned a structurally invalid
                batch (wrong length). Caught here so a contract breach cannot
                reach the normalizer as a silent mismatch.
        """
        started = self._clock.monotonic().ns
        async with self._gate:
            try:
                results = await asyncio.to_thread(self._detector.detect, views, request)
            except DetectorContractError:
                self._stats = _bump(self._stats, failures=1)
                raise
            except Exception as exc:  # noqa: BLE001 - normalise every adapter failure
                self._stats = _bump(self._stats, failures=1)
                raise DetectionFailedError(
                    f"detector on device '{self._device_id}' failed for a batch of "
                    f"{len(views)}: {type(exc).__name__}: {exc}",
                    device_id=self._device_id,
                ) from exc

        if len(results) != len(views):
            self._stats = _bump(self._stats, failures=1)
            raise DetectorContractError(
                f"detector returned {len(results)} results for {len(views)} frames; "
                f"batch results must map 1:1 and in order (obligation D6)",
                device_id=self._device_id,
            )

        elapsed_ms = (self._clock.monotonic().ns - started) / 1_000_000
        self._stats = _bump(
            self._stats,
            batches=1,
            frames=len(views),
            total_inference_ms=elapsed_ms,
        )
        return results

    async def warm(self) -> None:
        """Warm the adapter off the event loop.

        Mandatory before a detector counts as ready: a cold first inference can
        be 10-100x slower and would otherwise read as a performance regression.
        """
        await asyncio.to_thread(self._detector.warm)

    def health(self) -> ComponentHealth:
        try:
            return self._detector.health()
        except Exception as exc:  # noqa: BLE001 - a broken health check is itself unhealthy
            from ...core.model.ids import ModuleId

            return ComponentHealth(
                component_id=ModuleId(f"detector.{self._device_id}"),
                state=HealthState.DEGRADED,
                reported_at=self._clock.now(),
                detail=f"health check raised {type(exc).__name__}: {exc}",
            )


def _bump(
    stats: WorkerStats,
    *,
    batches: int = 0,
    frames: int = 0,
    failures: int = 0,
    total_inference_ms: float = 0.0,
) -> WorkerStats:
    return WorkerStats(
        batches=stats.batches + batches,
        frames=stats.frames + frames,
        failures=stats.failures + failures,
        total_inference_ms=stats.total_inference_ms + total_inference_ms,
    )
