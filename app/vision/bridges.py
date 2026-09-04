"""The L3 → L4 bridge: tracking outcomes become registry updates.

### The defect this replaces

Phase 4 connected tracking to the registry like this:

    tracking.runtime._sink = registry_layer.runtime   # noqa: SLF001

Two things are wrong with it, and together they produced a total, silent
outage of everything above tracking.

**It reaches into a private attribute.** `_sink` is not a declared seam.

**The shapes do not match.** `TrackingRuntime._publish` calls
`self._sink(result)` — a *callable* taking a `TrackingOutcome`. `RegistryRuntime`
is not callable; its entry point is `async def on_tracked(camera_id, update)`
taking a `TrackUpdate`. So every call raised `TypeError`, and tracking's sink
guard — correctly, by invariant V9 — swallowed it:

    except Exception:                    # a bad sink must not break tracking
        self._stats.sink_failures += 1

Detection ran. Tracking ran. `registry.created`, `cropping.requested`,
`understanding.results`, `synthesis.built` and `state.appended` all stayed at
zero, and nothing anywhere said why. The platform behaved exactly as designed;
the composition was wrong, and a composition error is invisible precisely
because each layer is doing its job.

### What this does instead

Adapts the shape, and only the shape. It reads `TrackingOutcome`, builds the
`TrackUpdate` the registry declares, and awaits `on_tracked`. It makes no
tracking decision, invents no track, and drops nothing silently — a failure here
is counted and named.

### Why it schedules rather than awaits

`TrackingRuntime._publish` is synchronous and `on_tracked` is a coroutine. The
bridge schedules onto the running loop and keeps a strong reference to the task,
because a bare `create_task` result that nobody holds may be garbage-collected
mid-flight. Ordering per camera is preserved by `RegistryRuntime`'s own
per-camera lock, which is why this does not need one.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass(slots=True)
class BridgeAudit:
    """What crossed the seam, and what did not."""

    outcomes_seen: int = 0
    outcomes_failed: int = 0
    updates_forwarded: int = 0
    forward_failures: int = 0
    #: Set when the bridge is called with no running loop — which would mean
    #: tracking is publishing from a thread the application does not own.
    no_loop: int = 0
    last_error: str = ""
    failure_kinds: dict[str, int] = field(default_factory=dict)

    def note(self, exc: BaseException) -> None:
        kind = type(exc).__name__
        self.failure_kinds[kind] = self.failure_kinds.get(kind, 0) + 1
        self.last_error = f"{kind}: {exc}"

    def to_wire(self) -> dict[str, Any]:
        return {
            "outcomes_seen": self.outcomes_seen,
            "outcomes_failed": self.outcomes_failed,
            "updates_forwarded": self.updates_forwarded,
            "forward_failures": self.forward_failures,
            "no_loop": self.no_loop,
            "failure_kinds": dict(self.failure_kinds),
            "last_error": self.last_error,
            # Zero forwarded against a non-zero seen count is the exact shape of
            # the defect this module exists to fix, and it is now a number
            # somebody can read rather than a silence.
            "forwarded_fraction": (
                self.updates_forwarded / self.outcomes_seen if self.outcomes_seen else 0.0
            ),
        }


class TrackingToRegistryBridge:
    """Callable sink for `TrackingRuntime`. Adapts to `RegistryRuntime`."""

    __slots__ = ("_registry", "_tasks", "audit")

    def __init__(self, registry_runtime: Any) -> None:
        self._registry = registry_runtime
        self._tasks: set[asyncio.Task[None]] = set()
        self.audit = BridgeAudit()

    def __call__(self, outcome: Any) -> None:
        """Tracking's sink contract: synchronous, one `TrackingOutcome`."""
        self.audit.outcomes_seen += 1

        if getattr(outcome, "failed", False):
            # Forwarded anyway. A failed tracking frame is when the registry
            # ages and invalidates state, so skipping it would leave objects
            # looking fresher than the platform can justify.
            self.audit.outcomes_failed += 1

        try:
            update = self._to_update(outcome)
        except Exception as exc:  # noqa: BLE001 - one frame, not the pipeline
            self.audit.forward_failures += 1
            self.audit.note(exc)
            logger.warning(
                "tracking outcome could not be adapted for the registry: {}: {}",
                type(exc).__name__,
                exc,
            )
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.audit.no_loop += 1
            self.audit.last_error = "no running event loop at the tracking sink"
            return

        task = loop.create_task(self._forward(outcome.camera_id, update))
        # Held until done: a task referenced only by the event loop may be
        # collected before it runs.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _forward(self, camera_id: Any, update: Any) -> None:
        try:
            await self._registry.on_tracked(camera_id, update)
            self.audit.updates_forwarded += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the seam is a firewall
            self.audit.forward_failures += 1
            self.audit.note(exc)
            logger.warning(
                "registry refused a track update for {}: {}: {}",
                camera_id,
                type(exc).__name__,
                exc,
            )

    @staticmethod
    def _to_update(outcome: Any) -> Any:
        """`TrackingOutcome` → `TrackUpdate`. Shape only, no judgement.

        ### What this used to do, and why it was wrong

        `TrackingOutcome` carried the tracks plus *counts* of what changed, so
        this method reconstructed the changed ids by reading each track's
        **state**:

            new=ids_in(TrackState.TENTATIVE)
            coasting=ids_in(TrackState.COASTING)

        A state is not an event. `TENTATIVE` persists for
        `min_hits_to_confirm` frames, so a single track was reported as *new* on
        every one of those frames, while a track created and confirmed in one
        frame was never reported as new at all. `terminated`, `recovered`,
        `associations`, `refused` and `unmatched_detections` had no counterpart
        in the counts, so they silently took their empty defaults on every
        frame — the registry was told, forever, that nothing ever ended and
        nothing ever came back.

        ### What it does now

        The tracker already produces a complete `TrackUpdate` and the engine now
        carries it through, so the honest adaptation is to pass it along. No
        state is inspected, no transition is inferred, and the frame reference
        is the value object the tracker actually used rather than one re-parsed
        from its rendering.

        The reconstruction is kept **only** for an outcome with no update — a
        failed frame, or an older producer — so this bridge still degrades to
        exactly its previous behaviour rather than dropping the frame.
        """
        from vision_os.core.model.ids import FrameRef
        from vision_os.core.model.track import TrackState, TrackUpdate

        update = getattr(outcome, "update", None)
        if update is not None:
            return update

        tracks = tuple(getattr(outcome, "tracks", ()) or ())

        def ids_in(*states: Any) -> tuple[Any, ...]:
            return tuple(t.track_id for t in tracks if getattr(t, "state", None) in states)

        frame_ref = getattr(outcome, "frame_ref", "")
        if isinstance(frame_ref, str):
            frame_ref = _parse_frame_ref(frame_ref, outcome.camera_id) or FrameRef(
                camera_id=outcome.camera_id, stream_epoch=0, frame_seq=0
            )

        return TrackUpdate(
            camera_id=outcome.camera_id,
            frame_ref=frame_ref,
            tracker_epoch=int(getattr(outcome, "tracker_epoch", 0) or 0),
            active=tracks,
            new=ids_in(TrackState.TENTATIVE),
            coasting=ids_in(TrackState.COASTING),
            failed=bool(getattr(outcome, "failed", False)),
            reason=str(getattr(outcome, "reason", "") or ""),
        )


def _parse_frame_ref(rendered: str, camera_id: Any) -> Any | None:
    """`cam/e0/f12` back into a `FrameRef`.

    Tracking renders the reference as a string; the registry wants the value
    object. Parsing rather than fabricating keeps the frame a track update cites
    the same frame detection actually ran on — which is what makes a stored
    attribute traceable to a picture.
    """
    from vision_os.core.model.ids import FrameRef, StreamEpoch

    parts = rendered.rsplit("/", 2)
    if len(parts) != 3:
        return None
    _, epoch_part, seq_part = parts
    try:
        epoch = int(epoch_part.lstrip("eE") or 0)
        sequence = int(seq_part.lstrip("fF") or 0)
    except ValueError:
        return None
    return FrameRef(camera_id=camera_id, stream_epoch=StreamEpoch(epoch), frame_seq=sequence)


__all__ = ["BridgeAudit", "TrackingToRegistryBridge"]
