"""Live production frame collector — P9.5.

    python -m tools.p9_dataset.collect --seconds 120 --cameras 11,12,13,14

### What this is, and what it deliberately is not

A **read-only data acquisition tool**. It opens its own RTSP session, decodes,
samples, writes JPEGs and a session record. It imports no compliance code, holds
no registry, publishes no observation and cannot reach an alert. Nothing it does
can change a verdict, and nothing in production depends on it running.

It reuses `app.vision.sources.rtsp.RtspCameraConfig` rather than building a
second camera-ingestion mechanism, so the URL construction, credential reference
and redaction behaviour are the production ones. The password is resolved at the
moment of dialling and never stored, never logged, never written to a manifest.

### Sampling

Two strategies, selected with `--mode`.

**`event` (P9.6, the default).** The frame is offered to an `EventSampler`, which
keeps it only when the scene or the people in it changed, and records *why*. Each
camera therefore finds its own rate: a busy one samples often, a still one falls
back to the low-rate heartbeat. This exists because P9.5's wall-clock corpus
measured 55.8 % near-duplicate, 28.1 % bit-identical and 34 % person-free, and
because the duplicate rate did **not** respond to the interval — 55.0 % at 3 s,
53.9 % at 4 s, 60.8 % at 5 s. A timer was the wrong instrument, not a mistuned
one. See `P9_6_SAMPLING_BASELINE.md`.

**`interval` (P9.5).** The original wall-clock timer, kept so the comparison
remains runnable and the earlier corpus reproducible. Not deleted: a superseded
strategy that can still be executed is evidence; one that can only be described
is a claim.

### Sampling is blind to PPE in both modes

Nothing here looks for hairnets, gloves, masks or violations, and no PPE model,
classifier or VLM output can reach the sampling decision. The event sampler is
allowed to know that a *person* is present and where — nothing about what they
are wearing. A collector that searched for uncovered heads would build a corpus
around the classes someone expected to find, and every distribution computed from
it afterwards would describe the search rather than the restaurant.

### Failures

A camera that will not open, a stream that stalls mid-session, a frame that will
not decode: each is caught, counted, recorded in the session record, and does not
stop the other cameras. A silently dropped camera is a blind kitchen, so every
failure is data.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "datasets" / "p9-live"

#: Wall-clock seconds between kept frames.
#:
#: Derived, not chosen: at the measured 15–25 fps a 3 s interval discards ~98 %
#: of frames before any similarity test, and 3 s is long enough that a worker's
#: posture, hand position and occlusion state have usually changed. The residual
#: duplication that survives it is measured by `dedupe.py` and published — if
#: that number is high the interval is wrong and the report says so.
DEFAULT_INTERVAL_S = 3.0

#: Bumped when the collector's own behaviour changes in a way that could alter
#: which frames reach disk, independently of the sampling policy. Recorded per
#: session so a corpus names both the policy AND the machinery that applied it.
COLLECTOR_VERSION = "p9.7-collector-1"

#: Per-camera cap, so one camera cannot dominate the corpus (Phase 6).
DEFAULT_MAX_FRAMES = 120

_TRANSPORT = {
    "rtsp_transport": "tcp",
    "stimeout": "15000000",
    "max_delay": "5000000",
    "fflags": "nobuffer",
}


@dataclass(slots=True)
class CameraResult:
    camera_id: str
    channel: int
    redacted_uri: str
    width: int = 0
    height: int = 0
    codec: str = ""
    stream_fps: float = 0.0
    frames_decoded: int = 0
    frames_kept: int = 0
    frames_failed: int = 0
    reconnects: int = 0
    seconds: float = 0.0
    status: str = "not_attempted"
    error: str = ""
    files: list[str] = field(default_factory=list)

    samples: list[dict] = field(default_factory=list)
    """Per-kept-frame provenance: file, reason, offset, people, tracks.

    Phase 5's requirement — every retained sample carries a machine-readable
    sampling reason — plus the camera/day/session/sequence chain Phase 19 needs
    to prevent leakage later. Written for both modes; in `interval` mode the
    reason is `wall_clock_interval`, which is honest about what selected it."""

    by_reason: dict[str, int] = field(default_factory=dict)
    event_triggered: int = 0
    baseline_triggered: int = 0
    suppressed: dict[str, int] = field(default_factory=dict)
    retrospective_captures: int = 0
    retrospective_missed: int = 0
    """Departures whose evidence frame had already left the ring buffer. Counted
    rather than hidden: the frame kept was then the expiry frame, not the one the
    policy names, and a corpus must not claim a rule it did not apply."""

    candidate_subject_tracks: int = 0
    """Distinct confirmed tracks seen. **Not** a count of people: a track is an
    association id that survives no occlusion and no exit, so one person may
    produce several. Reported as an upper bound on subject diversity and never
    promoted to an identity."""


def _env() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _config(channel: int):
    from app.vision.sources.rtsp import RtspCameraConfig

    return RtspCameraConfig(
        camera_id=f"cam-{channel:02d}",
        host=os.environ["CCTV_HOST"],
        channel=channel,
        port=int(os.environ.get("CCTV_RTSP_PORT") or 554),
        stream_type=os.environ.get("CCTV_STREAM_TYPE", "sub"),
        username=os.environ.get("CCTV_USERNAME", ""),
        credential_ref="env:CCTV_PASSWORD",
    )


#: Frames held for retrospective capture. At 15-25 fps this is a few seconds —
#: comfortably longer than `track_max_age + departure_confirm_misses` detections,
#: and bounded so a long session cannot grow memory without limit.
RETROSPECTIVE_BUFFER = 120


#: Intra-op threads per detector when cameras are observed concurrently.
#:
#: Four cameras at one thread per core ask for ~48 threads on a 12-core machine.
#: Capping is a precaution against oversubscription; two threads each leaves the
#: decoders room, and the detector is rate-limited anyway.
DETECTOR_THREADS = 2


class _LazyFrame:
    """A decoded frame that becomes a PIL image only when something needs one.

    The event sampler hashes **every** decoded frame and detects on a small
    fraction of them. Converting each one to RGB up front is a full-resolution
    colour-space conversion per frame — pure waste on the ~95 % that are hashed
    and discarded — so the hash comes from `libswscale` at 9x8 grey, which is
    close to free, and `to_image()` runs only when the detector is due or the
    frame is being kept.

    Worth recording honestly: this was written in response to an apparent
    throughput collapse that turned out to be an orphaned collector competing for
    the same cameras, not conversion cost. With the machine clean, four
    concurrent cameras decode at full stream rate — 862–1,463 frames per camera
    per 60 s window against nominal 15–25 fps. The change stands on its own
    merits; the collapse it was written to explain was never real.
    """

    __slots__ = ("frame", "_image")

    def __init__(self, frame) -> None:
        self.frame = frame
        self._image = None

    @property
    def image(self):
        if self._image is None:
            self._image = self.frame.to_image().convert("RGB")
        return self._image


def _dhash_video_frame(lazy: _LazyFrame) -> int:
    """64-bit difference hash straight off the decoded frame.

    Downscaled by `libswscale` rather than Pillow, so this is **not** bit-for-bit
    the hash `dedupe.dhash` computes from the saved JPEG. It does not need to be:
    this one is the live *sampling* signal and that one is the corpus
    *deduplication* signal. Both are deterministic, both measure the same thing,
    and neither is a label.
    """
    import numpy as np

    grid = np.asarray(
        lazy.frame.reformat(width=9, height=8, format="gray").to_ndarray(),
        dtype=np.int16,
    )
    value = 0
    for bit in (grid[:, 1:] > grid[:, :-1]).flatten():
        value = (value << 1) | int(bit)
    return value


def _perception(threads: int | None = DETECTOR_THREADS):
    """Bind the hash and the **person** detector for the event sampler.

    Deliberately narrow. The sampler is handed exactly two capabilities — hash a
    frame, find people in it — and there is no third callable through which a PPE
    signal could arrive. That is a structural guarantee, not a convention: the
    `EventSampler` constructor takes no other perception argument.
    """
    from .candidates import _detector, propose_people

    detector = _detector(threads)
    return _dhash_video_frame, lambda lazy: propose_people(detector, lazy.image)


def collect_camera(
    channel: int,
    *,
    session_id: str,
    seconds: float,
    interval: float,
    max_frames: int,
    attempts: int = 2,
    sampling: "SamplingConfig | None" = None,
) -> CameraResult:
    """Sample one camera for a bounded wall-clock window.

    `sampling` selects the strategy: an `EventSampler` when given, the P9.5
    wall-clock interval when `None`.

    Reconnects once on a mid-stream failure, because a DVR dropping a session is
    ordinary and losing the remainder of the window to it is not acceptable. Both
    the reconnect and its cause are recorded.
    """
    import av

    from .events import EventSampler

    config = _config(channel)
    result = CameraResult(
        camera_id=config.camera_id,
        channel=channel,
        redacted_uri=config.redacted_uri(),
    )
    password = os.environ.get("CCTV_PASSWORD", "")
    frames_dir = OUT / session_id / config.camera_id
    # A session is immutable, like a dataset version. Re-running a plan on the
    # same calendar day reuses its labels, and without this guard the second run
    # appends frames into the first run's directory and overwrites its
    # session.json — silently merging two collections under one provenance
    # record. Caught in P9.8 when a plan was re-run hours later.
    if frames_dir.exists() and any(frames_dir.glob("*.jpg")):
        raise FileExistsError(
            f"session {session_id} already holds frames for {config.camera_id}. "
            f"Sessions are immutable; choose a distinct --label rather than "
            f"merging two collections under one provenance record."
        )
    frames_dir.mkdir(parents=True, exist_ok=True)

    sampler = None
    recent: "collections.OrderedDict[int, object]" = collections.OrderedDict()
    if sampling is not None:
        hash_of, detect = _perception()
        sampler = EventSampler(sampling, hash_of=hash_of, detect=detect)
        max_frames = min(max_frames, sampling.max_samples)

    started = time.perf_counter()
    last_kept = -interval

    for attempt in range(1, attempts + 1):
        if result.frames_kept >= max_frames or time.perf_counter() - started >= seconds:
            break
        if attempt > 1:
            result.reconnects += 1
        try:
            with av.open(
                config.dial_uri(password), options=dict(_TRANSPORT), timeout=25
            ) as container:
                stream = container.streams.video[0]
                result.codec = stream.codec_context.name
                result.stream_fps = round(float(stream.average_rate or 0), 2)

                for frame in container.decode(stream):
                    now = time.perf_counter()
                    elapsed = now - started
                    if elapsed >= seconds or result.frames_kept >= max_frames:
                        break
                    result.frames_decoded += 1

                    if sampler is None and now - last_kept < interval:
                        continue

                    reason = "wall_clock_interval"
                    sample_class = "event"
                    people = tracks = 0
                    retrospective = False
                    try:
                        lazy = _LazyFrame(frame)
                        image = None
                        if sampler is not None:
                            index = result.frames_decoded
                            decision = sampler.offer(index, elapsed, lazy)
                            # A bounded ring of recent frames, so a departure can
                            # keep the last frame the person was actually in.
                            # Without it `LAST_CONFIRMED` degrades silently to
                            # `ON_EXPIRY` and the corpus would claim a rule it
                            # did not apply.
                            recent[index] = lazy
                            while len(recent) > RETROSPECTIVE_BUFFER:
                                recent.popitem(last=False)
                            if not decision.accepted:
                                continue
                            reason = decision.reason.value
                            sample_class = decision.sample_class.value
                            people = decision.people
                            tracks = len(decision.tracks)
                            wanted = decision.capture_frame_index
                            if wanted >= 0:
                                held = recent.get(wanted)
                                if held is None:
                                    result.retrospective_missed += 1
                                else:
                                    image = held.image
                                    retrospective = True
                        if image is None:
                            image = lazy.image
                    except Exception:  # noqa: BLE001 - a bad frame is data
                        result.frames_failed += 1
                        continue

                    last_kept = now
                    result.width, result.height = image.size
                    stamp = datetime.now(UTC).strftime("%H%M%S_%f")[:-3]
                    name = f"{config.camera_id}_{stamp}.jpg"
                    image.save(frames_dir / name, quality=92)
                    result.files.append(name)
                    result.samples.append(
                        {
                            "file": name,
                            "sequence": result.frames_kept,
                            "offset_seconds": round(elapsed, 3),
                            "sampling_reason": reason,
                            "sample_class": sample_class,
                            "retrospective_capture": retrospective,
                            "people_detected": people,
                            "tracks_live": tracks,
                        }
                    )
                    result.frames_kept += 1
            result.status = "ok"
        except Exception as error:  # noqa: BLE001 - never let one camera stop another
            result.error = f"{type(error).__name__}: {str(error)[:200]}"
            result.status = "failed" if result.frames_kept == 0 else "partial"

    if sampler is not None:
        statistics = sampler.statistics()
        result.by_reason = statistics["by_reason"]
        result.event_triggered = statistics["event_triggered"]
        result.baseline_triggered = statistics["baseline_triggered"]
        result.suppressed = statistics["suppressed"]
        result.candidate_subject_tracks = statistics["candidate_subject_tracks"]
        result.retrospective_captures = sum(
            1 for s in result.samples if s.get("retrospective_capture")
        )

    result.seconds = round(time.perf_counter() - started, 1)
    if result.frames_kept and result.status == "not_attempted":
        result.status = "ok"
    return result


def collect(
    channels: list[int],
    *,
    label: str,
    seconds: float,
    interval: float,
    max_frames: int,
    sampling: "SamplingConfig | None" = None,
    period: str = "unspecified",
) -> dict:
    """One collection session across several cameras.

    `period` names the operating condition — `morning-prep`, `lunch-peak`,
    `cleaning` — and is recorded verbatim. It is an **operator's assertion**, not
    a measurement, and the report must present it as one: nothing here verifies
    that the kitchen was doing what the label says.
    """
    _env()
    now = datetime.now(UTC)
    session_id = f"live-{now.strftime('%Y%m%d')}-{label}"

    # Cameras are observed **concurrently**, not one after another.
    #
    # Sequentially, a four-camera "session" spans four consecutive windows, so
    # cam-11 and cam-14 record different times and any within-session comparison
    # across cameras is comparing different moments. A session has to name one
    # slice of wall clock or it is not a unit of anything. It is also four times
    # faster, which buys more sessions — and the session is the group a split may
    # not straddle, so more sessions is the scarce resource.
    #
    # Each thread opens its own RTSP session and its own detector; nothing is
    # shared between them, and `av` and `onnxruntime` both release the GIL during
    # the work that matters.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(channels)) as pool:
        futures = [
            pool.submit(
                collect_camera,
                channel,
                session_id=session_id,
                seconds=seconds,
                interval=interval,
                max_frames=max_frames,
                sampling=sampling,
            )
            for channel in channels
        ]
        results = [future.result() for future in futures]

    reasons: dict[str, int] = {}
    for result in results:
        for reason, count in result.by_reason.items():
            reasons[reason] = reasons.get(reason, 0) + count

    record = {
        "_comment": [
            "A LIVE PRODUCTION collection session. Provenance: LIVE_PRODUCTION.",
            "Sampling is blind to PPE state. The event sampler may know that a",
            "PERSON is present and where; it is told nothing about coverings, and",
            "no PPE model, classifier or VLM output can reach the decision. A",
            "collector that looked for violations would describe the search,",
            "not the kitchen.",
            "'period' is an operator's assertion about the operating condition,",
            "not a measurement — nothing here verifies it.",
            "No credential appears here; URIs are redacted by the production",
            "config object that built them.",
        ],
        "session_id": session_id,
        "provenance": "LIVE_PRODUCTION",
        "collected_at": now.isoformat(timespec="seconds"),
        "ended_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "collection_day": now.strftime("%Y-%m-%d"),
        "label": label,
        "period": period,
        "collector_version": COLLECTOR_VERSION,
        "camera_set": [f"cam-{c:02d}" for c in channels],
        # SHA-256 over the serialised sampling configuration. Two sessions with
        # the same hash were collected under byte-identical policy; two with
        # different hashes were not, however similar their version strings look.
        "configuration_hash": hashlib.sha256(
            json.dumps(
                sampling.as_dict() if sampling else {"strategy": "wall-clock", "interval": interval},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "sampling": {
            "strategy": "event-aware" if sampling else "wall-clock interval",
            "sampling_config_version": sampling.version if sampling else "p9.5-interval",
            "interval_seconds": interval,
            "window_seconds": seconds,
            "max_frames_per_camera": max_frames,
            "config": sampling.as_dict() if sampling else None,
        },
        "cameras_observed_concurrently": True,
        "cameras": [asdict(r) for r in results],
        "totals": {
            "frames_decoded": sum(r.frames_decoded for r in results),
            "frames_kept": sum(r.frames_kept for r in results),
            "frames_failed": sum(r.frames_failed for r in results),
            "reconnects": sum(r.reconnects for r in results),
            "cameras_ok": sum(1 for r in results if r.status == "ok"),
            "cameras_failed": sum(1 for r in results if r.status == "failed"),
            "event_triggered": sum(r.event_triggered for r in results),
            "baseline_triggered": sum(r.baseline_triggered for r in results),
            "candidate_subject_tracks": sum(r.candidate_subject_tracks for r in results),
            "by_reason": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        },
    }
    path = OUT / session_id / "session.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", default="11,12,13,14")
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--label", default="a", help="distinguishes sessions on one day")
    parser.add_argument(
        "--mode",
        choices=("event", "interval"),
        default="event",
        help="event-aware sampling (P9.6) or the P9.5 wall-clock timer",
    )
    parser.add_argument(
        "--period",
        default="unspecified",
        help="operating condition asserted by the operator, e.g. lunch-peak",
    )
    parser.add_argument("--heartbeat", type=float, default=None)
    parser.add_argument("--detect-every", type=float, default=None)
    args = parser.parse_args()

    sampling = None
    if args.mode == "event":
        from dataclasses import replace

        from .events import SamplingConfig

        sampling = SamplingConfig()
        if args.heartbeat is not None:
            sampling = replace(sampling, heartbeat_seconds=args.heartbeat)
        if args.detect_every is not None:
            sampling = replace(sampling, detect_every_seconds=args.detect_every)

    channels = [int(c) for c in args.cameras.split(",") if c.strip()]
    record = collect(
        channels,
        label=args.label,
        seconds=args.seconds,
        interval=args.interval,
        max_frames=args.max_frames,
        sampling=sampling,
        period=args.period,
    )

    print(f"\nsession {record['session_id']}  [{record['sampling']['strategy']}]")
    for camera in record["cameras"]:
        line = (
            f"  {camera['camera_id']}  {camera['status']:8s} "
            f"kept={camera['frames_kept']:4d} decoded={camera['frames_decoded']:5d} "
            f"{camera['width']}x{camera['height']} @{camera['stream_fps']}fps"
        )
        if camera["error"]:
            line += f"  ERROR {camera['error'][:70]}"
        print(line)
        if camera["by_reason"]:
            print(f"      reasons: {camera['by_reason']}  suppressed: {camera['suppressed']}")
    totals = record["totals"]
    print(
        f"  decoded={totals['frames_decoded']} kept={totals['frames_kept']} "
        f"event={totals['event_triggered']} baseline={totals['baseline_triggered']} "
        f"cameras_ok={totals['cameras_ok']} failed={totals['cameras_failed']} "
        f"reconnects={totals['reconnects']}"
    )
    if totals["by_reason"]:
        print(f"  by reason: {totals['by_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
