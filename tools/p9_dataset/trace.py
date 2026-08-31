"""Perception traces — replayable evidence that stores no pixels.

    python -m tools.p9_dataset.trace record --seconds 240 --label a
    python -m tools.p9_dataset.trace replay

### The problem this solves

P9.6 Phase 1's corpus cannot be replayed. It holds 1,095 JPEGs selected *by* the
Phase 1 policy out of 123,628 decoded frames; feeding those back to a different
policy would ask it to choose from a set the old policy already chose, which
measures nothing. A valid A/B needs the **unfiltered** stream both policies would
have seen.

A trace is that stream, reduced to what the sampler actually consumes: a
perceptual hash and a list of person boxes, per observation. Policies are
replayed against it offline, deterministically, as often as needed.

### It stores no images, and that is a privacy property as well as a size one

A trace of four cameras over four minutes is a few hundred kilobytes of numbers.
The same window as JPEGs is roughly 90 MB of identifiable production CCTV. Since
the trace holds no pixels, **no face, no uniform and no person is recoverable
from it** — it records that a person-shaped box existed at a coordinate, and
nothing about who. That makes it the right artefact to iterate sampling policy
against, and it means policy work no longer requires growing the image corpus.

### The cadence caveat, stated rather than buried

A trace samples at a fixed rate — the live sampler hashes every decoded frame and
detects on a rate limit. Replay therefore operates on a coarser stream than
production. Every policy in a comparison sees **the identical trace**, so the A/B
between them is sound; what a replay cannot claim is the absolute rate a policy
would achieve live. Absolute figures come from live collection, deltas come from
here, and the report keeps them apart.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "datasets" / "p9-traces"

#: Observations per second of wall clock. 4 Hz supports replaying any
#: `detect_every_seconds >= 0.25` and keeps a four-camera trace small.
DEFAULT_HZ = 4.0

#: Observations to look back over when finding the audit's reference frame.
#: 12 at 4 Hz is three seconds — long enough to find a resembling frame, short
#: enough that the comparison is still about the recent past.
REFERENCE_WINDOW = 12

_TRANSPORT = {
    "rtsp_transport": "tcp",
    "stimeout": "15000000",
    "max_delay": "5000000",
    "fflags": "nobuffer",
}


def _record_camera(channel: int, *, seconds: float, hz: float, threads: int = 2) -> dict:
    """Decode one camera and record hashes and person boxes. No pixels kept."""
    import av

    from .candidates import _detector, propose_people
    from .collect import _LazyFrame, _config, _dhash_video_frame

    config = _config(channel)
    detector = _detector(threads)
    password = os.environ.get("CCTV_PASSWORD", "")

    result = {
        "camera_id": config.camera_id,
        "channel": channel,
        "redacted_uri": config.redacted_uri(),
        "status": "not_attempted",
        "error": "",
        "frames_decoded": 0,
        "observations": [],
    }
    interval = 1.0 / hz
    started = time.perf_counter()
    last = -interval
    try:
        with av.open(config.dial_uri(password), options=dict(_TRANSPORT), timeout=25) as container:
            stream = container.streams.video[0]
            result["codec"] = stream.codec_context.name
            result["stream_fps"] = round(float(stream.average_rate or 0), 2)
            result["width"] = stream.codec_context.width
            result["height"] = stream.codec_context.height
            for frame in container.decode(stream):
                now = time.perf_counter()
                elapsed = now - started
                if elapsed >= seconds:
                    break
                result["frames_decoded"] += 1
                if now - last < interval:
                    continue
                last = now
                lazy = _LazyFrame(frame)
                result["observations"].append(
                    {
                        "i": len(result["observations"]),
                        "t": round(elapsed, 3),
                        "hash": _dhash_video_frame(lazy),
                        "boxes": [
                            [[round(v, 5) for v in box], round(score, 4)]
                            for box, score in propose_people(detector, lazy.image)
                        ],
                    }
                )
        result["status"] = "ok"
    except Exception as error:  # noqa: BLE001 - a failed camera is data
        result["error"] = f"{type(error).__name__}: {str(error)[:200]}"
        result["status"] = "failed" if not result["observations"] else "partial"
    result["seconds"] = round(time.perf_counter() - started, 1)
    return result


def record(channels: list[int], *, label: str, seconds: float, hz: float = DEFAULT_HZ,
           period: str = "unspecified") -> dict:
    from .collect import _env

    _env()
    now = datetime.now(UTC)
    trace_id = f"trace-{now.strftime('%Y%m%d')}-{label}"
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(channels)) as pool:
        futures = [
            pool.submit(_record_camera, channel, seconds=seconds, hz=hz)
            for channel in channels
        ]
        cameras = [future.result() for future in futures]

    payload = {
        "_comment": [
            "A PERCEPTION TRACE. Hashes and PERSON boxes only — no pixels, so no",
            "person is identifiable from this file. Recorded so sampling policies",
            "can be A/B tested against the unfiltered stream both would have seen;",
            "the Phase 1 image corpus cannot serve that purpose because it was",
            "already filtered by the policy under test.",
            "No PPE signal is recorded or consulted.",
        ],
        "trace_id": trace_id,
        "provenance": "LIVE_PRODUCTION_TRACE",
        "recorded_at": now.isoformat(timespec="seconds"),
        "collection_day": now.strftime("%Y-%m-%d"),
        "period": period,
        "hz": hz,
        "window_seconds": seconds,
        "cameras": cameras,
        "totals": {
            "frames_decoded": sum(c["frames_decoded"] for c in cameras),
            "observations": sum(len(c["observations"]) for c in cameras),
            "boxes": sum(len(o["boxes"]) for c in cameras for o in c["observations"]),
            "cameras_ok": sum(1 for c in cameras if c["status"] == "ok"),
            "cameras_failed": sum(1 for c in cameras if c["status"] == "failed"),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{trace_id}.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return payload


def load_traces(root: Path = OUT) -> list[dict]:
    if not root.exists():
        return []
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("trace-*.json"))
    ]


def replay(trace: dict, config) -> dict:
    """Run one policy over one trace. Deterministic and side-effect free."""
    from .events import EventSampler, SampleClass

    per_camera = {}
    for camera in trace["cameras"]:
        observations = camera["observations"]
        if not observations:
            continue
        index = {o["i"]: o for o in observations}
        sampler = EventSampler(
            config,
            hash_of=lambda key: index[key]["hash"],
            detect=lambda key: [(tuple(b), s) for b, s in index[key]["boxes"]],
        )
        # A POLICY-INDEPENDENT reference for the Phase 5 audit.
        #
        # Comparing a retained frame against another *retained* frame makes the
        # reference depend on the policy under test, so two policies get
        # different baselines and their per-reason rates stop being comparable.
        # Measured: that alone moved `person_entered` from 80.0 % to 25.7 %
        # between two policies whose entry logic is identical. The reference
        # therefore comes from the raw trace, which no policy can move.
        from .dedupe import DEFAULT_THRESHOLD, hamming

        def _resembling(target: int) -> list[dict]:
            out = []
            for earlier in range(max(0, target - REFERENCE_WINDOW), target):
                other = index.get(earlier)
                if other is None:
                    continue
                if hamming(index[target]["hash"], other["hash"]) <= DEFAULT_THRESHOLD:
                    out.append(other)
            return out

        def _as_reference(other: dict | None) -> dict | None:
            if other is None:
                return None
            return {
                "people": len(other["boxes"]),
                "boxes": [b for b, _ in other["boxes"]],
                "hash": other["hash"],
            }

        kept = []
        for observation in observations:
            decision = sampler.offer(observation["i"], observation["t"], observation["i"])
            if not decision.accepted:
                continue
            # Honour retrospective capture: the frame actually kept may be an
            # earlier one, and its person count is what the corpus would hold.
            target = (
                decision.capture_frame_index
                if decision.capture_frame_index >= 0
                else observation["i"]
            )
            # Two references, both drawn from the raw trace, both published.
            #
            # `reference` is the most recent resembling frame — often only
            # 0.25 s earlier and therefore frequently *after* the change the
            # event is claiming, which makes it a conservative test.
            # `reference_earliest` is the oldest resembling frame in the window
            # and is more likely to precede the change. Neither is obviously the
            # right choice, so the audit runs under both rather than picking the
            # one that flatters.
            resembling = _resembling(target)
            kept.append(
                {
                    "offered": observation["i"],
                    "captured": target,
                    "hash": index[target]["hash"],
                    "people": len(index[target]["boxes"]),
                    # Boxes are carried so the Phase 5 audit can test the
                    # GEOMETRIC consequence an event predicts, not only the
                    # person count. Without them, occlusion and region events
                    # are untestable and must be reported as such.
                    "boxes": [b for b, _ in index[target]["boxes"]],
                    "reference": _as_reference(resembling[-1] if resembling else None),
                    "reference_earliest": _as_reference(
                        resembling[0] if resembling else None
                    ),
                    "reason": decision.reason.value,
                    "sample_class": decision.sample_class.value,
                    "retrospective": decision.capture_frame_index >= 0,
                    "fallback": decision.capture_fallback,
                }
            )
        statistics = sampler.statistics()
        per_camera[camera["camera_id"]] = {
            "observations": len(observations),
            "frames_decoded": camera["frames_decoded"],
            "kept": kept,
            "statistics": statistics,
        }
    return {"trace_id": trace["trace_id"], "by_camera": per_camera}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("record")
    r.add_argument("--cameras", default="11,12,13,14")
    r.add_argument("--seconds", type=float, default=240.0)
    r.add_argument("--hz", type=float, default=DEFAULT_HZ)
    r.add_argument("--label", default="a")
    r.add_argument("--period", default="unspecified")

    c = sub.add_parser(
        "campaign",
        help="record repeatedly with real gaps, so the set spans operating periods",
    )
    c.add_argument("--cameras", default="11,12,13,14")
    c.add_argument("--seconds", type=float, default=240.0)
    c.add_argument("--hz", type=float, default=DEFAULT_HZ)
    c.add_argument("--repeat", type=int, default=6)
    c.add_argument("--gap-minutes", type=float, default=12.0)
    c.add_argument("--prefix", default="c")

    sub.add_parser("list")

    args = parser.parse_args()

    if args.command == "campaign":
        # Spacing is the point. P9.6 Phase 1's twelve sessions fell inside a
        # 35-minute window, which cannot evidence shift diversity however many
        # sessions it contains. Real gaps are the only way to sample a kitchen
        # in different states, and a trace costs kilobytes rather than the ~90 MB
        # of identifiable CCTV the same window would cost as frames.
        channels = [int(x) for x in args.cameras.split(",") if x.strip()]
        for n in range(args.repeat):
            label = f"{args.prefix}{n:02d}"
            stamp = datetime.now(UTC)
            payload = record(
                channels,
                label=label,
                seconds=args.seconds,
                hz=args.hz,
                period=f"observed-{stamp.strftime('%H%MZ')}",
            )
            totals = payload["totals"]
            print(
                f"{payload['trace_id']:26s} {stamp.strftime('%H:%M:%SZ')} "
                f"obs={totals['observations']:5d} boxes={totals['boxes']:5d} "
                f"ok={totals['cameras_ok']}/{len(channels)} "
                f"failed={totals['cameras_failed']}",
                flush=True,
            )
            if n + 1 < args.repeat:
                time.sleep(args.gap_minutes * 60)
        return 0

    if args.command == "record":
        channels = [int(c) for c in args.cameras.split(",") if c.strip()]
        payload = record(
            channels,
            label=args.label,
            seconds=args.seconds,
            hz=args.hz,
            period=args.period,
        )
        print(f"trace {payload['trace_id']}  period={payload['period']}")
        for camera in payload["cameras"]:
            print(
                f"  {camera['camera_id']}  {camera['status']:8s} "
                f"decoded={camera['frames_decoded']:6d} "
                f"observations={len(camera['observations']):5d} "
                f"boxes={sum(len(o['boxes']) for o in camera['observations']):5d}"
                + (f"  ERROR {camera['error'][:60]}" if camera["error"] else "")
            )
        print(f"  totals: {payload['totals']}")
        size = (OUT / f"{payload['trace_id']}.json").stat().st_size
        print(f"  trace size: {size / 1024:.0f} KB (no pixels stored)")
        return 0

    for trace in load_traces():
        print(
            f"{trace['trace_id']:26s} {trace['recorded_at']} {trace['period']:16s} "
            f"obs={trace['totals']['observations']:5d} "
            f"boxes={trace['totals']['boxes']:5d} "
            f"ok={trace['totals']['cameras_ok']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
