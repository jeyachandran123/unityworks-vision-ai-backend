"""Prepare frames for human PPE annotation.

    python -m tools.vision_eval.annotate_ppe extract  <video> --dataset <root> \
        --restaurant r1 --camera cam-1 --fps 0.5
    python -m tools.vision_eval.annotate_ppe review   --dataset <root>
    python -m tools.vision_eval.annotate_ppe validate --dataset <root>

Produces the material a person needs to make a defensible judgement, and nothing
that would bias it.

**No model output is rendered.** No boxes, no detections, no pose skeletons, no
VLM answers. The annotator sees the footage. Showing a proposal first anchors the
judgement to it — and an annotator who agrees with the detector 95 % of the time
has produced a very expensive copy of the detector rather than ground truth.

Whether the bundled detector found each person is recorded **afterwards**, by
matching finished annotations against detections, purely so detection recall
becomes computable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ppe_dataset import load, quality_report, validate

#: What an annotator is asked to look at, in order.
#:
#: Full frame first so people are counted before anyone thinks about PPE — the
#: opposite order makes it easy to miss a worker in the background who was never
#: cropped, and a person missed here is invisible to every metric downstream.
REVIEW_LEVELS = ("full_frame", "person_crop", "evidence_region", "enlarged_region")


def extract(args) -> int:
    """Sample frames from footage at a fixed interval, and nothing more."""
    import av
    from PIL import Image

    root: Path = args.dataset
    frames_dir = root / "review" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    container = av.open(str(args.video))
    stream = container.streams.video[0]
    native_fps = float(stream.average_rate)
    step = max(1, round(native_fps / args.fps))

    written = []
    for index, frame in enumerate(container.decode(video=0)):
        if index % step:
            continue
        image = frame.to_image().convert("RGB")
        name = f"{args.camera}_{index:06d}.jpg"
        image.save(frames_dir / name, quality=95)
        written.append({
            "frame_id": f"{args.video_id or args.video.stem}/{index:06d}",
            "video_id": args.video_id or args.video.stem,
            "camera_id": args.camera,
            "restaurant_id": args.restaurant,
            "frame_index": index,
            "timestamp_ms": index / native_fps * 1000.0,
            "image_path": f"review/frames/{name}",
            "persons": [],
            "annotation_source": "human_visual_inspection",
        })
    container.close()

    stub = root / "annotations" / f"{args.camera}.todo.json"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(
        json.dumps({"schema_version": "2.0.0", "source": args.video.name,
                    "frames": written}, indent=2),
        encoding="utf-8",
    )
    print(f"extracted {len(written)} frames at {args.fps} fps -> {frames_dir}")
    print(f"annotation stub -> {stub}")
    print("\nEvery person visible in each frame must be annotated, including any")
    print("the detector would miss. Leave `persons` empty only for empty frames.")
    return 0


def review(args) -> int:
    """Render per-person review material for frames already annotated.

    Run *after* a first pass, to check work — the crops are cut from the
    annotator's own boxes, so this cannot introduce a detector's opinion.
    """
    from PIL import Image, ImageDraw

    root: Path = args.dataset
    out = root / "review" / "persons"
    out.mkdir(parents=True, exist_ok=True)

    count = 0
    for path in sorted((root / "annotations").glob("*.json")):
        if path.name.endswith(".todo.json"):
            continue
        for frame in load(path):
            image_path = root / frame.image_path
            if not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            for person in frame.persons:
                b = person.box
                pad_x = (b.x2 - b.x1) * width * 0.25
                pad_y = (b.y2 - b.y1) * height * 0.25
                crop = image.crop((
                    max(0, int(b.x1 * width - pad_x)), max(0, int(b.y1 * height - pad_y)),
                    min(width, int(b.x2 * width + pad_x)), min(height, int(b.y2 * height + pad_y)),
                ))
                scale = 640 / max(crop.height, 1)
                crop = crop.resize(
                    (max(1, int(crop.width * scale)), 640), Image.LANCZOS
                )
                banner = Image.new("RGB", (crop.width, crop.height + 26), (15, 15, 15))
                banner.paste(crop, (0, 26))
                states = "  ".join(
                    f"{k}={v.state.value}/{v.observability.value}"
                    for k, v in person.ppe.items()
                )
                ImageDraw.Draw(banner).text(
                    (4, 7), f"{frame.frame_id} {person.person_id}  {states}",
                    fill=(0, 255, 120),
                )
                banner.save(out / f"{frame.frame_index:06d}_{person.person_id}.jpg", quality=94)
                count += 1
    print(f"rendered {count} person reviews -> {out}")
    return 0


def check(args) -> int:
    """Validate annotations and write the quality report."""
    root: Path = args.dataset
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    attributes = manifest["attributes"]

    frames = []
    for path in sorted((root / "annotations").glob("*.json")):
        if path.name.endswith(".todo.json"):
            continue
        frames.extend(load(path))

    issues = validate(frames, attributes=attributes)
    report = quality_report(frames, attributes)
    report["validation_issues"] = [
        {"frame_id": i.frame_id, "person_id": i.person_id, "attribute": i.attribute,
         "rule": i.rule, "detail": i.detail}
        for i in issues
    ]

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "phase5_dataset_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"frames {report['frames']}   annotated persons {report['annotated_persons']}")
    print(f"restaurants {report['restaurants']}  cameras {report['cameras']}")
    for attribute, block in report["attributes"].items():
        print(f"\n{attribute}: {block['counts']}")
        print(f"  {block['verdict']}")
    if issues:
        print(f"\n{len(issues)} validation issue(s):")
        for issue in issues[:20]:
            print(f"  {issue}")
    else:
        print("\nno validation issues")
    print(f"\nwritten: {reports / 'phase5_dataset_report.json'}")
    return 1 if any(i.rule == "decided_state_without_observability" for i in issues) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract", help="sample frames from footage")
    e.add_argument("video", type=Path)
    e.add_argument("--dataset", type=Path, required=True)
    e.add_argument("--restaurant", required=True)
    e.add_argument("--camera", required=True)
    e.add_argument("--video-id", default="")
    e.add_argument("--fps", type=float, default=0.5,
                   help="sampling rate; low by default because adjacent CCTV "
                        "frames are near-duplicates and inflate a dataset "
                        "without adding information")
    e.set_defaults(func=extract)

    r = sub.add_parser("review", help="render per-person review crops")
    r.add_argument("--dataset", type=Path, required=True)
    r.set_defaults(func=review)

    v = sub.add_parser("validate", help="validate and write the quality report")
    v.add_argument("--dataset", type=Path, required=True)
    v.set_defaults(func=check)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
