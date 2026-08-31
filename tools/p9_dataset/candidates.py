"""Build the human review queue — candidates, which are **not** annotations.

### Why this is a separate artefact

It would have been easy to emit `SubjectAnnotation`s with machine-guessed states
and a `MACHINE_PROPOSED` provenance flag. That is the shape of every dataset that
later turns out to have been scoring a model against its own opinions.

So a candidate carries **no attribute label at all**. It carries a frame, a
proposed box, a crop, and geometric hints — and the hints are geometry, not
judgements: where the head *is*, never what is *on* it. The covering question is
left entirely for a person, because it is the only question that matters and the
only one a model cannot be trusted to seed.

A candidate becomes an annotation when a human answers it. There is no code path
that promotes one automatically, and `validate.py` rejects a machine provenance
in any evaluation split.

### Where candidates come from

Three sources, deliberately different from each other:

| source | why it is here |
|---|---|
| `Screen Recording 2026-08-17 122553` (cam-11) | second camera, close overhead prep line |
| `Screen Recording 2026-08-17 122832` (cam-13) | third camera, wide zone, more subjects and distances |
| `data/evidence/` | real production frames from 4 cameras, including the only observed uncovered heads |

The evidence store matters most. It is the only source in the workspace known to
contain `head_covering = ABSENT` — the class the corpus has zero of, and without
which PPE violation recall cannot be measured at all.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT.parent
OUT = ROOT / "datasets" / "p9-candidates"

#: Sources, with the camera each was observed to carry in its overlay.
#:
#: Camera ids are read from the recordings' own burnt-in channel number, not
#: assumed. Where a source carries no legible channel, the id is `unknown` and
#: the manifest says so rather than inventing diversity.
VIDEO_SOURCES = (
    {
        "session_id": "kitchen-20260817-a",
        "camera_id": "cam-11",
        "path": ATLAS / "media" / "Screen Recording 2026-08-17 122553.mp4",
        "note": "close overhead, prep line, 1-2 subjects, 1714x966",
    },
    {
        "session_id": "kitchen-20260817-b",
        "camera_id": "cam-13",
        "path": ATLAS / "media" / "Screen Recording 2026-08-17 122832.mp4",
        "note": "wide, storage/cook zone, up to 4 subjects, more distance, 1718x978",
    },
)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One person proposed for annotation. **Carries no PPE label.**"""

    candidate_id: str
    session_id: str
    camera_id: str
    frame_id: str
    source: str
    frame_path: str
    box: tuple[float, float, float, float]
    box_provenance: str = "detector_derived"
    detector_confidence: float = 0.0

    #: Geometry only. Where the head is, never what is on it.
    head_observability_hint: str = "unknown"
    hint_provenance: str = "machine_proposed"
    hint_confidence: float = 0.0

    #: Deliberately absent: head_covering, face_covering, glove states.
    #: A human supplies those, or they do not exist.
    review_status: str = "awaiting_human_annotation"
    suggested_tags: tuple[str, ...] = field(default_factory=tuple)


def _detector(threads: int | None = None):
    """The production detector, bound exactly as the composition root binds it.

    `threads` caps intra-op parallelism, which matters when several detectors run
    at once. The live collector observes four cameras concurrently, so on a
    12-core machine four sessions at the default of one thread per core ask for
    roughly 48 threads on 12 cores.

    This is a precaution against oversubscription, not the fix for a measured
    collapse — the collapse that prompted it turned out to be an orphaned
    collector holding its own four RTSP sessions, and with the machine clean the
    four cameras decode at full stream rate either way.

    Left as `None` — every core — for the offline paths, which run one detector
    at a time and want the throughput.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(ROOT / "models" / "yolov8n.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _pose():
    import onnxruntime as ort

    path = ROOT / "models" / "yolov8n-pose.onnx"
    if not path.exists():
        return None
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _letterbox(image, size: int = 640):
    import numpy as np

    from vision_os.adapters.detection.letterbox import compute_transform

    width, height = image.size
    transform = compute_transform(
        source_width=width, source_height=height, target_width=size, target_height=size
    )
    scaled = image.resize(
        (max(1, round(width * transform.scale)), max(1, round(height * transform.scale)))
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    ox, oy = int(transform.pad_x), int(transform.pad_y)
    canvas[oy : oy + scaled.height, ox : ox + scaled.width] = np.array(scaled)
    tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return tensor, transform, width, height


def propose_people(session, image, *, confidence: float = 0.35):
    """Person boxes from the production detector, in normalised coordinates.

    Provenance is `detector_derived` and it is recorded on every candidate. This
    is exactly the property that makes kitchen-01 unable to measure detection
    recall, and repeating it silently would repeat that mistake — a human
    reviewer must be able to add boxes the detector missed.
    """
    import numpy as np

    tensor, transform, width, height = _letterbox(image)
    name = session.get_inputs()[0].name
    raw = session.run(None, {name: tensor})[0][0]  # (84, N)

    out = []
    for column in raw.T:
        scores = column[4:]
        best = int(np.argmax(scores))
        if best != 0:  # COCO class 0 is person
            continue
        score = float(scores[best])
        if score < confidence:
            continue
        cx, cy, bw, bh = (float(v) for v in column[:4])
        box = (
            max(0.0, (cx - bw / 2 - transform.pad_x) / transform.scale / width),
            max(0.0, (cy - bh / 2 - transform.pad_y) / transform.scale / height),
            min(1.0, (cx + bw / 2 - transform.pad_x) / transform.scale / width),
            min(1.0, (cy + bh / 2 - transform.pad_y) / transform.scale / height),
        )
        if box[2] > box[0] and box[3] > box[1]:
            out.append((box, score))
    return _nms(out)


def _nms(boxes, threshold: float = 0.55):
    kept = []
    for box, score in sorted(boxes, key=lambda b: -b[1]):
        if all(_iou(box, k[0]) < threshold for k in kept):
            kept.append((box, score))
    return kept


def _iou(a, b) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def head_hint(pose_adapter, image, box) -> tuple[str, float]:
    """Geometric observability hint from P8's pose producer.

    Reused rather than reimplemented, so the hint a reviewer sees is the same
    signal production acts on. It answers *where the head is*; it has no opinion
    about coverings and the port's state enum offers none.
    """
    if pose_adapter is None:
        return "unknown", 0.0
    import numpy as np

    from vision_os.adapters.perception import PoseRegionObservability
    from vision_os.core.model.ids import AttributeKey, CameraId
    from vision_os.core.model.space import Box
    from vision_os.core.ports.region_observability import RegionObservabilityRequest

    producer = PoseRegionObservability(session=pose_adapter)
    verdict = producer.assess(
        RegionObservabilityRequest(
            camera_id=CameraId("p9"),
            box=Box(*box),
            attributes=(AttributeKey("head_covering"),),
            source_width=image.width,
            source_height=image.height,
            pixels=memoryview(np.asarray(image, dtype=np.uint8).tobytes()),
            frame_key="",
        )
    )[0]
    return verdict.state.value, round(verdict.confidence, 3)


def from_video(spec: dict, *, every: int = 60, limit: int = 40) -> list[Candidate]:
    """Sample frames at a fixed stride and propose people in each.

    A stride, never a random sample: neighbouring frames of one worker are near
    duplicates, and 60 frames at 30 fps is 2 seconds — far enough apart to be
    different postures, close enough to keep a session's coverage honest.
    """
    import av

    path = spec["path"]
    if not path.exists():
        return []
    detector, pose = _detector(), _pose()
    frames_dir = OUT / "frames" / spec["session_id"]
    frames_dir.mkdir(parents=True, exist_ok=True)

    out: list[Candidate] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        index = kept = 0
        for frame in container.decode(stream):
            if index % every == 0 and kept < limit:
                image = frame.to_image().convert("RGB")
                frame_id = f"{spec['session_id']}.f{index:06d}"
                frame_path = frames_dir / f"f{index:06d}.jpg"
                if not frame_path.exists():
                    image.save(frame_path, quality=90)
                for n, (box, score) in enumerate(propose_people(detector, image)):
                    hint, hint_confidence = head_hint(pose, image, box)
                    out.append(
                        Candidate(
                            candidate_id=f"{frame_id}.p{n}",
                            session_id=spec["session_id"],
                            camera_id=spec["camera_id"],
                            frame_id=frame_id,
                            source=path.name,
                            frame_path=str(frame_path.relative_to(ROOT)),
                            box=box,
                            detector_confidence=round(score, 3),
                            head_observability_hint=hint,
                            hint_confidence=hint_confidence,
                        )
                    )
                kept += 1
            index += 1
            if kept >= limit:
                break
    return out


def from_evidence(*, limit: int = 60) -> list[Candidate]:
    """Candidates from real production frames.

    The highest-value source: these are the frames the live system actually acted
    on, from four cameras, and the only place in the workspace observed to
    contain uncovered heads.

    Camera and timestamp are **not** recoverable from a content-addressed blob
    without the incident database, so both are recorded as `unknown` rather than
    guessed. A reviewer reads the burnt-in overlay; the manifest does not pretend
    to know it.
    """
    from PIL import Image

    store = ROOT / "data" / "evidence"
    if not store.exists():
        return []
    detector, pose = _detector(), _pose()
    frames_dir = OUT / "frames" / "production-evidence"
    frames_dir.mkdir(parents=True, exist_ok=True)

    blobs = []
    for blob in sorted(store.rglob("*")):
        if not blob.is_file():
            continue
        try:
            with Image.open(blob) as probe:
                if probe.size in ((1920, 1080), (960, 576)):
                    blobs.append((blob, probe.size))
        except Exception:
            continue

    step = max(1, len(blobs) // limit)
    out: list[Candidate] = []
    for blob, _size in blobs[::step][:limit]:
        image = Image.open(blob).convert("RGB")
        digest = hashlib.sha256(blob.read_bytes()).hexdigest()[:16]
        frame_id = f"evidence.{digest}"
        frame_path = frames_dir / f"{digest}.jpg"
        if not frame_path.exists():
            image.save(frame_path, quality=90)
        for n, (box, score) in enumerate(propose_people(detector, image)):
            hint, hint_confidence = head_hint(pose, image, box)
            out.append(
                Candidate(
                    candidate_id=f"{frame_id}.p{n}",
                    session_id="production-evidence",
                    camera_id="unknown-read-from-overlay",
                    frame_id=frame_id,
                    source="data/evidence",
                    frame_path=str(frame_path.relative_to(ROOT)),
                    box=box,
                    detector_confidence=round(score, 3),
                    head_observability_hint=hint,
                    hint_confidence=hint_confidence,
                )
            )
    return out


def build_queue() -> dict:
    candidates: list[Candidate] = []
    for spec in VIDEO_SOURCES:
        candidates.extend(from_video(spec))
    candidates.extend(from_evidence())

    by_session: dict[str, int] = {}
    by_hint: dict[str, int] = {}
    for candidate in candidates:
        by_session[candidate.session_id] = by_session.get(candidate.session_id, 0) + 1
        by_hint[candidate.head_observability_hint] = (
            by_hint.get(candidate.head_observability_hint, 0) + 1
        )

    return {
        "_comment": [
            "A HUMAN REVIEW QUEUE. These are CANDIDATES, not annotations.",
            "No entry carries a PPE label of any kind — not even a guessed one.",
            "Boxes are detector proposals: a reviewer must be able to ADD people",
            "the detector missed, or this corpus repeats kitchen-01's inability",
            "to measure detection recall.",
            "head_observability_hint is GEOMETRY from the P8 pose producer. It",
            "says where a head is, never what is on it, and it is advisory.",
        ],
        "built_at": __import__("datetime").datetime.now(
            __import__("datetime").UTC
        ).isoformat(timespec="seconds"),
        "candidate_count": len(candidates),
        "by_session": by_session,
        "by_head_hint": by_hint,
        "candidates": [asdict(c) for c in candidates],
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    queue = build_queue()
    path = OUT / "review_queue.json"
    path.write_text(json.dumps(queue, indent=1) + "\n", encoding="utf-8")
    print(f"{queue['candidate_count']} candidates -> {path.relative_to(ROOT)}")
    print("  by session:", queue["by_session"])
    print("  by head hint:", queue["by_head_hint"])
