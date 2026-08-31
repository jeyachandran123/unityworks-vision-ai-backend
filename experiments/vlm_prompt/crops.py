"""Generate the evidence crops ONCE, so every variant sees identical bytes.

The single most important control in this experiment. If each variant re-cut its
own crop, a difference in results could be a difference in pixels, and the whole
comparison would be uninterpretable.

Geometry is the production geometry, taken from the shipped policy document
rather than restated here: `head_covering` and `face_covering` declare the band
`(top 0.0, height 0.45)` at output size 448, and share one crop because M8 groups
by exact declared band.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets" / "kitchen-01"
POLICY = ROOT / "config" / "policies" / "kitchen-safety.example.json"
OUT = Path(__file__).resolve().parent / "crops"

#: The attribute under study. `face_covering` rides the same crop but has no
#: ground truth in this corpus, so it is not scored.
ATTRIBUTE = "head_covering"


@dataclass(frozen=True, slots=True)
class Subject:
    frame: str
    subject: str
    truth: str
    note: str
    crop_path: Path
    crop_sha256: str
    width: int
    height: int

    @property
    def key(self) -> str:
        return f"{self.frame}/{self.subject}"


def _policy():
    from vision_os.adapters.configuration.semantic_policy import SemanticPolicy

    return SemanticPolicy.from_document(json.loads(POLICY.read_text(encoding="utf-8")))


def build(force: bool = False) -> list[Subject]:
    """Cut every subject's head-band crop with the real production strategy."""
    from PIL import Image

    from vision_os.adapters.cropping.strategies import PartFocusedCropStrategy
    from vision_os.core.model.ids import AttributeKey, ClassId
    from vision_os.core.model.space import Box

    policy = _policy()
    strategy = PartFocusedCropStrategy(
        regions=policy.evidence_regions, output_sizes=policy.output_sizes
    )
    attributes = (AttributeKey(ATTRIBUTE), AttributeKey("face_covering"))

    OUT.mkdir(parents=True, exist_ok=True)
    annotations = json.loads(
        (DATASET / "annotations" / "kitchen-01.json").read_text(encoding="utf-8")
    )

    subjects: list[Subject] = []
    for frame in annotations["frames"]:
        name = frame["frame_id"].split("/")[-1]
        image = Image.open(DATASET / frame["image_path"]).convert("RGB")
        width, height = image.size
        for entry in frame["subjects"]:
            box = entry["box"]
            plan = strategy.plan(
                box=Box(box["x1"], box["y1"], box["x2"], box["y2"]),
                class_id=ClassId("person"),
                source_width=width,
                source_height=height,
                attributes=attributes,
            )
            padded = plan.padded_box
            cut = image.crop(
                (
                    int(padded.x1 * width),
                    int(padded.y1 * height),
                    int(padded.x2 * width),
                    int(padded.y2 * height),
                )
            )
            canvas = Image.new("RGB", (plan.output_width, plan.output_height), (0, 0, 0))
            thumb = cut.copy()
            thumb.thumbnail((plan.output_width, plan.output_height), Image.LANCZOS)
            canvas.paste(
                thumb,
                (
                    (plan.output_width - thumb.width) // 2,
                    (plan.output_height - thumb.height) // 2,
                ),
            )

            path = OUT / f"{name}_{entry['subject_id']}.png"
            if force or not path.exists():
                canvas.save(path, format="PNG", optimize=False)
            subjects.append(
                Subject(
                    frame=name,
                    subject=entry["subject_id"],
                    truth=entry["attributes"].get(ATTRIBUTE, ""),
                    note=entry.get("note", ""),
                    crop_path=path,
                    crop_sha256=hashlib.sha256(path.read_bytes()).hexdigest()[:16],
                    width=canvas.width,
                    height=canvas.height,
                )
            )
    return subjects


if __name__ == "__main__":
    built = build(force=True)
    print(f"{len(built)} crops -> {OUT}")
    sizes = {(s.width, s.height) for s in built}
    print("sizes:", sizes)
    digest = hashlib.sha256(
        "".join(s.crop_sha256 for s in built).encode()
    ).hexdigest()[:16]
    print("corpus digest:", digest)
