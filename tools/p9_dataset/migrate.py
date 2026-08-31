"""Carry kitchen-01's human labels into schema v2.0.0 — faithfully, including
what cannot be carried.

### What migrates cleanly

`head_covering` maps one-to-one, because v1's value already encoded both
questions and the split is mechanical:

```
present      -> HEAD  VISIBLE      + PRESENT
not_visible  -> HEAD  NOT_VISIBLE  + NOT_EVALUATED
absent       -> HEAD  VISIBLE      + ABSENT        (never occurs in kitchen-01)
```

### What does not, and why it is left empty rather than guessed

v1 carries **one** `hand_covering` for both hands. P9 needs left and right
separately, and there is no way to recover which hand a single value described.

Three options were available and two of them are wrong:

* stamp the v1 value onto both hands — **invents** per-hand detail nobody
  recorded, and does it for the exact class the corpus is shortest of;
* parse the free-text note ("both hands bare on the mixing bowl") into labels —
  **inference dressed as annotation**, and it would silently be wrong wherever
  the note is less explicit;
* emit no hand regions and preserve the v1 value verbatim in the note, flagged
  for human re-annotation.

The third is taken. It costs the 3 `absent` hand examples until someone
re-annotates them, and that cost is visible in the report rather than hidden
inside a fabricated left/right split.

### Provenance is split, because the two halves differ

`box_provenance = DETECTOR_DERIVED` — kitchen-01's manifest states the boxes are
detector proposals, which is why that corpus cannot measure detection recall.
`label_provenance = HUMAN_VERIFIED` — the attribute values are human judgements.

v1 could not express that difference. Recording it is most of the reason the
schema separates the two fields.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import (
    AttributeState,
    LabelProvenance,
    Observability,
    QualityStatus,
    Region,
    RegionAnnotation,
    Split,
    SubjectAnnotation,
)

ROOT = Path(__file__).resolve().parents[2]
KITCHEN01 = ROOT / "datasets" / "kitchen-01"

#: kitchen-01's source recording, and the camera it came from.
#:
#: The camera is recorded as `unknown` rather than guessed. The 2026-08-13
#: recording carries no channel overlay that was captured in the extracted
#: frames, and inventing an id would create a false diversity count — the
#: manifest would claim three cameras when it can only prove two.
SESSION_ID = "kitchen01-20260813"
CAMERA_ID = "cam-unknown-kitchen01"
SOURCE_VIDEO = "Screen Recording 2026-08-13 112749.mp4"

_HEAD = {
    "present": (Observability.VISIBLE, AttributeState.PRESENT),
    "absent": (Observability.VISIBLE, AttributeState.ABSENT),
    "not_visible": (Observability.NOT_VISIBLE, AttributeState.NOT_EVALUATED),
    "unknown": (Observability.UNCERTAIN, AttributeState.NOT_EVALUATED),
}


def migrate(annotations_path: Path | None = None) -> list[SubjectAnnotation]:
    """Every kitchen-01 subject, in schema v2.0.0."""
    path = annotations_path or (KITCHEN01 / "annotations" / "kitchen-01.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    annotated_by = document.get("annotated_by", "unknown")

    out: list[SubjectAnnotation] = []
    for frame in document["frames"]:
        frame_name = frame["frame_id"].split("/")[-1]
        for entry in frame["subjects"]:
            attributes = entry.get("attributes", {})
            head_value = str(attributes.get("head_covering", "unknown")).lower()
            observability, state = _HEAD.get(
                head_value, (Observability.UNCERTAIN, AttributeState.NOT_EVALUATED)
            )

            note_parts = [entry.get("note", "")]
            if "hand_covering" in attributes:
                note_parts.append(
                    f"[v1 migration] hand_covering={attributes['hand_covering']!r} "
                    f"applied to BOTH hands jointly; per-hand re-annotation required"
                )
            box = entry["box"]
            out.append(
                SubjectAnnotation(
                    sample_id=f"kitchen01.{frame_name}.{entry['subject_id']}",
                    subject_id=entry["subject_id"],
                    session_id=SESSION_ID,
                    camera_id=CAMERA_ID,
                    frame_id=f"kitchen01.{frame_name}",
                    source_video=SOURCE_VIDEO,
                    box=(box["x1"], box["y1"], box["x2"], box["y2"]),
                    regions=(
                        RegionAnnotation(
                            region=Region.HEAD,
                            observability=observability,
                            state=state,
                            hard_case_tags=(),
                            note=entry.get("note", "")[:160],
                        ),
                    ),
                    label_provenance=LabelProvenance.HUMAN_VERIFIED,
                    box_provenance=LabelProvenance.DETECTOR_DERIVED,
                    annotator=annotated_by,
                    annotated_at="2026-08-13",
                    quality_status=QualityStatus.ACCEPTED,
                    split=Split.UNASSIGNED,
                    note=" | ".join(p for p in note_parts if p)[:400],
                )
            )
    return out


if __name__ == "__main__":
    import collections

    subjects = migrate()
    print(f"migrated {len(subjects)} subjects from kitchen-01 -> schema v2.0.0")
    counts = collections.Counter()
    for subject in subjects:
        head = subject.region_of(Region.HEAD)
        counts[(head.observability.value, head.state.value)] += 1
    for key, n in sorted(counts.items()):
        print(f"  HEAD {key[0]:12s} + {key[1]:14s} {n:3d}")
    print("  hands: 0 regions migrated (per-hand detail not recoverable from v1)")
