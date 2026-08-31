"""Person candidates from the retained live frames — Phases 17 and 19.

    python -m tools.p9_dataset.live_queue

Candidates, not annotations: **no PPE label of any kind**, not even a guessed
one. Boxes are detector proposals and are marked as such, because a human must
be able to reject a false detection and **add a person the detector missed** —
the only route to a measurable detection recall, and the thing kitchen-01 could
never do.

### What P9.6 adds

**Provenance.** Every candidate now carries camera, calendar day, session,
sequence, offset and the sampling reason that selected its frame, so a
leakage-safe split remains constructible later (Phase 19). P9.5's candidates
carried camera and session but no reason, because nothing had chosen them except
a clock.

**Event-aware retention.** Deduplication consults the sampling reason before
removing a frame, so a distant worker crossing a wide shot is not deleted for
resembling the frame before it (Phase 12).

**Review priority that is not "where the model failed".** Phase 17 asks for
representative production samples, not a queue of the current model's mistakes —
that would rebuild the corpus around the model's beliefs, which is exactly the
failure the event sampler was designed to avoid. The priority signals here are
therefore properties of the *observation*: a low-confidence detection, a
population change, an occlusion transition, a crowded frame. None of them is a
PPE signal.
"""

from __future__ import annotations

import collections
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .candidates import Candidate, _detector, _pose, head_hint, propose_people
from .dedupe import (
    ROOT,
    Redundancy,
    classification_summary,
    classify,
    frame_key,
    load_frames,
    session_reasons,
)

LIVE = ROOT / "datasets" / "p9-live"

#: Detector confidence below which a proposal is flagged for careful review.
#: Not a rejection: a weak detection is often a distant or occluded person, and
#: those are the hard cases the corpus is short of.
LOW_CONFIDENCE = 0.50

#: Person count at or above which a frame is flagged as crowded.
CROWDED = 3


@dataclass(frozen=True, slots=True)
class LiveCandidate:
    """A candidate with the full provenance chain Phase 19 requires."""

    candidate: Candidate
    collection_day: str
    period: str
    sequence: int
    offset_seconds: float
    sampling_reason: str
    people_in_frame: int
    review_flags: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        payload = asdict(self.candidate)
        payload.update(
            collection_day=self.collection_day,
            period=self.period,
            sequence=self.sequence,
            offset_seconds=self.offset_seconds,
            sampling_reason=self.sampling_reason,
            people_in_frame=self.people_in_frame,
            review_flags=list(self.review_flags),
        )
        return payload


def _sessions() -> dict:
    out = {}
    for directory in sorted(LIVE.glob("live-*")):
        record = directory / "session.json"
        if record.exists():
            payload = json.loads(record.read_text(encoding="utf-8"))
            out[payload["session_id"]] = payload
    return out


def _flags(sample: dict, score: float, people: int) -> tuple[str, ...]:
    """Why a reviewer might want to look at this one first.

    Observation properties only. Nothing here consults a PPE model, and nothing
    here is a label.
    """
    flags = []
    reason = sample.get("sampling_reason", "")
    if score < LOW_CONFIDENCE:
        flags.append("low_detector_confidence")
    if reason in ("person_entered", "person_left", "person_count_changed"):
        flags.append("population_change")
    if reason == "occlusion_changed":
        flags.append("occlusion_transition")
    if reason == "region_transition":
        flags.append("region_change")
    if reason == "periodic_heartbeat":
        flags.append("baseline_sample")
    if people >= CROWDED:
        flags.append("crowded")
    return tuple(flags)


def build() -> dict:
    from PIL import Image

    sessions = _sessions()
    frames = load_frames()
    verdicts = classify(frames)
    retained = [
        f
        for f in frames
        if verdicts.get(frame_key(f.path))
        in (Redundancy.UNIQUE, Redundancy.MEANINGFUL_CHANGE)
    ]

    samples_by_session = {
        session_id: session_reasons(LIVE / session_id) for session_id in sessions
    }

    detector, pose = _detector(), _pose()
    out: list[LiveCandidate] = []
    for frame in retained:
        session = sessions.get(frame.session_id, {})
        sample = samples_by_session.get(frame.session_id, {}).get(frame.path.name, {})
        image = Image.open(frame.path).convert("RGB")
        proposals = propose_people(detector, image)
        for n, (box, score) in enumerate(proposals):
            hint, confidence = head_hint(pose, image, box)
            out.append(
                LiveCandidate(
                    candidate=Candidate(
                        candidate_id=(
                            f"{frame.session_id}.{frame.camera_id}."
                            f"{frame.order:04d}.p{n}"
                        ),
                        session_id=frame.session_id,
                        camera_id=frame.camera_id,
                        frame_id=f"{frame.session_id}.{frame.camera_id}.{frame.order:04d}",
                        source="LIVE_PRODUCTION",
                        frame_path=frame_key(frame.path),
                        box=box,
                        detector_confidence=round(score, 3),
                        head_observability_hint=hint,
                        hint_confidence=confidence,
                    ),
                    collection_day=session.get("collection_day", ""),
                    period=session.get("period", "unspecified"),
                    sequence=sample.get("sequence", frame.order),
                    offset_seconds=sample.get("offset_seconds", 0.0),
                    sampling_reason=sample.get("sampling_reason", ""),
                    people_in_frame=len(proposals),
                    review_flags=_flags(sample, score, len(proposals)),
                )
            )

    by_camera: collections.Counter = collections.Counter()
    by_session: collections.Counter = collections.Counter()
    by_day: collections.Counter = collections.Counter()
    by_period: collections.Counter = collections.Counter()
    by_reason: collections.Counter = collections.Counter()
    by_hint: collections.Counter = collections.Counter()
    by_flag: collections.Counter = collections.Counter()
    for entry in out:
        by_camera[entry.candidate.camera_id] += 1
        by_session[entry.candidate.session_id] += 1
        by_day[entry.collection_day] += 1
        by_period[entry.period] += 1
        by_reason[entry.sampling_reason or "unrecorded"] += 1
        by_hint[entry.candidate.head_observability_hint] += 1
        for flag in entry.review_flags:
            by_flag[flag] += 1

    frames_with_people = {e.candidate.frame_id for e in out}
    return {
        "_comment": [
            "LIVE_PRODUCTION human review queue. CANDIDATES, not annotations.",
            "No PPE label of any kind appears here — not even a guessed one.",
            "Boxes are DETECTOR PROPOSALS. A reviewer must reject false ones and",
            "ADD people the detector missed, or detection recall stays unmeasurable.",
            "head_observability_hint is geometry from P8: where a head is, never",
            "what is on it. Advisory. Where the image disagrees, the image wins.",
            "review_flags are properties of the OBSERVATION — confidence, crowding,",
            "population and occlusion transitions. None is a PPE signal, and the",
            "queue is deliberately not ordered by where the model failed.",
        ],
        "provenance": "LIVE_PRODUCTION",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "deduplication": classification_summary(verdicts),
        "frames_retained": len(retained),
        "frames_with_at_least_one_person": len(frames_with_people),
        "frames_with_no_person_detected": len(retained) - len(frames_with_people),
        "candidate_count": len(out),
        "by_camera": dict(sorted(by_camera.items())),
        "by_session": dict(sorted(by_session.items())),
        "by_day": dict(sorted(by_day.items())),
        "by_period": dict(sorted(by_period.items())),
        "by_sampling_reason": dict(sorted(by_reason.items(), key=lambda kv: -kv[1])),
        "by_head_hint": dict(sorted(by_hint.items())),
        "by_review_flag": dict(sorted(by_flag.items(), key=lambda kv: -kv[1])),
        "candidates": [entry.as_dict() for entry in out],
    }


if __name__ == "__main__":
    queue = build()
    path = LIVE / "review_queue.json"
    path.write_text(json.dumps(queue, indent=1) + "\n", encoding="utf-8")
    print(
        f"{queue['candidate_count']} candidates from {queue['frames_retained']} "
        f"retained frames"
    )
    print(f"  dedup      : {queue['deduplication']}")
    print(f"  with person: {queue['frames_with_at_least_one_person']}")
    print(f"  none       : {queue['frames_with_no_person_detected']}")
    print(f"  by camera  : {queue['by_camera']}")
    print(f"  by day     : {queue['by_day']}")
    print(f"  by period  : {queue['by_period']}")
    print(f"  by reason  : {queue['by_sampling_reason']}")
    print(f"  by flag    : {queue['by_review_flag']}")
    print(f"  -> {path.relative_to(ROOT)}")
