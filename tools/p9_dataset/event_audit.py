"""Does each event predict something, and did it happen? — Phase 5.

An audit that measures the wrong consequence is worse than no audit, because it
manufactures confidence. This programme has already made that mistake once: the
P9.6 Phase 1 rescue audit scored `occlusion_changed` by person-count change and
reported 31 % corroboration, when an unchanged count is precisely what that event
*predicts*. The number was an artefact of the wrong denominator.

So this module starts from an explicit contract. For every reason:

1. what observable consequence does it claim?
2. is that consequence derivable from the evidence in hand?
3. only if both are answered is it scored.

Anything else is reported `UNTESTABLE` or `TAUTOLOGICAL` and **carries no rate**.

### The comparison pair

Not "this frame versus the previous one" — that would be tautological, since the
event fired on exactly that comparison. The audit compares a retained frame with
**the earlier frame it perceptually resembles**, which is the pair deduplication
would have collapsed. The question it answers is the one that matters for the
corpus: *given that these two frames look alike, did the thing the event claimed
actually differ between them?*

### Tautological is not a passing grade

`SCENE_CHANGED` and `PERIODIC_HEARTBEAT` are true by construction — one fires on a
hash distance and the other on a clock, and both would trivially "corroborate"
against any pair. They are labelled as such rather than credited, because a
metric that cannot fail is decoration.
"""

from __future__ import annotations

import enum

from .events import (
    DepartureRule,
    SamplingConfig,
    Track,
    _cell,
    _occlusion_state,
    iou,
)


class Testability(enum.Enum):
    TESTABLE = "testable"
    """The consequence is derivable from the evidence and can fail."""

    TAUTOLOGICAL = "tautological"
    """True by construction. Reported, never scored."""

    UNTESTABLE = "untestable"
    """The consequence is real but not derivable from what was recorded."""


#: The contract. One row per reason: what it claims, and whether we can check.
#:
#: **`person_left`'s claim depends on the departure rule**, and getting that wrong
#: is not a rounding error — it inverts the prediction. Under `ON_EXPIRY` the
#: retained frame is the one in which the absence was admitted, so it holds
#: *fewer* people than the frame it resembles. Under `LAST_CONFIRMED` the
#: retained frame is the departing person's last sighting, so it holds *more*.
#: Scoring the second with the first's contract reports a collapse that is
#: entirely an artefact of the auditor. Use `consequences_for(config)`, never this
#: dict directly, wherever the rule matters.
CONSEQUENCES = {
    "person_entered": (
        "the person count is higher than in the frame it resembles",
        Testability.TESTABLE,
    ),
    "person_left": (
        "the person count is lower than in the frame it resembles",
        Testability.TESTABLE,
    ),
    "person_count_changed": (
        "the person count differs from the frame it resembles",
        Testability.TESTABLE,
    ),
    "occlusion_changed": (
        "some person's overlap-or-edge state differs; the count need not",
        Testability.TESTABLE,
    ),
    "region_transition": (
        "the multiset of occupied grid cells differs",
        Testability.TESTABLE,
    ),
    "bbox_changed": (
        "a matched box differs in IoU or area beyond threshold",
        Testability.TESTABLE,
    ),
    "person_moved": (
        "a matched box centre has moved at least move_fraction of its diagonal",
        Testability.TESTABLE,
    ),
    "scene_changed": (
        "the perceptual hash differs — true by construction of the trigger",
        Testability.TAUTOLOGICAL,
    ),
    "periodic_heartbeat": (
        "time has passed — true by construction of the trigger",
        Testability.TAUTOLOGICAL,
    ),
    "manual_review": (
        "a person wanted it; no observable consequence exists",
        Testability.UNTESTABLE,
    ),
}


def consequences_for(config: SamplingConfig) -> dict:
    """The contract, specialised to the policy actually in force.

    Only `person_left` differs, and it differs completely: see `CONSEQUENCES`.
    Under `LAST_CONFIRMED` the retained frame is the departing person's final
    sighting, so the claim worth testing is that **the evidence frame is not
    empty** — which is the entire purpose of the rule and the thing P9.6 Phase 1
    failed at 65.9 % of the time.
    """
    contract = dict(CONSEQUENCES)
    if config.departure_rule is DepartureRule.LAST_CONFIRMED:
        contract["person_left"] = (
            "the retained frame still shows the departing person (people > 0)",
            Testability.TESTABLE,
        )
    return contract


def _tracks(boxes) -> list[Track]:
    return [
        Track(track_id=n, box=tuple(box), confidence=1.0, hits=9, confirmed=True)
        for n, box in enumerate(boxes)
    ]


def _occlusion_signature(boxes, config: SamplingConfig) -> tuple:
    tracks = _tracks(boxes)
    return tuple(sorted(_occlusion_state(t, tracks, config) for t in tracks))


def _cell_signature(boxes, config: SamplingConfig) -> tuple:
    return tuple(sorted(_cell(t, config.grid) for t in _tracks(boxes)))


def _best_pairs(a, b):
    """Greedy IoU pairing between two box sets, for geometry comparisons."""
    used = set()
    for box_a in a:
        best, score = None, 0.0
        for n, box_b in enumerate(b):
            if n in used:
                continue
            overlap = iou(tuple(box_a), tuple(box_b))
            if overlap > score:
                best, score = n, overlap
        if best is not None and score > 0:
            used.add(best)
            yield tuple(box_a), tuple(b[best]), score


def _holds(reason: str, current: dict, match: dict, config: SamplingConfig) -> bool | None:
    """Did the predicted consequence actually occur between the two frames?"""
    boxes_a, boxes_b = current.get("boxes") or [], match.get("boxes") or []
    people_a, people_b = current["people"], match["people"]

    if reason == "person_entered":
        return people_a > people_b
    if reason == "person_left":
        if config.departure_rule is DepartureRule.LAST_CONFIRMED:
            # A different claim, because the rule keeps a different frame.
            return people_a > 0
        return people_a < people_b
    if reason == "person_count_changed":
        return people_a != people_b
    if reason == "occlusion_changed":
        return _occlusion_signature(boxes_a, config) != _occlusion_signature(boxes_b, config)
    if reason == "region_transition":
        return _cell_signature(boxes_a, config) != _cell_signature(boxes_b, config)
    if reason == "bbox_changed":
        for box_a, box_b, overlap in _best_pairs(boxes_a, boxes_b):
            area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
            area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
            if min(area_a, area_b) <= 0:
                continue
            ratio = max(area_a, area_b) / min(area_a, area_b)
            if overlap < config.bbox_iou or ratio >= config.bbox_area_ratio:
                return True
        return False
    if reason == "person_moved":
        for box_a, box_b, _ in _best_pairs(boxes_a, boxes_b):
            ax = (box_a[0] + box_a[2]) / 2 - (box_b[0] + box_b[2]) / 2
            ay = (box_a[1] + box_a[3]) / 2 - (box_b[1] + box_b[3]) / 2
            width, height = box_b[2] - box_b[0], box_b[3] - box_b[1]
            diagonal = (width * width + height * height) ** 0.5
            if diagonal > 0 and (ax * ax + ay * ay) ** 0.5 / diagonal >= config.move_fraction:
                return True
        return False
    return None


def audit(
    kept: list[dict], config: SamplingConfig, *, reference: str = "reference"
) -> dict:
    """Score every retained frame that resembles an earlier one.

    Frames with no perceptual match are not evidence either way and are counted
    separately rather than treated as passes — a claim that was never put to the
    test has not survived one.
    """
    contract = consequences_for(config)
    report: dict[str, dict] = {}
    for reason, (claim, testability) in contract.items():
        report[reason] = {
            "claim": claim,
            "testability": testability.value,
            "frames": 0,
            "compared": 0,
            "no_match": 0,
            "held": 0,
            "failed": 0,
            "rate": None,
        }

    for entry in kept:
        reason = entry["reason"]
        cell = report.setdefault(
            reason,
            {
                "claim": "unknown reason",
                "testability": Testability.UNTESTABLE.value,
                "frames": 0,
                "compared": 0,
                "no_match": 0,
                "held": 0,
                "failed": 0,
                "rate": None,
            },
        )
        cell["frames"] += 1

        # The reference is attached by `trace.replay` and comes from the RAW
        # trace, not from the retained set, so it does not move when the policy
        # does. Frames without one were never put to the test.
        match = entry.get(reference)

        if cell["testability"] != Testability.TESTABLE.value:
            continue
        if match is None:
            cell["no_match"] += 1
            continue
        verdict = _holds(reason, entry, match, config)
        if verdict is None:
            cell["no_match"] += 1
            continue
        cell["compared"] += 1
        if verdict:
            cell["held"] += 1
        else:
            cell["failed"] += 1

    for cell in report.values():
        if cell["testability"] == Testability.TESTABLE.value and cell["compared"]:
            cell["rate"] = round(cell["held"] / cell["compared"], 4)

    testable = [c for c in report.values() if c["testability"] == Testability.TESTABLE.value]
    compared = sum(c["compared"] for c in testable)
    held = sum(c["held"] for c in testable)
    return {
        "_comment": [
            "Each event is scored ONLY against the consequence it actually claims,",
            "and only where that consequence is derivable from the evidence.",
            "TAUTOLOGICAL rows fire by construction and are never credited;",
            "UNTESTABLE rows carry no rate. A frame with no perceptual match was",
            "never put to the test and is counted apart rather than passed.",
            "The reference is an earlier RAW-TRACE frame that resembles this one",
            "— never another retained frame, which would make the baseline move",
            "with the policy under test and stop the rates being comparable.",
            "person_left's claim depends on the departure rule: ON_EXPIRY keeps",
            "the frame after the person is gone and predicts a LOWER count;",
            "LAST_CONFIRMED keeps their final sighting and predicts a NON-EMPTY",
            "frame. Scoring the second with the first's contract reports a",
            "collapse that belongs entirely to the auditor.",
        ],
        "departure_rule": config.departure_rule.value,
        "reference": reference,
        "by_reason": report,
        "overall_testable": {
            "compared": compared,
            "held": held,
            "rate": round(held / compared, 4) if compared else None,
        },
    }
