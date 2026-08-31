"""Event-aware sampling — P9.6 Phases 2–6.

A deterministic state machine that decides, frame by frame, whether a frame
carries information the corpus does not already have. It holds no camera, opens
no file and imports nothing from production: perception enters through two
injected callables, which is what makes it testable without a DVR.

### What it detects, and what it must never detect

It detects **change**. It does not detect **violations**, and the distinction is
the reason this module exists rather than a simpler one.

A sampler triggered by a hairnet classifier — or by a VLM, or by any PPE signal —
builds a corpus around what the current model already finds interesting. Every
distribution computed from that corpus afterwards describes the model's beliefs,
and a benchmark built on it scores the model against its own priors. So the
trigger vocabulary here is exhausted by geometry and pixels:

* pixels changed (`SCENE_CHANGED`)
* the number of people changed (`PERSON_ENTERED`, `PERSON_LEFT`, `PERSON_COUNT_CHANGED`)
* a person moved, resized, crossed the frame, or started/stopped overlapping
  something (`PERSON_MOVED`, `BBOX_CHANGED`, `REGION_TRANSITION`, `OCCLUSION_CHANGED`)
* nothing happened for a long time (`PERIODIC_HEARTBEAT`)

The person detector is a *person* detector. It is asked "is someone there and
where", never "is that person compliant", and no covering signal reaches this
file.

### The cascade: deterministic signal first, model second

Phase 1 measured that 48 % of consecutive sampled frames differed by ≤ 1 bit of
64 — the kitchen is motionless for long stretches. Running a neural detector on
every decoded frame of four 15–25 fps streams to discover that is waste, so a
64-bit difference hash gates it: the detector runs on a rate limit, or early when
the hash says something moved. Frames between detections can still be accepted
for a scene change, but they are not credited with person events, because no
observation of people was made and inventing one would be a lie about provenance.

### Why each camera finds its own rate

Phase 1 also measured that cam-12's median consecutive distance is 7 while
cam-14's is 0. A single global interval must be simultaneously too slow for one
and far too fast for the other. Nothing here sets a rate: the rate is whatever
the scene produces, so a busy camera samples often and a frozen one falls back to
the heartbeat.

### The heartbeat measures from the last *accepted* sample

Deliberately, and it gives the baseline exactly the semantics Phase 6 asks for:
in a busy scene the heartbeat never fires, because event sampling already has
coverage; in a still scene, or one where event detection has broken, it is the
only thing firing. Its share of the corpus is therefore a direct readout of how
much of the session the event layer failed to explain, and it is reported
separately for that reason.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

Box = tuple[float, float, float, float]

#: Bumped whenever a change alters which frames would be accepted. Recorded in
#: every session manifest, so a corpus can always be traced to the policy that
#: produced it.
#: `-2b` adopts `DepartureRule.LAST_CONFIRMED`, selected by the Phase 2 replay
#: over 8 traces / 26,510 observations: it removed all 149 person-free
#: `PERSON_LEFT` frames while keeping candidate subject tracks at 1,002 — an
#: identical coverage figure, which is what makes it an improvement rather than
#: a trade. `-2c` was rejected for buying one further point of person-free rate
#: with a 40 % loss of tracks.
SAMPLING_CONFIG_VERSION = "p9.6-events-2b"


class SamplingReason(enum.Enum):
    """Why a frame was kept. Every retained sample carries exactly one.

    Ordered by `priority`: rarer and more informative events outrank common ones,
    so when several fire at once the frame is attributed to the most specific
    thing that happened rather than to whichever check ran first.
    """

    MANUAL_REVIEW = "manual_review"
    """Requested by a person. Never suppressed."""

    PERSON_ENTERED = "person_entered"
    """A person box appeared with no predecessor. The rarest and most valuable
    moment on a static camera."""

    PERSON_LEFT = "person_left"
    """A tracked person box vanished."""

    PERSON_COUNT_CHANGED = "person_count_changed"
    """The population changed size without a clean entry or exit — typically a
    person becoming separable from, or merging into, a group."""

    OCCLUSION_CHANGED = "occlusion_changed"
    """A **geometric proxy**: a person's box began or stopped overlapping another
    person's box, or began or stopped touching the frame edge. Real occlusion is
    not measurable from boxes; this catches the transitions that produce the hard
    cases, and it is named for what it approximates, not for what it proves."""

    REGION_TRANSITION = "region_transition"
    """A person's centre crossed into a different cell of a coarse grid — new
    lighting, new distance, new camera geometry."""

    BBOX_CHANGED = "bbox_changed"
    """A matched box changed shape or scale materially: approach, recession,
    crouching, truncation by the frame."""

    PERSON_MOVED = "person_moved"
    """A matched box translated by a material fraction of its own size."""

    SCENE_CHANGED = "scene_changed"
    """Global pixel change with no person event to explain it — equipment, doors,
    lighting, trolleys, or a person the detector did not find."""

    PERIODIC_HEARTBEAT = "periodic_heartbeat"
    """The low-rate baseline. Nothing happened for a long time, or nothing was
    detected happening, and both of those are conditions the corpus must be able
    to represent."""

    @property
    def priority(self) -> int:
        return _PRIORITY[self]

    @property
    def is_person_event(self) -> bool:
        return self in _PERSON_EVENTS


_PRIORITY = {
    SamplingReason.MANUAL_REVIEW: 100,
    SamplingReason.PERSON_ENTERED: 90,
    SamplingReason.PERSON_LEFT: 85,
    SamplingReason.PERSON_COUNT_CHANGED: 80,
    SamplingReason.OCCLUSION_CHANGED: 70,
    SamplingReason.REGION_TRANSITION: 60,
    SamplingReason.BBOX_CHANGED: 50,
    SamplingReason.PERSON_MOVED: 40,
    SamplingReason.SCENE_CHANGED: 30,
    SamplingReason.PERIODIC_HEARTBEAT: 10,
}

_PERSON_EVENTS = frozenset(
    {
        SamplingReason.PERSON_ENTERED,
        SamplingReason.PERSON_LEFT,
        SamplingReason.PERSON_COUNT_CHANGED,
        SamplingReason.OCCLUSION_CHANGED,
        SamplingReason.REGION_TRANSITION,
        SamplingReason.BBOX_CHANGED,
        SamplingReason.PERSON_MOVED,
    }
)

#: Priority at or above which an event may override the ordinary cooldown, down
#: to `hard_floor_seconds`. A person walking into frame is precisely the instant
#: a cooldown must not swallow.
PRIORITY_OVERRIDES_COOLDOWN = _PRIORITY[SamplingReason.OCCLUSION_CHANGED]


class SampleClass(enum.Enum):
    """What kind of evidence a kept frame is. **Never mixed in a metric.**

    Phase 1 reported one efficiency number over both classes and it was
    misleading in both directions: the baseline dragged the person-positive rate
    down while the event frames hid how little of the session the baseline was
    actually covering. They answer different questions and are scored apart.
    """

    EVENT = "event"
    """Something changed. Judged on person-positive rate, duplication and
    corroboration — this is the pool an annotator draws from."""

    BASELINE = "baseline"
    """Nothing changed, or nothing was detected changing. Judged on temporal
    coverage and camera health. **It is not supposed to contain people**, and
    scoring it on person yield punishes it for doing its job."""

    @property
    def is_event(self) -> bool:
        return self is SampleClass.EVENT


class DepartureRule(enum.Enum):
    """Which frame a `PERSON_LEFT` event should keep.

    The defect P9.6 Phase 1 measured. `PERSON_LEFT` supplied 151 of the 323
    person-free retained frames — 65.9 % of everything that event kept — because
    it fires when a track *expires*, and by then the person has been gone for
    `track_max_age` detections.
    """

    ON_EXPIRY = "on_expiry"
    """Phase 1 behaviour: keep the frame in which the absence was admitted. Kept
    executable so the A/B is against the policy that actually ran, not a
    reconstruction of it."""

    LAST_CONFIRMED = "last_confirmed"
    """Keep the last frame in which the track was actually confirmed — the
    departure *evidence* rather than its aftermath. Requires the caller to be
    able to produce an earlier frame; a caller that cannot falls back to
    `ON_EXPIRY` and the decision records that it did, because silently
    substituting a different frame would corrupt the provenance."""


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Every threshold the sampler consults. Recorded with the corpus.

    Defaults are the values Phase 4's sweep selected on recorded production
    footage; they are a reading, not a preference, and `sweep.py` reproduces the
    comparison that chose them.
    """

    version: str = SAMPLING_CONFIG_VERSION

    scene_change_bits: int = 8
    """Hamming distance from the last accepted frame that counts as a scene
    change. Phase 1 measured the consecutive-distance distribution: median 2,
    p75 6, p90 11. Eight sits above the noise floor of every camera (cam-14's
    p90 is 3) while remaining below cam-12's median of 7 plus its spread."""

    move_fraction: float = 0.15
    """Centre displacement as a fraction of the box diagonal."""

    bbox_iou: float = 0.70
    """IoU with the matched predecessor below which the box materially changed."""

    bbox_area_ratio: float = 1.25
    """Area ratio outside [1/r, r] counts as a material scale change."""

    overlap_iou: float = 0.10
    """Person-person IoU at which the occlusion proxy turns **on**."""

    overlap_release_iou: float = 0.05
    """Person-person IoU at which it turns **off** again.

    A Schmitt trigger, not a threshold. Two people working side by side sit near
    the boundary and a single threshold chatters across it every detection,
    producing a stream of `OCCLUSION_CHANGED` on a scene that is not changing.
    Releasing lower than it engages costs nothing and removes the oscillation."""

    edge_epsilon: float = 0.01
    """Normalised distance from a frame border that counts as contact."""

    grid: int = 3
    """Region grid per axis for `REGION_TRANSITION`."""

    match_iou: float = 0.30
    """IoU below which two boxes are not the same track. Governs entry/exit."""

    match_distance_diagonals: float = 1.20
    """Fallback association gate, in box diagonals.

    Pure IoU association breaks on a walking person: at a 0.5 s detection
    interval someone at 1.4 m/s covers ~0.7 m, more than a body's width, so
    consecutive boxes can fail to overlap **at all**. Without this gate that
    reads as one person leaving and another arriving, which does not change
    whether the frame is kept but does corrupt the event distribution the
    report publishes.

    Set to 0 to disable the fallback and use IoU alone."""

    match_area_ratio: float = 2.0
    """Area agreement required by the fallback. Two boxes of wildly different
    size near each other are not the same person moving."""

    track_min_hits: int = 2
    """Detections a new box must survive before it counts as a person arriving.

    Without this the sampler reports the detector's stutter as human traffic.
    Measured on 226 s of recorded production footage, a naive version fired 65
    `PERSON_ENTERED` and 64 `PERSON_LEFT` events — nobody entered 65 times. The
    detector was dropping a box for a frame and re-acquiring it, and because both
    events are high priority they punched straight through the cooldown and
    resampled an unchanged scene."""

    track_max_age: int = 2
    """Detections a track may go unmatched before it counts as having left.

    The other half of the hysteresis. A person briefly occluded by equipment has
    not left the kitchen, and saying they did costs a spurious exit now and a
    spurious entry moments later."""

    min_gap_seconds: float = 2.0
    """Ordinary cooldown between accepted samples."""

    hard_floor_seconds: float = 0.5
    """The floor a high-priority event may override the cooldown down to."""

    heartbeat_seconds: float = 45.0
    """Low-rate baseline, measured from the last accepted sample."""

    detect_every_seconds: float = 0.5
    """Rate limit for the person detector. A scene change runs it early.

    The sweep's most consequential choice, and it was not made on the duplicate
    rate. Coarsening to 2 s gives the best numbers on paper — 26.8 % duplicates
    against 40.0 % at 0.5 s — but it sees **27 candidate subject tracks instead
    of 39**. It buys the duplicate rate by not seeing people.

    Those two costs are not symmetric. Redundancy is recoverable: deduplication
    removes it afterwards, and the frames are on disk either way. A person who
    was never sampled cannot be annotated, ever. So coverage wins.

    Going finer than 0.5 s does not help: 0.2 s reports 44 tracks but its
    rescue corroboration collapses to 60 % against 79 %, which says the extra
    tracks are one person fragmenting rather than more people."""

    max_per_reason: int = 10
    """Per camera-session cap, so one event type cannot monopolise a corpus.

    Chosen for event **diversity**, which Phase 18 audits. Tightening 40 → 10
    cost 6 frames and 8 candidates out of ~250 and raised the number of distinct
    event types represented from 7 to 8, while improving rescue corroboration
    from 79.0 % to 82.3 %. Cheap.

    Note the interaction with session length: the cap is absolute, so a longer
    window does not produce proportionally more frames. Coverage is therefore
    bought with **more sessions**, not longer ones — which is also what the split
    needs, since the session is the group a split may not straddle."""

    max_samples: int = 150
    """Per camera-session cap on accepted frames."""

    departure_rule: DepartureRule = DepartureRule.LAST_CONFIRMED
    """Which frame `PERSON_LEFT` keeps. See `DepartureRule`."""

    departure_confirm_misses: int = 0
    """Extra missed detections required before a departure is believed at all.

    Zero reproduces Phase 1: a track expires at `track_max_age` and that is
    treated as a departure. Raising it separates *"the detector stopped seeing
    them"* from *"they left"*, at the cost of reporting real departures later.
    Both costs are measured in the replay rather than argued about."""

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "scene_change_bits": self.scene_change_bits,
            "move_fraction": self.move_fraction,
            "bbox_iou": self.bbox_iou,
            "bbox_area_ratio": self.bbox_area_ratio,
            "overlap_iou": self.overlap_iou,
            "overlap_release_iou": self.overlap_release_iou,
            "edge_epsilon": self.edge_epsilon,
            "grid": self.grid,
            "match_iou": self.match_iou,
            "match_distance_diagonals": self.match_distance_diagonals,
            "match_area_ratio": self.match_area_ratio,
            "track_min_hits": self.track_min_hits,
            "track_max_age": self.track_max_age,
            "min_gap_seconds": self.min_gap_seconds,
            "hard_floor_seconds": self.hard_floor_seconds,
            "heartbeat_seconds": self.heartbeat_seconds,
            "detect_every_seconds": self.detect_every_seconds,
            "max_per_reason": self.max_per_reason,
            "max_samples": self.max_samples,
            "departure_rule": self.departure_rule.value,
            "departure_confirm_misses": self.departure_confirm_misses,
        }


@dataclass(frozen=True, slots=True)
class Track:
    """A person box with a temporary id.

    `track_id` is an **engineering aid, not an identity**. It is stable only
    while IoU matching succeeds within one camera-session, and it survives no
    occlusion, no exit and no session boundary. Nothing downstream may promote it
    to a subject identity: P9's `identity_verified` stays False for every sample
    this module produces, which collapses the session to a single split group and
    makes the thinness visible rather than assumed away.
    """

    track_id: int
    box: Box
    confidence: float = 0.0

    hits: int = 1
    """Detections this track has been matched in. Governs confirmation."""

    misses: int = 0
    """Consecutive detections this track went unmatched. Governs expiry."""

    confirmed: bool = False
    """Whether the track has survived `track_min_hits`. Only confirmed tracks
    are counted as people or generate events: an unconfirmed track is a
    detection that has not yet distinguished itself from a flicker."""

    last_seen_index: int = -1
    """Frame index of the most recent detection that actually matched this
    track. `DepartureRule.LAST_CONFIRMED` keeps this frame rather than the one
    where the absence was finally admitted — the person is still visible in it,
    which is the entire point."""

    last_seen_timestamp: float = 0.0

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])

    @property
    def diagonal(self) -> float:
        width = self.box[2] - self.box[0]
        height = self.box[3] - self.box[1]
        return (width * width + height * height) ** 0.5


@dataclass(frozen=True, slots=True)
class Decision:
    """The verdict on one frame — returned whether or not it was accepted.

    A rejected frame carries `suppressed_by`, so Phase 4 can measure what the
    cooldown and the caps actually cost instead of guessing at it.
    """

    frame_index: int
    timestamp: float
    accepted: bool
    reason: SamplingReason | None = None
    reasons: tuple[SamplingReason, ...] = ()
    people: int = 0
    tracks: tuple[Track, ...] = ()
    scene_distance: int = 0
    detected: bool = False
    suppressed_by: str = ""
    detail: str = ""

    sample_class: SampleClass = SampleClass.EVENT
    """Event evidence or periodic baseline. Reported apart, never averaged."""

    capture_frame_index: int = -1
    """The frame the caller should actually keep, when that is **not** the frame
    being offered. Set only by `DepartureRule.LAST_CONFIRMED`, and only to a
    frame the sampler has already seen. `-1` means "keep the offered frame"."""

    capture_fallback: str = ""
    """Set when a retrospective capture was wanted and could not be honoured.
    Recorded rather than silently ignored: a frame kept under a different rule
    from the one the policy names is a provenance error."""


def iou(a: Box, b: Box) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_tracks(
    previous: Sequence[Track],
    boxes: Sequence[tuple[Box, float]],
    *,
    threshold: float,
    next_id: int,
    distance_diagonals: float = 0.0,
    area_ratio: float = 2.0,
) -> tuple[list[Track], list[tuple[Track, Track]], list[Track], int]:
    """Greedy association, IoU first and a gated centre-distance fallback second.

    Returns `(current, matched_pairs, lost, next_id)`. Deterministic: ties break
    on index order, and the fallback pass only ever considers boxes the IoU pass
    left unclaimed.

    This is association, not tracking. It has no motion model, no appearance
    model and no re-identification, and it is not trying to acquire any: it
    answers "is this plausibly the same box as a moment ago", which is all the
    event vocabulary needs. When it guesses wrong the cost is a mislabelled
    *reason* on a frame that was going to be kept regardless — never a
    mislabelled sample, because `track_id` is not an identity and nothing
    downstream may treat it as one.
    """
    used_previous: set[int] = set()
    used_box: set[int] = set()
    assignment: dict[int, int] = {}

    def commit(scored: list[tuple[float, int, int]]) -> None:
        for _, index_previous, index_box in scored:
            if index_previous in used_previous or index_box in used_box:
                continue
            used_previous.add(index_previous)
            used_box.add(index_box)
            assignment[index_box] = index_previous

    overlaps: list[tuple[float, int, int]] = []
    for index_previous, track in enumerate(previous):
        for index_box, (box, _) in enumerate(boxes):
            score = iou(track.box, box)
            if score >= threshold:
                overlaps.append((-score, index_previous, index_box))
    overlaps.sort()
    commit(overlaps)

    if distance_diagonals > 0:
        nearby: list[tuple[float, int, int]] = []
        for index_previous, track in enumerate(previous):
            if index_previous in used_previous or track.diagonal <= 0:
                continue
            for index_box, (box, _) in enumerate(boxes):
                if index_box in used_box:
                    continue
                candidate = Track(track_id=-1, box=box)
                if candidate.area <= 0 or track.area <= 0:
                    continue
                ratio = max(candidate.area, track.area) / min(candidate.area, track.area)
                if ratio > area_ratio:
                    continue
                bx, by = track.centre
                cx, cy = candidate.centre
                separation = ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5 / track.diagonal
                if separation <= distance_diagonals:
                    nearby.append((separation, index_previous, index_box))
        nearby.sort()
        commit(nearby)

    current: list[Track] = []
    matched: list[tuple[Track, Track]] = []
    for index_box, (box, score) in enumerate(boxes):
        if index_box in assignment:
            before = previous[assignment[index_box]]
            after = Track(track_id=before.track_id, box=box, confidence=score)
            matched.append((before, after))
        else:
            after = Track(track_id=next_id, box=box, confidence=score)
            next_id += 1
        current.append(after)

    lost = [t for i, t in enumerate(previous) if i not in used_previous]
    return current, matched, lost, next_id


def advance_tracks(
    previous: Sequence[Track],
    boxes: Sequence[tuple[Box, float]],
    *,
    config: SamplingConfig,
    next_id: int,
    frame_index: int = -1,
    timestamp: float = 0.0,
) -> tuple[list[Track], list[tuple[Track, Track]], list[Track], list[Track], int]:
    """Associate, then age. Returns `(live, matched, confirmed_new, expired, next_id)`.

    The lifecycle is the whole point. `match_tracks` answers "which box is which";
    this answers "did a person actually arrive or leave", and those are different
    questions — the second one needs evidence over time, because a single-frame
    detector gap is indistinguishable from a departure until the next frame
    disagrees.

    A track that goes unmatched is **coasted**, keeping its last known box, for up
    to `track_max_age` detections. A new box is **provisional** until it has been
    matched `track_min_hits` times. Only the transitions across those two
    boundaries produce `PERSON_ENTERED` and `PERSON_LEFT`.
    """
    associated, matched, unmatched, next_id = match_tracks(
        previous,
        boxes,
        threshold=config.match_iou,
        next_id=next_id,
        distance_diagonals=config.match_distance_diagonals,
        area_ratio=config.match_area_ratio,
    )
    before_of = {after.track_id: before for before, after in matched}

    live: list[Track] = []
    confirmed_new: list[Track] = []
    aged_pairs: list[tuple[Track, Track]] = []

    for track in associated:
        before = before_of.get(track.track_id)
        if before is None:
            hits, was_confirmed = 1, False
        else:
            hits, was_confirmed = before.hits + 1, before.confirmed
        confirmed = was_confirmed or hits >= config.track_min_hits
        aged = Track(
            track_id=track.track_id,
            box=track.box,
            confidence=track.confidence,
            hits=hits,
            misses=0,
            confirmed=confirmed,
            last_seen_index=frame_index,
            last_seen_timestamp=timestamp,
        )
        live.append(aged)
        if confirmed and not was_confirmed:
            confirmed_new.append(aged)
        if before is not None and was_confirmed and confirmed:
            aged_pairs.append((before, aged))

    # A track is only believed to have departed after `track_max_age` misses
    # plus `departure_confirm_misses` more. The second term separates "the
    # detector stopped seeing them" from "they left": a person behind a passing
    # trolley is not a departure, and Phase 1 had no way to say so.
    expiry = config.track_max_age + config.departure_confirm_misses
    expired: list[Track] = []
    for track in unmatched:
        coasted = Track(
            track_id=track.track_id,
            box=track.box,
            confidence=track.confidence,
            hits=track.hits,
            misses=track.misses + 1,
            confirmed=track.confirmed,
            last_seen_index=track.last_seen_index,
            last_seen_timestamp=track.last_seen_timestamp,
        )
        if coasted.misses > expiry:
            if coasted.confirmed:
                expired.append(coasted)
        else:
            live.append(coasted)

    live.sort(key=lambda t: t.track_id)
    return live, aged_pairs, confirmed_new, expired, next_id


def _cell(track: Track, grid: int) -> tuple[int, int]:
    cx, cy = track.centre
    return (
        min(grid - 1, max(0, int(cx * grid))),
        min(grid - 1, max(0, int(cy * grid))),
    )


def _touches_edge(track: Track, epsilon: float) -> bool:
    x1, y1, x2, y2 = track.box
    return x1 <= epsilon or y1 <= epsilon or x2 >= 1 - epsilon or y2 >= 1 - epsilon


def _occlusion_state(
    track: Track,
    everyone: Sequence[Track],
    config: SamplingConfig,
    *,
    was: bool | None = None,
) -> bool:
    """The geometric proxy: overlapping another person, or cut by the frame.

    Hysteretic. `was` is the previous state; the engage threshold applies when
    turning on and the lower release threshold when turning off, so a pair of
    workers hovering at the boundary does not emit an event per detection.
    """
    if _touches_edge(track, config.edge_epsilon):
        return True
    threshold = (
        config.overlap_release_iou if was else config.overlap_iou
    )
    return any(
        other.track_id != track.track_id and iou(track.box, other.box) >= threshold
        for other in everyone
    )


class EventSampler:
    """One camera, one session. Deterministic given the same frame sequence.

    Perception is injected:

    * `hash_of(image) -> int` — a 64-bit perceptual hash
    * `detect(image) -> list[(box, score)]` — normalised person boxes

    Both are cheap to fake, so every rule below is tested without a camera, a
    model file or a network. A test that needs a DVR is a test that stops running.
    """

    __slots__ = (
        "config",
        "_hash_of",
        "_detect",
        "_last_hash",
        "_last_accepted_at",
        "_last_detect_at",
        "_tracks",
        "_occluded",
        "_cells",
        "_next_id",
        "_accepted",
        "_by_reason",
        "_suppressed",
        "_seen",
        "_subject_tracks",
        "_departed",
        "_by_class",
        "_fallbacks",
    )

    def __init__(
        self,
        config: SamplingConfig | None = None,
        *,
        hash_of: Callable[[object], int],
        detect: Callable[[object], Sequence[tuple[Box, float]]],
    ) -> None:
        self.config = config or SamplingConfig()
        self._hash_of = hash_of
        self._detect = detect
        self._last_hash: int | None = None
        self._last_accepted_at: float | None = None
        self._last_detect_at: float | None = None
        self._tracks: tuple[Track, ...] = ()
        self._occluded: dict[int, bool] = {}
        self._cells: dict[int, tuple[int, int]] = {}
        self._next_id = 0
        self._accepted = 0
        self._seen = 0
        self._by_reason: dict[SamplingReason, int] = {}
        self._suppressed: dict[str, int] = {}
        self._subject_tracks: set[int] = set()
        self._departed = -1
        self._by_class: dict[SampleClass, int] = {}
        self._fallbacks = 0

    # -- reporting ---------------------------------------------------------

    @property
    def accepted(self) -> int:
        return self._accepted

    @property
    def seen(self) -> int:
        return self._seen

    def statistics(self) -> dict:
        return {
            "sampling_config_version": self.config.version,
            "frames_offered": self._seen,
            "frames_accepted": self._accepted,
            "acceptance_rate": (
                round(self._accepted / self._seen, 5) if self._seen else None
            ),
            "by_reason": {r.value: n for r, n in sorted(
                self._by_reason.items(), key=lambda kv: -kv[0].priority
            )},
            "event_triggered": sum(
                n for r, n in self._by_reason.items()
                if r is not SamplingReason.PERIODIC_HEARTBEAT
            ),
            # Person-driven vs pixel-driven, reported apart. A session carried
            # mostly by SCENE_CHANGED saw movement it could not attribute to
            # anybody — equipment, lighting, or a person the detector missed —
            # and that distinction matters more than the total.
            "person_event_triggered": sum(
                n for r, n in self._by_reason.items() if r.is_person_event
            ),
            "scene_event_triggered": self._by_reason.get(
                SamplingReason.SCENE_CHANGED, 0
            ),
            "baseline_triggered": self._by_reason.get(SamplingReason.PERIODIC_HEARTBEAT, 0),
            "suppressed": dict(sorted(self._suppressed.items())),
            "candidate_subject_tracks": len(self._subject_tracks),
            # Reported apart, never averaged. A baseline frame is not supposed
            # to contain a person, so folding it into a person-positive rate
            # penalises it for working correctly.
            "by_class": {c.value: n for c, n in sorted(
                self._by_class.items(), key=lambda kv: kv[0].value
            )},
            "departure_rule": self.config.departure_rule.value,
            "retrospective_fallbacks": self._fallbacks,
        }

    # -- the decision ------------------------------------------------------

    def offer(self, frame_index: int, timestamp: float, image: object) -> Decision:
        """Consider one frame. Returns a `Decision` either way."""
        self._seen += 1
        config = self.config
        bits = self._hash_of(image)
        opening = self._last_hash is None
        distance = 64 if opening else _hamming(bits, self._last_hash)

        run_detector = (
            self._last_detect_at is None
            or timestamp - self._last_detect_at >= config.detect_every_seconds
            or distance >= config.scene_change_bits
        )

        reasons: list[SamplingReason] = []
        details: list[str] = []
        tracks = self._tracks
        self._departed = -1

        if run_detector:
            boxes = list(self._detect(image))
            tracks, matched, arrived, left, self._next_id = advance_tracks(
                self._tracks,
                boxes,
                config=config,
                next_id=self._next_id,
                frame_index=frame_index,
                timestamp=timestamp,
            )
            self._subject_tracks.update(t.track_id for t in tracks if t.confirmed)
            if self._last_detect_at is not None:
                reasons.extend(self._person_events(matched, arrived, left, tracks, details))
            self._last_detect_at = timestamp

        # Not on the opening frame: there is no earlier frame for the scene to
        # have changed *from*, and claiming a 64-bit change against nothing would
        # mislabel every session's first sample.
        if not opening and distance >= config.scene_change_bits:
            reasons.append(SamplingReason.SCENE_CHANGED)
            details.append(f"scene distance {distance}")

        elapsed = None if self._last_accepted_at is None else timestamp - self._last_accepted_at
        if elapsed is None or elapsed >= config.heartbeat_seconds:
            reasons.append(SamplingReason.PERIODIC_HEARTBEAT)
            details.append(
                "session opener" if elapsed is None else f"{elapsed:.0f}s since last sample"
            )

        decision = self._resolve(
            frame_index=frame_index,
            timestamp=timestamp,
            reasons=reasons,
            details=details,
            tracks=tracks,
            distance=distance,
            detected=run_detector,
            elapsed=elapsed,
        )

        self._tracks = tuple(tracks)
        confirmed = [t for t in tracks if t.confirmed]
        self._occluded = {
            t.track_id: _occlusion_state(
                t, confirmed, config, was=self._occluded.get(t.track_id)
            )
            for t in confirmed
        }
        self._cells = {t.track_id: _cell(t, config.grid) for t in confirmed}
        if decision.accepted:
            self._last_hash = bits
            self._last_accepted_at = timestamp
        return decision

    def offer_manual(self, frame_index: int, timestamp: float, note: str) -> Decision:
        """Accept a frame a person asked for. Bypasses cooldown and every cap."""
        self._accepted += 1
        self._by_reason[SamplingReason.MANUAL_REVIEW] = (
            self._by_reason.get(SamplingReason.MANUAL_REVIEW, 0) + 1
        )
        self._last_accepted_at = timestamp
        return Decision(
            frame_index=frame_index,
            timestamp=timestamp,
            accepted=True,
            reason=SamplingReason.MANUAL_REVIEW,
            reasons=(SamplingReason.MANUAL_REVIEW,),
            sample_class=SampleClass.EVENT,
            people=len(self._tracks),
            tracks=self._tracks,
            detail=note,
        )

    # -- internals ---------------------------------------------------------

    def _person_events(
        self,
        matched: Sequence[tuple[Track, Track]],
        arrived: Sequence[Track],
        left: Sequence[Track],
        tracks: Sequence[Track],
        details: list[str],
    ) -> Iterable[SamplingReason]:
        config = self.config
        found: list[SamplingReason] = []
        confirmed = [t for t in tracks if t.confirmed]
        was = sum(1 for t in self._tracks if t.confirmed)

        if arrived:
            found.append(SamplingReason.PERSON_ENTERED)
            details.append(f"{len(arrived)} entered")
        if left:
            found.append(SamplingReason.PERSON_LEFT)
            details.append(f"{len(left)} left")
            # The most recent sighting among the departing tracks. Under
            # LAST_CONFIRMED this is the frame worth keeping: the person is
            # still in it.
            self._departed = max(
                (t.last_seen_index for t in left if t.last_seen_index >= 0),
                default=-1,
            )
        if len(confirmed) != was:
            found.append(SamplingReason.PERSON_COUNT_CHANGED)
            details.append(f"{was} -> {len(confirmed)} people")

        for before, after in matched:
            was_occluded = self._occluded.get(before.track_id)
            now_occluded = _occlusion_state(after, confirmed, config, was=was_occluded)
            if was_occluded is not None and was_occluded != now_occluded:
                found.append(SamplingReason.OCCLUSION_CHANGED)
                details.append(
                    f"track {after.track_id} "
                    f"{'entered' if now_occluded else 'left'} overlap/edge"
                )

            bx, by = before.centre
            ax, ay = after.centre
            displacement = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            travelled = displacement / before.diagonal if before.diagonal > 0 else 0.0

            was_cell = self._cells.get(before.track_id)
            now_cell = _cell(after, config.grid)
            # A boundary crossing AND real travel. A person standing on a cell
            # edge jitters across it every detection; without the magnitude
            # requirement that reads as repeated traversal of the kitchen.
            if (
                was_cell is not None
                and was_cell != now_cell
                and travelled >= config.move_fraction
            ):
                found.append(SamplingReason.REGION_TRANSITION)
                details.append(f"track {after.track_id} {was_cell} -> {now_cell}")

            overlap = iou(before.box, after.box)
            area_before, area_after = before.area, after.area
            ratio = (
                max(area_before, area_after) / min(area_before, area_after)
                if min(area_before, area_after) > 0
                else float("inf")
            )
            if overlap < config.bbox_iou or ratio >= config.bbox_area_ratio:
                found.append(SamplingReason.BBOX_CHANGED)
                details.append(f"track {after.track_id} iou {overlap:.2f} ratio {ratio:.2f}")

            if travelled >= config.move_fraction:
                found.append(SamplingReason.PERSON_MOVED)
                details.append(f"track {after.track_id} moved {travelled:.2f} diag")

        return found

    def _resolve(
        self,
        *,
        frame_index: int,
        timestamp: float,
        reasons: list[SamplingReason],
        details: list[str],
        tracks: Sequence[Track],
        distance: int,
        detected: bool,
        elapsed: float | None,
    ) -> Decision:
        config = self.config
        unique = tuple(sorted(set(reasons), key=lambda r: -r.priority))
        # Confirmed tracks only. A provisional track is a detection that has not
        # yet distinguished itself from a flicker, and counting it as a person
        # would put the flicker into the published statistics.
        settled = tuple(t for t in tracks if t.confirmed)
        base = {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "reasons": unique,
            "people": len(settled),
            "tracks": settled,
            "scene_distance": distance,
            "detected": detected,
            "detail": "; ".join(details),
        }

        if not unique:
            return Decision(accepted=False, suppressed_by="no_event", **base)

        if self._accepted >= config.max_samples:
            self._suppressed["session_cap"] = self._suppressed.get("session_cap", 0) + 1
            return Decision(accepted=False, suppressed_by="session_cap", **base)

        # The highest-priority reason that still has budget. Falling through to a
        # lesser reason rather than dropping the frame is deliberate: the cap
        # exists to stop one event type monopolising the corpus, not to discard a
        # frame that some other event also justifies.
        chosen: SamplingReason | None = None
        for reason in unique:
            if self._by_reason.get(reason, 0) < config.max_per_reason:
                chosen = reason
                break
        if chosen is None:
            self._suppressed["reason_cap"] = self._suppressed.get("reason_cap", 0) + 1
            return Decision(accepted=False, suppressed_by="reason_cap", **base)

        if elapsed is not None:
            floor = (
                config.hard_floor_seconds
                if chosen.priority >= PRIORITY_OVERRIDES_COOLDOWN
                else config.min_gap_seconds
            )
            if elapsed < floor:
                self._suppressed["cooldown"] = self._suppressed.get("cooldown", 0) + 1
                return Decision(accepted=False, suppressed_by="cooldown", **base)

        self._accepted += 1
        self._by_reason[chosen] = self._by_reason.get(chosen, 0) + 1

        sample_class = (
            SampleClass.BASELINE
            if chosen is SamplingReason.PERIODIC_HEARTBEAT
            else SampleClass.EVENT
        )
        self._by_class[sample_class] = self._by_class.get(sample_class, 0) + 1

        # Retrospective capture. Under LAST_CONFIRMED a departure keeps the last
        # frame the person was actually in, not the frame in which their absence
        # was finally admitted — the defect that supplied 151 of Phase 1's 323
        # person-free frames.
        capture = -1
        fallback = ""
        if (
            chosen is SamplingReason.PERSON_LEFT
            and config.departure_rule is DepartureRule.LAST_CONFIRMED
        ):
            if self._departed >= 0 and self._departed != frame_index:
                capture = self._departed
            else:
                fallback = "no_retained_frame"
                self._fallbacks += 1

        return Decision(
            accepted=True,
            reason=chosen,
            sample_class=sample_class,
            capture_frame_index=capture,
            capture_fallback=fallback,
            **base,
        )


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
