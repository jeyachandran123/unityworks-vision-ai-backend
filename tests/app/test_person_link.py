"""One person, one tag, across the platform's object-id hand-overs.

cam-13, 2026-10-22: one chef walking past became three alerts tagged 3RQHQY,
Q858TP and X84R8K within eleven seconds - each a new platform object id on the
spot the previous one had just left.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.vision.person_link import PersonLinker

S = 1_000_000_000


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Spatial:
    bbox: Box


@dataclass(frozen=True)
class Instant:
    ns: int


@dataclass(frozen=True)
class Obj:
    object_id: str
    camera_id: str
    spatial: Spatial
    last_seen: Instant


@dataclass(frozen=True)
class Snapshot:
    objects: tuple


def at(t: float, *people: tuple[str, tuple[float, float, float, float]], camera="cam-13"):
    return Snapshot(
        objects=tuple(
            Obj(oid, camera, Spatial(Box(*box)), Instant(int(t * S))) for oid, box in people
        )
    )


CHEF = (0.40, 0.20, 0.50, 0.70)
MOVED = (0.42, 0.21, 0.52, 0.71)
ELSEWHERE = (0.05, 0.30, 0.15, 0.90)


def link(linker, t, *people, camera="cam-13"):
    return linker.link(at(t, *people, camera=camera), organization_id="org", now_ns=int(t * S))


def test_a_new_id_on_the_spot_a_person_just_left_is_the_same_person() -> None:
    people = PersonLinker()
    link(people, 100, ("3RQHQY", CHEF))
    assert link(people, 101, ("Q858TP", MOVED)) == {"Q858TP": "3RQHQY"}
    assert link(people, 102, ("X84R8K", CHEF)) == {"X84R8K": "3RQHQY"}


def test_someone_elsewhere_is_a_different_person() -> None:
    people = PersonLinker()
    link(people, 100, ("A", CHEF))
    assert link(people, 101, ("B", ELSEWHERE)) == {"B": "B"}


def test_a_person_still_in_view_is_never_given_away() -> None:
    """Two people side by side: the newcomer must not take the one who stayed."""
    people = PersonLinker()
    link(people, 100, ("A", CHEF))
    assert link(people, 101, ("A", CHEF), ("B", MOVED)) == {"A": "A", "B": "B"}


def test_too_long_a_gap_starts_a_new_person() -> None:
    people = PersonLinker()
    link(people, 100, ("A", CHEF))
    assert link(people, 111, ("B", CHEF)) == {"B": "B"}


def test_cameras_do_not_share_people() -> None:
    people = PersonLinker()
    link(people, 100, ("A", CHEF), camera="cam-12")
    assert link(people, 101, ("B", CHEF), camera="cam-13") == {"B": "B"}


def test_a_linked_id_stays_linked() -> None:
    people = PersonLinker()
    link(people, 100, ("A", CHEF))
    link(people, 101, ("B", MOVED))
    assert people.person("org", "B") == "A"
    assert link(people, 102, ("B", MOVED)) == {"B": "A"}


@dataclass(frozen=True)
class Tracked(Obj):
    lifecycle: str = "active"


def test_a_dropped_id_lingering_as_occluded_does_not_block_its_successor() -> None:
    people = PersonLinker()
    people.link(
        Snapshot(objects=(Tracked("A", "cam-13", Spatial(Box(*CHEF)), Instant(100 * S)),)),
        organization_id="org",
        now_ns=100 * S,
    )
    later = Snapshot(
        objects=(
            Tracked("A", "cam-13", Spatial(Box(*CHEF)), Instant(100 * S), lifecycle="occluded"),
            Tracked("B", "cam-13", Spatial(Box(*MOVED)), Instant(102 * S)),
        )
    )
    assert people.link(later, organization_id="org", now_ns=102 * S)["B"] == "A"
