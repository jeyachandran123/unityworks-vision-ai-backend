"""A person's last clear answer survives a following "could not see".

cam-12, 2026-10-22: v1 saw a chef's bare hands at 04:40:54; the next crops could
not see his hands, `not_visible` replaced `none`, and the alert opened at
04:41:33 without gloves in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.vision.recall import DecidedRecall

S = 1_000_000_000


@dataclass(frozen=True)
class Instant:
    ns: int


@dataclass(frozen=True)
class View:
    value: str
    observed_at: Instant


@dataclass(frozen=True)
class Obj:
    object_id: str
    attributes: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Snapshot:
    objects: tuple


def snap(t: int, **values: str) -> Snapshot:
    return Snapshot(objects=(Obj("p1", {k: View(v, Instant(t * S)) for k, v in values.items()}),))


def hands(recall: DecidedRecall, t: int, value: str) -> str:
    result = recall.apply(snap(t, hand_covering=value), organization_id="org", now_ns=t * S)
    return result.objects[0].attributes["hand_covering"].value


def test_a_bare_hand_seen_a_moment_ago_survives_a_crop_that_missed_it() -> None:
    recall = DecidedRecall()
    assert hands(recall, 100, "none") == "none"
    assert hands(recall, 101, "not_visible") == "none"


def test_the_recalled_value_keeps_its_own_observation_time() -> None:
    recall = DecidedRecall()
    recall.apply(snap(100, hand_covering="none"), organization_id="org", now_ns=100 * S)
    result = recall.apply(
        snap(130, hand_covering="not_visible"), organization_id="org", now_ns=130 * S
    )
    assert result.objects[0].attributes["hand_covering"].observed_at.ns == 100 * S


def test_a_later_clear_answer_replaces_the_earlier_one() -> None:
    recall = DecidedRecall()
    hands(recall, 100, "none")
    assert hands(recall, 105, "gloves") == "gloves"
    assert hands(recall, 106, "not_visible") == "gloves"


def test_nothing_older_than_the_rules_window_is_recalled() -> None:
    recall = DecidedRecall()
    hands(recall, 100, "none")
    assert hands(recall, 100 + 91, "not_visible") == "not_visible"


def test_nothing_is_invented_for_a_person_never_seen_clearly() -> None:
    assert hands(DecidedRecall(), 100, "not_visible") == "not_visible"


def test_people_and_organizations_do_not_share_memory() -> None:
    recall = DecidedRecall()
    recall.apply(snap(100, hand_covering="none"), organization_id="org-a", now_ns=100 * S)
    other = recall.apply(
        snap(101, hand_covering="not_visible"), organization_id="org-b", now_ns=101 * S
    )
    assert other.objects[0].attributes["hand_covering"].value == "not_visible"
