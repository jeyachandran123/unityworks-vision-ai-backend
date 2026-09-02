"""The product observation surface.

Two properties carry the weight here, and both are about honesty rather than
about data:

* **Unavailable is not empty.** A platform that is not assembled says so. A
  hygiene screen that rendered "no subjects observed" for a system that was not
  watching would be the exact failure this product exists to prevent.

* **The four states survive transport.** `not_visible` arrives at the client as
  `not_visible`. Nothing between the log and the wire is allowed to decide that
  it means "absent", because one of those is a violation and the other never is.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from httpx import AsyncClient

from app.api.product import _observation_window
from app.domain.observations import query_observations
from app.authorization.model import AccessDecision, CameraScope, Role, ScopeBreadth
from app.errors import ValidationError

from .conftest import bearer

pytestmark = pytest.mark.asyncio


# ── the window ───────────────────────────────────────────────────────────────


async def test_window_defaults_to_the_last_day() -> None:
    start, end = _observation_window(None, None)
    assert 23.9 < (end - start).total_seconds() / 3600 < 24.1


async def test_window_refuses_to_end_before_it_starts() -> None:
    later = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    earlier = datetime(2026, 9, 1, 6, 0, tzinfo=UTC)
    with pytest.raises(ValidationError):
        _observation_window(later, earlier)


# ── folding observations into subjects ───────────────────────────────────────


def _access() -> AccessDecision:
    return AccessDecision(
        subject="manager@example.com",
        tenant_id="org-test",
        roles=frozenset({Role.RESTAURANT_MANAGER}),
        cameras=CameraScope(breadth=ScopeBreadth.LISTED, camera_ids=("cam-01",)),
    )


def _attribute(key: str, value: str, ns: int):
    return SimpleNamespace(
        key=key,
        value=value,
        observed_at=SimpleNamespace(ns=ns),
        valid_until=None,
        confidence=SimpleNamespace(value=0.9, semantics=SimpleNamespace(value="self_reported"), calibrated=False),
    )


def _observation(object_id: str, ns: int, attributes):
    return SimpleNamespace(
        object_id=object_id,
        camera_id="cam-01",
        class_id="person",
        t_capture=SimpleNamespace(ns=ns),
        attributes=tuple(attributes),
    )


class _FakeApi:
    """Stands in for the platform's ObservationApi. Records what it was asked."""

    def __init__(self, observations, fully_observable: bool = True) -> None:
        self.observations = observations
        self.fully_observable = fully_observable
        self.calls: list = []

    def query_observations(self, principal, scope, window, **kwargs):
        self.calls.append((principal, scope, window, kwargs))
        return SimpleNamespace(
            observations=tuple(self.observations),
            cursor=None,
            window_fully_observable=self.fully_observable,
        )


def _fold(observations, **kwargs):
    api = _FakeApi(observations, **kwargs)
    subjects, count, observable = query_observations(
        api,
        _access(),
        ("cam-01",),
        datetime(2026, 9, 1, tzinfo=UTC),
        datetime(2026, 9, 2, tzinfo=UTC),
        200,
    )
    return api, subjects, count, observable


async def test_not_visible_survives_the_fold_unchanged() -> None:
    """The value the platform reported is the value the client receives."""
    _, subjects, _, _ = _fold(
        [_observation("obj-1", 1_000, [_attribute("head_covering", "not_visible", 1_000)])]
    )

    assert len(subjects) == 1
    assert subjects[0]["attributes"][0]["value"] == "not_visible"


async def test_every_state_passes_through_verbatim() -> None:
    observations = [
        _observation("obj-1", 1_000, [_attribute("head_covering", "hairnet", 1_000)]),
        _observation("obj-2", 2_000, [_attribute("head_covering", "none", 2_000)]),
        _observation("obj-3", 3_000, [_attribute("head_covering", "not_visible", 3_000)]),
    ]
    _, subjects, _, _ = _fold(observations)

    by_id = {s["object_id"]: s["attributes"][0]["value"] for s in subjects}
    assert by_id == {"obj-1": "hairnet", "obj-2": "none", "obj-3": "not_visible"}


async def test_the_freshest_reading_per_attribute_wins() -> None:
    """A later observation revises an earlier one; the earlier one is not shown."""
    observations = [
        _observation("obj-1", 1_000, [_attribute("head_covering", "none", 1_000)]),
        _observation("obj-1", 5_000, [_attribute("head_covering", "hairnet", 5_000)]),
    ]
    _, subjects, count, _ = _fold(observations)

    assert len(subjects) == 1, "the same subject must not become two rows"
    assert subjects[0]["attributes"][0]["value"] == "hairnet"
    assert subjects[0]["first_seen"] == 1_000
    assert subjects[0]["last_seen"] == 5_000
    assert count == 2, "the observation count reports what was read, not what was kept"


async def test_an_older_observation_does_not_overwrite_a_newer_one() -> None:
    """Log order is oldest-first, but the fold must not trust that blindly."""
    observations = [
        _observation("obj-1", 9_000, [_attribute("head_covering", "hairnet", 9_000)]),
        _observation("obj-1", 1_000, [_attribute("head_covering", "none", 1_000)]),
    ]
    _, subjects, _, _ = _fold(observations)
    assert subjects[0]["attributes"][0]["value"] == "hairnet"


async def test_a_coverage_observation_is_not_a_subject() -> None:
    """`object_id is None` is a statement about the platform, not about a person."""
    observations = [
        _observation("obj-1", 1_000, [_attribute("head_covering", "hairnet", 1_000)]),
        SimpleNamespace(
            object_id=None,
            camera_id="cam-01",
            class_id="",
            t_capture=SimpleNamespace(ns=2_000),
            attributes=(),
        ),
    ]
    _, subjects, _, _ = _fold(observations)
    assert [s["object_id"] for s in subjects] == ["obj-1"]


async def test_confidence_carries_its_semantics() -> None:
    """A self-reported score must not be presentable as a probability."""
    _, subjects, _, _ = _fold(
        [_observation("obj-1", 1_000, [_attribute("head_covering", "hairnet", 1_000)])]
    )
    confidence = subjects[0]["attributes"][0]["confidence"]
    assert confidence["semantics"] == "self_reported"
    assert confidence["calibrated"] is False


async def test_the_scope_handed_to_the_platform_is_the_callers_cameras() -> None:
    api, _, _, _ = _fold([])
    _principal, scope, _window, _kwargs = api.calls[0]
    assert [str(c) for c in scope.camera_ids] == ["cam-01"]
    assert str(scope.tenant_id) == "org-test"


# ── the route ────────────────────────────────────────────────────────────────


async def test_observations_require_authentication(client: AsyncClient, seeded) -> None:
    response = await client.get("/api/v1/observations")
    assert response.status_code == 401


async def test_unavailable_platform_says_so_rather_than_reporting_nothing_seen(
    client: AsyncClient, seeded
) -> None:
    """The property this endpoint exists to protect.

    `vision_autostart` is false in the test settings, so nothing is assembled.
    The answer must be *unavailable with a reason* — never a 200 carrying an
    empty list, which a screen would render as "nobody broke the rules".
    """
    headers = await bearer(client, "manager@example.com")
    response = await client.get("/api/v1/observations", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["reason"], "an unavailable platform must say why"
    assert body["subjects"] == []
    assert body["window_fully_observable"] is False


async def test_an_empty_camera_grant_matches_nothing_and_is_not_a_wildcard(
    client: AsyncClient, seeded
) -> None:
    """`ScopeBreadth.NONE` returns available-and-empty, never everything."""
    headers = await bearer(client, "nocameras@example.com")
    response = await client.get("/api/v1/observations", headers=headers)

    assert response.status_code == 200
    body = response.json()
    # Available: the platform's state is irrelevant, this account simply reaches
    # no camera. That is a different sentence from "the platform is down".
    assert body["available"] is True
    assert body["subjects"] == []
    assert body["cameras_queried"] == []


async def test_a_malformed_window_is_rejected(client: AsyncClient, seeded) -> None:
    headers = await bearer(client, "manager@example.com")
    response = await client.get(
        "/api/v1/observations", params={"since": "not-a-timestamp"}, headers=headers
    )
    assert response.status_code == 422


# ── the shared fold: one implementation, two consumers ───────────────────────


async def test_both_consumers_fold_identically(seeded) -> None:
    """The Phase 3 close-out, asserted rather than asserted-to-have-been-done.

    `app/api/product.py` and `app/reporting/sources.py` must call the *same*
    implementation. A second copy is the one place `not_visible` could later be
    collapsed into `none` in a reporting path while the API path stayed correct,
    and nothing would fail until somebody read a report.

    So this drives both consumers over identical input and compares their subject
    records byte for byte. The fixture deliberately includes all four states.
    """
    import json

    from app.domain import observations as fold_module
    from app.reporting.model import Granularity, ReportRequest
    from app.reporting.periods import resolve_timezone
    from app.reporting.sources import collect_observations

    observations = [
        _observation("obj-present", 1_000, [_attribute("head_covering", "hairnet", 1_000)]),
        _observation("obj-absent", 2_000, [_attribute("head_covering", "none", 2_000)]),
        _observation("obj-refused", 3_000, [_attribute("head_covering", "not_visible", 3_000)]),
        _observation("obj-unknown", 4_000, [_attribute("head_covering", "unknown", 4_000)]),
    ]
    start = datetime(2026, 9, 1, tzinfo=UTC)
    end = datetime(2026, 9, 2, tzinfo=UTC)

    # Consumer A: exactly what the product route calls.
    from app.api import product as product_module

    api_a = _FakeApi(observations)
    via_api, count_a, observable_a = product_module.observation_fold.query_observations(
        api_a, _access(), ("cam-01",), start, end, 200
    )

    # Consumer B: exactly what the reporting collector calls, reached through the
    # collector itself so the binding is under test rather than a direct call.
    from app.reporting import sources as reporting_module

    api_b = _FakeApi(observations)
    captured: dict[str, object] = {}
    real_fold = reporting_module.observation_fold.query_observations

    def recording(*args, **kwargs):
        result = real_fold(*args, **kwargs)
        captured["subjects"] = result[0]
        captured["count"] = result[1]
        return result

    request = ReportRequest(
        report_id="hygiene_observations",
        since=start,
        until=end,
        granularity=Granularity.TOTAL,
        timezone="UTC",
        timezone_resolved=True,
        organization_id="org-test",
        camera_keys=("cam-01",),
        row_limit=200,
    )
    database = seeded.state.database
    async with database.session_scope() as session:
        original = reporting_module.observation_fold.query_observations
        reporting_module.observation_fold.query_observations = recording
        try:
            await collect_observations(
                session,
                request,
                resolve_timezone("UTC"),
                exposure_api=api_b,
                unavailable_reason="",
                access=_access(),
            )
        finally:
            reporting_module.observation_fold.query_observations = original

    via_reporting = captured["subjects"]

    # The reporting collector attaches zone attribution afterwards; the fold's
    # own output is what must match, so those keys are dropped before comparing.
    stripped = [
        {k: v for k, v in subject.items() if k not in {"zone_id", "zone_name", "zone_recorded"}}
        for subject in via_reporting
    ]

    assert json.dumps(stripped, sort_keys=True) == json.dumps(via_api, sort_keys=True)
    assert captured["count"] == count_a
    assert observable_a is True

    # And the property the sharing exists to protect, checked on the result both
    # consumers received.
    values = {s["object_id"]: s["attributes"][0]["value"] for s in via_api}
    assert values == {
        "obj-present": "hairnet",
        "obj-absent": "none",
        "obj-refused": "not_visible",
        "obj-unknown": "unknown",
    }

    # Both consumers reach the same module object, not two lookalikes.
    assert product_module.observation_fold is fold_module
    assert reporting_module.observation_fold is fold_module


async def test_the_fold_lives_in_exactly_one_place() -> None:
    """No second implementation anywhere in `app/`.

    Guards the close-out against being quietly undone: a future contributor who
    copies the fold into the reporting layer to avoid an import fails here rather
    than in a report six months later.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "app"
    definitions = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
        and "def query_observations(" in path.read_text(encoding="utf-8")
    ]
    assert definitions == ["domain/observations.py"], definitions
