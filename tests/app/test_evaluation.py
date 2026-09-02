"""Reading real evaluation artifacts, and refusing to embellish them.

These tests run against the **committed artifacts**, not against fixtures. That
is deliberate: an adapter tested only on a synthetic file proves it can parse a
file somebody wrote to make it pass, and the artifacts in this repository are
irregular in exactly the ways that matter — one family has no timestamps at all,
one metric is undefined for want of ground truth, and one run produced nothing
because a provider throttled it.

The properties that carry the weight:

* a metric traces to its artifact, dataset, split and definition;
* a malformed artifact is an unavailable state, never a zero;
* incomparable runs are never one trend;
* sparse history stays sparse;
* configured thresholds match the policy file they claim to come from;
* no image path reaches the API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.authorization.model import Permission, Role, permissions_for
from app.evaluation import catalogue
from app.evaluation.adapters import common, datasets, policy, ppe, regression, vlm_prompt
from app.evaluation.model import (
    ComparabilityKey,
    Freshness,
    MetricEntry,
    MetricKind,
    Provenance,
)

from .conftest import bearer, make_user

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
async def developer(seeded):
    """The seeded `developer@example.com` already holds VIEW_MODEL_EVALUATION."""
    return seeded


@pytest.fixture
async def admin(seeded):
    database = seeded.state.database
    async with database.session_scope() as session:
        _, user = make_user(
            email="admin@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(user)
    return seeded


# ── 1. A metric traces to its provenance ─────────────────────────────────────


def test_a_metric_cannot_exist_without_provenance_and_a_definition() -> None:
    """Structural. Forgetting either is a TypeError, not a number nobody can read."""
    with pytest.raises(TypeError):
        MetricEntry(  # type: ignore[call-arg]
            key="x", label="X", kind=MetricKind.COUNT, value=1
        )


def test_every_surfaced_metric_names_its_artifact_dataset_and_definition() -> None:
    """Over every real artifact, not a sample.

    A number without a dataset, a split and a sentence saying what it measures is
    a number somebody will read as "model accuracy". This walks the whole
    catalogue and refuses to let one through.
    """
    checked = 0
    for family in catalogue.families():
        for run in family.runs:
            if not run.available:
                continue
            for group in run.groups:
                for metric in group.metrics:
                    prov = metric.provenance
                    assert prov.artifact, f"{metric.key} names no artifact"
                    assert (REPO / prov.artifact).is_file(), (
                        f"{metric.key} names {prov.artifact}, which does not exist"
                    )
                    assert prov.source, metric.key
                    assert len(metric.definition) > 30, (
                        f"{metric.key} has no usable definition: {metric.definition!r}"
                    )
                    assert prov.dataset, f"{metric.key} names no dataset"
                    assert prov.split, f"{metric.key} names no split"
                    checked += 1
    assert checked > 60, f"only {checked} metrics checked — the catalogue looks empty"


def test_the_ppe_agreement_metric_matches_its_source_file() -> None:
    """The number on the dashboard is the number in the file. No rounding, no rescaling."""
    raw = json.loads(
        (REPO / "datasets" / "kitchen-01" / "results" / "baseline.json").read_text(
            encoding="utf-8"
        )
    )
    head = next(a for a in raw["attributes"] if a["attribute"] == "head_covering")

    family = ppe.load()
    run = next(r for r in family.runs if r.run_id == "baseline")
    group = next(g for g in run.groups if g.key == "head_covering")
    metric = next(m for m in group.metrics if m.key == "head_covering.agreement")

    assert metric.value == head["accuracy"]
    assert metric.support == head["matched"]
    assert group.confusion == head["confusion"]
    # And it is not called accuracy on the wire.
    assert metric.kind is MetricKind.ATTRIBUTE_AGREEMENT
    assert "Not overall model accuracy" in metric.definition


def test_the_vlm_metric_keeps_the_experiments_own_name() -> None:
    """`accuracy_over_parsed` is not renamed to accuracy.

    The experiment gave it that name because runs with unparsed answers have a
    different denominator. Relabelling it would be the exact metric-laundering
    this phase exists to avoid.
    """
    raw = json.loads(
        (REPO / "experiments" / "vlm_prompt" / "runs" / "scores.json").read_text(
            encoding="utf-8"
        )
    )
    family = vlm_prompt.load()
    run = next(r for r in family.runs if r.run_id == "variant_A")
    metric = next(
        m
        for g in run.groups
        for m in g.metrics
        if m.key == "ungated.accuracy_over_parsed"
    )

    assert metric.value == raw["A"]["ungated"]["accuracy_over_parsed"]
    assert metric.kind is MetricKind.ACCURACY_OVER_PARSED
    assert "parsed" in metric.definition


# ── 2. Missing or malformed data is an honest absence, never a zero ──────────


def test_a_malformed_artifact_reads_as_unavailable_not_as_zero(tmp_path, monkeypatch) -> None:
    """A file that will not parse must not become a model scoring nothing."""
    broken = tmp_path / "results"
    broken.mkdir()
    (broken / "baseline.json").write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(ppe, "RESULTS", broken)
    # `read_json` sandboxes to the repository, so the temp path is refused too —
    # both failure modes land in the same honest state, which is the point.

    family = ppe.load()
    run = next(r for r in family.runs if r.run_id == "baseline")

    assert run.available is False
    assert run.reason
    assert run.groups == ()
    # Nothing anywhere claims a value.
    assert all(m.value is None for g in run.groups for m in g.metrics)


def test_a_missing_artifact_family_says_what_it_looked_for(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ppe, "RESULTS", tmp_path / "nowhere")
    family = ppe.load()

    assert family.available is False
    assert family.reason
    assert family.expected_artifacts, "an absent family must name what it expected"


def test_an_undefined_metric_is_none_and_says_why() -> None:
    """`null` in the source stays `null` here.

    `experiments/vlm_prompt/score.py`: *"a metric with no support is undefined,
    not bad, and printing a number there would be inventing one."* Ground truth
    holds no `absent` examples, so recall for that state has no denominator.
    """
    family = vlm_prompt.load()
    run = next(r for r in family.runs if r.run_id == "variant_A")
    recall = next(
        m for g in run.groups for m in g.metrics if m.key == "ungated.absent.recall"
    )

    assert recall.value is None
    assert "undefined" in recall.undefined_reason.lower()
    assert "zero" in recall.undefined_reason.lower()


def test_an_empty_dataset_is_not_a_dataset_of_zero() -> None:
    """`vision-phase5` is awaiting footage. It has no counts, not zero counts."""
    coverages = datasets.load()
    phase5 = next(c for c in coverages if c.name == "vision-phase5")

    assert phase5.status == "AWAITING FOOTAGE"
    assert phase5.frames is None
    assert phase5.subjects is None


# ── 3. Incomparable evaluations are never one trend ──────────────────────────


def test_a_different_model_never_shares_an_axis() -> None:
    """The minimax probe and the llama variants are never grouped.

    Same corpus, same attribute, different model — and the probe answered
    nothing at all. A chart joining them would compare a rate-limit against a
    prompt.
    """
    loaded = catalogue.families()
    sets = catalogue.comparison_sets(loaded)

    llama = next(s for s in sets if "variant_A" in s["run_ids"])
    minimax = next(s for s in sets if "minimax_m3_20260831" in s["run_ids"])

    assert llama is not minimax
    assert set(llama["run_ids"]) == {"variant_A", "variant_B", "variant_C", "variant_D"}
    assert minimax["run_ids"] == ["minimax_m3_20260831"]
    assert minimax["comparable"] is False


def test_a_run_with_no_recorded_configuration_is_never_joined_to_another() -> None:
    """Two runs that both failed to record a configuration are not the same run."""
    a = ComparabilityKey("attribute_agreement", "m", "d", "test", "")
    b = ComparabilityKey("attribute_agreement", "m", "d", "test", "")
    assert a.comparable_with(b) is False

    with_config = ComparabilityKey("attribute_agreement", "m", "d", "test", "cfg")
    assert with_config.comparable_with(with_config) is True


def test_the_regression_recording_never_joins_the_ppe_runs() -> None:
    """Both call a number 'agreement'. They are not the same measurement.

    The harness computes agreement over detector-matched subjects; the recording
    compares two fields directly. Sharing an axis would be exactly the
    metric-conflation this phase forbids.
    """
    loaded = catalogue.families()
    sets = catalogue.comparison_sets(loaded)

    recording = next(s for s in sets if "kitchen01_model_answers" in s["run_ids"])
    assert recording["run_ids"] == ["kitchen01_model_answers"]

    ppe_set = next(s for s in sets if "baseline" in s["run_ids"])
    assert "kitchen01_model_answers" not in ppe_set["run_ids"]


def test_a_single_run_group_is_labelled_a_snapshot_not_a_trend() -> None:
    loaded = catalogue.families()
    for group in catalogue.comparison_sets(loaded):
        if len(group["run_ids"]) == 1:
            assert group["comparable"] is False
            assert "snapshot" in group["why"] or "on its own" in group["why"]


# ── 4. Sparse history stays sparse ───────────────────────────────────────────


def test_undated_runs_are_undated_rather_than_stale() -> None:
    """Four PPE runs carry no evaluation date. That is its own state.

    Rendering them as very old would assert something nobody recorded, and
    reading a file's mtime would assert when git touched it.
    """
    family = ppe.load()
    for run in family.runs:
        assert run.provenance.evaluated_at is None
        assert run.provenance.timestamp_source == "absent"
        assert run.freshness is Freshness.UNDATED


def test_no_timestamp_is_ever_taken_from_the_filesystem() -> None:
    """`parse_instant` has no mtime fallback, and nothing calls one."""
    assert common.parse_instant(None) is None
    assert common.parse_instant("") is None
    assert common.parse_instant("not a date") is None

    source = (REPO / "app" / "evaluation").rglob("*.py")
    for path in source:
        text = path.read_text(encoding="utf-8")
        for banned in ("st_mtime", "getmtime", "st_ctime"):
            assert banned not in text, f"{path.name} reads a filesystem time"


def test_the_summary_counts_dated_and_undated_separately() -> None:
    """A reader must be able to see how much of the history has a date at all."""
    summary = catalogue.summary()
    totals = summary["totals"]

    assert totals["runs_dated"] > 0
    assert totals["runs_undated"] > 0
    assert totals["runs_dated"] + totals["runs_undated"] == totals["runs_available"]


def test_there_is_no_aggregate_model_score() -> None:
    """No headline number, and the payload says why rather than omitting it."""
    summary = catalogue.summary()
    assert summary["headline_metric"] is None
    assert "no definition" in summary["headline_reason"]


# ── 5. Configured thresholds match their source ──────────────────────────────


def test_the_confidence_threshold_matches_the_policy_file() -> None:
    """Read from the document the deployment uses, not restated in code."""
    document = json.loads(
        (REPO / "config" / "policies" / "kitchen-safety.example.json").read_text(
            encoding="utf-8"
        )
    )
    available, _reason, groups = policy.load()
    assert available is True

    example = next(g for g in groups if g.key.endswith("kitchen-safety.example.json"))
    threshold = next(m for m in example.metrics if m.key == "scope.min_confidence")

    assert threshold.value == document["scope"]["min_confidence"]
    assert threshold.kind is MetricKind.CONFIGURED_THRESHOLD
    # A threshold is configuration, and its provenance says so rather than
    # pretending it has an evaluation date.
    assert threshold.provenance.timestamp_source == "not_applicable"
    assert threshold.provenance.configuration.startswith(document["policy_id"])


def test_the_attribute_validity_window_matches_the_policy_file() -> None:
    document = json.loads(
        (REPO / "config" / "policies" / "kitchen-safety.example.json").read_text(
            encoding="utf-8"
        )
    )
    head = next(a for a in document["attributes"] if a["key"] == "head_covering")

    _available, _reason, groups = policy.load()
    example = next(g for g in groups if g.key.endswith("kitchen-safety.example.json"))
    validity = next(
        m for m in example.metrics if m.key == "attribute.head_covering.validity_ms"
    )
    assert validity.value == head["validity_ms"]


# ── 6. No imagery, and no filesystem structure ───────────────────────────────


async def test_no_image_reference_reaches_the_api(client: AsyncClient, developer) -> None:
    """4,036 dataset frames and 43 crops exist on disk. None of them is reachable.

    Asserted over the serialised response rather than over the adapters, because
    the response is what a browser receives — and an image path in a JSON blob is
    an image path whether or not a component renders it.
    """
    headers = await bearer(client, "developer@example.com")
    body = (await client.get("/api/v1/evaluation", headers=headers)).text

    for banned in (".jpg", ".jpeg", ".png", "frames/", "crops/"):
        assert banned not in body, f"{banned!r} appeared in the evaluation payload"


async def test_no_absolute_path_reaches_the_api(client: AsyncClient, developer) -> None:
    """Provenance paths are repository-relative. Nothing says where the checkout is."""
    headers = await bearer(client, "developer@example.com")
    body = (await client.get("/api/v1/evaluation", headers=headers)).text

    assert "C:\\" not in body
    assert "/home/" not in body
    assert "\\\\" not in body, "a Windows path survived into the payload"


async def test_per_case_failure_detail_is_not_surfaced(
    client: AsyncClient, developer
) -> None:
    """The reports carry per-person failure cases. Only the tallies leave.

    Each case names a frame, a subject and what the person was doing. The
    dashboard's question is answered by the confusion matrix and the category
    counts.
    """
    raw = json.loads(
        (REPO / "datasets" / "kitchen-01" / "results" / "baseline.json").read_text(
            encoding="utf-8"
        )
    )
    sample = raw["failures"][0]

    headers = await bearer(client, "developer@example.com")
    body = (await client.get("/api/v1/evaluation", headers=headers)).text

    assert sample["frame_id"] not in body
    assert sample["detail"] not in body
    # The category tally is present, because that is the useful part.
    assert "vlm_failure" in body


def test_the_regression_adapter_drops_case_notes() -> None:
    """The recording's per-case notes describe people. They do not leave the file."""
    raw = json.loads(
        (REPO / "tests" / "compliance" / "kitchen01_model_answers.json").read_text(
            encoding="utf-8"
        )
    )
    note = raw["cases"][0]["note"]
    assert note, "fixture assumption: the recording has case notes"

    payload = json.dumps(regression.load().as_dict())
    assert note not in payload
    assert raw["cases"][0]["frame"] not in payload


def test_an_artifact_read_is_sandboxed_to_the_repository(tmp_path) -> None:
    """`read_json` refuses anything outside the allowed roots.

    Nothing takes a path from a caller, so this is the second lock on a door with
    no handle — and the one that stays useful if a future adapter grows one.
    """
    outside = tmp_path / "secrets.json"
    outside.write_text('{"a": 1}', encoding="utf-8")
    assert common.read_json(outside) is None

    # And a real artifact inside the roots still reads.
    assert common.read_json(REPO / "datasets" / "kitchen-01" / "dataset.json") is not None


def test_an_oversized_artifact_is_refused() -> None:
    """`p9-live/review_queue.json` is 4 MB of candidate rows with image paths."""
    big = REPO / "datasets" / "p9-live" / "review_queue.json"
    if not big.is_file():
        pytest.skip("the large p9 review queue is not present in this checkout")
    assert big.stat().st_size > common.MAX_ARTIFACT_BYTES
    assert common.read_json(big) is None


# ── The dataset's own limitations survive ────────────────────────────────────


def test_the_dataset_note_about_detection_recall_is_carried_through() -> None:
    """The most important sentence in the inventory, surfaced rather than dropped.

    `kitchen-01/dataset.json` says the set cannot measure detection recall. A
    coverage panel showing 43 subjects and 0 unmatched without it would invite
    exactly the wrong conclusion.
    """
    coverage = next(c for c in datasets.load() if c.name == "kitchen-01")
    joined = " ".join(coverage.limitations)
    assert "CANNOT measure detection recall" in joined


def test_no_recall_is_derived_from_the_detection_counts() -> None:
    """Counts are counts. The split cannot support a recall figure and none is made."""
    family = ppe.load()
    run = next(r for r in family.runs if r.run_id == "baseline")
    run_group = next(g for g in run.groups if g.key == "run")

    unmatched = next(m for m in run_group.metrics if m.key == "unmatched_truth")
    assert unmatched.kind is MetricKind.COUNT
    assert "cannot measure what the detector missed" in unmatched.definition

    assert not any(
        m.kind is MetricKind.RECALL and m.key.startswith("detection")
        for g in run.groups
        for m in g.metrics
    )


# ── Routes ───────────────────────────────────────────────────────────────────


async def test_the_evaluation_route_is_permission_gated(
    client: AsyncClient, admin
) -> None:
    """A restaurant manager does not read model evaluation."""
    manager = await bearer(client, "manager@example.com")
    assert (await client.get("/api/v1/evaluation", headers=manager)).status_code == 403

    developer = await bearer(client, "developer@example.com")
    assert (await client.get("/api/v1/evaluation", headers=developer)).status_code == 200


async def test_an_organisation_administrator_no_longer_reads_evaluation(
    client: AsyncClient, admin
) -> None:
    """The Phase 4 role correction, asserted at the route rather than the map.

    Hiding the navigation entry is not closing the door. This is the door: an
    org_admin who types the address, follows a bookmark or replays a saved
    request is refused by the server, with the refusal audited exactly as every
    other refusal on this route already is.
    """
    headers = await bearer(client, "admin@example.com")
    assert (await client.get("/api/v1/evaluation", headers=headers)).status_code == 403
    assert (
        await client.get("/api/v1/evaluation/artifacts", headers=headers)
    ).status_code == 403


async def test_the_evaluation_route_refuses_an_unauthenticated_caller(
    client: AsyncClient, seeded
) -> None:
    assert (await client.get("/api/v1/evaluation")).status_code == 401


async def test_an_unknown_run_is_a_404_not_an_empty_run(
    client: AsyncClient, developer
) -> None:
    """An empty run would read as a run that measured nothing."""
    headers = await bearer(client, "developer@example.com")
    assert (
        await client.get("/api/v1/evaluation/runs/no-such-run", headers=headers)
    ).status_code == 404


async def test_a_run_route_carries_full_provenance(client: AsyncClient, developer) -> None:
    headers = await bearer(client, "developer@example.com")
    body = (await client.get("/api/v1/evaluation/runs/variant_A", headers=headers)).json()

    assert body["provenance"]["model"] == "meta/llama-3.2-11b-vision-instruct"
    assert body["provenance"]["evaluated_at"]
    assert body["provenance"]["timestamp_source"] == "artifact"
    assert body["provenance"]["limitations"]


async def test_the_artifact_route_states_what_is_deliberately_absent(
    client: AsyncClient, developer
) -> None:
    """Imagery and the run trigger, both named where somebody would look."""
    headers = await bearer(client, "developer@example.com")
    body = (await client.get("/api/v1/evaluation/artifacts", headers=headers)).json()

    assert body["imagery_available"] is False
    assert body["imagery_reason"]
    assert body["run_evaluation_available"] is False
    assert body["run_evaluation_reason"]


async def test_there_is_no_write_route_on_the_evaluation_surface(developer) -> None:
    """Read-only, asserted over the whole application rather than one router."""
    writes = [
        route
        for route in developer.routes
        if "/evaluation" in getattr(route, "path", "")
        and set(getattr(route, "methods", set())) - {"GET", "HEAD", "OPTIONS"}
    ]
    assert writes == []


def test_the_evaluation_permission_is_not_granted_broadly() -> None:
    """Two roles, and no operational or administrative one is among them.

    ORG_ADMIN was removed deliberately. The reasoning that put it there —
    an organisation administrator answers for what the system claims — is
    served by VIEW_REPORTS, which that role still holds and which carries
    coverage, completeness and the ruleset version behind every figure.
    What this permission actually opens is attribute agreement on a
    43-subject split and per-state confusion matrices, which answer a
    shipping question rather than an operational one.
    """
    holders = {
        role.value
        for role in Role
        if Permission.VIEW_MODEL_EVALUATION in permissions_for(frozenset({role}))
    }
    assert holders == {"super_admin", "developer"}


def test_nothing_in_the_evaluation_package_writes_to_disk() -> None:
    """Artifacts are immutable snapshots. Nothing here can modify one."""
    for path in (REPO / "app" / "evaluation").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for banned in ("write_text(", "write_bytes(", "open(", "unlink(", "mkdir("):
            assert banned not in text, f"{path.name} contains {banned!r}"
