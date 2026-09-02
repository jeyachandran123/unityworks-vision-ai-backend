"""Model evaluation artifacts, read-only.

Three GET routes over `app/evaluation`. Nothing here writes, and nothing here
runs anything.

### There is no "run evaluation" endpoint, deliberately

The brief permitted one only if an existing harness could be invoked safely with
bounded parameters, and none can:

* `tools/vision_eval` needs the NVIDIA VLM. The baseline run made 86 calls at a
  mean 11 seconds each — a quarter of an hour of somebody's API budget per press,
  with a network dependency and a key, which is not a bounded button.
* `experiments/vlm_prompt/score.py` **is** pure, offline and deterministic, and
  was the one real candidate. It re-scores already-recorded runs and writes
  `scores.json` — overwriting a historical artifact, which guardrail 6 forbids.
  Rescoring identical inputs to identical outputs is also not a new evaluation;
  it is a recomputation dressed as one.
* The dataset regression already runs on every `pytest` invocation. A button that
  re-ran it would report what CI reports, one layer further from where it is
  actionable.

So this surface is read-only, and the phase report says so rather than shipping a
trigger that would have to be trusted. A read-only dashboard that is truthful
beats a button nobody can audit.

### Why this is its own permission

`VIEW_MODEL_EVALUATION`, not `VIEW_REPORTS`. Evaluation artifacts answer "should
we ship this model" rather than "is the kitchen clean", and they are candid about
the product's weaknesses in a way an operational report is not — the shipped
configuration agrees with human annotation on head covering 23% of the time, and
that is a number for the people who answer for the system.

### No imagery, and no paths outside the repository

`datasets/kitchen-01/frames/` holds 4,036 JPEGs of real people and
`experiments/vlm_prompt/crops/` holds 43 crops. No route here reads, lists or
references any of them, and per-case failure records — which name a frame, a
subject and what the person was doing — are dropped in the adapters. Paths in
responses are repository-relative and appear only as provenance.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import CurrentAccess, DbSession, requires
from app.authorization.model import Permission
from app.errors import NotFoundError
from app.evaluation import catalogue

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])


@router.get("", dependencies=[Depends(requires(Permission.VIEW_MODEL_EVALUATION))])
async def evaluation_summary(access: CurrentAccess) -> dict[str, Any]:
    """Every artifact family, dataset coverage, configuration and comparison set.

    Assembled from files on disk rather than from the database, so it is the same
    answer for every tenant in this deployment — these are properties of the
    build, not of an organisation's data. The tenant is echoed so a reader can
    see which deployment they are looking at.
    """
    payload = catalogue.summary()
    payload["tenant_id"] = access.tenant_id
    return payload


@router.get("/runs/{run_id}", dependencies=[Depends(requires(Permission.VIEW_MODEL_EVALUATION))])
async def evaluation_run(run_id: str) -> dict[str, Any]:
    """One run in full, including every metric's provenance.

    Serves the detail panel. The summary already carries these, so this exists
    for a caller that wants one run without the rest — and it 404s on an unknown
    id rather than returning an empty run, which would read as a run that found
    nothing.
    """
    loaded = catalogue.families()
    for run in catalogue.all_runs(loaded):
        if run.run_id == run_id:
            return run.as_dict()
    raise NotFoundError(f"no evaluation run '{run_id}'")


@router.get(
    "/artifacts", dependencies=[Depends(requires(Permission.VIEW_MODEL_EVALUATION))]
)
async def evaluation_artifacts(
    session: DbSession,
    family: Annotated[str, Query()] = "",
) -> dict[str, Any]:
    """Which artifact files each family expects, and whether each was read.

    An operator asking "why is this family empty" needs to know what was looked
    for. Repository-relative paths only — enough to find the file in the tree,
    and nothing about where the tree lives.
    """
    loaded = catalogue.families()
    if family:
        loaded = tuple(f for f in loaded if f.key == family)
        if not loaded:
            raise NotFoundError(f"no artifact family '{family}'")

    return {
        "families": [
            {
                "key": f.key,
                "title": f.title,
                "available": f.available,
                "reason": f.reason,
                "expected_artifacts": list(f.expected_artifacts),
                "runs": [
                    {
                        "run_id": r.run_id,
                        "available": r.available,
                        "reason": r.reason,
                        "artifact": r.provenance.artifact,
                        "timestamp_source": r.provenance.timestamp_source,
                        "freshness": r.freshness.value,
                    }
                    for r in f.runs
                ],
            }
            for f in loaded
        ],
        # Stated on the surface that lists artifacts, because this is exactly
        # where somebody would expect to find a way to open one.
        "imagery_available": False,
        "imagery_reason": (
            "Dataset frames and evaluation crops exist on disk and are "
            "deliberately not reachable through this API. They are imagery of "
            "identifiable people, and exposing them would need the same "
            "click-to-retrieve and audited-access discipline evidence has — "
            "which this phase did not build."
        ),
        "run_evaluation_available": False,
        "run_evaluation_reason": (
            "No evaluation harness in this repository can be invoked with "
            "bounded parameters without either a paid network dependency or "
            "overwriting a historical artifact. See app/api/evaluation.py."
        ),
    }


__all__ = ["router"]
