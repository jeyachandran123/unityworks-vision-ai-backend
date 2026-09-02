"""Five product modules whose schema exists and whose data source does not.

People counting, demography, table occupancy, cutting-board compliance and meal
detection. Each route is permission-gated, tenant-scoped, and answers with the
capability shape from `app/api/capability.py` rather than a 404 or an empty list.

### Every route here is a read that returns no readings

There is no detection logic in this module, no computed count and no verdict.
The correct output of this phase is that none exists, and each route says so in
the words of the specific input it is waiting for — a trained detector, a floor
plan, a site's colour scheme — rather than a generic "coming soon".

The counts are real: `stored_records` is `SELECT count(*)` over the caller's own
tenant. It is zero because the tables are empty, and the day something writes to
one it stops being zero without a line of this file changing.

### Permissions are not implied by one another

`VIEW_DEMOGRAPHY` is not implied by `VIEW_PEOPLE_COUNT`. Counting how many
people passed a door is a footfall figure; inferring their age or gender is a
different purpose under PDPA with its own lawful basis and its own notice
obligation, and a role that may read the first has no automatic claim on the
second.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.api.capability import ModuleCapability, Requirement, render
from app.api.dependencies import CurrentAccess, DbSession, requires
from app.authorization.model import Permission
from app.domain import modules as module_models

router = APIRouter(prefix="/api/v1/modules", tags=["modules"])


# ── People counting ──────────────────────────────────────────────────────────

PEOPLE_COUNTING = ModuleCapability(
    module="people_counting",
    title="People Counting",
    purpose=(
        "Entries and exits per zone over closed time buckets, with the "
        "coverage each bucket was computed from, so a quiet hour and a camera "
        "that was down never read the same."
    ),
    reason=(
        "No counting line is configured and no person-counting detector is "
        "bound. The platform detects and tracks people, but nothing turns those "
        "tracks into directional crossings yet."
    ),
    requirements=(
        Requirement(
            "counting_geometry",
            "A counting line or region per camera, in normalised frame "
            "coordinates, with which side is 'in'. This is a per-camera survey "
            "decision — a doorway line drawn for one mounting angle is wrong "
            "for another — and nobody has supplied one.",
        ),
        Requirement(
            "crossing_logic_validation",
            "A validated line-crossing rule over existing tracks, including "
            "what happens when a track is lost and re-acquired mid-crossing. "
            "Without it a reconnect double-counts, and a footfall figure that "
            "quietly inflates is worse than none.",
        ),
        Requirement(
            "coverage_accounting",
            "A per-bucket record of how many seconds were actually observed, "
            "read from camera health. The schema has the column; the source "
            "that fills it does not exist, and a count without its coverage "
            "cannot be read.",
        ),
        Requirement(
            "bucket_policy",
            "The bucket size and the site's operating-day boundary, so peak-"
            "hour analysis and branch comparison are computed on the same "
            "clock. Comparing a 15-minute series with an hourly one is the "
            "usual way these reports become quietly wrong.",
        ),
    ),
    tables=("people_count_intervals",),
    documentation="docs/architecture/NOT_YET_CONNECTED.md#people-counting",
)


@router.get(
    "/people-counting",
    dependencies=[Depends(requires(Permission.VIEW_PEOPLE_COUNT))],
)
async def people_counting(access: CurrentAccess, session: DbSession) -> dict[str, Any]:
    """Footfall capability. Reports no count, because none can be produced."""
    return await render(
        PEOPLE_COUNTING,
        session,
        organization_id=access.tenant_id,
        models=(module_models.PeopleCountInterval,),
    )


# ── Demography ───────────────────────────────────────────────────────────────

DEMOGRAPHY = ModuleCapability(
    module="demography",
    title="Demography",
    purpose=(
        "Aggregate category counts per zone and time bucket — never a person. "
        "The schema carries no subject reference of any kind, so this cannot "
        "become per-individual without a migration somebody has to sign."
    ),
    reason=(
        "No demographic classifier is bound, and none may be until this "
        "collection has its own lawful basis. Inferring age or gender from a "
        "camera is a different purpose from food-safety monitoring, and it does "
        "not inherit that purpose's basis or its notice."
    ),
    requirements=(
        Requirement(
            "lawful_basis_and_notice",
            "A PDPA lawful basis for demographic inference specifically, and "
            "signage or notice that states it. This is the gating item, not the "
            "model: without it the classifier must not run at all.",
        ),
        Requirement(
            "k_anonymity_threshold",
            "The minimum bucket size below which a count is suppressed rather "
            "than stored. A category containing one person, combined with a "
            "shift roster, names them — so the floor is a privacy control and a "
            "person must choose it.",
        ),
        Requirement(
            "classifier_and_bias_evaluation",
            "A demographic classifier with a published evaluation across the "
            "populations it will be pointed at. Age and gender classifiers are "
            "known to degrade unevenly by skin tone and age, and shipping one "
            "unevaluated turns a bias into a reported statistic.",
        ),
        Requirement(
            "category_vocabulary",
            "The category axes and values the deployment will report, agreed "
            "in advance. An open vocabulary means the model's own guesses "
            "become the product's categories.",
        ),
    ),
    tables=("demography_snapshots",),
    documentation="docs/architecture/NOT_YET_CONNECTED.md#demography",
)


@router.get("/demography", dependencies=[Depends(requires(Permission.VIEW_DEMOGRAPHY))])
async def demography(access: CurrentAccess, session: DbSession) -> dict[str, Any]:
    """Aggregate demography capability. Never per-person, by schema."""
    return await render(
        DEMOGRAPHY,
        session,
        organization_id=access.tenant_id,
        models=(module_models.DemographySnapshot,),
        extra={
            # Stated in the payload so the frontend renders the guarantee from
            # the server rather than asserting it on its own authority.
            "aggregate_only": True,
            "aggregate_only_detail": (
                "demography_snapshots has no object_id, track_id or evidence "
                "reference. There is no column that could link a row to an "
                "individual."
            ),
        },
    )


# ── Table occupancy ──────────────────────────────────────────────────────────

TABLE_OCCUPANCY = ModuleCapability(
    module="table_occupancy",
    title="Table Occupancy",
    purpose=(
        "Each table's state over time — vacant, occupied, needs cleaning — with "
        "turnover derived from the transitions, and where each transition "
        "happened frozen onto the record."
    ),
    reason=(
        "No floor plan is configured and no table-state detector is bound. "
        "There are no tables to report on and nothing watching them."
    ),
    requirements=(
        Requirement(
            "floor_plan",
            "The tables at each site: code, seats, zone, which camera sees "
            "them and the region of that camera's frame they occupy. This is a "
            "survey per site and cannot be inferred from a video stream.",
        ),
        Requirement(
            "state_detector",
            "A detector that distinguishes vacant, occupied and needs-cleaning "
            "from a fixed region. A table with plates on it and a table with "
            "diners at it are different states, and 'something is on the "
            "table' does not separate them.",
        ),
        Requirement(
            "cleaning_sla",
            "The minutes after which a needs-cleaning table becomes an alert, "
            "per site. A threshold nobody has set cannot raise an alert, and "
            "inventing one would generate a work queue out of a guess.",
        ),
        Requirement(
            "occlusion_policy",
            "What to report when a table is out of view — a passing trolley, a "
            "standing group. `not_visible` and `unknown` exist in the state "
            "enum for exactly this, and the policy for when each applies has "
            "not been written.",
        ),
    ),
    tables=("dining_tables", "table_status_events"),
    documentation="docs/architecture/NOT_YET_CONNECTED.md#table-occupancy",
)


@router.get(
    "/table-occupancy",
    dependencies=[Depends(requires(Permission.VIEW_TABLE_OCCUPANCY))],
)
async def table_occupancy(access: CurrentAccess, session: DbSession) -> dict[str, Any]:
    """Table state capability, plus the real (empty) floor plan."""
    return await render(
        TABLE_OCCUPANCY,
        session,
        organization_id=access.tenant_id,
        models=(module_models.DiningTable, module_models.TableStatusEvent),
        extra={
            # The four-plus-two states are declared by the server so a client
            # cannot invent a fifth or collapse two into one.
            "states": [state.value for state in module_models.TableState],
        },
    )


# ── Cutting board compliance ─────────────────────────────────────────────────

CUTTING_BOARD = ModuleCapability(
    module="cutting_board",
    title="Cutting Board Compliance",
    purpose=(
        "Colour-coded board readings against the ingredient category being "
        "prepared, evaluated against the site's own colour scheme and frozen "
        "with the policy version that produced the verdict."
    ),
    reason=(
        "No colour scheme is configured for this organisation, and no board or "
        "ingredient attribute is declared in the perception vocabulary. There "
        "is nothing to read and no rule to read it against."
    ),
    requirements=(
        Requirement(
            "colour_scheme",
            "The site's own colour-to-ingredient mapping, as a policy version. "
            "Colour coding is not universal — a Singapore chain, a UK caterer "
            "and a US franchise use overlapping but different schemes — so this "
            "is data a person supplies, never a default this system picks.",
        ),
        Requirement(
            "attribute_vocabulary",
            "`board_colour` and `ingredient_category` declared in the attribute "
            "registry with their permitted values and their unknown values. "
            "Until they are declared the pipeline has no legal place to put "
            "such a reading.",
        ),
        Requirement(
            "colour_under_kitchen_light",
            "Validation that board colour survives real kitchen lighting and "
            "camera white balance. A blue board under sodium light reads green, "
            "and a mis-read colour is a false accusation about a named shift.",
        ),
        Requirement(
            "ingredient_recognition",
            "A way to identify the ingredient category on the board. This is "
            "the hard half: 'raw chicken' versus 'cooked chicken' is a food-"
            "safety distinction and a very fine visual one.",
        ),
    ),
    tables=("cutting_board_policies", "board_usage_events"),
    documentation="docs/architecture/NOT_YET_CONNECTED.md#cutting-board-compliance",
)


@router.get(
    "/cutting-board",
    dependencies=[Depends(requires(Permission.VIEW_CUTTING_BOARD))],
)
async def cutting_board(access: CurrentAccess, session: DbSession) -> dict[str, Any]:
    """Board compliance capability, and the real (empty) policy set."""
    return await render(
        CUTTING_BOARD,
        session,
        organization_id=access.tenant_id,
        models=(module_models.CuttingBoardPolicy, module_models.BoardUsageEvent),
        extra={
            # Named by the server so the page renders the same four states the
            # hygiene surface does, resolved by the same rule. A board whose
            # colour could not be seen is `not_visible` and never a mismatch.
            "reading_states": ["present", "absent", "not_visible", "unknown"],
        },
    )


# ── Meal detection ───────────────────────────────────────────────────────────

MEAL_DETECTION = ModuleCapability(
    module="meal_detection",
    title="Meal Detection",
    purpose=(
        "Dishes recognised at the pass or the table, held separately from what "
        "the till says was sold. The value of the module is the difference "
        "between the two, which disappears if either is stored as the other."
    ),
    reason=(
        "No dish-recognition model is bound and no menu mapping exists. "
        "Reconciliation additionally needs a POS connector, which is its own "
        "module and is also unconfigured."
    ),
    requirements=(
        Requirement(
            "dish_dataset_and_model",
            "A labelled dataset of this operator's actual menu and a model "
            "trained on it. A general food classifier recognises 'noodles'; a "
            "menu contains four noodle dishes at different prices, and telling "
            "them apart is the entire job.",
        ),
        Requirement(
            "menu_mapping",
            "A mapping from detector classes to POS menu item identifiers, "
            "maintained as the menu changes. An unmapped detection is not a "
            "menu item and this system will not pretend otherwise.",
        ),
        Requirement(
            "pos_connector",
            "A configured POS connector for the site. Without one, every "
            "detection stays `unreconciled` — which is honest, and is why that "
            "is the default rather than `matched`.",
        ),
        Requirement(
            "camera_placement",
            "Cameras positioned over the pass or the tables. Kitchen hygiene "
            "cameras are mounted for people, not for plates, and a dish model "
            "pointed at a hygiene camera reports on whatever is in that frame.",
        ),
    ),
    tables=("dish_detections",),
    documentation="docs/architecture/NOT_YET_CONNECTED.md#meal-detection",
)


@router.get(
    "/meal-detection",
    dependencies=[Depends(requires(Permission.VIEW_MEAL_DETECTION))],
)
async def meal_detection(access: CurrentAccess, session: DbSession) -> dict[str, Any]:
    """Dish recognition capability, and the reconciliation states it would use."""
    return await render(
        MEAL_DETECTION,
        session,
        organization_id=access.tenant_id,
        models=(module_models.DishDetection,),
        extra={
            "reconciliation_states": [
                state.value for state in module_models.ReconciliationState
            ],
        },
    )


__all__ = ["router"]
