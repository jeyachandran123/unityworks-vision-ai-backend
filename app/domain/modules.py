"""Seven product modules, modelled but not yet fed.

Every table here is empty, and will stay empty until somebody supplies an input
this repository does not have: a trained detector, a site's HACCP colour scheme,
a POS vendor's credentials and API documentation, a floor plan, or — for patron
identification — a completed DPIA and a lawful basis. The schemas exist now so
that connecting the real source later is a *binding*, not a redesign.

### What "modelled but not fed" deliberately does not include

There is no detection logic in this module, no placeholder count, and no code
path that writes a row from anything other than a real source. An empty table is
an honest answer; a table seeded with plausible numbers would teach an operator
to trust a figure the system cannot produce, and on the day it became real
nobody could tell which readings were which.

### The rule every event table here follows

**Where something happened is frozen onto the record at write time.**

`cameras.zone_id` and `dining_tables.zone_id` are *current state*. A record that
stored only `camera_key` and left the zone to a join would have its history
rewritten the moment a camera moved or a table was renumbered — a whole quarter
of prep-line readings silently relocating because of one dropdown. So every row
describing something that happened at a place and a time carries
`restaurant_id`, `zone_id` and `zone_name` as they were **then**, exactly as
`Incident.finding_snapshot` freezes the finding and `CameraZoneAssignment`
freezes the camera's zone.

Configuration tables — `DiningTable`, `CuttingBoardPolicy`, `PosConnector` — do
not follow the rule, because they *are* the current state. That is the point of
the distinction: one kind of row answers "where is this now", the other answers
"where was this then", and conflating them is the bug.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


# ── People counting ──────────────────────────────────────────────────────────


class PeopleCountInterval(Base):
    """Entries and exits over one closed time bucket, at one place.

    ### Why a bucket rather than a running total

    A running counter cannot be corrected, cannot be attributed to a window, and
    cannot say how much of that window it actually saw. A bucket can: it is a
    fact about a stated interval, it is idempotent to re-derive, and a gap in
    the series is visible as a gap rather than as a smaller number.

    ### `observed_seconds` is what makes the count readable

    A camera that was down for forty of a sixty-minute bucket produces a count
    that looks exactly like a quiet hour. `observed_seconds` against
    `bucket_start`/`bucket_end` is the coverage the reader needs to tell those
    apart, and it is the same discipline the dashboard applies to every other
    figure: a count without its coverage cannot be read.

    Nothing computes any of this yet. See `docs/architecture/NOT_YET_CONNECTED.md`.
    """

    __tablename__ = "people_count_intervals"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "camera_key",
            "bucket_start",
            "bucket_seconds",
            name="uq_people_count_bucket",
        ),
        Index("ix_people_count_org_time", "organization_id", "bucket_start"),
        Index("ix_people_count_zone_time", "zone_id", "bucket_start"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    #: Frozen at write time. See the module docstring.
    restaurant_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("restaurants.id", ondelete="SET NULL"), nullable=True
    )
    zone_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    zone_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    camera_key: Mapped[str] = mapped_column(String(64), nullable=False)

    bucket_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    bucket_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    #: Stored as well as derivable, so the uniqueness constraint can name it and
    #: so a fifteen-minute bucket is never silently compared with an hourly one.
    bucket_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    entries: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    exits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Occupancy at the close of the bucket. Nullable because a counting line
    #: gives flow and not occupancy, and a zero would claim an empty room.
    occupancy_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    #: How much of the bucket was actually observed. Never defaulted to the full
    #: bucket: unknown coverage and complete coverage are different facts.
    observed_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    #: Which counting geometry produced this, and which build of which detector.
    #: Frozen so a series is never silently compared across a reconfiguration.
    line_config_ref: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    detector_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    detector_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )


# ── Demography ───────────────────────────────────────────────────────────────


class DemographySnapshot(Base):
    """An **aggregate** count for one category over one bucket. Never a person.

    ### The schema is the privacy control

    This table has no `object_id`, no `track_id`, no `patron_token_id` and no
    evidence reference, and it never will. There is deliberately no column that
    could carry a link to an individual, so "aggregate-only" is a property of the
    shape rather than a promise about the code that writes it. A future
    contributor cannot accidentally make this per-person without an explicit,
    reviewable migration that adds such a column — which is exactly the review
    that should happen.

    The unique constraint enforces the same thing from the other direction: one
    row per (camera, bucket, axis, value). A per-person write path would collide
    on its second subject.

    ### Small counts are suppressed, and say so

    A bucket containing one person in a category is a re-identification risk, not
    a statistic — combined with a shift roster it names somebody. `suppressed`
    records that the true count fell below `min_bucket_size` and was **not
    stored**, which is different from a count of zero and must render differently.

    Nothing writes to this table. Under PDPA, inferring age or gender from a
    camera is a distinct purpose from food-safety monitoring and needs its own
    lawful basis and notice — see `docs/architecture/NOT_YET_CONNECTED.md`.
    """

    __tablename__ = "demography_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "camera_key",
            "bucket_start",
            "category_axis",
            "category_value",
            name="uq_demography_bucket",
        ),
        Index("ix_demography_org_time", "organization_id", "bucket_start"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    restaurant_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("restaurants.id", ondelete="SET NULL"), nullable=True
    )
    zone_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    zone_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    camera_key: Mapped[str] = mapped_column(String(64), nullable=False)

    bucket_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    bucket_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    bucket_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: `age_band`, `apparent_gender`, `group_size`. The axis is stored rather
    #: than columned so adding one is configuration, not a migration — and so
    #: no axis is privileged by having a column of its own.
    category_axis: Mapped[str] = mapped_column(String(64), nullable=False)
    category_value: Mapped[str] = mapped_column(String(64), nullable=False)
    subject_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: The k-anonymity floor in force when this row was written, frozen so a
    #: later policy change cannot make an old row look more precise than it was.
    min_bucket_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    suppressed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    observed_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    classifier_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    classifier_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    #: The consent or notice basis this collection relied on, recorded per row.
    #: Aggregate data is still personal data at the moment it is derived.
    lawful_basis_ref: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )


# ── Table occupancy ──────────────────────────────────────────────────────────


class TableState(enum.Enum):
    """What a dining table is doing. Five states, and two of them are *not knowing*.

    `NOT_VISIBLE` and `UNKNOWN` exist here for the same reason they exist for
    PPE: a table the camera could not see is not a vacant table, and rendering
    it as one would tell a host to seat a party at an occupied table.
    """

    VACANT = "vacant"
    OCCUPIED = "occupied"
    NEEDS_CLEANING = "needs_cleaning"
    OUT_OF_SERVICE = "out_of_service"
    NOT_VISIBLE = "not_visible"
    UNKNOWN = "unknown"


class DiningTable(Base):
    """A table on the floor plan. **Current state, and mutable.**

    Deliberately does not follow the freeze rule: this row answers "where is
    table 12 now", and it is `TableStatusEvent` that answers "where was table 12
    when this happened".
    """

    __tablename__ = "dining_tables"
    __table_args__ = (
        UniqueConstraint("organization_id", "table_code", name="uq_dining_table_code"),
        Index("ix_dining_tables_restaurant", "restaurant_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    restaurant_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("restaurants.id", ondelete="CASCADE"), nullable=False
    )
    zone_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("zones.id", ondelete="SET NULL"), nullable=True
    )
    #: What the staff call it — "12", "Bar 3". Not an id.
    table_code: Mapped[str] = mapped_column(String(64), nullable=False)
    seats: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Which camera watches it, and where in that camera's frame. Empty until a
    #: floor plan is supplied; a table nothing watches is still a real table.
    camera_key: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    #: JSON: the normalised polygon this table occupies in the camera's frame.
    #: Text rather than columns for the same reason `finding_snapshot` is — its
    #: shape belongs to whatever configures it.
    region: Mapped[str] = mapped_column(Text, nullable=False, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )


class TableStatusEvent(Base):
    """A table changed state, at a time, in a place. Append-only.

    `table_code`, `zone_id` and `zone_name` are copied here at write time. Renumber
    table 12 to 14 next month and this row still says what it said — which is
    what makes a turnover figure from last quarter defensible.

    ### Turnover and cleaning SLAs are derived, never stored as a verdict

    `dwell_seconds` is the measured gap since the previous state, and that is a
    measurement. "This table has needed cleaning for too long" is a *policy*
    judgement against a threshold no deployment has set yet; it belongs with the
    rule engine, arrives as an incident, and is not invented here.
    """

    __tablename__ = "table_status_events"
    __table_args__ = (
        Index("ix_table_events_org_time", "organization_id", "observed_at"),
        Index("ix_table_events_table_time", "table_id", "observed_at"),
        Index("ix_table_events_zone_time", "zone_id", "observed_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    #: SET NULL rather than CASCADE: deleting a table from the floor plan must
    #: not delete the record of what happened at it.
    table_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("dining_tables.id", ondelete="SET NULL"), nullable=True
    )
    #: Frozen at write time — the whole reason this column duplicates the one on
    #: `dining_tables`.
    table_code: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    restaurant_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("restaurants.id", ondelete="SET NULL"), nullable=True
    )
    zone_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    zone_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    camera_key: Mapped[str] = mapped_column(String(64), nullable=False, default="")

    state: Mapped[str] = mapped_column(String(32), nullable=False, default=TableState.UNKNOWN.value)
    #: What it was before, so a transition reads without a self-join.
    previous_state: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    #: Seconds spent in `previous_state`. `None` when the previous state is not
    #: known — never zero, which would read as an instant turnover.
    dwell_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: `None` rather than 0: an occupied table whose party could not be counted
    #: is not an empty one.
    party_size: Mapped[int | None] = mapped_column(Integer, nullable=True)

    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    detector_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    detector_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )


# ── Cutting board compliance ─────────────────────────────────────────────────


class CuttingBoardPolicy(Base):
    """One colour → permitted ingredient categories, in one version of a policy.

    ### Versioned and append-only, like a ruleset

    Colour coding is not universal: a chain in Singapore, a UK caterer and a US
    franchise use overlapping but different schemes, and a site may run its own.
    So the mapping is data, and it is *versioned* data — an event evaluated last
    March must be explicable against the policy in force last March, never
    against the one in force today. `Incident.ruleset_version` exists for the
    same reason and this mirrors it.

    A row is never edited. A change is a new `policy_version`.
    """

    __tablename__ = "cutting_board_policies"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "restaurant_id",
            "policy_version",
            "board_colour",
            name="uq_board_policy_colour",
        ),
        Index("ix_board_policy_org", "organization_id", "policy_version"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    #: `None` is the organisation-wide default; a row with a restaurant overrides
    #: it for that site. Sites genuinely differ, and forcing one scheme would
    #: mean a site silently evaluated against somebody else's kitchen.
    restaurant_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("restaurants.id", ondelete="CASCADE"), nullable=True
    )
    policy_version: Mapped[str] = mapped_column(String(64), nullable=False)

    #: The colour as the site names it. Not an enum: "duck egg blue" is a real
    #: answer in a real kitchen, and an enum would force a lossy mapping at the
    #: one point where the mapping is the whole subject.
    board_colour: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Comma-separated ingredient categories this colour may be used for.
    permitted_categories: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: What a person should read when this rule is broken. Stored so it renders
    #: identically six months later.
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    created_by: Mapped[str] = mapped_column(String(255), nullable=False, default="")


class BoardUsageEvent(Base):
    """A board was seen in use with an ingredient. **A reading, not a verdict.**

    ### Both readings keep their four states

    `board_colour_state` and `ingredient_state` are `present | absent |
    not_visible | unknown`, resolved by the same rule PPE uses. A board whose
    colour the camera could not make out is `not_visible`, and a `not_visible`
    reading can never produce a mismatch — accusing a chef of using the wrong
    board because the lighting was poor is the identical failure the hygiene
    surface exists to prevent, in a kitchen where it would be even harder to
    argue with.

    ### `verdict` is nullable and stays null

    A mismatch is a judgement against a `CuttingBoardPolicy`, and no deployment
    has one yet. `verdict = None` with `policy_version = ""` means *no policy was
    in force*, which is emphatically not "compliant".
    """

    __tablename__ = "board_usage_events"
    __table_args__ = (
        Index("ix_board_events_org_time", "organization_id", "observed_at"),
        Index("ix_board_events_zone_time", "zone_id", "observed_at"),
        Index("ix_board_events_verdict", "organization_id", "verdict", "observed_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    restaurant_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("restaurants.id", ondelete="SET NULL"), nullable=True
    )
    zone_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    zone_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    camera_key: Mapped[str] = mapped_column(String(64), nullable=False)

    #: The tracked object, as Vision OS identified it. Not a person.
    object_id: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    #: Raw readings, passed through exactly as the platform reported them.
    board_colour: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    board_colour_state: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    ingredient_category: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    ingredient_state: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")

    #: `match`, `mismatch`, or `None` for "no policy was in force". Never a
    #: default of `match`: an unevaluated event is not a clean one.
    verdict: Mapped[str | None] = mapped_column(String(32), nullable=True)
    #: The policy this was evaluated against, frozen. Empty means none.
    policy_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")

    board_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ingredient_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    detector_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    detector_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )


# ── Meal detection ───────────────────────────────────────────────────────────


class ReconciliationState(enum.Enum):
    """Whether a detected dish was ever matched to a POS line.

    `UNRECONCILED` is the default and the honest one: a dish nobody has compared
    against a ticket is not evidence of anything, and defaulting to `MATCHED`
    would silently manufacture a reconciliation that never happened.
    """

    UNRECONCILED = "unreconciled"
    MATCHED = "matched"
    #: Detected, compared, and no ticket line corresponds. A real finding.
    UNMATCHED = "unmatched"
    #: No POS connector covers this site, so reconciliation does not apply.
    NOT_APPLICABLE = "not_applicable"


class DishDetection(Base):
    """A dish was recognised at a place and a time. Not an order, not a sale.

    Separate from the POS record on purpose. A detection is what a camera
    believed it saw; a ticket line is what the till says was sold. The value of
    this module is the *difference* between them, and it disappears the moment
    one is stored as the other.
    """

    __tablename__ = "dish_detections"
    __table_args__ = (
        Index("ix_dish_org_time", "organization_id", "observed_at"),
        Index("ix_dish_zone_time", "zone_id", "observed_at"),
        Index("ix_dish_reconciliation", "organization_id", "reconciliation_state", "observed_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    restaurant_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("restaurants.id", ondelete="SET NULL"), nullable=True
    )
    zone_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    zone_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    camera_key: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Frozen, like the zone: which table this was plated at, as it was called
    #: then. Nullable — a pass counter has no table.
    table_code: Mapped[str] = mapped_column(String(64), nullable=False, default="")

    object_id: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    #: What the detector called it, in the detector's own vocabulary.
    dish_class: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    taxonomy_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    #: The menu item this class maps to, once somebody supplies the mapping.
    #: Empty until then — an unmapped detection is not a menu item.
    menu_item_ref: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    reconciliation_state: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ReconciliationState.UNRECONCILED.value
    )
    #: The POS ticket line, when one was matched. A reference to the vendor's
    #: identifier, never a copy of the ticket — see `PosSyncRun`.
    pos_ticket_ref: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    reconciled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    detector_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    detector_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )


# ── Unique patron identification ─────────────────────────────────────────────


class PatronToken(Base):
    """A pseudonymous handle for a returning patron. **Never a face.**

    ### This table is the safety control, and it is built out of what it lacks

    There is no `image_ref`, no `template`, no `embedding`, no `descriptor`, and
    no binary column of any kind — not one place a raw biometric could be put.
    `token_hash` is `String(64)`: exactly wide enough for a hex SHA-256 digest
    and far too narrow for a face template, so an attempt to widen the design
    into storing one requires a migration somebody has to write, review and
    sign. A test asserts these properties, so a later change that reintroduces a
    binary column fails the suite rather than shipping quietly.

    That matters because the platform's own posture is unambiguous. Vision OS
    declares `EmbeddingPort` (P10) as **C2 · Biometric** and leaves it *"declared,
    unbound, and unimplemented … deliberately"*, and `IdentityResolverPort` (P11)
    is likewise unimplemented; 07_STATE §8.2 states UWV *"holds no persistent
    biometric identity, which is a deliberate privacy posture, not a limitation."*
    Re-identification would be the first thing in this product to contradict
    that, so it must be the hardest thing to do by accident.

    ### `consent_ref` and `legal_gate_ref` are NOT NULL with no default

    A row cannot exist without naming the consent that permits it and the
    approval that authorised the capability. Not a check performed by a service
    that a future caller might bypass — a column the database refuses to leave
    empty.

    ### The erasure fields are the point, not an afterthought

    Re-identification is precisely the processing a person is most likely to
    object to, and PDPA/GDPR erasure has to be answerable. The tombstone pattern
    matches `EvidenceRecord`: the hash is cleared, the row survives, and the
    deletion is provable afterwards. The platform cannot help here — its
    `EraseScope` is *deliberately* not "by subject", because it has no subject to
    name — so this is the only place an erasure can be honoured.

    **Nothing writes to this table, and the write path refuses unconditionally
    until the legal gate is satisfied.** See `app/domain/patron.py`.
    """

    __tablename__ = "patron_tokens"
    __table_args__ = (
        UniqueConstraint("organization_id", "token_hash", name="uq_patron_token"),
        Index("ix_patron_org_seen", "organization_id", "last_seen_at"),
        Index("ix_patron_consent", "organization_id", "consent_expires_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    #: Frozen at write time, like every other located record here.
    restaurant_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("restaurants.id", ondelete="SET NULL"), nullable=True
    )

    #: Hex SHA-256 of a site-scoped pepper concatenated with a template digest.
    #: 64 characters, and the width is load-bearing: it cannot hold a template,
    #: and the peppering means a token from one site does not match the same
    #: person at another — which is what stops this becoming a cross-site
    #: tracking gallery.
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    hash_algorithm: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    #: Which pepper epoch produced the hash. Rotating the pepper invalidates
    #: every token, which is the intended way to end re-identification.
    key_version: Mapped[str] = mapped_column(String(32), nullable=False, default="")

    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    visit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: **NOT NULL, no default.** A token that cannot name its consent has no
    #: lawful basis, and the schema refuses to hold one.
    consent_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    consent_basis: Mapped[str] = mapped_column(String(64), nullable=False)
    consent_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    #: The DPIA / DPO sign-off that authorised the capability at all. Also NOT
    #: NULL: the approval is part of the record, not a file in somebody's inbox.
    legal_gate_ref: Mapped[str] = mapped_column(String(255), nullable=False)

    erasure_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    erased_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    erasure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )


# ── POS / ERP integration ────────────────────────────────────────────────────


class PosConnector(Base):
    """A configured link to a point-of-sale or ERP system. Current state.

    **No credential lives here.** `credential_ref` is a pointer the
    `SecretProvider` resolves at connect time, exactly as `Camera.credential_ref`
    is, so a database dump is not a credential dump. That is not a detail
    inherited by accident — a POS credential reaches sales and often payment
    data, and it is the most valuable secret this application would ever hold.

    Created **inactive**, like a camera: registering a connector and letting it
    exchange data are two decisions.
    """

    __tablename__ = "pos_connectors"
    __table_args__ = (
        UniqueConstraint("organization_id", "connector_key", name="uq_pos_connector_key"),
        Index("ix_pos_connectors_org", "organization_id", "is_active"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    #: `None` when one connector serves the whole organisation.
    restaurant_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("restaurants.id", ondelete="SET NULL"), nullable=True
    )
    connector_key: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Which vendor's adapter serves this row. The adapter is chosen by this
    #: value, so a new vendor is a sibling adapter and a new row — never a
    #: change to the port or to any caller.
    vendor: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    display_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    base_url: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    #: `env:POS_TOKEN`, `file:/run/secrets/pos`. A REFERENCE, never a secret.
    credential_ref: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    #: Comma-separated capability names the adapter declares it supports.
    capabilities: Mapped[str] = mapped_column(Text, nullable=False, default="")

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_success_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_error_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: The last failure, in words an operator can act on. Never a stack trace and
    #: never a response body — a POS error body can echo a ticket.
    last_error: Mapped[str] = mapped_column(String(512), nullable=False, default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )


class PosSyncRun(Base):
    """One exchange with a POS system: when, which way, and what happened.

    ### The payload is not stored, and that is deliberate

    A POS payload carries ticket lines, staff identifiers, discounts and often
    partial card data. Keeping one to "help debugging" would quietly turn a
    compliance product into a store of retail and payment records, under a
    retention policy written for camera observations. `payload_digest` is a hash:
    enough to prove two runs saw the same data and to detect a replay, and
    useless for reading anybody's lunch order.
    """

    __tablename__ = "pos_sync_runs"
    __table_args__ = (
        Index("ix_pos_runs_org_time", "organization_id", "started_at"),
        Index("ix_pos_runs_connector_time", "connector_id", "started_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    connector_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("pos_connectors.id", ondelete="SET NULL"), nullable=True
    )
    #: Frozen, like every other historical attribution here: which vendor and
    #: which site this run belonged to when it ran.
    vendor: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    restaurant_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    #: `pull` reads from the POS; `push` sends to it. Named rather than inferred
    #: from counts, because a push that sent nothing is not a pull.
    direction: Mapped[str] = mapped_column(String(16), nullable=False, default="pull")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: `succeeded`, `failed`, or `refused` — the last meaning the adapter
    #: declined to run at all, which is what an unconfigured connector does.
    outcome: Mapped[str] = mapped_column(String(32), nullable=False, default="refused")

    records_in: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_out: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_kind: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    error_detail: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    #: A hash of what was exchanged. Never the payload itself.
    payload_digest: Mapped[str] = mapped_column(String(128), nullable=False, default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )


__all__ = [
    "BoardUsageEvent",
    "CuttingBoardPolicy",
    "DemographySnapshot",
    "DiningTable",
    "DishDetection",
    "PatronToken",
    "PeopleCountInterval",
    "PosConnector",
    "PosSyncRun",
    "ReconciliationState",
    "TableState",
    "TableStatusEvent",
]
