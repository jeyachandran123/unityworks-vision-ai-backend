# Not yet connected

Seven product modules have a schema, a permission, a route, a page and a nav
entry. None of them has a data source, and **none of them contains any detection
logic** — no people counter, no dish classifier, no board-colour reader, no face
matcher. That is deliberate and it is the correct output of the phase that built
them: a plausible number from a system that cannot produce one teaches an
operator to trust a reading, and on the day it becomes real nobody can tell which
readings were which.

This document is the activation checklist. Each section lists the **exact
real-world inputs** required before that module can be connected for real, in the
order a deployment would obtain them. The same list is served by the module's own
API route and rendered on its page, so an operator sees it without reading this
file — the wording here and there comes from one place in the code.

Each module's schema is empty and its route answers `available: false` with the
reason. Connecting a real source is a **binding**, not a redesign.

---

## Contents

- [People Counting](#people-counting)
- [Demography](#demography)
- [Table Occupancy](#table-occupancy)
- [Cutting Board Compliance](#cutting-board-compliance)
- [Meal Detection](#meal-detection)
- [POS / ERP Integration](#pos--erp-integration)
- [Unique Patron ID](#unique-patron-id)
- [Open questions carried from earlier phases](#open-questions-carried-from-earlier-phases)

---

## People Counting

**Tables** `people_count_intervals` · **Permission** `view_people_count` ·
**Route** `GET /api/v1/modules/people-counting` · **Page** `/people-counting`

Entries and exits per zone over closed time buckets, with the coverage each
bucket was computed from.

### Required before this can be connected

1. **`counting_geometry`** — a counting line or region per camera, in normalised
   frame coordinates, with which side is "in". This is a survey decision per
   camera: a doorway line drawn for one mounting angle is wrong for another, and
   it cannot be inferred from a stream.
2. **`crossing_logic_validation`** — a validated line-crossing rule over existing
   tracks, including what happens when a track is lost and re-acquired
   mid-crossing. Without it a reconnect double-counts, and a footfall figure that
   quietly inflates is worse than none.
3. **`coverage_accounting`** — a per-bucket record of how many seconds were
   actually observed, read from camera health. The column exists
   (`observed_seconds`); the source that fills it does not.
4. **`bucket_policy`** — the bucket size and the site's operating-day boundary, so
   peak-hour analysis and branch comparison run on the same clock. Comparing a
   15-minute series against an hourly one is the usual way these reports become
   quietly wrong.

### Notes

No new Vision OS port. People are already detected and tracked; turning tracks
into directional crossings is aggregation, which belongs in `app`/`compliance`.

`observed_seconds` is nullable and is never defaulted to the full bucket.
*Unknown coverage* and *complete coverage* are different facts, and a count
without its coverage cannot be read — the same rule the dashboard applies to
every other figure.

---

## Demography

**Tables** `demography_snapshots` · **Permission** `view_demography` ·
**Route** `GET /api/v1/modules/demography` · **Page** `/demography`

Aggregate category counts per zone and time bucket. Never per person.

### Required before this can be connected

1. **`lawful_basis_and_notice`** — a PDPA lawful basis for demographic inference
   *specifically*, and signage or notice that states it. **This is the gating
   item, not the model.** Inferring age or gender from a camera is a different
   purpose from food-safety monitoring and does not inherit that purpose's basis.
2. **`k_anonymity_threshold`** — the minimum bucket size below which a count is
   suppressed rather than stored. A category containing one person, read
   alongside a shift roster, names them. This is a privacy control and a person
   must choose it; there is deliberately no default.
3. **`classifier_and_bias_evaluation`** — a classifier with a published
   evaluation across the populations it will be pointed at. Age and gender
   classifiers are known to degrade unevenly by skin tone and age, and shipping
   one unevaluated turns a bias into a reported statistic.
4. **`category_vocabulary`** — the axes and values the deployment will report,
   agreed in advance. An open vocabulary means the model's own guesses become the
   product's categories.

### The schema is the privacy control

`demography_snapshots` has **no `object_id`, no `track_id`, no
`patron_token_id`, no evidence reference** — no column that could link a row to
an individual. Aggregate-only is therefore a property of the shape rather than a
promise about the code that writes it, and making this per-person requires a
migration somebody has to write, review and sign.

The unique constraint on `(organization_id, camera_key, bucket_start,
category_axis, category_value)` enforces the same thing from the other direction:
a per-person write path collides on its second subject.

`suppressed = true` records that a true count fell below `min_bucket_size` and
was **not stored**. That is a different fact from a count of zero and must render
differently.

### Permission note

`view_demography` is **not** implied by `view_people_count`. A restaurant manager
holds the second and not the first: running a site does not confer a marketing
purpose.

---

## Table Occupancy

**Tables** `dining_tables`, `table_status_events` ·
**Permissions** `view_table_occupancy`, `manage_table_occupancy` ·
**Route** `GET /api/v1/modules/table-occupancy` · **Page** `/tables`

Each table's state over time, with turnover derived from the transitions.

### Required before this can be connected

1. **`floor_plan`** — the tables at each site: code, seats, zone, which camera
   sees them, and the region of that camera's frame they occupy. A survey per
   site; it cannot be inferred from a video stream.
2. **`state_detector`** — a detector that distinguishes vacant, occupied and
   needs-cleaning from a fixed region. A table with plates on it and a table with
   diners at it are different states, and "something is on the table" does not
   separate them.
3. **`cleaning_sla`** — the minutes after which a needs-cleaning table becomes an
   alert, per site. A threshold nobody has set cannot raise an alert, and
   inventing one would generate a work queue out of a guess.
4. **`occlusion_policy`** — what to report when a table is out of view. Both
   `not_visible` and `unknown` exist in the state enum for exactly this, and the
   policy for when each applies has not been written.

### Six states, two of which are not knowing

`vacant · occupied · needs_cleaning · out_of_service · not_visible · unknown`

A table hidden behind a standing group is `not_visible`, not vacant. Collapsing
the two would seat a party at an occupied table — the same failure mode as
collapsing `NOT_VISIBLE` into `ABSENT` for PPE, with a different victim.

### Turnover is derived, never stored as a verdict

`dwell_seconds` is a measurement. "This table has needed cleaning too long" is a
policy judgement against a threshold that does not exist yet; it belongs with the
rule engine and arrives as an incident.

---

## Cutting Board Compliance

**Tables** `cutting_board_policies`, `board_usage_events` ·
**Permissions** `view_cutting_board`, `manage_cutting_board` ·
**Route** `GET /api/v1/modules/cutting-board` · **Page** `/cutting-boards`

Board colour against the ingredient being prepared, evaluated against the site's
own colour scheme.

### Required before this can be connected

1. **`colour_scheme`** — the site's own colour-to-ingredient mapping, as a policy
   version. Colour coding is **not universal**: a Singapore chain, a UK caterer
   and a US franchise use overlapping but different schemes, and a site may run
   its own. This is data a person supplies, never a default this system picks.
2. **`attribute_vocabulary`** — `board_colour` and `ingredient_category` declared
   in the attribute registry with their permitted values and their unknown
   values. Until declared, the pipeline has no legal place to put such a reading.
3. **`colour_under_kitchen_light`** — validation that board colour survives real
   kitchen lighting and camera white balance. A blue board under sodium light
   reads green, and a mis-read colour is a false accusation about a named shift.
4. **`ingredient_recognition`** — a way to identify the ingredient category on the
   board. This is the hard half: "raw chicken" versus "cooked chicken" is a
   food-safety distinction and a very fine visual one.

### Both readings keep four states

`board_colour_state` and `ingredient_state` are
`present | absent | not_visible | unknown`, resolved by the same rule PPE uses
and rendered by the same `StateBadge`. **A `not_visible` reading can never
produce a mismatch.**

`verdict` is nullable with **no default**. `verdict = null` with
`policy_version = ""` means *no policy was in force* — emphatically not
"compliant". A default of `match` would make every unevaluated reading look
clean, which is the one thing this table must not do.

### Versioned like a ruleset

`cutting_board_policies` is append-only and versioned. An event evaluated last
March must stay explicable against the policy in force last March, exactly as
`Incident.ruleset_version` requires. A change is a new `policy_version`, never an
edit.

---

## Meal Detection

**Tables** `dish_detections` · **Permission** `view_meal_detection` ·
**Route** `GET /api/v1/modules/meal-detection` · **Page** `/meals`

Dishes recognised at the pass or the table, held **separately** from what the
till says was sold.

### Required before this can be connected

1. **`dish_dataset_and_model`** — a labelled dataset of this operator's actual
   menu, and a model trained on it. A general food classifier recognises
   "noodles"; a menu contains four noodle dishes at different prices, and telling
   them apart is the entire job.
2. **`menu_mapping`** — a mapping from detector classes to POS menu item
   identifiers, maintained as the menu changes. "Chicken rice" in a model's
   vocabulary is not the same string as a menu code, and it cannot be inferred.
3. **`pos_connector`** — see [POS / ERP Integration](#pos--erp-integration).
   Without one, every detection stays `unreconciled`.
4. **`camera_placement`** — cameras over the pass or the tables. Kitchen hygiene
   cameras are mounted for people, not for plates, and a dish model pointed at a
   hygiene camera reports on whatever is in that frame.

### Detection and sale stay separate

The value of the module is the *difference* between what a camera saw and what
the till recorded, and it disappears the moment one is stored as the other.

`reconciliation_state` defaults to `unreconciled`, never `matched`. A dish nobody
has compared against a ticket is not evidence of anything.

---

## POS / ERP Integration

**Tables** `pos_connectors`, `pos_sync_runs` ·
**Permissions** `view_pos_integration`, `manage_pos_integration` ·
**Routes** `GET /api/v1/modules/pos-integration`, `GET /api/v1/pos-connectors` ·
**Page** `/integrations/pos`

The seam between this system and a point-of-sale or ERP.

### Required before this can be connected

1. **`vendor_selection`** — which POS or ERP each site actually runs, per site.
   Chains commonly run more than one, and an adapter chosen for the group is an
   adapter wrong for half the estate.
2. **`api_documentation`** — the vendor's API documentation, specifically its
   ticket-line schema and pagination model. Reconciliation is a join on the
   vendor's own item identifiers; without their shape there is nothing to join on.
3. **`credentials`** — sandbox and production credentials, supplied as a
   `SecretProvider` reference for `PosConnector.credential_ref`. Never a value in
   the database, exactly as camera credentials are handled.
4. **`menu_mapping`** — see Meal Detection. The same mapping serves both.
5. **`rate_and_egress_agreement`** — the vendor's rate limits, and written
   agreement that this system may read sales data at all. A till belongs to the
   operator, and pulling from it is a commercial decision before it is a
   technical one.

### The port is in `app/`, not `vision_os/`

`PosGatewayPort` lives in `app/integrations/pos.py`. Vision OS's ports describe
*perception*: acquiring frames, detecting, tracking, understanding, publishing
observations. A till is not perception — it produces no frame, has no capture
instant, and cannot be scoped by camera. Reasoning about it in the platform's
vocabulary would mean inventing a `FrameRef` for a receipt.

Adding a vendor later is a sibling adapter behind this port, selected by
`PosConnector.vendor`, with no change to the port and none to any caller.

### The bound adapter refuses rather than returning nothing

`NotConfiguredPosGateway` raises `CapabilityNotConfiguredError` naming every
missing input. It does **not** return an empty ticket list, because a caller
handed `()` records a successful sync that found no sales, and every dish
detected in the window then reconciles as `unmatched` — manufacturing a
discrepancy report out of the fact that nothing was plugged in.

### Payloads are never stored

A POS payload carries ticket lines, staff identifiers, discounts and sometimes
partial card data. `PosSyncRun.payload_digest` is a hash: enough to prove two
runs saw the same data and to detect a replay, and useless for reading anybody's
lunch order. An adapter that persisted the body would turn a compliance product
into a store of retail and payment records, governed by a retention policy
written for camera observations.

---

## Unique Patron ID

**Tables** `patron_tokens` ·
**Permissions** `view_patron_id`, `manage_patron_id` ·
**Routes** `GET /api/v1/modules/patron-id`,
`GET /api/v1/modules/patron-id/gate` · **Page** `/patron-id`

> **This module reports `state: "blocked"`, not `"not_configured"`.**
> Every other module here is waiting for *work*. This one is waiting for
> *permission*, and the distinction is load-bearing.

### The legal artifact required before the first real write

**A completed Data Protection Impact Assessment for biometric re-identification,
signed off by a named Data Protection Officer, recorded as
`PATRON_ID_LEGAL_GATE_REF` and carried on every token row via
`patron_tokens.legal_gate_ref` (NOT NULL).**

Nothing less unblocks the backend. A DPIA that exists but is not recorded as that
reference does not unblock it either — the point of storing the reference on
every row is that no token can exist without naming the authority that permitted
it, and an approval that lives only in somebody's inbox cannot do that.

Alongside it, and equally required:

1. **`legal_review`** — the DPIA and DPO sign-off above.
2. **`consent_mechanism`** — a working way for a patron to **give and withdraw**
   consent, and a reference this backend can store per token. Consent that cannot
   be withdrawn is not consent. *There is deliberately no setting for this,
   because there is no mechanism: naming it as permanently missing is more honest
   than a flag somebody could set without building anything.*
3. **`site_pepper`** — a site-scoped pepper, as `PATRON_ID_PEPPER_REF` (a
   reference the `SecretProvider` resolves, never the value). Without it tokens
   are portable between sites and the module becomes a cross-site tracking
   gallery rather than a returning-visitor count.
4. **`deliberate_enablement`** — `PATRON_ID_ENABLED=true`, set by a deployment
   that has done the three above. Last on purpose: it is the switch, not the
   decision.
5. **`biometric_source`** — a bound biometric source. There is currently nothing
   that could produce a digest to hash; see below.

**Setting `PATRON_ID_ENABLED=true` alone changes nothing.**
`app/domain/patron.require_writable` refuses unconditionally and names every
outstanding item. The refusal is `PATRON_IDENTIFICATION_BLOCKED`, not a
permission error: an operator holding every permission in the product still gets
it, and granting more access cannot resolve it.

### The platform already encodes this position

No Vision OS adapter was written for this module, and that is the correct
architectural answer rather than an omission. Both ports it would need already
exist and are deliberately unbound:

- **`EmbeddingPort` (P10)** — classified **C2 · Biometric**, and its own
  docstring reads *"Declared, unbound, and unimplemented in this flow,
  deliberately."* It names threat #4, *identity linkage* — "any persistent
  mapping that links sightings across time or cameras" — as precisely what a
  retained embedding gallery is.
- **`IdentityResolverPort` (P11)** — *"Phase 2 and unimplemented."*
- **07_STATE §8.2** — UWV *"holds no persistent biometric identity, which is a
  deliberate privacy posture, not a limitation."*

Binding an adapter to either port **is** the act that requires the legal
artifact. Writing one in this phase would have quietly done the thing the DPIA is
supposed to authorise.

### Schema guarantees that hold regardless

| Guarantee | How |
|---|---|
| A raw face image cannot be stored | No binary column exists on `patron_tokens`, and no `image_ref`, `template`, `embedding` or `descriptor` column |
| A biometric template cannot be stored | `token_hash` is `String(64)` — a hex SHA-256 digest fits, a template does not |
| No token without a lawful basis | `consent_ref`, `consent_basis` and `legal_gate_ref` are NOT NULL with no default |
| Tokens do not travel between sites | The hash is peppered per site, and rotating the pepper invalidates every token — which is the intended way to end re-identification |
| Erasure stays provable | Tombstone pattern matching `EvidenceRecord`: the hash is cleared, the row survives |

A test (`tests/app/test_modules.py::test_patron_tokens_cannot_hold_a_biometric`)
asserts each of these against the model, so a later change that reintroduces a
binary column fails the suite rather than shipping quietly.

### Erasure has to happen here

The platform cannot help. `EraseScope` is *deliberately* not "by subject" —
07_STATE §8.2 again — because the platform has no subject to name. A patron's
erasure request is therefore an application-side operation against
`patron_tokens`, and it is the only place it can be honoured.

---

## Open questions carried from earlier phases

These are decisions for a person, not tasks. They are recorded here because they
have real data-protection consequences and no engineering default is the right
answer.

### A-01 · Observation retention is an interim figure

`observation_retention_days = 90` is a **placeholder pending formal legal/DPO
review**, not a decided policy.

It was set because the alternative was worse: binding `FileObservationLog` made
PPE readings outlive the process with **no expiry at all**, which is collection
without a limit. 90 days sits deliberately between the imagery clock (30) and the
compliance-record clock (365) — long enough to evidence a quarterly hygiene
review, short enough not to be a standing behavioural archive of a named shift
team.

An observation is not imagery and names nobody. It is still a record about
identifiable staff at work, and the lawful basis that justifies keeping it for a
week may not justify a quarter. **The final figure is a legal determination.**

Enforcement: `RetentionService._truncate_observations`, gated on
`RETENTION_SWEEP_ENABLED` like every other erasure.

### A-02 · Erasure does not reach the observation log

`ObservationLogPort.truncate` removes only a **time-bounded prefix** — it is what
its own contract says it is for. There is no per-subject erasure, because the
platform holds no subject to erase by. A staff erasure request that needs to
reach observations currently has no mechanism.

### A-03 · A deleted camera stops being swept

Observation-log partitions are enumerated from the `cameras` table, so every
truncation belongs to a tenant and can be audited to one. A camera row that is
deleted therefore leaves its partition unswept, and its observations outlive
their retention.

The fix belongs at the delete — erase the partition when the camera row goes —
rather than in a sweep that would have to guess which orphaned directories were
once cameras. Recorded rather than papered over.

### A-04 · Zone attribution: resolved, with a stated gap

Closed by `camera_zone_assignments` (migration `b7c41e08d5aa`). A camera's zone
is recorded as closed intervals, so moving a camera never rewrites where its past
readings happened. `/observations` returns `zone_id`, `zone_name` and
`zone_recorded` per subject, and Staff Hygiene renders the zone the camera was in
**at the time**.

**The stated gap:** intervals begin the first time a camera's zone is written
*after* this table existed. Observations older than that resolve to
`zone_recorded: false` and render as *Not recorded*. This was **not backfilled**
from `cameras.zone_id` on purpose — doing so would assert that every camera has
always been where it is now, which is the exact error the table exists to
prevent.
