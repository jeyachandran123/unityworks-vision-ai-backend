"""product module scaffold

Ten tables for seven product modules that have no data source yet.

Nothing writes to any of them. They exist now so that connecting a real
detector, a real POS vendor or a real consent mechanism later is a *binding*
rather than a redesign — and so the shape of each record is reviewed while it is
still cheap to change, rather than under delivery pressure with rows in it.

### Two decisions this migration encodes

**Location is frozen on every event row.** Each table describing something that
happened at a place carries `restaurant_id`, `zone_id` and `zone_name` of its
own rather than leaving them to a join through `cameras` or `dining_tables`.
Those are current-state tables; joining them would rewrite where a past event
happened the moment a camera moves or a table is renumbered. Same reasoning as
`camera_zone_assignments` in the preceding revision.

**`patron_tokens` is deliberately narrow.** `token_hash` is `String(64)` — a hex
SHA-256 digest fits, a face template does not — and there is no binary column
anywhere in the table. `consent_ref` and `legal_gate_ref` are NOT NULL with no
server default, so the database itself refuses a token that cannot name the
consent permitting it and the approval authorising the capability. Widening any
of this needs a migration somebody has to write, review and sign, which is the
point.

Additive: ten new tables, no existing table altered, no data touched.

Revision ID: c9d5f21ab340
Revises: b7c41e08d5aa
Create Date: 2026-09-02 09:20:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = 'c9d5f21ab340'
down_revision: str | None = 'b7c41e08d5aa'
branch_labels: str | None = None
depends_on: str | None = None


def _org_fk() -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE')


def _restaurant_fk_set_null() -> sa.ForeignKeyConstraint:
    """SET NULL, never CASCADE.

    Deleting a site must not delete the record of what happened there — the same
    rule `incidents.restaurant_id` follows.
    """
    return sa.ForeignKeyConstraint(['restaurant_id'], ['restaurants.id'], ondelete='SET NULL')


def upgrade() -> None:
    # ── People counting ──────────────────────────────────────────────────────
    op.create_table(
        'people_count_intervals',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('organization_id', sa.String(length=64), nullable=False),
        sa.Column('restaurant_id', sa.String(length=64), nullable=True),
        sa.Column('zone_id', sa.String(length=64), nullable=True),
        sa.Column('zone_name', sa.String(length=255), nullable=False, server_default=''),
        sa.Column('camera_key', sa.String(length=64), nullable=False),
        sa.Column('bucket_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('bucket_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('bucket_seconds', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('entries', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('exits', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('occupancy_end', sa.Integer(), nullable=True),
        sa.Column('observed_seconds', sa.Integer(), nullable=True),
        sa.Column('line_config_ref', sa.String(length=255), nullable=False, server_default=''),
        sa.Column('detector_id', sa.String(length=128), nullable=False, server_default=''),
        sa.Column('detector_version', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        _org_fk(),
        _restaurant_fk_set_null(),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'organization_id',
            'camera_key',
            'bucket_start',
            'bucket_seconds',
            name='uq_people_count_bucket',
        ),
    )
    op.create_index(
        'ix_people_count_org_time', 'people_count_intervals', ['organization_id', 'bucket_start']
    )
    op.create_index(
        'ix_people_count_zone_time', 'people_count_intervals', ['zone_id', 'bucket_start']
    )

    # ── Demography ───────────────────────────────────────────────────────────
    #
    # No object_id, no track_id, no evidence reference. There is deliberately no
    # column here that could link a row to an individual, so aggregate-only is a
    # property of the schema rather than a promise about the writer.
    op.create_table(
        'demography_snapshots',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('organization_id', sa.String(length=64), nullable=False),
        sa.Column('restaurant_id', sa.String(length=64), nullable=True),
        sa.Column('zone_id', sa.String(length=64), nullable=True),
        sa.Column('zone_name', sa.String(length=255), nullable=False, server_default=''),
        sa.Column('camera_key', sa.String(length=64), nullable=False),
        sa.Column('bucket_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('bucket_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('bucket_seconds', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('category_axis', sa.String(length=64), nullable=False),
        sa.Column('category_value', sa.String(length=64), nullable=False),
        sa.Column('subject_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('min_bucket_size', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('suppressed', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('observed_seconds', sa.Integer(), nullable=True),
        sa.Column('classifier_id', sa.String(length=128), nullable=False, server_default=''),
        sa.Column('classifier_version', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('lawful_basis_ref', sa.String(length=255), nullable=False, server_default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        _org_fk(),
        _restaurant_fk_set_null(),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'organization_id',
            'camera_key',
            'bucket_start',
            'category_axis',
            'category_value',
            name='uq_demography_bucket',
        ),
    )
    op.create_index(
        'ix_demography_org_time', 'demography_snapshots', ['organization_id', 'bucket_start']
    )

    # ── Table occupancy ──────────────────────────────────────────────────────
    op.create_table(
        'dining_tables',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('organization_id', sa.String(length=64), nullable=False),
        sa.Column('restaurant_id', sa.String(length=64), nullable=False),
        sa.Column('zone_id', sa.String(length=64), nullable=True),
        sa.Column('table_code', sa.String(length=64), nullable=False),
        sa.Column('seats', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('camera_key', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('region', sa.Text(), nullable=False, server_default=''),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        _org_fk(),
        sa.ForeignKeyConstraint(['restaurant_id'], ['restaurants.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['zone_id'], ['zones.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('organization_id', 'table_code', name='uq_dining_table_code'),
    )
    op.create_index('ix_dining_tables_restaurant', 'dining_tables', ['restaurant_id'])

    op.create_table(
        'table_status_events',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('organization_id', sa.String(length=64), nullable=False),
        sa.Column('table_id', sa.String(length=64), nullable=True),
        sa.Column('table_code', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('restaurant_id', sa.String(length=64), nullable=True),
        sa.Column('zone_id', sa.String(length=64), nullable=True),
        sa.Column('zone_name', sa.String(length=255), nullable=False, server_default=''),
        sa.Column('camera_key', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('state', sa.String(length=32), nullable=False, server_default='unknown'),
        sa.Column('previous_state', sa.String(length=32), nullable=False, server_default=''),
        sa.Column('observed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('dwell_seconds', sa.Integer(), nullable=True),
        sa.Column('party_size', sa.Integer(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('detector_id', sa.String(length=128), nullable=False, server_default=''),
        sa.Column('detector_version', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        _org_fk(),
        # SET NULL: removing a table from the floor plan must not delete the
        # record of what happened at it.
        sa.ForeignKeyConstraint(['table_id'], ['dining_tables.id'], ondelete='SET NULL'),
        _restaurant_fk_set_null(),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_table_events_org_time', 'table_status_events', ['organization_id', 'observed_at']
    )
    op.create_index(
        'ix_table_events_table_time', 'table_status_events', ['table_id', 'observed_at']
    )
    op.create_index('ix_table_events_zone_time', 'table_status_events', ['zone_id', 'observed_at'])

    # ── Cutting board compliance ─────────────────────────────────────────────
    op.create_table(
        'cutting_board_policies',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('organization_id', sa.String(length=64), nullable=False),
        sa.Column('restaurant_id', sa.String(length=64), nullable=True),
        sa.Column('policy_version', sa.String(length=64), nullable=False),
        sa.Column('board_colour', sa.String(length=64), nullable=False),
        sa.Column('permitted_categories', sa.Text(), nullable=False, server_default=''),
        sa.Column('summary', sa.Text(), nullable=False, server_default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_by', sa.String(length=255), nullable=False, server_default=''),
        _org_fk(),
        sa.ForeignKeyConstraint(['restaurant_id'], ['restaurants.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'organization_id',
            'restaurant_id',
            'policy_version',
            'board_colour',
            name='uq_board_policy_colour',
        ),
    )
    op.create_index(
        'ix_board_policy_org', 'cutting_board_policies', ['organization_id', 'policy_version']
    )

    op.create_table(
        'board_usage_events',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('organization_id', sa.String(length=64), nullable=False),
        sa.Column('restaurant_id', sa.String(length=64), nullable=True),
        sa.Column('zone_id', sa.String(length=64), nullable=True),
        sa.Column('zone_name', sa.String(length=255), nullable=False, server_default=''),
        sa.Column('camera_key', sa.String(length=64), nullable=False),
        sa.Column('object_id', sa.String(length=255), nullable=False, server_default=''),
        sa.Column('observed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('board_colour', sa.String(length=64), nullable=False, server_default=''),
        # Four states, kept four. `not_visible` never becomes a mismatch.
        sa.Column('board_colour_state', sa.String(length=32), nullable=False, server_default='unknown'),
        sa.Column('ingredient_category', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('ingredient_state', sa.String(length=32), nullable=False, server_default='unknown'),
        # Nullable, and no default: an unevaluated event is not a clean one.
        sa.Column('verdict', sa.String(length=32), nullable=True),
        sa.Column('policy_version', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('board_confidence', sa.Float(), nullable=True),
        sa.Column('ingredient_confidence', sa.Float(), nullable=True),
        sa.Column('detector_id', sa.String(length=128), nullable=False, server_default=''),
        sa.Column('detector_version', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        _org_fk(),
        _restaurant_fk_set_null(),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_board_events_org_time', 'board_usage_events', ['organization_id', 'observed_at']
    )
    op.create_index('ix_board_events_zone_time', 'board_usage_events', ['zone_id', 'observed_at'])
    op.create_index(
        'ix_board_events_verdict',
        'board_usage_events',
        ['organization_id', 'verdict', 'observed_at'],
    )

    # ── Meal detection ───────────────────────────────────────────────────────
    op.create_table(
        'dish_detections',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('organization_id', sa.String(length=64), nullable=False),
        sa.Column('restaurant_id', sa.String(length=64), nullable=True),
        sa.Column('zone_id', sa.String(length=64), nullable=True),
        sa.Column('zone_name', sa.String(length=255), nullable=False, server_default=''),
        sa.Column('camera_key', sa.String(length=64), nullable=False),
        sa.Column('table_code', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('object_id', sa.String(length=255), nullable=False, server_default=''),
        sa.Column('observed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('dish_class', sa.String(length=128), nullable=False, server_default=''),
        sa.Column('taxonomy_version', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('menu_item_ref', sa.String(length=255), nullable=False, server_default=''),
        sa.Column('confidence', sa.Float(), nullable=True),
        # `unreconciled`, never `matched`: a dish nobody compared against a
        # ticket is not evidence of anything.
        sa.Column(
            'reconciliation_state', sa.String(length=32), nullable=False, server_default='unreconciled'
        ),
        sa.Column('pos_ticket_ref', sa.String(length=255), nullable=False, server_default=''),
        sa.Column('reconciled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('detector_id', sa.String(length=128), nullable=False, server_default=''),
        sa.Column('detector_version', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        _org_fk(),
        _restaurant_fk_set_null(),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_dish_org_time', 'dish_detections', ['organization_id', 'observed_at'])
    op.create_index('ix_dish_zone_time', 'dish_detections', ['zone_id', 'observed_at'])
    op.create_index(
        'ix_dish_reconciliation',
        'dish_detections',
        ['organization_id', 'reconciliation_state', 'observed_at'],
    )

    # ── Unique patron identification ─────────────────────────────────────────
    #
    # `token_hash` is String(64): a hex SHA-256 digest fits and a biometric
    # template does not. There is no binary column in this table, and
    # `consent_ref` / `legal_gate_ref` are NOT NULL with no server default, so
    # the database refuses a token that names neither its consent nor the
    # approval that authorised the capability.
    op.create_table(
        'patron_tokens',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('organization_id', sa.String(length=64), nullable=False),
        sa.Column('restaurant_id', sa.String(length=64), nullable=True),
        sa.Column('token_hash', sa.String(length=64), nullable=False),
        sa.Column('hash_algorithm', sa.String(length=32), nullable=False, server_default=''),
        sa.Column('key_version', sa.String(length=32), nullable=False, server_default=''),
        sa.Column('first_seen_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('visit_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('consent_ref', sa.String(length=255), nullable=False),
        sa.Column('consent_basis', sa.String(length=64), nullable=False),
        sa.Column('consent_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('legal_gate_ref', sa.String(length=255), nullable=False),
        sa.Column('erasure_requested_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('erased_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('erasure_reason', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        _org_fk(),
        _restaurant_fk_set_null(),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('organization_id', 'token_hash', name='uq_patron_token'),
    )
    op.create_index('ix_patron_org_seen', 'patron_tokens', ['organization_id', 'last_seen_at'])
    op.create_index('ix_patron_consent', 'patron_tokens', ['organization_id', 'consent_expires_at'])

    # ── POS / ERP integration ────────────────────────────────────────────────
    op.create_table(
        'pos_connectors',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('organization_id', sa.String(length=64), nullable=False),
        sa.Column('restaurant_id', sa.String(length=64), nullable=True),
        sa.Column('connector_key', sa.String(length=64), nullable=False),
        sa.Column('vendor', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('display_name', sa.String(length=255), nullable=False, server_default=''),
        sa.Column('base_url', sa.String(length=512), nullable=False, server_default=''),
        # A reference the SecretProvider resolves, never a secret.
        sa.Column('credential_ref', sa.String(length=512), nullable=False, server_default=''),
        sa.Column('capabilities', sa.Text(), nullable=False, server_default=''),
        # Inactive on creation, like a camera: registering and connecting are
        # two decisions.
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('last_success_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_error_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_error', sa.String(length=512), nullable=False, server_default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        _org_fk(),
        _restaurant_fk_set_null(),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('organization_id', 'connector_key', name='uq_pos_connector_key'),
    )
    op.create_index('ix_pos_connectors_org', 'pos_connectors', ['organization_id', 'is_active'])

    op.create_table(
        'pos_sync_runs',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('organization_id', sa.String(length=64), nullable=False),
        sa.Column('connector_id', sa.String(length=64), nullable=True),
        sa.Column('vendor', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('restaurant_id', sa.String(length=64), nullable=True),
        sa.Column('direction', sa.String(length=16), nullable=False, server_default='pull'),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('outcome', sa.String(length=32), nullable=False, server_default='refused'),
        sa.Column('records_in', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('records_out', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('error_kind', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('error_detail', sa.String(length=512), nullable=False, server_default=''),
        # A hash of what was exchanged. Never the payload: a POS payload carries
        # ticket lines, staff ids and sometimes partial card data.
        sa.Column('payload_digest', sa.String(length=128), nullable=False, server_default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        _org_fk(),
        sa.ForeignKeyConstraint(['connector_id'], ['pos_connectors.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_pos_runs_org_time', 'pos_sync_runs', ['organization_id', 'started_at'])
    op.create_index(
        'ix_pos_runs_connector_time', 'pos_sync_runs', ['connector_id', 'started_at']
    )


def downgrade() -> None:
    for index, table in (
        ('ix_pos_runs_connector_time', 'pos_sync_runs'),
        ('ix_pos_runs_org_time', 'pos_sync_runs'),
        ('ix_pos_connectors_org', 'pos_connectors'),
        ('ix_patron_consent', 'patron_tokens'),
        ('ix_patron_org_seen', 'patron_tokens'),
        ('ix_dish_reconciliation', 'dish_detections'),
        ('ix_dish_zone_time', 'dish_detections'),
        ('ix_dish_org_time', 'dish_detections'),
        ('ix_board_events_verdict', 'board_usage_events'),
        ('ix_board_events_zone_time', 'board_usage_events'),
        ('ix_board_events_org_time', 'board_usage_events'),
        ('ix_board_policy_org', 'cutting_board_policies'),
        ('ix_table_events_zone_time', 'table_status_events'),
        ('ix_table_events_table_time', 'table_status_events'),
        ('ix_table_events_org_time', 'table_status_events'),
        ('ix_dining_tables_restaurant', 'dining_tables'),
        ('ix_demography_org_time', 'demography_snapshots'),
        ('ix_people_count_zone_time', 'people_count_intervals'),
        ('ix_people_count_org_time', 'people_count_intervals'),
    ):
        op.drop_index(index, table_name=table)

    for table in (
        'pos_sync_runs',
        'pos_connectors',
        'patron_tokens',
        'dish_detections',
        'board_usage_events',
        'cutting_board_policies',
        'table_status_events',
        'dining_tables',
        'demography_snapshots',
        'people_count_intervals',
    ):
        op.drop_table(table)
