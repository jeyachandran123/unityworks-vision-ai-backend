"""The VisualObject model — what a canonical object may and may not claim.

Most of these assert *refusals*. An object that can be constructed incoherent is
one every consumer must defend against; making the state unconstructible moves
the check from N readers to one constructor.

The field set is checked against 02_VOM section 10.6 directly, because the brief
forbids renaming, removing, simplifying, merging in ``Track`` fields, or
inventing new ones — and a test that pins the schema is the only thing that keeps
that true through later edits.
"""

from __future__ import annotations

import pytest

from vision_os.core.model.confidence import Confidence, ConfidenceSemantics
from vision_os.core.model.ids import (
    AttributeKey,
    BindingId,
    ClassId,
    ObjectId,
)
from vision_os.core.model.space import Box
from vision_os.core.model.visual_object import (
    OBJECT_SCHEMA_VERSION,
    Attribute,
    BindingMethod,
    ClassObservation,
    IdentityAssertion,
    LifecycleState,
    RevisionReason,
    TrackBinding,
    VisualObject,
)

from ..conftest import CAMERA, PERSON, SITE, TENANT, at, spatial, track_id

OBJECT_ID = ObjectId("01JB0000000000000000000001")
OTHER_ID = ObjectId("01JB0000000000000000000002")


def make_object(**overrides) -> VisualObject:
    defaults = dict(
        object_id=OBJECT_ID,
        tenant_id=TENANT,
        site_id=SITE,
        camera_id=CAMERA,
        class_id=PERSON,
        confidence=Confidence.uncalibrated(0.9, ConfidenceSemantics.IDENTITY),
        lifecycle=LifecycleState.ACTIVE,
        class_history=(),
        track_bindings=(),
        current_spatial=spatial(Box(0.3, 0.4, 0.5, 0.8)),
        spatial_history=(),
        attributes={},
        first_seen=at(0),
        last_seen=at(5),
        last_confirmed=at(5),
        observation_count=5,
        provenance=_provenance(),
    )
    defaults.update(overrides)
    return VisualObject(**defaults)


def _provenance():
    from vision_os.core.model.ids import ConfigRevision, ModuleId
    from vision_os.core.model.provenance import Provenance

    return Provenance(
        producer_module=ModuleId("object_registry"),
        producer_version="1.0.0",
        config_revision=ConfigRevision("test"),
    )


class TestSchemaMatchesTheArchitecture:
    """02_VOM section 10.6, field for field."""

    def test_every_documented_field_is_present(self) -> None:
        fields = set(VisualObject.__dataclass_fields__)
        for required in (
            "object_id",
            "class_id",
            "class_history",
            "lifecycle",
            "track_bindings",
            "current_spatial",
            "spatial_history",
            "attributes",
            "first_seen",
            "last_seen",
            "last_confirmed",
            "observation_count",
        ):
            assert required in fields, f"02_VOM section 10.6 requires '{required}'"

    def test_the_substrate_is_present(self) -> None:
        """Every object kind carries the same base (02_VOM section 3)."""
        fields = set(VisualObject.__dataclass_fields__)
        for required in ("tenant_id", "site_id", "provenance", "confidence",
                         "lineage", "labels", "schema_version"):
            assert required in fields

    def test_no_track_field_leaked_in(self) -> None:
        """Track and Object are separate kinds; merging them is forbidden."""
        fields = set(VisualObject.__dataclass_fields__)
        for forbidden in (
            "track_id", "tracker_epoch", "coast_frames", "hit_count",
            "age_frames", "break_reason", "motion", "motion_state",
            "measurement_basis",
        ):
            assert forbidden not in fields, (
                f"'{forbidden}' belongs to Track (M6); merging it into "
                f"VisualObject collapses the boundary V10 exists to hold"
            )

    def test_no_projection_field_leaked_in(self) -> None:
        """``ObjectState`` (07_STATE section 3.1) is the L6 projection, not this.

        Its extra fields — ``regions``, ``trajectory``, ``last_observation`` —
        belong to Vision State, which is Flow 7.
        """
        fields = set(VisualObject.__dataclass_fields__)
        for forbidden in (
            "regions", "trajectory", "last_observation", "provenance_summary",
            "staleness", "identity",
        ):
            assert forbidden not in fields

    def test_no_business_field_exists(self) -> None:
        fields = set(VisualObject.__dataclass_fields__)
        for forbidden in (
            "person_id", "name", "role", "is_employee", "is_customer",
            "alert", "violation", "embedding", "face_id",
        ):
            assert forbidden not in fields

    def test_schema_version_is_declared(self) -> None:
        assert make_object().schema_version == OBJECT_SCHEMA_VERSION


class TestLifecycleStates:
    def test_the_state_set_is_exactly_the_architecture_s_seven(self) -> None:
        assert {s.value for s in LifecycleState} == {
            "provisional",
            "active",
            "occluded",
            "dormant",
            "departed",
            "merged_into",
            "expired",
        }

    def test_terminal_states_are_merged_and_expired(self) -> None:
        for state in LifecycleState:
            expected = state in (LifecycleState.MERGED_INTO, LifecycleState.EXPIRED)
            assert state.is_terminal is expected

    def test_present_means_believed_in_the_observable_area(self) -> None:
        """``occluded`` counts — the object is believed present, merely unmeasured.

        ``dormant`` does not: it is retained for re-entry, not asserted present.
        """
        assert LifecycleState.PROVISIONAL.is_present
        assert LifecycleState.ACTIVE.is_present
        assert LifecycleState.OCCLUDED.is_present
        assert not LifecycleState.DORMANT.is_present
        assert not LifecycleState.DEPARTED.is_present

    def test_measurable_is_narrower_than_present(self) -> None:
        assert LifecycleState.ACTIVE.is_measurable
        assert not LifecycleState.OCCLUDED.is_measurable


class TestConstruction:
    def test_a_valid_object_constructs(self) -> None:
        assert make_object().lifecycle is LifecycleState.ACTIVE

    def test_identity_confidence_is_required(self) -> None:
        """An object asserts identity; a track asserts association (02_VOM 7)."""
        with pytest.raises(ValueError, match="IDENTITY"):
            make_object(
                confidence=Confidence.uncalibrated(
                    0.9, ConfidenceSemantics.ASSOCIATION
                )
            )

    def test_merged_into_requires_the_merged_state(self) -> None:
        with pytest.raises(ValueError, match="V5"):
            make_object(merged_into=OTHER_ID)

    def test_the_merged_state_requires_a_target(self) -> None:
        """A merged object without a target is unresolvable history."""
        with pytest.raises(ValueError, match="V5"):
            make_object(lifecycle=LifecycleState.MERGED_INTO)

    def test_an_object_cannot_merge_into_itself(self) -> None:
        with pytest.raises(ValueError, match="into itself"):
            make_object(lifecycle=LifecycleState.MERGED_INTO, merged_into=OBJECT_ID)

    def test_negative_observation_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            make_object(observation_count=-1)

    def test_last_seen_cannot_precede_first_seen(self) -> None:
        with pytest.raises(ValueError, match="precede"):
            make_object(first_seen=at(9), last_seen=at(1), last_confirmed=at(1))

    def test_last_confirmed_cannot_follow_last_seen(self) -> None:
        """A measurement cannot be newer than the most recent update."""
        with pytest.raises(ValueError, match="last_confirmed"):
            make_object(last_seen=at(3), last_confirmed=at(9))


class TestMeasuredVersusBelieved:
    """V8 at object scale — 02_VOM section 10.6 notes."""

    def test_staleness_measures_from_the_last_confirmed_sighting(self) -> None:
        obj = make_object(last_seen=at(10), last_confirmed=at(4))
        assert obj.staleness(at(10)).millis == pytest.approx(6 * 200)

    def test_staleness_never_goes_negative(self) -> None:
        assert make_object(last_confirmed=at(9), last_seen=at(9)).staleness(at(1)).ns == 0

    def test_a_coasting_object_reports_growing_staleness(self) -> None:
        obj = make_object(
            lifecycle=LifecycleState.OCCLUDED, last_seen=at(10), last_confirmed=at(4)
        )
        assert obj.staleness(at(20)) > obj.staleness(at(12))

    def test_age_spans_the_whole_lifetime(self) -> None:
        assert make_object(first_seen=at(0)).age(at(10)).millis == pytest.approx(2_000)


class TestBindings:
    def test_a_binding_requires_identity_confidence(self) -> None:
        with pytest.raises(ValueError, match="IDENTITY"):
            TrackBinding(
                binding_id=BindingId("b1"),
                track_id=track_id(),
                bound_from=at(0),
                confidence=Confidence.uncalibrated(
                    0.9, ConfidenceSemantics.ASSOCIATION
                ),
            )

    def test_a_binding_cannot_close_before_it_opened(self) -> None:
        with pytest.raises(ValueError, match="close before"):
            TrackBinding(
                binding_id=BindingId("b1"),
                track_id=track_id(),
                bound_from=at(5),
                bound_to=at(1),
            )

    def test_an_open_binding_is_the_bound_track(self) -> None:
        binding = TrackBinding(
            binding_id=BindingId("b1"), track_id=track_id(7), bound_from=at(0)
        )
        obj = make_object(track_bindings=(binding,))
        assert obj.open_binding is binding
        assert obj.bound_track == track_id(7)

    def test_a_closed_binding_leaves_the_object_unbound(self) -> None:
        binding = TrackBinding(
            binding_id=BindingId("b1"),
            track_id=track_id(7),
            bound_from=at(0),
            bound_to=at(5),
        )
        obj = make_object(track_bindings=(binding,))
        assert obj.open_binding is None
        assert obj.bound_track is None

    def test_a_superseded_binding_is_not_the_open_one(self) -> None:
        """A revision does not delete the original (V5); it supersedes it."""
        old = TrackBinding(
            binding_id=BindingId("b1"),
            track_id=track_id(7),
            bound_from=at(0),
            superseded_by=BindingId("b2"),
        )
        new = TrackBinding(
            binding_id=BindingId("b2"), track_id=track_id(8), bound_from=at(3)
        )
        obj = make_object(track_bindings=(old, new))
        assert obj.open_binding is new
        assert old in obj.track_bindings, "the superseded binding must be retained"

    def test_bindings_for_a_track_are_findable(self) -> None:
        first = TrackBinding(
            binding_id=BindingId("b1"),
            track_id=track_id(7),
            bound_from=at(0),
            bound_to=at(2),
        )
        second = TrackBinding(
            binding_id=BindingId("b2"), track_id=track_id(7), bound_from=at(4)
        )
        obj = make_object(track_bindings=(first, second))
        assert len(obj.bindings_for(track_id(7))) == 2
        assert obj.bindings_for(track_id(9)) == ()

    def test_binding_duration_uses_now_while_open(self) -> None:
        binding = TrackBinding(
            binding_id=BindingId("b1"), track_id=track_id(), bound_from=at(0)
        )
        assert binding.duration(at(5)).millis == pytest.approx(1_000)

    def test_binding_methods_cover_the_documented_strategies(self) -> None:
        values = {m.value for m in BindingMethod}
        for required in (
            "first_sight", "track_continuity", "spatio_temporal",
            "epoch_rebind", "appearance", "resolver",
        ):
            assert required in values


class TestAttributes:
    def _attribute(self, **overrides) -> Attribute:
        defaults = dict(
            key=AttributeKey("posture"),
            schema_version="1.0.0",
            value="standing",
            confidence=Confidence.uncalibrated(0.8, ConfidenceSemantics.ATTRIBUTE),
            observed_at=at(5),
            producer=_provenance(),
        )
        defaults.update(overrides)
        return Attribute(**defaults)

    def test_an_attribute_requires_attribute_confidence(self) -> None:
        with pytest.raises(ValueError, match="ATTRIBUTE"):
            self._attribute(
                confidence=Confidence.uncalibrated(0.8, ConfidenceSemantics.IDENTITY)
            )

    def test_self_reported_confidence_is_accepted(self) -> None:
        """A model's own certainty is weaker but still a legitimate basis."""
        assert self._attribute(
            confidence=Confidence.uncalibrated(
                0.8, ConfidenceSemantics.SELF_REPORTED
            )
        )

    def test_an_attribute_without_a_horizon_never_goes_stale(self) -> None:
        assert not self._attribute(valid_until=None).is_stale(at(10_000))

    def test_an_attribute_past_its_horizon_is_stale(self) -> None:
        attribute = self._attribute(observed_at=at(5), valid_until=at(10))
        assert not attribute.is_stale(at(9))
        assert attribute.is_stale(at(11))

    def test_stale_attributes_are_listed_on_the_object(self) -> None:
        fresh = self._attribute(key=AttributeKey("posture"), valid_until=at(100))
        stale = self._attribute(key=AttributeKey("headwear_present"), valid_until=at(6))
        obj = make_object(
            attributes={fresh.key: fresh, stale.key: stale}
        )
        assert obj.stale_attributes(at(50)) == (AttributeKey("headwear_present"),)

    def test_attribute_lookup_returns_none_when_absent(self) -> None:
        assert make_object().attribute(AttributeKey("posture")) is None


class TestClassResolution:
    def test_class_history_records_what_was_seen(self) -> None:
        history = (
            ClassObservation(
                class_id=PERSON,
                observed_at=at(0),
                confidence=Confidence.uncalibrated(
                    0.9, ConfidenceSemantics.CLASSIFICATION
                ),
            ),
            ClassObservation(
                class_id=ClassId("person.child"),
                observed_at=at(1),
                confidence=Confidence.uncalibrated(
                    0.6, ConfidenceSemantics.CLASSIFICATION
                ),
            ),
        )
        obj = make_object(class_history=history)
        assert len(obj.class_history) == 2
        assert obj.class_history[0].class_id == PERSON

    def test_is_a_matches_hierarchically(self) -> None:
        obj = make_object(class_id=ClassId("vehicle.forklift"))
        assert obj.is_a(ClassId("vehicle"))
        assert obj.is_a(ClassId("vehicle.forklift"))
        assert not obj.is_a(ClassId("person"))

    def test_is_a_does_not_match_a_bare_prefix(self) -> None:
        assert not make_object(class_id=ClassId("vehicles")).is_a(ClassId("vehicle"))


class TestIdentityAssertion:
    def _assertion(self, **overrides) -> IdentityAssertion:
        defaults = dict(
            binding_id=BindingId("b1"),
            object_id=OBJECT_ID,
            track_id=track_id(),
            asserted_at=at(3),
            confidence=Confidence.uncalibrated(0.8, ConfidenceSemantics.IDENTITY),
            method=BindingMethod.SPATIO_TEMPORAL,
        )
        defaults.update(overrides)
        return IdentityAssertion(**defaults)

    def test_it_requires_identity_confidence(self) -> None:
        with pytest.raises(ValueError, match="IDENTITY"):
            self._assertion(
                confidence=Confidence.uncalibrated(
                    0.8, ConfidenceSemantics.ASSOCIATION
                )
            )

    def test_alternatives_make_it_ambiguous(self) -> None:
        """Section M7: never guess silently — publish the candidates."""
        assertion = self._assertion(alternatives=((OTHER_ID, 0.71), (OBJECT_ID, 0.68)))
        assert assertion.is_ambiguous
        assert len(assertion.alternatives) == 2

    def test_no_alternatives_means_unambiguous(self) -> None:
        assert not self._assertion().is_ambiguous

    def test_supersedes_marks_a_revision(self) -> None:
        assertion = self._assertion(
            supersedes=BindingId("b0"), reason=RevisionReason.BETTER_EVIDENCE
        )
        assert assertion.is_revision
        assert assertion.reason is RevisionReason.BETTER_EVIDENCE


class TestImmutability:
    def test_an_object_cannot_be_mutated(self) -> None:
        """V5 and the canonical-ownership rule: consumers read, never write."""
        obj = make_object()
        with pytest.raises(AttributeError):
            obj.class_id = ClassId("vehicle")  # type: ignore[misc]

    def test_lifecycle_cannot_be_reassigned(self) -> None:
        obj = make_object()
        with pytest.raises(AttributeError):
            obj.lifecycle = LifecycleState.EXPIRED  # type: ignore[misc]

    def test_a_binding_cannot_be_mutated(self) -> None:
        binding = TrackBinding(
            binding_id=BindingId("b1"), track_id=track_id(), bound_from=at(0)
        )
        with pytest.raises(AttributeError):
            binding.bound_to = at(5)  # type: ignore[misc]

    def test_two_objects_with_the_same_content_are_equal(self) -> None:
        assert make_object() == make_object()
