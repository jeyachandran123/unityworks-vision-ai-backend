"""Observation tests — the envelope itself (02_VOM §11).

The brief's *Observation* category. The envelope is the platform's permanent
record, so these assertions are about what a record must always carry and what it
must never be able to become.

An observation that validates today and cannot be interpreted in five years is a
failure the type system should have prevented, which is why so much of this file
is about construction refusing rather than about behaviour.
"""

from __future__ import annotations

import dataclasses

import pytest

from vision_os.core.errors import ObservationError
from vision_os.core.model.ids import ObjectId, ObservationId
from vision_os.core.model.observation import (
    CoverageWindow,
    LifecycleTransition,
    MeasurementBasis,
    ObservabilityReason,
    ObservabilityStatus,
    Observation,
    ObservationType,
    ViolationKind,
    coverage_gap,
)
from vision_os.core.model.visual_object import LifecycleState

from .conftest import (
    CAMERA,
    OTHER_CAMERA,
    POSTURE,
    at,
    attribute,
    builder_provenance,
    context,
    frame_ref,
    make_object,
    presence_of,
    spatial,
)


class TestTheEnvelopeIsComplete:
    """§M11: an observation carries everything needed to interpret it later."""

    def test_a_presence_observation_carries_the_full_envelope(self, builder) -> None:
        observation = presence_of(builder)

        assert observation.observation_type is ObservationType.PRESENCE
        assert observation.camera_id == CAMERA
        assert observation.object_id == ObjectId("obj-1")
        assert observation.t_capture == at(3)
        assert observation.provenance.producer_module == "observation_builder"
        assert observation.taxonomy_version
        assert observation.measurement_basis is MeasurementBasis.MEASURED

    def test_capture_and_publish_times_are_both_present_and_distinct_fields(
        self, builder
    ) -> None:
        """V11. When a thing happened and when we said so are different facts.

        A record with only one of them cannot distinguish a slow pipeline from a
        late event, and every latency question the platform will ever be asked
        needs both.
        """
        observation = presence_of(builder)
        assert observation.t_capture is not None
        assert observation.t_published is not None
        assert "t_capture" in Observation.__dataclass_fields__
        assert "t_published" in Observation.__dataclass_fields__

    def test_clock_quality_travels_with_the_timestamp(self, builder) -> None:
        """A timestamp without its quality invites false precision (V11)."""
        observation = presence_of(builder)
        assert observation.clock_quality.typical_uncertainty_ms > 0
        assert observation.t_capture_unc.ns >= 0

    def test_a_poor_clock_floors_the_uncertainty(self, clock, metrics, bus,
                                                 attribute_registry, taxonomy) -> None:
        """A source claiming 0ms uncertainty on an unsynchronised clock is wrong.

        §M11 stamps what is true, not what was claimed: the builder floors the
        declared uncertainty at the clock class's own typical value, so a
        producer cannot assert a precision its clock cannot support.
        """
        from vision_os.adapters.synthesis import AlwaysPublish
        from vision_os.core.model.timebase import ClockQuality
        from vision_os.synthesis import CeilingGate, ObservationBuilder

        from .conftest import synthesis_config

        builder = ObservationBuilder(
            clock=clock,
            metrics=metrics,
            events=bus,
            config=synthesis_config(suppression_policy="suppression.always"),
            gate=CeilingGate(attribute_registry, taxonomy),
            provenance=builder_provenance(),
            suppression_policy=AlwaysPublish(),
        )
        observation = builder.build_presence(
            make_object(),
            context(clock_quality=ClockQuality.UNKNOWN, uncertainty_ms=0.0),
        )
        assert observation is not None
        assert observation.t_capture_unc.millis >= ClockQuality.UNKNOWN.typical_uncertainty_ms


class TestImmutability:
    """V5. *"Never edit. Never mutate. Never overwrite."*"""

    def test_an_observation_cannot_be_mutated(self, builder) -> None:
        observation = presence_of(builder)
        with pytest.raises(dataclasses.FrozenInstanceError):
            observation.t_capture = at(99)  # type: ignore[misc]

    def test_an_observation_has_no_dict_to_smuggle_a_field_into(self, builder) -> None:
        """Slotted, so a caller cannot attach a field the schema never had.

        Frozen alone stops assignment to declared fields; ``__dict__`` would
        still accept a new one, and a record carrying an undeclared attribute is
        exactly what the ceiling exists to prevent.
        """
        observation = presence_of(builder)
        assert not hasattr(observation, "__dict__")
        assert Observation.__slots__

    def test_correction_is_a_new_observation_that_names_the_old_one(
        self, loud_builder
    ) -> None:
        """07_STATE §2: correction without mutation.

        The superseding record is *new*; the superseded one is untouched and
        still readable, which is what makes the log auditable rather than merely
        current.
        """
        original = presence_of(loud_builder)
        corrected = dataclasses.replace(
            original,
            observation_id=ObservationId("obs-corrected"),
            supersedes=original.observation_id,
        )
        assert corrected.supersedes == original.observation_id
        assert original.supersedes is None
        assert corrected.observation_id != original.observation_id

    def test_an_observation_cannot_supersede_itself(self, builder) -> None:
        observation = presence_of(builder)
        with pytest.raises(ValueError, match="supersede itself"):
            dataclasses.replace(observation, supersedes=observation.observation_id)

    def test_lineage_may_not_contain_the_observations_own_id(self, builder) -> None:
        """A cycle in lineage is a graph that cannot be walked to an origin."""
        observation = presence_of(builder)
        with pytest.raises(ValueError):
            dataclasses.replace(observation, lineage=(observation.observation_id,))


class TestProvenanceIsMandatory:
    """V4. An observation nobody can explain is worse than no observation."""

    def test_provenance_names_a_module_and_a_version(self, builder) -> None:
        provenance = presence_of(builder).provenance
        assert provenance.producer_module
        assert provenance.producer_version
        assert provenance.config_revision

    def test_an_envelope_without_provenance_is_refused(self, builder) -> None:
        """And refused with a message that names the reason, not an AttributeError.

        This test originally passed against a ``NoneType has no attribute``
        crash, which is a refusal only by accident. The check is explicit now.
        """
        observation = presence_of(builder)
        with pytest.raises(ValueError, match="requires provenance"):
            dataclasses.replace(observation, provenance=None)

    def test_the_camera_must_match_the_frame_it_came_from(self, builder) -> None:
        """A frame from cam-02 in an observation labelled cam-01 is unreplayable."""
        observation = presence_of(builder)
        with pytest.raises(ValueError, match="camera"):
            dataclasses.replace(observation, frame_ref=frame_ref(3, camera=OTHER_CAMERA))


class TestTypePayloadsAreEnforced:
    """02_VOM §11.2 assigns different content to different types."""

    def test_a_coverage_observation_needs_a_coverage_window(self, builder) -> None:
        observation = builder.build_coverage(
            context(),
            status=ObservabilityStatus.DEGRADED,
            reason=ObservabilityReason.SCENE_OBSCURED,
            since=at(1),
            effective_rate=0.4,
        )
        assert observation.coverage is not None
        assert observation.coverage.status is ObservabilityStatus.DEGRADED
        with pytest.raises(ValueError):
            dataclasses.replace(observation, coverage=None)

    def test_a_lifecycle_observation_needs_a_transition(self, builder) -> None:
        transition = LifecycleTransition(
            previous=LifecycleState.ACTIVE,
            current=LifecycleState.OCCLUDED,
            trigger="occlusion",
        )
        observation = builder.build_lifecycle(make_object(), transition, context())
        assert observation is not None
        assert observation.lifecycle_transition == transition
        with pytest.raises(ValueError):
            dataclasses.replace(observation, lifecycle_transition=None)

    def test_a_transition_to_the_same_state_is_not_a_transition(self) -> None:
        """A record asserting a change that did not happen is a false fact."""
        with pytest.raises(ValueError, match="must record a change"):
            LifecycleTransition(
                previous=LifecycleState.ACTIVE, current=LifecycleState.ACTIVE
            )

    def test_an_attribute_observation_carries_attribute_confidence(
        self, builder
    ) -> None:
        """02_VOM §7.1: confidence is meaningless without its semantics.

        An attribute claim scored with detection-presence confidence would be
        comparing two different probabilities under one name.
        """
        from .conftest import understanding

        published = builder.build_attribute(make_object(), understanding(), context())
        assert len(published) == 1
        for attr in published[0].attributes:
            assert attr.confidence.semantics.value in ("attribute", "self_reported")


class TestCoverageIsNeverOptional:
    """02_VOM §11.2: *"not optional"*. V8's teeth."""

    def test_a_coverage_observation_is_published_even_when_nothing_changed(
        self, builder
    ) -> None:
        first = builder.build_coverage(
            context(seq=1),
            status=ObservabilityStatus.BLIND,
            reason=ObservabilityReason.STREAM_DISCONNECTED,
            since=at(1),
            effective_rate=0.0,
        )
        second = builder.build_coverage(
            context(seq=2),
            status=ObservabilityStatus.BLIND,
            reason=ObservabilityReason.STREAM_DISCONNECTED,
            since=at(1),
            effective_rate=0.0,
        )
        assert first is not None and second is not None
        assert first.observation_id != second.observation_id

    def test_build_coverage_returns_an_observation_not_an_optional(self) -> None:
        """The signature itself is the guarantee.

        Every other builder returns ``Observation | None``. This one cannot,
        because a policy that could suppress it would be the platform deciding
        its own blindness was not worth mentioning.
        """
        import inspect

        from vision_os.synthesis import ObservationBuilder

        signature = inspect.signature(ObservationBuilder.build_coverage)
        assert "None" not in str(signature.return_annotation)

    def test_a_restart_is_a_declarable_blindness_reason(self) -> None:
        """10_RELIABILITY: the gap across a restart is a real gap.

        A platform that resumed silently would have a hole in its record that
        nothing in the record mentions.
        """
        assert ObservabilityReason.RESTART in tuple(ObservabilityReason)

    def test_observable_fraction_distinguishes_empty_from_blind(self) -> None:
        """07_STATE §7.3. Without this number, *"the region was empty"* and
        *"the camera was disconnected"* are the same answer.
        """
        blind = CoverageWindow(
            status=ObservabilityStatus.BLIND,
            reason=ObservabilityReason.STREAM_DISCONNECTED,
            since=at(10),
            effective_rate=0.0,
            until=at(20),
        )
        fraction = coverage_gap((blind,), since=at(0), until=at(20))
        assert fraction == pytest.approx(0.5)

    def test_a_fully_observed_window_reports_complete_coverage(self) -> None:
        observing = CoverageWindow(
            status=ObservabilityStatus.OBSERVING,
            reason=ObservabilityReason.NORMAL,
            since=at(0),
            effective_rate=1.0,
            until=at(20),
        )
        assert coverage_gap((observing,), since=at(0), until=at(20)) == 1.0


class TestUnknownIsNotFalse:
    """*"Unknown is always better than fabricated."*"""

    def test_a_failed_understanding_produces_no_attribute_observation(
        self, builder
    ) -> None:
        """`NO_ATTRIBUTES` is an outcome, not a fact.

        Publishing an empty attribute observation would assert the platform
        looked and found nothing, when it may simply have failed to look.
        """
        from vision_os.core.model.understanding import UnderstandingOutcome

        from .conftest import understanding

        for outcome in (
            UnderstandingOutcome.UNAVAILABLE,
            UnderstandingOutcome.TIMED_OUT,
            UnderstandingOutcome.NO_ATTRIBUTES,
        ):
            result = understanding(outcome=outcome, attributes=())
            assert builder.build_attribute(make_object(), result, context()) == []

    def test_a_missing_attribute_is_absent_rather_than_negative(self, builder) -> None:
        """The headwear attribute is *not asserted false* — it is not there.

        A consumer can tell "we did not determine this" from "we determined it is
        false", which is the distinction V8 exists to preserve.
        """
        from .conftest import HEADWEAR, understanding

        published = builder.build_attribute(
            make_object(), understanding(attributes=(attribute(POSTURE, "standing"),)),
            context(),
        )
        keys = {a.key for a in published[0].attributes}
        assert POSTURE in keys
        assert HEADWEAR not in keys


class TestViolationKindSplitsTwoFailures:
    """§M11 names two failures with two different responses."""

    def test_an_unregistered_attribute_does_not_reject_the_envelope(self) -> None:
        assert not ViolationKind.UNREGISTERED_ATTRIBUTE.rejects_the_envelope

    def test_a_missing_envelope_field_rejects_the_whole_observation(self) -> None:
        for kind in (
            ViolationKind.MISSING_PROVENANCE,
            ViolationKind.MISSING_TIMING,
            ViolationKind.MISSING_EVIDENCE,
            ViolationKind.MISSING_MEASUREMENT_BASIS,
            ViolationKind.TAXONOMY_VERSION_MISMATCH,
            ViolationKind.UNREGISTERED_CLASS,
        ):
            assert kind.rejects_the_envelope, f"{kind} must reject the envelope"

    def test_every_violation_kind_declares_which_response_it_gets(self) -> None:
        """No kind may be ambiguous — the property is total, not partial."""
        for kind in ViolationKind:
            assert isinstance(kind.rejects_the_envelope, bool)


class TestSpatialAndMeasurementBasis:
    def test_a_measured_observation_says_so(self, builder) -> None:
        observation = presence_of(builder)
        assert observation.measurement_basis is MeasurementBasis.MEASURED

    def test_an_inferred_position_is_labelled_inferred(self, loud_builder) -> None:
        """02_VOM: a believed position and a measured one are different facts.

        A consumer averaging the two without knowing which is which would be
        treating a guess as a measurement.
        """
        observation = loud_builder.build_spatial(
            make_object(position=spatial(0.5, 0.5)),
            context(),
            basis=MeasurementBasis.PREDICTED,
        )
        assert observation is not None
        assert observation.measurement_basis is MeasurementBasis.PREDICTED

    def test_normalized_coordinates_need_no_calibration(self, builder) -> None:
        """V9: an uncalibrated camera degrades to normalized space, it does not fail."""
        observation = presence_of(builder)
        assert observation.spatial is not None
        assert observation.spatial.calibration_id is None


class TestObservationErrorsAreTyped:
    def test_observation_error_carries_structured_context(self) -> None:
        """An operator reading a log needs the values, not a formatted sentence."""
        error = ObservationError("bad policy", requested="suppression.nope")
        assert error.context["requested"] == "suppression.nope"
