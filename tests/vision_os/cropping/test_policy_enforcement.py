"""A policy's declared subject scope must be enforced, not merely recorded.

### The defect this pins

`config/policies/kitchen-safety.example.json` declares:

    "scope": {
      "object_classes": ["person"],
      "lifecycle": ["active", "occluded"],
      "min_confidence": 0.4
    }

The author's intent is unambiguous: do not spend a model call on a provisional
object, and do not spend one on an object whose identity is barely believed.

`SubjectFilter` carried all three fields. `DemandRegistry.matching` consulted
exactly one of them — `matches_class` — and no accessor for the other two
existed anywhere in the repository. They were parsed, carried and dropped.

The consequence is the false-positive path. `VisualObject.is_present`
deliberately admits `PROVISIONAL`, so a single-frame detector false positive was
minted as an object, included in `RegistryUpdate.present`, matched by a demand
that checked only its class, cropped, sent to the model, answered
`head_covering: none`, and raised a `high` severity compliance incident.

Every one of those layers was doing what its own contract said. The two gates
written to stop it were inert.
"""

from __future__ import annotations

import pytest

from vision_os.core.model.demand import SubjectFilter


class TestSubjectFilterAnswersAboutLifecycle:
    def test_an_unnarrowed_filter_covers_every_state(self):
        """Empty means *every* state, matching `covers_camera`'s
        empty-means-all convention. A policy that does not narrow is site-wide,
        not silent — the opposite reading would disable analysis everywhere the
        moment this field started being read."""
        assert SubjectFilter().matches_lifecycle("provisional") is True

    def test_an_unsupplied_state_is_not_read_as_a_state_name(self):
        """The two emptinesses are different questions. Treating "the caller did
        not say" as a lifecycle name would make every older three-argument
        caller match nothing and stop analysis outright — the exact class of
        silent shutdown this whole repair exists to remove."""
        assert SubjectFilter(lifecycle=("active",)).matches_lifecycle("") is True
        assert SubjectFilter(lifecycle=("active",)).matches_lifecycle(None) is True

    def test_a_declared_scope_excludes_states_outside_it(self):
        filter_ = SubjectFilter(lifecycle=("active", "occluded"))
        assert filter_.matches_lifecycle("active") is True
        assert filter_.matches_lifecycle("occluded") is True
        assert filter_.matches_lifecycle("provisional") is False
        assert filter_.matches_lifecycle("dormant") is False


class TestSubjectFilterAnswersAboutConfidence:
    def test_no_floor_admits_everything(self):
        assert SubjectFilter().matches_confidence(0.01) is True

    def test_a_floor_excludes_what_falls_below_it(self):
        filter_ = SubjectFilter(min_confidence=0.4)
        assert filter_.matches_confidence(0.39) is False
        assert filter_.matches_confidence(0.40) is True
        assert filter_.matches_confidence(0.95) is True

    def test_an_unknown_confidence_is_not_treated_as_a_failure(self):
        """Refusing on absence would turn a missing input into a policy
        decision nobody wrote, silently excluding every object whose confidence
        a caller did not supply."""
        assert SubjectFilter(min_confidence=0.4).matches_confidence(None) is True


class TestTheRegistryActuallyConsultsThem:
    """The half that was missing. The methods above are only worth having if
    `matching` calls them."""

    @pytest.fixture
    def registry_with_policy(self, demand_registry, clock):
        """A demand carrying the same scope `kitchen-safety.example.json`
        declares — the real shape, not a contrived one."""
        from dataclasses import replace

        from tests.vision_os.cropping.conftest import make_demand

        base = make_demand(demand_id="kitchen-safety@2.1.0")
        demand = replace(
            base,
            subject_filter=SubjectFilter(
                class_ids=base.subject_filter.class_ids,
                lifecycle=("active", "occluded"),
                min_confidence=0.4,
            ),
        )
        demand_registry.register(demand, now=clock.now())
        return demand_registry

    def test_a_provisional_object_is_not_matched(self, registry_with_policy):
        """The false-positive path, closed. A one-frame detection becomes a
        PROVISIONAL object, and a PROVISIONAL object is not what this policy
        asked about."""
        matched = registry_with_policy.matching(
            camera_id="cam-01",
            class_id="person",
            lifecycle="provisional",
            confidence=0.9,
        )
        assert matched == ()

    def test_a_low_confidence_object_is_not_matched(self, registry_with_policy):
        matched = registry_with_policy.matching(
            camera_id="cam-01",
            class_id="person",
            lifecycle="active",
            confidence=0.2,
        )
        assert matched == ()

    def test_an_eligible_object_still_matches(self, registry_with_policy):
        """The repair must not close the door on the objects the policy is
        actually about."""
        matched = registry_with_policy.matching(
            camera_id="cam-01",
            class_id="person",
            lifecycle="active",
            confidence=0.9,
        )
        assert len(matched) == 1

    def test_an_occluded_object_still_matches(self, registry_with_policy):
        """`occluded` is in scope: believed present, merely unmeasured. A person
        behind a counter is still the person the policy is about."""
        matched = registry_with_policy.matching(
            camera_id="cam-01",
            class_id="person",
            lifecycle="occluded",
            confidence=0.9,
        )
        assert len(matched) == 1

    def test_omitting_the_context_still_matches(self, registry_with_policy):
        """Backward compatibility. A caller that cannot say what state an object
        is in must not have that silence read as a policy decision — the old
        three-argument call keeps working exactly as it did."""
        matched = registry_with_policy.matching(camera_id="cam-01", class_id="person")
        assert len(matched) == 1

    def test_required_attributes_applies_the_same_gate(self, registry_with_policy):
        """`required_attributes` is what the trigger policy actually calls, so
        the gate has to hold there too — enforcing it only in `matching` would
        leave the real path open."""
        wanted = registry_with_policy.required_attributes(
            camera_id="cam-01",
            class_id="person",
            lifecycle="provisional",
            confidence=0.9,
        )
        assert wanted == {}
