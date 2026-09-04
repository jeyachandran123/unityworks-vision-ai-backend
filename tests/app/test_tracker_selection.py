"""Production must not silently run the fallback tracker.

### The defect this pins

`app/vision/runtime.py` hard-coded:

    "tracking": {"enabled": True, "tracker_id": "tracker.iou"}

`tracker.iou` is the platform's documented **universal fallback**: no motion
model, greedy single-stage association, `handles_occlusion="none"`. It exists so
that tracking degrades in *accuracy* when a better tracker fails, never in
*availability*.

Nothing had fallen back. The composition root asked for it by name, so
`TrackingManager.is_fallback` correctly reported `False` and every health check
agreed the tracker was fine — a deliberately-constructed fallback is invisible
to a check that only asks "did we fall back?".

With no motion model the predictor asserts the box does not move, so a detection
must overlap the track's **previous** box. A person walking at ordinary pace
does not, the track fragments, and each fragment mints a fresh logical object.
That is the upstream cause of one person becoming many incidents.
"""

from __future__ import annotations

import pytest

from app.configuration.settings import Settings
from app.vision.runtime import VisionRuntime

POLICY = "config/policies/kitchen-safety.example.json"


def _document(**overrides):
    return VisionRuntime(Settings(vision_semantic_policy=POLICY, **overrides))._config_document()


class TestTheTrackerIsConfigurable:
    def test_the_tracker_is_not_hard_coded_any_more(self):
        """A site's tracking choice is a deployment fact. Naming it in source
        meant nobody could change it without a release, and nobody reviewing a
        deployment could see what it was."""
        assert _document(vision_tracker_id="tracker.bytetrack")["tracking"]["tracker_id"] == (
            "tracker.bytetrack"
        )

    def test_the_default_is_not_the_fallback_tracker(self):
        """The whole point. `tracker.iou` remains available and remains the
        automatic fallback; it is no longer what a default deployment chooses."""
        assert _document()["tracking"]["tracker_id"] != "tracker.iou"

    def test_the_default_carries_a_motion_model(self):
        """Motion prediction is the property whose absence caused fragmentation:
        it moves the predicted box to where the person is going, restoring the
        overlap that association needs at any frame rate.

        Asserted on the tracker's own config, which is the mechanism, rather
        than on a capability string that only describes it.
        """
        from vision_os.adapters.tracking import TRACKER_FACTORIES

        chosen = _document()["tracking"]["tracker_id"]
        tracker = TRACKER_FACTORIES[chosen]()
        assert tracker._config.use_motion_model is True  # noqa: SLF001

    def test_the_default_declares_it_can_survive_a_short_occlusion(self):
        """The honestly-declared capability, which is what the platform reads.
        `tracker.iou` declares `none`; a person stepping behind a counter ends
        its track outright."""
        from vision_os.adapters.tracking import TRACKER_FACTORIES

        chosen = _document()["tracking"]["tracker_id"]
        assert TRACKER_FACTORIES[chosen]().capabilities().handles_occlusion != "none"

    def test_the_fallback_is_still_selectable(self):
        """Reverting must stay a one-line change, not a rollback."""
        assert _document(vision_tracker_id="tracker.iou")["tracking"]["tracker_id"] == (
            "tracker.iou"
        )


class TestTheChoiceIsRealAndBounded:
    def test_every_selectable_name_actually_builds(self):
        """A typo in settings must fail at boot with the list of valid names,
        not at the first frame with a lookup error."""
        from vision_os.adapters.tracking import TRACKER_FACTORIES

        for name, factory in TRACKER_FACTORIES.items():
            assert factory().capabilities().tracker_id == name

    def test_the_default_needs_no_new_dependency(self):
        """All three shipped trackers are the same `GeometricTracker` class with
        a different config, so this is a configuration change and not a new
        tracking framework."""
        from vision_os.adapters.tracking import TRACKER_FACTORIES, GeometricTracker

        chosen = _document()["tracking"]["tracker_id"]
        assert isinstance(TRACKER_FACTORIES[chosen](), GeometricTracker)

    @pytest.mark.parametrize("bound", ["min_hits_to_confirm", "max_coast_frames", "max_lost_frames"])
    def test_the_lifecycle_bounds_are_unchanged_by_the_switch(self, bound):
        """`tracker_factory` passes the config-derived `LifecyclePolicy` to every
        tracker, so selecting a different one must not silently move the track
        memory bounds a site already runs on."""
        from vision_os.kernel.config.schema import TrackingSection
        from vision_os.perception.tracking.lifecycle import LifecyclePolicy

        assert getattr(TrackingSection(), bound) == getattr(LifecyclePolicy(), bound)
