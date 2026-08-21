"""M16 Configuration Manager — layering, the closed schema, and safe reload."""

from __future__ import annotations

import json

import pytest

from vision_os.adapters.configuration import (
    InMemoryConfigSource,
    InMemorySecretProvider,
    JsonFileConfigSource,
)
from vision_os.core.errors import (
    ConfigurationError,
    SecretResolutionError,
    ValidationError,
)
from vision_os.core.model.timebase import Duration
from vision_os.kernel.clock import VirtualClock
from vision_os.kernel.config import (
    ConfigLayer,
    ConfigurationManager,
    validate,
)

from ..conftest import base_config_document


class TestClosedSchema:
    """Invariant V2 — the schema is closed, and that is the point.

    Closing it turns "don't put business logic in config" from a code-review
    convention into a structural property.
    """

    def test_unknown_section_is_rejected(self) -> None:
        violations = validate({"restaurant_rules": {"max_wait_seconds": 60}})
        assert violations
        assert "unknown configuration section 'restaurant_rules'" in violations[0]
        assert "schema is closed" in violations[0]

    def test_unknown_key_within_a_known_section_is_rejected(self) -> None:
        violations = validate({"scheduler": {"kitchen_priority_boost": 3}})
        assert any("unknown key 'scheduler.kitchen_priority_boost'" in v for v in violations)

    def test_there_is_no_slot_for_a_business_threshold(self) -> None:
        """A vertical may supply geometry, profiles and labels — never a rule."""
        for attempt in (
            {"rules": [{"if": "dwell>60", "then": "alert"}]},
            {"alerts": {"enabled": True}},
            {"business": {"vertical": "restaurant"}},
            {"analytics": {"hourly_counts": True}},
        ):
            assert validate(attempt), f"{attempt} should have been rejected"

    def test_valid_document_passes(self) -> None:
        assert validate(base_config_document()) == ()

    def test_enum_values_are_validated(self) -> None:
        violations = validate({"platform": {"deployment_profile": "spaceship"}})
        assert any("deployment_profile" in v for v in violations)


class TestProvisioningFailsFast:
    def test_camera_referencing_undeclared_profile_is_rejected(self) -> None:
        document = base_config_document()
        document["cameras"][0]["profile_id"] = "does-not-exist"
        violations = validate(document)
        assert any("is not declared" in v for v in violations)
        assert any("not at first frame" in v for v in violations)

    def test_camera_referencing_undeclared_region_is_rejected(self) -> None:
        document = base_config_document()
        document["cameras"][0]["region_ids"] = ["Z99"]
        assert any("undeclared region 'Z99'" in v for v in validate(document))

    def test_duplicate_camera_ids_are_rejected(self) -> None:
        document = base_config_document(cameras=2)
        document["cameras"][1]["camera_id"] = document["cameras"][0]["camera_id"]
        assert any("duplicate camera_id" in v for v in validate(document))

    def test_invalid_source_semantics_is_rejected(self) -> None:
        document = base_config_document()
        document["cameras"][0]["source_semantics"] = "streaming"
        assert any("source_semantics" in v for v in validate(document))


class TestLayering:
    def test_later_layers_override_earlier_ones(self, clock: VirtualClock) -> None:
        manager = ConfigurationManager(
            clock=clock,
            sources={
                ConfigLayer.DEPLOYMENT: InMemoryConfigSource(
                    {**base_config_document(), "scheduler": {"global_budget_fps": 100.0}},
                    source_id="deployment",
                ),
                ConfigLayer.SITE: InMemoryConfigSource(
                    {"scheduler": {"global_budget_fps": 250.0}}, source_id="site"
                ),
            },
        )
        manager.load()
        assert manager.scheduler().global_budget_fps == 250.0

    def test_explain_names_the_layer_that_set_a_value(self, clock: VirtualClock) -> None:
        """Without this, "why is this camera at 2 fps?" is an afternoon of archaeology."""
        manager = ConfigurationManager(
            clock=clock,
            sources={
                ConfigLayer.DEPLOYMENT: InMemoryConfigSource(
                    {**base_config_document(), "scheduler": {"global_budget_fps": 100.0}},
                    source_id="deployment-defaults",
                ),
                ConfigLayer.SITE: InMemoryConfigSource(
                    {"scheduler": {"global_budget_fps": 250.0}}, source_id="site-sg-01"
                ),
            },
        )
        manager.load()
        origin = manager.explain("scheduler.global_budget_fps")
        assert origin.layer is ConfigLayer.SITE
        assert origin.source_id == "site-sg-01"
        assert origin.value == 250.0

    def test_explain_on_an_unset_path_is_an_explicit_error(
        self, config: ConfigurationManager
    ) -> None:
        with pytest.raises(ConfigurationError, match="no configured value"):
            config.explain("scheduler.nonexistent")


class TestValidationFailure:
    def test_invalid_config_fails_load_loudly(self, clock: VirtualClock) -> None:
        manager = ConfigurationManager(
            clock=clock,
            sources={ConfigLayer.SITE: InMemoryConfigSource({"nonsense": {"a": 1}})},
        )
        with pytest.raises(ValidationError) as exc:
            manager.load()
        assert exc.value.violations

    def test_failed_reload_keeps_the_current_revision(self, clock: VirtualClock) -> None:
        """Never degrade a running system for a bad reload (05_KERNEL M16)."""
        source = InMemoryConfigSource(base_config_document())
        manager = ConfigurationManager(clock=clock, sources={ConfigLayer.SITE: source})
        good_revision = manager.load()
        good_budget = manager.scheduler().global_budget_fps

        source.replace({**base_config_document(), "totally_invalid": {"x": 1}})
        with pytest.raises(ValidationError):
            manager.reload()

        assert manager.revision() == good_revision
        assert manager.scheduler().global_budget_fps == good_budget

    def test_successful_reload_reports_changed_paths(self, clock: VirtualClock) -> None:
        source = InMemoryConfigSource(base_config_document())
        manager = ConfigurationManager(clock=clock, sources={ConfigLayer.SITE: source})
        manager.load()

        document = base_config_document()
        document["scheduler"]["global_budget_fps"] = 42.0
        source.replace(document)
        result = manager.reload()

        assert "scheduler.global_budget_fps" in result.changed_paths
        assert manager.scheduler().global_budget_fps == 42.0

    def test_non_reloadable_changes_are_reported_not_ignored(
        self, clock: VirtualClock
    ) -> None:
        """Never silently ignore a change an operator made."""
        source = InMemoryConfigSource(base_config_document())
        manager = ConfigurationManager(clock=clock, sources={ConfigLayer.SITE: source})
        manager.load()

        document = base_config_document()
        document["buffer"]["slots_per_camera"] = 12
        source.replace(document)
        result = manager.reload()

        assert "buffer.slots_per_camera" in result.requires_restart


class TestRevisions:
    def test_revision_is_stable_for_identical_content(self, clock: VirtualClock) -> None:
        first = ConfigurationManager(
            clock=clock, sources={ConfigLayer.SITE: InMemoryConfigSource(base_config_document())}
        )
        second = ConfigurationManager(
            clock=clock, sources={ConfigLayer.SITE: InMemoryConfigSource(base_config_document())}
        )
        assert first.load() == second.load()

    def test_revision_changes_with_content(self, clock: VirtualClock) -> None:
        document = base_config_document()
        first = ConfigurationManager(
            clock=clock, sources={ConfigLayer.SITE: InMemoryConfigSource(document)}
        )
        revision_a = first.load()
        document["scheduler"]["global_budget_fps"] = 99.0
        second = ConfigurationManager(
            clock=clock, sources={ConfigLayer.SITE: InMemoryConfigSource(document)}
        )
        assert second.load() != revision_a

    def test_history_is_recorded(self, config: ConfigurationManager) -> None:
        assert len(config.history()) >= 1


class TestOverrides:
    def test_override_takes_precedence_and_is_attributed(self, clock: VirtualClock) -> None:
        source = InMemoryConfigSource(base_config_document())
        manager = ConfigurationManager(clock=clock, sources={ConfigLayer.SITE: source})
        manager.load()

        manager.override(
            "scheduler.global_budget_fps", 7.0, Duration.from_millis(10_000), actor="oncall"
        )
        assert manager.scheduler().global_budget_fps == 7.0
        origin = manager.explain("scheduler.global_budget_fps")
        assert origin.layer is ConfigLayer.OVERRIDE
        assert "oncall" in origin.source_id

    def test_override_expires(self, clock: VirtualClock) -> None:
        """Operational overrides are time-boxed, always."""
        source = InMemoryConfigSource(base_config_document())
        manager = ConfigurationManager(clock=clock, sources={ConfigLayer.SITE: source})
        manager.load()
        baseline = manager.scheduler().global_budget_fps

        manager.override(
            "scheduler.global_budget_fps", 7.0, Duration.from_millis(1_000), actor="oncall"
        )
        assert manager.scheduler().global_budget_fps == 7.0

        clock.advance(Duration.from_millis(2_000))
        manager.reload()
        assert manager.scheduler().global_budget_fps == baseline

    def test_clear_override_restores_the_underlying_value(self, clock: VirtualClock) -> None:
        source = InMemoryConfigSource(base_config_document())
        manager = ConfigurationManager(clock=clock, sources={ConfigLayer.SITE: source})
        manager.load()
        baseline = manager.scheduler().global_budget_fps
        manager.override(
            "scheduler.global_budget_fps", 7.0, Duration.from_millis(10_000), actor="oncall"
        )
        manager.clear_override("scheduler.global_budget_fps")
        assert manager.scheduler().global_budget_fps == baseline


class TestSecrets:
    def test_secret_is_resolved_through_the_provider(self, clock: VirtualClock) -> None:
        manager = ConfigurationManager(
            clock=clock,
            sources={ConfigLayer.SITE: InMemoryConfigSource(base_config_document())},
            secrets=InMemorySecretProvider({"cam-01-creds": "hunter2"}),
        )
        manager.load()
        assert manager.resolve_secret("cam-01-creds") == "hunter2"

    def test_none_reference_resolves_to_none(self, config: ConfigurationManager) -> None:
        assert config.resolve_secret(None) is None

    def test_unknown_reference_raises_without_leaking(self, clock: VirtualClock) -> None:
        manager = ConfigurationManager(
            clock=clock,
            sources={ConfigLayer.SITE: InMemoryConfigSource(base_config_document())},
            secrets=InMemorySecretProvider({"known": "s3cret"}),
        )
        manager.load()
        with pytest.raises(SecretResolutionError) as exc:
            manager.resolve_secret("unknown")
        assert "s3cret" not in str(exc.value)

    def test_credential_ref_without_a_provider_is_an_explicit_error(
        self, clock: VirtualClock
    ) -> None:
        manager = ConfigurationManager(
            clock=clock, sources={ConfigLayer.SITE: InMemoryConfigSource(base_config_document())}
        )
        manager.load()
        with pytest.raises(ConfigurationError, match="no secret provider"):
            manager.resolve_secret("cam-01-creds")


class TestFileSource:
    def test_missing_required_file_is_an_error(self, tmp_path) -> None:
        source = JsonFileConfigSource(tmp_path / "absent.json", required=True)
        with pytest.raises(ConfigurationError, match="not found"):
            source.load()

    def test_missing_optional_file_is_empty(self, tmp_path) -> None:
        """A missing optional layer and an unparseable one are different failures."""
        source = JsonFileConfigSource(tmp_path / "absent.json", required=False)
        assert source.load() == {}

    def test_malformed_json_names_the_line(self, tmp_path) -> None:
        path = tmp_path / "broken.json"
        path.write_text('{"platform": ', encoding="utf-8")
        with pytest.raises(ConfigurationError, match="malformed JSON"):
            JsonFileConfigSource(path).load()

    def test_valid_file_loads(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps(base_config_document()), encoding="utf-8")
        assert "platform" in JsonFileConfigSource(path).load()


class TestTypedSlices:
    def test_every_module_receives_a_typed_slice(self, config: ConfigurationManager) -> None:
        """No module reads files or environment; all receive validated objects."""
        assert config.buffer().slots_per_camera == 3
        assert config.scheduler().global_budget_fps == 1000.0
        assert config.source().stall_watchdog_ms == 1000
        assert config.health().aggregation_interval_ms == 100
        assert config.runtime().attach_stagger_ms == 0
        assert len(config.cameras()) == 1
        assert len(config.profiles()) == 1
        assert len(config.regions()) == 1

    def test_effective_before_load_is_an_explicit_error(self, clock: VirtualClock) -> None:
        manager = ConfigurationManager(clock=clock)
        with pytest.raises(ConfigurationError, match="has not been loaded"):
            manager.effective()
