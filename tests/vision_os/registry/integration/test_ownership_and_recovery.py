"""Canonical ownership, durable state, versioning, and recovery.

Flow 4 introduces the platform's first **canonical owner**: the registry is the
only writer of Vision Objects, and every future module reads. That is
structurally enforced rather than documented, and the first class below is what
enforces it.

``07_STATE`` section 9.3 states the recovery contract precisely:

> *Object identity — **Preserved** — durable in the registry. Tracks — **Lost** —
> new `TrackerEpoch`; re-binding to objects happens with explicitly reduced
> confidence.*

Both halves are tested: what survives, and what must not pretend to.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from vision_os.adapters.registry import (
    SNAPSHOT_FORMAT_VERSION,
    FileObjectStore,
    InMemoryObjectStore,
)
from vision_os.conformance import OBJECT_STORE_KIT
from vision_os.core.errors import ObjectStoreError
from vision_os.core.model.ids import CameraId
from vision_os.core.model.space import Box
from vision_os.core.model.visual_object import (
    BindingMethod,
    LifecycleState,
)
from vision_os.core.ports.registry import (
    ObjectStorePort,
    PartitionSnapshot,
)
from vision_os.kernel.metrics import MetricName
from vision_os.kernel.plugins.manifest import BINDABLE_PORTS, PortCatalogue

from ..conftest import CAMERA, OTHER_CAMERA, SITE, age, at, coast, drive, make_track, make_update


class TestCanonicalOwnership:
    """Single writer, multiple readers — structurally, not by convention."""

    def test_a_published_object_cannot_be_mutated(self, registry) -> None:
        drive(registry, 5)
        obj = registry.active(CAMERA)[0]
        with pytest.raises(AttributeError):
            obj.lifecycle = LifecycleState.EXPIRED  # type: ignore[misc]

    def test_a_reader_snapshot_does_not_drift(self, registry) -> None:
        """A consumer holding an object holds a snapshot that cannot move."""
        drive(registry, 5)
        snapshot = registry.active(CAMERA)[0]
        drive(registry, 5, start=10)
        assert snapshot.observation_count == 5
        assert registry.active(CAMERA)[0].observation_count == 10

    def test_two_readers_get_independent_snapshots(self, registry) -> None:
        drive(registry, 5)
        first = registry.active(CAMERA)[0]
        second = registry.active(CAMERA)[0]
        assert first == second
        assert first is not second or first == second

    def test_only_the_registry_mints_object_ids(self) -> None:
        """01_LAYERED section 8: exactly one module may mint an identity.

        Three constructions are not minting and are excluded by name:

        * ``adapters/registry`` **decodes** a persisted id — reconstruction of an
          identity the registry already minted, not creation of a new one.
        * ``adapters/synthesis/decode.py`` decodes the same way, one layer up: a
          P20 log record carries the id the registry minted, and reading it back
          off disk reconstructs it. Named as a single file rather than a
          directory so the rest of the synthesis adapters stay guarded.
        * ``adapters/persistence/evidence.py`` reconstructs the ``object_id`` an
          evidence record was indexed by, so 07_STATE §8.2's *"erasure by
          object"* survives a restart. Again reconstruction, not creation — the
          id was minted by the registry and written down.
        * ``conformance`` builds **fixtures** to exercise a port; those ids never
          enter a pipeline.

        Everything else creating an ``ObjectId`` is diffusing identity, which
        01_LAYERED names as how ID chaos begins.
        """
        import ast
        from pathlib import Path as P

        import vision_os as pkg

        root = P(pkg.__file__).parent
        allowed_prefixes = (
            "perception/registry/",
            "adapters/registry/",
            "adapters/synthesis/decode.py",
            "adapters/persistence/evidence.py",
            "conformance/",
        )
        offenders = []
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            relative = str(path.relative_to(root)).replace("\\", "/")
            if relative.startswith(allowed_prefixes):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "ObjectId"
                ):
                    offenders.append(f"{relative}:{node.lineno}")
        assert not offenders, (
            "only the Object Registry may mint an ObjectId:\n" + "\n".join(offenders)
        )

    def test_the_minting_call_site_is_singular(self) -> None:
        """Inside the registry, ``new_ulid`` for an object appears once.

        Two call sites would be two policies, and the second would drift.
        """
        import ast
        from pathlib import Path as P

        import vision_os.perception.registry as pkg

        sites = []
        for path in P(pkg.__file__).parent.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "ObjectId"
                ):
                    sites.append(f"{path.name}:{node.lineno}")
        assert len(sites) <= 2, (
            f"ObjectId is constructed at {len(sites)} sites inside the registry "
            f"({', '.join(sites)}); minting should be one policy"
        )

    def test_the_registry_exposes_no_write_path_for_consumers(self, registry) -> None:
        """A consumer can read and can request a correction; it cannot write."""
        public = {
            name
            for name in dir(registry)
            if not name.startswith("_") and callable(getattr(registry, name))
        }
        assert public == {
            "ingest",
            "get",
            "resolve",
            "active",
            "objects",
            "bind",
            "merge",
            "split",
            "apply_attribute",
            "expire_stale",
            # Advancing horizons and *reporting* what that changed. `ingest`
            # already publishes its lifecycle changes; the scheduled pass did
            # not, so nothing downstream could learn that a departed object had
            # aged out — it stayed a live subject. Read-and-age, like
            # `expire_stale` beside it; still no write path for a consumer.
            "sweep",
            "restore",
            "set_regions",
            "partition_stats",
            "region_tracker",
            "health",
        }

    def test_partitions_are_independent_writers(self, registry) -> None:
        """07_STATE section 4.1: the camera is the partition, one writer each."""
        drive(registry, 5, camera=CAMERA)
        drive(registry, 5, camera=OTHER_CAMERA)
        first = registry.active(CAMERA)[0]
        second = registry.active(OTHER_CAMERA)[0]
        assert first.object_id != second.object_id
        assert first.camera_id != second.camera_id


class TestVersioning:
    def test_a_partition_version_advances_on_write(self, registry) -> None:
        drive(registry, 3)
        first = registry.partition_stats(CAMERA).version
        drive(registry, 3, start=10)
        assert registry.partition_stats(CAMERA).version > first

    def test_reads_do_not_advance_the_version(self, registry) -> None:
        drive(registry, 3)
        version = registry.partition_stats(CAMERA).version
        registry.active(CAMERA)
        registry.objects(CAMERA)
        assert registry.partition_stats(CAMERA).version == version

    def test_an_unknown_partition_has_no_stats(self, registry) -> None:
        assert registry.partition_stats(CameraId("never-seen")) is None

    def test_a_snapshot_carries_its_version(self, registry, object_store) -> None:
        drive(registry, 5)
        stats = registry.partition_stats(CAMERA)
        snapshot = PartitionSnapshot(
            camera_id=CAMERA,
            site_id=SITE,
            version=stats.version,
            taken_at=at(5),
            objects=registry.objects(CAMERA),
        )
        object_store.save(snapshot)
        assert object_store.load(CAMERA).version == stats.version


class TestDurableState:
    def test_a_round_trip_preserves_identity(self, registry, object_store) -> None:
        drive(registry, 6)
        before = registry.objects(CAMERA)
        object_store.save(
            PartitionSnapshot(
                camera_id=CAMERA, site_id=SITE, version=1, taken_at=at(6), objects=before
            )
        )
        restored = object_store.load(CAMERA)
        assert restored.objects[0].object_id == before[0].object_id

    def test_a_file_store_round_trips(self, registry) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = FileObjectStore(Path(directory))
            drive(registry, 6)
            before = registry.objects(CAMERA)
            store.save(
                PartitionSnapshot(
                    camera_id=CAMERA,
                    site_id=SITE,
                    version=3,
                    taken_at=at(6),
                    objects=before,
                )
            )
            after = store.load(CAMERA)
            assert after is not None
            assert after.objects[0].object_id == before[0].object_id
            assert after.objects[0].first_seen == before[0].first_seen

    def test_absence_is_not_an_error(self) -> None:
        assert InMemoryObjectStore().load(CameraId("never-written")) is None

    def test_a_corrupt_partition_fails_loudly(self) -> None:
        """Obligation S3 — never present data loss as a fresh start."""
        with tempfile.TemporaryDirectory() as directory:
            store = FileObjectStore(Path(directory))
            store.save(
                PartitionSnapshot(
                    camera_id=CAMERA, site_id=SITE, version=1, taken_at=at(0)
                )
            )
            store._path(CAMERA).write_text("{ broken", encoding="utf-8")  # noqa: SLF001
            with pytest.raises(ObjectStoreError, match="fresh start"):
                store.load(CAMERA)

    def test_an_unknown_format_version_is_refused(self) -> None:
        """A format change is a migration, never a silent reinterpretation."""
        with tempfile.TemporaryDirectory() as directory:
            store = FileObjectStore(Path(directory))
            store.save(
                PartitionSnapshot(
                    camera_id=CAMERA, site_id=SITE, version=1, taken_at=at(0)
                )
            )
            path = store._path(CAMERA)  # noqa: SLF001
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["format"] = SNAPSHOT_FORMAT_VERSION + 99
            path.write_text(json.dumps(payload), encoding="utf-8")
            with pytest.raises(ObjectStoreError, match="structurally invalid"):
                store.load(CAMERA)

    def test_writes_are_atomic(self) -> None:
        """A partially written partition reloads as plausible corruption."""
        with tempfile.TemporaryDirectory() as directory:
            store = FileObjectStore(Path(directory))
            store.save(
                PartitionSnapshot(
                    camera_id=CAMERA, site_id=SITE, version=1, taken_at=at(0)
                )
            )
            store.save(
                PartitionSnapshot(
                    camera_id=CAMERA, site_id=SITE, version=2, taken_at=at(1)
                )
            )
            leftovers = list(Path(directory).glob("*.tmp"))
            assert not leftovers, f"temporary files were left behind: {leftovers}"
            assert store.load(CAMERA).version == 2

    def test_forget_removes_durable_state(self, object_store) -> None:
        object_store.save(
            PartitionSnapshot(camera_id=CAMERA, site_id=SITE, version=1, taken_at=at(0))
        )
        object_store.forget(CAMERA)
        assert object_store.load(CAMERA) is None


class TestRecovery:
    """07_STATE section 9.3 — object identity survives, tracks do not."""

    def _rebuild(self, clock, bus, metrics, registry_config, registry_provenance):
        from vision_os.perception.registry import LifecyclePolicy, ObjectRegistry

        from ..conftest import TENANT

        return ObjectRegistry(
            clock=clock,
            bus=bus,
            metrics=metrics,
            config=registry_config,
            tenant_id=TENANT,
            site_id=SITE,
            provenance=registry_provenance,
            lifecycle=LifecyclePolicy(min_observations_to_confirm=3),
        )

    def test_object_identity_survives_a_restart(
        self, registry, object_store, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        drive(registry, 8)
        before = registry.objects(CAMERA)
        object_store.save(
            PartitionSnapshot(
                camera_id=CAMERA, site_id=SITE, version=1, taken_at=at(8), objects=before
            )
        )

        revived = self._rebuild(clock, bus, metrics, registry_config, registry_provenance)
        restored = revived.restore(object_store.load(CAMERA))

        assert restored == len(before)
        after = revived.objects(CAMERA)
        assert after[0].object_id == before[0].object_id

    def test_a_long_lived_object_does_not_become_new(
        self, registry, object_store, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        """*"An object present for 20 minutes must not become a new object
        because a process recycled."*"""
        drive(registry, 8)
        before = registry.objects(CAMERA)[0]
        object_store.save(
            PartitionSnapshot(
                camera_id=CAMERA,
                site_id=SITE,
                version=1,
                taken_at=at(8),
                objects=(before,),
            )
        )
        revived = self._rebuild(clock, bus, metrics, registry_config, registry_provenance)
        revived.restore(object_store.load(CAMERA))
        assert revived.get(before.object_id).first_seen == before.first_seen

    def test_tracks_do_not_survive_a_restart(
        self, registry, object_store, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        """Every binding closes: the tracks that produced them died with the
        process, and leaving one open would let a recycled track id inherit it.
        """
        drive(registry, 8)
        before = registry.objects(CAMERA)
        assert before[0].bound_track is not None

        object_store.save(
            PartitionSnapshot(
                camera_id=CAMERA, site_id=SITE, version=1, taken_at=at(8), objects=before
            )
        )
        revived = self._rebuild(clock, bus, metrics, registry_config, registry_provenance)
        revived.restore(object_store.load(CAMERA))

        after = revived.objects(CAMERA)[0]
        assert after.bound_track is None
        assert after.track_bindings, "the binding history is retained, merely closed"

    def test_re_binding_after_a_restart_carries_reduced_confidence(
        self, registry, object_store, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        drive(registry, 8)
        coast(registry, 1, start=10)
        age(registry, 19)
        before = registry.objects(CAMERA)

        object_store.save(
            PartitionSnapshot(
                camera_id=CAMERA, site_id=SITE, version=1, taken_at=at(19), objects=before
            )
        )
        revived = self._rebuild(clock, bus, metrics, registry_config, registry_provenance)
        revived.restore(object_store.load(CAMERA))

        # A new epoch after restart, at the object's last known position.
        result = revived.ingest(
            CAMERA,
            make_update(
                [
                    make_track(
                        local=0,
                        box=before[0].current_spatial.bbox,
                        seq=20,
                        epoch=1,
                    )
                ],
                seq=20,
                epoch=1,
            ),
        )
        rebinds = [
            a for a in result.assertions if a.method is BindingMethod.EPOCH_REBIND
        ]
        if rebinds:
            assert rebinds[0].confidence.value < 1.0, (
                "07_STATE section 9.3 requires explicitly reduced confidence"
            )

    def test_merged_objects_survive_a_restart(
        self, registry, object_store, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        """V5 must hold across a deployment, not merely within one process."""
        for seq in range(5):
            tracks = [
                make_track(local=0, box=Box(0.1, 0.4, 0.2, 0.8), seq=seq),
                make_track(local=1, box=Box(0.7, 0.4, 0.8, 0.8), seq=seq),
            ]
            registry.ingest(CAMERA, make_update(tracks, seq=seq))
        objects = registry.active(CAMERA)
        source, target = objects[0].object_id, objects[1].object_id
        registry.merge(source, target)

        object_store.save(
            PartitionSnapshot(
                camera_id=CAMERA,
                site_id=SITE,
                version=1,
                taken_at=at(6),
                objects=registry.objects(CAMERA),
            )
        )
        revived = self._rebuild(clock, bus, metrics, registry_config, registry_provenance)
        revived.restore(object_store.load(CAMERA))

        assert revived.resolve(source).object_id == target

    def test_restoring_twice_is_idempotent(
        self, registry, object_store, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        drive(registry, 5)
        snapshot = PartitionSnapshot(
            camera_id=CAMERA,
            site_id=SITE,
            version=1,
            taken_at=at(5),
            objects=registry.objects(CAMERA),
        )
        revived = self._rebuild(clock, bus, metrics, registry_config, registry_provenance)
        first = revived.restore(snapshot)
        second = revived.restore(snapshot)
        assert first == 1
        assert second == 0, "a duplicate restore must not double the population"

    def test_restoring_an_empty_snapshot_is_safe(
        self, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        revived = self._rebuild(clock, bus, metrics, registry_config, registry_provenance)
        assert (
            revived.restore(
                PartitionSnapshot(
                    camera_id=CAMERA, site_id=SITE, version=1, taken_at=at(0)
                )
            )
            == 0
        )


class TestRuntime:
    async def test_it_consumes_track_updates(self, registry_runtime) -> None:
        await registry_runtime.start()
        for seq in range(5):
            await registry_runtime.on_tracked(
                CAMERA, make_update([make_track(seq=seq)], seq=seq)
            )
        assert registry_runtime.stats.frames_consumed == 5
        assert registry_runtime.stats.updates_applied == 5

    async def test_it_ignores_frames_before_start(self, registry_runtime) -> None:
        await registry_runtime.on_tracked(CAMERA, make_update([make_track()], seq=0))
        assert registry_runtime.stats.frames_consumed == 0

    async def test_it_never_raises(self, registry_runtime, registry, monkeypatch) -> None:
        await registry_runtime.start()
        monkeypatch.setattr(
            registry, "ingest", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
        )
        await registry_runtime.on_tracked(CAMERA, make_update([make_track()], seq=0))
        assert registry_runtime.stats.frames_failed >= 1

    async def test_one_camera_is_serialized(self, registry_runtime) -> None:
        """07_STATE section 4.1 — exactly one writer per partition."""
        await registry_runtime.start()
        await asyncio.gather(
            *(
                registry_runtime.on_tracked(
                    CAMERA, make_update([make_track(seq=seq)], seq=seq)
                )
                for seq in range(20)
            )
        )
        assert registry_runtime.stats.frames_consumed == 20
        assert registry_runtime.stats.frames_failed == 0

    async def test_cameras_run_independently(self, registry_runtime) -> None:
        await registry_runtime.start()
        cameras = [CameraId(f"cam-{i:02d}") for i in range(12)]
        await asyncio.gather(
            *(
                registry_runtime.on_tracked(
                    camera,
                    make_update([make_track(seq=seq, camera=camera)], seq=seq, camera=camera),
                )
                for camera in cameras
                for seq in range(4)
            )
        )
        assert registry_runtime.cameras_seen == 12
        assert registry_runtime.stats.frames_failed == 0

    async def test_a_failing_sink_does_not_break_the_registry(
        self, clock, metrics, health, registry, registry_config, object_store
    ) -> None:
        from vision_os.perception.registry import RegistryRuntime

        def exploding(_result):
            raise ValueError("bad consumer")

        runtime = RegistryRuntime(
            clock=clock,
            metrics=metrics,
            health=health,
            registry=registry,
            config=registry_config,
            store=object_store,
            sink=exploding,
        )
        await runtime.start()
        await runtime.on_tracked(CAMERA, make_update([make_track()], seq=0))
        assert runtime.stats.updates_applied == 1
        assert runtime.stats.sink_failures == 1

    async def test_a_detached_camera_releases_its_lock(self, registry_runtime) -> None:
        await registry_runtime.start()
        await registry_runtime.on_tracked(CAMERA, make_update([make_track()], seq=0))
        assert registry_runtime.cameras_seen == 1
        registry_runtime.forget(CAMERA)
        assert registry_runtime.cameras_seen == 0

    async def test_stop_flushes_durable_state(
        self, clock, metrics, health, registry, object_store
    ) -> None:
        """A shutdown that discarded unflushed objects would make restart depend
        on whether the last flush happened to run."""
        from vision_os.kernel.config.schema import RegistrySection
        from vision_os.perception.registry import RegistryRuntime

        runtime = RegistryRuntime(
            clock=clock,
            metrics=metrics,
            health=health,
            registry=registry,
            config=RegistrySection(enabled=True, persistence_enabled=True),
            store=object_store,
        )
        await runtime.start()
        for seq in range(5):
            await runtime.on_tracked(CAMERA, make_update([make_track(seq=seq)], seq=seq))
        await runtime.stop()
        assert object_store.load(CAMERA) is not None

    async def test_a_store_failure_does_not_stop_ingestion(
        self, clock, metrics, health, registry
    ) -> None:
        """Durability degrades; the pipeline does not."""
        from vision_os.kernel.config.schema import RegistrySection
        from vision_os.perception.registry import RegistryRuntime

        class _Failing(ObjectStorePort):
            @property
            def store_id(self) -> str:
                return "failing"

            def save(self, snapshot) -> None:
                raise ObjectStoreError("disk is gone")

            def load(self, camera_id):
                return None

            def forget(self, camera_id) -> None:
                return None

        runtime = RegistryRuntime(
            clock=clock,
            metrics=metrics,
            health=health,
            registry=registry,
            config=RegistrySection(
                enabled=True, persistence_enabled=True, persistence_interval_ms=1
            ),
            store=_Failing(),
        )
        await runtime.start()
        for seq in range(5):
            await runtime.on_tracked(CAMERA, make_update([make_track(seq=seq)], seq=seq))
        await runtime.flush_now()

        assert runtime.stats.updates_applied == 5
        assert runtime.stats.persist_failures >= 1
        assert metrics.snapshot().counters_matching(MetricName.OBJECT_STORE_FAILURES)

    async def test_health_is_reported(self, registry_runtime, health) -> None:
        from vision_os.perception.registry import REGISTRY_RUNTIME_ID

        await registry_runtime.start()
        assert health.component_health(REGISTRY_RUNTIME_ID).state.value == "healthy"


class TestObjectStoreConformance:
    def test_the_in_memory_store_passes_its_kit(self) -> None:
        report = OBJECT_STORE_KIT.run(InMemoryObjectStore())
        assert report.passed, report.failures

    def test_the_file_store_passes_its_kit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = OBJECT_STORE_KIT.run(FileObjectStore(Path(directory)))
            assert report.passed, report.failures

    def test_the_kit_runs_every_check(self) -> None:
        report = OBJECT_STORE_KIT.run(InMemoryObjectStore())
        assert len(report.executed) == len(OBJECT_STORE_KIT.checks)

    def test_the_kit_is_registered_for_the_platform(self, conformance) -> None:
        assert conformance.get(PortCatalogue.STATE_STORE) is not None

    def test_a_store_that_loses_identity_is_caught(self) -> None:
        """A kit that passes everything is indistinguishable from no kit."""

        class _Amnesiac(ObjectStorePort):
            @property
            def store_id(self) -> str:
                return "amnesiac"

            def save(self, snapshot) -> None:
                self._last = PartitionSnapshot(
                    camera_id=snapshot.camera_id,
                    site_id=snapshot.site_id,
                    version=snapshot.version,
                    taken_at=snapshot.taken_at,
                    objects=(),
                )

            def load(self, camera_id):
                return getattr(self, "_last", None)

            def forget(self, camera_id) -> None:
                return None

        report = OBJECT_STORE_KIT.run(_Amnesiac())
        assert not report.passed

    def test_a_store_that_drops_merged_objects_is_caught(self) -> None:
        class _DropsTerminal(InMemoryObjectStore):
            @property
            def store_id(self) -> str:
                return "drops-terminal"

            def save(self, snapshot) -> None:
                super().save(
                    PartitionSnapshot(
                        camera_id=snapshot.camera_id,
                        site_id=snapshot.site_id,
                        version=snapshot.version,
                        taken_at=snapshot.taken_at,
                        objects=tuple(
                            o for o in snapshot.objects if not o.lifecycle.is_terminal
                        ),
                    )
                )

        report = OBJECT_STORE_KIT.run(_DropsTerminal())
        assert not report.passed
        assert any("merged" in failure for failure in report.failures)

    def test_a_store_that_swallows_corruption_is_caught(self) -> None:
        class _Silent(FileObjectStore):
            def load(self, camera_id):
                try:
                    return super().load(camera_id)
                except ObjectStoreError:
                    return None  # the defect under test

        with tempfile.TemporaryDirectory() as directory:
            report = OBJECT_STORE_KIT.run(_Silent(Path(directory)))
            assert not report.passed
            assert any("decode_failure" in failure for failure in report.failures)


class TestIdentityResolverStaysUnbound:
    """15_ROADMAP section 3: P11 has no implementations in Phase 1."""

    def test_the_port_is_not_bindable(self) -> None:
        assert PortCatalogue.IDENTITY_RESOLVER not in BINDABLE_PORTS

    def test_no_resolver_adapter_ships(self) -> None:
        from pathlib import Path as P

        import vision_os.adapters.registry as pkg

        for path in P(pkg.__file__).parent.rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            assert "class identityresolver" not in text
            assert "def resolve(" not in text

    def test_its_kit_is_not_registered(self, conformance) -> None:
        """Registering a kit for a port with no implementations would suggest
        one is expected."""
        assert conformance.get(PortCatalogue.IDENTITY_RESOLVER) is None

    def test_the_kit_exists_for_phase_two(self) -> None:
        """The contract waits for the adapter, not the other way round."""
        from vision_os.conformance import IDENTITY_RESOLVER_KIT

        assert len(IDENTITY_RESOLVER_KIT.checks) >= 5
        obligations = {c.obligation for c in IDENTITY_RESOLVER_KIT.checks if c.obligation}
        assert {"I1", "I3", "I4", "I5"} <= obligations

    def test_the_registry_runs_without_a_resolver(self, registry) -> None:
        """M7's native binding is mandatory behaviour, not an extension."""
        results = drive(registry, 8)
        assert not any(r.failed for r in results)
        assert len(registry.active(CAMERA)) == 1
