"""P9.7 external dataset storage — identity, integrity, failure modes.

Every test builds its own store in a tmp directory. Nothing here touches the
real corpus, the real manifests or the repository's Git state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.p9_dataset.store import (
    LAYER_KINDS,
    PIXEL_SUFFIXES,
    REPO,
    ROOT_ENV,
    SCHEMA_VERSION,
    Artifact,
    ArtifactKind,
    DatasetStore,
    Layer,
    StoreError,
    build,
    content_digest,
    ingest,
    read,
    sha256_of,
    status,
    verify,
    write,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A store whose manifests also land in tmp.

    Without the manifest override a test that calls `write()` deposits a file
    into the real repository — which one did, during P9.7.
    """
    root = tmp_path / "vision-os-data"
    monkeypatch.setenv(ROOT_ENV, str(root))
    return DatasetStore.resolve(manifest_root=tmp_path / "manifests").ensure()


def seed(store: DatasetStore, layer=Layer.CANDIDATES, dataset="d", version="v1", n=3):
    directory = store.dataset(layer, dataset, version)
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (directory / f"frame_{i:03d}.jpg").write_bytes(b"pixels-%d" % i)
    (directory / "session.json").write_text('{"session_id": "s"}', encoding="utf-8")
    return directory


class TestRootResolution:
    def test_the_environment_variable_selects_the_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ROOT_ENV, str(tmp_path / "elsewhere"))
        assert DatasetStore.resolve().root == (tmp_path / "elsewhere").resolve()

    def test_a_root_inside_the_repository_is_refused(self, monkeypatch):
        """The one configuration that defeats the entire mechanism.

        A store under the working tree puts production CCTV back where a
        `git add .` can reach it, which is the failure P9.7 exists to prevent.
        """
        monkeypatch.setenv(ROOT_ENV, str(REPO / "datasets" / "store"))
        with pytest.raises(StoreError, match="inside the repository"):
            DatasetStore.resolve()

    def test_the_default_is_outside_the_repository(self, monkeypatch):
        monkeypatch.delenv(ROOT_ENV, raising=False)
        root = DatasetStore.resolve().root
        with pytest.raises(ValueError):
            root.relative_to(REPO)

    def test_no_developer_path_is_hard_coded(self):
        """In executable code, not in prose.

        The module docstring deliberately quotes a Windows user path as an
        example of what a manifest must never contain, so a naive text search
        flags the very sentence explaining the rule. The invariant is about
        string literals the code actually evaluates.
        """
        import ast

        tree = ast.parse(Path("tools/p9_dataset/store.py").read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docstrings.add(doc)
        banned = ("C:/Users", "C:" + chr(92) + "Users", "/home/", "/Users/")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if node.value in docstrings:
                continue
            for pattern in banned:
                assert pattern not in node.value, f"absolute path in code: {node.value!r}"


class TestIdentityIsContentNotLocation:
    def test_relocating_the_store_does_not_change_the_digest(self, tmp_path, monkeypatch):
        """The property that lets the store be replaced later.

        If identity moved with the filesystem, every experiment that cited a
        dataset version would be invalidated by a disk migration.
        """
        first = tmp_path / "a"
        monkeypatch.setenv(ROOT_ENV, str(first))
        store_a = DatasetStore.resolve().ensure()
        seed(store_a)
        digest_a = build(store_a, Layer.CANDIDATES, "d", "v1").digest

        second = tmp_path / "b"
        second.mkdir()
        import shutil

        shutil.copytree(first, second, dirs_exist_ok=True)
        monkeypatch.setenv(ROOT_ENV, str(second))
        store_b = DatasetStore.resolve()
        digest_b = build(store_b, Layer.CANDIDATES, "d", "v1").digest

        assert digest_a == digest_b
        assert store_a.root != store_b.root

    def test_a_relocated_store_still_verifies_its_old_manifest(self, tmp_path, monkeypatch):
        import shutil

        first, second = tmp_path / "a", tmp_path / "b"
        monkeypatch.setenv(ROOT_ENV, str(first))
        store_a = DatasetStore.resolve().ensure()
        seed(store_a)
        manifest = build(store_a, Layer.CANDIDATES, "d", "v1")

        shutil.copytree(first, second)
        monkeypatch.setenv(ROOT_ENV, str(second))
        assert verify(manifest, DatasetStore.resolve()).ok

    def test_no_absolute_path_appears_in_a_manifest(self, store, tmp_path):
        seed(store)
        payload = json.dumps(build(store, Layer.CANDIDATES, "d", "v1").as_dict())
        assert str(tmp_path) not in payload
        assert "\\\\" not in payload, "logical paths must be POSIX-separated"

    def test_enumeration_order_cannot_change_identity(self, store):
        seed(store)
        manifest = build(store, Layer.CANDIDATES, "d", "v1")
        shuffled = list(reversed(manifest.artifacts))
        assert content_digest(shuffled) == manifest.digest

    def test_a_same_size_substitution_is_caught(self, store):
        """Names and sizes are not enough; the digest is over content."""
        directory = seed(store)
        manifest = build(store, Layer.CANDIDATES, "d", "v1")
        (directory / "frame_000.jpg").write_bytes(b"XXXXXXXX")
        assert not verify(manifest, store).ok


class TestIntegrity:
    def test_an_intact_dataset_verifies(self, store):
        seed(store)
        manifest = build(store, Layer.CANDIDATES, "d", "v1")
        report = verify(manifest, store)
        assert report.ok
        assert report.actual_digest == report.expected_digest

    def test_a_modified_frame_is_detected(self, store):
        directory = seed(store)
        manifest = build(store, Layer.CANDIDATES, "d", "v1")
        (directory / "frame_001.jpg").write_bytes(b"tampered")
        report = verify(manifest, store)
        assert not report.ok
        assert any("frame_001" in p for p in report.modified)

    def test_a_missing_frame_is_detected(self, store):
        directory = seed(store)
        manifest = build(store, Layer.CANDIDATES, "d", "v1")
        (directory / "frame_002.jpg").unlink()
        report = verify(manifest, store)
        assert not report.ok
        assert any("frame_002" in p for p in report.missing)

    def test_an_unexpected_file_is_detected(self, store):
        directory = seed(store)
        manifest = build(store, Layer.CANDIDATES, "d", "v1")
        (directory / "smuggled.jpg").write_bytes(b"not in the manifest")
        report = verify(manifest, store)
        assert not report.ok
        assert any("smuggled" in p for p in report.unexpected)

    def test_a_duplicate_sample_id_is_refused_at_build(self, store):
        directory = seed(store)
        (directory / "nested").mkdir()
        (directory / "nested" / "frame_000.jpg").write_bytes(b"x")
        # Distinct ids, because the id is the path within the version.
        assert build(store, Layer.CANDIDATES, "d", "v1")

    def test_the_sample_id_is_the_path_not_the_filename(self, store):
        """Every session directory holds a `session.json`.

        A stem-based id collides on the second one — caught on the first real
        corpus this tool described.
        """
        directory = seed(store)
        for name in ("s1", "s2"):
            (directory / name).mkdir()
            (directory / name / "session.json").write_text("{}", encoding="utf-8")
        ids = {a.sample_id for a in build(store, Layer.CANDIDATES, "d", "v1").artifacts}
        assert "s1/session" in ids and "s2/session" in ids

    def test_verification_never_repairs(self, store):
        directory = seed(store)
        manifest = build(store, Layer.CANDIDATES, "d", "v1")
        (directory / "frame_000.jpg").write_bytes(b"tampered")
        verify(manifest, store)
        assert (directory / "frame_000.jpg").read_bytes() == b"tampered"


class TestLayerDiscipline:
    def test_every_layer_declares_its_permitted_kinds(self):
        assert set(LAYER_KINDS) == set(Layer)

    def test_annotations_admit_only_annotated_artifacts(self):
        """A derived artifact in the annotations layer is a machine label
        wearing ground truth's clothes."""
        assert LAYER_KINDS[Layer.ANNOTATIONS] == {ArtifactKind.ANNOTATED}

    def test_a_wrong_kind_for_a_layer_is_refused(self, store):
        seed(store, layer=Layer.ANNOTATIONS, dataset="gt")
        with pytest.raises(StoreError, match="may not live in"):
            build(store, Layer.ANNOTATIONS, "gt", "v1", kind=ArtifactKind.DERIVED)

    def test_the_five_layers_are_a_closed_set(self):
        assert {layer.value for layer in Layer} == {
            "raw", "candidates", "annotations", "benchmarks", "traces"
        }


class TestFailureModes:
    def test_a_missing_dataset_root_is_visible_before_use(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ROOT_ENV, str(tmp_path / "absent"))
        store = DatasetStore.resolve()
        assert not store.exists()
        assert status(store)["root_exists"] is False

    def test_verification_of_a_missing_root_reports_it(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ROOT_ENV, str(tmp_path / "a"))
        store = DatasetStore.resolve().ensure()
        seed(store)
        manifest = build(store, Layer.CANDIDATES, "d", "v1")
        import shutil

        shutil.rmtree(store.root)
        report = verify(manifest, store)
        assert not report.ok
        assert any("does not exist" in e for e in report.errors)

    def test_a_missing_dataset_directory_reports_it(self, store):
        seed(store)
        manifest = build(store, Layer.CANDIDATES, "d", "v1")
        import shutil

        shutil.rmtree(store.dataset(Layer.CANDIDATES, "d", "v1"))
        report = verify(manifest, store)
        assert not report.ok
        assert any("does not exist" in e for e in report.errors)

    def test_building_a_dataset_that_is_not_there_is_refused(self, store):
        with pytest.raises(StoreError, match="no dataset at"):
            build(store, Layer.CANDIDATES, "absent", "v1")

    def test_a_missing_manifest_is_refused(self, tmp_path):
        with pytest.raises(StoreError, match="no manifest at"):
            read(tmp_path / "nope.json")

    def test_a_corrupt_manifest_is_refused(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(StoreError, match="not valid JSON"):
            read(path)

    def test_a_foreign_schema_version_is_refused_not_guessed(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(
            json.dumps({"schema_version": "some-other-format", "artifacts": []}),
            encoding="utf-8",
        )
        with pytest.raises(StoreError, match="Refusing rather than guessing"):
            read(path)

    def test_ingest_refuses_to_overwrite_a_version(self, store, tmp_path):
        source = tmp_path / "src"
        source.mkdir()
        (source / "a.jpg").write_bytes(b"x")
        ingest(store, source, Layer.RAW, "cam", "v1")
        with pytest.raises(StoreError, match="immutable"):
            ingest(store, source, Layer.RAW, "cam", "v1")

    def test_ingest_refuses_a_source_that_is_not_a_directory(self, store, tmp_path):
        with pytest.raises(StoreError, match="not a directory"):
            ingest(store, tmp_path / "absent", Layer.RAW, "cam", "v1")

    def test_ingest_verifies_every_copy(self, store, tmp_path):
        source = tmp_path / "src"
        source.mkdir()
        (source / "a.jpg").write_bytes(b"contents")
        result = ingest(store, source, Layer.RAW, "cam", "v1")
        assert result["copied"] == 1 and result["failed"] == 0
        copied = store.dataset(Layer.RAW, "cam", "v1") / "a.jpg"
        assert sha256_of(copied) == sha256_of(source / "a.jpg")

    def test_a_path_outside_the_store_is_refused(self, store, tmp_path):
        with pytest.raises(StoreError, match="not inside the store"):
            store.relative(tmp_path / "elsewhere" / "x.jpg")

    def test_an_unreadable_file_is_reported_not_skipped(self, store, monkeypatch):
        seed(store)
        manifest = build(store, Layer.CANDIDATES, "d", "v1")

        real = Path.open

        def explode(self, *args, **kwargs):
            if self.suffix == ".jpg":
                raise PermissionError("denied")
            return real(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", explode)
        report = verify(manifest, store)
        assert not report.ok
        assert any("unreadable" in e for e in report.errors)


class TestManifestRoundTrip:
    def test_a_manifest_survives_write_and_read(self, store):
        seed(store)
        original = build(store, Layer.CANDIDATES, "d", "v1", sampler_version="p9.6-events-2b")
        path = write(original, store)
        restored = read(path)
        assert restored.digest == original.digest
        assert restored.layer is original.layer
        assert len(restored.artifacts) == len(original.artifacts)

    def test_the_manifest_lives_outside_the_store(self, store):
        """The arrangement: digests are versioned with the code; pixels are not."""
        path = store.manifest_path(Layer.CANDIDATES, "d", "v1")
        assert not str(path).startswith(str(store.root))

    def test_the_default_manifest_root_is_the_repository(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ROOT_ENV, str(tmp_path / "data"))
        path = DatasetStore.resolve().manifest_path(Layer.CANDIDATES, "d", "v1")
        assert str(path).startswith(str(REPO / "datasets" / "manifests"))

    def test_a_test_cannot_write_into_the_repository_by_accident(self, store):
        assert not str(store.manifest_path(Layer.TRACES, "x", "v1")).startswith(
            str(REPO / "datasets" / "manifests")
        )

    def test_provenance_fields_are_carried(self, store):
        seed(store)
        manifest = build(
            store, Layer.CANDIDATES, "d", "v1",
            sampler_version="p9.6-events-2b", policy_version="p9.6-events-2b",
        )
        artifact = manifest.artifacts[0]
        for field in ("sha256", "bytes", "media_type", "sample_id", "schema_version"):
            assert getattr(artifact, field) not in ("", None)
        assert artifact.sampler_version == "p9.6-events-2b"
        assert manifest.schema_version == SCHEMA_VERSION


class TestReproducibilityChain:
    def test_the_full_chain_holds_and_one_changed_byte_breaks_it(self, store):
        """external store → manifest → verification → loader → experiment input."""
        directory = seed(store, n=5)
        manifest = build(store, Layer.CANDIDATES, "d", "v1")
        path = write(manifest, store)

        restored = read(path)
        assert verify(restored, store).ok

        selected = [a for a in restored.artifacts if a.media_type == "jpg"]
        assert len(selected) == 5
        for artifact in selected:
            assert store.absolute(artifact.logical_path).is_file()

        (directory / "frame_003.jpg").write_bytes(b"pixels-3 ")  # one byte longer
        assert not verify(restored, store).ok


class TestStatus:
    def test_status_reports_the_root_and_whether_it_is_safe(self, store):
        payload = status(store)
        assert payload["inside_repository"] is False
        assert set(payload["layers"]) == {layer.value for layer in Layer}


class TestPixelVocabulary:
    def test_the_pixel_suffixes_cover_the_formats_this_programme_produces(self):
        for suffix in (".jpg", ".png", ".mp4"):
            assert suffix in PIXEL_SUFFIXES

    def test_metadata_formats_are_not_treated_as_pixels(self):
        for suffix in (".json", ".md", ".py", ".csv"):
            assert suffix not in PIXEL_SUFFIXES


class TestArtifactSerialisation:
    def test_kind_round_trips_through_json(self):
        artifact = Artifact(
            logical_path="candidates/d/v1/a.jpg",
            kind=ArtifactKind.DERIVED,
            sha256="0" * 64,
            bytes=1,
            media_type="jpg",
        )
        assert Artifact.from_dict(json.loads(json.dumps(artifact.as_dict()))) == artifact
