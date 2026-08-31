"""P9.7 repository hygiene and backward compatibility.

These tests run against the **real** repository, because that is the thing whose
state matters: a guard that only passes on a fixture proves nothing about
whether production CCTV is about to reach GitHub.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.p9_dataset.guard import KNOWN_TRACKED_PIXELS, audit
from tools.p9_dataset.store import PIXEL_SUFFIXES, REPO


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout


class TestNoNewCctvExposure:
    def test_no_new_pixel_file_is_tracked(self):
        """The headline invariant of P9.7.

        `KNOWN_TRACKED_PIXELS` records what was already committed when the guard
        was written. Anything outside it is new exposure and fails here.
        """
        report = audit()
        assert report["new_tracked"] == [], (
            f"{len(report['new_tracked'])} newly tracked image/video file(s): "
            f"{report['new_tracked'][:5]}"
        )

    def test_nothing_is_staged(self):
        assert audit()["staged_pixel_files"] == []

    def test_no_untracked_pixel_file_could_be_staged(self):
        """`git add .` must not be able to publish the corpus.

        This is the failure mode the phase exists to close: before P9.7 the
        working tree held 723 MB of CCTV frames with no ignore rule at all.
        """
        assert audit()["stageable_pixel_files"] == []

    def test_the_known_list_is_not_a_licence_to_add_more(self):
        """It names directories that predate the guard, and only those."""
        assert all(k.startswith("datasets/") for k in KNOWN_TRACKED_PIXELS)
        assert len(KNOWN_TRACKED_PIXELS) <= 3


class TestGitignoreCoversTheCorpus:
    def _ignored(self, path: str) -> bool:
        result = subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=REPO, capture_output=True, check=False,
        )
        return result.returncode == 0

    def test_a_live_frame_is_ignored(self):
        frames = sorted(Path(REPO, "datasets", "p9-live").rglob("*.jpg"))
        if not frames:
            pytest.skip("no live corpus present in this checkout")
        assert self._ignored(str(frames[0].relative_to(REPO)).replace("\\", "/"))

    def test_session_metadata_is_not_ignored(self):
        """The trap the brief warns about, and which this phase walked into.

        Git cannot re-include a file whose parent directory is excluded, so a
        `datasets/p9-live/` rule also swallowed `session.json` — the record that
        makes a collection reproducible.
        """
        records = sorted(Path(REPO, "datasets", "p9-live").rglob("session.json"))
        if not records:
            pytest.skip("no live corpus present in this checkout")
        assert not self._ignored(str(records[0].relative_to(REPO)).replace("\\", "/"))

    def test_manifests_are_not_ignored(self):
        manifests = sorted(Path(REPO, "datasets", "manifests").rglob("*.json"))
        if not manifests:
            pytest.skip("no manifests built in this checkout")
        assert not self._ignored(str(manifests[0].relative_to(REPO)).replace("\\", "/"))

    def test_traces_are_not_ignored(self):
        """A trace holds hashes and boxes. Nobody is identifiable from it, and
        it is the evidence that makes a sampling policy re-testable."""
        traces = sorted(Path(REPO, "datasets", "p9-traces").glob("trace-*.json"))
        if not traces:
            pytest.skip("no traces present in this checkout")
        assert not self._ignored(str(traces[0].relative_to(REPO)).replace("\\", "/"))

    def test_the_ignore_rules_name_every_pixel_format_the_store_knows(self):
        rules = Path(REPO, ".gitignore").read_text(encoding="utf-8")
        for suffix in PIXEL_SUFFIXES:
            assert f"*{suffix}" in rules, f"{suffix} is unguarded in .gitignore"


class TestManifestsAreSmallEnoughToCommit:
    def test_a_manifest_is_orders_of_magnitude_smaller_than_its_corpus(self):
        path = Path(REPO, "datasets", "manifests", "candidates", "p9-live", "v1.json")
        if not path.is_file():
            pytest.skip("p9-live manifest not built in this checkout")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert path.stat().st_size < payload["total_bytes"] / 100
        assert payload["artifact_count"] > 1000

    def test_a_manifest_records_no_absolute_path(self):
        for path in sorted(Path(REPO, "datasets", "manifests").rglob("*.json")):
            text = path.read_text(encoding="utf-8")
            assert ":\\\\" not in text and ":/" not in text.replace("https://", "")


class TestP9BackwardCompatibility:
    """Rule 5: P9-v1 and P9-v2 are historical evidence and are not touched."""

    def test_both_annotation_manifests_still_verify(self):
        from tools.p9_dataset.manifest import verify

        for version in ("p9-v1", "p9-v2"):
            path = Path(REPO, "datasets", version, "manifest.json")
            if not path.is_file():
                pytest.skip(f"{version} not present in this checkout")
            ok, detail = verify(path)
            assert ok, detail

    def test_the_digest_is_unchanged(self):
        for version in ("p9-v1", "p9-v2"):
            path = Path(REPO, "datasets", version, "manifest.json")
            if not path.is_file():
                pytest.skip(f"{version} not present in this checkout")
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["digest"].startswith("fe16a44bc39e01e4")

    def test_p9_annotation_manifests_are_not_ignored_by_the_new_rules(self):
        """Historical ground truth must remain in version control."""
        for version in ("p9-v1", "p9-v2"):
            path = Path(REPO, "datasets", version, "manifest.json")
            if not path.is_file():
                continue
            result = subprocess.run(
                ["git", "check-ignore", "-q", f"datasets/{version}/manifest.json"],
                cwd=REPO, capture_output=True, check=False,
            )
            assert result.returncode != 0, f"{version}/manifest.json became ignored"

    def test_the_kitchen01_frames_p9_depends_on_are_still_tracked(self):
        """P9-v1/v2 annotate these frames. They predate P9.7 and stay tracked;
        removing them from Git would break reproduction of the only human PPE
        ground truth this programme has."""
        tracked = git("ls-files", "datasets/kitchen-01/frames/").splitlines()
        assert len([t for t in tracked if t.endswith(".jpg")]) == 15


class TestP96SamplerStillReproducible:
    """Rule 13: P9.6 sampling behaviour must remain reproducible."""

    def test_the_frozen_phase1_policy_is_unchanged(self):
        from tools.p9_dataset.baselines import PHASE1
        from tools.p9_dataset.events import DepartureRule

        assert PHASE1.version == "p9.6-events-1"
        assert PHASE1.departure_rule is DepartureRule.ON_EXPIRY

    def test_the_selected_policy_is_unchanged(self):
        from tools.p9_dataset.baselines import PHASE2_B
        from tools.p9_dataset.events import DepartureRule

        assert PHASE2_B.version == "p9.6-events-2b"
        assert PHASE2_B.departure_rule is DepartureRule.LAST_CONFIRMED

    def test_replay_is_still_deterministic(self):
        from tools.p9_dataset.baselines import PHASE2_B
        from tools.p9_dataset.trace import replay

        observations = [
            {
                "i": i,
                "t": i * 0.25,
                "hash": 0xFFFF if (i % 24) < 10 else 0x00FF,
                "boxes": [[[0.1, 0.1, 0.3, 0.9], 0.9]] if (i % 24) < 10 else [],
            }
            for i in range(48)
        ]
        trace = {
            "trace_id": "t",
            "cameras": [
                {"camera_id": "cam-11", "frames_decoded": 600, "observations": observations}
            ],
        }
        first = replay(trace, PHASE2_B)["by_camera"]["cam-11"]["kept"]
        second = replay(trace, PHASE2_B)["by_camera"]["cam-11"]["kept"]
        assert [e["captured"] for e in first] == [e["captured"] for e in second]

    def test_the_trace_corpus_is_still_loadable(self):
        from tools.p9_dataset.trace import load_traces

        traces = load_traces()
        if not traces:
            pytest.skip("no traces present in this checkout")
        assert all("cameras" in t and "totals" in t for t in traces)


class TestProductionUntouched:
    def test_production_never_imports_the_dataset_tooling(self):
        """The boundary, in the direction that actually matters permanently.

        This replaces an earlier `git status` assertion that the production
        directories were unmodified. That check encoded a *phase* claim — "the
        P9 dataset work touched no production code" — which was true when P9.7
        made it and is recorded in its report. It is not a repository invariant:
        production code must be free to change, and Pre-P9.9 changes
        `app/main.py` deliberately to fix camera bootstrap. A test that fails
        whenever anyone fixes a bug is a test that gets deleted.

        What is permanent, and stronger, is the **dependency direction**. The
        dataset tooling may read production code; production must never depend
        on the dataset tooling, or `tools/p9_dataset` becomes a runtime
        requirement of the product it was built to measure. Together with
        `test_the_store_imports_no_production_code` below, this pins the
        boundary from both sides — which a mutable status snapshot never did.
        """
        import ast

        offenders: dict[str, list[str]] = {}
        for directory in ("app", "vision_os", "compliance"):
            root = Path(REPO, directory)
            if not root.is_dir():
                continue
            for path in root.rglob("*.py"):
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"))
                except (OSError, SyntaxError):
                    continue
                hits = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if node.module.startswith("tools"):
                            hits.append(node.module)
                    elif isinstance(node, ast.Import):
                        hits.extend(
                            a.name for a in node.names if a.name.startswith("tools")
                        )
                if hits:
                    offenders[str(path.relative_to(REPO))] = hits
        assert offenders == {}, f"production imports dataset tooling: {offenders}"

    def test_the_store_imports_no_production_code(self):
        import ast

        tree = ast.parse(Path(REPO, "tools", "p9_dataset", "store.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
        assert imported <= {
            "__future__", "argparse", "enum", "hashlib", "json", "os",
            "shutil", "dataclasses", "datetime", "pathlib",
        }
