"""External dataset storage, provenance and integrity — P9.7.

    python -m tools.p9_dataset.store status
    python -m tools.p9_dataset.store build   --layer candidates --dataset p9-live --version v1
    python -m tools.p9_dataset.store verify  --layer candidates --dataset p9-live --version v1

### The problem

`datasets/` holds 723 MB of production CCTV frames inside the Git working tree,
and `.gitignore` does not mention it. A single `git add .` would publish
identifiable footage of real people to GitHub, where deletion does not delete.
Meanwhile the programme needs *more* collection, not less — multi-day, multi-shift
— which makes the working tree exactly the wrong place for it.

### The separation

The repository keeps what makes an experiment **reproducible**; the external
store keeps what makes it **large and sensitive**.

    repository                     external store (VISION_OS_DATA_ROOT)
    ├── code                       ├── raw/
    ├── schemas                    ├── candidates/
    ├── manifests  ──references──▶ ├── annotations/
    └── tooling                    ├── benchmarks/
                                   └── traces/

A manifest is a few hundred kilobytes of digests and provenance. It goes in Git.
The pixels it describes do not.

### Identity is content, not location

A dataset's identity is a SHA-256 over its sorted `(logical_path, digest)` pairs.
Logical paths are store-relative and always `/`-separated, so **the identity does
not change when the store moves** — between machines, between drives, or from a
local filesystem to object storage. That is the property that lets the store be
replaced later without invalidating any experiment that cited a dataset version.

Absolute paths appear nowhere in a manifest. A manifest that recorded
`C:/Users/someone/...` would be a manifest that only verified on one laptop.

### What this module deliberately does not do

It does not move, copy or delete anything on its own. Ingestion is explicit and
verified; deletion is left to a human with a retention policy, because *"do not
silently discard inconvenient data"* is a standing rule of this programme and a
storage tool is exactly where that rule gets broken.
"""

from __future__ import annotations

import argparse
import enum
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]

#: Environment variable naming the external dataset root.
#:
#: No default inside the repository, by design: a default under `REPO` would put
#: CCTV back in the Git tree the moment someone forgot to set it, which is the
#: failure this module exists to prevent.
ROOT_ENV = "VISION_OS_DATA_ROOT"

#: Fallback when the variable is unset: a sibling of the repository. Outside the
#: working tree by construction, and portable — it is derived from the repo's own
#: location rather than from anyone's home directory.
DEFAULT_ROOT = REPO.parent / "vision-os-data"

SCHEMA_VERSION = "p9.7-store-1"


class Layer(enum.Enum):
    """The five data layers, kept strictly apart.

    The separation is not tidiness. Each layer has a different retention need, a
    different access boundary and a different answer to "may a model write this",
    and collapsing any two of them loses that distinction.
    """

    RAW = "raw"
    """Original production captures. Immutable, never edited in place."""

    CANDIDATES = "candidates"
    """Frames the validated sampler selected. Derived; reconstructible from raw
    plus a policy version, which is why they may expire before raw does."""

    ANNOTATIONS = "annotations"
    """Human-created ground truth **only**. No model output may be written
    here — the rule the whole programme rests on."""

    BENCHMARKS = "benchmarks"
    """Immutable evaluation sets derived from human-verified annotations."""

    TRACES = "traces"
    """Non-pixel evidence: hashes, boxes, event decisions, session metadata. No
    person is identifiable from a trace, which is why it may live longer and,
    unlike the other layers, may reasonably sit in Git."""


class ArtifactKind(enum.Enum):
    """What an artifact *is*, independent of where it sits."""

    RAW = "raw"
    DERIVED = "derived"
    ANNOTATED = "annotated"
    BENCHMARK = "benchmark"
    TRACE = "trace"


#: Which kinds may legally appear in which layer. Enforced at manifest build.
#:
#: The load-bearing row is `ANNOTATIONS: {ANNOTATED}` — a `DERIVED` artifact in
#: the annotations layer is a machine label wearing ground truth's clothes, and
#: the manifest refuses to describe one.
LAYER_KINDS = {
    Layer.RAW: {ArtifactKind.RAW},
    Layer.CANDIDATES: {ArtifactKind.DERIVED},
    Layer.ANNOTATIONS: {ArtifactKind.ANNOTATED},
    Layer.BENCHMARKS: {ArtifactKind.BENCHMARK},
    Layer.TRACES: {ArtifactKind.TRACE},
}

#: Extensions that carry recoverable images of identifiable people.
PIXEL_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
     ".mp4", ".avi", ".mkv", ".mov", ".webm"}
)


class StoreError(RuntimeError):
    """Raised when the store cannot be used safely.

    Every failure in this module is loud. A dataset tool that degrades quietly
    hands an experiment a partial corpus and lets it publish a number.
    """


@dataclass(frozen=True, slots=True)
class Artifact:
    """One file, described portably.

    `logical_path` is store-relative and POSIX-separated. It is the artifact's
    name for all time; the absolute path is a detail of the machine that happens
    to be holding it.
    """

    logical_path: str
    kind: ArtifactKind
    sha256: str
    bytes: int
    media_type: str

    sample_id: str = ""
    camera_id: str = ""
    session_id: str = ""
    captured_at: str = ""
    source_artifact: str = ""
    """The logical path this was derived from, when known. Empty for RAW."""

    sampler_version: str = ""
    policy_version: str = ""
    schema_version: str = ""
    provenance: str = ""
    created_at: str = ""

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> Artifact:
        payload = dict(payload)
        payload["kind"] = ArtifactKind(payload["kind"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class DatasetStore:
    """A dataset root somewhere outside the repository."""

    root: Path

    manifest_root: Path | None = None
    """Where manifests are written. Defaults to `<repo>/datasets/manifests`.

    Configurable for the same reason `root` is: a test that writes a manifest
    must not write it into the repository. One did, during P9.7, and left a
    stray `candidates/d/v1.json` behind — the argument exists so that cannot
    recur."""

    @classmethod
    def resolve(cls, root: str | Path | None = None, *, manifest_root: Path | None = None) -> DatasetStore:
        """From an argument, then the environment, then the sibling default."""
        chosen = Path(root) if root else Path(os.environ.get(ROOT_ENV) or DEFAULT_ROOT)
        chosen = chosen.expanduser().resolve()
        if _is_inside(chosen, REPO):
            raise StoreError(
                f"{ROOT_ENV}={chosen} is inside the repository at {REPO}. The "
                f"store exists to keep production CCTV out of the Git tree; a "
                f"root inside it defeats the entire mechanism."
            )
        return cls(root=chosen, manifest_root=manifest_root)

    def layer(self, layer: Layer) -> Path:
        return self.root / layer.value

    def dataset(self, layer: Layer, dataset: str, version: str) -> Path:
        return self.layer(layer) / dataset / version

    def manifest_path(self, layer: Layer, dataset: str, version: str) -> Path:
        """Manifests live in the **repository**, not the store.

        That is the whole arrangement: the small, reviewable, diff-able record of
        what a dataset contains is versioned with the code that used it, while
        the pixels it describes stay outside.
        """
        base = self.manifest_root or (REPO / "datasets" / "manifests")
        return base / layer.value / dataset / f"{version}.json"

    def exists(self) -> bool:
        return self.root.is_dir()

    def ensure(self) -> DatasetStore:
        for layer in Layer:
            self.layer(layer).mkdir(parents=True, exist_ok=True)
        return self

    def relative(self, path: Path) -> str:
        """The portable logical name for an absolute path inside the store."""
        try:
            return PurePosixPath(path.resolve().relative_to(self.root)).as_posix()
        except ValueError as error:
            raise StoreError(f"{path} is not inside the store at {self.root}") from error

    def absolute(self, logical_path: str) -> Path:
        return self.root / PurePosixPath(logical_path)


def _is_inside(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def sha256_of(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def content_digest(artifacts: list[Artifact]) -> str:
    """Dataset identity: a digest over sorted `(logical_path, sha256)` pairs.

    Sorted, so file-system enumeration order cannot change it. Over logical
    paths, so **relocating the store does not change it**. Over digests rather
    than sizes or names, so a same-size substitution is caught.
    """
    digest = hashlib.sha256()
    for artifact in sorted(artifacts, key=lambda a: a.logical_path):
        digest.update(artifact.logical_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(artifact.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(slots=True)
class Manifest:
    dataset_id: str
    version: str
    layer: Layer
    artifacts: list[Artifact] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    created_at: str = ""
    parent_dataset: str = ""
    parent_version: str = ""
    notes: str = ""

    @property
    def digest(self) -> str:
        return content_digest(self.artifacts)

    @property
    def total_bytes(self) -> int:
        return sum(a.bytes for a in self.artifacts)

    def as_dict(self) -> dict:
        return {
            "_comment": [
                "P9.7 dataset manifest. It describes artifacts held in an",
                "EXTERNAL store; the pixels are deliberately not in this repo.",
                "Paths are logical and store-relative, so dataset identity",
                "survives relocation — moving the store, or replacing it with",
                "object storage, does not change the digest.",
                "The digest is SHA-256 over sorted (logical_path, sha256) pairs.",
            ],
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "version": self.version,
            "layer": self.layer.value,
            "created_at": self.created_at,
            "parent_dataset": self.parent_dataset,
            "parent_version": self.parent_version,
            "notes": self.notes,
            "digest": self.digest,
            "artifact_count": len(self.artifacts),
            "total_bytes": self.total_bytes,
            "artifacts": [a.as_dict() for a in self.artifacts],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Manifest:
        return cls(
            dataset_id=payload["dataset_id"],
            version=payload["version"],
            layer=Layer(payload["layer"]),
            artifacts=[Artifact.from_dict(a) for a in payload["artifacts"]],
            schema_version=payload.get("schema_version", ""),
            created_at=payload.get("created_at", ""),
            parent_dataset=payload.get("parent_dataset", ""),
            parent_version=payload.get("parent_version", ""),
            notes=payload.get("notes", ""),
        )


def build(
    store: DatasetStore,
    layer: Layer,
    dataset: str,
    version: str,
    *,
    kind: ArtifactKind | None = None,
    sampler_version: str = "",
    policy_version: str = "",
    parent_dataset: str = "",
    parent_version: str = "",
    notes: str = "",
) -> Manifest:
    """Walk a dataset directory in the store and describe every file in it."""
    directory = store.dataset(layer, dataset, version)
    if not directory.is_dir():
        raise StoreError(
            f"no dataset at {directory}. Ingest it first; `build` describes what "
            f"is there and never invents it."
        )

    kind = kind or next(iter(LAYER_KINDS[layer]))
    if kind not in LAYER_KINDS[layer]:
        raise StoreError(
            f"a {kind.value!r} artifact may not live in the {layer.value!r} "
            f"layer (permitted: {sorted(k.value for k in LAYER_KINDS[layer])})"
        )

    artifacts: list[Artifact] = []
    seen: dict[str, str] = {}
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        logical = store.relative(path)
        # The sample id is the path **within the dataset version**, minus the
        # extension — not the filename stem. Every collection session contains a
        # `session.json`, so a stem-based id collides on the second one; the
        # duplicate guard caught that on the first real corpus it saw.
        within = PurePosixPath(path.relative_to(directory).as_posix())
        sample_id = str(within.with_suffix("")) if within.suffix else str(within)
        if sample_id in seen:
            raise StoreError(
                f"duplicate sample id {sample_id!r}: {seen[sample_id]} and "
                f"{logical}. A dataset with two samples of one name cannot be "
                f"split, cited or audited."
            )
        seen[sample_id] = logical
        statistics = path.stat()
        artifacts.append(
            Artifact(
                logical_path=logical,
                kind=kind,
                sha256=sha256_of(path),
                bytes=statistics.st_size,
                media_type=path.suffix.lower().lstrip(".") or "bin",
                sample_id=sample_id,
                sampler_version=sampler_version,
                policy_version=policy_version,
                schema_version=SCHEMA_VERSION,
                provenance=layer.value.upper(),
                created_at=datetime.fromtimestamp(statistics.st_mtime, UTC).isoformat(
                    timespec="seconds"
                ),
            )
        )

    return Manifest(
        dataset_id=dataset,
        version=version,
        layer=layer,
        artifacts=artifacts,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        parent_dataset=parent_dataset,
        parent_version=parent_version,
        notes=notes,
    )


def write(manifest: Manifest, store: DatasetStore) -> Path:
    path = store.manifest_path(manifest.layer, manifest.dataset_id, manifest.version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.as_dict(), indent=1) + "\n", encoding="utf-8")
    return path


def read(path: Path) -> Manifest:
    if not path.is_file():
        raise StoreError(f"no manifest at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise StoreError(f"manifest at {path} is not valid JSON: {error}") from error
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise StoreError(
            f"manifest at {path} declares schema {payload.get('schema_version')!r}, "
            f"this tool speaks {SCHEMA_VERSION!r}. Refusing rather than guessing "
            f"at a format it may not share."
        )
    return Manifest.from_dict(payload)


@dataclass(slots=True)
class Verification:
    """Every way a stored dataset can disagree with its manifest."""

    dataset_id: str
    version: str
    layer: str
    expected_digest: str
    actual_digest: str = ""
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    duplicate_sample_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.missing
            or self.unexpected
            or self.modified
            or self.duplicate_sample_ids
            or self.errors
            or self.actual_digest != self.expected_digest
        )

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "dataset_id": self.dataset_id,
            "version": self.version,
            "layer": self.layer,
            "expected_digest": self.expected_digest,
            "actual_digest": self.actual_digest,
            "digest_matches": self.actual_digest == self.expected_digest,
            "missing": self.missing,
            "unexpected": self.unexpected,
            "modified": self.modified,
            "duplicate_sample_ids": self.duplicate_sample_ids,
            "errors": self.errors,
        }

    def summary(self) -> str:
        if self.ok:
            return (
                f"{self.dataset_id}/{self.version} intact "
                f"({len(self.missing) == 0 and 'all files present'}) "
                f"digest {self.expected_digest[:16]}"
            )
        parts = []
        if self.missing:
            parts.append(f"{len(self.missing)} missing")
        if self.unexpected:
            parts.append(f"{len(self.unexpected)} unexpected")
        if self.modified:
            parts.append(f"{len(self.modified)} modified")
        if self.duplicate_sample_ids:
            parts.append(f"{len(self.duplicate_sample_ids)} duplicate ids")
        if self.actual_digest != self.expected_digest:
            parts.append("digest mismatch")
        parts.extend(self.errors)
        return f"{self.dataset_id}/{self.version} FAILED: " + ", ".join(parts)


def verify(manifest: Manifest, store: DatasetStore) -> Verification:
    """Compare the store against the manifest. Never repairs, only reports."""
    report = Verification(
        dataset_id=manifest.dataset_id,
        version=manifest.version,
        layer=manifest.layer.value,
        expected_digest=manifest.digest,
    )

    directory = store.dataset(manifest.layer, manifest.dataset_id, manifest.version)
    if not store.exists():
        report.errors.append(f"dataset root {store.root} does not exist")
        return report
    if not directory.is_dir():
        report.errors.append(f"dataset directory {directory} does not exist")
        return report

    declared = {a.logical_path: a for a in manifest.artifacts}
    seen_ids: dict[str, str] = {}
    for artifact in manifest.artifacts:
        if artifact.sample_id in seen_ids:
            report.duplicate_sample_ids.append(artifact.sample_id)
        seen_ids[artifact.sample_id] = artifact.logical_path

    present = {store.relative(p) for p in directory.rglob("*") if p.is_file()}
    report.missing = sorted(set(declared) - present)
    report.unexpected = sorted(present - set(declared))

    observed: list[Artifact] = []
    for logical in sorted(present & set(declared)):
        artifact = declared[logical]
        path = store.absolute(logical)
        try:
            actual = sha256_of(path)
        except OSError as error:
            report.errors.append(f"{logical}: unreadable ({error.__class__.__name__})")
            continue
        if actual != artifact.sha256:
            report.modified.append(logical)
        observed.append(
            Artifact(
                logical_path=logical,
                kind=artifact.kind,
                sha256=actual,
                bytes=path.stat().st_size,
                media_type=artifact.media_type,
            )
        )

    report.actual_digest = content_digest(observed) if not report.missing else ""
    return report


def ingest(
    store: DatasetStore,
    source: Path,
    layer: Layer,
    dataset: str,
    version: str,
    *,
    move: bool = False,
) -> dict:
    """Copy (or move) a directory into the store, then verify byte-for-byte.

    Copy is the default. A move that fails halfway has destroyed the only copy
    of production evidence, and this programme's rules do not permit losing
    evidence to a convenience.

    Refuses to overwrite an existing version: an immutable layer that can be
    overwritten in place is not immutable.
    """
    source = Path(source).resolve()
    if not source.is_dir():
        raise StoreError(f"source {source} is not a directory")
    target = store.dataset(layer, dataset, version)
    if target.exists() and any(target.iterdir()):
        raise StoreError(
            f"{target} already exists and is not empty. Versions are immutable; "
            f"write a new version rather than overwriting evidence."
        )

    store.ensure()
    target.mkdir(parents=True, exist_ok=True)
    copied = failed = 0
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(path, destination)
            if sha256_of(path) != sha256_of(destination):
                failed += 1
                raise StoreError(f"copy of {path} did not verify; store may be full")
            copied += 1
        except OSError as error:
            raise StoreError(f"copying {path} failed: {error}") from error

    if move:
        for path in sorted((p for p in source.rglob("*") if p.is_file()), reverse=True):
            path.unlink()

    return {"copied": copied, "failed": failed, "target": str(target)}


def status(store: DatasetStore) -> dict:
    """What the store holds, and what the repository knows about it."""
    layers: dict[str, dict] = {}
    for layer in Layer:
        directory = store.layer(layer)
        datasets: dict[str, list[str]] = {}
        if directory.is_dir():
            for dataset_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
                datasets[dataset_dir.name] = sorted(
                    p.name for p in dataset_dir.iterdir() if p.is_dir()
                )
        layers[layer.value] = {"exists": directory.is_dir(), "datasets": datasets}

    manifests = []
    root = REPO / "datasets" / "manifests"
    if root.is_dir():
        manifests = sorted(
            PurePosixPath(p.relative_to(root)).as_posix() for p in root.rglob("*.json")
        )

    return {
        "root": str(store.root),
        "root_env": ROOT_ENV,
        "root_configured": bool(os.environ.get(ROOT_ENV)),
        "root_exists": store.exists(),
        "inside_repository": _is_inside(store.root, REPO),
        "layers": layers,
        "manifests_in_repo": manifests,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help=f"overrides ${ROOT_ENV}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status")
    sub.add_parser("init")

    for name in ("build", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--layer", required=True, choices=[layer.value for layer in Layer])
        p.add_argument("--dataset", required=True)
        p.add_argument("--version", required=True)
        if name == "build":
            p.add_argument("--sampler-version", default="")
            p.add_argument("--policy-version", default="")
            p.add_argument("--notes", default="")

    p = sub.add_parser("ingest")
    p.add_argument("--source", required=True)
    p.add_argument("--layer", required=True, choices=[layer.value for layer in Layer])
    p.add_argument("--dataset", required=True)
    p.add_argument("--version", required=True)

    args = parser.parse_args()
    store = DatasetStore.resolve(args.root)

    if args.command == "status":
        payload = status(store)
        print(f"root            : {payload['root']}")
        print(f"configured via  : ${ROOT_ENV} = {payload['root_configured']}")
        print(f"exists          : {payload['root_exists']}")
        print(f"inside repo     : {payload['inside_repository']}  (must be False)")
        for name, entry in payload["layers"].items():
            print(f"  {name:12s} {'ok' if entry['exists'] else 'missing':8s} {entry['datasets']}")
        print(f"manifests in repo: {len(payload['manifests_in_repo'])}")
        for name in payload["manifests_in_repo"]:
            print(f"  {name}")
        return 0

    if args.command == "init":
        store.ensure()
        print(f"initialised {store.root}")
        for layer in Layer:
            print(f"  {store.layer(layer)}")
        return 0

    if args.command == "ingest":
        result = ingest(
            store, Path(args.source), Layer(args.layer), args.dataset, args.version
        )
        print(f"ingested {result['copied']} file(s) -> {result['target']}")
        return 0

    layer = Layer(args.layer)
    if args.command == "build":
        manifest = build(
            store,
            layer,
            args.dataset,
            args.version,
            sampler_version=args.sampler_version,
            policy_version=args.policy_version,
            notes=args.notes,
        )
        path = write(manifest, store)
        print(f"{len(manifest.artifacts)} artifact(s)")
        print(f"digest  : {manifest.digest}")
        print(f"bytes   : {manifest.total_bytes:,}")
        print(f"manifest: {path.relative_to(REPO)}")
        return 0

    manifest = read(store.manifest_path(layer, args.dataset, args.version))
    report = verify(manifest, store)
    print(report.summary())
    for name in ("missing", "unexpected", "modified", "duplicate_sample_ids"):
        for entry in getattr(report, name)[:10]:
            print(f"  {name}: {entry}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
