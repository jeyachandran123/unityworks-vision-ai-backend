"""P25 ``ArtifactStorePort`` — fetch and verify model artifacts.

**Fails closed on a hash mismatch.** Loading unverified weights is a supply-chain
vulnerability, and a mismatch is a security event rather than a network glitch —
so it is never retried into success (12_SECURITY section 6).

An object-storage or OCI-registry store is a sibling adapter behind the same
port; only the fetch changes, never the verification.
"""

from __future__ import annotations

import hashlib
import shutil
import threading
from pathlib import Path

from ...core.errors import ArtifactIntegrityError, ArtifactUnavailableError
from ...core.ports.models import ArtifactRef

_HASH_ALGORITHMS = {"blake2b": hashlib.blake2b, "sha256": hashlib.sha256}


def compute_hash(path: Path, algorithm: str = "blake2b") -> str:
    """Hash a file's contents as ``algorithm:hexdigest``.

    Streams in chunks: a model artifact is routinely hundreds of megabytes, and
    reading one into memory to hash it would be a needless spike at exactly the
    moment the platform is already allocating device memory.
    """
    factory = _HASH_ALGORITHMS.get(algorithm)
    if factory is None:
        raise ArtifactIntegrityError(
            f"unsupported hash algorithm '{algorithm}'", algorithm=algorithm
        )
    digest = factory(digest_size=32) if algorithm == "blake2b" else factory()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"{algorithm}:{digest.hexdigest()}"


class LocalArtifactStore:
    """Verified artifacts from the local filesystem, cached by content hash.

    Caching by hash rather than by name means an artifact is fetched once per
    node ever, and two models that happen to share weights share one copy.
    """

    def __init__(self, cache_dir: Path | str, *, verify: bool = True) -> None:
        self._cache_dir = Path(cache_dir)
        self._verify = verify
        self._lock = threading.Lock()

    @property
    def store_id(self) -> str:
        return f"local:{self._cache_dir}"

    def _cache_path(self, ref: ArtifactRef) -> Path:
        safe = ref.expected_hash.replace(":", "_")
        return self._cache_dir / safe

    def has(self, ref: ArtifactRef) -> bool:
        return self._cache_path(ref).exists()

    def fetch(self, ref: ArtifactRef) -> str:
        """Return a local path to the verified artifact.

        Raises:
            ArtifactUnavailableError: the source is missing. Transient — a cached
                known-good copy may still serve.
            ArtifactIntegrityError: the content does not match its declared hash.
                Fails closed; the Model Manager marks the version bad.
        """
        cached = self._cache_path(ref)
        with self._lock:
            if cached.exists():
                return str(cached)

            source = Path(ref.uri.removeprefix("file://"))
            if not source.exists():
                raise ArtifactUnavailableError(
                    f"artifact '{ref.uri}' is not present", uri=ref.uri
                )

            if self._verify:
                algorithm = ref.expected_hash.split(":", 1)[0] if ":" in ref.expected_hash else "blake2b"
                actual = compute_hash(source, algorithm)
                if actual != ref.expected_hash:
                    raise ArtifactIntegrityError(
                        f"artifact '{ref.uri}' hashes to {actual} but was declared "
                        f"{ref.expected_hash}. Loading unverified weights is a "
                        f"supply-chain vulnerability; this is never retried.",
                        uri=ref.uri,
                        expected=ref.expected_hash,
                        actual=actual,
                    )

            self._cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, cached)
            return str(cached)


class InMemoryArtifactStore:
    """Artifacts held in memory, for tests and embedded deployments.

    Verifies exactly as the local store does, so a test that would fail on a hash
    mismatch in production fails here too.
    """

    def __init__(self, artifacts: dict[str, bytes] | None = None) -> None:
        self._artifacts = dict(artifacts or {})
        self._lock = threading.Lock()

    @property
    def store_id(self) -> str:
        return "in-memory"

    def put(self, uri: str, payload: bytes) -> str:
        """Store an artifact and return its computed hash."""
        with self._lock:
            self._artifacts[uri] = payload
        digest = hashlib.blake2b(payload, digest_size=32).hexdigest()
        return f"blake2b:{digest}"

    def has(self, ref: ArtifactRef) -> bool:
        return ref.uri in self._artifacts

    def fetch(self, ref: ArtifactRef) -> str:
        with self._lock:
            payload = self._artifacts.get(ref.uri)
        if payload is None:
            raise ArtifactUnavailableError(
                f"artifact '{ref.uri}' is not present", uri=ref.uri
            )
        digest = hashlib.blake2b(payload, digest_size=32).hexdigest()
        actual = f"blake2b:{digest}"
        if actual != ref.expected_hash:
            raise ArtifactIntegrityError(
                f"artifact '{ref.uri}' hashes to {actual} but was declared "
                f"{ref.expected_hash}",
                uri=ref.uri,
            )
        return ref.uri
