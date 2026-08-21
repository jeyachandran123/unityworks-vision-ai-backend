"""Registry adapters — durable object state behind ``ObjectStorePort``.

No ``IdentityResolverPort`` adapter ships. ``15_ROADMAP`` section 3: P11 is
*"already specified, no implementations in Phase 1"*, and cross-camera identity
is classified C2 and policy-gated (``12_SECURITY`` section 2.3).
"""

from .stores import (
    ENCODED_OBJECT_KEYS,
    SNAPSHOT_FORMAT_VERSION,
    FileObjectStore,
    InMemoryObjectStore,
)

__all__ = [
    "ENCODED_OBJECT_KEYS",
    "SNAPSHOT_FORMAT_VERSION",
    "FileObjectStore",
    "InMemoryObjectStore",
]
