"""Reference adapters for M13's P22 `EvidenceStorePort`.

Two shipped: an in-memory store honest about being volatile, and a file store
that survives a restart. A deployment with real retention obligations binds an
encrypted or object-storage adapter; §M13's Extension Points name *"tiered storage
(hot local → warm object store → cold archive); encryption at rest per privacy
class; regional pinning for data residency."*

Both honour `RetentionMode.NEVER_PERSIST` by storing nothing at all — 12_SECURITY
§2.3's no-evidence mode is a hard guarantee, and an adapter that stored it anyway
"just in case" would break a promise a deployment made to its regulator.
"""

from __future__ import annotations

from .evidence import (
    EVIDENCE_FACTORIES,
    FileEvidenceStore,
    InMemoryEvidenceStore,
    NullEvidenceStore,
)

__all__ = [
    "EVIDENCE_FACTORIES",
    "FileEvidenceStore",
    "InMemoryEvidenceStore",
    "NullEvidenceStore",
]
