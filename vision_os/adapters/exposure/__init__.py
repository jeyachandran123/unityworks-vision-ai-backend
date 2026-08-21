"""Reference adapters for P31 and P32.

Every one is honest about being a reference: `StaticAuthorizer` reads grants from
configuration rather than from a policy service, and `InProcessTransport` carries
a call across a function boundary rather than a network. A deployment with real
authorization needs binds an RBAC or ABAC adapter; the port is where that choice
belongs (§M14 Extension Points).

What they are *not* is permissive. `StaticAuthorizer` fails closed on an unknown
principal (obligation Z5) and denies across tenants unconditionally (Z2), because
a reference adapter that granted broadly would make every test pass for the wrong
reason.
"""

from __future__ import annotations

from .authorization import (
    AUTHORIZER_FACTORIES,
    DenyAll,
    Grant,
    StaticAuthorizer,
)
from .transport import InProcessTransport, RecordingTransport

__all__ = [
    "AUTHORIZER_FACTORIES",
    "DenyAll",
    "Grant",
    "InProcessTransport",
    "RecordingTransport",
    "StaticAuthorizer",
]
