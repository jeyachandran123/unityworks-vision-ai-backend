"""How a recorder password held in the database reaches the stream that needs it.

### The invariant this preserves

`RtspCameraConfig` carries a credential *reference*, never a password, so a
config can be logged, compared and put in an error message safely. The RTSP
source resolves the reference through a `SecretProvider` at the moment it dials,
and forgets the value when it closes. That design is correct and nothing here
changes it — this adds one more reference scheme it understands.

### `recorder:<id>`

A recorder whose password was typed into the application has its
`credential_ref` set to `recorder:<its id>` and the sealed password in its own
row. Two problems had to be bridged:

* **Sync versus async.** `SecretProvider.resolve` is synchronous and runs on the
  camera's own thread; the sealed value lives behind an async database session.
  So the application keeps a small store of *sealed* values, filled from the
  database whenever cameras are started or a password changes, and the provider
  opens one only inside `resolve`.
* **What sits in memory.** The store holds ciphertext. Plaintext exists only for
  the instant a URL is being built, exactly as it did when every password came
  from the environment.

One store and one provider serve the whole process, so the camera wall and the
analysis runtime can never disagree about whether a camera can authenticate.

### Fails as a missing credential, never as a crash

If a sealed value cannot be opened — no master key configured, the key changed,
the row was altered — `resolve` raises `MissingSecretError`, which the RTSP
source already treats as "no credential available" and reports on that camera.
One recorder with an unreadable password stops its own cameras and nobody
else's.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING

from app.domain.recorder_secrets import (
    MissingMasterKeyError,
    SealedSecret,
    SecretSealError,
    master_key,
    open_sealed,
)
from app.vision.secrets import MissingSecretError, SecretProvider

if TYPE_CHECKING:  # pragma: no cover
    from app.configuration.settings import Settings

RECORDER_SCHEME = "recorder:"


def recorder_reference(recorder_id: str) -> str:
    """The credential reference for a password sealed on this recorder's row."""
    return f"{RECORDER_SCHEME}{recorder_id}"


class RecorderCredentialStore:
    """Sealed recorder passwords by recorder id, and the key that opens them.

    Written from the API's event loop and read from camera threads. Every write
    replaces one immutable `SealedSecret` under a lock, and a read takes one
    reference, so a reader never sees half of an update.
    """

    __slots__ = ("_key", "_key_problem", "_lock", "_sealed")

    def __init__(self) -> None:
        self._sealed: dict[str, SealedSecret] = {}
        self._key: bytes | None = None
        self._key_problem = "the recorder master key has not been configured"
        self._lock = threading.Lock()

    # ── the key ──────────────────────────────────────────────────────────────

    def configure(self, settings: Settings) -> None:
        """Read the master key once. Its absence is recorded, not raised.

        Raising here would stop the application from starting on a deployment
        that keeps every password in the environment and never needs the key.
        Whether its absence is fatal depends on whether any recorder actually
        holds a sealed password — see `require_key_for`.
        """
        try:
            self._key = master_key(settings)
            self._key_problem = ""
        except SecretSealError as exc:
            self._key = None
            self._key_problem = str(exc)

    @property
    def key_available(self) -> bool:
        return self._key is not None

    def key(self) -> bytes:
        if self._key is None:
            raise MissingMasterKeyError(self._key_problem)
        return self._key

    def require_key_for(self, sealed_count: int) -> None:
        """Fail loudly when stored passwords exist and nothing can open them.

        Booting anyway would leave every camera on those recorders sitting at
        CONNECTING with an authentication error nobody reads. Refusing to start
        puts the missing setting in front of whoever is deploying.
        """
        if sealed_count and self._key is None:
            raise MissingMasterKeyError(
                f"{sealed_count} recorder(s) keep their password in the database, and "
                f"it cannot be read: {self._key_problem}"
            )

    # ── the values ───────────────────────────────────────────────────────────

    def replace(self, sealed: Mapping[str, SealedSecret]) -> None:
        with self._lock:
            self._sealed = dict(sealed)

    def put(self, recorder_id: str, sealed: SealedSecret | None) -> None:
        with self._lock:
            if sealed is None:
                self._sealed.pop(recorder_id, None)
            else:
                self._sealed[recorder_id] = sealed

    def __contains__(self, recorder_id: object) -> bool:
        return recorder_id in self._sealed

    def __len__(self) -> int:
        return len(self._sealed)

    def open(self, recorder_id: str) -> str:
        """The plaintext password, for the instant a URL is being built."""
        sealed = self._sealed.get(recorder_id)
        if sealed is None:
            raise MissingSecretError(f"recorder '{recorder_id}' has no stored password")
        return open_sealed(sealed, key=self.key())


class RecorderSecretProvider:
    """A `SecretProvider` that also understands `recorder:<id>`.

    Every other scheme — `env:`, `file:` — goes to the provider it wraps, so a
    deployment that keeps its passwords in the environment behaves exactly as
    it did before recorders existed.
    """

    __slots__ = ("_fallback", "_store")

    def __init__(self, store: RecorderCredentialStore, *, fallback: SecretProvider) -> None:
        self._store = store
        self._fallback = fallback

    @property
    def store(self) -> RecorderCredentialStore:
        return self._store

    def resolve(self, reference: str) -> str:
        candidate = (reference or "").strip()
        if not candidate.startswith(RECORDER_SCHEME):
            return self._fallback.resolve(candidate)

        recorder_id = candidate[len(RECORDER_SCHEME) :]
        try:
            return self._store.open(recorder_id)
        except MissingSecretError:
            raise
        except SecretSealError as exc:
            # Reported as a missing credential, which the RTSP source already
            # turns into a clear state on that one camera. The reason names the
            # recorder and the cause; it never carries a byte of the secret.
            raise MissingSecretError(
                f"the stored password for recorder '{recorder_id}' cannot be read: {exc}"
            ) from exc

    def has(self, reference: str) -> bool:
        try:
            self.resolve(reference)
            return True
        except Exception:  # noqa: BLE001 - `has` never raises, by contract
            return False


__all__ = [
    "RECORDER_SCHEME",
    "RecorderCredentialStore",
    "RecorderSecretProvider",
    "recorder_reference",
]
