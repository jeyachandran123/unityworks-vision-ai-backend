"""Sealing a recorder's password, so the database alone recovers nothing.

A recorder that can be onboarded through the application has to keep its
password somewhere the application can reach. Storing it in plaintext would make
a database dump — or a backup, or a support export — a list of working
credentials for every camera on the estate.

So the value is sealed with **AES-GCM** and the key lives in the environment.
The row and the key are two halves: whoever holds only the database holds
nothing usable.

AES-GCM rather than a plain cipher because it is *authenticated*. A wrong key,
an edited row or a swapped nonce raises instead of decrypting into garbage, and
garbage handed to a DVR as a password surfaces as a login failure — sending
whoever debugs it to the credential rather than to the row that was tampered
with.

This module is pure: given a key it seals and opens, and it knows nothing about
sessions, recorders or HTTP. `app/domain/recorders.py` decides when to call it.
"""

from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, types only
    from app.configuration.settings import Settings

#: AES-GCM's standard nonce width. Twelve bytes is what the mode is defined for;
#: other lengths are permitted by the API and weaken it.
NONCE_BYTES = 12

#: AES-256. The key is a fixed width, so a wrong one is caught at load time
#: rather than at the first connection attempt.
KEY_BYTES = 32


class SecretSealError(RuntimeError):
    """A password could not be sealed or opened.

    Deliberately not carrying the underlying library's message: it quotes
    internals that are noise in a log, and on the open path the one thing worth
    saying is that authentication failed.
    """


class MissingMasterKeyError(SecretSealError):
    """No master key is configured, so nothing sealed can be read.

    Its own type because the remedy is different from every other failure here:
    a deployment step was missed, rather than a value being wrong.
    """


@dataclass(frozen=True, slots=True)
class SealedSecret:
    """A password at rest. Never a password in use."""

    ciphertext: bytes
    nonce: bytes
    #: Which master key sealed this. Carried so a rotation can tell rows apart
    #: without re-typing every password.
    key_id: str

    def __repr__(self) -> str:
        # A sealed secret is still a secret's shadow, and this object ends up in
        # tracebacks and debugger output. Nothing about the bytes is printed.
        return f"SealedSecret(key_id={self.key_id!r}, bytes={len(self.ciphertext)})"


def seal(plaintext: str, *, key: bytes, key_id: str = "k1") -> SealedSecret:
    """Seal a password for storage.

    A fresh nonce every time, so sealing one password twice gives different
    ciphertext. Without that, anyone with read access to the table could see
    which recorders share a password — a real fact about the estate, leaked
    without decrypting anything.
    """
    if not plaintext:
        # An empty password seals and opens perfectly well, then builds a URL the
        # recorder rejects — reported as a connection failure rather than as the
        # missing credential it is.
        raise SecretSealError("refusing to seal an empty password")
    _check_key(key)

    nonce = os.urandom(NONCE_BYTES)
    try:
        ciphertext = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    except Exception as exc:  # noqa: BLE001 - never leak the library's message
        raise SecretSealError(f"could not seal the password: {type(exc).__name__}") from exc
    return SealedSecret(ciphertext=ciphertext, nonce=nonce, key_id=key_id)


def open_sealed(sealed: SealedSecret, *, key: bytes) -> str:
    """Recover a password, for the instant a stream URL is being built."""
    _check_key(key)
    try:
        plaintext = AESGCM(key).decrypt(sealed.nonce, sealed.ciphertext, None)
    except InvalidTag as exc:
        # The interesting case, and worth naming precisely: the key is wrong, or
        # the row was edited. Both mean "do not trust this value", and neither
        # means "the DVR refused us".
        raise SecretSealError(
            f"could not open the recorder credential sealed with key '{sealed.key_id}': "
            "the master key does not match, or the stored value was altered"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - never leak the library's message
        raise SecretSealError(f"could not open the password: {type(exc).__name__}") from exc
    return plaintext.decode("utf-8")


def master_key(settings: Settings) -> bytes:
    """The configured master key, or a refusal that says what to do about it."""
    configured = settings.recorder_secret_key.get_secret_value().strip()
    if not configured:
        raise MissingMasterKeyError(
            "RECORDER_SECRET_KEY is not set, so no recorder credential held in the "
            "database can be read. Set it to base64 of 32 random bytes."
        )

    try:
        key = base64.b64decode(configured, validate=True)
    except (binascii.Error, ValueError) as exc:
        # Told apart from a short key on purpose: this is a raw value pasted
        # where an encoded one belongs, and the fix is to encode it.
        raise SecretSealError(
            "RECORDER_SECRET_KEY is not valid base64; it must be base64 of 32 random bytes"
        ) from exc

    _check_key(key)
    return key


def _check_key(key: bytes) -> None:
    if len(key) != KEY_BYTES:
        raise SecretSealError(
            f"a recorder master key must be exactly {KEY_BYTES} bytes "
            f"(base64 of 32 random bytes); got {len(key)}"
        )


__all__ = [
    "KEY_BYTES",
    "NONCE_BYTES",
    "MissingMasterKeyError",
    "SealedSecret",
    "SecretSealError",
    "master_key",
    "open_sealed",
    "seal",
]
