"""A recorder's password is the most confidential thing this application stores.

Two properties matter more than the rest, and both are here as tests rather than
as comments in the module:

1. It round-trips exactly. A password that comes back subtly different is worse
   than one that fails, because the failure surfaces as a DVR refusing a login
   and sends whoever is debugging it to the wrong place entirely.
2. A wrong key **raises**. AES-GCM is authenticated, which is the reason it was
   chosen over a cipher that would decrypt happily into garbage and hand that
   garbage to a recorder as a password.
"""

from __future__ import annotations

import base64

import pytest
from pydantic import SecretStr

from app.domain.recorder_secrets import (
    MissingMasterKeyError,
    SealedSecret,
    SecretSealError,
    master_key,
    open_sealed,
    seal,
)

KEY = bytes(range(32))
OTHER_KEY = bytes(reversed(range(32)))


def test_a_password_survives_the_round_trip_exactly() -> None:
    sealed = seal("Dvr@Gayathri#2026", key=KEY)

    assert open_sealed(sealed, key=KEY) == "Dvr@Gayathri#2026"


def test_a_password_with_awkward_characters_survives_too() -> None:
    """DVR passwords are frequently pasted from a label and contain anything.
    An encoding that mangled one would present as a credential failure."""
    awkward = "päss/wörd:with@symbols&=?# and spaces"

    assert open_sealed(seal(awkward, key=KEY), key=KEY) == awkward


def test_the_ciphertext_does_not_contain_the_password() -> None:
    sealed = seal("hunter2hunter2", key=KEY)

    assert b"hunter2" not in sealed.ciphertext


def test_sealing_the_same_password_twice_gives_different_ciphertext() -> None:
    """A fresh nonce every time.

    Identical ciphertext would let anyone with read access to the table see
    which recorders share a password — which is a real fact about the estate,
    leaked without decrypting anything.
    """
    first, second = seal("same", key=KEY), seal("same", key=KEY)

    assert first.nonce != second.nonce
    assert first.ciphertext != second.ciphertext


def test_the_wrong_key_raises_rather_than_returning_plausible_bytes() -> None:
    sealed = seal("correct horse", key=KEY)

    with pytest.raises(SecretSealError):
        open_sealed(sealed, key=OTHER_KEY)


def test_tampered_ciphertext_is_refused() -> None:
    """Authenticated encryption, and this is what that buys: a row edited in the
    database cannot be made to decrypt into an attacker's chosen password."""
    sealed = seal("correct horse", key=KEY)
    flipped = SealedSecret(
        ciphertext=sealed.ciphertext[:-1] + bytes([sealed.ciphertext[-1] ^ 0x01]),
        nonce=sealed.nonce,
        key_id=sealed.key_id,
    )

    with pytest.raises(SecretSealError):
        open_sealed(flipped, key=KEY)


def test_a_swapped_nonce_is_refused() -> None:
    first = seal("first", key=KEY)
    second = seal("second", key=KEY)

    with pytest.raises(SecretSealError):
        open_sealed(SealedSecret(first.ciphertext, second.nonce, first.key_id), key=KEY)


def test_the_key_id_travels_with_the_secret() -> None:
    """Rotation later has to know which key sealed which row. Without this,
    rotating means re-typing every password, which in practice means never
    rotating at all."""
    assert seal("x", key=KEY).key_id == "k1"
    assert seal("x", key=KEY, key_id="k2").key_id == "k2"


def test_an_empty_password_is_refused() -> None:
    """An empty password seals and opens perfectly well, and then builds a URL
    that the recorder rejects — reported as a connection failure rather than as
    the missing credential it is."""
    with pytest.raises(SecretSealError, match="empty"):
        seal("", key=KEY)


def test_a_missing_master_key_is_an_error_not_an_empty_key(settings) -> None:
    without = settings.model_copy(update={"recorder_secret_key": SecretStr("")})

    with pytest.raises(MissingMasterKeyError):
        master_key(without)


def test_a_master_key_of_the_wrong_length_is_refused(settings) -> None:
    short = settings.model_copy(
        update={"recorder_secret_key": SecretStr(base64.b64encode(b"tooshort").decode())}
    )

    with pytest.raises(SecretSealError, match="32"):
        master_key(short)


def test_a_master_key_that_is_not_base64_says_so(settings) -> None:
    """Told apart from a short key deliberately: one is a truncated paste and the
    other is a raw value pasted where an encoded one belongs, and the fixes
    differ."""
    broken = settings.model_copy(update={"recorder_secret_key": SecretStr("not base64 at all!!")})

    with pytest.raises(SecretSealError, match="base64"):
        master_key(broken)


def test_the_configured_master_key_round_trips_a_password(settings) -> None:
    """The whole path as the application uses it, rather than with a literal."""
    key = master_key(settings)

    assert open_sealed(seal("Dvr@Gayathri#2026", key=key), key=key) == "Dvr@Gayathri#2026"
