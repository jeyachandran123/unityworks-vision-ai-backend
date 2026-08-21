"""Password and API-key handling.

bcrypt directly rather than through passlib: the reference backend already
imported bcrypt itself, passlib has been unmaintained for years, and a wrapper
that adds no capability adds only a dependency.
"""

from __future__ import annotations

import hashlib
import secrets

import bcrypt

from app.errors import ValidationError

#: bcrypt silently truncates at 72 bytes. Rejecting rather than truncating means
#: a 200-character passphrase is never quietly reduced to its first 72 bytes,
#: which would make two different passphrases interchangeable.
MAX_PASSWORD_BYTES = 72

API_KEY_PREFIX = "uwv"


def hash_password(password: str, *, min_length: int = 12) -> str:
    """Hash a password with a per-password salt.

    Raises:
        ValidationError: too short, or longer than bcrypt can represent.
    """
    if len(password) < min_length:
        raise ValidationError(
            f"password must be at least {min_length} characters",
            details={"min_length": min_length},
        )
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise ValidationError(
            f"password must be at most {MAX_PASSWORD_BYTES} bytes; bcrypt "
            f"truncates beyond that, which would silently weaken it"
        )
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Check a password against its hash. Never raises on malformed input.

    A corrupt or empty stored hash is a failed verification, not a 500. Letting
    it raise would turn a data problem into an availability problem, and would
    distinguish "no such user" from "corrupt record" to an attacker.
    """
    if not plain or not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def generate_api_key() -> tuple[str, str]:
    """Return ``(raw_key, stored_hash)``.

    The raw key is shown to the user **once** and never stored. Only the SHA-256
    hash is persisted, so a database disclosure does not yield usable keys.

    SHA-256 rather than bcrypt here on purpose: an API key is 256 bits of
    generated entropy, not a human-chosen secret, so it is not brute-forceable
    and does not need a slow hash. A slow hash on every API request would be a
    self-inflicted rate limit.
    """
    raw = f"{API_KEY_PREFIX}_{secrets.token_urlsafe(32)}"
    return raw, hash_api_key(raw)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def api_keys_match(presented: str, stored_hash: str) -> bool:
    """Constant-time comparison, so response timing reveals no prefix."""
    return secrets.compare_digest(hash_api_key(presented), stored_hash)


__all__ = [
    "API_KEY_PREFIX",
    "MAX_PASSWORD_BYTES",
    "api_keys_match",
    "generate_api_key",
    "hash_api_key",
    "hash_password",
    "verify_password",
]
