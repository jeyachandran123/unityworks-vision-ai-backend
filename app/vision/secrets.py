"""Secret resolution, at the application boundary.

Vision OS declares `SecretProviderPort` and has never had an implementation. This
is the minimum one, and it lives **here** rather than inside the platform on
purpose: the platform receives *resolved* configuration and never learns how a
secret is fetched. Teaching `vision_os` about environment variables, files or a
vault would give it a dependency on a deployment's infrastructure, which is
precisely what the port exists to prevent.

### References, not values

A camera row stores `credential_ref`, never a password:

    env:CCTV_PASSWORD          an environment variable
    file:/run/secrets/dvr_pw   a mounted secret file, trailing newline stripped
    literal:hunter2            development only, and it says so

Adding a `vault:` scheme later is a new resolver here and nothing anywhere else.

### What this module refuses to do

It never logs a resolved value, never puts one in an exception message, and
never returns one from anything that reaches an API response. `MissingSecretError`
names the *reference*, which is safe — that a secret is called
`env:CCTV_PASSWORD` is not itself a secret — and never the value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

LITERAL_SCHEME = "literal:"
ENV_SCHEME = "env:"
FILE_SCHEME = "file:"


class SecretResolutionError(RuntimeError):
    """A reference could not be resolved. Names the reference, never a value."""


class MissingSecretError(SecretResolutionError):
    """The reference is well-formed and the secret is not there."""


@runtime_checkable
class SecretProvider(Protocol):
    """The application-side shape of Vision OS's `SecretProviderPort`."""

    def resolve(self, reference: str) -> str: ...
    def has(self, reference: str) -> bool: ...


class EnvironmentSecretProvider:
    """Resolves `env:`, `file:` and `literal:` references.

    The provider a single-node deployment needs, and the seam a vault-backed one
    replaces without touching a caller.
    """

    __slots__ = ("_environ",)

    def __init__(self, environ: dict[str, str] | None = None) -> None:
        # Injectable so tests never mutate the process environment, and so a
        # future provider can be composed rather than monkey-patched.
        import os

        self._environ = os.environ if environ is None else environ

    def resolve(self, reference: str) -> str:
        candidate = (reference or "").strip()
        if not candidate:
            raise MissingSecretError("no credential reference was configured")

        if candidate.startswith(LITERAL_SCHEME):
            value = candidate[len(LITERAL_SCHEME) :]
            if not value:
                raise MissingSecretError(f"'{_safe(candidate)}' resolves to an empty value")
            return value

        if candidate.startswith(ENV_SCHEME):
            name = candidate[len(ENV_SCHEME) :].strip()
            value = self._environ.get(name, "")
            if not value:
                raise MissingSecretError(
                    f"environment variable '{name}' is unset or empty"
                )
            return value

        if candidate.startswith(FILE_SCHEME):
            path = Path(candidate[len(FILE_SCHEME) :].strip())
            if not path.is_file():
                raise MissingSecretError(f"secret file '{path}' does not exist")
            # A file written by `echo` carries a newline that is not part of the
            # password. Stripping it here saves an authentication failure that
            # looks exactly like a wrong password.
            value = path.read_text(encoding="utf-8").strip()
            if not value:
                raise MissingSecretError(f"secret file '{path}' is empty")
            return value

        raise SecretResolutionError(
            f"unsupported credential reference scheme in '{_safe(candidate)}'; "
            f"expected one of env:, file:, literal:"
        )

    def has(self, reference: str) -> bool:
        """Whether the reference resolves. Never raises, never returns the value."""
        try:
            self.resolve(reference)
            return True
        except SecretResolutionError:
            return False


def _safe(reference: str) -> str:
    """A reference safe to put in an error message.

    A `literal:` reference *contains* the secret, so only the scheme survives.
    Every other scheme names a location, which is not sensitive.
    """
    return LITERAL_SCHEME + "***" if reference.startswith(LITERAL_SCHEME) else reference


__all__ = [
    "ENV_SCHEME",
    "EnvironmentSecretProvider",
    "FILE_SCHEME",
    "LITERAL_SCHEME",
    "MissingSecretError",
    "SecretProvider",
    "SecretResolutionError",
]
