"""P23/P24 — configuration source and secret provider adapters.

Secrets never enter the configuration tree. A ``Camera`` record travels to config
repositories, logs, diagnostics and support bundles; a design where credentials
are values guarantees they eventually appear in a file that gets emailed to a
vendor (12_SECURITY §9.1).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from ...core.errors import ConfigurationError, SecretResolutionError


class InMemoryConfigSource:
    """A literal document. The default for tests and embedded deployments."""

    __slots__ = ("_source_id", "_document", "_lock")

    def __init__(self, document: dict[str, Any], *, source_id: str = "in-memory") -> None:
        self._source_id = source_id
        self._document = document
        self._lock = threading.Lock()

    @property
    def source_id(self) -> str:
        return self._source_id

    def load(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._document, default=str))

    def replace(self, document: dict[str, Any]) -> None:
        """Swap the document, to exercise hot reload."""
        with self._lock:
            self._document = document


class JsonFileConfigSource:
    """A JSON document on disk.

    A *missing optional* file returns ``{}`` rather than raising; a file that
    exists but cannot be parsed raises. Collapsing those two cases is how a
    typo'd config silently becomes an empty one.
    """

    __slots__ = ("_source_id", "_path", "_required")

    def __init__(self, path: Path | str, *, required: bool = True, source_id: str | None = None):
        self._path = Path(path)
        self._required = required
        self._source_id = source_id or f"file:{self._path}"

    @property
    def source_id(self) -> str:
        return self._source_id

    def load(self) -> dict[str, Any]:
        if not self._path.exists():
            if self._required:
                raise ConfigurationError(
                    f"required config file not found: {self._path}", path=str(self._path)
                )
            return {}
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(
                f"cannot read config file {self._path}: {exc}", path=str(self._path)
            ) from exc
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"malformed JSON in {self._path} at line {exc.lineno}: {exc.msg}",
                path=str(self._path),
            ) from exc
        if not isinstance(document, dict):
            raise ConfigurationError(
                f"config root must be an object, got {type(document).__name__}",
                path=str(self._path),
            )
        return document


class InMemorySecretProvider:
    """Secrets held in process memory.

    Never logs, never writes to disk, and never includes a resolved value in an
    exception message.
    """

    __slots__ = ("_secrets",)

    def __init__(self, secrets: dict[str, str] | None = None) -> None:
        self._secrets = dict(secrets or {})

    def resolve(self, reference: str) -> str:
        try:
            return self._secrets[reference]
        except KeyError:
            raise SecretResolutionError(
                f"unknown secret reference '{reference}'", reference=reference
            ) from None

    def has(self, reference: str) -> bool:
        return reference in self._secrets

    def put(self, reference: str, value: str) -> None:
        self._secrets[reference] = value


class EnvironmentSecretProvider:
    """Resolve references from environment variables."""

    __slots__ = ("_prefix",)

    def __init__(self, *, prefix: str = "UWV_SECRET_") -> None:
        self._prefix = prefix

    def _key(self, reference: str) -> str:
        return f"{self._prefix}{reference.upper().replace('-', '_').replace('.', '_')}"

    def resolve(self, reference: str) -> str:
        value = os.environ.get(self._key(reference))
        if value is None:
            raise SecretResolutionError(
                f"secret reference '{reference}' is not present in the environment",
                reference=reference,
            )
        return value

    def has(self, reference: str) -> bool:
        return self._key(reference) in os.environ
