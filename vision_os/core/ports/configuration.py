"""Configuration ports — P23 ``ConfigSourcePort``, P24 ``SecretProviderPort``.

Owner: M16 Configuration Manager.

The Configuration Manager is the **only** component that reads the outside world
for settings. Every other module receives a validated, typed slice by injection,
so every module is constructible in a test with a literal config.

Secrets are resolved through a separate port and are never placed in the
configuration tree. A ``Camera`` record travels to config repositories, logs,
diagnostics and support bundles; a design where credentials are values guarantees
they eventually appear in a file that gets emailed to a vendor (12_SECURITY §9.1).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ConfigSourcePort(Protocol):
    """P23 — supply raw configuration documents.

    Implementations: file, environment, git, config service, Kubernetes
    ConfigMap, cloud parameter store.
    """

    @property
    def source_id(self) -> str:
        """Stable identifier used in value-origin reporting (``explain``)."""
        ...

    def load(self) -> dict[str, Any]:
        """Return the raw document for this layer.

        Raises:
            ConfigurationError: when the source exists but cannot be parsed. A
                *missing* optional source returns ``{}`` rather than raising.
        """
        ...


@runtime_checkable
class SecretProviderPort(Protocol):
    """P24 — resolve a secret reference to its value.

    Implementations: environment, file, vault, cloud secret manager.

    Implementations must never log, cache to disk, or include the resolved value
    in an exception message.
    """

    def resolve(self, reference: str) -> str:
        """Resolve ``reference``.

        Raises:
            SecretResolutionError: never containing the secret itself.
        """
        ...

    def has(self, reference: str) -> bool: ...
