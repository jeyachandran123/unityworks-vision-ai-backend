"""Loading the verification document — a composition-time act.

The sibling of ``semantic_policy``, and separated from the policy it configures
for the reason M8's architecture test states plainly:

> *"A vocabulary guard can be worked around by naming a method something else; an
> import guard cannot, because writing a file requires reaching for something
> that writes files."*

The crop path may not import ``pathlib`` or ``os`` anywhere, so a loader inside
``adapters.cropping`` would fail that check no matter how careful it was — and
correctly, because the check is defending something real: M8 holds no durable
store, and a module that can read a file is one frame away from writing one.

So the split is clean. ``VerificationRules.from_document`` is a pure function of
a parsed mapping and lives with the policy. Everything that touches a filesystem
or an environment lives here, runs once at composition, and hands the policy a
value.

### No rules is a supported configuration

Absent a document: no verification policy is built, the trigger policy is the
unwrapped one, and no corroborating model call is ever made. That is a platform
without an identity-sensitive use case, not a broken one — the same stance
``load_policy`` takes for semantics.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from ...core.errors import ConfigurationError
from ..cropping.verification import VerificationRules

#: Where a deployment names its verification rules. A path, or empty for none.
RULES_ENV = "VISION_VERIFICATION_RULES"


def rules_from_file(path: Path | str) -> VerificationRules:
    """Parse a rules document.

    Raises:
        ConfigurationError: the file is missing or is not valid JSON. Both are
            composition-time failures and neither falls back to an empty rule
            set: a deployment that believed it had configured corroboration and
            silently got none would spend nothing and notice nothing.
    """
    file = Path(path)
    if not file.is_file():
        raise ConfigurationError(f"verification rules not found at '{file}'")
    try:
        document = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"'{file}' is not valid JSON: {exc}") from exc
    return VerificationRules.from_document(document)


def load_verification_rules(
    path: Path | str | None = None, *, env: Mapping[str, str] | None = None
) -> VerificationRules | None:
    """The active rules, or ``None``.

    ``None`` is a supported configuration and never an error.
    """
    source = os.environ if env is None else env
    chosen = str(path or source.get(RULES_ENV, "")).strip()
    if not chosen:
        return None
    return rules_from_file(chosen)


__all__ = ["RULES_ENV", "load_verification_rules", "rules_from_file"]
