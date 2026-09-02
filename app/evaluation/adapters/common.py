"""Shared reading rules every adapter obeys.

Three of them matter enough to live in one place:

**Paths never leave the repository.** Artifacts record things like
`config\\policies\\kitchen-safety.example.json` and `models\\yolov8n.onnx` —
Windows paths from whatever machine produced them. `repo_relative` and
`basename` strip them to something meaningful, so an API response describes a
policy or a weights file without revealing where a checkout lives.

**Reads are sandboxed to a fixed set of roots.** `read_json` refuses anything
resolving outside them, so no request parameter can ever become a file path. In
practice nothing here takes a path from a caller at all — the artifact set is a
constant — and this is the second lock on a door that has no handle.

**A missing or malformed artifact is a stated absence.** `read_json` returns
`None` and the adapter turns that into `available: false` with the reason. It
never returns `{}`, because an empty dict flows downstream as zeros.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: The repository root. `app/evaluation/adapters/common.py` → four levels up.
ROOT = Path(__file__).resolve().parents[3]

#: Everything an adapter may read, and nothing else. A path that does not resolve
#: inside one of these is refused rather than read.
ALLOWED_ROOTS: tuple[Path, ...] = (
    ROOT / "datasets",
    ROOT / "experiments",
    ROOT / "config",
    ROOT / "tests",
)

#: Size ceiling per artifact. `datasets/p9-live/review_queue.json` is 4 MB of
#: candidate rows with image paths in it; nothing here should be reading a file
#: that size, and the limit makes that a refusal rather than a slow request.
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024


def repo_relative(path: Path) -> str:
    """A repository-relative POSIX path, for provenance. Never absolute."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:  # pragma: no cover - only if an adapter is misconfigured
        return path.name


def basename(recorded: str) -> str:
    """The last component of a path an artifact recorded.

    `models\\yolov8n-pose.onnx` → `yolov8n-pose.onnx`. The weights file is the
    fact worth surfacing; the directory it sat in on somebody's laptop is not,
    and it is the kind of detail that ends up in a screenshot.
    """
    if not recorded:
        return ""
    return recorded.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def read_json(path: Path) -> Any | None:
    """Read one artifact, or `None` with nothing invented.

    Returns `None` for missing, oversized, unreadable and malformed alike. The
    caller turns that into a stated reason; what it must never do is proceed with
    an empty structure, because a dashboard cannot tell zeros apart from a file
    that would not parse.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return None

    if not any(_within(resolved, root) for root in ALLOWED_ROOTS):
        return None
    if not resolved.is_file():
        return None
    try:
        if resolved.stat().st_size > MAX_ARTIFACT_BYTES:
            return None
        return json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def parse_instant(raw: object) -> datetime | None:
    """An ISO-8601 instant recorded *inside* an artifact, or `None`.

    Deliberately has no filesystem fallback. A file's modification time records
    when git touched it, not when anybody measured anything, and using one as an
    evaluation date would manufacture provenance — the one thing this phase must
    not do.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def ratio(value: object) -> float | None:
    """A float, or `None` for a metric the artifact left undefined.

    `null` in these artifacts means *no support*, and
    `experiments/vlm_prompt/score.py` is explicit that it "reports `null`, never
    `0.0` — a metric with no support is undefined, not bad". This preserves that
    distinction across the wire; coercing to `0.0` here would erase it.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def whole(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return int(value)
    return None


__all__ = [
    "ALLOWED_ROOTS",
    "MAX_ARTIFACT_BYTES",
    "ROOT",
    "basename",
    "parse_instant",
    "ratio",
    "read_json",
    "repo_relative",
    "whole",
]
