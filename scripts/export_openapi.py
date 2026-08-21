"""Export the OpenAPI schema to a file.

    python scripts/export_openapi.py [--out docs/api/openapi.json] [--check]

### Why a file rather than a live endpoint

`/openapi.json` is mounted only when `APP_DEBUG=true`, because a published schema
is a map of the attack surface and a production deployment should not hand one
out. But the frontend needs the schema to generate its types, and pointing a
build step at a debug-only endpoint would mean either running the API in debug to
build the UI, or exposing it in production. Neither is acceptable.

So the schema is **exported and committed**. The build reads a file; the API
serves nothing extra.

### Drift

`--check` regenerates and compares against the committed copy, exiting non-zero
if they differ. Run it in CI: a route added without regenerating is a frontend
that types against an API that no longer exists.

    python scripts/export_openapi.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs" / "api" / "openapi.json"


def build_schema() -> dict:
    """Generate the schema from a fully-configured application.

    DevTools is mounted deliberately: those routes are part of the contract the
    frontend types against, and their absence from the schema would leave the
    DevTools client hand-writing the very types this replaces. Whether they are
    *reachable* remains a deployment decision, and it is enforced at request
    time by `FEATURE_DEVTOOLS` and `ACCESS_DEVTOOLS` — not by hiding them here.
    """
    from app.configuration.settings import Settings
    from app.main import create_app

    settings = Settings(
        app_env="development",
        app_debug=True,
        feature_devtools=True,
        redis_enabled=False,
        metrics_enabled=False,
        secret_key="schema-export-only-never-used-to-sign-anything",
    )
    return create_app(settings).openapi()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed schema is out of date; changes nothing",
    )
    args = parser.parse_args()

    # `sort_keys` and a trailing newline so the file is diffable and a
    # regeneration that changed nothing produces no diff at all.
    rendered = json.dumps(build_schema(), indent=2, sort_keys=True) + "\n"

    if args.check:
        if not args.out.is_file():
            print(f"{args.out} does not exist; run without --check", file=sys.stderr)
            return 1
        if args.out.read_text(encoding="utf-8") != rendered:
            print(
                f"{args.out} is out of date.\n"
                f"Run: python scripts/export_openapi.py",
                file=sys.stderr,
            )
            return 1
        print(f"{args.out} is up to date")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    paths = len(json.loads(rendered).get("paths", {}))
    print(f"wrote {args.out} — {paths} paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
