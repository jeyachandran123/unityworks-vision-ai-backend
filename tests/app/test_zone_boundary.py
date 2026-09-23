"""Nothing in `app/` reaches for a site.

`restaurants` survives as a table until a later migration drops it, so that the
fold can be checked against the 1,990 incidents that were attributed through it.
Surviving is not the same as being readable: a query still joining sites would
quietly answer for the old shape of the estate, and the answer would look fine.

The rule is stated once, here, rather than in each of the nineteen modules that
used to name one.
"""

from __future__ import annotations

from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"

#: The one place a site may still be named: the model of the table that is
#: deliberately left standing, and the migration note that explains why.
ALLOWED = {Path("domain/models.py")}


def test_no_module_still_reaches_for_a_site() -> None:
    offenders: dict[str, list[int]] = {}
    for path in sorted(APP.rglob("*.py")):
        relative = path.relative_to(APP)
        if relative in ALLOWED:
            continue
        hits = [
            number
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if "restaurant_id" in line or "Restaurant" in line
        ]
        if hits:
            offenders[str(relative)] = hits
    assert offenders == {}, "these still name a site; placement is a zone now: " + ", ".join(
        f"{name}:{lines}" for name, lines in offenders.items()
    )


def test_the_model_that_is_allowed_to_name_one_only_defines_the_table() -> None:
    """`app/domain/models.py` keeps `Restaurant` so the table it maps still
    exists to be read by a person checking the fold. Nothing may query it."""
    text = (APP / "domain" / "models.py").read_text(encoding="utf-8")
    assert "class Restaurant(Base)" in text
    assert "zones: Mapped" not in text.split("class Zone(Base)")[0]
