"""How each family of recorder wants its stream URL built.

The brands themselves live in `config/recorders/brands.json`, beside the
semantic policies and compliance rules, because which devices this deployment
can talk to is configuration rather than logic. Adding another vendor is an edit
to that file.

A brand carries **two** facts. `path_template` is the shape of the path, and
`main`/`sub` are the integers that vendor uses for full-resolution and reduced
streams. They genuinely differ — Dahua's main stream is `subtype=0`, Hikvision's
is `1` — so a template on its own would leave whoever adds a recorder supplying
the number from memory. That is the sort of mistake which does not fail at save
time; it fails later as a decoder error on a camera that looks correctly
configured.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from app.configuration.settings import REPO_ROOT

#: The escape hatch. A device matching nothing in the file carries its own
#: template on the recorder row instead of a key into this one.
CUSTOM = "custom"

BRANDS_PATH = REPO_ROOT / "config" / "recorders" / "brands.json"

STREAM_TYPES = ("main", "sub")


class BrandError(ValueError):
    """A brand, channel or stream type that cannot produce a usable path."""


@dataclass(frozen=True, slots=True)
class Brand:
    """One family of recorder, and everything needed to address a channel on it."""

    key: str
    label: str
    path_template: str
    #: The integer this vendor uses for the full-resolution stream.
    main: int
    #: …and for the reduced one.
    sub: int
    default_port: int

    def __post_init__(self) -> None:
        if "{channel}" not in self.path_template:
            # A template without the channel builds one identical URL for every
            # channel on the recorder. That connects, and shows the wrong
            # camera — worse than failing, because nothing reports it.
            raise BrandError(
                f"brand '{self.key}' has no '{{channel}}' in its path template, so every "
                "channel would resolve to the same stream"
            )
        if self.main == self.sub:
            raise BrandError(
                f"brand '{self.key}' uses {self.main} for both main and sub, so the "
                "stream choice on a camera would have no effect"
            )
        if self.default_port < 1:
            raise BrandError(f"brand '{self.key}' has no usable default port")
        if not self.label:
            raise BrandError(f"brand '{self.key}' has no label to show in a list")

    def subtype(self, stream_type: str) -> int:
        if stream_type not in STREAM_TYPES:
            raise BrandError(
                f"stream_type must be one of {list(STREAM_TYPES)}, not {stream_type!r}; "
                "defaulting to main would pull full-resolution video on every channel, "
                "which is a bandwidth decision nobody made"
            )
        return self.main if stream_type == "main" else self.sub

    def path(self, *, channel: int, stream_type: str) -> str:
        """The RTSP path for one channel of this recorder."""
        if channel < 1:
            raise BrandError("channel numbering starts at 1 on every supported device")
        return self.path_template.format(channel=channel, subtype=self.subtype(stream_type))


def custom_brand(
    *,
    path_template: str,
    main: int,
    sub: int,
    default_port: int = 554,
    label: str = "Custom",
) -> Brand:
    """A brand defined on a recorder row rather than in the file."""
    return Brand(
        key=CUSTOM,
        label=label,
        path_template=path_template,
        main=main,
        sub=sub,
        default_port=default_port,
    )


@functools.lru_cache(maxsize=1)
def load_brands() -> Mapping[str, Brand]:
    """Every brand this deployment knows, read once.

    Read-only to callers: a mapping that could be edited would change how every
    later recorder builds its URL, from anywhere in the process.
    """
    try:
        raw = json.loads(BRANDS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - a broken deployment
        raise BrandError(
            f"no recorder brands at {BRANDS_PATH}; no recorder could be addressed"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BrandError(f"recorder brands at {BRANDS_PATH} are not valid JSON: {exc}") from exc

    brands: dict[str, Brand] = {}
    for key, entry in raw.items():
        # Keys beginning with an underscore are commentary. JSON has no comments
        # and the file needs to explain itself to whoever adds the next device.
        if key.startswith("_"):
            continue
        if key == CUSTOM:
            raise BrandError(
                f"'{CUSTOM}' is reserved for a template held on a recorder row and "
                "cannot be declared in the brands file"
            )
        try:
            brands[key] = Brand(
                key=key,
                label=str(entry["label"]),
                path_template=str(entry["path_template"]),
                main=int(entry["main"]),
                sub=int(entry["sub"]),
                default_port=int(entry.get("default_port", 554)),
            )
        except KeyError as exc:
            raise BrandError(f"brand '{key}' is missing {exc.args[0]!r}") from exc

    if not brands:
        raise BrandError(f"recorder brands at {BRANDS_PATH} declare no usable brand")
    return MappingProxyType(brands)


def brand_for(key: str) -> Brand:
    """One brand by key, with a message naming what is available.

    Separate from `load_brands()[key]` so a stale key stored on a recorder row —
    a brand removed from the file after that row was written — reports the
    actual problem rather than a bare `KeyError`.
    """
    brands = load_brands()
    try:
        return brands[key]
    except KeyError as exc:
        raise BrandError(
            f"unknown recorder brand '{key}'; this deployment knows "
            f"{sorted(brands)} and '{CUSTOM}'"
        ) from exc


__all__ = [
    "BRANDS_PATH",
    "CUSTOM",
    "STREAM_TYPES",
    "Brand",
    "BrandError",
    "brand_for",
    "custom_brand",
    "load_brands",
]
