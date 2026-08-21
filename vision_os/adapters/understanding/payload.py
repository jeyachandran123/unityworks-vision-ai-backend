"""Shared payload work for VLM understanders: encoding in, schema discipline out.

Two adapters — one hosted, one local — face the same three problems: a crop has
to become an image a model will accept, an answer has to be recovered from text
that was never constrained, and the result has to be split against the schema
the platform declared. Solving them twice would let the copies drift, and the
one that drifts is the one that starts letting undeclared fields through.

Nothing here knows a provider, a model or an attribute name.
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

from ...core.model.ids import AttributeKey


def split_by_schema(
    decoded: Mapping[str, Any], schema
) -> tuple[dict[str, Any], str | None]:
    """U1 — declared fields in, everything else to ``unparsed``.

    An adapter never invents a schema and never silently drops what a model
    volunteered. A key outside the schema is preserved as evidence rather than
    discarded, so a model that has started answering a different question is
    discoverable instead of merely ignored.

    This is the *first* of two gates. ``AttributeValidator`` checks the survivors
    again against the registry, because this one runs inside adapter code that a
    vendor integration could get wrong.
    """
    declared: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in decoded.items():
        if schema.declares(AttributeKey(key)):
            declared[key] = value
        else:
            extra[key] = value
    return declared, (json.dumps(extra, default=str) if extra else None)


def extract_json(text: str) -> dict[str, Any] | None:
    """Recover a JSON object from prose. ``None`` when there is none.

    Needed wherever decoding is not constrained: the model is asked for JSON and
    usually complies, but wraps it in a fence or a sentence often enough to
    matter. Recovering it is legitimate; *inventing* it is not, so anything
    unparseable returns ``None`` and the caller records the answer as unparseable
    with the original text preserved (U2, U3).
    """
    stripped = text.strip()
    if not stripped:
        return None

    if stripped.startswith("```"):
        parts = stripped.split("```")
        stripped = parts[1] if len(parts) > 1 else stripped[3:]
        if stripped.lstrip().lower().startswith("json"):
            stripped = stripped.lstrip()[4:]

    try:
        decoded = json.loads(stripped)
        return decoded if isinstance(decoded, dict) else None
    except json.JSONDecodeError:
        pass

    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        decoded = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def encode_png_base64(
    pixels, width: int, height: int, *, colour_space: str = "bgr24", max_side: int = 448
) -> str:
    """BGR24 crop -> base64 PNG, downscaled to ``max_side``.

    Hand-rolled rather than via Pillow or OpenCV so understanding adapters carry
    **no new dependency**: a VLM adapter that could not run without an imaging
    library would be a worse trade than fifty lines of zlib.

    Downscaling is nearest-neighbour and matters more than it looks: vision
    tokens scale with *area*, so halving the longest side quarters the prompt
    cost of every call the platform makes.
    """
    data = bytes(pixels)
    stride = width * 3
    expected = stride * height
    if len(data) < expected:
        raise ValueError(f"crop is {len(data)}B, expected {expected}B for {width}x{height}")

    scale = max(1, (max(width, height) + max_side - 1) // max_side)
    out_w, out_h = max(1, width // scale), max(1, height // scale)

    rows: list[bytes] = []
    swap = colour_space.startswith("bgr")
    for y in range(out_h):
        src_row = (y * scale) * stride
        row = bytearray(out_w * 3)
        for x in range(out_w):
            offset = src_row + (x * scale) * 3
            b, g, r = data[offset], data[offset + 1], data[offset + 2]
            target = x * 3
            if swap:
                row[target], row[target + 1], row[target + 2] = r, g, b
            else:
                row[target], row[target + 1], row[target + 2] = b, g, r
        rows.append(bytes(row))

    return base64.b64encode(_png(out_w, out_h, rows)).decode("ascii")


def _png(width: int, height: int, rgb_rows: Sequence[bytes]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanlines = b"".join(b"\x00" + row for row in rgb_rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines, 6))
        + chunk(b"IEND", b"")
    )


__all__ = ["encode_png_base64", "extract_json", "split_by_schema"]
