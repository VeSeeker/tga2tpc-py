"""TXI (texture information) parsing and rendering."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TXIInfo:
    text: str
    is_animated: bool = False
    is_cube: bool = False
    is_bumpmap: bool = False
    compress_texture_false: bool = False
    has_proceduretype: bool = False
    numx: int | None = None
    numy: int | None = None
    default_width: int | None = None
    default_height: int | None = None
    alpha_blending: float | None = None


def _search_int(text: str, keyword: str) -> int | None:
    match = re.search(rf"^\s*{keyword}\s+(\d+)", text, re.IGNORECASE | re.MULTILINE)
    return int(match.group(1)) if match else None


def parse_txi(text: str) -> TXIInfo:
    """Parse the TXI directives that affect TPC layout."""

    if not text:
        return TXIInfo(text="")

    animated = bool(
        re.search(r"^\s*proceduretype\s+cycle", text, re.IGNORECASE | re.MULTILINE)
    )
    cube = bool(re.search(r"^\s*cube\s+1", text, re.IGNORECASE | re.MULTILINE))
    bumpmap = bool(
        re.search(r"^\s*isbumpmap\s+[1TYty]", text, re.IGNORECASE | re.MULTILINE)
    )
    compress_false = bool(
        re.search(
            r"^\s*compresstexture\s+[0FNfn]", text, re.IGNORECASE | re.MULTILINE
        )
    )
    has_proceduretype = bool(
        re.search(r"^\s*proceduretype", text, re.IGNORECASE | re.MULTILINE)
    )

    alpha_blending: float | None = None
    match = re.search(
        r"^\s*#\s*alphablending\s+([.0-9]+)",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if match:
        try:
            value = float(match.group(1))
        except ValueError:
            value = math.nan
        if math.isfinite(value):
            alpha_blending = value

    return TXIInfo(
        text=text,
        is_animated=animated,
        is_cube=cube,
        is_bumpmap=bumpmap,
        compress_texture_false=compress_false,
        has_proceduretype=has_proceduretype,
        numx=_search_int(text, "numx"),
        numy=_search_int(text, "numy"),
        default_width=_search_int(text, "defaultwidth"),
        default_height=_search_int(text, "defaultheight"),
        alpha_blending=alpha_blending,
    )


def strip_alpha_blending(text: str) -> str:
    """Remove ``# alphablending`` comment lines, like the original UI."""

    return re.sub(
        r"^\s*#\s*alphablending.+$",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )


def render_txi(text: str) -> bytes:
    """Normalise TXI text and encode it the way tga2tpc writes it."""

    if not text:
        return b""
    normalised = text.strip().replace("\r\n", "\n").replace("\n", "\r\n")
    if not normalised:
        return b""
    return (normalised + "\r\n").encode("utf-8", errors="replace")


def decode_txi_bytes(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.decode("utf-8", errors="replace")


def read_txi_file(path: str | Path) -> str:
    return decode_txi_bytes(Path(path).read_bytes())
